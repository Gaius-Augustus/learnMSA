import tensorflow as tf
import numpy as np
import learnMSA.msa_hmm.Training as train
from learnMSA.msa_hmm.SequenceDataset import SequenceDataset
import time
import math



def save_log(x, log_zero_val=-1e3):
    """ Computes element-wise logarithm with output_i=log_zero_val where x_i=0.
    """
    epsilon = tf.constant(np.finfo(np.float32).tiny)
    log_x = tf.math.log(tf.maximum(x, epsilon))
    zero_mask = tf.cast(tf.equal(x, 0), dtype=log_x.dtype)
    log_x = (1-zero_mask) * log_x + zero_mask * log_zero_val
    return log_x



def save_log(x, log_zero_val=-1e3):
    """ Computes element-wise logarithm with output_i=log_zero_val where x_i=0.
    """
    epsilon = tf.constant(np.finfo(np.float32).tiny)
    log_x = tf.math.log(tf.maximum(x, epsilon))
    zero_mask = tf.cast(tf.equal(x, 0), dtype=log_x.dtype)
    log_x = (1-zero_mask) * log_x + zero_mask * log_zero_val
    return log_x


def viterbi_step(gamma_prev, emission_probs_i, hmm_cell):
    """ Computes one Viterbi dynamic programming step.
    Args:
        gamma_prev: Viterbi values of the previous recursion. Shape (num_models, b, q)
        emission_probs_i: Emission probabilities of the i-th vertical input slice. Shape (num_models, b, q)
        hmm_cell: HMM cell with the models under which decoding should happen.
    """
    epsilon = tf.constant(np.finfo(np.float32).tiny)
    #very very inefficient!?
    a = tf.expand_dims(hmm_cell.log_A_dense, 1) + tf.expand_dims(gamma_prev, -1)
    a = tf.reduce_max(a, axis=-2)
    b = save_log(emission_probs_i)
    gamma_next = a + b
    return gamma_next


def viterbi_dyn_prog(sequences, hmm_cell, end_hints=None):
    """ Logarithmic (underflow safe) viterbi capable of decoding many sequences in parallel on the GPU.
    Args:
        sequences: Tensor. Shape (num_models, b, L, s).
        hmm_cell: HMM cell with the models under which decoding should happen.
        end_hints: A optional tensor of shape (..., 2, num_states) that contains the correct state for the left and right ends of each chunk.
    Returns:
        Viterbi values (gamma) per model. Shape (num_model, b, L, q)
    """
    epsilon = tf.constant(np.finfo(np.float32).tiny)
    init = tf.transpose(hmm_cell.init_dist, (1,0,2)) #(num_models, 1, q)
    emission_probs = hmm_cell.emission_probs(sequences, end_hints=end_hints, training=False)
    b0 = emission_probs[:,:,0]
    gamma_val = save_log(init) + save_log(b0)
    gamma_val = tf.cast(gamma_val, dtype=hmm_cell.dtype) 
    L = tf.shape(sequences)[-2]
    #tf.function-compatible accumulation of results in a dynamically unrolled loop using TensorArray
    gamma = tf.TensorArray(hmm_cell.dtype, size=L)
    gamma = gamma.write(0, gamma_val)
    for i in tf.range(1, L):
        gamma_val = viterbi_step(gamma_val, emission_probs[:,:,i], hmm_cell)
        gamma = gamma.write(i, gamma_val) 
    gamma = tf.transpose(gamma.stack(), [1,2,0,3])
    return gamma


def viterbi_backtracking_step(q, gamma_state, hmm_cell):
    """ Computes a Viterbi backtracking step in parallel for all models and batch elements.
    Args:
        q: Previously decoded states. Shape: (num_model, b, 1)
        gamma_state: Viterbi values of the previously decoded states. Shape: (num_model, b, q)
        hmm_cell: HMM cell with the models under which decoding should happen.
    """
    #since A is transposed, we gather columns (i.e. all starting states q' when transitioning to q)
    A_q = tf.gather_nd(hmm_cell.log_A_dense_t, q, batch_dims=1)
    q = tf.math.argmax(A_q + gamma_state, axis=-1)
    q = tf.expand_dims(q, -1)
    return q

    
def viterbi_backtracking(hmm_cell, gamma):
    """ Performs backtracking on Viterbi score tables.
    Args:
        hmm_cell: HMM cell with the models under which decoding should happen.
        gamma: A Viterbi score table per model and batch element. Shape (num_model, b, L, q)
    """
    q = tf.math.argmax(gamma[:,:,-1], axis=-1)
    q = tf.expand_dims(q, -1)
    L = tf.shape(gamma)[2]
    #tf.function-compatible accumulation of results in a dynamically unrolled loop using TensorArray
    state_seqs_max_lik = tf.TensorArray(q.dtype, size=L)
    state_seqs_max_lik = state_seqs_max_lik.write(L-1, q)
    for i in tf.range(L-2, -1, -1):
        q = viterbi_backtracking_step(q, gamma[:,:,i], hmm_cell)
        state_seqs_max_lik = state_seqs_max_lik.write(i, q)
    state_seqs_max_lik = tf.transpose(state_seqs_max_lik.stack(), [1,2,0,3])
    state_seqs_max_lik = tf.cast(state_seqs_max_lik, dtype=tf.int16)
    state_seqs_max_lik = state_seqs_max_lik[:,:,:,0]
    return state_seqs_max_lik


def viterbi(sequences, hmm_cell, end_hints=None):
    """ Computes the most likely sequence of hidden states given unaligned sequences and a number of models.
        The implementation is logarithmic (underflow safe) and capable of decoding many sequences in parallel 
        on the GPU.
    Args:
        sequences: Input sequences. Shape (num_models, b, L, s) or (num_models, b, L)
        hmm_cell: A HMM cell representing k models used for decoding.
    Returns:
        State sequences. Shape (num_models, b, L)
    """
    if len(sequences.shape) == 3:
        sequences = tf.one_hot(sequences, hmm_cell.dim, dtype=hmm_cell.dtype)
    else:
        sequences = tf.cast(sequences, hmm_cell.dtype)
    gamma = viterbi_dyn_prog(sequences, hmm_cell, end_hints=end_hints)
    return viterbi_backtracking(hmm_cell, gamma)


def get_state_seqs_max_lik(data : SequenceDataset,
                           batch_generator,
                           indices,
                           batch_size,
                           hmm_cell, 
                           model_ids,
                           encoder=None):
    """ Runs batch-wise viterbi on all sequences in the dataset as specified by indices.
    Args:
        data: The sequence dataset.
        batch_generator: Batch generator.
        indices: Indices that specify which sequences in the dataset should be decoded. 
        batch_size: Specifies how many sequences will be decoded in parallel. 
        hmm_cell: MsaHmmCell object. 
        encoder: Optional eget_state_seqs_max_likncoder model that is applied to the sequences before Viterbi.
    Returns:
        A dense integer representation of the most likely state sequences. Shape: (num_model, num_seq, L)
    """
    #does currently not support multi-GPU, scale the batch size to account for that and prevent overflow
    num_gpu = len([x.name for x in tf.config.list_logical_devices() if x.device_type == 'GPU']) 
    num_devices = num_gpu + int(num_gpu==0) #account for the CPU-only case 
    batch_size = int(batch_size / num_devices)
    hmm_cell.recurrent_init()
    old_crop_long_seqs = batch_generator.crop_long_seqs
    batch_generator.crop_long_seqs = math.inf #do not crop sequences
    ds = train.make_dataset(indices, 
                            batch_generator, 
                            batch_size,
                            shuffle=False,
                            bucket_by_seq_length=True,
                            model_lengths=hmm_cell.length)
    seq_len = np.amax(data.seq_lens[indices]+1)
    #initialize with terminal states
    state_seqs_max_lik = np.zeros((hmm_cell.num_models, indices.size, seq_len), 
                                  dtype=np.uint16) 
    if encoder:
        @tf.function(input_signature=[[tf.TensorSpec(x.shape, dtype=x.dtype) for x in encoder.inputs]])
        def call_viterbi(inputs):
            encoded_seq = encoder(inputs)
            #todo: this can be improved by encoding only for required models, not all
            encoded_seq = tf.gather(encoded_seq, model_ids, axis=0)
            viterbi_seq = viterbi(encoded_seq, hmm_cell)
            return viterbi_seq
    
    @tf.function(input_signature=(tf.TensorSpec(shape=[None, hmm_cell.num_models, None], dtype=tf.uint8),))
    def call_viterbi_single(inputs):
        if encoder is None:
            seq = tf.transpose(inputs, [1,0,2])
        else:
            seq = encoder(inputs) 
        #todo: this can be improved by encoding only for required models, not all
        seq = tf.gather(seq, model_ids, axis=0)
        return viterbi(seq, hmm_cell)
    
    for i,q in enumerate(hmm_cell.num_states):
        state_seqs_max_lik[i] = q-1 #terminal state
    for (*inputs, batch_indices), _ in ds:
        if hasattr(batch_generator, "return_only_sequences") and batch_generator.return_only_sequences:
            state_seqs_max_lik_batch = call_viterbi_single(inputs[0]).numpy()
        else:
            state_seqs_max_lik_batch = call_viterbi(inputs).numpy()
        _,b,l = state_seqs_max_lik_batch.shape
        state_seqs_max_lik[:, batch_indices, :l] = state_seqs_max_lik_batch

    # revert batch generator state
    batch_generator.crop_long_seqs = old_crop_long_seqs

    return state_seqs_max_lik