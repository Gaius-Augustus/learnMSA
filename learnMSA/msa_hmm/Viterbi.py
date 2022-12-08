import tensorflow as tf
import numpy as np
import learnMSA.msa_hmm.Training as train


@tf.function
def viterbi_step(hmm_cell, gamma_prev, sequences_i, log_A_dense, epsilon):
    """ Computes one Viterbi dynamic programming step.
    Args:
        hmm_cell: HMM cell with the models under which decoding should happen.
        gamma_prev: Viterbi values of the previous recursion. Shape (num_models, b, q)
        sequences_i: i-th vertical sequence slice. Shape (num_models, b, s)
        log_A_dense: Logarithmic transition matricies. Shape (num_model,q, q)
        epsilon: A small constant for numeric stability. 
    """
    a = tf.reduce_max(tf.expand_dims(log_A_dense, 1) + tf.expand_dims(gamma_prev, -1), axis=-2)
    b = hmm_cell.emission_probs(sequences_i)
    b += epsilon                         
    b = tf.math.log(b)
    gamma_next = a + b
    return gamma_next


def viterbi_dyn_prog(sequences, hmm_cell, epsilon=np.finfo(np.float32).tiny):
    """ Logarithmic (underflow safe) viterbi capable of decoding many sequences in parallel on the GPU.
    Args:
        sequences: Tensor. Shape (num_models, b, L, s).
        hmm_cell: HMM cell with the models under which decoding should happen.
        epsilon: A small constant for numeric stability. 
    Returns:
        Viterbi values (gamma) per model. Shape (num_model, b, L, q)
    """
    epsilon = tf.cast(epsilon, hmm_cell.dtype)
    init = hmm_cell.make_initial_distribution()
    init = tf.transpose(init, (1,0,2)) #(num_models, 1, q)
    log_A_dense = hmm_cell.transitioner.make_log_A()
    b0 = hmm_cell.emission_probs(sequences[:,:,0])
    gamma0 = tf.math.log(init+epsilon) + tf.math.log(b0+epsilon)
    gamma0 = tf.cast(gamma0, dtype=sequences.dtype) 
    gamma = [gamma0]  
    for i in range(1,tf.shape(sequences)[-2]):
        gamma_next = viterbi_step(hmm_cell, 
                                    gamma[-1], 
                                    sequences[:,:,i], 
                                    log_A_dense,
                                    epsilon)
        gamma.append(gamma_next) 
    gamma = tf.stack(gamma, axis=2)
    return gamma


@tf.function
def viterbi_backtracking_step(q, gamma_state, log_A_dense_t):
    """ Computes a Viterbi backtracking step in parallel for all models and batch elements.
    Args:
        q: Previously decoded states. Shape: (num_model, b, 1)
        gamma_state: Viterbi values of the previously decoded states. Shape: (num_model, b, q)
        log_A_dense_t: Transposed, logarithmic transition matrices per model in dense format. Shape: (num_model, q, q)
    """
    #since A is transposed, we gather columns (i.e. all starting states q' when transitioning to q)
    A_q = tf.gather_nd(log_A_dense_t, q, batch_dims=1)
    q = tf.compat.v1.argmax(A_q + gamma_state, axis=-1)
    q = tf.expand_dims(q, -1)
    return q

    
def viterbi_backtracking(hmm_cell, gamma):
    """ Performs backtracking on Viterbi score tables.
    Args:
        hmm_cell: HMM cell with the models under which decoding should happen.
        gamma: A Viterbi score table per model and batch element. Shape (num_model, b, L, q)
    """
    log_A_dense = hmm_cell.transitioner.make_log_A()
    log_A_dense_t = tf.transpose(log_A_dense, [0,2,1])
    state_seqs_max_lik = []
    q = tf.compat.v1.argmax(gamma[:,:,-1], axis=-1)
    q = tf.expand_dims(q, -1)
    L = tf.shape(gamma)[2]
    for i in range(L):
        q = viterbi_backtracking_step(q, gamma[:,:,L-i-1], log_A_dense_t)
        state_seqs_max_lik.insert(0, q)
    state_seqs_max_lik = tf.stack(state_seqs_max_lik, axis=2)
    state_seqs_max_lik = tf.cast(state_seqs_max_lik, tf.int16)[:,:,:,0]
    return state_seqs_max_lik


def viterbi(sequences, hmm_cell):
    """ Computes the most likely sequence of hidden states given unaligned sequences and a number of models.
        The implementation is logarithmic (underflow safe) and capable of decoding many sequences in parallel 
        on the GPU.
    Args:
        sequences: Input sequences. Shape (num_models, b, L, s) or (num_models, b, L)
        hmm_cell: A HMM cell representing k models used for decoding.
    Returns:
        State sequences. Shape (num_models, b, L)
    """
    hmm_cell.recurrent_init()
    if len(sequences.shape) == 3:
        sequences = tf.one_hot(sequences, hmm_cell.dim, dtype=hmm_cell.dtype)
    else:
        sequences = tf.cast(sequences, hmm_cell.dtype)
    gamma = viterbi_dyn_prog(sequences, hmm_cell)
    return viterbi_backtracking(hmm_cell, gamma).numpy()


def get_state_seqs_max_lik(fasta_file,
                           batch_generator,
                           indices,
                           batch_size,
                           msa_hmm_cell, 
                           encoder=None):
    """ Runs batch-wise viterbi on all sequences in fasta_file as specified by indices.
    Args:
        fasta_file: Fasta file object.
        batch_generator: Batch generator.
        indices: Indices that specify which sequences in fasta_file should be decoded. 
        batch_size: Specifies how many sequences will be decoded in parallel. 
        msa_hmm_cell: MsaHmmCell object. 
        encoder: Optional encoder model that is applied to the sequences before Viterbi.
    Returns:
        A dense integer representation of the most likely state sequences. Shape: (num_model, num_seq, L)
    """
    ds = train.make_dataset(indices, 
                            batch_generator, 
                            batch_size,
                            shuffle=False)
    seq_len = np.amax(fasta_file.seq_lens[indices])+1
    #initialize with terminal states
    state_seqs_max_lik = np.zeros((msa_hmm_cell.num_models, indices.size, seq_len), 
                                  dtype=np.uint16) 
    for i,q in enumerate(msa_hmm_cell.num_states):
        state_seqs_max_lik[i] = q-1 #terminal state
    i = 0                              
    for inputs, _ in ds:
        if encoder is None:
            if batch_generator.return_only_sequences:
                seq = inputs
            else:
                seq = inputs[0]
        else:
            seq = encoder(inputs) 
        state_seqs_max_lik_batch = viterbi(seq, msa_hmm_cell) 
        _,b,l = state_seqs_max_lik_batch.shape
        state_seqs_max_lik[:, i:i+b, :l] = state_seqs_max_lik_batch
        i += b
    return state_seqs_max_lik