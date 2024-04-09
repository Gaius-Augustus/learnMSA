import tensorflow as tf
import numpy as np
import learnMSA.msa_hmm.Training as train
from learnMSA.msa_hmm.SequenceDataset import SequenceDataset
import time
import math



def safe_log(x, log_zero_val=-1e3):
    """ Computes element-wise logarithm with output_i=log_zero_val where x_i=0.
    """
    epsilon = tf.constant(np.finfo(np.float32).tiny)
    log_x = tf.math.log(tf.maximum(x, epsilon))
    zero_mask = tf.cast(tf.equal(x, 0), dtype=log_x.dtype)
    log_x = (1-zero_mask) * log_x + zero_mask * log_zero_val
    return log_x


@tf.function
def viterbi_step(gamma_prev, emission_probs_i, transition_matrix=None):
    """ Computes one Viterbi dynamic programming step. z is a helper dimension for parallelization and not used in the final result.
    Args:
        gamma_prev: Viterbi values of the previous recursion. Shape (num_models, b, z, q)
        emission_probs_i: Emission probabilities of the i-th vertical input slice. Shape (num_models, b, q) or (num_models, b, q, q)
        transition_matrix: Logarithmic transition matricies describing the Markov chain. Shape (num_models, q, q)
    Returns:
        Viterbi values of the current recursion (gamma_next). Shape (num_models, b, z, q)
    """
    if transition_matrix is not None:
        gamma_next = transition_matrix[:,tf.newaxis,tf.newaxis] + tf.expand_dims(gamma_prev, -1)
        gamma_next = tf.reduce_max(gamma_next, axis=-2)
    else:
        gamma_next = gamma_prev
    if len(tf.shape(emission_probs_i)) == 3:
        #classic Viterbi step
        gamma_next += safe_log(emission_probs_i[:,:,tf.newaxis])
    else:
        #variant used for parallel Viterbi where (something like) emission probabilities is defined for state pairs
        gamma_next = tf.expand_dims(gamma_next, -1)
        gamma_next += emission_probs_i[:,:,tf.newaxis]
        gamma_next = tf.reduce_max(gamma_next, axis=-2)
    return gamma_next


def viterbi_dyn_prog(emission_probs, init, transition_matrix):
    """ Logarithmic (underflow safe) viterbi capable of decoding many sequences in parallel on the GPU.
    z is a helper dimension for parallelization and not used in the final result.
    Args:
        emission_probs: Tensor. Shape (num_models, b, L, q) or (num_models, b, L, q, q).
        init: Initial state distribution. Shape (num_models, z, q).
        transition_matrix: Logarithmic transition matricies describing the Markov chain. Shape (num_models, q, q)
    Returns:
        Viterbi values (gamma) per model. Shape (num_models, b, z, L, q)
    """
    gamma_val = safe_log(init)[:,tf.newaxis] 
    gamma_val = tf.cast(gamma_val, dtype=transition_matrix.dtype) 
    b0 = emission_probs[:,:,0]
    q = tf.shape(transition_matrix)[-1]
    if len(tf.shape(emission_probs)) == 4:
        #classic Viterbi initialization
        gamma_val += safe_log(b0)[:,:,tf.newaxis]
    else:
        #variant used for parallel Viterbi 
        gamma_val = viterbi_step(gamma_val, b0)
    L = tf.shape(emission_probs)[2]
    #tf.function-compatible accumulation of results in a dynamically unrolled loop using TensorArray
    gamma = tf.TensorArray(transition_matrix.dtype, size=L)
    gamma = gamma.write(0, gamma_val)
    for i in tf.range(1, L):
        gamma_val = viterbi_step(gamma_val, emission_probs[:,:,i], transition_matrix)
        gamma = gamma.write(i, gamma_val) 
    gamma = tf.transpose(gamma.stack(), [1,2,3,0,4])
    return gamma


@tf.function
def viterbi_backtracking_step(prev_states, gamma_state, transition_matrix_transposed):
    """ Computes a Viterbi backtracking step in parallel for all models and batch elements.
    Args:
        prev_states: Previously decoded states. Shape: (num_model, b, 1)
        gamma_state: Viterbi values of the previously decoded states. Shape: (num_model, b, q)
        transition_matrix_transposed: Transposed logarithmic transition matricies describing the Markov chain. 
                                        Shape (num_models, q, q) 
    """
    #since A is transposed, we gather columns (i.e. all starting states q' when transitioning to q)
    A_prev_states = tf.gather_nd(transition_matrix_transposed, prev_states, batch_dims=1)
    next_states = tf.math.argmax(A_prev_states + gamma_state, axis=-1)
    next_states = tf.expand_dims(next_states, -1)
    return next_states

    
def viterbi_backtracking(gamma, transition_matrix_transposed):
    """ Performs backtracking on Viterbi score tables.
    Args:
        gamma: A Viterbi score table per model and batch element. Shape (num_model, b, L, q)
        transition_matrix_transposed: Transposed logarithmic transition matricies describing the Markov chain. 
                                            Shape (num_models, q, q)
    Returns:
        State sequences. Shape (num_model, b, L) of type int16
    """
    cur_states = tf.math.argmax(gamma[:,:,-1], axis=-1)
    cur_states = tf.expand_dims(cur_states, -1)
    L = tf.shape(gamma)[2]
    #tf.function-compatible accumulation of results in a dynamically unrolled loop using TensorArray
    state_seqs_max_lik = tf.TensorArray(cur_states.dtype, size=L)
    state_seqs_max_lik = state_seqs_max_lik.write(L-1, cur_states)
    for i in tf.range(L-2, -1, -1):
        cur_states = viterbi_backtracking_step(cur_states, gamma[:,:,i], transition_matrix_transposed)
        state_seqs_max_lik = state_seqs_max_lik.write(i, cur_states)
    state_seqs_max_lik = tf.transpose(state_seqs_max_lik.stack(), [1,2,0,3])
    state_seqs_max_lik = tf.cast(state_seqs_max_lik, dtype=tf.int16)
    state_seqs_max_lik = state_seqs_max_lik[:,:,:,0]
    return state_seqs_max_lik


@tf.function
def viterbi_chunk_backtracking_step(end_states, gamma_start, transition_matrix_transposed, gamma_end=None, emission_probs_transposed=None):
    """ Computes a chunk-wise backtracking step used in parallel Viterbi decoding. Given the most likely end state of a chunk,
        computes the most likely state the chunk started with and the most likely end state of the previous chunk.
    Args:
        end_states: Most likely end states of the current chunk. Shape: (num_model, b, 1)
        gamma_start: Viterbi values at the start of the current chunk. Shape: (num_model, b, q)
        transition_matrix_transposed: Transposed logarithmic transition matricies describing the Markov chain. 
                                        Shape (num_models, q, q) 
        gamma_end: Viterbi values at the end of the previous chunk. Shape: (num_model, b, q)
        emission_probs_transposed: Pair-wise emission probs used during chunk-wise Viterbi dynamic programming. 
                                    Shape (num_models, b, q, q).
    Returns:
        next_states: Most likely end states of the previous chunk. Shape: (num_model, b, 1) 
                    (only returned if gamma_end is provided)
        chunk_start_states: Most likely start states of the current chunk. Shape: (num_model, b)
    """
    #since A is transposed, we gather columns (i.e. all starting states q' when transitioning to q)
    chunk_transition_probs = tf.gather_nd(emission_probs_transposed, end_states, batch_dims=2) #(num_model, b, q)
    chunk_start_states = tf.math.argmax(chunk_transition_probs + gamma_start, axis=-1)
    if gamma_end is not None:
        tf.print(gamma_end[1,1], summarize=-1)
        tf.print(chunk_start_states[1,1], summarize=-1)
        next_states = viterbi_backtracking_step(tf.expand_dims(chunk_start_states, -1), gamma_end, transition_matrix_transposed)
        tf.print(next_states[1,1], summarize=-1)
        return next_states, chunk_start_states
    else:
        return 0, chunk_start_states

    
def viterbi_chunk_wise_backtracking(gamma_start, gamma_end, transition_matrix_transposed, emission_probs_transposed):
    """ Performs backtracking on Viterbi score tables.
    Args:
        gamma_start: A Viterbi score table at the starting time of a chunk. Shape (num_model, b, num_chunks, q)
        gamma_end: A Viterbi score table at the end time of a chunk. Shape (num_model, b, num_chunks, q)
        transition_matrix_transposed: Transposed logarithmic transition matricies describing the Markov chain. 
                                            Shape (num_models, q, q)
        emission_probs_transposed: Pair-wise emission probs used during chunk-wise Viterbi dynamic programming. 
                                    Shape (num_models, b, num_chunks, q, q).
    Returns:
        State sequences. Shape (num_model, b, L) of type int16
    """
    end_states = tf.math.argmax(gamma_end[:,:,-1], axis=-1)
    end_states = tf.expand_dims(end_states, -1)
    num_chunks = tf.shape(gamma_start)[2]
    #tf.function-compatible accumulation of results in a dynamically unrolled loop using TensorArray
    optimal_chunks = tf.TensorArray(end_states.dtype, size=num_chunks)
    for i in tf.range(num_chunks-1, -1, -1):
        tf.print(i)
        end_states, chunk_start_states = viterbi_chunk_backtracking_step(end_states, 
                                                gamma_start[:,:,i], 
                                                transition_matrix_transposed, 
                                                gamma_end[:,:,i-1] if i > 0 else None,
                                                emission_probs_transposed[:,:,i])
        optimal_chunks = optimal_chunks.write(i, chunk_start_states)
    optimal_chunks = tf.transpose(optimal_chunks.stack(), [1,2,0])
    optimal_chunks = tf.cast(optimal_chunks, dtype=tf.int16)
    return optimal_chunks


def viterbi(sequences, hmm_cell, end_hints=None, parallel_factor=1, return_variables=False):
    """ Computes the most likely sequence of hidden states given unaligned sequences and a number of models.
        The implementation is logarithmic (underflow safe) and capable of decoding many sequences in parallel 
        on the GPU.
    Args:
        sequences: Input sequences. Shape (num_models, b, L, s) or (num_models, b, L)
        hmm_cell: A HMM cell representing k models used for decoding.
        end_hints: A optional tensor of shape (..., 2, num_states) that contains the correct state for the left and right ends of each chunk. (experimental)
        parallel_factor: Increasing this number allows computing likelihoods and posteriors chunk-wise in parallel at the cost of memory usage.
                        The parallel factor has to be a divisor of the sequence length.
        return_variables: If True, the function also returns the Viterbi variables (gamma). 
                        If parallel_factor > 1, the variables are returned only for the last chunk positions and are currently
                        not equivalent to a non-parallel call. Use for debugging only.
    Returns:
        State sequences. Shape (num_models, b, L)
    """
    if len(sequences.shape) == 3:
        sequences = tf.one_hot(sequences, hmm_cell.dim, dtype=hmm_cell.dtype)
    else:
        sequences = tf.cast(sequences, hmm_cell.dtype)
    #compute all emission probabilities in parallel
    emission_probs = hmm_cell.emission_probs(sequences, end_hints=end_hints, training=False)
    num_model, b, seq_len, q = tf.unstack(tf.shape(emission_probs))
    tf.debugging.assert_equal(seq_len % parallel_factor, 0, 
        f"The sequence length ({seq_len}) has to be divisible by the parallel factor ({parallel_factor}).")
    #compute max probabilities of chunks given starting states in parallel
    chunk_size = seq_len // parallel_factor
    emission_probs = tf.reshape(emission_probs, (num_model, b*parallel_factor, chunk_size, q))
    init_dist = tf.transpose(hmm_cell.init_dist, (1,0,2)) #(num_models, 1, q)
    init = init_dist if parallel_factor == 1 else tf.eye(q, dtype=hmm_cell.dtype)[tf.newaxis] 
    z = tf.shape(init)[1] #1 if parallel_factor == 1, q otherwise
    A = hmm_cell.log_A_dense
    At = hmm_cell.log_A_dense_t
    gamma = viterbi_dyn_prog(emission_probs, init, A) 
    gamma = tf.reshape(gamma, (num_model, b*parallel_factor*z, chunk_size, q))
    if parallel_factor == 1:
        viterbi_paths = viterbi_backtracking(gamma, At)
        variables_out = gamma
    else:
        gamma_local_chunk_ends = tf.reshape(gamma[:,:,-1], (num_model, b, parallel_factor, q, q))
        gamma_global_chunk_ends = viterbi_dyn_prog(gamma_local_right, init_dist, A)[:,:,0] #(num_model, b, parallel_factor, q)

        #first pass of backtracking to determine the most likely chunk end states
        most_likely_chunk_ends = viterbi_backtracking(gamma_global_chunk_ends, At)

        variables_out = chunk_probs

        #idea for backtracking: do a first pass that determines the most likely chunk start- and end states
        #then do a second pass that determines the most piecewise most likely chunk sequences given the correct border states


        gamma_start = tf.reshape(chunk_probs[:,:,:-1], (num_model, -1, 1, q))
        emission_probs = tf.reshape(emission_probs, (num_model, b, parallel_factor, chunk_size, q))
        chunk_start_probs = viterbi_step(gamma_start, tf.reshape(emission_probs[:,:,1:,0], (num_model, -1, q)), A) 
        gamma_init = safe_log(init_dist) + safe_log(tf.reshape(emission_probs, (num_model, b, seq_len, q))[:,:,0])
        gamma_init = gamma_init[:,:,tf.newaxis]
        chunk_start_probs = tf.reshape(chunk_start_probs, (num_model, b, parallel_factor-1, q))
        chunk_start_probs = tf.concat([gamma_init, chunk_start_probs], axis=2)
        optimal_chunks = viterbi_chunk_wise_backtracking(chunk_start_probs, chunk_probs, At, tf.transpose(gamma_last, (0,1,2,4,3))) 
        viterbi_paths = tf.reshape(viterbi_paths, (num_model, b, parallel_factor, q, chunk_size))
        optimal_chunks = tf.cast(optimal_chunks, dtype=tf.int32)
        optimal_chunks = tf.expand_dims(optimal_chunks, 3)
        viterbi_paths = tf.gather_nd(viterbi_paths, optimal_chunks, batch_dims=3) #(num_model, b, L)
        viterbi_paths = tf.reshape(viterbi_paths, (num_model, b, seq_len))
    if return_variables:
        return viterbi_paths, variables_out
    else:
        return viterbi_paths


def get_state_seqs_max_lik(data : SequenceDataset,
                           batch_generator,
                           indices,
                           batch_size,
                           hmm_cell, 
                           model_ids,
                           encoder=None,
                           parallel_factor=1):
    """ Runs batch-wise viterbi on all sequences in the dataset as specified by indices.
    Args:
        data: The sequence dataset.
        batch_generator: Batch generator.
        indices: Indices that specify which sequences in the dataset should be decoded. 
        batch_size: Specifies how many sequences will be decoded in parallel. 
        hmm_cell: MsaHmmCell object. 
        encoder: Optional encoder model that is applied to the sequences before Viterbi.
        parallel_factor: Increasing this number allows computing likelihoods and posteriors chunk-wise in parallel at the cost of memory usage.
                        The parallel factor has to be a divisor of the sequence length.
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
            viterbi_seq = viterbi(encoded_seq, hmm_cell, parallel_factor=parallel_factor)
            return viterbi_seq
    
    @tf.function(input_signature=(tf.TensorSpec(shape=[None, hmm_cell.num_models, None], dtype=tf.uint8),))
    def call_viterbi_single(inputs):
        if encoder is None:
            seq = tf.transpose(inputs, [1,0,2])
        else:
            seq = encoder(inputs) 
        #todo: this can be improved by encoding only for required models, not all
        seq = tf.gather(seq, model_ids, axis=0)
        return viterbi(seq, hmm_cell, parallel_factor=parallel_factor)
    
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