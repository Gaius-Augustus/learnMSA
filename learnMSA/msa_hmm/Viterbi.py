import tensorflow as tf
import numpy as np
import learnMSA.msa_hmm.Training as train
from learnMSA.msa_hmm.SequenceDataset import SequenceDataset
import time
import math
from functools import partial



def safe_log(x, log_zero_val=-1e3):
    """ Computes element-wise logarithm with output_i=log_zero_val where x_i=0.
    """
    epsilon = tf.constant(np.finfo(np.float32).tiny)
    log_x = tf.math.log(tf.maximum(x, epsilon))
    zero_mask = tf.cast(tf.equal(x, 0), dtype=log_x.dtype)
    log_x = (1-zero_mask) * log_x + zero_mask * log_zero_val
    return log_x


@tf.function
def viterbi_step(gamma_prev, emission_probs_i, transition_matrix, non_homogeneous_mask=None):
    """ Computes one Viterbi dynamic programming step. z is a helper dimension for parallelization and not used in the final result.
    Args:
        gamma_prev: Viterbi values of the previous recursion. Shape (num_models, b, z, q)
        emission_probs_i: Emission probabilities of the i-th vertical input slice. Shape (num_models, b, q)
        transition_matrix: Logarithmic transition matricies describing the Markov chain. Shape (num_models, q, q)
        non_homogeneous_mask: Optional mask of shape (num_models, b, q, q) that specifies which transitions are allowed.
    Returns:
        Viterbi values of the current recursion (gamma_next). Shape (num_models, b, z, q)
    """
    gamma_next = transition_matrix[:,tf.newaxis,tf.newaxis] + tf.expand_dims(gamma_prev, -1) #(n,b,z,q,q)
    if non_homogeneous_mask is not None:
        gamma_next += safe_log(non_homogeneous_mask[:,:,tf.newaxis])
    gamma_next = tf.reduce_max(gamma_next, axis=-2)
    gamma_next += safe_log(emission_probs_i[:,:,tf.newaxis])
    return gamma_next


def viterbi_dyn_prog(emission_probs, init, transition_matrix, use_first_position_emission=True, non_homogeneous_mask_func=None):
    """ Logarithmic (underflow safe) viterbi capable of decoding many sequences in parallel on the GPU.
    z is a helper dimension for parallelization and not used in the final result.
    Args:
        emission_probs: Tensor. Shape (num_models, b, L, q).
        init: Initial state distribution. Shape (num_models, z, q).
        transition_matrix: Logarithmic transition matricies describing the Markov chain. Shape (num_models, q, q)
        use_first_position_emission: If True, the first position of the sequence is considered to have an emission.
        non_homogeneous_mask_func: Optional function that maps a sequence index i to a num_models x q x q mask that specifies which transitions are allowed.
    Returns:
        Viterbi values (gamma) per model. Shape (num_models, b, z, L, q)
    """
    gamma_val = safe_log(init)[:,tf.newaxis] 
    gamma_val = tf.cast(gamma_val, dtype=transition_matrix.dtype) 
    b0 = emission_probs[:,:,0]
    if use_first_position_emission:
        gamma_val += safe_log(b0)[:,:,tf.newaxis]
    else:
        gamma_val += tf.zeros_like(b0)[:,:,tf.newaxis]
    L = tf.shape(emission_probs)[2]
    #tf.function-compatible accumulation of results in a dynamically unrolled loop using TensorArray
    gamma = tf.TensorArray(transition_matrix.dtype, size=L)
    gamma = gamma.write(0, gamma_val)
    for i in tf.range(1, L):
        gamma_val = viterbi_step(gamma_val, emission_probs[:,:,i], transition_matrix,
                                 non_homogeneous_mask_func(i) if non_homogeneous_mask_func is not None else None)
        gamma = gamma.write(i, gamma_val) 
    gamma = tf.transpose(gamma.stack(), [1,2,3,0,4])
    return gamma


@tf.function
def viterbi_chunk_step(gamma_prev, local_gamma):
    """ A variant of the Viterbi step that is used in the parallel variant of Viterbi.
    Args:
        gamma_prev: Viterbi values of the previous recursion. Shape (num_models, b, q)
        local_gamma: Logarithmic transition matricies describing the transition from chunk start to end. Shape (num_models, b, q, q)
    Returns:
        Viterbi values of the current recursion (gamma_next). Shape (num_models, b, q)
    """
    gamma_next = local_gamma + tf.expand_dims(gamma_prev, -1)
    gamma_next = tf.reduce_max(gamma_next, axis=-2)
    return gamma_next


def viterbi_chunk_dyn_prog(emission_probs, init, transition_matrix, local_gamma, non_homogeneous_mask=None):    
    """ A variant of Viterbi that computes the gamma values at the begin and end positions of chunks.
    Args:
        emission_probs: Emission probabilities at the starting positions of each chunk. Shape (num_models, b, num_chunks, q).
        init: Initial state distribution. Shape (num_models, q).
        transition_matrix: Logarithmic transition matricies describing the Markov chain. Shape (num_models, q, q)
        local_gamma: Local viterbi values at the end of each chunk. Shape (num_models, b, num_chunks, q, q)
        non_homogeneous_mask: Optional mask of shape (num_models, b, q, q) that specifies which transitions are allowed.
    Returns:
        Viterbi values (gamma) of begin and end positions per chunk. Shape (num_models, b, num_chunks, 2, q)
    """
    gamma_val = safe_log(init)[:,tf.newaxis] 
    gamma_val = tf.cast(gamma_val, dtype=transition_matrix.dtype) 
    b0 = emission_probs[:,:,0]
    gamma_val += safe_log(b0)
    L = tf.shape(emission_probs)[2]
    #tf.function-compatible accumulation of results in a dynamically unrolled loop using TensorArray
    gamma = tf.TensorArray(emission_probs.dtype, size=2*L)
    gamma = gamma.write(0, gamma_val)
    gamma_val = viterbi_chunk_step(gamma_val, local_gamma[:,:,0])
    gamma = gamma.write(1, gamma_val)
    for i in tf.range(1, L):
        gamma_val = viterbi_step(tf.expand_dims(gamma_val, -2), emission_probs[:,:,i], transition_matrix, non_homogeneous_mask)[...,0,:]
        gamma = gamma.write(2*i, gamma_val)
        gamma_val = viterbi_chunk_step(gamma_val, local_gamma[:,:,i])
        gamma = gamma.write(2*i+1, gamma_val)
    gamma = tf.transpose(gamma.stack(), [1,2,0,3])
    gamma = tf.reshape(gamma, (tf.shape(gamma)[0], tf.shape(gamma)[1], L, 2, tf.shape(gamma)[-1]))
    return gamma


@tf.function
def viterbi_backtracking_step(prev_states, gamma_state, transition_matrix_transposed, output_type, non_homogeneous_mask=None):
    """ Computes a Viterbi backtracking step in parallel for all models and batch elements.
    Args:
        prev_states: Previously decoded states. Shape: (num_model, b, 1)
        gamma_state: Viterbi values of the previously decoded states. Shape: (num_model, b, q)
        transition_matrix_transposed: Transposed logarithmic transition matricies describing the Markov chain. 
                                        Shape (num_models, q, q) or (num_models, b, q, q)
        output_type: Datatype of the output states.
        non_homogeneous_mask: Optional mask of shape (num_models, b, q, q) that specifies which transitions are allowed.
    """
    #since A is transposed, we gather columns (i.e. all starting states q' when transitioning to q)
    if non_homogeneous_mask is None:
        A_prev_states = tf.gather_nd(transition_matrix_transposed, prev_states, 
                                        batch_dims=len(tf.shape(transition_matrix_transposed))-2)
    else:
        A_prev_states = tf.gather_nd(transition_matrix_transposed  + safe_log(tf.transpose(non_homogeneous_mask, [0,1,3,2])), prev_states, 
                                        batch_dims=2)
    next_states = tf.math.argmax(A_prev_states + gamma_state, axis=-1, output_type=output_type)
    next_states = tf.expand_dims(next_states, -1)
    return next_states

    
def viterbi_backtracking(gamma, transition_matrix_transposed, output_type=tf.int32, non_homogeneous_mask_func=None):
    """ Performs backtracking on Viterbi score tables.
    Args:
        gamma: A Viterbi score table per model and batch element. Shape (num_model, b, L, q)
        transition_matrix_transposed: Transposed logarithmic transition matricies describing the Markov chain. 
                                            Shape (num_models, q, q) 
        output_type: Output type of the state sequences.
        non_homogeneous_mask_func: Optional function that maps a sequence index i to a num_model x batch x q x q mask that specifies which transitions are allowed.
    Returns:
        State sequences. Shape (num_model, b, L).
    """
    cur_states = tf.math.argmax(gamma[:,:,-1], axis=-1, output_type=output_type)
    cur_states = tf.expand_dims(cur_states, -1)
    L = tf.shape(gamma)[2]
    #tf.function-compatible accumulation of results in a dynamically unrolled loop using TensorArray
    state_seqs_max_lik = tf.TensorArray(output_type, size=L)
    state_seqs_max_lik = state_seqs_max_lik.write(L-1, cur_states)
    for i in tf.range(L-2, -1, -1):
        cur_states = viterbi_backtracking_step(cur_states, gamma[:,:,i], transition_matrix_transposed, output_type,
                                                non_homogeneous_mask_func(i+1) if non_homogeneous_mask_func is not None else None)
        state_seqs_max_lik = state_seqs_max_lik.write(i, cur_states)
    state_seqs_max_lik = tf.transpose(state_seqs_max_lik.stack(), [1,2,0,3])
    state_seqs_max_lik = state_seqs_max_lik[:,:,:,0]
    return state_seqs_max_lik


def viterbi_chunk_backtracking(gamma, local_gamma_end_transposed, transition_matrix_transposed, output_type=tf.int32, non_homogeneous_mask_func=None):
    """Performs backtracking on chunk-wise Viterbi score tables.
    Args:
        gamma: Viterbi values of begin and end positions per chunk. Shape (num_model, b, num_chunks, 2, q)
        local_gamma_end_transposed: Local viterbi values at the end of each chunk (transposed output of viterbi_chunk_dyn_prog).
                                Shape (num_models, b, num_chunks, q, q)
        transition_matrix_transposed: Transposed logarithmic transition matricies describing the Markov chain. 
                                            Shape (num_models, q, q) 
        output_type: Output type of the state sequences.
        non_homogeneous_mask_func: Optional function that maps a sequence index i to a num_model x batch x q x q mask that specifies which transitions are allowed.
    Returns:
        Most likely states at the chunk borders. Shape (num_model, b, num_chunks, 2).
    """
    cur_states = tf.math.argmax(gamma[:,:,-1,1], axis=-1, output_type=output_type)
    cur_states = tf.expand_dims(cur_states, -1)
    num_chunks = tf.shape(gamma)[2]
    #tf.function-compatible accumulation of results in a dynamically unrolled loop using TensorArray
    state_seqs_max_lik = tf.TensorArray(output_type, size=2*num_chunks)
    state_seqs_max_lik = state_seqs_max_lik.write(2*num_chunks-1, cur_states)
    cur_states = viterbi_backtracking_step(cur_states, gamma[:,:,-1,0], local_gamma_end_transposed[:,:,-1], output_type)
    state_seqs_max_lik = state_seqs_max_lik.write(2*num_chunks-2, cur_states)
    L = tf.shape(gamma)[2]
    for i in tf.range(1, num_chunks):
        cur_states = viterbi_backtracking_step(cur_states, gamma[:,:,-1-i,1], transition_matrix_transposed, output_type,
                                                non_homogeneous_mask_func(L-i) if non_homogeneous_mask_func is not None else None)
        state_seqs_max_lik = state_seqs_max_lik.write(2*num_chunks-2*i-1, cur_states)
        cur_states = viterbi_backtracking_step(cur_states, gamma[:,:,-1-i,0], local_gamma_end_transposed[:,:,-1-i], output_type)
        state_seqs_max_lik = state_seqs_max_lik.write(2*num_chunks-2*i-2, cur_states)
    state_seqs_max_lik = tf.transpose(state_seqs_max_lik.stack(), [1,2,0,3])
    state_seqs_max_lik = tf.reshape(state_seqs_max_lik, (tf.shape(state_seqs_max_lik)[0], tf.shape(state_seqs_max_lik)[1], num_chunks, 2))
    return state_seqs_max_lik


def viterbi_full_chunk_backtracking(viterbi_chunk_borders, local_gamma, transition_matrix_transposed, 
                                    output_type=tf.int32, non_homogeneous_mask_func=None):
    """ Given the optimal end points for each chunk, determines the full Viterbi state sequence.
    Args:
        viterbi_chunk_borders: Most likely states at the chunk borders. Shape (num_model, b, num_chunks, 2)
        local_gamma: Local viterbi values for all chunks Shape (num_models, b, num_chunks, q, chunk_length, q)
        transition_matrix_transposed: Transposed logarithmic transition matricies describing the Markov chain. 
                                            Shape (num_models, q, q) 
        output_type: Output type of the state sequences.
        non_homogeneous_mask_func: Optional function that maps a sequence index i to a num_model x batch x q x q mask that specifies which transitions are allowed.
    Returns:
        State sequences. Shape (num_model, b, num_chunks*chunk_length).
    """
    num_model, b, num_chunks, q, chunk_length, _ = tf.unstack(tf.shape(local_gamma))
    local_gamma = tf.reshape(local_gamma, (num_model, -1, q, chunk_length, q))
    state_seqs_max_lik = tf.TensorArray(output_type, size=chunk_length)
    start_states = tf.reshape(viterbi_chunk_borders[:,:,:,0], (num_model, -1, 1))
    end_states = tf.reshape(viterbi_chunk_borders[:,:,:,1], (num_model, -1, 1))
    #since we already know the optimal starting states for each chunk, we can choose the corresponding local gamma values
    local_gamma = tf.gather(local_gamma, start_states, batch_dims=2)[:,:,0] #(num_model, b*num_chunks, chunk_length, q)
    #regular backtracking on chunks 
    cur_states = end_states
    state_seqs_max_lik = state_seqs_max_lik.write(chunk_length-1, cur_states)
    for i in tf.range(chunk_length-2, 0, -1):
        cur_states = viterbi_backtracking_step(cur_states, local_gamma[:,:,i], transition_matrix_transposed, output_type,
                                                non_homogeneous_mask_func(i+1) if non_homogeneous_mask_func is not None else None)
        state_seqs_max_lik = state_seqs_max_lik.write(i, cur_states)
    state_seqs_max_lik = state_seqs_max_lik.write(0, start_states)
    state_seqs_max_lik = tf.transpose(state_seqs_max_lik.stack(), [1,2,0,3])
    state_seqs_max_lik = tf.reshape(state_seqs_max_lik, (num_model, b, num_chunks*chunk_length))
    return state_seqs_max_lik


def viterbi(sequences, hmm_cell, end_hints=None, parallel_factor=1, return_variables=False, non_homogeneous_mask_func=None):
    """ Computes the most likely sequence of hidden states given unaligned sequences and a number of models.
        The implementation is logarithmic (underflow safe) and capable of decoding many sequences in parallel 
        on the GPU. Optionally the function can also parallelize over the sequence length at the cost of memory usage.
        (recommended for long sequences and HMMs with few states)
    Args:
        sequences: Input sequences. Shape (num_models, b, L, s) or (num_models, b, L)
        hmm_cell: A HMM cell representing k models used for decoding.
        end_hints: A optional tensor of shape (..., 2, num_states) that contains the correct state for the left and right ends of each chunk. (experimental)
        parallel_factor: Increasing this number allows computing likelihoods and posteriors chunk-wise in parallel at the cost of memory usage.
                        The parallel factor has to be a divisor of the sequence length.
        return_variables: If True, the function also returns the Viterbi variables (gamma). 
                        If parallel_factor > 1, the variables are returned only for the first and last chunk positions and are currently
                        not equivalent to a non-parallel call. Use for debugging only.
        non_homogeneous_mask_func: Optional function that maps a sequence index i to a num_model x batch x q x q mask that specifies which transitions are allowed.
    Returns:
        State sequences. Shape (num_models, b, L)
    """
    if len(sequences.shape) == 3:
        sequences = tf.one_hot(sequences, hmm_cell.dim, dtype=hmm_cell.dtype)
    else:
        sequences = tf.cast(sequences, hmm_cell.dtype)
    seq_lens = tf.reduce_sum(tf.cast(sequences[..., -1]==0, tf.int32), axis=-1)
    #compute all emission probabilities in parallel
    emission_probs = hmm_cell.emission_probs(sequences, end_hints=end_hints, training=False)
    num_model, b, seq_len, q = tf.unstack(tf.shape(emission_probs))
    tf.debugging.assert_equal(seq_len % parallel_factor, 0, 
        f"The sequence length ({seq_len}) has to be divisible by the parallel factor ({parallel_factor}).")
    #compute max probabilities of chunks given all possible starting states in parallel
    chunk_size = seq_len // parallel_factor
    emission_probs = tf.reshape(emission_probs, (num_model, b*parallel_factor, chunk_size, q))
    init_dist = tf.transpose(hmm_cell.init_dist, (1,0,2)) #(num_models, 1, q)
    init = init_dist if parallel_factor == 1 else tf.eye(q, dtype=hmm_cell.dtype)[tf.newaxis] 
    z = tf.shape(init)[1] #1 if parallel_factor == 1, q otherwise
    A = hmm_cell.log_A_dense
    At = hmm_cell.log_A_dense_t
    if non_homogeneous_mask_func is not None:
        non_homogeneous_mask_func = partial(non_homogeneous_mask_func, seq_lens=seq_lens, hmm_cell=hmm_cell)
    gamma = viterbi_dyn_prog(emission_probs, init, A, 
                             use_first_position_emission=parallel_factor==1, 
                             non_homogeneous_mask_func=non_homogeneous_mask_func)
    gamma = tf.reshape(gamma, (num_model, b*parallel_factor*z, chunk_size, q))
    if parallel_factor == 1:
        viterbi_paths = viterbi_backtracking(gamma, At, non_homogeneous_mask_func=non_homogeneous_mask_func)
        variables_out = gamma
    else:
        #compute (global) Viterbi values at the chunk borders
        emission_probs_at_chunk_start = tf.reshape(emission_probs[:,:,0], (num_model, b, parallel_factor, q))
        gamma_local_at_chunk_end = gamma[:,:,-1]
        gamma_local_at_chunk_end = tf.reshape(gamma_local_at_chunk_end, (num_model, b, parallel_factor, q, q))
        gamma_at_chunk_borders = viterbi_chunk_dyn_prog(emission_probs_at_chunk_start, init_dist[:,0], A, gamma_local_at_chunk_end)
        #compute optimal states at the chunk borders
        gamma_local_at_chunk_end = tf.transpose(gamma_local_at_chunk_end, [0,1,2,4,3])
        viterbi_chunk_borders = viterbi_chunk_backtracking(gamma_at_chunk_borders, gamma_local_at_chunk_end, At)
        #compute full state sequences
        gamma = tf.reshape(gamma, (num_model, b, parallel_factor, z, chunk_size, q))
        viterbi_paths = viterbi_full_chunk_backtracking(viterbi_chunk_borders, gamma, At)
        variables_out = gamma_at_chunk_borders
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
                           non_homogeneous_mask_func=None,
                           parallel_factor=1):
    """ Runs batch-wise viterbi on all sequences in the dataset as specified by indices.
    Args:
        data: The sequence dataset.
        batch_generator: Batch generator.
        indices: Indices that specify which sequences in the dataset should be decoded. 
        batch_size: Specifies how many sequences will be decoded in parallel. 
        hmm_cell: MsaHmmCell object. 
        encoder: Optional encoder model that is applied to the sequences before Viterbi.
        non_homogeneous_mask_func: Optional function that maps a sequence index i to a num_model x batch x q x q mask that specifies which transitions are allowed.
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
                                  dtype=np.uint32) 
    if encoder:
        @tf.function(input_signature=[[tf.TensorSpec(x.shape, dtype=x.dtype) for x in encoder.inputs]])
        def call_viterbi(inputs):
            encoded_seq = encoder(inputs)
            #todo: this can be improved by encoding only for required models, not all
            encoded_seq = tf.gather(encoded_seq, model_ids, axis=0)
            viterbi_seq = viterbi(encoded_seq, hmm_cell, parallel_factor=parallel_factor, non_homogeneous_mask_func=non_homogeneous_mask_func)
            return viterbi_seq
    
    @tf.function(input_signature=(tf.TensorSpec(shape=[None, hmm_cell.num_models, None], dtype=tf.uint8),))
    def call_viterbi_single(inputs):
        if encoder is None:
            seq = tf.transpose(inputs, [1,0,2])
        else:
            seq = encoder(inputs) 
        #todo: this can be improved by encoding only for required models, not all
        seq = tf.gather(seq, model_ids, axis=0)
        return viterbi(seq, hmm_cell, parallel_factor=parallel_factor, non_homogeneous_mask_func=non_homogeneous_mask_func)
    
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