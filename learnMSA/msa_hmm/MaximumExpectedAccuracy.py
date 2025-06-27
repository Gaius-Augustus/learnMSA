import tensorflow as tf
import learnMSA.msa_hmm.Viterbi as viterbi
from functools import partial




def maximum_expected_accuracy(
    posterior_probs,
    hmm_cell,
    parallel_factor=1,
    non_homogeneous_mask_func=None
):
    """ Computes the state sequences with maximum expected accuracy (MEA).
        The implementation is logarithmic (underflow safe) and capable of 
        decoding many sequences in parallel on the GPU. 
        Optionally the function can also parallelize over the sequence length 
        at the cost of memory usage.
        (recommended for long sequences and HMMs with few states)
    Args:
        posterior_probs: Posterior state probabilities. 
            Shape (num_models, b, L, q)
        hmm_cell: A HMM cell representing k models used for decoding.
        parallel_factor: Increasing this number allows computing likelihoods 
            and posteriors chunk-wise in parallel at the cost of memory usage.
            The parallel factor has to be a divisor of the sequence length.
        non_homogeneous_mask_func: Optional function that maps a sequence 
            index i to a num_model x batch x q x q mask that specifies which 
            transitions are allowed.
    Returns:
        State sequences. Shape (num_models, b, L)
    """
    hmm_cell.recurrent_init()
    posterior_probs = tf.exp(posterior_probs) # see docs for explanation
    num_model, b, seq_len, q = tf.unstack(tf.shape(posterior_probs))
    seq_lens = None
    tf.debugging.assert_equal(
        seq_len % parallel_factor, 0, 
        f"The sequence length ({seq_len}) has to be divisible "
        f"by the parallel factor ({parallel_factor})."
    )
    # compute max probabilities of chunks given all possible starting states 
    # in parallel
    chunk_size = seq_len // parallel_factor
    posterior_probs = tf.reshape(
        posterior_probs, (num_model, b*parallel_factor, chunk_size, q)
    )
    init_dist = tf.transpose(hmm_cell.init_dist, (1,0,2)) #(num_models, 1, q)
    if parallel_factor == 1:
        init = init_dist 
    else: 
        init = tf.eye(q, dtype=hmm_cell.dtype)[tf.newaxis] 
    z = tf.shape(init)[1] #1 if parallel_factor == 1, q otherwise
    A = unify_tensor(hmm_cell.log_A_dense)
    At = unify_tensor(hmm_cell.log_A_dense_t)
    if non_homogeneous_mask_func is not None:
        non_homogeneous_mask_func = partial(
            non_homogeneous_mask_func, seq_lens=seq_lens, hmm_cell=hmm_cell
        )
    gamma = viterbi.viterbi_dyn_prog(
        posterior_probs,
        init, 
        A, 
        use_first_position_emission=parallel_factor==1, 
        non_homogeneous_mask_func=non_homogeneous_mask_func
    )
    gamma = tf.reshape(gamma, (num_model, b*parallel_factor*z, chunk_size, q))
    if parallel_factor == 1:
        mea_paths = viterbi.viterbi_backtracking(
            gamma, At, non_homogeneous_mask_func=non_homogeneous_mask_func
        )
    else:
        #compute (global) Viterbi values at the chunk borders
        posterior_probs_at_chunk_start = tf.reshape(
            posterior_probs[:,:,0], (num_model, b, parallel_factor, q)
        )
        gamma_local_at_chunk_end = gamma[:,:,-1]
        gamma_local_at_chunk_end = tf.reshape(
            gamma_local_at_chunk_end, (num_model, b, parallel_factor, q, q)
        )
        gamma_at_chunk_borders = viterbi.viterbi_chunk_dyn_prog(
            posterior_probs_at_chunk_start, 
            init_dist[:,0], 
            A, 
            gamma_local_at_chunk_end
        )
        #compute optimal states at the chunk borders
        gamma_local_at_chunk_end = tf.transpose(
            gamma_local_at_chunk_end, [0,1,2,4,3]
        )
        mea_chunk_borders = viterbi.viterbi_chunk_backtracking(
            gamma_at_chunk_borders, gamma_local_at_chunk_end, At
        )
        #compute full state sequences
        gamma = tf.reshape(
            gamma, (num_model, b, parallel_factor, z, chunk_size, q)
        )
        mea_paths = viterbi.viterbi_full_chunk_backtracking(
            mea_chunk_borders, gamma, At
        )
    return mea_paths


def unify_tensor(A):
    """
    Unifies a tensor by replacing all non-zero elements with 1.
    """
    A = tf.where(A > 0, 1.0, 0.0)
    return A