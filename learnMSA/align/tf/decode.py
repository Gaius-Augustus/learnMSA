"""
TensorFlow implementations of the core MSA decoding operations.

These functions mirror :func:`AlignmentModel.decode_core` and
:func:`AlignmentModel.decode_flank` exactly but operate on ``tf.Tensor``
inputs so they can be JIT-compiled and executed on GPU.

Each function:
  * accepts either a ``tf.Tensor`` or a ``np.ndarray`` (TF will up-cast numpy
    inputs automatically).
  * returns plain **numpy arrays** – the small decoded metadata arrays are
    CPU-bound anyway, and transferring them once per batch is cheap compared
    to keeping the full ``(n, T)`` state-sequence tensor in CPU RAM.

Wrapping in ``@tf.function`` (optionally with ``jit_compile=True``) is left
to the caller so that tracing / XLA compilation can be shared across multiple
calls with the same shapes.
"""

import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tf_decode_flank_core(state_seqs, flank_state_id, indices):
    """Shared TF logic for decode_flank.  All args are tf.Tensor."""
    T = tf.shape(state_seqs)[1]
    T_i64 = tf.cast(T, tf.int64)

    pos = tf.cast(tf.range(T), tf.int32)  # (T,)
    # active[i, t] = True when t >= indices[i]
    active = pos[tf.newaxis, :] >= tf.cast(indices[:, tf.newaxis], tf.int32)  # (n, T)
    non_flank = active & tf.not_equal(state_seqs, flank_state_id)            # (n, T)

    has_non_flank = tf.reduce_any(non_flank, axis=1)                         # (n,)
    first_non_flank = tf.cast(tf.argmax(tf.cast(non_flank, tf.int8), axis=1), tf.int32)  # (n,)
    end_pos = tf.where(has_non_flank, first_non_flank, tf.fill(tf.shape(indices), T))     # (n,)

    insertion_lens = tf.cast(end_pos - indices, tf.int16)
    return insertion_lens, indices, end_pos


def _tf_decode_core_inner(state_seqs, c, indices):
    """Shared TF logic for decode_core.  All args are tf.Tensor."""
    n = tf.shape(state_seqs)[0]
    T = tf.shape(state_seqs)[1]

    pos = tf.cast(tf.range(T), tf.int32)                                     # (T,)
    active = pos[tf.newaxis, :] >= tf.cast(indices[:, tf.newaxis], tf.int32) # (n, T)
    s = state_seqs                                                             # (n, T) int32

    # ---- locate terminal positions ----------------------------------------
    is_unannotated = active & tf.equal(s, 2 * c)
    is_at_end      = active & (tf.equal(s, 2 * c + 1) | tf.equal(s, 2 * c + 2))
    is_terminal    = is_unannotated | is_at_end

    has_terminal = tf.reduce_any(is_terminal, axis=1)                        # (n,)
    first_term   = tf.cast(tf.argmax(tf.cast(is_terminal, tf.int8), axis=1), tf.int32)  # (n,)
    end_pos      = tf.where(has_terminal, first_term, tf.fill([n], T))        # (n,)

    # ---- core region mask ---------------------------------------------------
    in_core   = active & (pos[tf.newaxis, :] < end_pos[:, tf.newaxis])
    is_match  = in_core & (s >= 0) & (s < c)
    is_insert = in_core & (s >= c) & (s < 2 * c - 1)

    # ---- consensus columns --------------------------------------------------
    # For every (seq, pos) where is_match: consensus_columns[seq, state] = pos
    match_coords = tf.cast(tf.where(is_match), tf.int32)                     # (K, 2)
    seq_idx_m = match_coords[:, 0]
    pos_idx_m = match_coords[:, 1]
    state_m   = tf.gather_nd(s, match_coords)                                # (K,)
    cc_indices = tf.stack([seq_idx_m, state_m], axis=1)                      # (K, 2)
    consensus_columns = tf.tensor_scatter_nd_update(
        tf.fill([n, c], tf.cast(-1, tf.int32)),
        cc_indices,
        pos_idx_m,
    )                                                                         # (n, c)

    # ---- insertion lengths --------------------------------------------------
    ins_coords  = tf.cast(tf.where(is_insert), tf.int32)                     # (M, 2)
    seq_idx_i   = ins_coords[:, 0]
    ins_state_i = tf.gather_nd(s, ins_coords) - c                            # (M,) insert-state idx
    il_indices  = tf.stack([seq_idx_i, ins_state_i], axis=1)                 # (M, 2)
    insertion_lens = tf.tensor_scatter_nd_add(
        tf.zeros([n, c - 1], dtype=tf.int32),
        il_indices,
        tf.ones(tf.shape(ins_coords)[0], dtype=tf.int32),
    )                                                                         # (n, c-1)

    # ---- insertion starts (min pos per seq×insert-state) -------------------
    pos_idx_i = ins_coords[:, 1]
    lin_idx   = seq_idx_i * (c - 1) + ins_state_i                           # (M,) flat index
    ins_start_flat = tf.math.unsorted_segment_min(
        pos_idx_i,
        lin_idx,
        num_segments=n * (c - 1),
    )                                                                         # (n*(c-1),)
    # Positions equal to T indicate "no insertion" – remap to -1
    ins_start_2d = tf.reshape(ins_start_flat, [n, c - 1])
    insertion_start = tf.where(
        tf.equal(ins_start_2d, T),
        tf.fill([n, c - 1], tf.cast(-1, tf.int32)),
        ins_start_2d,
    )                                                                         # (n, c-1)

    # ---- finished flag ------------------------------------------------------
    ep_clamped   = tf.minimum(end_pos, T - 1)                               # (n,)
    row_indices  = tf.range(n)
    gather_idx   = tf.stack([row_indices, ep_clamped], axis=1)               # (n, 2)
    finished     = has_terminal & tf.gather_nd(is_at_end, gather_idx)       # (n,)

    return consensus_columns, insertion_lens, insertion_start, finished, end_pos


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def decode_flank_tf(state_seqs, flank_state_id: int, indices):
    """TensorFlow equivalent of :meth:`AlignmentModel.decode_flank`.

    Args:
        state_seqs: ``(n, T)`` int32 tensor / array of state sequences.
        flank_state_id: Integer id of the flank state.
        indices: ``(n,)`` int32 tensor / array of per-sequence start
            positions.  **Modified in-place** (like the numpy version) if
            passed as a numpy array; a new tensor is returned otherwise.

    Returns:
        insertion_lens: ``(n,)`` int16 numpy array.
        insertion_start: ``(n,)`` int32 numpy array (copy of the input
            indices *before* advancing).
    """
    state_seqs_t = tf.cast(state_seqs, tf.int32)
    indices_t    = tf.cast(indices, tf.int32)

    insertion_start_np = np.array(indices, dtype=np.int32)

    insertion_lens_t, _, end_pos_t = _tf_decode_flank_core(
        state_seqs_t, tf.cast(flank_state_id, tf.int32), indices_t
    )

    insertion_lens_np = insertion_lens_t.numpy().astype(np.int16)
    end_pos_np        = end_pos_t.numpy().astype(indices.dtype)
    indices[:]        = end_pos_np
    return insertion_lens_np, insertion_start_np


def decode_core_tf(model_length: int, state_seqs, indices):
    """TensorFlow equivalent of :meth:`AlignmentModel.decode_core`.

    Args:
        model_length: Number of match states ``c``.
        state_seqs: ``(n, T)`` int32 tensor / array of state sequences.
        indices: ``(n,)`` int32 numpy array of per-sequence start positions.
            **Modified in-place** to point past the decoded core block.

    Returns:
        consensus_columns: ``(n, c)`` int16 numpy array.
        insertion_lens: ``(n, c-1)`` int16 numpy array.
        insertion_start: ``(n, c-1)`` int16 numpy array.
        finished: ``(n,)`` bool numpy array.
    """
    c             = tf.cast(model_length, tf.int32)
    state_seqs_t  = tf.cast(state_seqs, tf.int32)
    indices_t     = tf.cast(indices, tf.int32)

    consensus_columns_t, insertion_lens_t, insertion_start_t, finished_t, end_pos_t = \
        _tf_decode_core_inner(state_seqs_t, c, indices_t)

    consensus_columns = consensus_columns_t.numpy().astype(np.int16)
    insertion_lens    = insertion_lens_t.numpy().astype(np.int16)
    insertion_start   = insertion_start_t.numpy().astype(np.int16)
    finished          = finished_t.numpy()
    indices[:]        = end_pos_t.numpy().astype(indices.dtype)
    return consensus_columns, insertion_lens, insertion_start, finished


def decode_tf(model_length: int, state_seqs):
    """TensorFlow equivalent of :meth:`AlignmentModel.decode`.

    Runs the full decode loop (flank → core* → flank) using TF ops for each
    individual step while keeping the repeat-detection loop in Python.

    Args:
        model_length: Number of match states.
        state_seqs: ``(n, T)`` int32 array/tensor of Viterbi/MEA paths
            (single model, already sliced to ``[:, :, model_idx]``).

    Returns:
        :class:`~learnMSA.align.alignment_metadata.AlignmentMetaData`
    """
    from learnMSA.align.alignment_metadata import AlignmentMetaData

    # Ensure numpy so we can do in-place updates on `indices`
    if isinstance(state_seqs, tf.Tensor):
        state_seqs = state_seqs.numpy()

    n = state_seqs.shape[0]
    c = model_length
    indices = np.zeros(n, dtype=np.int32)

    left_flank_len, left_flank_start = decode_flank_tf(state_seqs, 2 * c - 1, indices)

    core_blocks      = []
    insertion_lens   = []
    insertion_starts = []
    unannotated_segs = []
    core_starts      = []
    core_ends        = []
    finished_blocks  = []

    while True:
        core_starts.append(indices.copy())
        C, IL, IS, finished = decode_core_tf(c, state_seqs, indices)
        core_ends.append(indices.copy())
        core_blocks.append(C)
        insertion_lens.append(IL)
        insertion_starts.append(IS)
        finished_blocks.append(finished)

        if np.all(finished):
            break

        uns_len, uns_start = decode_flank_tf(state_seqs, 2 * c, indices)
        unannotated_segs.append((uns_len, uns_start))

    right_flank_len, right_flank_start = decode_flank_tf(state_seqs, 2 * c + 1, indices)

    # Compute num_repeats_per_row from finished_blocks
    max_R = len(core_blocks)
    finished_stack = np.stack(finished_blocks, axis=0)   # (max_R, n)
    first_finish   = np.argmax(finished_stack, axis=0)    # (n,)
    num_repeats_per_row = (first_finish + 1).astype(np.int32)

    row_offsets = np.concatenate([[0], np.cumsum(num_repeats_per_row)]).astype(np.int32)
    total_R = int(row_offsets[-1])
    M = core_blocks[0].shape[1] if max_R > 0 else 0

    domain_hit_flat  = np.full((total_R, M), -1, dtype=np.int16)
    domain_loc_flat  = np.full((total_R, 2), -1, dtype=np.int32)
    ins_lens_flat    = np.zeros((total_R, max(0, M - 1)), dtype=np.int16)
    ins_start_flat   = np.full((total_R, max(0, M - 1)), -1, dtype=np.int16)

    for r in range(max_R):
        rows     = np.where(num_repeats_per_row > r)[0]
        flat_idx = row_offsets[rows] + r
        domain_hit_flat[flat_idx]      = core_blocks[r][rows]
        domain_loc_flat[flat_idx, 0]   = core_starts[r][rows]
        domain_loc_flat[flat_idx, 1]   = core_ends[r][rows]
        ins_lens_flat[flat_idx]        = insertion_lens[r][rows]
        ins_start_flat[flat_idx]       = insertion_starts[r][rows]

    num_uns_per_row = np.maximum(num_repeats_per_row - 1, 0)
    uns_offsets = np.concatenate([[0], np.cumsum(num_uns_per_row)]).astype(np.int32)
    total_U = int(uns_offsets[-1])
    uns_len_flat   = np.zeros(total_U, dtype=np.int16)
    uns_start_flat = np.full(total_U, -1, dtype=np.int32)
    for r, (seg_len, seg_start) in enumerate(unannotated_segs):
        rows     = np.where(num_repeats_per_row > r + 1)[0]
        flat_idx = uns_offsets[rows] + r
        uns_len_flat[flat_idx]   = seg_len[rows]
        uns_start_flat[flat_idx] = seg_start[rows]

    return AlignmentMetaData(
        num_rows            = n,
        num_match           = c,
        num_repeats_per_row = num_repeats_per_row,
        domain_hit          = domain_hit_flat,
        domain_loc          = domain_loc_flat,
        insertion_lens      = ins_lens_flat,
        insertion_start     = ins_start_flat,
        left_flank_len      = left_flank_len,
        left_flank_start    = left_flank_start,
        right_flank_len     = right_flank_len,
        right_flank_start   = right_flank_start,
        unannotated_segments_len   = uns_len_flat,
        unannotated_segments_start = uns_start_flat,
    )
