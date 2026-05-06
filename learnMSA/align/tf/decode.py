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
    """Shared TF logic for decode_core.

    Args:
        state_seqs: ``(n, T)`` int32 tensor.
        c: Python int – number of match states.  Used as a Python constant so
           that ``tf.function(jit_compile=True)`` can unroll loops and infer
           static output shapes.
        indices: ``(n,)`` int32 tensor of per-sequence start positions.
    """
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

    # ---- consensus columns, insertion lengths, insertion starts -----------
    # All three use per-state Python loops (unrolled at XLA trace time) so
    # that every intermediate tensor has a statically-known shape.  The
    # previous tf.where(is_match/is_insert) approach returned variable-length
    # coordinate tensors, which forced XLA into slow dynamic-shape handling.
    cc_list, il_list, ins_start_list = [], [], []
    for j in range(c):
        is_j   = in_core & tf.equal(s, j)                                   # (n, T)
        has_j  = tf.reduce_any(is_j, axis=1)                                # (n,)
        first_j = tf.cast(
            tf.argmax(tf.cast(is_j, tf.int8), axis=1), tf.int32
        )                                                                    # (n,)
        cc_list.append(
            tf.where(has_j, first_j, tf.fill([n], tf.cast(-1, tf.int32)))
        )
    consensus_columns = tf.stack(cc_list, axis=1)                           # (n, c)

    for j in range(c - 1):
        is_j = is_insert & tf.equal(s, c + j)                               # (n, T)
        il_list.append(tf.reduce_sum(tf.cast(is_j, tf.int32), axis=1))      # (n,)
        has_j  = tf.reduce_any(is_j, axis=1)                                # (n,)
        first_j = tf.cast(
            tf.argmax(tf.cast(is_j, tf.int8), axis=1), tf.int32
        )                                                                    # (n,)
        ins_start_list.append(
            tf.where(has_j, first_j, tf.fill([n], tf.cast(-1, tf.int32)))
        )
    if c > 1:
        insertion_lens  = tf.stack(il_list,        axis=1)                  # (n, c-1)
        insertion_start = tf.stack(ins_start_list, axis=1)                  # (n, c-1)
    else:
        insertion_lens  = tf.zeros([n, 0], dtype=tf.int32)
        insertion_start = tf.zeros([n, 0], dtype=tf.int32)

    # ---- finished flag ------------------------------------------------------
    # finished = True when the sequence reached a right-flank / end state
    # OR when no terminal state was found at all (sequence ended with padding,
    # meaning this was the last repeat block).
    ep_clamped   = tf.minimum(end_pos, T - 1)                               # (n,)
    row_indices  = tf.range(n)
    gather_idx   = tf.stack([row_indices, ep_clamped], axis=1)               # (n, 2)
    finished     = tf.logical_not(has_terminal) | tf.gather_nd(is_at_end, gather_idx)  # (n,)

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
    # model_length is passed as Python int so _tf_decode_core_inner can use
    # it as a compile-time constant for static shape inference.
    c             = int(model_length)
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


# ---------------------------------------------------------------------------
# Batch-decode infrastructure for OOM-safe GPU processing
# ---------------------------------------------------------------------------

def _count_repeats_batch(batch, unann_state):
    """Per-batch repeat counter (no decorator; wrap via _get_count_fn).

    ``unann_state`` is a scalar int32 tensor baked in at wrapping time so
    ``tf.function`` never retraces due to different model lengths.
    """
    is_unann = tf.equal(batch, unann_state)
    entries  = tf.logical_and(
        tf.logical_not(is_unann[:, :-1]),
        is_unann[:, 1:],
    )
    return tf.reduce_sum(tf.cast(entries, tf.int32), axis=1) + 1  # (B,)


# Cache of tf.function-wrapped counters keyed by (c, jit_compile).
_COUNT_FN_CACHE: dict = {}


def _get_count_fn(c: int, jit_compile: bool):
    """Return (and cache) a ``tf.function``-wrapped repeat counter that has
    the unannotated-state id ``2*c`` baked in as a Python constant.

    Baking ``c`` prevents retracing when the same function object is called
    with different model lengths; a separate compiled function is stored per
    ``(c, jit_compile)`` pair instead.
    """
    key = (c, jit_compile)
    if key in _COUNT_FN_CACHE:
        return _COUNT_FN_CACHE[key]
    unann = 2 * c  # baked Python constant
    def _count(batch):
        return _count_repeats_batch(batch, tf.constant(unann, dtype=tf.int32))
    # jit_compile=False: the count op is just tf.equal + tf.reduce_sum —
    # no XLA benefit.  With a single model and single bucket the concrete
    # (B, T) shape is always the same, so one trace is all we ever get.
    _COUNT_FN_CACHE[key] = tf.function(_count, jit_compile=False)
    return _COUNT_FN_CACHE[key]


def _tf_count_repeats(state_seqs_t: tf.Tensor, c: int,
                      batch_size: int = 512,
                      jit_compile: bool = True) -> np.ndarray:
    """Count repeats per sequence by detecting unannotated-state transitions.

    Processes sequences in batches padded to ``batch_size`` so that the
    compiled kernel is traced only once per unique (batch_size, T) shape.

    Args:
        state_seqs_t: ``(n, T)`` int32 tensor (GPU or CPU).
        c: Python int – model length.
        batch_size: Sequences per GPU batch (also used as static batch dim).
        jit_compile: Whether to JIT-compile the counting kernel.

    Returns:
        ``(n,)`` int32 numpy array with the number of repeats per sequence.
    """
    n        = int(state_seqs_t.shape[0])
    T        = int(state_seqs_t.shape[1])
    pad_rows = tf.zeros([batch_size, T], dtype=tf.int32)
    count_fn = _get_count_fn(c, jit_compile)
    result   = np.empty(n, dtype=np.int32)
    for start in range(0, n, batch_size):
        end  = min(start + batch_size, n)
        B    = end - start
        rows = state_seqs_t[start:end]
        if B < batch_size:
            rows = tf.concat([rows, pad_rows[:batch_size - B]], axis=0)
        result[start:end] = count_fn(rows).numpy()[:B]
    return result


# Cache of compiled decode functions keyed by (num_repeats, model_length, jit_compile).
_DECODE_BATCH_CACHE: dict = {}


def _get_decode_batch_fn(num_repeats: int, model_length: int,
                         jit_compile: bool = True):
    """Return (and cache) a compiled decode function for exactly
    ``num_repeats`` repeats and ``model_length`` match states.

    The returned callable takes a single ``(n, T)`` int32 tensor and returns
    a tuple of tensors kept on the GPU until ``.numpy()`` is called by the
    caller.  The Python-level loop over repeats is unrolled at trace time so
    XLA can fuse the entire decode into a single kernel.

    Returns:
        A ``tf.function``-wrapped callable ``fn(state_seqs_t)`` whose output
        is a 12-tuple::

            (left_flank_len,    # (n,) int32
             left_flank_start,  # (n,) int32
             cc_stack,          # (R, n, c) int32
             il_stack,          # (R, n, c-1) int32
             is_stack,          # (R, n, c-1) int32
             cs_stack,          # (R, n) int32  – core start positions
             ce_stack,          # (R, n) int32  – core end positions
             fin_stack,         # (R, n) bool   – finished flag per repeat
             ul_stack,          # (R-1, n) int32 or shape-(0,) for R=1
             us_stack,          # (R-1, n) int32 or shape-(0,) for R=1
             right_flank_len,   # (n,) int32
             right_flank_start) # (n,) int32
    """
    key = (num_repeats, model_length, jit_compile)
    if key in _DECODE_BATCH_CACHE:
        return _DECODE_BATCH_CACHE[key]

    R = num_repeats
    c = model_length

    def _decode_batch(state_seqs):
        # Left flank ---------------------------------------------------------
        lfl, lfs, indices = _tf_decode_flank_core(state_seqs, 2 * c - 1,
                                                   tf.zeros([tf.shape(state_seqs)[0]],
                                                            dtype=tf.int32))

        # Core blocks (R) and unannotated segments (R-1) ---------------------
        core_cc, core_il, core_is_, core_cs, core_ce, core_fin = [], [], [], [], [], []
        uns_l, uns_s = [], []

        for r in range(R):
            core_cs.append(indices)
            cc, il, is_, fin, indices = _tf_decode_core_inner(
                state_seqs, c, indices
            )
            core_cc.append(cc)
            core_il.append(il)
            core_is_.append(is_)
            core_ce.append(indices)
            core_fin.append(fin)

            if r < R - 1:
                ul, us, indices = _tf_decode_flank_core(state_seqs, 2 * c, indices)
                uns_l.append(ul)
                uns_s.append(us)

        # Right flank --------------------------------------------------------
        rfl, rfs, _ = _tf_decode_flank_core(state_seqs, 2 * c + 1, indices)

        # Stack results ------------------------------------------------------
        cc_stack  = tf.stack(core_cc,  axis=0)   # (R, n, c)
        il_stack  = tf.stack(core_il,  axis=0)   # (R, n, c-1)
        is_stack  = tf.stack(core_is_, axis=0)   # (R, n, c-1)
        cs_stack  = tf.stack(core_cs,  axis=0)   # (R, n)
        ce_stack  = tf.stack(core_ce,  axis=0)   # (R, n)
        fin_stack = tf.stack(core_fin, axis=0)   # (R, n)

        if R > 1:
            ul_stack = tf.stack(uns_l, axis=0)  # (R-1, n)
            us_stack = tf.stack(uns_s, axis=0)  # (R-1, n)
        else:
            # No unannotated segments for single-repeat sequences.
            # Return empty 1-D tensors (shape (0,)) as placeholders.
            ul_stack = tf.constant([], dtype=tf.int32)
            us_stack = tf.constant([], dtype=tf.int32)

        return lfl, lfs, cc_stack, il_stack, is_stack, cs_stack, ce_stack, \
               fin_stack, ul_stack, us_stack, rfl, rfs

    # No reduce_retracing: with a single model and a single bucket all
    # batches share the same concrete (B, T) shape, so TF traces once and
    # XLA receives exact dimensions for tighter kernel specialisation.
    _DECODE_BATCH_CACHE[key] = tf.function(_decode_batch, jit_compile=jit_compile)
    return _DECODE_BATCH_CACHE[key]


def decode_batch_to_meta(outputs_np: tuple, model_length: int):
    """Assemble :class:`~learnMSA.align.alignment_metadata.AlignmentMetaData`
    from the dense numpy arrays produced by :func:`_get_decode_batch_fn`.

    This is the CPU post-processing step.  It extracts the *valid* repeat
    entries (determined by the ``finished`` flag) from the padded
    ``(R_max, B, ...)`` arrays and builds the flat storage layout expected by
    :class:`~learnMSA.align.alignment_metadata.AlignmentMetaData`.

    Args:
        outputs_np: 12-tuple of numpy arrays as returned by calling
            :func:`_get_decode_batch_fn` and converting each element with
            ``.numpy()``:

            ``(lfl, lfs, cc, il, is_, cs, ce, fin, ul, us, rfl, rfs)``

            where shapes (with ``R = num_repeats``, ``B = batch``,
            ``c = model_length``) are:

            * ``lfl``  : (B,) int32  – left-flank lengths
            * ``lfs``  : (B,) int32  – left-flank starts
            * ``cc``   : (R, B, c)   int32
            * ``il``   : (R, B, c-1) int32
            * ``is_``  : (R, B, c-1) int32
            * ``cs``   : (R, B)      int32
            * ``ce``   : (R, B)      int32
            * ``fin``  : (R, B)      bool
            * ``ul``   : (R-1, B) or (0,) int32  – unannotated-seg lengths
            * ``us``   : (R-1, B) or (0,) int32  – unannotated-seg starts
            * ``rfl``  : (B,) int32  – right-flank lengths
            * ``rfs``  : (B,) int32  – right-flank starts

        model_length: Number of match states ``c``.

    Returns:
        :class:`~learnMSA.align.alignment_metadata.AlignmentMetaData` with
        ``sort_perm=None``.
    """
    from learnMSA.align.alignment_metadata import AlignmentMetaData

    lfl, lfs, cc, il, is_, cs, ce, fin, ul, us, rfl, rfs = outputs_np
    c  = int(model_length)
    R  = cc.shape[0]   # num_repeats (dense upper bound)
    B  = cc.shape[1]   # batch size (may include padding rows)

    # num_repeats_per_row[i] = index of first True in fin[:, i] + 1
    # fin is (R, B) bool; argmax on axis=0 finds the first True.
    # If no repeat is finished (all False), argmax returns 0 — so we use
    # tf.reduce_any to detect this case and clamp to R.
    has_any_finished = np.any(fin, axis=0)          # (B,)
    first_finished   = np.argmax(fin, axis=0)        # (B,) index of first True
    num_repeats_per_row = np.where(
        has_any_finished, first_finished + 1, R
    ).astype(np.int32)                               # (B,)

    # Build flat output arrays.
    row_off  = np.concatenate([[0], np.cumsum(num_repeats_per_row)]).astype(np.int32)
    total_R  = int(row_off[-1])
    uns_per  = np.maximum(num_repeats_per_row - 1, 0)
    uns_off  = np.concatenate([[0], np.cumsum(uns_per)]).astype(np.int32)
    total_U  = int(uns_off[-1])
    M        = c

    # -----------------------------------------------------------------------
    # Assemble flat output arrays – vectorized, no Python loop over B.
    # -----------------------------------------------------------------------
    if np.all(num_repeats_per_row == R):
        # Fast path: every sequence has exactly R repeats.  Arrays can be
        # transposed and reshaped without any masked scatter.
        if R == 1:
            domain_hit_flat  = cc[0].astype(np.int16)                          # (B, c)
            domain_loc_flat  = np.stack(
                [cs[0], ce[0]], axis=1
            ).astype(np.int32)                                                  # (B, 2)
            if M > 1:
                ins_lens_flat  = il[0].astype(np.int16)                        # (B, c-1)
                ins_start_flat = is_[0].astype(np.int16)                       # (B, c-1)
            else:
                ins_lens_flat  = np.zeros((B, 0), dtype=np.int16)
                ins_start_flat = np.zeros((B, 0), dtype=np.int16)
            uns_len_flat   = np.zeros(0, dtype=np.int16)
            uns_start_flat = np.zeros(0, dtype=np.int32)
        else:
            # (R, B, c) -> transpose (B, R, c) -> reshape (B*R, c)
            domain_hit_flat  = cc.transpose(1, 0, 2).reshape(
                total_R, c).astype(np.int16)
            domain_loc_flat  = np.stack(
                [cs, ce], axis=2
            ).transpose(1, 0, 2).reshape(total_R, 2).astype(np.int32)
            if M > 1:
                ins_lens_flat  = il.transpose(1, 0, 2).reshape(
                    total_R, M - 1).astype(np.int16)
                ins_start_flat = is_.transpose(1, 0, 2).reshape(
                    total_R, M - 1).astype(np.int16)
            else:
                ins_lens_flat  = np.zeros((total_R, 0), dtype=np.int16)
                ins_start_flat = np.zeros((total_R, 0), dtype=np.int16)
            if R > 1:
                # ul: (R-1, B) -> transpose (B, R-1) -> ravel (B*(R-1),)
                uns_len_flat   = ul.T.ravel().astype(np.int16)
                uns_start_flat = us.T.ravel().astype(np.int32)
            else:
                uns_len_flat   = np.zeros(0, dtype=np.int16)
                uns_start_flat = np.zeros(0, dtype=np.int32)
    else:
        # Slow path: sequences have different repeat counts.  Use vectorized
        # numpy scatter (np.where returns valid index pairs in one call).
        r_arr  = np.arange(R, dtype=np.int32)[:, np.newaxis]   # (R, 1)
        valid  = r_arr < num_repeats_per_row[np.newaxis, :]     # (R, B)
        valid_r, valid_i = np.where(valid)                      # (total_R,) each
        flat_idx = row_off[valid_i] + valid_r

        domain_hit_flat  = np.full((total_R, M), -1, dtype=np.int16)
        domain_loc_flat  = np.full((total_R, 2),  -1, dtype=np.int32)
        domain_hit_flat[flat_idx]    = cc[valid_r, valid_i].astype(np.int16)
        domain_loc_flat[flat_idx, 0] = cs[valid_r, valid_i]
        domain_loc_flat[flat_idx, 1] = ce[valid_r, valid_i]

        if M > 1:
            ins_lens_flat  = np.zeros((total_R, M - 1), dtype=np.int16)
            ins_start_flat = np.full((total_R, M - 1), -1, dtype=np.int16)
            ins_lens_flat[flat_idx]  = il[valid_r, valid_i].astype(np.int16)
            ins_start_flat[flat_idx] = is_[valid_r, valid_i].astype(np.int16)
        else:
            ins_lens_flat  = np.zeros((total_R, 0), dtype=np.int16)
            ins_start_flat = np.zeros((total_R, 0), dtype=np.int16)

        if R > 1 and total_U > 0:
            r_uns = np.arange(R - 1, dtype=np.int32)[:, np.newaxis]
            valid_uns         = r_uns < (num_repeats_per_row[np.newaxis, :] - 1)
            vur, vui          = np.where(valid_uns)
            uf_idx            = uns_off[vui] + vur
            uns_len_flat      = np.zeros(total_U, dtype=np.int16)
            uns_start_flat    = np.zeros(total_U, dtype=np.int32)
            uns_len_flat[uf_idx]   = ul[vur, vui].astype(np.int16)
            uns_start_flat[uf_idx] = us[vur, vui].astype(np.int32)
        else:
            uns_len_flat   = np.zeros(0, dtype=np.int16)
            uns_start_flat = np.zeros(0, dtype=np.int32)

    return AlignmentMetaData(
        num_rows                   = B,
        num_match                  = c,
        num_repeats_per_row        = num_repeats_per_row,
        domain_hit                 = domain_hit_flat,
        domain_loc                 = domain_loc_flat,
        insertion_lens             = ins_lens_flat,
        insertion_start            = ins_start_flat,
        left_flank_len             = lfl.astype(np.int16),
        left_flank_start           = lfs.astype(np.int32),
        right_flank_len            = rfl.astype(np.int16),
        right_flank_start          = rfs.astype(np.int32),
        unannotated_segments_len   = uns_len_flat,
        unannotated_segments_start = uns_start_flat,
    )


def decode_tf(model_length: int, state_seqs, batch_size: int = 2048):
    """OOM-safe TensorFlow decode.

    Strategy
    --------
    1. **Count repeats** in a single O(n × T) GPU pass by detecting
       transitions into the unannotated state.
    2. **Sort** sequences by repeat count so all sequences in a batch have
       the same number of repeats – enabling a fully static loop that XLA can
       compile once per ``(num_repeats, model_length)`` pair.
    3. **Batch** over sequences to keep GPU memory bounded.
    4. Collect CPU results and construct
       :class:`~learnMSA.align.alignment_metadata.AlignmentMetaData` with
       ``sort_perm`` stored so that callers can map original-index queries
       back to the sorted storage layout.

    Args:
        model_length: Number of match states.
        state_seqs: ``(n, T)`` int32 array/tensor of Viterbi/MEA paths
            (single model, already sliced to ``[:, :, model_idx]``).
        batch_size: Maximum number of sequences per GPU batch.

    Returns:
        :class:`~learnMSA.align.alignment_metadata.AlignmentMetaData`
    """
    from learnMSA.align.alignment_metadata import AlignmentMetaData

    if isinstance(state_seqs, tf.Tensor):
        state_seqs = state_seqs.numpy()

    n = state_seqs.shape[0]
    T = state_seqs.shape[1]
    c = int(model_length)

    # ------------------------------------------------------------------
    # Step 1 – count repeats per sequence (single GPU pass)
    # ------------------------------------------------------------------
    state_seqs_t = tf.constant(state_seqs, dtype=tf.int32)
    num_repeats_per_row = _tf_count_repeats(state_seqs_t, c, batch_size)   # (n,) CPU

    # ------------------------------------------------------------------
    # Step 2 – sort by repeat count (stable, so equal-repeat groups are
    # contiguous and within a group the original order is preserved)
    # ------------------------------------------------------------------
    sort_perm   = np.argsort(num_repeats_per_row, kind='stable').astype(np.int32)
    sorted_nrpr = num_repeats_per_row[sort_perm]               # (n,) sorted

    # Pre-compute flat offsets in the sorted layout
    sorted_row_off = np.concatenate(
        [[0], np.cumsum(sorted_nrpr)]
    ).astype(np.int32)
    sorted_uns_per_row = np.maximum(sorted_nrpr - 1, 0)
    sorted_uns_off = np.concatenate(
        [[0], np.cumsum(sorted_uns_per_row)]
    ).astype(np.int32)

    total_R = int(sorted_row_off[-1])
    total_U = int(sorted_uns_off[-1])
    M       = c

    # ------------------------------------------------------------------
    # Step 3 – allocate output arrays (sorted layout)
    # ------------------------------------------------------------------
    domain_hit_flat  = np.full((total_R, M),          -1, dtype=np.int16)
    domain_loc_flat  = np.full((total_R, 2),           -1, dtype=np.int32)
    ins_lens_flat    = np.zeros((total_R, max(0, M - 1)),   dtype=np.int16)
    ins_start_flat   = np.full((total_R, max(0, M - 1)), -1, dtype=np.int16)

    left_flank_len_s   = np.zeros(n, dtype=np.int16)
    left_flank_start_s = np.zeros(n, dtype=np.int32)
    right_flank_len_s  = np.zeros(n, dtype=np.int16)
    right_flank_start_s= np.zeros(n, dtype=np.int32)

    uns_len_flat   = np.zeros(total_U, dtype=np.int16)
    uns_start_flat = np.full(total_U, -1, dtype=np.int32)

    # ------------------------------------------------------------------
    # Step 4 – process each unique-R group in batches
    # ------------------------------------------------------------------
    unique_Rs = np.unique(sorted_nrpr)
    for R_val in unique_Rs:
        R_int     = int(R_val)
        decode_fn = _get_decode_batch_fn(R_int, c, jit_compile=True)
        group_idx = np.where(sorted_nrpr == R_val)[0]   # indices in sorted order

        for b_start in range(0, len(group_idx), batch_size):
            batch_sorted = group_idx[b_start : b_start + batch_size]
            orig_idx     = sort_perm[batch_sorted]       # original sequence indices
            B            = len(batch_sorted)

            # GPU decode – pad to batch_size so XLA sees a constant shape
            rows = state_seqs[orig_idx]
            if B < batch_size:
                rows = np.concatenate(
                    [rows, np.zeros((batch_size - B, T), dtype=np.int32)], axis=0
                )
            batch_t = tf.constant(rows, dtype=tf.int32)
            (lfl_t, lfs_t, cc_t, il_t, is_t, cs_t, ce_t,
             fin_t, ul_t, us_t, rfl_t, rfs_t) = decode_fn(batch_t)

            # Transfer to CPU (slice [:B] to discard padding rows)
            lfl = lfl_t.numpy()[:B].astype(np.int16)
            lfs = lfs_t.numpy()[:B].astype(np.int32)
            cc  = cc_t.numpy()[:, :B].astype(np.int16)  # (R, B, c)
            il  = il_t.numpy()[:, :B].astype(np.int16)  # (R, B, c-1)
            is_ = is_t.numpy()[:, :B].astype(np.int16)  # (R, B, c-1)
            cs  = cs_t.numpy()[:, :B].astype(np.int32)  # (R, B)
            ce  = ce_t.numpy()[:, :B].astype(np.int32)  # (R, B)
            rfl = rfl_t.numpy()[:B].astype(np.int16)
            rfs = rfs_t.numpy()[:B].astype(np.int32)

            # Per-row arrays (indexed by sorted position)
            left_flank_len_s[batch_sorted]    = lfl
            left_flank_start_s[batch_sorted]  = lfs
            right_flank_len_s[batch_sorted]   = rfl
            right_flank_start_s[batch_sorted] = rfs

            # Flat repeat arrays
            for r in range(R_int):
                flat_idx = sorted_row_off[batch_sorted] + r
                domain_hit_flat[flat_idx]    = cc[r]
                domain_loc_flat[flat_idx, 0] = cs[r]
                domain_loc_flat[flat_idx, 1] = ce[r]
                ins_lens_flat[flat_idx]       = il[r]
                ins_start_flat[flat_idx]      = is_[r]

            # Unannotated segment arrays
            if R_int > 1:
                ul = ul_t.numpy()[:, :B].astype(np.int16)  # (R-1, B)
                us = us_t.numpy()[:, :B].astype(np.int32)  # (R-1, B)
                for r in range(R_int - 1):
                    uf_idx = sorted_uns_off[batch_sorted] + r
                    uns_len_flat[uf_idx]   = ul[r]
                    uns_start_flat[uf_idx] = us[r]

    return AlignmentMetaData(
        num_rows             = n,
        num_match            = c,
        num_repeats_per_row  = sorted_nrpr,
        sort_perm            = sort_perm,
        domain_hit           = domain_hit_flat,
        domain_loc           = domain_loc_flat,
        insertion_lens       = ins_lens_flat,
        insertion_start      = ins_start_flat,
        left_flank_len       = left_flank_len_s,
        left_flank_start     = left_flank_start_s,
        right_flank_len      = right_flank_len_s,
        right_flank_start    = right_flank_start_s,
        unannotated_segments_len   = uns_len_flat,
        unannotated_segments_start = uns_start_flat,
    )
