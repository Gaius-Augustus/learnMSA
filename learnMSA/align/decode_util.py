"""Backend-neutral post-processing of decoded state paths.

The framework computes the raw dense decode output; these helpers turn it into
the flat numpy arrays that
:class:`~learnMSA.align.alignment_metadata.AlignmentMetaData` consumes, and
reorder them when the input was sorted. Pure numpy, no framework involved.
"""

import numpy as np



def reorder_decode_arrays(flat_dict: dict, sorted_order: np.ndarray) -> dict:
    """Reorder all arrays in a flat-dict (as returned by
    :func:`decode_batch_to_arrays`) so that new row ``i`` comes from old row
    ``sorted_order[i]``.

    Args:
        flat_dict: Dict produced by :func:`decode_batch_to_arrays`.
        sorted_order: 1-D int array of length ``num_rows``.

    Returns:
        New dict with the same keys but reordered rows.
    """
    perm = np.asarray(sorted_order, dtype=np.int32)
    n    = len(perm)

    nrpr_old = flat_dict['num_repeats_per_row']
    nrpr_new = nrpr_old[perm]

    old_row_off = np.concatenate([[0], np.cumsum(nrpr_old)]).astype(np.int32)
    new_row_off = np.concatenate([[0], np.cumsum(nrpr_new)]).astype(np.int32)
    total_R_new = int(new_row_off[-1])

    if total_R_new > 0:
        row_of_flat  = np.repeat(np.arange(n, dtype=np.int32), nrpr_new)
        local_of_flat = (
            np.arange(total_R_new, dtype=np.int32) - new_row_off[row_of_flat]
        )
        flat_order = old_row_off[perm[row_of_flat]] + local_of_flat
        dh_new = flat_dict['domain_hit'][flat_order]
        dl_new = flat_dict['domain_loc'][flat_order]
        il_new = flat_dict['insertion_lens'][flat_order]
        is_new = flat_dict['insertion_start'][flat_order]
    else:
        dh_new = flat_dict['domain_hit'][:0]
        dl_new = flat_dict['domain_loc'][:0]
        il_new = flat_dict['insertion_lens'][:0]
        is_new = flat_dict['insertion_start'][:0]

    uns_per_new = np.maximum(nrpr_new - 1, 0)
    uns_per_old = np.maximum(nrpr_old - 1, 0)
    new_uns_off = np.concatenate([[0], np.cumsum(uns_per_new)]).astype(np.int32)
    old_uns_off = np.concatenate([[0], np.cumsum(uns_per_old)]).astype(np.int32)
    total_U_new = int(new_uns_off[-1])

    if total_U_new > 0:
        uns_row = np.repeat(np.arange(n, dtype=np.int32), uns_per_new)
        uns_loc = (
            np.arange(total_U_new, dtype=np.int32) - new_uns_off[uns_row]
        )
        uns_flat_order = old_uns_off[perm[uns_row]] + uns_loc
        ul_new = flat_dict['unannotated_segments_len'][uns_flat_order]
        us_new = flat_dict['unannotated_segments_start'][uns_flat_order]
    else:
        ul_new = np.zeros(0, dtype=flat_dict['unannotated_segments_len'].dtype)
        us_new = np.zeros(0, dtype=flat_dict['unannotated_segments_start'].dtype)

    return dict(
        num_repeats_per_row        = nrpr_new,
        domain_hit                 = dh_new,
        domain_loc                 = dl_new,
        insertion_lens             = il_new,
        insertion_start            = is_new,
        left_flank_len             = flat_dict['left_flank_len'][perm],
        left_flank_start           = flat_dict['left_flank_start'][perm],
        right_flank_len            = flat_dict['right_flank_len'][perm],
        right_flank_start          = flat_dict['right_flank_start'][perm],
        unannotated_segments_len   = ul_new,
        unannotated_segments_start = us_new,
    )


def decode_batch_to_arrays(outputs_np: tuple, model_length: int) -> dict:
    """Convert the dense GPU output of :func:`_get_decode_batch_fn` to the
    flat numpy arrays used by
    :class:`~learnMSA.align.alignment_metadata.AlignmentMetaData`.

    Args:
        outputs_np: 12-tuple ``(lfl, lfs, cc, il, is_, cs, ce, fin, ul, us,
            rfl, rfs)`` – numpy arrays from ``.numpy()`` on each tensor.
        model_length: Number of match states ``c``.

    Returns:
        dict with keys matching :class:`AlignmentMetaData` array fields
        (no ``num_rows``, ``num_match``).
    """
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

    return dict(
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
