"""Tests for the insertion-slice column maps.

These guard the memory rewrite of :mod:`learnMSA.align.align_inserts`: the
per-slice row lookup became sparse (a binary search over the contributing rows)
instead of a dense array with one entry per sequence in the dataset, and the
column maps are now read straight out of the aligner output instead of via an
:class:`~learnMSA.util.aligned_dataset.AlignedDataset`.
"""

import numpy as np
import pytest

from learnMSA.align.align_inserts import (AlignedInsertions, SliceColumns,
                                          _slice_columns)
from learnMSA.align.alignment_metadata import AlignmentMetaData
from learnMSA.util.aligned_dataset import AlignedDataset


GAPPED = [
    "AC-DE--",
    "-CID---",
    "A--DEFG",
    "-------",
    "AAAAAAA",
]


def _dense_custom_columns(batch_indices, index_map, sparse_cols, max_len):
    """The pre-rewrite lookup, kept here as the reference implementation."""
    result = np.tile(
        np.arange(max_len, dtype=np.int32), (batch_indices.shape[0], 1)
    )
    local_rows = index_map[batch_indices]
    has_ins = local_rows >= 0
    if np.any(has_ins):
        result[has_ins] = sparse_cols[local_rows[has_ins]]
    return result


def test_slice_columns_matches_aligned_dataset() -> None:
    """The column map must agree with the AlignedDataset it replaced."""
    gapped = [g for g in GAPPED if g.strip("-")]  # AlignedDataset rejects empty
    rows = np.array([3, 11, 12, 40])
    sc = _slice_columns(rows, gapped)

    data = AlignedDataset(
        sequences=[(f"s{i}", g) for i, g in enumerate(gapped)]
    )
    assert sc.width == data.alignment_len
    for i in range(len(gapped)):
        expected = data.get_column_map(i)
        np.testing.assert_array_equal(sc.cols[i, :expected.size], expected)
    np.testing.assert_array_equal(sc.rows, rows)
    assert sc.rows.dtype == np.int32


def test_slice_columns_treats_dots_as_gaps() -> None:
    sc = _slice_columns(np.array([0, 1]), ["A.C", "-BC"])
    np.testing.assert_array_equal(sc.cols[0, :2], [0, 2])
    np.testing.assert_array_equal(sc.cols[1, :2], [1, 2])


def test_slice_columns_is_compact() -> None:
    """Rows are only as wide as the longest fragment, not the whole MSA."""
    sc = _slice_columns(np.array([0, 1, 2]), ["A" + "-" * 50, "-B" + "-" * 49,
                                              "--C" + "-" * 48])
    assert sc.width == 51
    assert sc.cols.shape == (3, 1)
    assert sc.cols.dtype == np.int16


@pytest.mark.parametrize("batch", [
    np.arange(0, 6), np.arange(6, 12), np.arange(3, 9), np.arange(0, 20),
    np.array([0]), np.array([19]),
])
def test_custom_columns_match_dense_lookup(batch) -> None:
    """Sparse row lookup must reproduce the dense index map exactly."""
    n_total = 20
    rows = np.array([1, 4, 5, 11, 19], dtype=np.int32)
    sc = _slice_columns(rows, GAPPED)

    index_map = np.full(n_total, -1, dtype=np.int32)
    index_map[rows] = np.arange(rows.size, dtype=np.int32)

    ai = AlignedInsertions(n_total, aligned_left_flank=sc)
    got = ai.left_flank(batch)
    expected = _dense_custom_columns(
        batch, index_map, sc.cols, sc.cols.shape[1]
    )
    np.testing.assert_array_equal(got, expected)


def test_custom_columns_identity_for_missing_rows() -> None:
    rows = np.array([2], dtype=np.int32)
    sc = SliceColumns(rows, np.array([[5, 7, 9]], dtype=np.int16), 12)
    ai = AlignedInsertions(4, aligned_right_flank=sc)
    got = ai.right_flank(np.arange(4))
    np.testing.assert_array_equal(got[0], [0, 1, 2])   # not in the slice
    np.testing.assert_array_equal(got[2], [5, 7, 9])   # in the slice
    assert ai.ext_right_flank == 12


def test_empty_aligned_insertions_returns_none() -> None:
    ai = AlignedInsertions()
    batch = np.arange(3)
    assert ai.left_flank(batch) is None
    assert ai.right_flank(batch) is None
    assert ai.insertion(batch, 0) is None
    assert ai.unannotated_segment(batch, 0) is None
    assert ai.ext_insertions == 0
    assert ai.ext_left_flank == 0
    assert ai.ext_right_flank == 0
    assert ai.ext_unannotated == 0


def test_insertion_per_position_matches_full_list() -> None:
    """Materializing one position must equal the corresponding list entry."""
    sc = _slice_columns(np.array([0, 2, 3, 5, 7]), GAPPED)
    ai = AlignedInsertions(10, aligned_insertions=[[sc, None, sc]])
    batch = np.arange(10)
    full = ai.insertion(batch, 0)
    assert full[1] is None
    for i, entry in enumerate(full):
        single = ai.insertion(batch, 0, i)
        if entry is None:
            assert single is None
        else:
            np.testing.assert_array_equal(single, entry)
    np.testing.assert_array_equal(ai.ext_insertions, [[sc.width, 0, sc.width]])


def _meta(num_rows=6, num_match=4) -> AlignmentMetaData:
    rng = np.random.default_rng(0)
    nrpr = np.array([1, 2, 1, 3, 1, 2], dtype=np.int32)[:num_rows]
    total = int(nrpr.sum())
    return AlignmentMetaData(
        num_rows=num_rows,
        num_match=num_match,
        num_repeats_per_row=nrpr,
        domain_hit=rng.integers(-1, 30, (total, num_match)).astype(np.int16),
        domain_loc=rng.integers(0, 30, (total, 2)).astype(np.int16),
        insertion_lens=rng.integers(0, 30, (total, num_match - 1)).astype(np.int16),
        insertion_start=rng.integers(0, 30, (total, num_match - 1)).astype(np.int16),
        left_flank_len=rng.integers(0, 9, num_rows).astype(np.int16),
        left_flank_start=np.zeros(num_rows, np.int16),
        right_flank_len=rng.integers(0, 9, num_rows).astype(np.int16),
        right_flank_start=rng.integers(0, 9, num_rows).astype(np.int16),
        unannotated_segments_len=rng.integers(
            0, 9, total - num_rows).astype(np.int16),
        unannotated_segments_start=rng.integers(
            0, 9, total - num_rows).astype(np.int16),
    )


def test_get_repeat_insertions_matches_get_repeat_data() -> None:
    """The narrow accessor must equal a column slice of the wide one."""
    meta = _meta()
    rows = np.arange(meta.num_rows)
    for r in range(meta.num_repeats):
        _, il, is_, _, _ = meta.get_repeat_data(r, rows)
        for c in range(meta.num_match - 1):
            il_c, is_c = meta.get_repeat_insertions(r, rows, c)
            np.testing.assert_array_equal(il_c, il[:, c])
            np.testing.assert_array_equal(is_c, is_[:, c])
            assert il_c.dtype == il.dtype
            assert is_c.dtype == is_.dtype


def test_get_repeat_insertions_subset_of_rows() -> None:
    meta = _meta()
    rows = np.array([5, 1, 3])
    _, il, is_, _, _ = meta.get_repeat_data(1, rows)
    il_c, is_c = meta.get_repeat_insertions(1, rows, 2)
    np.testing.assert_array_equal(il_c, il[:, 2])
    np.testing.assert_array_equal(is_c, is_[:, 2])


def test_get_repeat_insertions_does_not_mutate_metadata() -> None:
    meta = _meta()
    before = meta.insertion_lens.copy()
    meta.get_repeat_insertions(2, np.arange(meta.num_rows), 0)
    np.testing.assert_array_equal(meta.insertion_lens, before)


def _reference_reductions(meta):
    """The pre-rewrite whole-alignment reductions, un-chunked."""
    R, M = meta.num_repeats, meta.num_match
    virt_rep, _ = meta._flat_virt_rep_and_row()

    occ = np.zeros((R, M), dtype=np.int8)
    if len(meta.domain_hit):
        np.maximum.at(occ, virt_rep, (meta.domain_hit != -1).astype(np.int8))

    ins = np.zeros(R * max(0, M - 1), dtype=np.int32)
    if len(meta.insertion_lens) and M > 1:
        lin = (virt_rep[:, None] * (M - 1)
               + np.arange(M - 1, dtype=np.int32)[None, :]).ravel()
        np.maximum.at(ins, lin, meta.insertion_lens.astype(np.int32).ravel())
    return occ.astype(bool), ins.reshape(R, max(0, M - 1))


@pytest.mark.parametrize("chunk", [8, 64, 1 << 20])
def test_chunked_reductions_match_reference(chunk, monkeypatch) -> None:
    """Chunking must not change the reduction, at any chunk size."""
    import learnMSA.align.alignment_metadata as amd
    monkeypatch.setattr(amd, "_REDUCE_CHUNK_BYTES", chunk)
    meta = _meta(num_rows=6, num_match=5)
    occ_ref, ins_ref = _reference_reductions(meta)
    np.testing.assert_array_equal(meta.repeat_occupancy_mask(), occ_ref)
    np.testing.assert_array_equal(meta.insertion_lens_total, ins_ref)
    assert meta.repeat_occupancy_mask().dtype == np.bool_
    assert meta.insertion_lens_total.dtype == np.int32


def test_reductions_on_empty_metadata() -> None:
    meta = AlignmentMetaData(
        num_rows=0, num_match=3,
        num_repeats_per_row=np.zeros(0, np.int32),
        domain_hit=np.zeros((0, 3), np.int16),
        domain_loc=np.zeros((0, 2), np.int16),
        insertion_lens=np.zeros((0, 2), np.int16),
        insertion_start=np.zeros((0, 2), np.int16),
        left_flank_len=np.zeros(0, np.int16),
        left_flank_start=np.zeros(0, np.int16),
        right_flank_len=np.zeros(0, np.int16),
        right_flank_start=np.zeros(0, np.int16),
        unannotated_segments_len=np.zeros(0, np.int16),
        unannotated_segments_start=np.zeros(0, np.int16),
    )
    assert meta.repeat_occupancy_mask().shape == (0, 3)
    assert meta.insertion_lens_total.shape == (0, 2)
    assert meta.get_repeat_insertions(0, np.arange(0), 0)[0].shape == (0,)
