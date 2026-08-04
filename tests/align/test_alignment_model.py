"""Tests for ``AlignmentModel`` -- decoding, output formats and metadata.

``learnMSA/align/alignment_model.py`` is backend-neutral, so these run under
whichever backend is installed; the fixtures in ``conftest.py`` reach the model
through :func:`make_learnmsa_model`. The tests that drive the fused TensorFlow
decoder live in ``tf/test_decode.py`` instead.
"""

import os

import numpy as np

from learnMSA import Configuration
from learnMSA.align.align import align
from learnMSA.align.align_hits import HitAlignmentMode
from learnMSA.align.alignment_metadata import AlignmentMetaData
from learnMSA.align.alignment_model import AlignmentModel
from learnMSA.model.model import LearnMSAModel
from learnMSA.util.aligned_dataset import AlignedDataset, SequenceDataset

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def test_subalignment(
    simple_data : SequenceDataset,
    simple_model : LearnMSAModel,
) -> None:
    """Test extraction of subalignments from AlignmentModel"""
    # subalignment
    subset = np.array([0, 2, 5])
    # create alignment after building model
    sub_am = AlignmentModel(simple_data, simple_model, subset)
    subalignment_strings = sub_am.to_string(0, add_block_sep=False)
    ref_subalignment = ["FE...LIK...", "FE...LIKhac", "FEahcLIK..."]
    for s, r in zip(subalignment_strings, ref_subalignment):
        assert s == r

def test_only_matches(
    simple_data : SequenceDataset,
    simple_model : LearnMSAModel,
) -> None:
    """Test writing only match columns to file"""
    # subalignment
    subset = np.array([0, 2, 5])
    # create alignment after building model
    sub_am = AlignmentModel(simple_data, simple_model, subset)
    subalignment_strings = sub_am.to_string(
        0, add_block_sep=False, only_matches=True
    )
    ref_subalignment = ["FELIK", "FELIK", "FELIK"]
    for s, r in zip(subalignment_strings, ref_subalignment):
        assert s == r

def test_mea(
    simple_data : SequenceDataset,
    simple_model : LearnMSAModel,
) -> None:
    """Test writing only match columns to file"""
    # subalignment
    subset = np.array([0, 2, 5])
    # create alignment after building model
    sub_am = AlignmentModel(simple_data, simple_model, subset)
    subalignment_strings = sub_am.to_string(
        0, add_block_sep=False, decoding_mode=AlignmentModel.DecodingMode.MEA
    )
    ref_subalignment = ["FE...LIK...", "FE...LIKhac", "FEahcLIK..."]
    for s, r in zip(subalignment_strings, ref_subalignment):
        assert s == r

def test_alignment_egf(tmp_path) -> None:
    """Test the high-level alignment function with real world data.

    The strongest end-to-end gate either backend has: training, model surgery,
    model selection, Viterbi decoding and insertion alignment all have to work
    together for the sum-of-pairs score to clear the threshold.
    """
    egf_fasta_path = os.path.join(DATA, "egf.fasta")
    egf_ref_path = os.path.join(DATA, "egf.ref")
    egf_out_path = str(tmp_path / "egf.out.fasta")

    with SequenceDataset(egf_fasta_path) as data:
        with AlignedDataset(egf_ref_path) as ref_msa:
            seq_ids = ref_msa.seq_ids

        config = Configuration()
        config.training.num_model = 1
        config.training.no_sequence_weights = True
        config.training.epochs = [5, 1, 5]
        config.training.max_iterations = 2
        config.training.length_init = [25]
        config.input_output.subset_ids = seq_ids
        config.training.crop = 999999
        config.training.auto_crop = False

        # Fit the alignment model
        am = align(data, config)
        am.select_best()

        # Evaluate the model
        eval_output = am.model.evaluate(data, models=[am.best_head])

    # Check some friendly thresholds to check if the alignment makes sense
    assert np.amin(eval_output["loglik"].mean()) > -70
    # Surgery should have added match states
    assert am.model.lengths[am.best_head] > 25

    am.to_file(egf_out_path, 0)
    with AlignedDataset(egf_out_path) as pred_msa:
        sp = pred_msa.SP_score(ref_msa)
        # based on experience, any half decent hyperparameter choice
        # should yield at least this score
        assert sp > 0.7


def test_aligned_insertions() -> None:
    """Test aligned insertion blocks."""
    sequences = np.array([[1, 2, 3, 4, 5],
                          [6, 7, 8, 9, 10],
                          [11, 12, 13, 14, 15]])
    lens = np.array([5, 4, 3])
    starts = np.array([0, 1, 2])
    custom_columns = np.array([[0, 1, 2, 3, 4, -1],
                               [0, 1, 4, 5, -1, -1],
                               [2, 3, 4, -1, -1, -1]])
    output_len= 27  # residues + gap
    gap = output_len- 1
    block = AlignmentModel.get_insertion_block(
        sequences, lens, 6, starts, output_len, custom_columns=custom_columns
    )
    expected_block = np.array([[1, 2, 3, 4, 5, gap],
                               [7, 8, gap, gap, 9, 10],
                               [gap, gap, 13, 14, 15, gap]])
    np.testing.assert_array_equal(
        block, expected_block + output_len
    )

def test_alignment_metadata(
    multi_hit_data : SequenceDataset,
    simple_model : LearnMSAModel,
) -> None:
    # create alignment after building model
    am = AlignmentModel(
        multi_hit_data, simple_model,
        hit_alignment_mode=HitAlignmentMode.LEFT_ALIGN
    )
    am.build_alignment([0], AlignmentModel.DecodingMode.VITERBI)
    np.testing.assert_equal(
        am.metadata[0].domain_hit,
        [
            [0, 1, 2, -1, 3],
            [3, -1, 4, 5, -1],
            [0, -1, 1, -1, 2], [6, 7, 8, 9, 10], [-1, 14, 15, 16, -1],
            [0, 1, 2, 3, 4], [-1, 8, 9, 10, -1],
            [0, -1, 1, -1, 2], [4, 5, 6, 7, 8]
        ]
    )
    np.testing.assert_equal(
        am.metadata[0].domain_loc,
        [
            [0, 4],
            [3, 6],
            [0, 3], [6, 11], [14, 17],
            [0, 5], [8, 11],
            [0, 3], [4, 9]
        ]
    )


def test_alignment_decoding_mode_left(
    multi_hit_data : SequenceDataset,
    simple_model : LearnMSAModel,
) -> None:
    # create alignment after building model
    am = AlignmentModel(
        multi_hit_data, simple_model,
        hit_alignment_mode=HitAlignmentMode.LEFT_ALIGN
    )
    subalignment_strings = am.to_string(0, add_block_sep=False)
    ref_subalignment = [
        "...FEL-K...-----...---...",
        "ahcF-LI-...-----...---...",
        "...F-L-KhacFELIKhaaELIaah",
        "...FELIKahc-ELI-...---...",
        "...F-L-Ka..FELIK...---..."
    ]
    for s, r in zip(subalignment_strings, ref_subalignment):
        assert s == r


def test_alignment_decoding_mode_right(
    multi_hit_data : SequenceDataset,
    simple_model : LearnMSAModel,
) -> None:
    # create alignment after building model
    sub_am = AlignmentModel(
        multi_hit_data, simple_model,
        hit_alignment_mode=HitAlignmentMode.RIGHT_ALIGN
    )
    subalignment_strings = sub_am.to_string(0, add_block_sep=False)
    ref_subalignment = [
        "...---...-----...FEL-K...",
        "ahc---...-----...F-LI-...",
        "...FLKhacFELIKhaa-ELI-aah",
        "...---...FELIKahc-ELI-...",
        "...---...F-L-Ka..FELIK..."
    ]
    for s, r in zip(subalignment_strings, ref_subalignment):
        assert s == r


def test_alignment_decoding_mode_greedy_scores(
    multi_hit_data : SequenceDataset,
    simple_model : LearnMSAModel,
) -> None:
    # create alignment after building model
    sub_am = AlignmentModel(
        multi_hit_data, simple_model,
        hit_alignment_mode=HitAlignmentMode.GREEDY_SCORES
    )
    subalignment_strings = sub_am.to_string(0, add_block_sep=False)
    ref_subalignment = [
        "...---...FEL-K...---...",
        "ahc---...F-LI-...---...",
        "...FLKhacFELIKhaaELIaah",
        "...---...FELIKahcELI...",
        "...FLKa..FELIK...---..."
    ]
    for s, r in zip(subalignment_strings, ref_subalignment):
        assert s == r


def test_alignment_decoding_mode_greedy_single(
    multi_hit_data : SequenceDataset,
    simple_model : LearnMSAModel,
) -> None:
    # create alignment after building model
    sub_am = AlignmentModel(
        multi_hit_data, simple_model,
        hit_alignment_mode=HitAlignmentMode.GREEDY_SINGLE
    )
    subalignment_strings = sub_am.to_string(0, add_block_sep=False)
    ref_subalignment = [
        "...---...FEL-K...---...",
        "ahc---...F-LI-...---...",
        "...FLKhacFELIKhaaELIaah",
        "...---...FELIKahcELI...",
        "...FLKa..FELIK...---..."
    ]
    for s, r in zip(subalignment_strings, ref_subalignment):
        assert s == r


def test_alignment_metadata_shift() -> None:
    """Test AlignmentMetaData.shift() with the flat data layout."""
    rng = np.random.default_rng(42)
    R, N, M = 3, 5, 4  # repeats per row, rows, match states
    total_R = R * N    # all rows have exactly R repeats

    # Build flat arrays (row-major: row 0 repeats, row 1 repeats, ...)
    orig_dh  = rng.integers(0, 10,  (total_R, M),     dtype=np.int16)
    orig_il  = rng.integers(0, 5,   (total_R, M - 1), dtype=np.int16)
    orig_is  = rng.integers(0, 100, (total_R, M - 1), dtype=np.int16)
    orig_dl  = rng.integers(0, 100, (total_R, 2),      dtype=np.int16)
    uns_len  = rng.integers(0, 5,   (R - 1) * N,      dtype=np.int16)
    uns_start= rng.integers(0, 100, (R - 1) * N,       dtype=np.int16)

    meta = AlignmentMetaData(
        num_rows=N,
        num_match=M,
        num_repeats_per_row=np.full(N, R, dtype=np.int32),
        domain_hit=orig_dh.copy(),
        domain_loc=orig_dl.copy(),
        insertion_lens=orig_il.copy(),
        insertion_start=orig_is.copy(),
        left_flank_len=rng.integers(0, 10, N),
        left_flank_start=rng.integers(0, 10, N),
        right_flank_len=rng.integers(0, 10, N),
        right_flank_start=rng.integers(0, 10, N),
        unannotated_segments_len=uns_len.copy(),
        unannotated_segments_start=uns_start.copy(),
    )

    # Before shift: flat data is unchanged, virtual offsets are 0
    assert meta.num_repeats == R
    assert meta._repeat_offset.tolist() == [0] * N

    # Each row gets a different shift amount
    shift_arr = np.array([0, 1, 0, 2, 1])
    assert len(shift_arr) == N

    meta.shift(shift_arr)

    # Virtual num_repeats = max(_repeat_offset + num_repeats_per_row)
    expected_num_repeats = int(np.amax(shift_arr + R))
    assert meta.num_repeats == expected_num_repeats

    # Flat data must be unchanged by shift
    np.testing.assert_array_equal(meta.domain_hit, orig_dh)
    np.testing.assert_array_equal(meta.insertion_lens, orig_il)

    # _repeat_offset must equal shift_arr
    np.testing.assert_array_equal(meta._repeat_offset, shift_arr)

    # For each row i with shift s, get_repeat_data(s+r, [i]) must return orig repeat r
    for row_i in range(N):
        s = shift_arr[row_i]
        for r in range(R):
            virt = s + r
            flat_orig = row_i * R + r   # flat index in orig arrays
            dh, il, is_, _, has_r = meta.get_repeat_data(virt, np.array([row_i]))
            assert has_r[0], f"row {row_i} repeat {virt} should exist"
            np.testing.assert_array_equal(dh[0], orig_dh[flat_orig])
            np.testing.assert_array_equal(il[0], orig_il[flat_orig])
            np.testing.assert_array_equal(is_[0], orig_is[flat_orig])

        # Positions before the shift should return padding
        for pre in range(s):
            dh, il, _, _, has_r = meta.get_repeat_data(pre, np.array([row_i]))
            assert not has_r[0], f"row {row_i} repeat {pre} should be empty (pre-shift)"
            np.testing.assert_array_equal(dh[0], np.full(M, -1, dtype=np.int16))
            np.testing.assert_array_equal(il[0], np.zeros(M - 1, dtype=np.int16))

        # Positions after the last repeat should return padding
        dh, il, _, _, has_r = meta.get_repeat_data(s + R, np.array([row_i]))
        assert not has_r[0], f"row {row_i} repeat {s+R} should be empty (post-shift)"
