"""Tests for MSA to HMM conversion functionality."""

import numpy as np
import pytest

from learnMSA.msa_hmm.MSA2HMM import PHMMTransitionIndexSet, PHMMValueSet
from learnMSA.msa_hmm.SequenceDataset import AlignedDataset


def test_index_set() -> None:
    """Test PHMMTransitionIndexSet initialization and structure."""
    ind = PHMMTransitionIndexSet(3)
    assert ind.L == 3
    np.testing.assert_equal(ind.begin_to_match, [[9, 0], [9, 1], [9, 2]])
    np.testing.assert_equal(ind.match_to_match, [[0, 1], [1, 2]])
    np.testing.assert_equal(ind.match_to_insert, [[0, 3], [1, 4]])
    np.testing.assert_equal(ind.match_to_delete, [[0, 6], [1, 7]])
    np.testing.assert_equal(ind.insert_to_match, [[3, 1], [4, 2]])
    np.testing.assert_equal(ind.insert_to_insert, [[3, 3], [4, 4]])
    np.testing.assert_equal(ind.delete_to_match, [[5, 1], [6, 2]])
    np.testing.assert_equal(ind.delete_to_delete, [[5, 6], [6, 7]])
    np.testing.assert_equal(ind.match_to_end, [[0, 10], [1, 10], [2, 10]])
    np.testing.assert_equal(ind.left_flank, [[8, 8], [8, 9]])
    np.testing.assert_equal(ind.unannotated, [[11, 11], [11, 9]])
    np.testing.assert_equal(ind.end, [[10, 11], [10, 12], [10, 13]])
    np.testing.assert_equal(ind.mask(), [
        [1., 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
        [1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
        [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    ])


def test_msa_to_counts() -> None:
    """Test conversion of MSA to HMM counts."""
    sequences = [
        ("seq1", "----AA---AA--"),
        ("seq2", "A---AC-C---A-"),
        ("seq3", "-C-C--A--CAAA"),
        ("seq4", "-CACA--AA----"),
        ("seq5", "--ACAAA-AA--A"),
    ]
    with AlignedDataset(aligned_sequences=sequences) as data:
        counts_local = PHMMValueSet.from_msa(
            data, match_threshold=0.5, global_factor=0.0
        )
        counts_global = PHMMValueSet.from_msa(
            data, match_threshold=0.5, global_factor=1.0
        )
        counts_mix = PHMMValueSet.from_msa(
            data, match_threshold=0.5, global_factor=0.6
        )
        a_ind = data.alphabet.index("A")
        c_ind = data.alphabet.index("C")

    assert counts_local.matches() == 4  # number of match states
    assert counts_global.matches() == 4  # number of match states

    # Test emission counts
    np.testing.assert_equal(
        counts_local.match_emissions[:, a_ind], [0, 4, 2, 2]
    )
    np.testing.assert_equal(
        counts_local.match_emissions[:, c_ind], [3, 0, 1, 1]
    )
    np.testing.assert_equal(
        np.sum(counts_local.match_emissions, -1), [3, 4, 3, 3]
    )
    np.testing.assert_equal(counts_local.insert_emissions[a_ind], 14)
    np.testing.assert_equal(counts_local.insert_emissions[c_ind], 3)
    np.testing.assert_equal(np.sum(counts_local.insert_emissions), 17)
    np.testing.assert_equal(
        counts_local.match_emissions, counts_local.match_emissions
    )
    np.testing.assert_equal(
        counts_local.insert_emissions, counts_local.insert_emissions
    )

    # Test transition counts
    np.testing.assert_equal(
        counts_local.transitions,
        [
            # matches
            [0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0],
            # inserts
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # deletes
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # L, B, E, C, R, T
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 0, 0, 0, 0],
            [3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 5],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    np.testing.assert_equal(
        counts_global.transitions,
        [
            # matches
            [0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 3, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0],
            # inserts
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
            # deletes
            [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
            # L, B, E, C, R, T
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 0, 0, 0, 0],
            [3, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    np.testing.assert_equal(
        counts_mix.transitions,
        counts_global.transitions * 0.6 + counts_local.transitions * 0.4
    )


def test_add_pseudocounts() -> None:
    """Test adding pseudocounts to HMM parameters."""
    L = 4
    S = 3
    counts = PHMMValueSet(
        match_emissions=np.zeros((L, S)),
        insert_emissions=np.zeros((S,)),
        transitions=np.zeros((3 * L + 5, 3 * L + 5)),
        start=np.zeros((2,))
    )
    counts.add_pseudocounts(
        aa=[0., 1., 2.],
        match_transition=[11., 22., 33.],
        insert_transition=[4., 5.],
        delete_transition=[6., 7.],
        begin_to_match=[101., 102.],
        begin_to_delete=767.,
        match_to_end=13.,
        left_flank=[103., 104.],
        right_flank=[105., 106.],
        unannotated=[107., 108.],
        end=[300., 301., 302.],
        flank_start=[109., 110.],
    )
    np.testing.assert_equal(
        counts.match_emissions, np.tile([[0., 1., 2.]], (L, 1))
    )
    np.testing.assert_equal(
        counts.insert_emissions, np.array([0., 1., 2.])
    )
    np.testing.assert_equal(
        counts.transitions,
        np.array([
            # matches
            [0, 11., 0, 0, 22., 0, 0, 0, 33., 0, 0, 0, 0, 13, 0, 0, 0],
            [0, 0, 11., 0, 0, 22., 0, 0, 0, 33., 0, 0, 0, 13, 0, 0, 0],
            [0, 0, 0, 11., 0, 0, 22., 0, 0, 0, 33., 0, 0, 13, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0],
            # inserts
            [0, 5, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 5, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 5, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # deletes
            [0, 7, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 7, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1e-8, 0, 0, 0],
            # L, B, E, C, R, T
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 103, 104, 0, 0, 0, 0],
            [101, 102, 102, 102, 0, 0, 0, 767, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 300, 301, 302],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 108, 0, 107, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 105, 106],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1e-8],
        ])
    )
    np.testing.assert_equal(
        counts.start, np.array([109., 110.])
    )


def test_log_normalize() -> None:
    """Test normalization of HMM parameters to probabilities."""
    sequences = [
        ("seq1", "----AA---AA--"),
        ("seq2", "A---AC-C---A-"),
        ("seq3", "-C-C--A--CAAA"),
        ("seq4", "-CACA--AA----"),
        ("seq5", "--ACAAA-AA--A"),
    ]
    with AlignedDataset(aligned_sequences=sequences) as data:
        probs = PHMMValueSet.from_msa(data, global_factor=1.0).add_pseudocounts(
            unannotated=1.0, insert_transition=1.0,
        ).normalize()
        a_ind = data.alphabet.index("A")
        c_ind = data.alphabet.index("C")
    np.testing.assert_almost_equal(np.sum(probs.match_emissions, -1), 1.0)
    np.testing.assert_almost_equal(
        probs.match_emissions[:, a_ind], [0., 1., 2. / 3, 2. / 3]
    )
    np.testing.assert_almost_equal(
        probs.match_emissions[:, c_ind], [1., 0., 1. / 3, 1. / 3]
    )
    np.testing.assert_almost_equal(np.sum(probs.insert_emissions), 1.)
    np.testing.assert_almost_equal(probs.insert_emissions[a_ind], 14. / 17)
    np.testing.assert_almost_equal(probs.insert_emissions[c_ind], 3. / 17)
    np.testing.assert_almost_equal(
        probs.transitions,
        [
            # matches
            [0, 2. / 3, 0, 0, 0, 0, 0, 0, 1. / 3, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 3. / 4, 0, 0, 0, 0, 0, 0, 1. / 4, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1. / 3, 0, 0, 2. / 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            # inserts
            [0, 0.5, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0.5, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 3. / 8, 0, 0, 3. / 8, 0, 0, 0, 2. / 8, 0, 0, 0, 0, 0, 0],
            # deletes
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            # L, B, E, C, R, T
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1. / 5, 4. / 5, 0, 0, 0, 0],
            [3. / 5, 0, 0, 0, 0, 0, 0, 2. / 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4. / 5, 1. / 5],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0.5, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2. / 6, 4. / 6],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )


def test_threshold_too_high() -> None:
    """Test that match_threshold=1.0 raises an assertion error."""
    sequences = [
        ("seq1", "----AA---AA--"),
        ("seq2", "A---AC-C---A-"),
        ("seq3", "-C-C--A--CAAA"),
        ("seq4", "-CACA--AA----"),
        ("seq5", "--ACAAA-AA--A"),
    ]
    with AlignedDataset(aligned_sequences=sequences) as data:
        with pytest.raises(AssertionError):
            counts = PHMMValueSet.from_msa(data, match_threshold=1.0)


def test_threshold_very_low() -> None:
    """Test that match_threshold=0.0 treats all columns as match states."""
    sequences = [
        ("seq1", "----AA---AA--"),
        ("seq2", "A---AC-C---A-"),
        ("seq3", "-C-C--A--CAAA"),
        ("seq4", "-CACA--AA----"),
        ("seq5", "--ACAAA-AA--A"),
    ]
    with AlignedDataset(aligned_sequences=sequences) as data:
        counts = PHMMValueSet.from_msa(data, match_threshold=0.0)
        # all columns are match states
        assert counts.matches() == len(sequences[0][1])


def test_gapless_msa_to_counts() -> None:
    """Test conversion of MSA to HMM counts."""
    sequences = [
        ("seq1", "ACGTACGT"),
        ("seq2", "ACGTACGT"),
        ("seq3", "ACGTACGT"),
        ("seq4", "ACGTACGT"),
        ("seq5", "ACGTACGT"),
    ]
    with AlignedDataset(aligned_sequences=sequences) as data:
        counts = PHMMValueSet.from_msa(data)

    assert counts.matches() == 8

    # Try to normalize counts, should not raise any warnings/errors
    counts.add_pseudocounts(
        aa=1e-8,
        match_transition=1e-8,
        insert_transition=1e-8,
        delete_transition=1e-8,
        begin_to_match=1e-8,
        begin_to_delete=1e-8,
        match_to_end=1e-2,
        left_flank=1e-8,
        right_flank=1e-8,
        unannotated=1e-8,
        end=1e-8,
        flank_start=1e-8,
    ).normalize()

    assert not np.any(np.isnan(counts.transitions))
