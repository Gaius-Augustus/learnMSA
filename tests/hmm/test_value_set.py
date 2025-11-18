"""Tests for MSA to HMM conversion functionality."""

import numpy as np
import pytest

from learnMSA.hmm.value_set import PHMMValueSet
from learnMSA.msa_hmm.SequenceDataset import AlignedDataset


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


def test_from_config_scalar_params() -> None:
    """Test from_config with scalar parameters."""
    from learnMSA.config.hmm import HMMConfig

    config = HMMConfig(
        p_begin_match=0.5,
        p_match_match=0.7,
        p_match_insert=0.1,
        p_match_end=0.05,
        p_insert_insert=0.4,
        p_delete_delete=0.3,
        p_begin_delete=0.2,
        p_left_left=0.8,
        p_right_right=0.8,
        p_unannot_unannot=0.7,
        p_end_unannot=1e-4,
        p_end_right=0.6,
        p_start_left_flank=0.3,
    )

    L = 3
    h = 0
    value_set = PHMMValueSet.from_config(L, h, config)

    # Check dimensions
    assert value_set.match_emissions.shape == (L, len(config.alphabet))
    assert value_set.insert_emissions.shape == (len(config.alphabet),)
    assert value_set.transitions.shape == (3*L + 5, 3*L + 5)
    assert value_set.start.shape == (2,)

    # Check starting probabilities
    np.testing.assert_almost_equal(value_set.start[0], 0.3)
    np.testing.assert_almost_equal(value_set.start[1], 0.7)

    # Check left flank transitions
    np.testing.assert_almost_equal(value_set.transitions[3*L-1, 3*L-1], 0.8)  # LF->LF
    np.testing.assert_almost_equal(value_set.transitions[3*L-1, 3*L], 0.2)  # LF->B

    # Check begin to match 1
    np.testing.assert_almost_equal(value_set.transitions[3*L, 0], 0.5)  # B->M1

    # Check begin to other matches (uniform distribution)
    np.testing.assert_almost_equal(value_set.transitions[3*L, 1], 0.25)  # B->M2
    np.testing.assert_almost_equal(value_set.transitions[3*L, 2], 0.25)  # B->M3

    # Check begin to delete
    np.testing.assert_almost_equal(value_set.transitions[3*L, 2*L-1], 0.2)  # B->D1

    # Check match to match
    for i in range(L-1):
        np.testing.assert_almost_equal(value_set.transitions[i, i+1], 0.7)  # Mi->Mi+1

    # Check match to insert
    for i in range(L-1):
        np.testing.assert_almost_equal(value_set.transitions[i, L+i], 0.1)  # Mi->Ii

    # Check insert self-loop
    for i in range(L-1):
        np.testing.assert_almost_equal(value_set.transitions[L+i, L+i], 0.4)  # Ii->Ii

    # Check insert to match
    for i in range(L-1):
        np.testing.assert_almost_equal(value_set.transitions[L+i, i+1], 0.6)  # Ii->Mi+1

    # Check match to delete (1 - p_match_match - p_match_insert - p_match_end)
    for i in range(L-1):
        expected = 1 - 0.7 - 0.1 - 0.05
        np.testing.assert_almost_equal(value_set.transitions[i, 2*L+i], expected)  # Mi->Di+1

    # Check delete to delete
    for i in range(L-1):
        np.testing.assert_almost_equal(value_set.transitions[2*L-1+i, 2*L+i], 0.3)  # Di->Di+1

    # Check delete to match
    for i in range(L-1):
        np.testing.assert_almost_equal(value_set.transitions[2*L-1+i, i+1], 0.7)  # Di->Mi+1

    # Check match L to end
    np.testing.assert_almost_equal(value_set.transitions[L-1, 3*L+1], 1.0)  # ML->E

    # Check delete L to end
    np.testing.assert_almost_equal(value_set.transitions[3*L-2, 3*L+1], 1.0)  # DL->E

    # Check end to unannotated
    np.testing.assert_almost_equal(value_set.transitions[3*L+1, 3*L+2], 1e-4)  # E->C

    # Check unannotated self-loop
    np.testing.assert_almost_equal(value_set.transitions[3*L+2, 3*L+2], 0.7)  # C->C

    # Check unannotated to begin
    np.testing.assert_almost_equal(value_set.transitions[3*L+2, 3*L], 0.3)  # C->B

    # Check end to right flank
    np.testing.assert_almost_equal(value_set.transitions[3*L+1, 3*L+3], 0.6)  # E->R

    # Check end to terminal
    expected = 1 - 1e-4 - 0.6
    np.testing.assert_almost_equal(value_set.transitions[3*L+1, 3*L+4], expected)  # E->T

    # Check right flank self-loop
    np.testing.assert_almost_equal(value_set.transitions[3*L+3, 3*L+3], 0.8)  # R->R

    # Check right flank to terminal
    np.testing.assert_almost_equal(value_set.transitions[3*L+3, 3*L+4], 0.2)  # R->T

    # Check terminal self-loop
    np.testing.assert_almost_equal(value_set.transitions[3*L+4, 3*L+4], 1.0)  # T->T


def test_from_config_multi_head() -> None:
    """Test from_config with head-dependent parameters."""
    from learnMSA.config.hmm import HMMConfig

    config = HMMConfig(
        p_begin_match=[0.4, 0.5, 0.6],  # Different for each head
        p_match_match=[0.65, 0.7, 0.75],
        p_match_insert=[0.15, 0.1, 0.08],
        p_left_left=[0.75, 0.8, 0.85],
        p_start_left_flank=[0.2, 0.3, 0.4],
    )

    L = 2

    # Test head 0
    value_set_h0 = PHMMValueSet.from_config(L, 0, config)
    np.testing.assert_almost_equal(value_set_h0.start[0], 0.2)
    np.testing.assert_almost_equal(value_set_h0.transitions[3*L, 0], 0.4)  # B->M1
    np.testing.assert_almost_equal(value_set_h0.transitions[0, 1], 0.65)  # M1->M2
    np.testing.assert_almost_equal(value_set_h0.transitions[0, L], 0.15)  # M1->I1
    np.testing.assert_almost_equal(value_set_h0.transitions[3*L-1, 3*L-1], 0.75)  # LF->LF

    # Test head 1
    value_set_h1 = PHMMValueSet.from_config(L, 1, config)
    np.testing.assert_almost_equal(value_set_h1.start[0], 0.3)
    np.testing.assert_almost_equal(value_set_h1.transitions[3*L, 0], 0.5)  # B->M1
    np.testing.assert_almost_equal(value_set_h1.transitions[0, 1], 0.7)  # M1->M2
    np.testing.assert_almost_equal(value_set_h1.transitions[0, L], 0.1)  # M1->I1
    np.testing.assert_almost_equal(value_set_h1.transitions[3*L-1, 3*L-1], 0.8)  # LF->LF

    # Test head 2
    value_set_h2 = PHMMValueSet.from_config(L, 2, config)
    np.testing.assert_almost_equal(value_set_h2.start[0], 0.4)
    np.testing.assert_almost_equal(value_set_h2.transitions[3*L, 0], 0.6)  # B->M1
    np.testing.assert_almost_equal(value_set_h2.transitions[0, 1], 0.75)  # M1->M2
    np.testing.assert_almost_equal(value_set_h2.transitions[0, L], 0.08)  # M1->I1
    np.testing.assert_almost_equal(value_set_h2.transitions[3*L-1, 3*L-1], 0.85)  # LF->LF


def test_from_config_position_dependent() -> None:
    """Test from_config with position-dependent parameters."""
    from learnMSA.config.hmm import HMMConfig

    L = 3
    config = HMMConfig(
        p_begin_match=[[0.5, 0.3, 0.2]],  # Position-dependent for 1 head
        p_match_match=[[0.7, 0.75, 0.8]],  # Different for each position
        p_match_insert=[[0.1, 0.12]],  # L-1 positions
        p_insert_insert=[[0.4, 0.35]],
        p_delete_delete=[[0.3, 0.25]],
        p_match_end=[[0.05, 0.03]],
    )

    h = 0
    value_set = PHMMValueSet.from_config(L, h, config)

    # Check begin to match states (explicit probabilities)
    np.testing.assert_almost_equal(value_set.transitions[3*L, 0], 0.5)  # B->M1
    np.testing.assert_almost_equal(value_set.transitions[3*L, 1], 0.3)  # B->M2
    np.testing.assert_almost_equal(value_set.transitions[3*L, 2], 0.2)  # B->M3

    # Check begin to delete (1 - sum of begin_to_match)
    np.testing.assert_almost_equal(value_set.transitions[3*L, 2*L-1], 0.0)  # B->D1

    # Check match to match (position-dependent)
    np.testing.assert_almost_equal(value_set.transitions[0, 1], 0.7)  # M1->M2
    np.testing.assert_almost_equal(value_set.transitions[1, 2], 0.75)  # M2->M3

    # Check match to insert (position-dependent)
    np.testing.assert_almost_equal(value_set.transitions[0, L], 0.1)  # M1->I1
    np.testing.assert_almost_equal(value_set.transitions[1, L+1], 0.12)  # M2->I2

    # Check insert self-loop (position-dependent)
    np.testing.assert_almost_equal(value_set.transitions[L, L], 0.4)  # I1->I1
    np.testing.assert_almost_equal(value_set.transitions[L+1, L+1], 0.35)  # I2->I2

    # Check delete to delete (position-dependent)
    np.testing.assert_almost_equal(value_set.transitions[2*L-1, 2*L], 0.3)  # D1->D2
    np.testing.assert_almost_equal(value_set.transitions[2*L, 2*L+1], 0.25)  # D2->D3

    # Check match to delete with position-dependent params
    # M1->D2: 1 - p_match_match[0] - p_match_insert[0] - p_match_end[0]
    expected = 1 - 0.7 - 0.1 - 0.05
    np.testing.assert_almost_equal(value_set.transitions[0, 2*L], expected)

    # M2->D3: 1 - p_match_match[1] - p_match_insert[1] - p_match_end[1]
    expected = 1 - 0.75 - 0.12 - 0.03
    np.testing.assert_almost_equal(value_set.transitions[1, 2*L+1], expected)


def test_from_config_multi_head_position_dependent() -> None:
    """Test from_config with both multi-head and position-dependent parameters."""
    from learnMSA.config.hmm import HMMConfig

    L = 2
    config = HMMConfig(
        p_begin_match=[
            [0.6, 0.4],  # Head 0: P(M1|B), P(M2|B)
            [0.7, 0.3]   # Head 1: P(M1|B), P(M2|B)
        ],
        p_match_match=[
            [0.65],  # Head 0: P(M2|M1)
            [0.75]   # Head 1: P(M2|M1)
        ],
        p_match_insert=[
            [0.15],  # Head 0: P(I1|M1)
            [0.10]   # Head 1: P(I1|M1)
        ],
    )

    # Test head 0
    value_set_h0 = PHMMValueSet.from_config(L, 0, config)
    print("value_set_h0", value_set_h0.transitions)
    np.testing.assert_almost_equal(value_set_h0.transitions[3*L, 0], 0.6)  # B->M1
    np.testing.assert_almost_equal(value_set_h0.transitions[3*L, 1], 0.4)  # B->M2
    np.testing.assert_almost_equal(value_set_h0.transitions[3*L, 2*L-1], 0.0)  # B->D1 (sum=1)
    np.testing.assert_almost_equal(value_set_h0.transitions[0, 1], 0.65)  # M1->M2
    np.testing.assert_almost_equal(value_set_h0.transitions[0, L], 0.15)  # M1->I1

    # Test head 1
    value_set_h1 = PHMMValueSet.from_config(L, 1, config)
    print("value_set_h1", value_set_h1.transitions)
    np.testing.assert_almost_equal(value_set_h1.transitions[3*L, 0], 0.7)  # B->M1
    np.testing.assert_almost_equal(value_set_h1.transitions[3*L, 1], 0.3)  # B->M2
    np.testing.assert_almost_equal(value_set_h1.transitions[3*L, 2*L-1], 0.0)  # B->D1 (sum=1)
    np.testing.assert_almost_equal(value_set_h1.transitions[0, 1], 0.75)  # M1->M2
    np.testing.assert_almost_equal(value_set_h1.transitions[0, L], 0.10)  # M1->I1


def test_from_config_edge_cases() -> None:
    """Test from_config with edge case values."""
    from learnMSA.config.hmm import HMMConfig

    # Test with L=1 (single match state)
    config = HMMConfig(p_begin_match=1.0)
    value_set = PHMMValueSet.from_config(L=1, h=0, config=config)

    assert value_set.match_emissions.shape == (1, len(config.alphabet))
    assert value_set.insert_emissions.shape == (len(config.alphabet),)
    np.testing.assert_almost_equal(value_set.transitions[3, 0], 1.0)  # B->M1
    np.testing.assert_almost_equal(value_set.transitions[0, 4], 1.0)  # M1->E

    # Test with minimum probabilities (near zero)
    config_min = HMMConfig(
        p_begin_match=0.0,
        p_match_match=0.0,
        p_match_insert=0.0,
        p_match_end=0.0,
        p_end_unannot=0.0,
    )
    value_set_min = PHMMValueSet.from_config(L=2, h=0, config=config_min)
    np.testing.assert_almost_equal(value_set_min.transitions[6, 0], 0.0)  # B->M1
    np.testing.assert_almost_equal(value_set_min.transitions[0, 1], 0.0)  # M1->M2

    # Test with maximum probabilities
    config_max = HMMConfig(
        p_begin_match=1.0,
        p_match_match=1.0,
        p_left_left=1.0,
        p_right_right=1.0,
    )
    value_set_max = PHMMValueSet.from_config(L=2, h=0, config=config_max)
    np.testing.assert_almost_equal(value_set_max.transitions[6, 0], 1.0)  # B->M1
    np.testing.assert_almost_equal(value_set_max.transitions[0, 1], 1.0)  # M1->M2
    np.testing.assert_almost_equal(value_set_max.transitions[5, 5], 1.0)  # LF->LF
