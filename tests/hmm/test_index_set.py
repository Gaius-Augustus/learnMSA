"""Tests for MSA to HMM conversion functionality."""

import numpy as np

from learnMSA.hmm.transition_index_set import PHMMTransitionIndexSet


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
    np.testing.assert_equal(ind.num_states, 14)
    np.testing.assert_equal(ind.num_transitions, 32)


def test_index_set_folded() -> None:
    """
    Test PHMMTransitionIndexSet initialization and structure for folded case.
    """
    ind = PHMMTransitionIndexSet(4, folded=True)
    assert ind.L == 4
    assert ind.folded is True

    # Basic transitions
    np.testing.assert_equal(ind.match_to_match, [[0, 1], [1, 2], [2, 3]])
    np.testing.assert_equal(ind.match_to_insert, [[0, 4], [1, 5], [2, 6]])
    np.testing.assert_equal(ind.insert_to_insert, [[4, 4], [5, 5], [6, 6]])
    np.testing.assert_equal(ind.insert_to_match, [[4, 1], [5, 2], [6, 3]])

    # Jump transitions (match to match, skipping states)
    # M0->M2, M0->M3, M1->M3
    np.testing.assert_equal(ind.match_to_match_jump, [[0, 2], [0, 3], [1, 3]])

    # Transitions to special states
    np.testing.assert_equal(
        ind.match_to_unannotated, [[0, 8], [1, 8], [2, 8], [3, 8]]
    )
    np.testing.assert_equal(
        ind.match_to_right, [[0, 9], [1, 9], [2, 9], [3, 9]]
    )
    np.testing.assert_equal(
        ind.match_to_terminal, [[0, 10], [1, 10], [2, 10], [3, 10]]
    )

    # Left flank: self-loop, to all matches, to unannotated, to right, to terminal
    np.testing.assert_equal(ind.left_flank, [
        [7, 7],  # L to L self-loop
        [7, 0],  # L to M1
        [7, 1],  # L to M2
        [7, 2],  # L to M3
        [7, 3],  # L to M4
        [7, 8],  # L to C (unannotated)
        [7, 9],  # L to R (right flank)
        [7, 10],  # L to T (terminal)
    ])

    # Right flank: self-loop and to terminal
    np.testing.assert_equal(ind.right_flank, [[9, 9], [9, 10]])

    # Unannotated: self-loop, to all matches, to right, to terminal
    np.testing.assert_equal(ind.unannotated, [
        [8, 8],  # C to C self-loop
        [8, 0],  # C to M1
        [8, 1],  # C to M2
        [8, 2],  # C to M3
        [8, 3],  # C to M4
        [8, 9],  # C to R (right flank)
        [8, 10],  # C to T (terminal)
    ])

    # Terminal self-loop
    np.testing.assert_equal(ind.terminal, [[10, 10]])
    np.testing.assert_equal(ind.num_states, 11)
    np.testing.assert_equal(ind.num_transitions, 45)


def test_num_transitions() -> None:
    """Test PHMMTransitionIndexSet num_transitions method."""
    ind = PHMMTransitionIndexSet(5)
    assert ind.num_transitions == 50
    ind_folded = PHMMTransitionIndexSet(5, folded=True)
    assert ind_folded.num_transitions == 57
