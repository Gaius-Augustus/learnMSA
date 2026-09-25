"""A float64 reference for the flank prior of a single-head pHMM, shared by
the backend tests.

The transition matrix has saturated flank loops: in float32 the loop
probabilities round to exactly 1 while the exits are small but non-zero.
Computing an exit as ``1 - loop`` then gives 0, whose log hits the log-zero
floor and turns the complement term into a huge reward.
"""

import numpy as np

from learnMSA.config.hmm import PHMMPriorConfig

#: Number of match states of the test model.
L = 3
LEFT, BEGIN, END, UNANNOTATED, RIGHT = 3 * L - 1, 3 * L, 3 * L + 1, \
    3 * L + 2, 3 * L + 3
NUM_STATES = 3 * L + 5  # the last state is the terminal state


def flank_matrix(exit_prob: float) -> np.ndarray:
    """A (1, Q, Q) float32 transition matrix whose flank and unannotated
    states leave with probability exit_prob."""
    A = np.zeros((1, NUM_STATES, NUM_STATES), dtype=np.float32)
    for state, target in ((LEFT, BEGIN), (UNANNOTATED, BEGIN), (RIGHT, -1)):
        A[0, state, state] = 1.0 - exit_prob
        A[0, state, target] = exit_prob
    A[0, END, RIGHT] = 0.5
    A[0, END, UNANNOTATED] = 0.25
    A[0, END, -1] = 0.25
    return A


def expected_flank_prior(A: np.ndarray, config: PHMMPriorConfig) -> float:
    """The flank prior in float64, with exits read from the exit entries."""
    A = A[0].astype(np.float64)
    a, a_c = config.alpha_flank, config.alpha_flank_compl
    score = 0.0
    for state, target in ((UNANNOTATED, BEGIN), (RIGHT, -1), (LEFT, BEGIN)):
        score += (a - 1) * np.log(A[state, state])
        score += (a_c - 1) * np.log(A[state, target])
    score += (a - 1) * np.log(A[END, RIGHT])
    score += (a_c - 1) * np.log(A[END, UNANNOTATED] + A[END, -1])
    return score


def start_distribution(other_prob: float) -> np.ndarray:
    """A (1, Q) float32 start distribution that starts in the left flank
    with probability 1 - other_prob and in the begin state otherwise."""
    start = np.zeros((1, NUM_STATES), dtype=np.float32)
    start[0, LEFT] = 1.0 - other_prob
    start[0, BEGIN] = other_prob
    return start


def expected_start_prior(start: np.ndarray, config: PHMMPriorConfig) -> float:
    """The start prior in float64, with the complement read from the other
    start probabilities."""
    start = start[0].astype(np.float64)
    a, a_c = config.alpha_flank, config.alpha_flank_compl
    other = np.delete(start, LEFT).sum()
    return (a - 1) * np.log(start[LEFT]) + (a_c - 1) * np.log(other)
