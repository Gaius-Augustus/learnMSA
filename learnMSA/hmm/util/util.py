import numpy as np


def make_phmm_explicit_transitions(length: int) -> np.ndarray:
    """
    Create a set of transitions for a profile HMM using the state order
    M1 ... ML I1 ... IL-1 D1 ... DL L B E C R T.

    Args:
        length (int): The number of match states / the length of the profile.

    Returns:
        np.ndarray: An array of shape (num_transitions, 2) where each row is a
        (from_state, to_state) pair.
    """
    num_transitions = 10000
    transitions = np.array((num_transitions, 2), dtype=int)
    # Transitions from L
    transitions[0] = (length * 2 - 2, length * 2 - 1)  # L to L
    transitions[1] = (length * 2 - 2, length * 2) # L to B
    # Transitions from B to all M
    transitions[2:2 + length] = [(length * 2, i) for i in range(length)]
    # Main model transitions
    idx = 2 + length
    for i in range(length - 1):
        # match to match
        transitions[idx] = (i, i + 1)
        idx += 1
        # match to insert
        transitions[idx] = (i, length + i)
        idx += 1
        # self-loop in insert
        transitions[idx] = (length + i, length + i)
        idx += 1
        # insert to match
        transitions[idx] = (length + i, i + 1)
        idx += 1
    transitions[idx] = (length - 1, length - 1)  # last match state self-loop
    return transitions
