import numpy as np


def make_phmm_transitions(length: int) -> np.ndarray:
    """
    Create a set of transitions for a profile HMM.

    Args:
        length (int): The number of match states / the length of the profile.

    Returns:
        np.ndarray: An array of shape (num_transitions, 2) where each row is a
        (from_state, to_state) pair.
    """
    transitions = []
    for i in range(length - 1):
        # match to match
        transitions.append((i, i + 1))
        # match to insert
        transitions.append((i, length + i))
        # self-loop in insert
        transitions.append((length + i, length + i))
        # insert to match
        transitions.append((length + i, i + 1))
    transitions.append((length - 1, length - 1))  # last match state self-loop
    return np.array(transitions)
