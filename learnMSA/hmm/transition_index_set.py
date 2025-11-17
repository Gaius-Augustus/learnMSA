from dataclasses import dataclass

import numpy as np


@dataclass
class PHMMTransitionIndexSet:
    """ Indices for accessing groups of values in a PHMM transition matrix.

    Args:
        L: Number of match states.
    """
    def __init__(self, L: int, dtype=np.int32) -> None:
        self.L = L
        self.matches_plus = np.arange(L, dtype=dtype)
        self.matches = self.matches_plus[:-1]
        self.begin_to_match = np.stack(
            (np.zeros(L, dtype=dtype) + 3*L, self.matches_plus),
            axis=1,
        )
        self.begin_to_delete = np.array([[3*L, 2*L-1]], dtype=dtype)
        self.match_to_match = np.stack(
            (self.matches, self.matches+1), axis=1
        )
        self.match_to_insert = np.stack(
            (self.matches, self.matches+self.L), axis=1
        )
        self.match_to_delete = np.stack(
            (self.matches, self.matches+2*self.L), axis=1
        )
        self.match_to_end = np.stack(
            (self.matches_plus, np.zeros(self.L, dtype=dtype)+(3*self.L+1)),
            axis=1,
        )
        self.insert_to_insert = np.stack(
            (self.matches+self.L, self.matches+self.L), axis=1
        )
        self.insert_to_match = np.stack(
            (self.matches+self.L, self.matches+1), axis=1
        )
        self.delete_to_match = np.stack(
            (self.matches+2*self.L-1, self.matches+1), axis=1
        )
        self.delete_to_delete = np.stack(
            (self.matches+2*self.L-1, self.matches+2*self.L), axis=1
        )
        self.left_flank = np.array(
            [[3*L-1, 3*L-1], [3*L-1, 3*L]], dtype=dtype
        )
        self.right_flank = np.array(
            [[3*L+3, 3*L+3], [3*L+3, 3*L+4]], dtype=dtype
        )
        self.unannotated = np.array(
            [[3*L+2, 3*L+2], [3*L+2, 3*L]], dtype=dtype
        )
        self.end = np.array(
            [[3*L+1, 3*L+2], [3*L+1, 3*L+3], [3*L+1, 3*L+4]], dtype=dtype
        )

    def as_array(self) -> np.ndarray:
        """
        Returns an array of shape `(num_transitions, 2)` where each row is a
        (from_state, to_state) pair.
        """
        return np.vstack([
            self.begin_to_match,
            self.begin_to_delete,
            self.match_to_match,
            self.match_to_insert,
            self.match_to_delete,
            self.insert_to_insert,
            self.insert_to_match,
            self.delete_to_delete,
            self.delete_to_match,
            [[3*self.L-2, 3*self.L+1]], # D_L to E
            self.match_to_end,
            self.left_flank,
            self.right_flank,
            self.unannotated,
            self.end,
            [[3*self.L+4, 3*self.L+4]], # terminal to terminal
        ])


    def mask(self, dtype=np.float32) -> np.ndarray:
        """
        Returns a mask matrix of shape `(3L+5, 3L+5)` with ones for invalid
        transitions and zeros for valid transitions.
        """
        M = np.ones((3*self.L+5, 3*self.L+5), dtype=dtype)
        M[self.begin_to_match[:,0], self.begin_to_match[:,1]] = 0
        M[self.begin_to_delete[0,0], self.begin_to_delete[0,1]] = 0
        M[self.match_to_match[:,0], self.match_to_match[:,1]] = 0
        M[self.match_to_insert[:,0], self.match_to_insert[:,1]] = 0
        M[self.match_to_delete[:,0], self.match_to_delete[:,1]] = 0
        M[self.insert_to_insert[:,0], self.insert_to_insert[:,1]] = 0
        M[self.insert_to_match[:,0], self.insert_to_match[:,1]] = 0
        M[self.delete_to_delete[:,0], self.delete_to_delete[:,1]] = 0
        M[self.delete_to_match[:,0], self.delete_to_match[:,1]] = 0
        M[3*self.L-2, 3*self.L+1] = 0 # D_L to E
        M[self.match_to_end[:,0], self.match_to_end[:,1]] = 0
        M[self.left_flank[:,0], self.left_flank[:,1]] = 0
        M[self.right_flank[:,0], self.right_flank[:,1]] = 0
        M[self.unannotated[:,0], self.unannotated[:,1]] = 0
        M[self.end[:,0], self.end[:,1]] = 0
        M[-1, -1] = 0 # terminal state can loop to itself
        return M
