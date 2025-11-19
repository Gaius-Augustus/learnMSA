import numpy as np


class PHMMTransitionIndexSet:
    """ Indices for accessing groups of values in a PHMM transition matrix.
    Uses the state order\\
    M1 ... ML I1 ... IL-1 D1 ... DL L B E C R T\\
    if not folded, and otherwise\\
    M1 ... ML I1 ... IL-1 L C R T.

    Attributes:
        L: Number of match states.
        folded: Whether the model is folded (does not contain silent states for
            deletions).
        match_to_match: Transition indices from match state i to match state i+1.
        match_to_insert: Transition indices from match state i to insert state i.
        insert_to_insert: Transition indices for insert state self-loops.
        insert_to_match: Transition indices from insert state i to match state i+1.
        match_to_delete: Transition indices from match state i to delete state i (unfolded only).
        delete_to_delete: Transition indices from delete state i to delete state i+1 (unfolded only).
        delete_to_match: Transition indices from delete state i to match state i+1 (unfolded only).
        begin_to_match: Transition indices from begin state to all match states (unfolded only).
        begin_to_delete: Transition indices from begin state to first delete state (unfolded only).
        match_to_end: Transition indices from all match states to end state (unfolded only).
        delete_to_end: Transition indices from last delete state to end state (unfolded only).
        end: Transition indices from end state to flanking states (unfolded only).
        match_to_match_jump: Transition indices from match state i to match state j where j > i+1 (folded only).
        match_to_unannotated: Transition indices from all match states to unannotated state (folded only).
        match_to_right: Transition indices from all match states to right flank state (folded only).
        match_to_terminal: Transition indices from all match states to terminal state (folded only).
        left_flank: Transition indices from left flank state to other states.
        right_flank: Transition indices from right flank state (self-loop and to terminal).
        unannotated: Transition indices from unannotated state to other states.
        terminal: Transition indices for terminal state self-loop.
        num_states: Total number of states in the PHMM.
        num_transitions: Total number of transitions in the PHMM.
    """
    def __init__(self, L: int, folded: bool = False, dtype=np.int32) -> None:
        """
        Args:
            L: Number of match states.
            folded: Whether the model is folded (does not contain silent states for
                deletions).
            dtype: The data type for the indices.
        """
        self.L = L
        self._folded = folded
        self._dtype = dtype

        # Helper arrays for construction
        matches_plus = np.arange(L, dtype=dtype)
        matches = matches_plus[:-1]

        # Create the buffer with the correct size
        self._buffer = np.empty((self.num_transitions, 2), dtype=dtype)
        self._row_offsets = {}

        # Track current position in buffer
        idx = 0

        # Common transitions for both folded and unfolded
        # Match to match
        n_trans = L - 1
        self._buffer[idx:idx+n_trans] = np.stack((matches, matches+1), axis=1)
        self._row_offsets['match_to_match'] = (idx, idx+n_trans)
        idx += n_trans

        # Match to insert
        n_trans = L - 1
        self._buffer[idx:idx+n_trans] = np.stack((matches, matches+L), axis=1)
        self._row_offsets['match_to_insert'] = (idx, idx+n_trans)
        idx += n_trans

        # Insert to insert
        n_trans = L - 1
        self._buffer[idx:idx+n_trans] = np.stack((matches+L, matches+L), axis=1)
        self._row_offsets['insert_to_insert'] = (idx, idx+n_trans)
        idx += n_trans

        # Insert to match
        n_trans = L - 1
        self._buffer[idx:idx+n_trans] = np.stack((matches+L, matches+1), axis=1)
        self._row_offsets['insert_to_match'] = (idx, idx+n_trans)
        idx += n_trans

        if folded:
            # Match to match jump transitions
            if L > 2:
                start_idx = idx
                for i in range(L-2):
                    n_jumps = L - i - 2
                    self._buffer[idx:idx+n_jumps, 0] = i
                    self._buffer[idx:idx+n_jumps, 1] = matches_plus[i+2:]
                    idx += n_jumps
                self._row_offsets['match_to_match_jump'] = (start_idx, idx)

            # Match to unannotated
            n_trans = L
            self._buffer[idx:idx+n_trans] = np.stack(
                (matches_plus, np.zeros(L, dtype=dtype)+(2*L)), axis=1
            )
            self._row_offsets['match_to_unannotated'] = (idx, idx+n_trans)
            idx += n_trans

            # Match to right
            n_trans = L
            self._buffer[idx:idx+n_trans] = np.stack(
                (matches_plus, np.zeros(L, dtype=dtype)+(2*L+1)), axis=1
            )
            self._row_offsets['match_to_right'] = (idx, idx+n_trans)
            idx += n_trans

            # Match to terminal
            n_trans = L
            self._buffer[idx:idx+n_trans] = np.stack(
                (matches_plus, np.zeros(L, dtype=dtype)+(2*L+2)), axis=1
            )
            self._row_offsets['match_to_terminal'] = (idx, idx+n_trans)
            idx += n_trans

            # Left flank
            start_idx = idx
            self._buffer[idx] = [2*L-1, 2*L-1]  # self-loop
            idx += 1
            self._buffer[idx:idx+L] = np.stack(
                (np.zeros(L, dtype=dtype) + (2*L - 1), matches_plus), axis=1
            )
            idx += L
            self._buffer[idx] = [2*L - 1, 2*L]  # to unannotated
            idx += 1
            self._buffer[idx] = [2*L - 1, 2*L + 1]  # to right
            idx += 1
            self._buffer[idx] = [2*L - 1, 2*L + 2]  # to terminal
            idx += 1
            self._row_offsets['left_flank'] = (start_idx, idx)

            # Right flank
            start_idx = idx
            self._buffer[idx] = [2*L+1, 2*L+1]  # self-loop
            idx += 1
            self._buffer[idx] = [2*L+1, 2*L+2]  # to terminal
            idx += 1
            self._row_offsets['right_flank'] = (start_idx, idx)

            # Unannotated
            start_idx = idx
            self._buffer[idx] = [2*L, 2*L]  # self-loop
            idx += 1
            self._buffer[idx:idx+L] = np.stack(
                (np.zeros(L, dtype=dtype) + (2*L), matches_plus), axis=1
            )
            idx += L
            self._buffer[idx] = [2*L, 2*L + 1]  # to right
            idx += 1
            self._buffer[idx] = [2*L, 2*L + 2]  # to terminal
            idx += 1
            self._row_offsets['unannotated'] = (start_idx, idx)

            # Terminal
            self._buffer[idx] = [2*L+2, 2*L+2]
            self._row_offsets['terminal'] = (idx, idx+1)
            idx += 1

        else:
            # Unfolded case
            # Match to delete
            n_trans = L - 1
            self._buffer[idx:idx+n_trans] = np.stack(
                (matches, matches+2*L), axis=1
            )
            self._row_offsets['match_to_delete'] = (idx, idx+n_trans)
            idx += n_trans

            # Delete to delete
            n_trans = L - 1
            self._buffer[idx:idx+n_trans] = np.stack(
                (matches+2*L-1, matches+2*L), axis=1
            )
            self._row_offsets['delete_to_delete'] = (idx, idx+n_trans)
            idx += n_trans

            # Delete to match
            n_trans = L - 1
            self._buffer[idx:idx+n_trans] = np.stack(
                (matches+2*L-1, matches+1), axis=1
            )
            self._row_offsets['delete_to_match'] = (idx, idx+n_trans)
            idx += n_trans

            # Begin to match
            n_trans = L
            self._buffer[idx:idx+n_trans] = np.stack(
                (np.zeros(L, dtype=dtype) + 3*L, matches_plus), axis=1
            )
            self._row_offsets['begin_to_match'] = (idx, idx+n_trans)
            idx += n_trans

            # Match to end
            n_trans = L
            self._buffer[idx:idx+n_trans] = np.stack(
                (matches_plus, np.zeros(L, dtype=dtype)+(3*L+1)), axis=1
            )
            self._row_offsets['match_to_end'] = (idx, idx+n_trans)
            idx += n_trans

            # Begin to delete
            self._buffer[idx] = [3*L, 2*L-1]
            self._row_offsets['begin_to_delete'] = (idx, idx+1)
            idx += 1

            # Delete to end
            self._buffer[idx] = [3*L-2, 3*L+1]
            self._row_offsets['delete_to_end'] = (idx, idx+1)
            idx += 1

            # Left flank
            start_idx = idx
            self._buffer[idx] = [3*L-1, 3*L-1]
            idx += 1
            self._buffer[idx] = [3*L-1, 3*L]
            idx += 1
            self._row_offsets['left_flank'] = (start_idx, idx)

            # Right flank
            start_idx = idx
            self._buffer[idx] = [3*L+3, 3*L+3]
            idx += 1
            self._buffer[idx] = [3*L+3, 3*L+4]
            idx += 1
            self._row_offsets['right_flank'] = (start_idx, idx)

            # Unannotated
            start_idx = idx
            self._buffer[idx] = [3*L+2, 3*L+2]
            idx += 1
            self._buffer[idx] = [3*L+2, 3*L]
            idx += 1
            self._row_offsets['unannotated'] = (start_idx, idx)

            # End
            start_idx = idx
            self._buffer[idx] = [3*L+1, 3*L+2]
            idx += 1
            self._buffer[idx] = [3*L+1, 3*L+3]
            idx += 1
            self._buffer[idx] = [3*L+1, 3*L+4]
            idx += 1
            self._row_offsets['end'] = (start_idx, idx)

            # Terminal
            self._buffer[idx] = [3*L+4, 3*L+4]
            self._row_offsets['terminal'] = (idx, idx+1)
            idx += 1

    @property
    def folded(self) -> bool:
        """Whether the model is folded."""
        return self._folded

    def _get_slice(self, name: str) -> np.ndarray:
        """Get a slice of the buffer for the given property name."""
        if name not in self._row_offsets:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'. "
                "This attribute is only available in "
                f"{'unfolded' if self._folded else 'folded'} mode."
            )
        start_row, end_row = self._row_offsets[name]
        return self._buffer[start_row:end_row]

    # Properties for all transition types
    @property
    def match_to_match(self) -> np.ndarray:
        """Transition indices from match state i to match state i+1."""
        return self._get_slice('match_to_match')

    @property
    def match_to_insert(self) -> np.ndarray:
        """Transition indices from match state i to insert state i."""
        return self._get_slice('match_to_insert')

    @property
    def insert_to_insert(self) -> np.ndarray:
        """Transition indices for insert state self-loops."""
        return self._get_slice('insert_to_insert')

    @property
    def insert_to_match(self) -> np.ndarray:
        """Transition indices from insert state i to match state i+1."""
        return self._get_slice('insert_to_match')

    @property
    def match_to_delete(self) -> np.ndarray:
        """Transition indices from match state i to delete state i (unfolded only)."""
        return self._get_slice('match_to_delete')

    @property
    def delete_to_delete(self) -> np.ndarray:
        """Transition indices from delete state i to delete state i+1 (unfolded only)."""
        return self._get_slice('delete_to_delete')

    @property
    def delete_to_match(self) -> np.ndarray:
        """Transition indices from delete state i to match state i+1 (unfolded only)."""
        return self._get_slice('delete_to_match')

    @property
    def begin_to_match(self) -> np.ndarray:
        """Transition indices from begin state to all match states (unfolded only)."""
        return self._get_slice('begin_to_match')

    @property
    def begin_to_delete(self) -> np.ndarray:
        """Transition indices from begin state to first delete state (unfolded only)."""
        return self._get_slice('begin_to_delete')

    @property
    def match_to_end(self) -> np.ndarray:
        """Transition indices from all match states to end state (unfolded only)."""
        return self._get_slice('match_to_end')

    @property
    def delete_to_end(self) -> np.ndarray:
        """Transition indices from last delete state to end state (unfolded only)."""
        return self._get_slice('delete_to_end')

    @property
    def end(self) -> np.ndarray:
        """Transition indices from end state to flanking states (unfolded only)."""
        return self._get_slice('end')

    @property
    def match_to_match_jump(self) -> np.ndarray:
        """Transition indices from match state i to match state j where j > i+1 (folded only)."""
        return self._get_slice('match_to_match_jump')

    @property
    def match_to_unannotated(self) -> np.ndarray:
        """Transition indices from all match states to unannotated state (folded only)."""
        return self._get_slice('match_to_unannotated')

    @property
    def match_to_right(self) -> np.ndarray:
        """Transition indices from all match states to right flank state (folded only)."""
        return self._get_slice('match_to_right')

    @property
    def match_to_terminal(self) -> np.ndarray:
        """Transition indices from all match states to terminal state (folded only)."""
        return self._get_slice('match_to_terminal')

    @property
    def left_flank(self) -> np.ndarray:
        """Transition indices from left flank state to other states."""
        return self._get_slice('left_flank')

    @property
    def right_flank(self) -> np.ndarray:
        """Transition indices from right flank state (self-loop and to terminal)."""
        return self._get_slice('right_flank')

    @property
    def unannotated(self) -> np.ndarray:
        """Transition indices from unannotated state to other states."""
        return self._get_slice('unannotated')

    @property
    def terminal(self) -> np.ndarray:
        """Transition indices for terminal state self-loop."""
        return self._get_slice('terminal')

    @property
    def start(self) -> np.ndarray:
        """Allowed starting states."""
        if self.folded:
            # Can start in M1, ..., ML, L, R, C or T
            start = np.arange(self.L+4)
            start[-4] = 2*self.L - 1  # L
            start[-3] = 2*self.L      # C
            start[-2] = 2*self.L + 1  # R
            start[-1] = 2*self.L + 2  # T
            return start
        else:
            # Can start in L or B
            return np.array([3*self.L-1, 3*self.L])  # L, B

    @property
    def num_states(self) -> int:
        """
        Returns the total number of states in the PHMM.
        """
        if self.folded:
            return self.num_states_folded(self.L)
        else:
            return self.num_states_unfolded(self.L)

    @staticmethod
    def num_states_folded(L: int) -> int:
        """
        Returns the total number of states in a folded PHMM with L match states.
        """
        return 2*L + 3

    @staticmethod
    def num_states_unfolded(L: int) -> int:
        """
        Returns the total number of states in an unfolded PHMM with L match states.
        """
        return 3*L + 5

    @property
    def num_transitions(self) -> int:
        """
        Returns the total number of transitions in the PHMM.
        """
        if self.folded:
            # Common: 4*(L-1) for match_to_match, match_to_insert,
            # insert_to_insert, insert_to_match
            # match_to_match_jump: (L-2)*(L-1)/2
            # match_to_unannotated: L
            # match_to_right: L
            # match_to_terminal: L
            # left_flank: 1 + L + 1 + 1 + 1 = L + 4
            # right_flank: 2
            # unannotated: 1 + L + 1 + 1 = L + 3
            # terminal: 1
            return 4*(self.L-1) + (self.L-2)*(self.L-1)//2 + 3*self.L\
                 + (self.L+4) + 2 + (self.L+3) + 1
        else:
            # Common: 4*(L-1)
            # match_to_delete, delete_to_delete, delete_to_match: 3*(L-1)
            # begin_to_match: L
            # match_to_end: L
            # begin_to_delete: 1
            # delete_to_end: 1
            # left_flank: 2
            # right_flank: 2
            # unannotated: 2
            # end: 3
            # terminal: 1
            return 4*(self.L-1) + 3*(self.L-1) + self.L + self.L\
                + 1 + 1 + 2 + 2 + 2 + 3 + 1

    def as_array(self) -> np.ndarray:
        """
        Returns an array of shape `(num_transitions, 2)` where each row is a
        (from_state, to_state) pair.
        """
        return self._buffer

    def mask(self, dtype=np.float32) -> np.ndarray:
        """
        Returns a mask matrix of shape `(num_states, num_states)` with ones for invalid
        transitions and zeros for valid transitions.
        """
        n = self.num_states
        M = np.ones((n, n), dtype=dtype)
        M[self._buffer[:, 0], self._buffer[:, 1]] = 0.
        return M
