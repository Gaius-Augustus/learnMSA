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
    def __init__(
        self, L: int,
        folded: bool = False,
        shared_flanks: bool = False,
        dtype=np.int32,
    ) -> None:
        """
        Args:
            L: Number of match states.
            folded: Whether the model is folded (does not contain silent states for
                deletions).
            shared_flanks: Whether to share transition parameters of flank
                states within each head.
            dtype: The data type for the indices.
        """
        assert not (shared_flanks and folded), \
            "shared_flanks is only supported in the unfolded case."
        self.L = L
        self._folded = folded
        self._shared_flanks = shared_flanks
        self._dtype = dtype

        # Helper arrays for construction
        matches_plus = np.arange(L, dtype=dtype)
        matches = matches_plus[:-1]

        # Create the buffer with the correct size
        self._buffer = np.empty((self.num_transitions, 2), dtype=dtype)
        self._shared_indices = np.empty(self.num_transitions, dtype=dtype)
        self._row_offsets = {}

        # Track current position in buffers
        idx = 0
        sidx = 0

        # Common transitions for both folded and unfolded
        # Match to match
        idx, sidx = self._add_transitions(
            idx, sidx, np.stack((matches, matches+1), axis=1), "match_to_match"
        )

        # Match to insert
        idx, sidx = self._add_transitions(
            idx, sidx, np.stack((matches, matches+L), axis=1), "match_to_insert"
        )

        # Insert to insert (independent parameters per position)
        idx, sidx = self._add_transitions(
            idx, sidx, np.stack((matches+L, matches+L), axis=1), "insert_to_insert"
        )

        # Insert to match (independent parameters per position)
        idx, sidx = self._add_transitions(
            idx, sidx, np.stack((matches+L, matches+1), axis=1), "insert_to_match"
        )

        if folded:
            # Match to match jump transitions
            if L > 2:
                start_idx = idx
                for i in range(L-2):
                    n_jumps = L - i - 2
                    self._buffer[idx:idx+n_jumps, 0] = i
                    self._buffer[idx:idx+n_jumps, 1] = matches_plus[i+2:]
                    self._shared_indices[idx:idx+n_jumps] = np.arange(
                        n_jumps, dtype=dtype
                    ) + sidx
                    idx += n_jumps
                    sidx += n_jumps
                self._row_offsets['match_to_match_jump'] = (start_idx, idx)

            # Match to unannotated
            idx, sidx = self._add_transitions(
                idx, sidx,
                np.stack((matches_plus, np.full(L, 2*L, dtype=dtype)), axis=1),
                "match_to_unannotated"
            )

            # Match to right
            idx, sidx = self._add_transitions(
                idx, sidx,
                np.stack((matches_plus, np.full(L, 2*L+1, dtype=dtype)), axis=1),
                "match_to_right"
            )

            # Match to terminal
            idx, sidx = self._add_transitions(
                idx, sidx,
                np.stack((matches_plus, np.full(L, -1, dtype=dtype)), axis=1),
                "match_to_terminal"
            )

            # Left flank: self-loop, to M1..ML, to unannotated, to right, to terminal
            lf_trans = np.empty((L+4, 2), dtype=dtype)
            lf_trans[0] = [2*L-1, 2*L-1]  # self-loop
            lf_trans[1:1+L] = np.stack(
                (np.full(L, 2*L-1, dtype=dtype), matches_plus), axis=1
            )
            lf_trans[1+L] = [2*L-1, 2*L]    # to unannotated
            lf_trans[2+L] = [2*L-1, 2*L+1]  # to right
            lf_trans[3+L] = [2*L-1, -1]     # to terminal
            idx, sidx = self._add_transitions(idx, sidx, lf_trans, "left_flank")

            # Right flank: self-loop, to terminal
            rf_trans = np.array([[2*L+1, 2*L+1], [2*L+1, -1]], dtype=dtype)
            idx, sidx = self._add_transitions(idx, sidx, rf_trans, "right_flank")

            # Unannotated: self-loop, to M1..ML, to right, to terminal
            un_trans = np.empty((L+3, 2), dtype=dtype)
            un_trans[0] = [2*L, 2*L]  # self-loop
            un_trans[1:1+L] = np.stack(
                (np.full(L, 2*L, dtype=dtype), matches_plus), axis=1
            )
            un_trans[1+L] = [2*L, 2*L+1]  # to right
            un_trans[2+L] = [2*L, -1]     # to terminal
            idx, sidx = self._add_transitions(idx, sidx, un_trans, "unannotated")

            # Terminal
            idx, sidx = self._add_transitions(
                idx, sidx, np.array([[-1, -1]], dtype=dtype), "terminal"
            )

        else:
            # Unfolded case
            # Match to delete
            idx, sidx = self._add_transitions(
                idx, sidx,
                np.stack((matches, matches+2*L), axis=1),
                "match_to_delete"
            )

            # Delete to delete
            idx, sidx = self._add_transitions(
                idx, sidx,
                np.stack((matches+2*L-1, matches+2*L), axis=1),
                "delete_to_delete"
            )

            # Delete to match
            idx, sidx = self._add_transitions(
                idx, sidx,
                np.stack((matches+2*L-1, matches+1), axis=1),
                "delete_to_match"
            )

            # Begin to match
            idx, sidx = self._add_transitions(
                idx, sidx,
                np.stack((np.full(L, 3*L, dtype=dtype), matches_plus), axis=1),
                "begin_to_match"
            )

            # Match to end
            idx, sidx = self._add_transitions(
                idx, sidx,
                np.stack((matches_plus, np.full(L, 3*L+1, dtype=dtype)), axis=1),
                "match_to_end"
            )

            # Begin to delete
            idx, sidx = self._add_transitions(
                idx, sidx, np.array([[3*L, 2*L-1]], dtype=dtype), "begin_to_delete"
            )

            # Delete to end
            idx, sidx = self._add_transitions(
                idx, sidx, np.array([[3*L-2, 3*L+1]], dtype=dtype), "delete_to_end"
            )

            # Left flank: self-loop, to begin
            lf_trans = np.array([[3*L-1, 3*L-1], [3*L-1, 3*L]], dtype=dtype)
            idx, sidx = self._add_transitions(idx, sidx, lf_trans, "left_flank")

            # Right flank: self-loop, to terminal
            rf_trans = np.array([[3*L+3, 3*L+3], [3*L+3, -1]], dtype=dtype)
            idx, sidx = self._add_transitions(
                idx, sidx, rf_trans, "right_flank",
                shared_with="left_flank" if shared_flanks else None
            )

            # Unannotated: self-loop, to begin
            un_trans = np.array([[3*L+2, 3*L+2], [3*L+2, 3*L]], dtype=dtype)
            idx, sidx = self._add_transitions(
                idx, sidx, un_trans, "unannotated",
                shared_with="left_flank" if shared_flanks else None
            )

            # End: to unannotated, to right, to terminal
            end_trans = np.array(
                [[3*L+1, 3*L+2], [3*L+1, 3*L+3], [3*L+1, -1]], dtype=dtype
            )
            idx, sidx = self._add_transitions(idx, sidx, end_trans, "end")

            # Terminal
            idx, sidx = self._add_transitions(
                idx, sidx, np.array([[-1, -1]], dtype=dtype), "terminal"
            )

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
            start[-1] = -1  # T
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

    def as_array(self, shared_flanks: bool = True) -> np.ndarray:
        """
        Returns an array of shape `(num_transitions, 2)` where each row is a
        (from_state, to_state) pair. If transitions are shared, only the first
        occurrence of each unique transition is included in the array.

        Args:
            shared_flanks: Whether to share transition parameters of flank
                states within each head.
        """
        return self._buffer

    def mask(self, dtype=np.float32) -> np.ndarray:
        """
        Returns a mask matrix of shape `(num_states, num_states)` with ones for
        invalid transitions and zeros for valid transitions.
        """
        n = self.num_states
        M = np.ones((n, n), dtype=dtype)
        M[self._buffer[:, 0], self._buffer[:, 1]] = 0.
        return M

    def shared_indices(self) -> np.ndarray:
        """
        Returns an array of indices into the values of this head, indicating
        which transitions are shared. This has the same length as the array
        returned by `as_array()`.
        """
        return self._shared_indices

    def _add_transitions(
        self,
        idx: int,
        sidx: int,
        transitions: np.ndarray,
        name: str,
        shared_with: str | None = None,
    ) -> tuple[int, int]:
        """
        Utility used to initially fill the buffers of the value set.

        Args:
            idx: Current index in the buffer to start filling from.
            sidx: Current index in the shared indices buffer to start filling
                from.
            transitions: An array of shape (n_transitions, 2) containing the
                transitions to add to the buffer, where each row is a
                (from_state, to_state) pair.
            shared_with: If not None, the name of another property whose
                transitions are shared with the transitions being added.
                The referenced group must have the same number of transitions
                and must have already been added to the buffer.

        Returns:
            The next index in the buffer and the next index in
            the shared indices buffer after filling.
        """
        n_trans = transitions.shape[0]
        self._buffer[idx:idx+n_trans] = transitions
        self._row_offsets[name] = (idx, idx+n_trans)
        if shared_with is None:
            self._shared_indices[idx:idx+n_trans] = np.arange(
                n_trans, dtype=self._dtype
            ) + sidx
            sidx += n_trans
        else:
            s = shared_with
            self._shared_indices[idx:idx+n_trans] = self._shared_indices[
                self._row_offsets[s][0]:self._row_offsets[s][1]
            ]
        idx += n_trans
        return idx, sidx
