import numpy as np


class PHMMTransitionIndexSet:
    """ Indices for accessing groups of values in a PHMM transition matrix.
    Uses the state order\\
    M1 ... ML I1 ... IL-1 D1 ... DL LF B E C RF T\\
    if not folded, and otherwise\\
    M1 ... ML I1 ... IL-1 LF C RF T.

    Attributes:
        L: Number of match states.
        folded: Whether the model is folded (does not contain silent states for
            deletions).
        num_states: Total number of states in the PHMM.
        num_transitions: Total number of transitions in the PHMM.
        LF: Index of the left flank state.
        RF: Index of the right flank state.
        C: Index of the unannotated (C) state.
        T: Index of the terminal state.
        B: Index of the begin state (unfolded only).
        E: Index of the end state (unfolded only).
        M: Indices of the match states (array of length L).
        I: Indices of the insert states (array of length L-1).
        D: Indices of the delete states (array of length L-1, unfolded only).
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
        matches_plus = self.M
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
            idx, sidx, np.stack((matches, self.I), axis=1), "match_to_insert"
        )

        # Insert to insert (independent parameters per position)
        idx, sidx = self._add_transitions(
            idx, sidx, np.stack((self.I, self.I), axis=1), "insert_to_insert"
        )

        # Insert to match (independent parameters per position)
        idx, sidx = self._add_transitions(
            idx, sidx, np.stack((self.I, matches+1), axis=1), "insert_to_match"
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
                np.stack((matches_plus, np.full(L, self.C, dtype=dtype)), axis=1),
                "match_to_unannotated"
            )

            # Match to right
            idx, sidx = self._add_transitions(
                idx, sidx,
                np.stack((matches_plus, np.full(L, self.RF, dtype=dtype)), axis=1),
                "match_to_right"
            )

            # Match to terminal
            idx, sidx = self._add_transitions(
                idx, sidx,
                np.stack((matches_plus, np.full(L, self.T, dtype=dtype)), axis=1),
                "match_to_terminal"
            )

            # Left flank: self-loop, to M1..ML, to unannotated, to right, to terminal
            lf = self.LF
            lf_trans = np.empty((L+4, 2), dtype=dtype)
            lf_trans[0] = [lf, lf]  # self-loop
            lf_trans[1:1+L] = np.stack(
                (np.full(L, lf, dtype=dtype), matches_plus), axis=1
            )
            lf_trans[1+L] = [lf, self.C]    # to unannotated
            lf_trans[2+L] = [lf, self.RF]    # to right
            lf_trans[3+L] = [lf, self.T]       # to terminal
            idx, sidx = self._add_transitions(idx, sidx, lf_trans, "left_flank")

            # Right flank: self-loop, to terminal
            rf = self.RF
            rf_trans = np.array([[rf, rf], [rf, self.T]], dtype=dtype)
            idx, sidx = self._add_transitions(idx, sidx, rf_trans, "right_flank")

            # Unannotated: self-loop, to M1..ML, to right, to terminal
            un = self.C
            un_trans = np.empty((L+3, 2), dtype=dtype)
            un_trans[0] = [un, un]  # self-loop
            un_trans[1:1+L] = np.stack(
                (np.full(L, un, dtype=dtype), matches_plus), axis=1
            )
            un_trans[1+L] = [un, self.RF]    # to right
            un_trans[2+L] = [un, self.T]       # to terminal
            idx, sidx = self._add_transitions(idx, sidx, un_trans, "unannotated")

            # Terminal
            idx, sidx = self._add_transitions(
                idx, sidx,
                np.array([[self.T, self.T]], dtype=dtype),
                "terminal"
            )

        else:
            # Unfolded case
            # Match to delete
            idx, sidx = self._add_transitions(
                idx, sidx,
                np.stack((matches, self.D + 1), axis=1),
                "match_to_delete"
            )

            # Delete to delete
            idx, sidx = self._add_transitions(
                idx, sidx,
                np.stack(
                    (self.D,
                     self.D + 1), axis=1
                ),
                "delete_to_delete"
            )

            # Delete to match
            idx, sidx = self._add_transitions(
                idx, sidx,
                np.stack((self.D, matches + 1), axis=1),
                "delete_to_match"
            )

            # Begin to match
            idx, sidx = self._add_transitions(
                idx, sidx,
                np.stack((np.full(L, self.B, dtype=dtype), matches_plus), axis=1),
                "begin_to_match"
            )

            # Match to end
            idx, sidx = self._add_transitions(
                idx, sidx,
                np.stack((matches_plus, np.full(L, self.E, dtype=dtype)), axis=1),
                "match_to_end"
            )

            # Begin to delete
            idx, sidx = self._add_transitions(
                idx, sidx,
                np.array([[self.B, self.first_delete_state]], dtype=dtype),
                "begin_to_delete"
            )

            # Delete to end
            idx, sidx = self._add_transitions(
                idx, sidx,
                np.array([[self.first_delete_state + self.L - 1, self.E]], dtype=dtype),
                "delete_to_end"
            )

            # Left flank: self-loop, to begin
            lf = self.LF
            lf_trans = np.array([[lf, lf], [lf, self.B]], dtype=dtype)
            idx, sidx = self._add_transitions(idx, sidx, lf_trans, "left_flank")

            # Right flank: self-loop, to terminal
            rf = self.RF
            rf_trans = np.array([[rf, rf], [rf, self.T]], dtype=dtype)
            idx, sidx = self._add_transitions(
                idx, sidx, rf_trans, "right_flank",
                shared_with="left_flank" if shared_flanks else None
            )

            # Unannotated: self-loop, to begin
            un = self.C
            un_trans = np.array([[un, un], [un, self.B]], dtype=dtype)
            idx, sidx = self._add_transitions(
                idx, sidx, un_trans, "unannotated",
            )

            # End: to unannotated, to right, to terminal
            es = self.E
            end_trans = np.array(
                [[es, self.C],
                 [es, self.RF],
                 [es, self.T]], dtype=dtype
            )
            idx, sidx = self._add_transitions(idx, sidx, end_trans, "end")

            # Terminal
            idx, sidx = self._add_transitions(
                idx, sidx,
                np.array([[self.T, self.T]], dtype=dtype),
                "terminal"
            )

    @property
    def folded(self) -> bool:
        """Whether the model is folded."""
        return self._folded

    @property
    def M(self) -> np.ndarray:
        """Indices of the match states (array of length L)."""
        return np.arange(self.L, dtype=self._dtype)

    @property
    def I(self) -> np.ndarray:
        """Indices of the insert states (array of length L-1)."""
        return self.M[:-1] + self.L

    @property
    def D(self) -> np.ndarray:
        """Indices of the delete states (array of length L-1, unfolded only)."""
        return self.M[:-1] + self.first_delete_state

    @property
    def LF(self) -> int:
        """Index of the left flank state (2*L-1 if folded, else 3*L-1)."""
        return 2*self.L - 1 if self._folded else 3*self.L - 1

    @property
    def RF(self) -> int:
        """Index of the right flank state (2*L+1 if folded, else 3*L+3)."""
        return 2*self.L + 1 if self._folded else 3*self.L + 3

    @property
    def C(self) -> int:
        """Index of the unannotated (C) state (2*L if folded, else 3*L+2)."""
        return 2*self.L if self._folded else 3*self.L + 2

    @property
    def T(self) -> int:
        """Index of the terminal state (last state, equivalent to num_states-1)."""
        return -1

    @property
    def B(self) -> int:
        """Index of the begin state (3*L, unfolded only)."""
        if self._folded:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute 'begin_state'. "
                "This attribute is only available in unfolded mode."
            )
        return 3*self.L

    @property
    def E(self) -> int:
        """Index of the end state (3*L+1, unfolded only)."""
        if self._folded:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute 'end_state'. "
                "This attribute is only available in unfolded mode."
            )
        return 3*self.L + 1

    @property
    def first_delete_state(self) -> int:
        """Index of the first delete state D0 (2*L-1, unfolded only)."""
        if self._folded:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute 'first_delete_state'. "
                "This attribute is only available in unfolded mode."
            )
        return 2*self.L - 1

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
            # Can start in M1, ..., ML, L, C, R or T
            start = np.arange(self.L+4)
            start[-4] = self.LF   # L
            start[-3] = self.C   # C
            start[-2] = self.RF   # R
            start[-1] = self.T      # T
            return start
        else:
            # Can start in L or B
            return np.array([self.LF, self.B])  # L, B

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
