from dataclasses import dataclass

import numpy as np

from learnMSA.util.aligned_dataset import AlignedDataset
from learnMSA.hmm.util.transition_index_set import PHMMTransitionIndexSet
from learnMSA.hmm.util.util import state_index_to_name
from learnMSA.config.hmm import PHMMConfig
from learnMSA.config.util import get_emission_dist


@dataclass
class PHMMValueSet:
    """ Parameter collection for a pHMM of length `L` with  `q=3L+5` states
    over an emission alphabet of size `s`.
    Uses the state order M1 ... ML I1 ... IL-1 D1 ... DL L B E C R T.

    Attributes:
        match_emissions (np.ndarray). Emission counts of shape `(L, s)`.
        insert_emissions (np.ndarray). Emission counts of shape `(s)`.
        transitions : np.ndarray of shape `(q, q)`.
        start : np.ndarray of shape `(2,)` counts of either starting in `B` or
            `L`.
    """
    L: int
    match_emissions : np.ndarray
    insert_emissions : np.ndarray
    transitions : np.ndarray
    start : np.ndarray


    @classmethod
    def from_config(cls, L: int, h: int, config: PHMMConfig) -> "PHMMValueSet":
        """Creates a PHMMValueSet from a PHMMConfig object.

        Args:
            length: The number of match states (L).
            h: The head index.
            config: An PHMMConfig object containing the transition and
                emission probabilities.

        Returns:
            A PHMMValueSet object with parameters initialized from the config.
        """
        from learnMSA.config.util import get_value
        from collections.abc import Sequence

        q = 3*L + 5
        s = len(config.alphabet)

        # Initialize empty arrays
        transitions = np.zeros((q, q), dtype=np.float32)
        ind = PHMMTransitionIndexSet(L)

        # Left flank: self-loop and to begin
        transitions[ind.left_flank[0, 0], ind.left_flank[0, 1]] = \
            get_value(config.p_left_left, h)
        transitions[ind.left_flank[1, 0], ind.left_flank[1, 1]] = \
            1 - get_value(config.p_left_left, h)

        # Begin to match 1
        bm1 = get_value(config.p_begin_match, h, 0)
        transitions[ind.begin_to_match[0, 0], ind.begin_to_match[0, 1]] = bm1

        # Check if transition probs to other match states are provided
        if isinstance(config.p_begin_match, (Sequence, np.ndarray)) and \
                isinstance(config.p_begin_match[h], (Sequence, np.ndarray)):
            p_begin_match_head = config.p_begin_match[h]
            assert isinstance(p_begin_match_head, (Sequence, np.ndarray))  # Type guard
            p_begin_match_inner = p_begin_match_head[1:]
            p_sum_prob_begin_match = sum(p_begin_match_head)
            assert p_sum_prob_begin_match <= 1, (
                f"Sum of p_begin_match is {p_sum_prob_begin_match}, which is > 1"
            )
            p_begin_delete = 1 - p_sum_prob_begin_match
        else:
            p = get_value(config.p_begin_match, h, 0)
            p_begin_delete = get_value(config.p_begin_delete, h)
            assert p + p_begin_delete <= 1, (
                f"Sum of p_begin_match and p_begin_delete is "
                f"{p + p_begin_delete}, which is > 1"
            )
            if L > 1:
                p_begin_match_inner = (1 - p - p_begin_delete) / (L - 1)
                p_begin_match_inner = max(1e-6, p_begin_match_inner)
            else:
                p_begin_match_inner = 0.0

        # Begin to delete 1
        transitions[ind.begin_to_delete[0, 0], ind.begin_to_delete[0, 1]] = \
            p_begin_delete

        # Handle p_match_end: compute automatic values if None
        if config.p_match_end is None:
            p_match_end_values = np.zeros(L - 1, dtype=np.float32)
            if L > 1:
                p_match_end_values[0] = max((1 - bm1), 1e-10) / (L - 1)
            # For i > 0: exit[i] = entry[i]
            if isinstance(p_begin_match_inner, (Sequence, np.ndarray)):
                # Use explicit entry probabilities
                for i in range(1, L - 1):
                    p_match_end_values[i] = p_begin_match_inner[i - 1]
            else:
                # Use uniform entry probability
                for i in range(1, L - 1):
                    p_match_end_values[i] = p_begin_match_inner

        for i in range(L - 1):
            # Begin to match i+1
            p_val = (p_begin_match_inner[i]
                    if isinstance(p_begin_match_inner, (Sequence, np.ndarray))
                    else p_begin_match_inner)
            transitions[ind.begin_to_match[i + 1, 0], ind.begin_to_match[i + 1, 1]] = \
                p_val

            # Match to match
            transitions[ind.match_to_match[i, 0], ind.match_to_match[i, 1]] = \
                get_value(config.p_match_match, h, i)

            # Match to insert
            transitions[ind.match_to_insert[i, 0], ind.match_to_insert[i, 1]] = \
                get_value(config.p_match_insert, h, i)

            # Insert self-loop
            transitions[ind.insert_to_insert[i, 0], ind.insert_to_insert[i, 1]] = \
                get_value(config.p_insert_insert, h, i)

            # Insert to match
            transitions[ind.insert_to_match[i, 0], ind.insert_to_match[i, 1]] = \
                1 - get_value(config.p_insert_insert, h, i)

            # Match to delete
            transitions[ind.match_to_delete[i, 0], ind.match_to_delete[i, 1]] = \
                get_value(config.p_match_delete, h, i)

            # Delete to delete
            transitions[ind.delete_to_delete[i, 0], ind.delete_to_delete[i, 1]] = \
                get_value(config.p_delete_delete, h, i)

            # Delete to match
            transitions[ind.delete_to_match[i, 0], ind.delete_to_match[i, 1]] = \
                1 - get_value(config.p_delete_delete, h, i)

            # Match i to end
            transitions[ind.match_to_end[i, 0], ind.match_to_end[i, 1]] = \
                (p_match_end_values[i] if config.p_match_end is None
                 else get_value(config.p_match_end, h, i))

        # Rescale other outgoing transitions from match i
        match_out_in_profile = np.sum(transitions[:L-1, :3*L-1], axis=-1)
        match_out_in_profile = np.maximum(match_out_in_profile, 1e-6)
        transitions[:L-1, :3*L-1] /= match_out_in_profile[:, np.newaxis]
        transitions[:L, :3*L-1] *=\
            (1 - transitions[ind.match_to_end[:, 0], ind.match_to_end[:, 1]])\
            [:, np.newaxis]

        # Match L to end
        transitions[ind.match_to_end[-1, 0], ind.match_to_end[-1, 1]] = 1.0

        # Delete L to end
        transitions[ind.delete_to_end[0, 0], ind.delete_to_end[0, 1]] = 1.0

        # End to unannotated
        transitions[ind.end[0, 0], ind.end[0, 1]] = \
            get_value(config.p_end_unannot, h)

        # Unannotated self-loop
        transitions[ind.unannotated[0, 0], ind.unannotated[0, 1]] = \
            get_value(config.p_unannot_unannot, h)

        # Unannotated to begin
        transitions[ind.unannotated[1, 0], ind.unannotated[1, 1]] = \
            1 - get_value(config.p_unannot_unannot, h)

        # End to right flank
        transitions[ind.end[1, 0], ind.end[1, 1]] = \
            get_value(config.p_end_right, h)

        # End to terminal
        transitions[ind.end[2, 0], ind.end[2, 1]] = (
            1 - get_value(config.p_end_unannot, h)
            - get_value(config.p_end_right, h)
        )

        # Right flank self-loop
        transitions[ind.right_flank[0, 0], ind.right_flank[0, 1]] = \
            get_value(config.p_right_right, h)

        # Right flank to terminal
        transitions[ind.right_flank[1, 0], ind.right_flank[1, 1]] = \
            1 - get_value(config.p_right_right, h)

        # Terminal self-loop
        transitions[ind.terminal[0, 0], ind.terminal[0, 1]] = 1.0

        # Check that all transition values are in [0, 1]
        if np.any((transitions < 0) | (transitions > 1)):
            invalid_mask = (transitions < 0) | (transitions > 1)
            invalid_indices = np.argwhere(invalid_mask)
            if len(invalid_indices) > 0:
                i, j = invalid_indices[0]
                raise ValueError(
                    f"Transition probability {state_index_to_name(i, L)} -> "
                    f"{state_index_to_name(j, L)} in head {h} with length {L} "
                    f"is {transitions[i, j]}, which is outside the "
                    "range [0, 1]"
                )

        # Check for that transitions is stochastic
        row_sums = np.sum(transitions, axis=-1)
        for i, row_sum in enumerate(row_sums):
            if not np.isclose(row_sum, 1.0):
                raise ValueError(
                    "Transition probabilities for state "
                    f"{state_index_to_name(i, L)} in head {h} with "
                    f"length {L} do not sum to 1 (sum={row_sum})"
                )

        # Starting probabilities
        start = np.array([
            get_value(config.p_start_left_flank, h),
            1 - get_value(config.p_start_left_flank, h)
        ], dtype=np.float32)

        # Match emissions
        if config.match_emissions is None:
            # Use background distribution for all match states
            match_emissions = np.stack(
                [config.background_distribution]*L, axis=0
            ).astype(np.float32)
        else:
            # Get emissions for each match state
            match_emissions_list = []
            for i in range(L):
                dist = get_emission_dist(
                    config.match_emissions,
                    head=h,
                    index=i,
                    default=config.background_distribution
                )
                match_emissions_list.append(np.array(dist, dtype=np.float32))
            match_emissions = np.stack(match_emissions_list, axis=0)

        # Insert emissions
        if config.insert_emissions is None:
            # Use background distribution
            insert_emissions = np.array(
                config.background_distribution,
                dtype=np.float32
            )
        else:
            dist = get_emission_dist(
                config.insert_emissions,
                head=h,
                default=config.background_distribution
            )
            insert_emissions = np.array(dist, dtype=np.float32)

        return cls(
            L=L,
            match_emissions=match_emissions,
            insert_emissions=insert_emissions,
            transitions=transitions,
            start=start,
        )

    @classmethod
    def from_msa(
        cls,
        data : AlignedDataset,
        match_threshold : float = 0.5,
        global_factor : float = 0.1,
    ) -> "PHMMValueSet":

        """
        Counts state emissions and transitions in an MSA.

        Args:
            data: An AlignedDataset object containing the MSA.
            match_threshold: A column is considered a match state if its occupancy
                (fraction of non-gap characters) is at least this value.
            global_factor: Mixing factor that describes the degree to which the
                MSA should be considered a global alignment. This is used
                to distinguish observed events that could either be attributed to
                local or global alignment. For example, we might observe that the
                first k match states are not used. This could either indicate
                deletions (global) or a "jump" into the profile (local).

        Returns:
            An HMMValueSet object with the counts.
        """
        gap_idx = data.alphabet.index('-')

        # Identify match columns based on occupancy threshold
        gaps = data.msa_matrix == gap_idx
        match_occupancy = np.mean(~gaps, axis=0)
        match_columns = match_occupancy >= match_threshold

        L = np.sum(match_columns) # number of match states
        N = data.msa_matrix.shape[0] # number of sequences
        assert L > 1, "Can not infer HMM from MSA: Not enough match states "\
            " found. Try lowering the match-threshold."

        # Count match emissions
        aa_indices = np.arange(len(data.alphabet))[np.newaxis, np.newaxis, :]
        matrix_expanded = data.msa_matrix[:, :, np.newaxis]
        # Sum over sequences
        counts = np.sum(matrix_expanded == aa_indices, axis=0)
        counts = counts[:, :-1] # exclude gaps
        counts = counts.astype(np.float32)

        # Separate counts for match and insert states
        match_counts = counts[match_columns, :]
        if np.any(~match_columns):
            insert_counts = counts[~match_columns, :]
            # Sum over insert states
            insert_counts = np.sum(insert_counts, axis=0)
        else:
            insert_counts = np.zeros(counts.shape[-1], dtype=np.float32)

        # -1 indicates a position that has no meaning in the HMM such as an
        # insertion with a "hole"
        state_seqs = -np.ones_like(data.msa_matrix, dtype=np.int32)

        # Compute match and delete states
        M = np.arange(L, dtype=np.int32)
        D = np.arange(L, dtype=np.int32) + 2*L - 1
        state_seqs[:, match_columns] = np.where(gaps[:, match_columns], D, M)

        # Compute insert states and flanks
        match_indices = np.arange(data.msa_matrix.shape[1])[match_columns]
        def _set_residues(i: int, j: int, state: int) -> None:
            # Sets all residues (not gaps) in columns i to j-1 to the given state
            state_seqs[:, i:j][~gaps[:, i:j]] = state
        # Left flank
        if match_indices[0] > 0:
            _set_residues(0, match_indices[0], state=3*L-1)
        # Inserts
        for i in range(L-1):
            _set_residues(match_indices[i]+1, match_indices[i+1], state=i+L)
        # Right flank
        _set_residues(match_indices[-1]+1, data.msa_matrix.shape[1], state=3*L+3)

        # Add terminal states
        state_seqs = np.pad(state_seqs, ((0,0),(0,1)), constant_values=3*L+4)

        state_seqs_global = state_seqs.copy()

        # Add B and E states
        M1 = match_indices[0]
        ML = match_indices[-1]
        state_seqs_global = np.insert(
            state_seqs_global, [M1, ML+1], [3*L, 3*L+1], axis=1
        )

        counts_global = _count_transitions(state_seqs_global, L)

        # Remove all cases where we counted transitions evolving deletes before the
        # first or after the last match states
        # Delete states are in range [2*L-1, 3*L-2], corresponding to match states
        # [0, L-1]

        # Indices of the first and last match states in {0, ..., L-1}
        first_match_indices = np.argmax(
            1.-gaps[:, match_columns], axis=1
        )
        last_match_indices = L - 1 - np.argmax(
            (1.-gaps[:, match_columns])[:, ::-1], axis=1
        )
        first_m = first_match_indices[:, np.newaxis]  # Shape (N, 1)
        last_m = last_match_indices[:, np.newaxis]    # Shape (N, 1)

        # Mark delete states before first match (D0 to D_{first_m-1})
        dont_count_before = (state_seqs >= 2*L-1) & (state_seqs < 2*L-1 + first_m)

        # Mark delete states after last match (D_{last_m+1} to D_{L-1})
        dont_count_after = (state_seqs > 2*L-1 + last_m) & (state_seqs < 3*L-1)

        # Combine the masks
        dont_count = dont_count_before | dont_count_after
        state_seqs[dont_count] = -1

        # Reassign all insertions before the first match to the left flank
        # Insert states are in range [L, 2*L-2] for I1 to IL-1
        # Reassign insert states I0 to I_{first_m-1} to left flank (3*L-1)
        reassign_inserts = (state_seqs >= L) & (state_seqs <= L + first_m - 1)
        state_seqs[reassign_inserts] = 3*L-1

        # Ensure the B state is to the right of all left flank states
        # We swap the position before the first match state and B
        #state_seqs[]

        # Reassign all insertions after the last match to the right flank
        # Reassign insert states I_{last_m+1} to IL-1 to right flank (3*L+3)
        reassign_inserts_right = (state_seqs >= L + last_m) & (state_seqs < 2*L-1)
        state_seqs[reassign_inserts_right] = 3*L+3

        # Add begin, end states at row-specific positions (vectorized)
        # For each row, insert B (3*L) at first_m and E (3*L+1) at last_m+1
        N, seq_len = state_seqs.shape
        new_state_seqs = np.full((N, seq_len + 2), -1, dtype=np.int32)

        # Create index arrays for vectorized assignment
        row_idx = np.arange(N)[:, np.newaxis]
        col_idx = np.arange(seq_len + 2)[np.newaxis, :]

        # Determine which source column to copy from (or -1 for inserted states)
        # Before first_pos: source_col = col_idx
        # At first_pos: insert B (source_col = -1)
        # Between first_pos+1 and last_pos+1: source_col = col_idx - 1
        # At last_pos+2: insert E (source_col = -1)
        # After last_pos+2: source_col = col_idx - 2

        first_m_global = match_indices[first_m]
        last_m_global = match_indices[last_m]

        source_col = np.where(
            col_idx < first_m_global, col_idx,
            np.where(col_idx == first_m_global, -1,
            np.where(col_idx <= last_m_global + 1, col_idx - 1,
            np.where(col_idx == last_m_global + 2, -1, col_idx - 2)))
        )

        # Copy values from state_seqs where source_col >= 0
        # We need to expand row_idx to match the shape of source_col for proper
        # broadcasting
        valid_mask = source_col >= 0
        row_indices = np.broadcast_to(row_idx, source_col.shape)
        new_state_seqs[valid_mask] = state_seqs[
            row_indices[valid_mask], source_col[valid_mask]
        ]

        # Insert B (3*L) at first_match_indices for each row
        new_state_seqs[row_idx[:, 0], first_m_global[:, 0]] = 3*L

        # Insert E (3*L+1) at last_match_indices + 2 for each row
        new_state_seqs[row_idx[:, 0], last_m_global[:, 0] + 2] = 3*L + 1

        state_seqs = new_state_seqs

        counts_local = _count_transitions(state_seqs, L)

        transitions = (1.-global_factor) * counts_local + global_factor * counts_global

        # Count how many sequences start in the left flanking state
        flank_start = np.sum(np.any(state_seqs == 0, axis=1))
        start = np.array([flank_start, N - flank_start], dtype=np.float32)

        return cls(
            L=L,
            match_emissions=match_counts,
            insert_emissions=insert_counts,
            transitions=transitions,
            start=start
        )



    def matches(self) -> int:
        """ Returns the number of match states `n`. """
        return self.match_emissions.shape[0]

    def add_pseudocounts(
        self,
        aa : np.ndarray | list | float = 0,
        match_transition : np.ndarray | list | float = 0,
        insert_transition : np.ndarray | list | float = 0,
        delete_transition : np.ndarray | list | float = 0,
        begin_to_match : np.ndarray | list | float = 0,
        begin_to_delete : float = 0,
        match_to_end : float = 0,
        left_flank : np.ndarray | list | float = 0,
        right_flank : np.ndarray | list | float = 0,
        unannotated : np.ndarray | list | float = 0,
        end : np.ndarray | list | float = 0,
        flank_start : np.ndarray | list | float = 0,
    ) -> "PHMMValueSet":
        """
        Adds pseudocounts to the given HMM counts in-place and returns a reference
        to the modified object.

        Args:
            counts: An HMMValueSet object containing the counts.
            aa: Optional pseudocounts for amino acids to add to
                emission counts (should be either a scalar or a 1D array of length
                equal to the alphabet size - 1).
            match_transition: Optional pseudocounts for match
                transition counts (should be a scalar or a 1D array of length 3
                [match, insert, delete]).
            insert_transition: Optional pseudocounts for insert
                transition counts (should be a scalar or a 1D array of length 2
                [loop, exit]).
            delete_transition: Optional pseudocounts for delete
                transition counts (should be a scalar or a 1D array of length 2
                [continue, exit]).
            begin_to_match: Optional pseudocounts for the counts of
                transitions from the begin state to the match states (should be a
                scalar or a 1D array of length 2 [first, others]).
            begin_to_delete: Optional pseudocount for the transition of the
                begin state to the first delete state (should be a scalar).
            match_to_end: Optional pseudocounts for the counts of
                transitions from the match states to the end state (should be a
                scalar).
            left_flank: Optional pseudocounts for the counts of
                transitions from the left flanking state (should be a scalar or a
                1D array of length 2 [loop, exit]).
            right_flank: Optional pseudocounts for the counts of
                transitions from the right flanking state (should be a scalar or a
                1D array of length 2 [loop, exit]).
            unannotated: Optional pseudocounts for the counts
                from the unannotated segment state (should be a scalar or a
                1D array of length 2 [loop, exit]).
            end: Optional pseudocounts for the counts of
                transitions from the end state (should be a scalar or a
                1D array of length 3 [unannotated, right_flank, terminal]).
            flank_start: Optional pseudocounts for the probability
                of starting in the left flanking state (should be a scalar or a
                1D array of length 2).
        """
        L = self.matches()

        self.match_emissions += aa
        self.insert_emissions += aa

        # Apply the pseudocounts for transitions
        match_transition = _expand(match_transition, 3)
        insert_transition = _expand(insert_transition, 2)
        delete_transition = _expand(delete_transition, 2)
        if isinstance(begin_to_match, (list, np.ndarray)):
            begin_to_match = np.concat((
                begin_to_match[0:1],
                _expand(begin_to_match[1], L-1)
            ))
        else: # assume scalar
            begin_to_match = _expand(begin_to_match, L)
        left_flank = _expand(left_flank, 2)
        right_flank = _expand(right_flank, 2)
        unannotated = _expand(unannotated, 2)
        flank_start = _expand(flank_start, 2)

        ind = PHMMTransitionIndexSet(L)

        _add(self.transitions, ind.match_to_match, match_transition[0])
        _add(self.transitions, ind.match_to_insert, match_transition[1])
        _add(self.transitions, ind.match_to_delete, match_transition[2])
        _add(self.transitions, ind.insert_to_insert, insert_transition[0])
        _add(self.transitions, ind.insert_to_match, insert_transition[1])
        _add(self.transitions, ind.delete_to_delete, delete_transition[0])
        _add(self.transitions, ind.delete_to_match, delete_transition[1])
        self.transitions[ind.begin_to_match[:,0], ind.begin_to_match[:,1]] += \
            begin_to_match
        self.transitions[ind.begin_to_delete[0,0], ind.begin_to_delete[0,1]] += \
            begin_to_delete
        _add(self.transitions, ind.match_to_end, match_to_end)

        self.transitions[ind.left_flank[:,0], ind.left_flank[:,1]] += \
            left_flank
        self.transitions[ind.right_flank[:,0], ind.right_flank[:,1]] += \
            right_flank
        self.transitions[ind.unannotated[:,0], ind.unannotated[:,1]] += \
            unannotated
        self.transitions[ind.end[:,0], ind.end[:,1]] += end

        self.start += flank_start

        ## add a small number to the out transitions of the last delete state
        ## and the terminal self loop to avoid zero rows
        self.transitions[2*L-1 + (L-1), 3*L+1] += 1e-8
        self.transitions[-1, -1] += 1e-8

        return self

    def normalize(self, log_zero_value=-1e8) -> "PHMMValueSet":
        """
        Normalizes the counts in the given HMMValueSet to probabilities
        in-place and returns a reference to the modified object.
        """
        # Normalize emissions
        self.match_emissions /= \
            np.sum(self.match_emissions, axis=-1, keepdims=True)
        self.insert_emissions /= np.sum(self.insert_emissions)

        # Normalize starting probabilities
        self.start /= np.sum(self.start)

        # Mask for invalid
        mask = PHMMTransitionIndexSet(L = self.matches()).mask()

        # Normalize transitions
        self.transitions[-1, -1] = 1 # terminal state can loop to itself
        self.transitions /= np.sum(self.transitions, axis=-1, keepdims=True)

        return self

    def log(self, log_zero_value=-1e8) -> "PHMMValueSet":
        """
        Applies element-wise logarithm to all values. Should be used after
        normalize(). Computes log(0) as log_zero_value.
        Operates in-place and returns a reference to the modified object.
        """

        # Apply log transform with error handling
        with np.errstate(divide='raise', invalid='raise'):
            try:
                self.match_emissions = safe_log(
                    self.match_emissions, log_zero_value
                )
                self.insert_emissions = safe_log(
                    self.insert_emissions, log_zero_value
                )
                self.start = safe_log(self.start, log_zero_value)
                self.transitions = safe_log(self.transitions, log_zero_value)
            except FloatingPointError as e:
                raise ValueError(
                    "Cannot compute log probabilities. "
                    "This typically indicates that an emission or transition has "
                    "been counted zero times. "
                    "Consider adding pseudocounts using add_pseudocounts() before "
                    "calling log_normalize()."
                ) from e

        return self

    def add_noise(
        self, concentration: float = 100.0
    ) -> "PHMMValueSet":
        """
        Adds Dirichlet noise to probability distributions.

        Args:
            concentration: Concentration parameter controlling noise strength.
                Higher values produce smaller perturbations.

        Returns:
            Reference to the modified PHMMValueSet object.
        """
        def add_dirichlet_noise_sparse(
            probs: np.ndarray, concentration: float
        ) -> np.ndarray:
            """Add Dirichlet noise while preserving zeros (sparsity)."""
            result = probs.copy()

            if probs.ndim == 1:
                # Single distribution
                mask = probs > 0
                if np.any(mask):
                    nonzero_probs = probs[mask]
                    alpha = concentration * nonzero_probs
                    alpha = np.maximum(alpha, 1e-6)
                    sampled = np.random.dirichlet(alpha)
                    result[mask] = sampled
            else:
                # Multiple distributions (rows)
                for i in range(probs.shape[0]):
                    mask = probs[i] > 0
                    if np.any(mask):
                        nonzero_probs = probs[i, mask]
                        alpha = concentration * nonzero_probs
                        alpha = np.maximum(alpha, 1e-6)
                        sampled = np.random.dirichlet(alpha)
                        result[i, mask] = sampled

            return result

        # Apply Dirichlet noise to emissions (all entries typically non-zero)
        self.match_emissions = add_dirichlet_noise_sparse(
            self.match_emissions, concentration
        )
        self.insert_emissions = add_dirichlet_noise_sparse(
            self.insert_emissions, concentration
        )
        self.start = add_dirichlet_noise_sparse(self.start, concentration)

        # Apply Dirichlet noise to transitions (preserving structural zeros!)
        self.transitions = add_dirichlet_noise_sparse(
            self.transitions, concentration
        )
        # Ensure terminal state loops to itself
        self.transitions[-1, -1] = 1.0

        return self


def safe_log(x: np.ndarray, log_zero_value: float = -1e8) -> np.ndarray:
    """
    Computes the logarithm of x, replacing -inf with log_zero_value.
    """
    with np.errstate(divide='ignore'):
        log_x = np.log(x)
    log_x[~np.isfinite(log_x)] = log_zero_value
    return log_x


def _count_transitions(state_seqs: np.ndarray, L: int) -> np.ndarray:
    """
    Counts all transitions in the given state sequences and returns
    a square counting matrix.
    """

    # Remove -1 and flatten the state sequences
    mask = state_seqs != -1
    state_seqs_flat = state_seqs[mask]

    # Make pairs for easier counting
    state_seqs_flat = np.stack(
        (state_seqs_flat[:-1], state_seqs_flat[1:]), axis=1
    )

    values, counts = np.unique(state_seqs_flat, axis=0, return_counts=True)

    # We'll compute two counting matrices since for some alignments, alternative
    # counting schemes might be desired - either global (using delete states
    # for flanks) or local (using jumping edges into the profile for flanks).

    # global counting scheme
    count_matrix = np.zeros((3*L+5, 3*L+5), dtype=np.float32)
    np.add.at(count_matrix, (values[:, 0], values[:, 1]), counts)
    count_matrix[-1] = 0 # don't count transitions from terminal state

    return count_matrix


def _expand(x: np.ndarray | list | float, n: int) -> np.ndarray:
    """
    Expands x to a vector of length n if x is a scalar. If x is already
    a vector of length n, it is returned as a numpy array. If x is a list or
    numpy array of different length, an error is raised.
    """
    if isinstance(x, (list, np.ndarray)):
        if len(x) != n:
            raise ValueError(f"Expected length {n}, got {len(x)}")
        return np.array(x, dtype=np.float32)
    else:
        return np.full((n,), x, dtype=np.float32)


def _add(
        counts: np.ndarray, indices: np.ndarray, values: float | np.ndarray
) -> None:
    """
    Adds the given values to the counts at the specified indices in-place.
    """
    np.add.at(counts, (indices[..., 0], indices[..., 1]), values)
