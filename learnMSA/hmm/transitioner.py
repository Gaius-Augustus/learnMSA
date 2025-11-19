from collections.abc import Sequence
from typing import override

import numpy as np
import tensorflow as tf
from hidten.hmm import HMMConfig as HidtenHMMConfig
from hidten.tf.transitioner import (T_TFTensor, TFTransitioner, shared_tensor,
                                    TransitionMode)
from hidten.tf.util import zero_row_softmax, log_zero

from learnMSA.hmm.transition_index_set import PHMMTransitionIndexSet
from learnMSA.hmm.value_set import PHMMValueSet


def logsumexp(x: T_TFTensor, y: T_TFTensor) -> T_TFTensor:
    """Compute log(exp(x) + exp(y)) in a numerically stable way."""
    return tf.math.log(tf.math.exp(x) + tf.math.exp(y))


class PHMMExplicitTransitioner(TFTransitioner):
    """A transitioner for explicit pHMMs with deletion states.
    This transitioner contains silent states and needs to be folded.

    The order of states in each head is:

    M1 ... ML I1 ... IL-1 D1 ... DL L B E C R T.

    where L is the number of match states in that head.

    Args:
        values (Sequence[PHMMValueSet]): A sequence of value sets, one per head.
        hidten_hmm_config (HidtenHMMConfig): The configuration of the hidten HMM.
    """
    def __init__(
        self,
        values: Sequence[PHMMValueSet],
        hidten_hmm_config: HidtenHMMConfig,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.hmm_config = hidten_hmm_config
        transitions, value_list = [], []
        start, start_values = [], []
        for h, value_set in enumerate(values):
            index_set = PHMMTransitionIndexSet(value_set.L, folded=False)
            # Transitions
            # get all index pairs (i,j) and add head index (h,i,j)
            indices = index_set.as_array()
            transitions.append(
               np.pad(indices, ((0,0),(1,0)), constant_values=h)
            )
            value_list.append(value_set.transitions[indices[:,0], indices[:,1]])

            # Start distribution
            start.append(np.pad(
                index_set.start[:,np.newaxis], ((0,0), (1,0)), constant_values=h
            ))
            start_values.append(value_set.start)

        self.allow = np.vstack(transitions).tolist()
        self.initializer = np.hstack(value_list)

        self.allow_start = np.vstack(start).tolist()
        self.initializer_start = np.hstack(start_values)


class PHMMTransitioner(TFTransitioner):
    """A transitioner for folded pHMMs without deletion states. Wraps an
    explicit transitioner which holds all the parameters. Overrides the matrix
    and start distribution methods to provide the folded versions.

    The order of states in each head is:

    M1 ... ML I1 ... IL-1 L B E C R T.

    where L is the number of match states in that head.
    """
    def __init__(
        self,
        values: Sequence[PHMMValueSet],
        hidten_hmm_config: HidtenHMMConfig,
        **kwargs
    ) -> None:
        """
        Args:
            values (Sequence[PHMMValueSet]): A sequence of value sets, one per head.
            hidten_hmm_config (HidtenHMMConfig): The configuration of the hidten HMM.
        """
        super().__init__(**kwargs)
        self.explicit_transitioner = self._make_explicit_transitioner(values)
        self.hmm_config = hidten_hmm_config
        self.lengths = [value_set.L for value_set in values]
        # Construct allow indices for the folded models
        transitions, start = [], []
        for h, L in enumerate(self.lengths):
            index_set = PHMMTransitionIndexSet(L, folded=True)
            # Transitions
            # get all index pairs (i,j) and add head index (h,i,j)
            indices = index_set.as_array()
            transitions.append(
               np.pad(indices, ((0,0),(1,0)), constant_values=h)
            )

            # Start distribution
            start.append(np.pad(
                index_set.start[:, np.newaxis], ((0,0),(1,0)), constant_values=h
            ))

        self.allow = np.vstack(transitions).tolist()
        self.allow_start = np.vstack(start).tolist()

    @override
    def build(self) -> None:
        # don't call super.build(), as this transitioner only folds and has no
        # own parameters
        self.explicit_transitioner.build()
        # Initialize the precomputed tensors required to build matrix and start
        self.refresh()

    @override
    def _launch(
        self,
        mode: TransitionMode = TransitionMode.SUM,
        use_padding: bool = False,
    ) -> T_TFTensor:
        # Every time we launch, we need to refresh
        self.refresh()
        return super()._launch(mode, use_padding)

    def refresh(self) -> None:
        self.log_explicit_matrix = tf.math.log(
            self.explicit_transitioner.matrix()
        )
        self.match_skip = []
        for h in range(self.hmm_config.heads):
            self.match_skip.append(
                self._compute_match_skip_matrix(h)
            )

    @override
    def matrix(self) -> T_TFTensor:
        # Construct the matrix like usual in the transitioner, but instead of
        # using a parameter kernel, we use the folded transition probabilities
        return tf.math.exp(shared_tensor(
            indices=tf.constant(self.allow, dtype=tf.int64),
            values=self._get_folded_transition_probs(),
            shape=tf.constant(
                [
                    self.hmm_config.heads,
                    self.hmm_config.max_states,
                    self.matrix_dim,
                ],
                dtype=tf.int64,
            ),
            share=None,
        ))

    @override
    def start_dist(self) -> T_TFTensor:
        # Same principle: construct start distribution from explicit transition
        # probabilities
        return tf.math.exp(shared_tensor(
            indices=tf.constant(self.allow_start, dtype=tf.int64),
            values=self._get_folded_start_probs(),
            shape=tf.constant(
                [self.hmm_config.heads, self.hmm_config.max_states],
                dtype=tf.int64,
            ),
            share=None,
        ))

    def _get_folded_transition_probs(self) -> T_TFTensor:
        """Computes folded transition probabilities by marginalizing over
        silent delete, begin, and end states.

        Returns:
            A 1D tensor containing all folded transition probabilities,
            ordered according to self.allow indices.
        """
        folded_probs = []

        for h, L in enumerate(self.lengths):
            idx = PHMMTransitionIndexSet(L, folded=False)
            log_mat = self.log_explicit_matrix[h]
            M_skip = self.match_skip[h]

            # Helper to extract log prob from matrix
            def get(indices):
                return tf.gather_nd(log_mat, indices)

            # Get begin and end related transitions
            BM = get(idx.begin_to_match)  # Shape: (L,)
            ME = get(idx.match_to_end)  # Shape: (L,)
            E = get(idx.end)  # Shape: (3,) - [to_unannot, to_right, to_terminal]

            log_z = log_zero(log_mat)

            # The added costs of entering all match states either with direct
            # edges or via deletes
            entry_add = logsumexp(
                BM,
                # Add log_zero at the first position, as M1 can not be reached
                # via deletes
                tf.pad(M_skip[0, :-1], [[1,0]], constant_values=log_z)
            ) # Shape: (L,)

            # The added costs of exiting from all match states (analogously)
            exit_add = logsumexp(
                ME,
                # Add log_zero at the last position, as ML can not go to End
                # via deletes
                tf.pad(M_skip[1:,-1], [[0,1]], constant_values=log_z)
            ) # Shape: (L,)

            # Now build the folded transition probabilities in the order
            # specified by folded_idx

            # Common transitions (same in both folded and unfolded)
            MM = get(idx.match_to_match)
            MI = get(idx.match_to_insert)
            II = get(idx.insert_to_insert)
            IM = get(idx.insert_to_match)

            folded_probs.extend([MM, MI, II, IM])

            # Match to match jump transitions
            # Extract upper triangle of M_skip
            for i in range(1, L - 1):
                jump_probs = M_skip[i, i:-1]
                # Jumps from M_{i+1} to M_{i+3}, M_{i+4}, ..., M_L
                folded_probs.append(jump_probs)

            # Match to unannotated: M_i -> End -> Unannotated
            MU = exit_add + E[0]
            folded_probs.append(MU)

            # Match to right flank: M_i -> End -> Right
            MR = exit_add + E[1]
            folded_probs.append(MR)

            # Match to terminal: M_i -> End -> Terminal
            MT = exit_add + E[2]
            folded_probs.append(MT)

            # Left flank transitions
            LF = get(idx.left_flank)

            # Left flank self-loop
            folded_probs.append(LF[:1])

            # Left flank to match states: L -> Begin -> M_i
            LFM = LF[1:2] + entry_add
            folded_probs.append(LFM)

            # For left flank to unannotated/right/terminal, we need the path:
            # LF -> Begin -> D_1 -> ... -> D_L -> End -> (Unannot/Right/Terminal)
            LFE = LF[1:2] + M_skip[0,-1]

            LFU = LFE + E[0]
            folded_probs.append(LFU)

            LFR = LFE + E[1]
            folded_probs.append(LFR)

            LFT = LFE + E[2]
            folded_probs.append(LFT)

            # Right flank transitions
            p_right_flank = get(idx.right_flank)
            folded_probs.extend([
                tf.expand_dims(p_right_flank[0], 0),
                tf.expand_dims(p_right_flank[1], 0)
            ])

            # Unannotated transitions
            U = get(idx.unannotated)

            # Compute path U -> Begin -> deletes -> End for reuse
            UE = U[1] + M_skip[0, -1]

            # Unannotated self-loop: direct loop OR U -> Begin -> all deletes -> End -> U
            UU = logsumexp(U[0], UE + E[0])
            folded_probs.append(tf.expand_dims(UU, 0))

            # Unannotated to match: U -> Begin -> M_i
            UM = U[1] + entry_add
            folded_probs.append(UM)

            # Unannotated to right: U -> Begin -> deletes -> End -> Right
            URF = UE + E[1]
            folded_probs.append(tf.expand_dims(URF, 0))

            # Unannotated to terminal
            UT = UE + E[2]
            folded_probs.append(tf.expand_dims(UT, 0))

            # Terminal self-loop
            folded_probs.append(get(idx.terminal))

        # Concatenate across all heads
        return tf.concat(folded_probs, axis=0)

    def _get_folded_start_probs(self) -> T_TFTensor:
        """Computes folded start probabilities.

        In the folded model, we can start in:
        - Left flank L
        - Match states M_1, ..., M_L (via Begin state)
        - Unannotated C (via Begin state)
        - Right flank R (via Begin state)
        - Terminal T (via Begin state)

        Returns:
            A 1D tensor containing start probabilities for allowed starting
            states.
        """
        # Get the explicit start distribution
        explicit_start = tf.math.log(self.explicit_transitioner.start_dist())

        start_probs = []

        for h, L in enumerate(self.lengths):
            idx = PHMMTransitionIndexSet(L, folded=False)
            log_mat = self.log_explicit_matrix[h]
            M_skip = self.match_skip[h]

            # Helper to extract log prob from matrix
            def get(indices):
                return tf.gather_nd(log_mat, indices)

            log_z = log_zero(log_mat)

            # The added costs of entering all match states either with direct
            # edges or via deletes
            entry_add = logsumexp(
                get(idx.begin_to_match),
                # Add log_zero at the first position, as M1 can not be reached
                # via deletes
                tf.pad(M_skip[0, :-1], [[1,0]], constant_values=log_z)
            ) # Shape: (L,)

            # In explicit model, we can start in:
            # - Left flank (index 3*L-1)
            # - Begin (index 3*L)

            start_left = explicit_start[h, 3*L - 1]
            start_begin = explicit_start[h, 3*L]

            # Match states
            p_start_match = start_begin + entry_add
            start_probs.append(p_start_match)

            # Left flank
            start_probs.append(tf.expand_dims(start_left, 0))

            # Unannotated - can be reached via Begin -> deletes -> End -> Unannot
            # But this is typically not a valid start state, so we use log_zero
            # However, looking at the folded index set, C is listed as a valid start state
            # Let's compute it properly
            p_delete_to_end = get(idx.delete_to_end)
            p_end = get(idx.end)

            BE = start_begin + M_skip[0, -1]

            p_start_unannot = BE + p_end[0]
            start_probs.append(tf.expand_dims(p_start_unannot, 0))

            # Right flank
            p_start_right = BE + p_end[1]
            start_probs.append(tf.expand_dims(p_start_right, 0))

            # Terminal
            p_start_terminal = BE + p_end[2]
            start_probs.append(tf.expand_dims(p_start_terminal, 0))

        # Concatenate across all heads
        return tf.concat(start_probs, axis=0)

    def _compute_match_skip_matrix(self, h: int) -> T_TFTensor:
        """
        Utility method that computes the `L x L` match skip transition matrix
        for head `h` with `match_skip(i,j) = P(Mj+2 | Mi)`.
        With `M0 := Begin` and `ML+1 := End`.
        """
        L = self.lengths[h]
        log_mat = self.log_explicit_matrix[h]

        # Create index sets for explicit and folded models
        idx = PHMMTransitionIndexSet(L, folded=False)

        # Helper to extract log prob from matrix
        def get(indices):
            return tf.gather_nd(log_mat, indices)

        # Get transition log probabilities
        MD = get(idx.match_to_delete)  # Shape: (L-1,)
        DD = get(idx.delete_to_delete)  # Shape: (L-1,)
        DM = get(idx.delete_to_match)  # Shape: (L-1,)

        # Concatenate B -> D1 and DL -> E transitions
        begin_to_delete = get(idx.begin_to_delete)  # Shape: scalar
        delete_to_end = get(idx.delete_to_end)  # Shape: scalar

        MD = tf.concat([begin_to_delete, MD], axis=0)  # Shape: (L,)
        DM = tf.concat([DM, delete_to_end], axis=0)    # Shape: (L,)

        # Compute cumulative sum of delete-to-delete transitions
        # Prepend 0 for the first delete state D_0
        # Shape: (L,)
        DD_cumsum = tf.pad(tf.cumsum(DD), [[1, 0]], constant_values=0.0)

        # Compute the difference matrix for cumulative sums
        # Shape: (L, L)
        DD_diff = tf.expand_dims(DD_cumsum, 0) - tf.expand_dims(DD_cumsum, 1)

        # Build M_skip matrix
        MD_expanded = tf.expand_dims(MD, -1)  # Shape: (L, 1)
        DM_expanded = tf.expand_dims(DM, 0)  # Shape: (1, L)

        M_skip = MD_expanded + DD_diff + DM_expanded  # Shape: (L, L)
        return M_skip

    def _make_explicit_transitioner(
        self, values: Sequence[PHMMValueSet]
    ) -> PHMMExplicitTransitioner:
        """Helper to create the explicit transitioner with the same parameters."""
        # Since the explicit transitioner has more states (including the silent
        # ones) we need to create a new HMMConfig for it
        states = [
            PHMMTransitionIndexSet.num_states_unfolded(L=value_set.L)
            for value_set in values
        ]
        return PHMMExplicitTransitioner(
            values=values,
            hidten_hmm_config=HidtenHMMConfig(states=states),
        )
