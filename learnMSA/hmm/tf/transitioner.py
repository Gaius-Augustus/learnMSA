from collections.abc import Sequence
from typing import override

import numpy as np
import tensorflow as tf
from hidten.hmm import HMMConfig as HidtenHMMConfig
from hidten.prior import Prior
from hidten.tf.transitioner import (T_TFTensor, TFTransitioner, TransitionMode,
                                    shared_tensor)
from hidten.tf.util import log_zero, safe_log, tiny, zero_row_softmax

from learnMSA.hmm.util.transition_index_set import PHMMTransitionIndexSet
from learnMSA.hmm.util.value_set import PHMMValueSet


def logsumexp(x: T_TFTensor, y: T_TFTensor) -> T_TFTensor:
    """Compute log(exp(x) + exp(y)) in a numerically stable way."""
    m = tf.maximum(x, y)
    return m + tf.math.log(tf.exp(x - m) + tf.exp(y - m))


class PHMMExplicitTransitioner(TFTransitioner):
    """A transitioner for explicit pHMMs with deletion states.
    This transitioner contains silent states and needs to be folded.

    The order of states in each head is:

    M1 ... ML I1 ... IL-1 D1 ... DL L B E C R [... T],

    where L is the number of match states in that head.
    [... T] indicates that the last position is always the terminal state, no
    matter how many states there are in the head, with optional padding states
    to fill up to the maximum number of states across all heads.
    """
    def __init__(
        self,
        values: Sequence[PHMMValueSet],
        **kwargs
    ) -> None:
        """
        Args:
            values (Sequence[PHMMValueSet]): A sequence of value sets,
                one per head, with probabilities.
            hidten_hmm_config (HidtenHMMConfig): The configuration of the
                hidten HMM.
        """
        super().__init__(**kwargs)
        transitions, value_list = [], []
        start, start_values = [], []
        states = []
        lengths = [value_set.L for value_set in values]
        max_states = PHMMTransitionIndexSet.num_states_unfolded(max(lengths))
        for h, value_set in enumerate(values):
            index_set = PHMMTransitionIndexSet(value_set.L, folded=False)
            # Transitions
            # get all index pairs (i,j) and add head index (h,i,j)
            indices = index_set.as_array()
            # Add the values
            value_list.append(value_set.transitions[indices[:,0], indices[:,1]])
            # Handle negative indices (access from the end)
            indices[indices < 0] += max_states
            # Add the indices with head index
            transitions.append(
               np.pad(indices, ((0,0),(1,0)), constant_values=h)
            )

            # Start distribution
            start_indices = index_set.start[:, np.newaxis]
            # Handle negative indices (access from the end)
            start_indices[start_indices < 0] += max_states
            start.append(np.pad(
                start_indices, ((0,0), (1,0)), constant_values=h
            ))
            start_values.append(value_set.start)

            states.append(
                PHMMTransitionIndexSet.num_states_unfolded(L=value_set.L)
            )

        # Set a custom HMMConfig for the explicit model
        # because the state count differs from the folded model
        self.hmm_config = HidtenHMMConfig(states=states)

        self.allow = np.vstack(transitions)
        self.initializer = np.hstack(value_list)

        self.allow_start = np.vstack(start)
        self.initializer_start = np.hstack(start_values)

    @override
    def matrix(self) -> T_TFTensor:
        """Override to add numerical stability to avoid numerical issues
        when folding."""
        # Add epsilon in probability space to ensure that no allowed
        # transition has vanishing probability
        kernel = tf.math.log(tf.math.exp(self.kernel) + 1e-16)
        dense_tensor = shared_tensor(
            indices=self.allow, # type: ignore
            values=kernel,
            shape=tf.constant(
                [
                    self.heads,
                    self.max_states,
                    self.matrix_dim,
                ],
                dtype=tf.int64,
            ),
            share=self.share, # type: ignore
        )
        return zero_row_softmax(dense_tensor)


class PHMMTransitioner(TFTransitioner):
    """A transitioner for folded pHMMs without deletion states. Wraps an
    explicit transitioner which holds all the parameters. Overrides the matrix
    and start distribution methods to provide the folded versions.

    The order of states in each head is:

    M1 ... ML I1 ... IL-1 L C R [... T],

    where L is the number of match states in that head.
    [... T] indicates that the last position is always the terminal state, no
    matter how many states there are in the head, with optional padding states
    to fill up to the maximum number of states across all heads.
    """
    @property
    def max_states(self) -> int:
        """The maximum number of states across all heads. May be restricted
        by head_subset."""
        return self.hmm_config.max_states + 1

    @property
    def states(self) -> list[int]:
        """The number of states for each head. May be restricted
        by head_subset."""
        return [Q+1 for Q in self.hmm_config.states]

    @property
    def prior(self) -> "Prior[T_TFTensor] | None":
        return self.explicit_transitioner.prior

    @prior.setter
    def prior(self, prior: "Prior[T_TFTensor]") -> None:
        self.explicit_transitioner.prior = prior

    @property
    def prior_start(self) -> "Prior[T_TFTensor] | None":
        return self.explicit_transitioner.prior_start

    @prior_start.setter
    def prior_start(self, prior_start: "Prior[T_TFTensor]") -> None:
        self.explicit_transitioner.prior_start = prior_start

    head_subset : Sequence[int] | None = None
    """If set, only these heads are used in computations."""

    def __init__(
        self,
        values: Sequence[PHMMValueSet],
        **kwargs
    ) -> None:
        """
        Args:
            values (Sequence[PHMMValueSet]): A sequence of value sets, one per head.
        """
        super().__init__(**kwargs)

        self.explicit_transitioner = self._make_explicit_transitioner(values)
        self.lengths = [value_set.L for value_set in values]

        # Construct allow indices for the folded models
        transitions, start = [], []
        states = []
        max_states = PHMMTransitionIndexSet.num_states_folded(max(self.lengths))
        for h, L in enumerate(self.lengths):
            index_set = PHMMTransitionIndexSet(L, folded=True)
            # Transitions
            # get all index pairs (i,j) and add head index (h,i,j)
            indices = index_set.as_array()
            # Handle negative indices (access from the end)
            indices[indices < 0] += max_states
            transitions.append(
               np.pad(indices, ((0,0),(1,0)), constant_values=h)
            )

            # Start distribution
            start_indices = index_set.start[:, np.newaxis]
            # Handle negative indices (access from the end)
            start_indices[start_indices < 0] += max_states
            start.append(np.pad(
                start_indices, ((0,0),(1,0)), constant_values=h
            ))

            states.append(PHMMTransitionIndexSet.num_states_folded(L=L))

        self.allow = np.vstack(transitions)
        self.allow_start = np.vstack(start)

    @override
    def build(self) -> None:
        # don't call super.build(), as this transitioner only folds and has no
        # own parameters
        self.explicit_transitioner.build()

    @override
    def _launch(
        self,
        mode: TransitionMode = TransitionMode.SUM,
        use_padding: bool = True, #not used
    ) -> T_TFTensor:
        # The HMM may pass use_padding=True, but this transitioner already
        # manages the terminal/padding semantics explicitly.
        # Keep the folded matrix/start dimensions at Q (no extra padding row/col).
        self.mode = mode

        log_explicit_matrix = safe_log(self.explicit_transitioner.matrix())
        folded_transition_probs, folded_start_probs = \
            self._compute_folded_prob_vectors(log_explicit_matrix)

        self._A = self._build_folded_matrix(folded_transition_probs)

        H, Q, _ = tf.unstack(tf.shape(self._A))

        # Compute a starting distribution depending on the mode
        if TransitionMode.REVERSE in mode:
            start_dist = tf.ones(shape=(H, Q), dtype=self._A.dtype)
        else:
            start_dist = self._build_folded_start_dist(folded_start_probs)

        if TransitionMode.ALLOWED in mode:
            self._A = tf.where(self._A > tiny(self._A), 1., 0.)

        if TransitionMode.LOG_SUM_EXP in mode or TransitionMode.MAX in mode:
            self._A_log = safe_log(self._A)
            self._A_log_T = tf.transpose(self._A_log, [0, 2, 1])

        return start_dist

    def _build_folded_matrix(
        self, folded_transition_probs: T_TFTensor
    ) -> T_TFTensor:
        matrix = tf.math.exp(shared_tensor(
            indices=tf.constant(self.allow, dtype=tf.int64),
            values=folded_transition_probs,
            shape=tf.constant(
                [self.heads, self.max_states, self.matrix_dim],
                dtype=tf.int64,
            ),
            share=None,
        ))
        if self.head_subset is not None:
            matrix = tf.gather(matrix, self.head_subset, axis=0)
            max_states_subset = max(
                [self.hmm_config.states[h] for h in self.head_subset]
            )
            terminal_state_in = matrix[:, :max_states_subset, -1:]
            terminal_state_out = tf.one_hot(
                [[max_states_subset]], depth=max_states_subset+1
            )
            matrix = matrix[:, :max_states_subset, :max_states_subset]
            matrix = tf.concat([matrix, terminal_state_in], axis=2)
            matrix = tf.concat([matrix, terminal_state_out], axis=1)
        return matrix

    def _build_folded_start_dist(
        self, folded_start_probs: T_TFTensor
    ) -> T_TFTensor:
        start_dist = tf.math.exp(shared_tensor(
            indices=tf.constant(self.allow_start, dtype=tf.int64),
            values=folded_start_probs,
            shape=tf.constant(
                [self.heads, self.max_states],
                dtype=tf.int64,
            ),
            share=None,
        ))
        if self.head_subset is not None:
            start_dist = tf.gather(start_dist, self.head_subset, axis=0)
            max_states_subset = max(
                [self.hmm_config.states[h] for h in self.head_subset]
            )
            terminal_state = start_dist[:, -1:]
            start_dist = start_dist[:, :max_states_subset]
            start_dist = tf.concat([start_dist, terminal_state], axis=1)
        return start_dist

    @override
    def matrix(self) -> T_TFTensor:
        log_explicit_matrix = safe_log(self.explicit_transitioner.matrix())
        folded_transition_probs, _ = self._compute_folded_prob_vectors(
            log_explicit_matrix
        )
        return self._build_folded_matrix(folded_transition_probs)

    @override
    def start_dist(self) -> T_TFTensor:
        log_explicit_matrix = safe_log(self.explicit_transitioner.matrix())
        _, folded_start_probs = self._compute_folded_prob_vectors(
            log_explicit_matrix
        )
        return self._build_folded_start_dist(folded_start_probs)

    @override
    def prior_scores(self) -> T_TFTensor:
        return self.explicit_transitioner.prior_scores()

    def _compute_folded_prob_vectors(
        self,
        log_explicit_matrix: T_TFTensor,
    ) -> tuple[T_TFTensor, T_TFTensor]:
        """Compute folded transition and start log-probabilities in one pass.

        Returns:
            Tuple of flat tensors (transition_probs, start_probs), each ordered
            according to self.allow and self.allow_start respectively.
        """
        explicit_start = safe_log(self.explicit_transitioner.start_dist())
        max_states = PHMMTransitionIndexSet.num_states_unfolded(max(self.lengths))

        folded_transition_probs = []
        folded_start_probs = []

        for h, L in enumerate(self.lengths):
            idx = PHMMTransitionIndexSet(L, folded=False)
            log_mat = log_explicit_matrix[h]
            M_skip = self._compute_match_skip_matrix(h, log_mat=log_mat)

            def get(indices):
                gather_indices = np.array(indices, copy=True)
                gather_indices[gather_indices < 0] += max_states
                return tf.gather_nd(log_mat, gather_indices)

            BM = get(idx.begin_to_match)
            ME = get(idx.match_to_end)
            E = get(idx.end)

            log_z = log_zero(log_mat)
            entry_add = logsumexp(
                BM,
                tf.pad(M_skip[0, :-1], [[1, 0]], constant_values=log_z)
            )
            exit_add = logsumexp(
                ME,
                tf.pad(M_skip[1:, -1], [[0, 1]], constant_values=log_z)
            )

            # Folded transition probabilities
            MM = get(idx.match_to_match)
            MI = get(idx.match_to_insert)
            II = get(idx.insert_to_insert)
            IM = get(idx.insert_to_match)
            folded_transition_probs.extend([MM, MI, II, IM])

            for i in range(1, L - 1):
                folded_transition_probs.append(M_skip[i, i:-1])

            MU = exit_add + E[0]
            MR = exit_add + E[1]
            MT = exit_add + E[2]
            folded_transition_probs.extend([MU, MR, MT])

            LF = get(idx.left_flank)
            folded_transition_probs.append(LF[:1])
            folded_transition_probs.append(LF[1:2] + entry_add)

            LFE = LF[1:2] + M_skip[0, -1]
            folded_transition_probs.append(LFE + E[0])
            folded_transition_probs.append(LFE + E[1])
            folded_transition_probs.append(LFE + E[2])

            p_right_flank = get(idx.right_flank)
            folded_transition_probs.extend([
                tf.expand_dims(p_right_flank[0], 0),
                tf.expand_dims(p_right_flank[1], 0),
            ])

            U = get(idx.unannotated)
            UE = U[1] + M_skip[0, -1]
            UU = logsumexp(U[0], UE + E[0])
            folded_transition_probs.append(tf.expand_dims(UU, 0))
            folded_transition_probs.append(U[1] + entry_add)
            folded_transition_probs.append(tf.expand_dims(UE + E[1], 0))
            folded_transition_probs.append(tf.expand_dims(UE + E[2], 0))

            folded_transition_probs.append(get(idx.terminal))

            # Folded start probabilities
            start_left = explicit_start[h, 3*L - 1]
            start_begin = explicit_start[h, 3*L]
            BE = start_begin + M_skip[0, -1]

            folded_start_probs.append(start_begin + entry_add)
            folded_start_probs.append(tf.expand_dims(start_left, 0))
            folded_start_probs.append(tf.expand_dims(BE + E[0], 0))
            folded_start_probs.append(tf.expand_dims(BE + E[1], 0))
            folded_start_probs.append(tf.expand_dims(BE + E[2], 0))

        return (
            tf.concat(folded_transition_probs, axis=0),
            tf.concat(folded_start_probs, axis=0),
        )

    def _compute_match_skip_matrix(
        self,
        h: int,
        log_mat: T_TFTensor | None = None,
    ) -> T_TFTensor:
        """
        Utility method that computes the `L x L` match skip transition matrix
        for head `h` with `match_skip(i,j) = P(Mj+2 | Mi)`.
        With `M0 := Begin` and `ML+1 := End`.
        """
        L = self.lengths[h]
        if log_mat is None:
            log_mat = safe_log(self.explicit_transitioner.matrix())[h]

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
        """Helper to create the explicit transitioner with the same
        parameters."""
        return PHMMExplicitTransitioner(
            values=values
        )
