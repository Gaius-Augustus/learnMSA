import enum
from typing import Sequence, override

import numpy as np
import tensorflow as tf
from hidten.tf.prior.dirichlet import T_TFTensor, TFDirichletPrior, TFPrior
from hidten.tf.util import epsilon, safe_log

from learnMSA.config.hmm import PHMMPriorConfig
from learnMSA.hmm.tf.util import load_dirichlet
from learnMSA.hmm.util.transition_index_set import PHMMTransitionIndexSet


class TFPHMMTransitionPrior(TFPrior):
    """ A prior that uses Dirichlet distributions to score the transition
    probabilities of a profile HMM. Uses sub-priors for match, insert, and delete
    transitions.
    """

    def __init__(
        self,
        lengths: Sequence[int] | np.ndarray,
        prior_config: PHMMPriorConfig,
        **kwargs
    ) -> None:
        """
        Args:
            lengths (Sequence[int] | np.ndarray): The number of match states in each head
                of the pHMM.
            prior_config (HMMPriorConfig): Prior configuration containing alpha parameters
                for transition priors.
        """
        super().__init__(**kwargs)
        self.lengths = np.asarray(lengths)
        self.prior_config = prior_config
        transition_indices = [PHMMTransitionIndexSet(L=L) for L in lengths]

        def _pad_head(arr: np.ndarray, h: int) -> np.ndarray:
            """Pad the head index to the front of the transition indices."""
            return np.pad(arr, ((0,0),(1,0)), constant_values=h)

        match_match, match_insert, match_delete = [], [], []
        insert_insert, insert_match = [], []
        delete_delete, delete_match = [], []
        for h, idx in enumerate(transition_indices):
            # Must be triples, add head
            match_match.append(_pad_head(idx.match_to_match, h))
            match_insert.append(_pad_head(idx.match_to_insert, h))
            match_delete.append(_pad_head(idx.match_to_delete, h))
            insert_insert.append(_pad_head(idx.insert_to_insert, h))
            insert_match.append(_pad_head(idx.insert_to_match, h))
            delete_delete.append(_pad_head(idx.delete_to_delete, h))
            delete_match.append(_pad_head(idx.delete_to_match, h))

        self.match_out_transitions = np.concatenate(
            match_match + match_insert + match_delete
        )
        self.insert_out_transitions = np.concatenate(
            insert_match + insert_insert
        )
        self.delete_out_transitions = np.concatenate(
            delete_match + delete_delete
        )

        # Set up the sub-priors
        self.match_prior: TFDirichletPrior = load_dirichlet(
            "transition_match_dirichlet.weights", dim =3
        )
        self.insert_prior: TFDirichletPrior = load_dirichlet(
            "transition_insert_dirichlet.weights", dim =2
        )
        self.delete_prior: TFDirichletPrior = load_dirichlet(
            "transition_delete_dirichlet.weights", dim =2
        )

    class TransitionType(enum.Enum):
        MATCH = 1
        INSERT = 2
        DELETE = 3

    def compute_transition_prior(
        self, transition_matrix: T_TFTensor, type: TransitionType
    ) -> T_TFTensor:
        """Compute the prior score for a given transition type.

        Args:
            transition_matrix (Tensor):
                The transition matrix of shape (H, Q, Q).
            type (TransitionType): The type of transition.

        Returns:
            Tensor: The output tensor of shape (H), with prior scores per
                head, summed over the match states.
        """
        match type:
            case TFPHMMTransitionPrior.TransitionType.MATCH:
                indices = self.match_out_transitions
                prior = self.match_prior
                dim = 3
            case TFPHMMTransitionPrior.TransitionType.INSERT:
                indices = self.insert_out_transitions
                prior = self.insert_prior
                dim = 2
            case TFPHMMTransitionPrior.TransitionType.DELETE:
                indices = self.delete_out_transitions
                prior = self.delete_prior
                dim = 2
            case _:
                raise ValueError(f"Unknown transition type: {type}")

        assert len(indices) % dim == 0, \
            f"Indices shape must be reshapeable to (?, {dim}). Got shape: " \
            f"{indices.shape}"
        transitions = tf.gather_nd(transition_matrix, indices)
        transitions = tf.reshape(transitions, (dim, -1))
        transitions = tf.transpose(transitions) # (sum_L - num_heads, dim)

        # Normalize (might be necessary when a subset of out-transitions is used)
        transitions /= tf.reduce_sum(transitions, axis=-1, keepdims=True)

        # Apply the prior to the transitions
        scores = prior.log_dirichlet_pdf(transitions)
        scores = tf.squeeze(scores) # (sum_L - num_heads)

        # Sum over the heads (each with various number of match states)
        # Use unsorted_segment_sum instead of segment_sum for XLA compatibility
        segment_ids = tf.constant(
            np.repeat(np.arange(len(self.lengths)), [L-1 for L in self.lengths]),
            dtype=tf.int32
        )
        scores = tf.math.unsorted_segment_sum(
            scores,
            segment_ids,
            num_segments=len(self.lengths)
        )

        return scores

    def compute_flank_prior(
        self, transition_matrix: T_TFTensor
    ) -> T_TFTensor:
        """Compute the prior score for the flanking transitions.

        Args:
            transition_matrix (Tensor):
                The transition matrix of shape (H, Q, Q).
        Returns:
            Tensor: The output tensor of shape (H), with prior scores per
                head.
        """
        scores = []

        for h in range(len(self.lengths)):
            L = self.lengths[h]
            # State indices (unfolded model):
            # L (left flank) = 3*L - 1
            # C (unannotated) = 3*L + 2
            # R (right flank) = 3*L + 3
            # E (end) = 3*L + 1
            # T (terminal) = -1

            left_idx = 3*L - 1
            unannot_idx = 3*L + 2
            right_idx = 3*L + 3
            end_idx = 3*L + 1
            terminal_idx = -1

            # Extract transition probabilities
            left_flank_loop = transition_matrix[h, left_idx, left_idx]  # type: ignore
            unannotated_loop = transition_matrix[h, unannot_idx, unannot_idx] # type: ignore
            right_flank_loop = transition_matrix[h, right_idx, right_idx] # type: ignore
            end_to_right_flank = transition_matrix[h, end_idx, right_idx] # type: ignore

            # Exit probabilities (1 - loop probability)
            left_flank_exit = 1.0 - left_flank_loop
            unannotated_exit = 1.0 - unannotated_loop
            right_flank_exit = 1.0 - right_flank_loop

            # End state transitions
            end_to_unannotated = transition_matrix[h, end_idx, unannot_idx] # type: ignore
            end_to_terminal = transition_matrix[h, end_idx, terminal_idx] # type: ignore

            # Compute flank prior (without start distribution terms)
            a = self.prior_config.alpha_flank
            a_c = self.prior_config.alpha_flank_compl
            flank = (a - 1) * safe_log(unannotated_loop)
            flank += (a - 1) * safe_log(right_flank_loop)
            flank += (a - 1) * safe_log(left_flank_loop)
            flank += (a - 1) * safe_log(end_to_right_flank)
            flank += (a_c - 1) * safe_log(unannotated_exit)
            flank += (a_c - 1) * safe_log(right_flank_exit)
            flank += (a_c - 1) * safe_log(left_flank_exit)
            flank += (a_c - 1) * safe_log(end_to_unannotated + end_to_terminal)

            scores.append(flank)

        return tf.stack(scores)

    def compute_hit_prior(
        self, transition_matrix: T_TFTensor
    ) -> T_TFTensor:
        """Compute the prior score for single-hit probability.

        Args:
            transition_matrix (Tensor):
                The transition matrix of shape (H, Q, Q).
        Returns:
            Tensor: The output tensor of shape (H), with prior scores per
                head.
        """
        scores = []

        for h in range(len(self.lengths)):
            L = self.lengths[h]
            # State indices
            end_idx = 3*L + 1
            unannotated_idx = 3*L + 2
            right_idx = 3*L + 3
            terminal_idx = -1

            # Extract transition probabilities
            end_to_right_flank = transition_matrix[h, end_idx, right_idx]
            end_to_terminal = transition_matrix[h, end_idx, terminal_idx]
            end_to_unannotated = transition_matrix[h, end_idx, unannotated_idx]

            # Compute hit prior
            a = self.prior_config.alpha_single
            a_c = self.prior_config.alpha_single_compl
            hit = (a - 1) * safe_log(
                end_to_right_flank + end_to_terminal
            )
            hit += (a_c - 1) * safe_log(end_to_unannotated)

            scores.append(hit)

        return tf.stack(scores)

    def compute_global_prior(
        self, transition_matrix: T_TFTensor
    ) -> T_TFTensor:
        """Compute the prior score for uniform entry/exit.

        Args:
            transition_matrix (Tensor):
                The transition matrix of shape (H, Q, Q).
        Returns:
            Tensor: The output tensor of shape (H), with prior scores per
                head.
        """
        scores = []
        e = self.prior_config.epsilon

        for h in range(len(self.lengths)):
            L = self.lengths[h]
            # State indices
            begin_idx = 3*L

            # Extract begin_to_match and match_to_end probabilities
            begin_to_match = transition_matrix[h, begin_idx, :L]  # (L,)
            match_to_end = transition_matrix[h, :L, 3*L + 1]  # (L,)

            begin_to_delete_0 = transition_matrix[h, begin_idx, 2*L - 1]

            # Rescale begin_to_match to sum to 1
            div = tf.maximum(e, 1.0 - begin_to_delete_0)
            btm = begin_to_match / div

            # Compute entry-exit matrix
            enex = tf.expand_dims(btm, 1) * tf.expand_dims(match_to_end, 0)
            # Keep only upper triangular part (including diagonal)
            enex = tf.linalg.band_part(enex, 0, -1)

            log_enex = safe_log(tf.maximum(e, 1.0 - enex))
            log_enex_compl = safe_log(tf.maximum(e, enex))

            # Compute global prior over all profile entry-exit pairs
            glob = (self.prior_config.alpha_global - 1) * (
                tf.reduce_sum(log_enex) - log_enex[0, -1]
            )
            glob += (self.prior_config.alpha_global_compl - 1) * (
                tf.reduce_sum(log_enex_compl) - log_enex_compl[0, -1]
            )

            scores.append(glob)

        return tf.stack(scores)

    @override
    def matrix(self) -> T_TFTensor:
        """Not implemented for this prior."""
        raise NotImplementedError(
            "TFPHMMTransitionPrior doesn't have a matrix. See the sub priors'" \
            " matrices instead, (e.g. )match_prior, insert_prior, delete_prior)"
        )

    @override
    def call(
        self,
        transition_matrix: T_TFTensor
    ) -> T_TFTensor:
        """Calls the prior with the given transition_matrix.

        Args:
            transition_matrix (Tensor):
                The transition matrix of shape (H, Q, Q).

        Returns:
            Tensor: The output tensor of shape (H), with prior scores per
                head, summed over all prior components.
        """
        # Compute all prior components
        match_scores = self.compute_transition_prior(
            transition_matrix,
            TFPHMMTransitionPrior.TransitionType.MATCH
        )
        insert_scores = self.compute_transition_prior(
            transition_matrix,
            TFPHMMTransitionPrior.TransitionType.INSERT
        )
        delete_scores = self.compute_transition_prior(
            transition_matrix,
            TFPHMMTransitionPrior.TransitionType.DELETE
        )

        flank_scores = self.compute_flank_prior(transition_matrix)
        hit_scores = self.compute_hit_prior(transition_matrix)
        global_scores = self.compute_global_prior(transition_matrix)

        # Sum all log densities
        return (match_scores + insert_scores + delete_scores +
                flank_scores + hit_scores + global_scores)


class TFPHMMStartPrior(TFPrior):
    """A prior that scores the starting distribution of a profile HMM.

    This prior scores the probability of starting in the left flank state
    versus starting in the begin state.
    """

    def __init__(
        self,
        lengths: Sequence[int] | np.ndarray,
        prior_config: PHMMPriorConfig,
        **kwargs
    ) -> None:
        """
        Args:
            lengths (Sequence[int] | np.ndarray): The number of match states in each head
                of the pHMM.
            prior_config (PHMMPriorConfig): Prior configuration containing alpha parameters
                for the start distribution prior.
        """
        super().__init__(**kwargs)
        self.lengths = np.asarray(lengths)
        self.prior_config = prior_config

    @override
    def matrix(self) -> T_TFTensor:
        """Not implemented for this prior."""
        raise NotImplementedError(
            "TFPHMMStartPrior doesn't have a matrix."
        )

    @override
    def call(self, start_dist: T_TFTensor) -> T_TFTensor:
        """Calls the prior with the given start distribution.

        Args:
            start_dist (Tensor):
                The start distribution of shape (H, Q), where Q is the number
                of states (including padding).

        Returns:
            Tensor: The output tensor of shape (H), with prior scores per head.
        """
        scores = []

        for h in range(len(self.lengths)):
            L = self.lengths[h]
            # State indices (unfolded model):
            # L (left flank) = 3*L - 1
            # B (begin) = 3*L
            left_idx = 3 * L - 1

            # Extract the probability of starting in the left flank state
            flank_init_prob = start_dist[h, left_idx]  # type: ignore

            # Compute start prior using the same alpha parameters as flank prior
            a = self.prior_config.alpha_flank
            a_c = self.prior_config.alpha_flank_compl
            score = (a - 1) * safe_log(flank_init_prob)
            score += (a_c - 1) * safe_log(1.0 - flank_init_prob)

            scores.append(score)

        return tf.stack(scores)
