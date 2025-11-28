import enum
from typing import Sequence, override

import numpy as np
import tensorflow as tf
from hidten.tf.prior.dirichlet import T_TFTensor, TFDirichletPrior, TFPrior

from learnMSA.hmm.tf_util import load_dirichlet
from learnMSA.hmm.transition_index_set import PHMMTransitionIndexSet


class TFPHMMTransitionPrior(TFPrior):
    """ A prior that uses Dirichlet distributions to score the transition
    probabilities of a profile HMM. Uses sub-priors for match, insert, and delete
    transitions.
    """

    def __init__(self, lengths: Sequence[int], **kwargs) -> None:
        """
        Args:
            lengths (Sequence[int]): The number of match states in each head
                of the pHMM.
        """
        super().__init__(**kwargs)
        self.lengths = lengths
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

        # Apply the prior to the transitions
        scores = prior.log_dirichlet_pdf(transitions)
        scores = tf.squeeze(scores) # (sum_L - num_heads)

        # Sum over the heads (each with various number of match states)
        scores = tf.math.segment_sum(
            scores,
            np.repeat(np.arange(len(self.lengths)), [L-1 for L in self.lengths])
        )

        return scores

    @override
    def matrix(self) -> T_TFTensor:
        """Not implemented for this prior."""
        raise NotImplementedError(
            "TFPHMMTransitionPrior doesn't have a matrix. See the sub priors'" \
            " matrices instead, (e.g. )match_prior, insert_prior, delete_prior)"
        )

    @override
    def call(self, transition_matrix: T_TFTensor) -> T_TFTensor:
        """Calls the prior with the given transition_matrix.

        Args:
            transition_matrix (Tensor):
                The transition matrix of shape (H, Q, Q).

        Returns:
            Tensor: The output tensor of shape (H), with prior scores per
                head, summed over the match states.
        """
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
        # Sum the log densities
        return match_scores + insert_scores + delete_scores
