import enum
import sys
from typing import Sequence

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import numpy as np
import torch
from hidten.torch.prior.dirichlet import (T_TorchTensor, TorchDirichletPrior,
                                          TorchPrior)
from hidten.torch.util import safe_log, safe_norm

from learnMSA.config.hmm import PHMMPriorConfig
from learnMSA.hmm.torch.util import load_dirichlet
from learnMSA.hmm.util.transition_index_set import PHMMTransitionIndexSet


class TorchPHMMTransitionPrior(TorchPrior):
    """A prior that uses Dirichlet distributions to score the transition
    probabilities of a profile HMM. Uses sub-priors for match, insert, and
    delete transitions.
    """

    def __init__(
        self,
        lengths: Sequence[int] | np.ndarray,
        prior_config: PHMMPriorConfig,
        **kwargs
    ) -> None:
        """
        Args:
            lengths: The number of match states in each head of the pHMM.
            prior_config: Prior configuration containing alpha parameters for
                transition priors.
        """
        super().__init__(**kwargs)
        self.lengths = np.asarray(lengths)
        self.prior_config = prior_config
        transition_indices = [PHMMTransitionIndexSet(L=L) for L in lengths]

        def _pad_head(arr: np.ndarray, h: int) -> np.ndarray:
            """Pad the head index to the front of the transition indices."""
            return np.pad(arr, ((0, 0), (1, 0)), constant_values=h)

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
        self.match_prior: TorchDirichletPrior = load_dirichlet(
            "transition_match_dirichlet.weights", dim=3
        )
        self.insert_prior: TorchDirichletPrior = load_dirichlet(
            "transition_insert_dirichlet.weights", dim=2
        )
        self.delete_prior: TorchDirichletPrior = load_dirichlet(
            "transition_delete_dirichlet.weights", dim=2
        )

    class TransitionType(enum.Enum):
        MATCH = 1
        INSERT = 2
        DELETE = 3

    def compute_transition_prior(
        self, transition_matrix: T_TorchTensor, type: TransitionType
    ) -> T_TorchTensor:
        """Compute the prior score for a given transition type.

        Args:
            transition_matrix: The transition matrix of shape (H, Q, Q).
            type: The type of transition.

        Returns:
            The output tensor of shape (H), with prior scores per head, summed
            over the match states.
        """
        match type:
            case TorchPHMMTransitionPrior.TransitionType.MATCH:
                indices = self.match_out_transitions
                prior = self.match_prior
                dim = 3
            case TorchPHMMTransitionPrior.TransitionType.INSERT:
                indices = self.insert_out_transitions
                prior = self.insert_prior
                dim = 2
            case TorchPHMMTransitionPrior.TransitionType.DELETE:
                indices = self.delete_out_transitions
                prior = self.delete_prior
                dim = 2
            case _:
                raise ValueError(f"Unknown transition type: {type}")

        assert len(indices) % dim == 0, \
            f"Indices shape must be reshapeable to (?, {dim}). Got shape: " \
            f"{indices.shape}"
        transitions = transition_matrix[
            indices[:, 0], indices[:, 1], indices[:, 2]
        ]
        transitions = transitions.reshape(dim, -1)
        transitions = transitions.T  # (sum_L - num_heads, dim)

        # Normalize (might be necessary when a subset of out-transitions is
        # used). safe_norm guards the 0/0 that occurs when a triple underflows
        # to zero.
        transitions = safe_norm(transitions)

        # Apply the prior to the transitions
        scores = prior.log_dirichlet_pdf(transitions)
        scores = scores.squeeze()  # (sum_L - num_heads)

        # Sum over the heads (each with various number of match states)
        segment_ids = torch.as_tensor(
            np.repeat(
                np.arange(len(self.lengths)), [L - 1 for L in self.lengths]
            ),
            dtype=torch.int64,
            device=scores.device,
        )
        scores = torch.zeros(
            len(self.lengths), dtype=scores.dtype, device=scores.device
        ).index_add(0, segment_ids, scores)

        return scores

    def compute_flank_prior(
        self, transition_matrix: T_TorchTensor
    ) -> T_TorchTensor:
        """Compute the prior score for the flanking transitions.

        Args:
            transition_matrix: The transition matrix of shape (H, Q, Q).

        Returns:
            The output tensor of shape (H), with prior scores per head.
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

            left_idx = 3 * L - 1
            unannot_idx = 3 * L + 2
            right_idx = 3 * L + 3
            end_idx = 3 * L + 1
            terminal_idx = -1

            # Extract transition probabilities
            left_flank_loop = transition_matrix[h, left_idx, left_idx]
            unannotated_loop = transition_matrix[h, unannot_idx, unannot_idx]
            right_flank_loop = transition_matrix[h, right_idx, right_idx]
            end_to_right_flank = transition_matrix[h, end_idx, right_idx]

            # Exit probabilities (1 - loop probability)
            left_flank_exit = 1.0 - left_flank_loop
            unannotated_exit = 1.0 - unannotated_loop
            right_flank_exit = 1.0 - right_flank_loop

            # End state transitions
            end_to_unannotated = transition_matrix[h, end_idx, unannot_idx]
            end_to_terminal = transition_matrix[h, end_idx, terminal_idx]

            # Compute flank prior (without start distribution terms)
            a = self.prior_config.alpha_flank
            a_c = self.prior_config.alpha_flank_compl
            flank = (a - 1) * safe_log(unannotated_loop)
            flank = flank + (a - 1) * safe_log(right_flank_loop)
            flank = flank + (a - 1) * safe_log(left_flank_loop)
            flank = flank + (a - 1) * safe_log(end_to_right_flank)
            flank = flank + (a_c - 1) * safe_log(unannotated_exit)
            flank = flank + (a_c - 1) * safe_log(right_flank_exit)
            flank = flank + (a_c - 1) * safe_log(left_flank_exit)
            flank = flank + (a_c - 1) * safe_log(
                end_to_unannotated + end_to_terminal
            )

            scores.append(flank)

        return torch.stack(scores)

    def compute_hit_prior(
        self, transition_matrix: T_TorchTensor
    ) -> T_TorchTensor:
        """Compute the prior score for single-hit probability.

        Args:
            transition_matrix: The transition matrix of shape (H, Q, Q).

        Returns:
            The output tensor of shape (H), with prior scores per head.
        """
        scores = []

        for h in range(len(self.lengths)):
            L = self.lengths[h]
            # State indices
            end_idx = 3 * L + 1
            unannotated_idx = 3 * L + 2
            right_idx = 3 * L + 3
            terminal_idx = -1

            # Extract transition probabilities
            end_to_right_flank = transition_matrix[h, end_idx, right_idx]
            end_to_terminal = transition_matrix[h, end_idx, terminal_idx]
            end_to_unannotated = transition_matrix[
                h, end_idx, unannotated_idx
            ]

            # Compute hit prior
            a = self.prior_config.alpha_single
            a_c = self.prior_config.alpha_single_compl
            hit = (a - 1) * safe_log(end_to_right_flank + end_to_terminal)
            hit = hit + (a_c - 1) * safe_log(end_to_unannotated)

            scores.append(hit)

        return torch.stack(scores)

    def compute_global_prior(
        self, transition_matrix: T_TorchTensor
    ) -> T_TorchTensor:
        """Compute the prior score for uniform entry/exit.

        Args:
            transition_matrix: The transition matrix of shape (H, Q, Q).

        Returns:
            The output tensor of shape (H), with prior scores per head.
        """
        scores = []
        e = self.prior_config.epsilon

        for h in range(len(self.lengths)):
            L = self.lengths[h]
            # State indices
            begin_idx = 3 * L

            # Extract begin_to_match and match_to_end probabilities
            begin_to_match = transition_matrix[h, begin_idx, :L]  # (L,)
            match_to_end = transition_matrix[h, :L, 3 * L + 1]  # (L,)

            begin_to_delete_0 = transition_matrix[h, begin_idx, 2 * L - 1]

            # Rescale begin_to_match to sum to 1
            div = (1.0 - begin_to_delete_0).clamp_min(e)
            btm = begin_to_match / div

            # Compute entry-exit matrix
            enex = btm.unsqueeze(1) * match_to_end.unsqueeze(0)
            # Keep only upper triangular part (including diagonal)
            enex = torch.triu(enex)

            log_enex = safe_log((1.0 - enex).clamp_min(e))
            log_enex_compl = safe_log(enex.clamp_min(e))

            # Compute global prior over all profile entry-exit pairs
            glob = (self.prior_config.alpha_global - 1) * (
                log_enex.sum() - log_enex[0, -1]
            )
            glob = glob + (self.prior_config.alpha_global_compl - 1) * (
                log_enex_compl.sum() - log_enex_compl[0, -1]
            )

            scores.append(glob)

        return torch.stack(scores)

    @override
    def matrix(self) -> T_TorchTensor:
        """Not implemented for this prior."""
        raise NotImplementedError(
            "TorchPHMMTransitionPrior doesn't have a matrix. See the sub "
            "priors' matrices instead (e.g. match_prior, insert_prior, "
            "delete_prior)."
        )

    @override
    def forward(self, transition_matrix: T_TorchTensor) -> T_TorchTensor:
        """Calls the prior with the given transition_matrix.

        Args:
            transition_matrix: The transition matrix of shape (H, Q, Q).

        Returns:
            The output tensor of shape (H), with prior scores per head, summed
            over all prior components.
        """
        # Compute all prior components
        match_scores = self.compute_transition_prior(
            transition_matrix,
            TorchPHMMTransitionPrior.TransitionType.MATCH
        )
        insert_scores = self.compute_transition_prior(
            transition_matrix,
            TorchPHMMTransitionPrior.TransitionType.INSERT
        )
        delete_scores = self.compute_transition_prior(
            transition_matrix,
            TorchPHMMTransitionPrior.TransitionType.DELETE
        )

        flank_scores = self.compute_flank_prior(transition_matrix)
        hit_scores = self.compute_hit_prior(transition_matrix)
        global_scores = self.compute_global_prior(transition_matrix)

        # Sum all log densities
        return (match_scores + insert_scores + delete_scores +
                flank_scores + hit_scores + global_scores)


class TorchPHMMStartPrior(TorchPrior):
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
            lengths: The number of match states in each head of the pHMM.
            prior_config: Prior configuration containing alpha parameters for
                the start distribution prior.
        """
        super().__init__(**kwargs)
        self.lengths = np.asarray(lengths)
        self.prior_config = prior_config

    @override
    def matrix(self) -> T_TorchTensor:
        """Not implemented for this prior."""
        raise NotImplementedError(
            "TorchPHMMStartPrior doesn't have a matrix."
        )

    @override
    def forward(self, start_dist: T_TorchTensor) -> T_TorchTensor:
        """Calls the prior with the given start distribution.

        Args:
            start_dist: The start distribution of shape (H, Q), where Q is the
                number of states (including padding).

        Returns:
            The output tensor of shape (H), with prior scores per head.
        """
        scores = []

        for h in range(len(self.lengths)):
            L = self.lengths[h]
            # State indices (unfolded model):
            # L (left flank) = 3*L - 1
            # B (begin) = 3*L
            left_idx = 3 * L - 1

            # Extract the probability of starting in the left flank state
            flank_init_prob = start_dist[h, left_idx]

            # Compute start prior using the same alphas as the flank prior
            a = self.prior_config.alpha_flank
            a_c = self.prior_config.alpha_flank_compl
            score = (a - 1) * safe_log(flank_init_prob)
            score = score + (a_c - 1) * safe_log(1.0 - flank_init_prob)

            scores.append(score)

        return torch.stack(scores)
