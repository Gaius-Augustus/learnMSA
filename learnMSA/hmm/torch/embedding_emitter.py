import sys
from collections.abc import Sequence

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import numpy as np
import torch
from hidten.torch.emitter.multivariate_normal import (T_shapelike,
                                                      T_TorchTensor,
                                                      TorchMVNormalEmitter,
                                                      mvn_log_prob)

from learnMSA.hmm.torch.util import (insertion_expansion_indices,
                                     sequence_mask)
from learnMSA.hmm.util.value_set_emb import PHMMEmbeddingValueSet


# guard against too large input embeddings
MAX_EMB_DIM = 128


class TorchEmbeddingEmitter(TorchMVNormalEmitter):
    """An emitter for continuous embedding vectors using a multivariate normal
    distribution and a multivariate normal prior.
    """

    head_subset: Sequence[int] | None = None
    """If set, only these heads are used in computations."""

    @property
    def lengths(self) -> np.ndarray:
        """The number of match states in each head of the pHMM."""
        if self.head_subset is not None:
            return self._lengths[self.head_subset]
        return self._lengths

    def __init__(
        self,
        values: Sequence[PHMMEmbeddingValueSet],
        trainable_insertions: bool = True,
        use_full_matmul: bool = False,
        **kwargs
    ) -> None:
        """
        Args:
            values: A sequence of value sets, one per head, with embedding
                parameters.
            trainable_insertions: Whether insertion emissions are trainable.
                Defaults to True.
            use_full_matmul: Whether to compute emission scores via a full
                matrix multiplication instead of copying insertion emissions.
        """
        super().__init__(**kwargs)

        self._lengths = np.array([value_set.L for value_set in values])
        self.trainable_insertions = trainable_insertions
        self.use_full_matmul = use_full_matmul
        assert len(values) > 0, "At least one value set must be provided."
        self._embedding_dim = values[0].match_expectations.shape[1]

        init_values = []
        # Initialization based on provided value sets
        for value_set in values:
            match_values = np.concatenate(
                [value_set.match_expectations, value_set.match_variance],
                axis=-1,
            )
            assert match_values.shape == \
                (value_set.L, self._embedding_dim * 2), \
                "Match values for each value set must have shape " \
                f"(L, embedding_dim * 2), but got {match_values.shape}."
            insert_values = np.concatenate(
                [value_set.insert_expectation, value_set.insert_variance],
                axis=-1,
            )
            assert insert_values.shape == (self._embedding_dim * 2,), \
                "Insert values for each value set must have shape " \
                f"(embedding_dim * 2,), but got {insert_values.shape}."
            init_values.append(match_values.flatten())
            init_values.append(insert_values.flatten())

        # The initializer is a flat array that is ordered as follows:
        # For each head (major) and state (minor) it contains
        # component means, components variances, mix coefficients
        # (in this order, omitting coefficients for single component).
        self.initializer = np.concatenate(init_values)

    def build(self, input_shape: T_shapelike | None = None) -> None:
        if input_shape is None:
            input_shape = (None, None, self._embedding_dim)
        self.input_dim = input_shape[-1]  # type: ignore
        self._check_input_dim()

        # Share all insertion emissions across positions
        # We need to provide an array with indices into the emitter's kernel
        # values, which is flat and sorted by head, states, emissions (major
        # to minor).
        i_sum = 0
        indices = []
        for L in self._lengths:  # use unrestricted lengths here
            # Nothing is shared for match states (L per head)
            share_match = np.arange(i_sum, i_sum + L * self.matrix_dim)
            i_sum += L * self.matrix_dim
            # Each head shares its insert state emissions (1 per head)
            share_insert = np.tile(
                share_match[-1] + 1 + np.arange(self.matrix_dim),
                reps=L + 2
            )
            i_sum += self.matrix_dim
            indices.extend([share_match, share_insert])
        self.share = np.concatenate(indices)

        super().build(input_shape)

    def _check_input_dim(self) -> None:
        """Check allowed embedding dimensions for the pHMM.
        """
        if self.input_dim > MAX_EMB_DIM:
            raise ValueError(
                f"The embeddings are {self.input_dim}-dimensional, which is "
                f"too wide for the pHMM to emit directly. High-dimensional "
                f"protein language model embeddings have to be reduced before "
                f"they reach the pHMM: pass --reduce_online to learn the "
                f"reduction jointly with the alignment, or supply embeddings "
                f"that were already reduced to --scoring_model_dim "
                f"({self._embedding_dim})."
            )
        if self.input_dim != self._embedding_dim:
            raise ValueError(
                f"The embeddings are {self.input_dim}-dimensional, but the "
                f"pHMM's embedding emitter was built for "
                f"{self._embedding_dim} dimensions. Set --scoring_model_dim "
                f"to {self.input_dim} to match the embeddings."
            )

    @override
    def matrix(self, sqrt_variance: bool = False) -> T_TorchTensor:
        matrix = super().matrix(sqrt_variance)
        if self.head_subset is not None:
            subset = torch.as_tensor(
                np.asarray(self.head_subset),
                dtype=torch.int64,
                device=matrix.device,
            )
            matrix = matrix.index_select(0, subset)
            max_states_subset = max(
                [self.hmm_config.states[h] for h in self.head_subset]
            )
            matrix = matrix[:, :max_states_subset, :]

        if not self.trainable_insertions:
            # Create mask for match states (1) vs insertion states (0)
            # self.lengths gives the number of match states per head
            mask = sequence_mask(
                self.lengths, matrix.shape[1], matrix.dtype, matrix.device
            )
            # Expand mask to cover the emission dimension
            mask = mask[:, :, None]
            # Apply mask: keep gradients for match states, stop for insertions
            matrix = mask * matrix + (1 - mask) * matrix.detach()

        return matrix

    def emission_scores(self, observations: T_TorchTensor) -> T_TorchTensor:
        if self.use_full_matmul:
            return super().emission_scores(observations)

        # Override to handle insertion state via copying instead of
        # explicit computations
        # Keep match states + single insertion state
        matrix = self.matrix(sqrt_variance=True)
        keep = self.lengths.max() + 1
        reduced_mean = self.mean(matrix)[:, :keep, :]
        reduced_sqrt_variance = self.sqrt_variance(matrix)[:, :keep, :]

        # Add Z dimension (unused, single mixture component)
        mean = reduced_mean.unsqueeze(2)
        sqrt_variance = reduced_sqrt_variance.unsqueeze(2)
        log_pdf = mvn_log_prob(observations, mean, sqrt_variance).squeeze(-1)

        emission_scores = torch.exp(
            log_pdf / (self.config.temperature * self.input_dim)
        )

        # Mask invalid positions in shorter heads
        emission_scores = emission_scores * sequence_mask(
            self.lengths + 1,
            emission_scores.shape[-1],
            emission_scores.dtype,
            emission_scores.device,
        )

        return emission_scores

    @override
    def forward(
        self,
        emissions: T_TorchTensor,
        use_padding: bool = True,
    ) -> T_TorchTensor:
        # Compute the emission scores for matches + single insertion
        emission_scores = super().forward(emissions, use_padding=False)

        if not self.use_full_matmul:
            # emission_scores has the form
            # [[..., head 1, L1 x match + 1 x insert + (padding)]
            # [..., head 2, L2 x match + 1 x insert + (padding)]]
            # Expand to the full form
            # [[..., head 1, L1 x match + (L1+3) x insert + (padding)]
            # [..., head 2, L2 x match + (L2+3) x insert + (padding)]]
            B, T, H, Q = emission_scores.shape
            emission_scores = emission_scores.reshape(B, T, H * Q)
            indices = torch.as_tensor(
                insertion_expansion_indices(self.lengths),
                device=emission_scores.device,
            )
            emission_scores = emission_scores.index_select(-1, indices)
            emission_scores = emission_scores.reshape(B, T, H, 2 * Q)

        if use_padding:
            emission_scores = torch.nn.functional.pad(
                emission_scores, (0, 1), value=1.0
            )

        return emission_scores
