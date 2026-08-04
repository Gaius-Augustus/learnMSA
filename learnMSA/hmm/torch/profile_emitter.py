import sys
from collections.abc import Sequence

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import numpy as np
import torch
from hidten.torch.emitter.categorical import (T_shapelike, T_TorchTensor,
                                              TorchCategoricalEmitter)

from learnMSA.hmm.torch.util import (insertion_expansion_indices,
                                     sequence_mask)
from learnMSA.hmm.util.value_set import PHMMValueSet


class TorchProfileEmitter(TorchCategoricalEmitter):
    """An emitter for a profile of sites in a protein and their amino acid
    distributions. Insertions are modeled by a shared background distribution.
    """

    head_subset: Sequence[int] | None = None
    """If set, only these heads are used in computations."""

    extra_dims: int = 0
    """Number of trailing emission columns that are folded uniformly into
    the leading columns before the (lower-dimensional) Dirichlet prior is
    applied. 0 disables the projection (the common case)."""

    @property
    def lengths(self) -> np.ndarray:
        """The number of match states in each head of the pHMM."""
        if self.head_subset is not None:
            return self._lengths[self.head_subset]
        return self._lengths

    def __init__(
        self,
        values: Sequence[PHMMValueSet],
        trainable_insertions: bool = True,
        use_full_matmul: bool = True,
        temperature: float = 1.0,
        **kwargs
    ) -> None:
        """
        Args:
            values: A sequence of value sets, one per head, with
                probabilities.
            trainable_insertions: Whether insertion emissions are trainable.
                Defaults to True.
            use_full_matmul: Whether to compute emission scores via a full
                matrix multiplication instead of copying insertion emissions.
            temperature: Temperature applied as an exponent (1/temperature)
                to the emission scores. Defaults to 1.0.
        """
        super().__init__(**kwargs)
        self.temperature = temperature

        if len(values) == 0:
            raise ValueError("At least one value set must be provided.")

        self.matrix_dim = values[0].alphabet_size
        self._lengths = np.array([value_set.L for value_set in values])
        self.trainable_insertions = trainable_insertions
        self.use_full_matmul = use_full_matmul

        init_values = []
        # Initialization based on provided value sets
        for value_set in values:
            assert value_set.alphabet_size == self.matrix_dim, \
                "All value sets must have the same alphabet size."
            assert value_set.match_emissions.shape == \
                (value_set.L, self.matrix_dim), \
                "Match emissions for each value set must have shape " \
                f"(L, matrix_dim), but got {value_set.match_emissions.shape}."
            init_values.append(value_set.match_emissions.flatten())
            init_values.append(value_set.insert_emissions)
        self.initializer = np.concatenate(init_values)

    def build(self, input_shape: T_shapelike | None = None) -> None:
        s = self.matrix_dim
        if input_shape is None:
            input_shape = (None, None, s)
        else:
            assert input_shape[-1] == s, \
                "Input feature dimension must match alphabet size provided " \
                f"via the ValueSets ({input_shape[-1]} != {s})."

        self.share = self._build_share()

        super().build(input_shape)

    @override
    def matrix(self) -> T_TorchTensor:
        matrix = super().matrix()
        matrix = self._prepare_matrix(matrix)
        matrix = self._anneal_matrix(matrix)
        return matrix

    def emission_scores(self, observations: T_TorchTensor) -> T_TorchTensor:
        if self.use_full_matmul:
            emission_scores = super().emission_scores(observations)
        else:
            # Override to handle insertion state via copying instead of
            # explicit computations
            # Keep match states + single insertion state
            reduced_matrix = self.matrix()[:, :self.lengths.max() + 1, :]
            if observations.ndim == 3:
                emission_scores = torch.einsum(
                    "btd,hqd->bthq", observations, reduced_matrix
                )
            else:
                emission_scores = torch.einsum(
                    "bthd,hqd->bthq", observations, reduced_matrix
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

    def prior_scores(self) -> T_TorchTensor:
        """Prior scores for the emission matrix. When ``extra_dims`` > 0 the
        emission matrix has extra U/O columns that are folded uniformly into
        the leading (standard) columns before applying the lower-dimensional
        Dirichlet prior (treating U/O like X).
        """
        if not hasattr(self, "_prior"):
            return 0.0  # type: ignore[return-value]
        matrix = self.matrix()
        if self.extra_dims > 0:
            d = self.matrix_dim - self.extra_dims
            fold = matrix[..., d:].sum(dim=-1, keepdim=True) / d
            matrix = matrix[..., :d] + fold
        return self._prior(matrix)

    def _build_share(self) -> np.ndarray:
        # Share all insertion emissions across positions
        # We need to provide an array with indices into the emitter's kernel
        # values, which is flat and sorted by head, states, emissions (major
        # to minor).
        s = self.matrix_dim
        i_sum = 0
        indices = []
        for L in self._lengths:  # use unrestricted lengths here
            # Nothing is shared for match states (L per head)
            share_match = np.arange(i_sum, i_sum + L * s)
            i_sum += L * s
            # Each head shares its insert state emissions (1 per head)
            share_insert = np.tile(
                share_match[-1] + 1 + np.arange(s),
                reps=L + 2
            )
            i_sum += s
            indices.extend([share_match, share_insert])
        return np.concatenate(indices)

    def _prepare_matrix(self, matrix: T_TorchTensor) -> T_TorchTensor:
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
            # Keep only relevant states
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

    def _anneal_matrix(self, matrix: T_TorchTensor) -> T_TorchTensor:
        if self.temperature != 1.0:
            nonzero_mask = matrix.sum(dim=-1, keepdim=True) > 0
            log_matrix = torch.log(matrix.clamp_min(1e-16))
            matrix = torch.softmax(log_matrix / self.temperature, dim=-1)
            matrix = torch.where(
                nonzero_mask, matrix, torch.zeros_like(matrix)
            )
        return matrix
