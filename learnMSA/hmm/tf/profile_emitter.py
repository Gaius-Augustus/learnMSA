from collections.abc import Sequence
from typing import override

import numpy as np
import tensorflow as tf
from hidten.tf.emitter.categorical import (T_shapelike, T_TFTensor,
                                           TFCategoricalEmitter)

from learnMSA.hmm.util.value_set import PHMMValueSet
from learnMSA.util.sequence_dataset import SequenceDataset


class ProfileEmitter(TFCategoricalEmitter):
    """An emitter for a profile of sites in a protein and their amino acid
    distributions. Insertions are modeled by a shared background distribution.
    """
    head_subset : Sequence[int] | None = None
    """If set, only these heads are used in computations."""

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
            values (Sequence[PHMMValueSet]): A sequence of value sets,
                one per head, with probabilities.
            trainable_insertions (bool): Whether insertion emissions are
                trainable. Defaults to True.
            use_full_matmul (bool): Whether to compute emission scores via
                a full matrix multiplication instead of copying insertion
                emissions.
            temperature (float): Temperature applied as an exponent
                (1/temperature) to the emission scores. Defaults to 1.0.
        """
        super().__init__(**kwargs)
        self.temperature = temperature

        if len(values) == 0:
            raise ValueError("At least one value set must be provided.")

        self.alphabet_size = values[0].alphabet_size
        self._lengths = np.array([value_set.L for value_set in values])
        self.trainable_insertions = trainable_insertions
        self.use_full_matmul = use_full_matmul

        init_values = []
        # Initialization based on provided value sets
        for value_set in values:
            assert value_set.alphabet_size == self.alphabet_size,\
                "All value sets must have the same alphabet size."
            assert value_set.match_emissions.shape ==\
                (value_set.L, self.alphabet_size),\
                "Match emissions for each value set must have shape "\
                f"(L, alphabet_size), but got {value_set.match_emissions.shape}."
            init_values.append(value_set.match_emissions.flatten())
            init_values.append(value_set.insert_emissions)
        self.initializer = np.concatenate(init_values)

    def build(self, input_shape: T_shapelike | None = None) -> None:
        s = self.alphabet_size
        if input_shape is None:
            input_shape = (None, None, s)
        else:
            assert input_shape[-1] == s,\
                "Input feature dimension must match alphabet size provided "\
                f"via the ValueSets ({input_shape[-1]} != {s})."

        # Share all insertion emissions across positions
        # We need to provide an array with indices into the emitter's kernel
        # values, which is flat and sorted by head, states, emissions (major
        # to minor).
        i_sum = 0
        indices = []
        for L in self._lengths: # use unrestricted lengths here
            # Nothing is shared for match states (L per head)
            share_match = np.arange(i_sum, i_sum + L*s)
            i_sum += L*s
            # Each head shares its insert state emissions (1 per head)
            share_insert = np.tile(
                share_match[-1] + 1 + np.arange(s),
                reps=L + 2
            )
            i_sum += s
            indices.extend([share_match, share_insert])
        self.share = np.concatenate(indices)

        super().build(input_shape)

    @override
    def matrix(self) -> T_TFTensor:
        matrix = super().matrix()
        if self.head_subset is not None:
            matrix = tf.gather(matrix, self.head_subset, axis=0)
            max_states_subset = max(
                [self.hmm_config.states[h] for h in self.head_subset]
            )
            # Keep only relevant states
            matrix = matrix[:, :max_states_subset, :]

        if not self.trainable_insertions:
            # Create mask for match states (True) vs insertion states (False)
            # self.lengths gives the number of match states per head
            max_states = tf.shape(matrix)[1]
            mask = tf.sequence_mask(
                self.lengths, maxlen=max_states, dtype=matrix.dtype
            )
            # Expand mask to cover the emission dimension
            mask = mask[:, :, tf.newaxis]
            # Apply mask: keep gradients for match states, stop for insertions
            matrix = mask * matrix + (1 - mask) * tf.stop_gradient(matrix)

        return matrix

    def emission_scores(self, observations: T_TFTensor) -> T_TFTensor:
        if self.use_full_matmul:
            emission_scores = super().emission_scores(observations)
        else:
            # Override to handle insertion state via copying instead of
            # explicit computations
            # Keep match states + single insertion state
            reduced_matrix = self.matrix()[:, :self.lengths.max()+1, :]
            if observations.ndim == 3:
                emission_scores = tf.einsum(
                    "btd,hqd->bthq", observations, reduced_matrix
                )
            else:
                emission_scores = tf.einsum(
                    "bthd,hqd->bthq", observations, reduced_matrix
                )

            # Mask invalid positions in shorter heads
            emission_scores *= tf.sequence_mask(
                self.lengths+1, dtype=emission_scores.dtype
            )

        if self.temperature != 1.0:
            # Clip to a small epsilon before raising to a fractional power.
            # Clipping to 0 would cause the gradient p*x^(p-1) -> +Inf at x=0
            # when temperature > 1 (p = 1/temperature < 1). Masked positions
            # with near-zero scores are irrelevant to the loss anyway.
            emission_scores = tf.pow(
                tf.maximum(emission_scores, 1e-12), 1.0 / self.temperature
            )

        return emission_scores

    @override
    def call(
        self,
        emissions: T_TFTensor,
        use_padding: bool = True,
    ) -> T_TFTensor:
        # Compute the emission scores for matches + single insertion
        emission_scores = super().call(emissions, use_padding=False)

        if not self.use_full_matmul:
            # emission_scores has the form
            # [[..., head 1, L1 x match + 1 x insert + (padding)]
            # [..., head 2, L2 x match + 1 x insert + (padding)]]
            # Expand to the full form
            # [[..., head 1, L1 x match + (L1+3) x insert + (padding)]
            # [..., head 2, L2 x match + (L2+3) x insert + (padding)]]
            B, T, H, Q = tf.unstack(tf.shape(emission_scores))
            emission_scores = tf.reshape(emission_scores, (B, T, H*Q))

            # Build gather indices for XLA compatibility (instead of tf.repeat)
            indices = []
            ML = self.lengths.max()
            offset = 0
            for L in self.lengths:
                # Match states: copy once each
                indices.extend(list(range(offset, offset + L)))
                # Insertion state: repeat L+2 times
                indices.extend([offset + L] * (L + 2))
                offset += L + 1  # Move to next head (L matches + 1 insert)
                # Padding states
                if L < ML:
                    indices.extend([offset] * (ML - L + 1))  # repeat first padding
                    indices.extend(list(range(offset + 1, offset + ML - L)))  # rest of padding
                    offset += ML - L

            # Use tf.gather instead of tf.repeat for XLA compatibility
            indices_tensor = tf.constant(indices, dtype=tf.int32)
            emission_scores = tf.gather(emission_scores, indices_tensor, axis=-1)
            emission_scores = tf.reshape(emission_scores, (B, T, H, 2*Q))

        if use_padding:
            emission_scores = tf.pad(
                emission_scores,
                [[0, 0], [0, 0], [0, 0], [0, 1]],
                constant_values=1.0,
            )

        return emission_scores
