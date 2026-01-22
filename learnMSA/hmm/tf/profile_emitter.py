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
        **kwargs
    ) -> None:
        """
        Args:
            values (Sequence[PHMMValueSet]): A sequence of value sets,
                one per head, with probabilities.
            trainable_insertions (bool): Whether insertion emissions are
                trainable. Defaults to True.
        """
        super().__init__(**kwargs)

        self._lengths = np.array([value_set.L for value_set in values])
        self.trainable_insertions = trainable_insertions

        init_values = []
        # Initialization based on provided value sets
        for value_set in values:
            init_values.append(value_set.match_emissions.flatten())
            init_values.append(value_set.insert_emissions)
        self.initializer = np.concatenate(init_values)

    def build(self, input_shape: T_shapelike | None = None) -> None:
        if input_shape is None:
            # Number of amino acids (including non-standard + X,
            # but excluding gap)
            s = len(SequenceDataset._default_alphabet)-1
            input_shape = (None, None, s)
        else:
            s = input_shape[-1]

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

        return emission_scores

    @override
    def call(
        self,
        emissions: T_TFTensor,
        use_padding: bool = True,
    ) -> T_TFTensor:
        # Compute the emission scores for matches + single insertion
        emission_scores = super().call(emissions, use_padding=False)
        # emission_scores has the form
        # [[..., head 1, L1 x match + 1 x insert + (padding)]
        # [..., head 2, L2 x match + 1 x insert + (padding)]]
        # Expand to the full form
        # [[..., head 1, L1 x match + (L1+3) x insert + (padding)]
        # [..., head 2, L2 x match + (L2+3) x insert + (padding)]]
        B, T, H, Q = tf.unstack(tf.shape(emission_scores))
        emission_scores = tf.reshape(emission_scores, (B, T, H*Q))
        repeats = []
        ML = self.lengths.max()
        for L in self.lengths:
            repeats.extend([1]*L)
            repeats.extend([L+2])  # repeat insertion
            if L < ML:
                repeats.extend([ML-L+1])  # repeat any padding
                repeats.extend([1]*(ML-L-1))  # keep rest of padding
        emission_scores = tf.repeat(emission_scores, repeats, axis=-1)
        emission_scores = tf.reshape(emission_scores, (B, T, H, 2*Q))
        if use_padding:
            emission_scores = tf.pad(
                emission_scores,
                [[0, 0], [0, 0], [0, 0], [0, 1]],
                constant_values=1.0,
            )
        return emission_scores
