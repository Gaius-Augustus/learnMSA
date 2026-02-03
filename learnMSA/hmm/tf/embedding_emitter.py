from collections.abc import Sequence
from typing import override

import numpy as np
import tensorflow as tf
from hidten.tf.emitter.multivariate_normal import (T_shapelike, T_TFTensor,
                                                   TFMVNormalEmitter)

from learnMSA.hmm.util.value_set_emb import PHMMEmbeddingValueSet
from learnMSA.util.sequence_dataset import SequenceDataset


class EmbeddingEmitter(TFMVNormalEmitter):
    """An emitter for continuous embedding vectors using a multivariate normal
    distribution and a multivariate normal prior.
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
        values: Sequence[PHMMEmbeddingValueSet],
        trainable_insertions: bool = True,
        **kwargs
    ) -> None:
        """
        Args:
            values (Sequence[PHMMEmbeddingValueSet]): A sequence of value sets,
                one per head, with embedding parameters.
            trainable_insertions (bool): Whether insertion emissions are
                trainable. Defaults to True.
        """
        super().__init__(**kwargs)

        self._lengths = np.array([value_set.L for value_set in values])
        self.trainable_insertions = trainable_insertions

        init_values = []
        # Initialization based on provided value sets
        for value_set in values:
            match_values = np.concatenate(
                [value_set.match_expectations, value_set.match_stddev],
                axis=-1,
            )
            insert_values = np.concatenate(
                [value_set.insert_expectation, value_set.insert_stddev],
                axis=-1,
            )
            init_values.append(match_values.flatten())
            init_values.append(insert_values.flatten())

        # The initializer is a flat array that is ordered as follows:
        # For each head (major) and state (minor) it contains
        # component means, components standard deviations, mix coefficients
        # (in this order, omitting coefficients for single component).
        self.initializer = np.concatenate(init_values)

    def build(self, input_shape: T_shapelike | None = None) -> None:
        if input_shape is None:
            # Number of amino acids
            # (including non-standard + X, but excluding gap)
            s = len(SequenceDataset._default_alphabet)
            input_shape = (None, None, s-1)
        super().build(input_shape)

    @override
    def matrix(self) -> T_TFTensor:
        matrix = super().matrix()
        if self.head_subset is not None:
            matrix = tf.gather(matrix, self.head_subset, axis=0)
            max_len_subset = max(self.lengths[h] for h in self.head_subset)
            matrix = matrix[:, :max_len_subset*2, :]

        # TODO: to be tested..
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
        if self.head_subset is not None:
            lengths = [self.lengths[h] for h in self.head_subset]
        else:
            lengths = self.lengths
        ML = max(lengths)
        for L in lengths:
            repeats.extend([1]*L)
            repeats.extend([L+2])  # repeat insertion
            repeats.extend([L-1]*int(L < ML))  # repeat any padding
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
