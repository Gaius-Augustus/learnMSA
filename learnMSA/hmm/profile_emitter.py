from collections.abc import Sequence
from typing import override, cast

import numpy as np
import tensorflow as tf
from hidten.hmm import HMMConfig as HidtenHMMConfig
from hidten.tf.emitter.categorical import (T_shapelike, T_TFTensor,
                                           TFCategoricalEmitter)
from hidten.tf.prior.dirichlet import TFDirichletPrior

from learnMSA.hmm.tf_util import load_weight_resource, make_dirichlet_model
from learnMSA.hmm.value_set import PHMMValueSet
from learnMSA.msa_hmm.SequenceDataset import SequenceDataset


class ProfileEmitter(TFCategoricalEmitter):
    """An emitter for a profile of sites in a protein and their amino acid
    distributions. Insertions are modeled by a shared background distribution.
    """
    def __init__(
        self,
        values: Sequence[PHMMValueSet],
        use_prior_aa_dist: bool = True,
        **kwargs
    ) -> None:
        """
        Args:
            values (Sequence[PHMMValueSet]): A sequence of value sets,
                one per head, with probabilities.
            hidten_hmm_config (HidtenHMMConfig): The configuration of the
                hidten HMM.
            use_prior_aa_dist (bool): Whether to use the amino acid prior
                distribution for initializing the emissions. If False, uniform
                distributions are used.
        """
        super().__init__(**kwargs)

        # The profile emitter has less states than the HMM since it does not
        # model all insertion states explicitly.
        # We create a special HMMConfig so HidTen will correctly construct the
        # matrix for the lower number of states.
        self.hmm_config = HidtenHMMConfig(states = [v.L+1 for v in values])
        self.lengths = [value_set.L for value_set in values]

        # Set up the Dirichlet prior
        model = make_dirichlet_model()
        load_weight_resource(model, "amino_acid_dirichlet.weights")
        self.prior: TFDirichletPrior = cast(TFDirichletPrior, model.layers[1])
        # Assign custom config for broadcasting
        self.prior.hmm_config = HidtenHMMConfig(states=[1])

        init_values = []
        if use_prior_aa_dist:
            # Initialization based on prior distribution
            prior_dist = self.prior.matrix().numpy().flatten()
            for value_set in values:
                init_values.append(np.tile(prior_dist, value_set.L).flatten())
                init_values.append(prior_dist)
        else:
            # Initialization based on provided value sets
            for value_set in values:
                init_values.append(value_set.match_emissions.flatten())
                init_values.append(value_set.insert_emissions)
        self.initializer = np.concatenate(init_values)

    def build(self, input_shape: T_shapelike | None = None) -> None:
        # Number of amino acids (including non-standard + X, but excluding gap)
        s = len(SequenceDataset.alphabet)-1
        if input_shape is not None:
            assert input_shape[-1] == s
        super().build((None, None, s))

    @override
    def call(
        self,
        emissions: T_TFTensor,
        use_padding: bool = True,
    ) -> T_TFTensor:
        assert emissions.shape[-1] == len(SequenceDataset.alphabet)-1,\
            "Input emissions must match the number of amino acids (excluding gap)."
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
        ML = max(self.lengths)
        for L in self.lengths:
            repeats.extend([1]*L)
            repeats.extend([L+3])  # repeat insertion
            repeats.extend([L-1]*int(L < ML))  # repeat any padding
            repeats.extend([1]*(ML-L-1))  # keep rest of padding
        emission_scores = tf.repeat(emission_scores, repeats, axis=-1)
        emission_scores = tf.reshape(emission_scores, (B, T, H, 2*Q+1))
        if use_padding:
            emission_scores = tf.pad(
                emission_scores,
                [[0, 0], [0, 0], [0, 0], [0, 1]],
                constant_values=1.0,
            )
        return emission_scores
