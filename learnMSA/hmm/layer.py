from collections.abc import Sequence

import numpy as np
import tensorflow as tf
from hidten import HMMMode
from hidten.tf.emitter import TFPaddingEmitter
from hidten.tf.hmm import TFHMM, T_shapelike

from learnMSA.config import PHMMConfig, PHMMPriorConfig, LanguageModelConfig
from learnMSA.hmm.profile_emitter import ProfileEmitter
from learnMSA.hmm.embedding_emitter import EmbeddingEmitter
from learnMSA.hmm.transitioner import PHMMTransitioner
from learnMSA.hmm.value_set import PHMMValueSet
from learnMSA.hmm.prior import TFPHMMTransitionPrior
from learnMSA.hmm.tf_util import load_dirichlet
from learnMSA.msa_hmm.SequenceDataset import SequenceDataset


class PHMMLayer(tf.keras.Layer):

    lengths: Sequence[int]
    """The number of match states in each head of the pHMM.
    """

    @property
    def heads(self) -> int:
        """The number of pHMM heads."""
        return len(self.lengths)

    @property
    def states(self) -> Sequence[int]:
        """The total number of states in each head. Without terminal state."""
        return [2*L+2 for L in self.lengths]

    @property
    def states_explicit(self) -> Sequence[int]:
        """The total number of states in each head including silent states
        such as deletes.
        """
        return [3*L+5 for L in self.lengths]

    _mode: HMMMode = HMMMode.LIKELIHOOD_LOG
    """Determines the return value of the layer."""

    def __init__(
        self,
        lengths: Sequence[int],
        config : PHMMConfig,
        prior_config: PHMMPriorConfig | None = None,
        plm_config: LanguageModelConfig | None = None,
        **kwargs
    ) -> None:
        """
        Args:
            lengths: The number of match states in each head of the pHMM.
            config: HMM configuration parameters.
        """
        super().__init__(**kwargs)
        self.lengths = lengths
        self.config = config
        if prior_config is None:
            prior_config = PHMMPriorConfig()
        self.prior_config = prior_config
        self.plm_config = plm_config

        values = [
            PHMMValueSet.from_config(L, h, config)
            for h, L in enumerate(lengths)
        ]

        # Set up the Dirichlet prior for emissions
        emission_prior = load_dirichlet(
            "amino_acid_dirichlet.weights",
            dim = len(SequenceDataset.alphabet)-1
        )
        # Share concentrations across all states
        emission_prior.share = np.tile(
            np.arange(len(SequenceDataset.alphabet)-1),
            reps=2 * sum(lengths) + 2 * len(lengths)
        )

        # Override emission values with prior distribution if requested
        if config.use_prior_for_emission_init:
            values = self._override_emissions_with_prior(values, emission_prior)

        # Create the HMM, with 2*L+2 states per head
        self.hmm = TFHMM(states=self.states, heads=self.heads)

        # Add the transitioner and prior
        self.hmm.transitioner = PHMMTransitioner(
            values = values
        )
        self.hmm.transitioner.prior = TFPHMMTransitionPrior(
            lengths, prior_config
        )

        # Add the profile emitter
        profile_emitter = ProfileEmitter(values=values)
        self.hmm.add_emitter(profile_emitter)
        profile_emitter.prior = emission_prior

        # Add the MVN emitter
        if self.plm_config is not None:
            self.hmm.add_emitter(EmbeddingEmitter())

        # Add the padding emitter
        self.hmm.add_emitter(TFPaddingEmitter())

    def loglik_mode(self) -> None:
        """Makes the layer return log-likelihoods.
        """
        self._mode = HMMMode.LIKELIHOOD_LOG

    def viterbi_mode(self) -> None:
        """Makes the layer return Viterbi paths.
        """
        self._mode = HMMMode.VITERBI

    def posterior_mode(self) -> None:
        """Makes the layer return state posterior probabilities.
        """
        self._mode = HMMMode.POSTERIOR

    def build(self, input_shape: T_shapelike) -> None:
        self.hmm.build(input_shape)

    def call(self, x: tf.Tensor, padding: tf.Tensor) -> tf.Tensor:
        return self.hmm(x, padding, mode=self._mode)

    def prior_scores(self) -> tf.Tensor:
        """Calculates the prior scores for all parameters in the pHMM.

        Returns:
            Tensor: The prior scores of shape ``(H,)``, where ``H`` is the
                number of heads in the pHMM.
        """
        return self.hmm.prior_scores()

    @staticmethod
    def _override_emissions_with_prior(
        values: Sequence[PHMMValueSet],
        prior
    ) -> list[PHMMValueSet]:
        """Override emission values in value sets with prior distribution.

        Args:
            values: The original value sets.
            prior: The Dirichlet prior for emissions.

        Returns:
            New value sets with emissions replaced by prior distribution.
        """
        prior_dist = prior.matrix().numpy().flatten()
        updated_values = []
        for value_set in values:
            updated_values.append(PHMMValueSet(
                L=value_set.L,
                match_emissions=np.tile(prior_dist, (value_set.L, 1)),
                insert_emissions=prior_dist,
                transitions=value_set.transitions,
                start=value_set.start,
            ))
        return updated_values
