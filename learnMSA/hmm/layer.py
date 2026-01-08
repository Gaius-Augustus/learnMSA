from collections.abc import Sequence

import numpy as np
import tensorflow as tf
from hidten import HMMMode
from hidten.tf.emitter import TFPaddingEmitter
from hidten.tf.hmm import TFHMM, T_shapelike
from hidten.tf.prior import TFCombinedPrior, TFInverseGammaPrior

from learnMSA.config import PHMMConfig, PHMMPriorConfig, LanguageModelConfig
from learnMSA.hmm.profile_emitter import ProfileEmitter
from learnMSA.hmm.embedding_emitter import EmbeddingEmitter
from learnMSA.hmm.transitioner import PHMMTransitioner
from learnMSA.hmm.value_set import PHMMValueSet
from learnMSA.hmm.value_set_emb import PHMMEmbeddingValueSet
from learnMSA.hmm.prior import TFPHMMTransitionPrior
from learnMSA.hmm.tf_util import load_dirichlet, load_mvn
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

        if self.plm_config != None:
            # Create embedding value sets
            embedding_values = [
                PHMMEmbeddingValueSet.from_config(L, h, self.plm_config)
                for h, L in enumerate(lengths)
            ]

            # Set up the MVN prior for mean embeddings
            mvn_prior = load_mvn(
                self.plm_config.id_string() + ".weights",
                dim=len(SequenceDataset.alphabet)-1
            )

            # Override embedding values with prior distribution if requested
            if config.use_prior_for_emission_init:
                embedding_values = self._override_embeddings_with_prior(
                    embedding_values, mvn_prior
                )

            # Set the inverse gamma prior for embedding variances
            inv_gamma_prior = TFInverseGammaPrior()
            inv_gamma_prior.share = [0, 1] * 3
            inv_gamma_prior.initializer = [
                self.plm_config.inverse_gamma_alpha,
                self.plm_config.inverse_gamma_beta,
            ]

            combined_prior = TFCombinedPrior()
            combined_prior.add_prior(mvn_prior)
            combined_prior.add_prior(inv_gamma_prior)

            # Add the embedding emitter
            embedding_emitter = EmbeddingEmitter(values=embedding_values)
            self.hmm.add_emitter(embedding_emitter)
            embedding_emitter.prior = combined_prior

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

    @staticmethod
    def _override_embeddings_with_prior(
        values: Sequence[PHMMEmbeddingValueSet],
        prior
    ) -> list[PHMMEmbeddingValueSet]:
        """Override embedding values with prior distribution.

        Args:
            values: The original embedding value sets.
            prior: The MVN prior for embeddings.

        Returns:
            New value sets with embeddings replaced by prior distribution.
        """
        # Get the mean from the mixture model prior
        mean_per_component = prior.mean().numpy()
        mix_coef = prior.mixture_coefficients().numpy()
        mix_coef = np.expand_dims(mix_coef, axis=-1)
        mean = np.sum(mean_per_component * mix_coef, axis=2)
        mean = np.squeeze(mean, axis=(0, 1))

        updated_values = []
        for value_set in values:
            updated_values.append(PHMMEmbeddingValueSet(
                L=value_set.L,
                match_expectations=np.tile(mean, (value_set.L, 1, 1)),
                match_stddev=value_set.match_stddev,
                insert_expectation=mean,
                insert_stddev=value_set.insert_stddev,
            ))
        return updated_values
