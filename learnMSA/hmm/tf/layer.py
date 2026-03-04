from collections.abc import Sequence

import numpy as np
import tensorflow as tf
from hidten import HMMMode
from hidten.tf.hmm import TFHMM, T_shapelike
from hidten.tf.prior import TFCombinedPrior, TFInverseGammaPrior

from learnMSA.config import LanguageModelConfig, PHMMConfig, PHMMPriorConfig
from learnMSA.hmm.tf.embedding_emitter import EmbeddingEmitter
from learnMSA.hmm.tf.prior import TFPHMMStartPrior, TFPHMMTransitionPrior
from learnMSA.hmm.tf.profile_emitter import ProfileEmitter
from learnMSA.hmm.tf.transitioner import PHMMTransitioner
from learnMSA.hmm.tf.padding_emitter import TFSubsetPaddingEmitter
from learnMSA.hmm.tf.util import load_dirichlet, load_mvn
from learnMSA.hmm.util.value_set import PHMMValueSet
from learnMSA.hmm.util.value_set_emb import PHMMEmbeddingValueSet
from learnMSA.util.sequence_dataset import SequenceDataset


class PHMMLayer(tf.keras.Layer):

    lengths: np.ndarray
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

    @property
    def head_subset(self) -> Sequence[int] | None:
        """If set, only these heads are used in computations."""
        return self.hmm.transitioner.head_subset

    @head_subset.setter
    def head_subset(self, subset: Sequence[int] | None) -> None:
        self.hmm.transitioner.head_subset = subset
        for emitter in self.hmm.emitter:
            if hasattr(emitter, "head_subset"):
                emitter.head_subset = subset

    _mode: HMMMode = HMMMode.LIKELIHOOD_LOG
    """Determines the return value of the layer."""

    def __init__(
        self,
        lengths: Sequence[int] | np.ndarray,
        config : PHMMConfig,
        prior_config: PHMMPriorConfig | None = None,
        plm_config: LanguageModelConfig | None = None,
        use_prior: bool = True,
        trainable_insertions: bool = True,
        **kwargs
    ) -> None:
        """
        Args:
            lengths: The number of match states in each head of the pHMM.
            config: HMM configuration parameters.
            prior_config: Prior configuration parameters.
            plm_config: Protein language model configuration.
            use_prior: Whether to use priors for regularization.
            trainable_insertions: Whether insertion emissions are trainable.
        """
        super().__init__(**kwargs)
        self.lengths = np.asarray(lengths, dtype=np.int32)
        if prior_config is None:
            prior_config = PHMMPriorConfig()
        self.use_prior = use_prior

        values = [
            PHMMValueSet.from_config(L, h, config)
            for h, L in enumerate(lengths)
        ]

        # Apply random noise
        if config.use_noise:
            for value_set in values:
                value_set.add_noise(concentration=config.noise_concentration)

        if prior_config.use_amino_acid_prior:
            # Set up the Dirichlet prior for emissions
            emission_prior = load_dirichlet(
                "amino_acid_dirichlet.weights",
                dim = len(SequenceDataset._default_alphabet)-1
            )
            # Share concentrations across all states
            emission_prior.share = np.tile(
                np.arange(len(SequenceDataset._default_alphabet)-1),
                reps=2 * sum(lengths) + 2 * len(lengths)
            )

        # Override emission values with prior distribution if requested
        if config.use_prior_for_emission_init:
            assert prior_config.use_amino_acid_prior, (
                "Cannot use prior for emission initialization if no "
                "emission prior is set."
            )
            values = self._override_emissions_with_prior(
                values,
                emission_prior,
                override_matches=config.match_emissions is None,
                override_insertions=config.insert_emissions is None,
            )

        # Create the HMM, with 2*L+2 states per head
        self.hmm = TFHMM(states=self.states, heads=self.heads)

        # Add the transitioner and prior
        self.hmm.transitioner = PHMMTransitioner(
            values = values
        )
        if self.use_prior:
            self.hmm.transitioner.prior = TFPHMMTransitionPrior(
                lengths, prior_config
            )
            self.hmm.transitioner.prior_start = TFPHMMStartPrior(
                lengths, prior_config
            )

        # Add the profile emitter
        profile_emitter = ProfileEmitter(
            values=values, trainable_insertions=trainable_insertions
        )
        self.hmm.add_emitter(profile_emitter)
        if self.use_prior and prior_config.use_amino_acid_prior:
            profile_emitter.prior = emission_prior

        self.use_language_model = plm_config != None\
            and plm_config.use_language_model
        self.plm_config = plm_config
        if self.use_language_model:
            # Create embedding value sets
            embedding_values = [
                PHMMEmbeddingValueSet.from_config(L, h, plm_config) # type: ignore
                for h, L in enumerate(lengths)
            ]

            # Set up the MVN prior for mean embeddings
            mvn_prior = load_mvn(
                self.plm_config.id_string() + ".weights",
                dim=self.plm_config.scoring_model_dim,
                components=self.plm_config.embedding_prior_components,
            )

            # Override embedding values with prior distribution if requested
            if config.use_prior_for_emission_init:
                embedding_values = self._override_embeddings_with_prior(
                    embedding_values, mvn_prior
                )

            # Set the inverse gamma prior for embedding variances
            inv_gamma_prior = TFInverseGammaPrior()
            inv_gamma_prior.share = np.tile(
                [0, 1],
                reps=2 * sum(lengths) + 2 * len(lengths)
            )
            inv_gamma_prior.initializer = [
                self.plm_config.inverse_gamma_alpha,
                self.plm_config.inverse_gamma_beta,
            ]

            combined_prior = TFCombinedPrior()
            combined_prior.add_prior(mvn_prior)
            combined_prior.add_prior(inv_gamma_prior)

            # Add the embedding emitter
            embedding_emitter = EmbeddingEmitter(
                values=embedding_values,
                trainable_insertions=trainable_insertions,
            )
            self.hmm.add_emitter(embedding_emitter)
            if self.use_prior:
                embedding_emitter.prior = combined_prior

        # Add the padding emitter
        self.hmm.add_emitter(TFSubsetPaddingEmitter())

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

    def is_loglik_mode(self) -> bool:
        """Check if the layer is in log-likelihood mode.
        """
        return self._mode == HMMMode.LIKELIHOOD_LOG

    def is_viterbi_mode(self) -> bool:
        """Check if the layer is in Viterbi mode.
        """
        return self._mode == HMMMode.VITERBI

    def is_posterior_mode(self) -> bool:
        """Check if the layer is in posterior mode.
        """
        return self._mode == HMMMode.POSTERIOR

    def build(self, input_shape: T_shapelike) -> None:
        self.hmm.build(input_shape)

    def call(
        self,
        x: tf.Tensor,
        padding: tf.Tensor,
        adds: tuple[tf.Tensor, ...] | None = None,
    ) -> tf.Tensor:
        if adds is None:
            return self.hmm(x, padding, mode=self._mode)
        else:
            return self.hmm(x, *adds, padding, mode=self._mode)

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
        prior,
        override_matches: bool,
        override_insertions: bool,
    ) -> list[PHMMValueSet]:
        """Override emission values in value sets with prior distribution.

        Args:
            values: The original value sets.
            prior: The Dirichlet prior for emissions.
            override_matches: Whether to override match state emissions.
            override_insertions: Whether to override insertion state emissions.

        Returns:
            New value sets with emissions replaced by prior distribution.
        """
        prior_dist = prior.matrix().numpy().flatten()
        prior_dist = prior_dist / np.sum(prior_dist)
        updated_values = []
        if not override_matches and not override_insertions:
            return list(values)
        for value_set in values:
            if override_matches:
                match_emissions = np.tile(prior_dist, (value_set.L, 1))
            else:
                match_emissions = value_set.match_emissions
            if override_insertions:
                insert_emissions = prior_dist
            else:
                insert_emissions = value_set.insert_emissions
            updated_values.append(PHMMValueSet(
                L=value_set.L,
                match_emissions=match_emissions,
                insert_emissions=insert_emissions,
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
                match_expectations=np.tile(mean, (value_set.L, 1)),
                match_stddev=value_set.match_stddev,
                insert_expectation=mean,
                insert_stddev=value_set.insert_stddev,
            ))
        return updated_values
