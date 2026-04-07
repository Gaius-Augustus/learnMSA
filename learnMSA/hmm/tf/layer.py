from collections.abc import Sequence

import numpy as np
import tensorflow as tf
from hidten import HMMMode
from hidten.tf.hmm import TFHMM, T_shapelike
from hidten.tf.prior import TFCombinedPrior, TFInverseGammaPrior

from learnMSA.config import LanguageModelConfig, PHMMConfig, PHMMPriorConfig
from learnMSA.config.structure import StructureConfig
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
        lengths: Sequence[int] | np.ndarray | None,
        config : PHMMConfig,
        prior_config: PHMMPriorConfig | None = None,
        plm_config: LanguageModelConfig | None = None,
        structural_config: StructureConfig | None = None,
        use_prior: bool = True,
        trainable_insertions: bool = True,
        value_sets: Sequence[PHMMValueSet] | None = None,
        no_aa: bool = False,
        **kwargs
    ) -> None:
        """
        Args:
            lengths: The number of match states in each head of the pHMM.
                May be ``None`` if ``value_sets`` is provided, in which case
                the lengths are inferred from the value sets.
            config: HMM configuration parameters. The initial emission and
                transition values in ``config`` are ignored when ``value_sets``
                is provided.
            prior_config: Prior configuration parameters.
            plm_config: Protein language model configuration.
            structural_config: Structural information configuration.
            use_prior: Whether to use priors for regularization.
            trainable_insertions: Whether insertion emissions are trainable.
            value_sets: Optional pre-built :class:`PHMMValueSet` objects, one
                per head. When provided, ``PHMMValueSet.from_config`` is
                skipped and ``lengths`` may be ``None``.
            no_aa: Whether to use amino acid emissions in the model.
        """
        super().__init__(**kwargs)
        if prior_config is None:
            prior_config = PHMMPriorConfig()
        self.use_prior = use_prior
        self.no_aa = no_aa

        if value_sets is not None:
            values = list(value_sets)
            self.lengths = np.array([vs.L for vs in values], dtype=np.int32)
        else:
            assert lengths is not None, (
                "lengths must be provided when value_sets is None"
            )
            self.lengths = np.asarray(lengths, dtype=np.int32)
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
                dim = len(SequenceDataset._default_alphabet)-1,
                states = self.states,
            )

        # Override emission values with prior distribution if requested
        if config.use_prior_for_emission_init and value_sets is None:
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
                self.lengths, prior_config
            )
            self.hmm.transitioner.prior_start = TFPHMMStartPrior(
                self.lengths, prior_config
            )

        if not no_aa:
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
        self.emb_mean = None
        if self.use_language_model:
            assert self.plm_config is not None,\
                "plm_config must be provided if use_language_model is True"
            # Create embedding value sets
            emb_values = [
                PHMMEmbeddingValueSet.from_config(L, h, plm_config) # type: ignore
                for h, L in enumerate(self.lengths)
            ]

            # Set up the MVN prior for mean embeddings
            mvn_prior = load_mvn(
                self.plm_config.id_string() + ".weights",
                dim=self.plm_config.scoring_model_dim,
                components=self.plm_config.embedding_prior_components,
                states=self.states,
            )

            # Override embedding values with prior distribution if requested
            if config.use_prior_for_emission_init:
                emb_values, emb_mean = self._override_embeddings_with_prior(
                    emb_values,
                    mvn_prior,
                    override_matches=config.match_emissions is None,
                    override_insertions=config.insert_emissions is None,
                )
                self.emb_mean = emb_mean

            # Set the inverse gamma prior for embedding variances
            inv_gamma_prior = TFInverseGammaPrior()
            inv_gamma_prior.share = np.tile(
                [0, 1],
                reps=2 * sum(self.lengths) + 2 * len(self.lengths)
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
                values=emb_values,
                trainable_insertions=trainable_insertions,
                temperature=self.plm_config.temperature,
            )
            self.hmm.add_emitter(embedding_emitter)
            if self.use_prior:
                embedding_emitter.prior = combined_prior

        self.structural_config = structural_config
        if structural_config and structural_config.use_structure:
            self.use_structure = True
            struct_values = [
                PHMMValueSet.from_structural_config(L, h, structural_config)
                for h, L in enumerate(self.lengths)
            ]
            structural_emitter = ProfileEmitter(
                values=struct_values, trainable_insertions=trainable_insertions
            )
            self.hmm.add_emitter(structural_emitter)

            # If specified, load and add a Dirichlet prior
            if structural_config.prior_name:
                struct_prior = load_dirichlet(
                    structural_config.prior_name+".weights",
                    dim=structural_config.alphabet_size,
                    components=structural_config.prior_components,
                    states=self.states,
                )
                struct_prior.temperature = structural_config.prior_temperature
                structural_emitter.prior = struct_prior
        else:
            self.use_structure = False

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
        args = () if self.no_aa else (x,)
        if adds is not None:
            args += tuple(adds)
        args += (padding,)
        return self.hmm(*args, mode=self._mode)

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
        prior_dist = prior.matrix().numpy()[0, 0]
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
        prior,
        override_matches: bool,
        override_insertions: bool,
    ) -> tuple[list[PHMMEmbeddingValueSet], np.ndarray]:
        """Override embedding values with prior distribution.

        Args:
            values: The original embedding value sets.
            prior: The MVN prior for embeddings.
            override_matches: Whether to override match state emissions.
            override_insertions: Whether to override insertion state emissions.

        Returns:
            New value sets with embeddings replaced by prior distribution.
        """
        # Get the mean from the mixture model prior
        mean_per_component = prior.mean().numpy()[0, 0]
        mix_coef = prior.mixture_coefficients().numpy()[0, 0]
        mix_coef = np.expand_dims(mix_coef, axis=-1)
        mean = np.sum(mean_per_component * mix_coef, axis=0)

        if not override_matches and not override_insertions:
            return list(values), mean

        updated_values = []
        for value_set in values:
            if override_matches:
                match_expectations = np.tile(mean, (value_set.L, 1))
            else:
                match_expectations = value_set.match_expectations
            if override_insertions:
                insert_expectation = mean
            else:
                insert_expectation = value_set.insert_expectation
            updated_values.append(PHMMEmbeddingValueSet(
                L=value_set.L,
                match_expectations=match_expectations,
                match_variance=value_set.match_variance,
                insert_expectation=insert_expectation,
                insert_variance=value_set.insert_variance,
            ))
        return updated_values, mean
