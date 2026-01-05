from collections.abc import Sequence

import tensorflow as tf
from hidten import HMMMode
from hidten.tf.emitter import TFPaddingEmitter
from hidten.tf.hmm import TFHMM, T_shapelike

from learnMSA.config import PHMMConfig, PHMMPriorConfig, LanguageModelConfig
from learnMSA.hmm.profile_emitter import ProfileEmitter
from learnMSA.hmm.embedding_emitter import EmbeddingEmitter
from learnMSA.hmm.transitioner import PHMMTransitioner
from learnMSA.hmm.value_set import PHMMValueSet


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

        # TODO: clean this mess up
        # 3 different HMM configs are currently needed
        # the HMM has a config with 2L + 2 states, excluding terminal, because
        # hidten expects the configuration to not count the padding state

        # the transitioner needs a custom configuration with 2L + 3 states,
        # because it handles the padding/terminal state explicitly

        # the emitter needs another custom config with L + 1 states, because it
        # shares all insertion states and does not use hidten's share system
        # for performance reasons

        self.hmm = TFHMM(states=self.states, heads=self.heads)

        # Add the transitioner
        # TODO: avoid the hack of keeping the custom config
        transitioner = PHMMTransitioner(
            values = values, prior_config=prior_config
        )
        transitioner_custom_config = transitioner.hmm_config
        self.hmm.transitioner = transitioner # this overwrites hmm_config
        # restore custom config
        transitioner.hmm_config = transitioner_custom_config

        # Add the profile emitter
        profile_emitter = ProfileEmitter(
            values = values,
            use_prior_aa_dist=config.use_prior_for_emission_init
        )
        emitter_custom_config = profile_emitter.hmm_config
        self.hmm.add_emitter(profile_emitter) # this overwrites hmm_config
        profile_emitter.hmm_config = emitter_custom_config # restore

        # Add the MVN emitter
        if self.plm_config is not None:
            emb_emitter = EmbeddingEmitter()
            emitter_custom_config = emb_emitter.hmm_config
            self.hmm.add_emitter(emb_emitter) # this overwrites hmm_config
            emb_emitter.hmm_config = emitter_custom_config # restore


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
