from collections.abc import Sequence

import tensorflow as tf
from hidten import HMMMode
from hidten.tf.hmm import TFHMM, T_shapelike
from hidten.tf.emitter import TFPaddingEmitter

from learnMSA.config.hmm import HMMConfig
from learnMSA.hmm.profile_emitter import ProfileEmitter
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

    def __init__(self, lengths: Sequence[int], config : HMMConfig, **kwargs) -> None:
        """
        Args:
            lengths: The number of match states in each head of the pHMM.
            config: HMM configuration parameters.
        """
        super().__init__(**kwargs)
        self.lengths = lengths
        self.config = config

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
        transitioner = PHMMTransitioner(values = values)
        transitioner_custom_config = transitioner.hmm_config
        self.hmm.transitioner = transitioner # this overwrites hmm_config
        transitioner.hmm_config = transitioner_custom_config # restore custom config

        # Add the profile emitter and padding emitter
        profile_emitter = ProfileEmitter(
            values = values,
            use_prior_aa_dist=config.use_prior_for_emission_init
        )
        emitter_custom_config = profile_emitter.hmm_config
        self.hmm.add_emitter(profile_emitter) # this overwrites hmm_config
        profile_emitter.hmm_config = emitter_custom_config # restore
        self.hmm.add_emitter(TFPaddingEmitter())

    def build(self, input_shape: T_shapelike) -> None:
        self.hmm.build(input_shape)

    def call(self, x: tf.Tensor, padding: tf.Tensor) -> tf.Tensor:
        return self.hmm(x, padding, mode=HMMMode.LIKELIHOOD_LOG)
