from collections.abc import Sequence

import tensorflow as tf
from hidten import HMMMode
from hidten.tf import TFHMM, TFCategoricalEmitter, TFPaddingEmitter

from learnMSA.config.hmm import HMMConfig

from learnMSA.phmm_config import ProfileHMMConfig
from learnMSA.msa_hmm.phmm_util import make_phmm_transitions


class ProfileHMMLayer(tf.keras.Layer):

    lengths: Sequence[int]
    """The number of match states in each head of the pHMM.
    """

    @property
    def heads(self) -> int:
        """The number of pHMM heads."""
        return len(self.lengths)

    @property
    def states(self) -> Sequence[int]:
        """The total number of states in each head."""
        return [2*L+3 for L in self.lengths]

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
        super().__init__()
        self.lengths = lengths
        self.config = config

        self.hmm = TFHMM(states=self.states, heads=self.heads)

        transitions, values = [], []
        for h, (L, pMM, pII) in enumerate(zip(model_length, p_match, p_insert)):
            for i in range(L - 1):
                # match to match
                transitions.append((h, i, i + 1))
                values.append(pMM)
                # match to insert
                transitions.append((h, i, L+i))
                values.append(1 - pMM)
                # self-loop in insert
                transitions.append((h, L+i, L+i))
                values.append(pII)
                # insert to match
                transitions.append((h, L+i, i + 1))
                values.append(1 - pII)
            transitions.append((h, L - 1, L - 1))  # last match state self-loop
            values.append(1)

        start, start_values = [], []
        # define starting states and values

        self.hmm.transitioner.allow = transitions
        self.hmm.transitioner.initializer = values
        self.hmm.transitioner.allow_start = start
        self.hmm.transitioner.initializer_start = start_values

        amino_emitter = TFCategoricalEmitter()
        self.hmm.add_emitter(amino_emitter)
        self.hmm.add_emitter(TFPaddingEmitter())


    def build(self, input_shape: tuple[int | None, ...]) -> None:
        self.hmm.build(input_shape)


    def call(self, x: tf.Tensor, padding: tf.Tensor) -> tf.Tensor:
        return self.hmm(x, padding, HMMMode.LIKELIHOOD)