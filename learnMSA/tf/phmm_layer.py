import tensorflow as tf
from hidten import HMMMode
from hidten.config import with_config
from hidten.tf import TFHMM, TFCategoricalEmitter, TFPaddingEmitter

from learnMSA.phmm_config import ProfileHMMConfig
from learnMSA.util import make_phmm_transitions


@with_config(ProfileHMMConfig)
class ProfileHMM(tf.keras.Layer):

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.config = ProfileHMMConfig(**kwargs)

        self.hmm = TFHMM(
            states=self.config.states,
            heads=self.config.heads,
        )

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