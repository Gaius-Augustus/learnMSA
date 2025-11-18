from collections.abc import Sequence

import numpy as np
import tensorflow as tf
from hidten.hmm import HMMConfig as HidtenHMMConfig
from hidten.tf.transitioner import T_TFTensor, TFTransitioner, shared_tensor

from learnMSA.hmm.transition_index_set import PHMMTransitionIndexSet
from learnMSA.hmm.value_set import PHMMValueSet


class PHMMExplicitTransitioner(TFTransitioner):
    """A transitioner for explicit pHMMs with deletion states.
    This transitioner contains silent states and needs to be folded.

    The order of states in each head is:

    M1 ... ML I1 ... IL-1 D1 ... DL L B E C R T.

    where L is the number of match states in that head.

    Args:
        values (Sequence[PHMMValueSet]): A sequence of value sets, one per head.
        hidten_hmm_config (HidtenHMMConfig): The configuration of the hidten HMM.
    """
    def __init__(
        self,
        values: Sequence[PHMMValueSet],
        hidten_hmm_config: HidtenHMMConfig,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.hmm_config = hidten_hmm_config
        transitions, value_list = [], []
        start, start_values = [], []
        for h, value_set in enumerate(values):
            index_set = PHMMTransitionIndexSet(value_set.L, folded=False)
            # Transitions
            # get all index pairs (i,j) and add head index (h,i,j)
            indices = index_set.as_array()
            transitions.append(
               np.pad(indices, ((0,0),(1,0)), constant_values=h)
            )
            value_list.append(value_set.transitions[indices[:,0], indices[:,1]])

            # Start distribution
            start.append(np.pad(
                index_set.start[:,np.newaxis], ((0,0), (1,0)), constant_values=h
            ))
            start_values.append(value_set.start)

        self.allow = np.vstack(transitions).tolist()
        self.initializer = np.hstack(value_list)

        self.allow_start = np.vstack(start).tolist()
        self.initializer_start = np.hstack(start_values)


class PHMMTransitioner(TFTransitioner):
    """A transitioner for folded pHMMs without deletion states. Wraps an
    explicit transitioner which holds all the parameters. Overrides the matrix
    and start distribution methods to provide the folded versions.

    The order of states in each head is:

    M1 ... ML I1 ... IL-1 L B E C R T.

    where L is the number of match states in that head.

    Args:
        values (Sequence[PHMMValueSet]): A sequence of value sets, one per head.
        hidten_hmm_config (HidtenHMMConfig): The configuration of the hidten HMM.
    """
    def __init__(
        self,
        values: Sequence[PHMMValueSet],
        hidten_hmm_config: HidtenHMMConfig,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.explicit_transitioner = PHMMExplicitTransitioner(
            values=values,
            hidten_hmm_config=hidten_hmm_config,
        )
        self.hmm_config = hidten_hmm_config
        # Construct allow indices for the folded models
        transitions, start = [], []
        for h, value_set in enumerate(values):
            index_set = PHMMTransitionIndexSet(value_set.L, folded=True)
            # Transitions
            # get all index pairs (i,j) and add head index (h,i,j)
            indices = index_set.as_array()
            transitions.append(
               np.pad(indices, ((0,0),(1,0)), constant_values=h)
            )

            # Start distribution
            start.append(np.pad(
                index_set.start[:, np.newaxis], ((0,0),(1,0)), constant_values=h
            ))

        self.allow = np.vstack(transitions).tolist()
        self.allow_start = np.vstack(start).tolist()

    def build(self) -> None:
        super().build()
        self.explicit_transitioner.build()

    def get_explicit_transition_probs(self) -> T_TFTensor:
        pass

    def matrix(self) -> T_TFTensor:
        # Construct the matrix like usual in the transitioner, but instead of
        # using a parameter kernel, we use the explicit transition probabilities
        return shared_tensor(
            indices=self.allow,
            values=self.get_explicit_transition_probs(),
            shape=tf.constant(
                [
                    self.hmm_config.heads,
                    self.hmm_config.max_states,
                    self.matrix_dim,
                ],
                dtype=tf.int64,
            ),
            share=self.share,
        ) # no activation needed

    def start_dist(self) -> T_TFTensor:
        # Same principle: construct start distribution from explicit transition
        # probabilities
        dense_tensor = shared_tensor(
            indices=self.allow_start,
            values=self.get_explicit_transition_probs(),
            shape=[self.hmm_config.heads, self.hmm_config.max_states],
            share=self.share_start,
        )
        init = zero_row_softmax(dense_tensor)
        return init
