from collections.abc import Sequence

import numpy as np
import tensorflow as tf
from hidten.tf import TFTransitioner

from learnMSA.config.hmm import HMMConfig
from learnMSA.hmm.transition_index_set import PHMMTransitionIndexSet
from learnMSA.hmm.value_set import PHMMValueSet


class PHMMExplicitTransitioner(TFTransitioner):
    """A transitioner for explicit pHMMs with deletion states.
    This transitioner contains silent states and needs to be folded.

    The order of states in each head is:

    M1 ... ML I1 ... IL-1 D1 ... DL L B E C R T.

    where L is the number of match states in that head.
    """
    def __init__(self, lengths: Sequence[int], config: HMMConfig, **kwargs) -> None:
        super().__init__()
        self.lengths = lengths
        self.config = config
        transitions, values = [], []
        start, start_values = [], []
        for h, L in enumerate(lengths):
            # Transitions
            index_set = PHMMTransitionIndexSet(L, folded=False)
            value_set = PHMMValueSet.from_config(L, config)
            # get all index pairs (i,j) and add head index (h,i,j)
            transitions.append(
               np.pad(index_set.as_array(), ((0,0),(1,0)), constant_values=h)
            )
            values.append(value_set.as_array())

            # Start distribution
            start.extend([(h, 0), (h, 1)])
            start_values.append(value_set.start)

        self.allow = np.vstack(transitions).tolist()
        self.initializer = np.hstack(values)

        self.allow_start = start
        self.initializer_start = start_values