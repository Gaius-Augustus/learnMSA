from collections.abc import Sequence
from typing import override

import numpy as np
import tensorflow as tf
from hidten.tf.emitter import TFPaddingEmitter

from learnMSA.hmm.util.value_set import PHMMValueSet
from learnMSA.util.sequence_dataset import SequenceDataset


class TFSubsetPaddingEmitter(TFPaddingEmitter):
    """A padding emitter that supports head subsets.
    """
    head_subset : Sequence[int] | None = None
    """If set, only these heads are used in computations."""

    @property
    def max_states(self) -> int:
        """The maximum number of states across all heads."""
        if self.head_subset is not None:
            return max(self.hmm_config.states[i] for i in self.head_subset)
        return self.hmm_config.max_states
