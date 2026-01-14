import numpy as np

from learnMSA.hmm.tf.layer import PHMMLayer


class HMMStatsMixin():
    phmm_layer: PHMMLayer
    """The PHMM layer used in the model."""

    @property
    def lengths(self) -> np.ndarray:
        """The number of match states in each head of the pHMM.
        """
        return self.phmm_layer.lengths

    @property
    def heads(self) -> int:
        """The number of pHMM heads."""
        return self.phmm_layer.heads
