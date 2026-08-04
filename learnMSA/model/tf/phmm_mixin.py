import numpy as np

from learnMSA.hmm.tf.layer import PHMMLayer


class PHMMMixin():
    phmm_layer: PHMMLayer
    """The PHMM layer used in the model."""

    _decode_msa: bool = False

    @property
    def lengths(self) -> np.ndarray:
        """The number of match states in each head of the pHMM.
        """
        return self.phmm_layer.lengths

    @property
    def heads(self) -> int:
        """The number of pHMM heads."""
        return self.phmm_layer.heads

    def loglik_mode(self) -> None:
        """Makes the model return log-likelihoods.
        """
        self.phmm_layer.loglik_mode()
        self._decode_msa = False

    def viterbi_mode(self) -> None:
        """Makes the model return Viterbi paths.
        """
        self.phmm_layer.viterbi_mode()
        self._decode_msa = False

    def mea_mode(self) -> None:
        """Makes the model return MEA paths.
        """
        self.phmm_layer.mea_mode()
        self._decode_msa = False

    def viterbi_decode_mode(self) -> None:
        """Makes the model return Viterbi paths.
        """
        self.phmm_layer.viterbi_mode()
        self._decode_msa = True

    def mea_decode_mode(self) -> None:
        """Makes the model return MEA paths.
        """
        self.phmm_layer.mea_mode()
        self._decode_msa = True

    def posterior_mode(self) -> None:
        """Makes the model return state posterior probabilities.
        """
        self.phmm_layer.posterior_mode()
        self._decode_msa = False
