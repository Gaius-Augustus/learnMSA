from typing import Literal

import numpy as np

from learnMSA.hmm.layer import PHMMLayer


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

    def viterbi_mode(
        self, posterior: Literal["decoded"] | None = None
    ) -> None:
        """Makes the model return Viterbi paths (see
        :meth:`PHMMLayer.viterbi_mode` for ``posterior``).
        """
        self.phmm_layer.viterbi_mode(posterior)
        self._decode_msa = False

    def mea_mode(
        self, posterior: Literal["decoded", "full"] | None = None
    ) -> None:
        """Makes the model return MEA paths (see
        :meth:`PHMMLayer.mea_mode` for ``posterior``).
        """
        self.phmm_layer.mea_mode(posterior)
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

    def posterior_mode(self, decoding_temperature: bool = False) -> None:
        """Makes the model return state posterior probabilities (see
        :meth:`PHMMLayer.posterior_mode` for ``decoding_temperature``).
        """
        self.phmm_layer.posterior_mode(decoding_temperature)
        self._decode_msa = False
