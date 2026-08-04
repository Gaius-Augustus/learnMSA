"""PyTorch profile HMM layer.

All of the pHMM's structure and configuration handling lives in the
backend-neutral :class:`learnMSA.hmm.layer.PHMMLayer`. This adds the torch
module type, the component table naming the torch transitioner/emitter/prior
classes, and the three methods that actually touch tensors.
"""

from collections.abc import Sequence

import numpy as np
import torch
from hidten.torch.hmm import T_shapelike, TorchHMM
from hidten.torch.prior import TorchCombinedPrior, TorchInverseGammaPrior
from hidten.torch.util import T_TorchTensor

from learnMSA.config import LanguageModelConfig, PHMMConfig, PHMMPriorConfig
from learnMSA.config.structure import StructureConfig
from learnMSA.hmm.layer import PHMMComponents, PHMMLayer
from learnMSA.hmm.torch.embedding_emitter import TorchEmbeddingEmitter
from learnMSA.hmm.torch.joint_profile_emitter import TorchJointProfileEmitter
from learnMSA.hmm.torch.padding_emitter import TorchSubsetPaddingEmitter
from learnMSA.hmm.torch.prior import (TorchPHMMStartPrior,
                                      TorchPHMMTransitionPrior)
from learnMSA.hmm.torch.profile_emitter import TorchProfileEmitter
from learnMSA.hmm.torch.transitioner import TorchPHMMTransitioner
from learnMSA.hmm.torch.util import load_dirichlet, load_mvn
from learnMSA.hmm.util.value_set import PHMMValueSet
from learnMSA.hmm.util.value_set_emb import PHMMEmbeddingValueSet
from learnMSA.util.tensor import to_numpy


class TorchPHMMLayer(torch.nn.Module, PHMMLayer[T_TorchTensor]):
    """A profile HMM as a torch module."""

    components = PHMMComponents(
        HMM=TorchHMM,
        Transitioner=TorchPHMMTransitioner,
        ProfileEmitter=TorchProfileEmitter,
        JointProfileEmitter=TorchJointProfileEmitter,
        EmbeddingEmitter=TorchEmbeddingEmitter,
        PaddingEmitter=TorchSubsetPaddingEmitter,
        TransitionPrior=TorchPHMMTransitionPrior,
        StartPrior=TorchPHMMStartPrior,
        InverseGammaPrior=TorchInverseGammaPrior,
        CombinedPrior=TorchCombinedPrior,
        load_dirichlet=load_dirichlet,
        load_mvn=load_mvn,
    )

    def __init__(
        self,
        lengths: Sequence[int] | np.ndarray | None,
        config: PHMMConfig,
        prior_config: PHMMPriorConfig | None = None,
        plm_config: LanguageModelConfig | None = None,
        struct_config: StructureConfig | None = None,
        use_prior: bool = True,
        trainable_insertions: bool = True,
        aa_value_sets: Sequence[PHMMValueSet] | None = None,
        emb_value_sets: Sequence[PHMMEmbeddingValueSet] | None = None,
        struct_value_sets: Sequence[PHMMValueSet] | None = None,
        joint_aa_struct_value_sets: Sequence[PHMMValueSet] | None = None,
        no_aa: bool = False,
        **kwargs
    ) -> None:
        """See :meth:`learnMSA.hmm.layer.PHMMLayer._init_phmm` for the
        arguments; ``kwargs`` are accepted for signature parity with the
        TensorFlow layer and ignored.
        """
        torch.nn.Module.__init__(self)
        self._init_phmm(
            lengths=lengths,
            config=config,
            prior_config=prior_config,
            plm_config=plm_config,
            struct_config=struct_config,
            use_prior=use_prior,
            trainable_insertions=trainable_insertions,
            aa_value_sets=aa_value_sets,
            emb_value_sets=emb_value_sets,
            struct_value_sets=struct_value_sets,
            joint_aa_struct_value_sets=joint_aa_struct_value_sets,
            no_aa=no_aa,
        )

    def build(self, input_shape: T_shapelike) -> None:
        self.hmm.build(input_shape)

    def forward(
        self,
        x: T_TorchTensor,
        padding: T_TorchTensor,
        adds: tuple[T_TorchTensor, ...] | None = None,
    ) -> T_TorchTensor:
        args = () if self.no_aa else (x,)
        if adds is not None:
            args += tuple(adds)
        args += (padding,)
        return self.hmm(*args, mode=self._mode, output_dtype=torch.int32)

    #: The neutral base and its callers use ``call``; torch modules use
    #: ``forward``. They are the same method.
    call = forward

    def prior_scores(self) -> T_TorchTensor:
        """Calculates the prior scores for all parameters in the pHMM.

        Returns:
            The prior scores of shape ``(H,)``, where ``H`` is the number of
            heads in the pHMM.
        """
        return self.hmm.prior_scores()

    def get_weights(self) -> list[np.ndarray]:
        """All parameters and buffers, in ``state_dict`` order.

        Keras layers provide this natively; torch's nearest equivalent is the
        state dict, whose ordering is stable for a given architecture.
        """
        return [to_numpy(tensor) for tensor in self.state_dict().values()]

    def set_weights(self, weights: list[np.ndarray]) -> None:
        """Overwrite all parameters and buffers, in :meth:`get_weights` order."""
        state = self.state_dict()
        if len(weights) != len(state):
            raise ValueError(
                f"Expected {len(state)} weight arrays for this pHMM layer, "
                f"got {len(weights)}."
            )
        self.load_state_dict({
            name: torch.as_tensor(
                weight, dtype=current.dtype, device=current.device
            )
            for (name, current), weight in zip(state.items(), weights)
        })
