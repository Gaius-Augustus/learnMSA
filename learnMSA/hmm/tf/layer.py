"""TensorFlow profile HMM layer.

All of the pHMM's structure and configuration handling lives in the
backend-neutral :class:`learnMSA.hmm.layer.PHMMLayer`. This adds the keras layer
type, the component table naming the TF transitioner/emitter/prior classes, and
the three methods that actually touch tensors.
"""

from collections.abc import Sequence

import numpy as np
import tensorflow as tf
from hidten.tf.hmm import TFHMM, T_shapelike
from hidten.tf.prior import TFCombinedPrior, TFInverseGammaPrior
from hidten.tf.util import T_TFTensor

from learnMSA.config import LanguageModelConfig, PHMMConfig, PHMMPriorConfig
from learnMSA.config.structure import StructureConfig
from learnMSA.hmm.layer import PHMMComponents, PHMMLayer
from learnMSA.hmm.tf.embedding_emitter import EmbeddingEmitter
from learnMSA.hmm.tf.joint_profile_emitter import JointProfileEmitter
from learnMSA.hmm.tf.padding_emitter import TFSubsetPaddingEmitter
from learnMSA.hmm.tf.prior import TFPHMMStartPrior, TFPHMMTransitionPrior
from learnMSA.hmm.tf.profile_emitter import ProfileEmitter
from learnMSA.hmm.tf.transitioner import PHMMTransitioner
from learnMSA.hmm.tf.util import load_dirichlet, load_mvn
from learnMSA.hmm.util.value_set import PHMMValueSet
from learnMSA.hmm.util.value_set_emb import PHMMEmbeddingValueSet


class TFPHMMLayer(tf.keras.Layer, PHMMLayer[T_TFTensor]):
    """A profile HMM as a keras layer."""

    components = PHMMComponents(
        HMM=TFHMM,
        Transitioner=PHMMTransitioner,
        ProfileEmitter=ProfileEmitter,
        JointProfileEmitter=JointProfileEmitter,
        EmbeddingEmitter=EmbeddingEmitter,
        PaddingEmitter=TFSubsetPaddingEmitter,
        TransitionPrior=TFPHMMTransitionPrior,
        StartPrior=TFPHMMStartPrior,
        InverseGammaPrior=TFInverseGammaPrior,
        CombinedPrior=TFCombinedPrior,
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
        arguments; ``kwargs`` go to the keras layer (``name``, ``dtype``, ...).
        """
        tf.keras.Layer.__init__(self, **kwargs)
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

    def call(
        self,
        x: tf.Tensor,
        padding: tf.Tensor,
        adds: tuple[tf.Tensor, ...] | None = None,
    ) -> tf.Tensor:
        args = () if self.no_aa else (x,)
        if adds is not None:
            args += tuple(adds)
        args += (padding,)
        return self.hmm(*args, mode=self._mode, output_dtype=tf.int32)

    def prior_scores(self) -> tf.Tensor:
        """Calculates the prior scores for all parameters in the pHMM.

        Returns:
            Tensor: The prior scores of shape ``(H,)``, where ``H`` is the
                number of heads in the pHMM.
        """
        return self.hmm.prior_scores()
