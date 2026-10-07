"""learnMSA's step-launch threshold reaches hidten's dispatch.

hidten keeps its own thresholds (257 states, 512 for multiples of 16) and
lets callers change them process-wide; learnMSA lowers them to
``TRITON_MIN_Q``, and only when it runs the Triton kernels at all.
"""

import numpy as np
import pytest
from hidten.torch.triton import step_launch

import tests.hmm.ref as ref
from learnMSA.config import (AdvancedConfig, Configuration, TrainingConfig,
                             TreeConfig)
from learnMSA.config.hmm import PHMMPriorConfig
from learnMSA.hmm.torch.layer import TRITON_MIN_Q
from learnMSA.model.context import LearnMSAContext
from learnMSA.model.torch.model import TorchLearnMSAModel


@pytest.fixture(autouse=True)
def restore_thresholds():
    saved = (step_launch.AUTO_MIN_Q, step_launch.AUTO_MIN_Q_UNALIGNED)
    yield
    step_launch.set_auto_min_q(*saved)


def _build(use_triton: bool) -> TorchLearnMSAModel:
    hmm_config = ref.config.model_copy(deep=True)
    hmm_config.use_prior_for_emission_init = False
    config = Configuration(
        training=TrainingConfig(length_init=[4, 3]),
        tree=TreeConfig(use_anc_probs=False),
        hmm=hmm_config,
        hmm_prior=PHMMPriorConfig(use_amino_acid_prior=False),
        advanced=AdvancedConfig(use_triton=use_triton),
    )
    context = LearnMSAContext(
        config=config,
        num_seq=10,
        sequence_weights=np.arange(10, dtype=float),
    )
    return TorchLearnMSAModel(context)


def test_triton_sets_the_threshold() -> None:
    _build(use_triton=True)
    assert step_launch.AUTO_MIN_Q == TRITON_MIN_Q
    assert step_launch.AUTO_MIN_Q_UNALIGNED == TRITON_MIN_Q
    assert step_launch.auto_selects_step_launch(16, TRITON_MIN_Q)
    assert not step_launch.auto_selects_step_launch(16, TRITON_MIN_Q - 2)


def test_without_triton_hidten_keeps_its_thresholds() -> None:
    before = (step_launch.AUTO_MIN_Q, step_launch.AUTO_MIN_Q_UNALIGNED)
    _build(use_triton=False)
    assert (step_launch.AUTO_MIN_Q,
            step_launch.AUTO_MIN_Q_UNALIGNED) == before
