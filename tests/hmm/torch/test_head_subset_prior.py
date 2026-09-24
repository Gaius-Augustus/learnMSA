"""Prior scores under a head subset.

The priors cover all heads, so modules score their full parameter matrix and
select the subset afterwards. Under ``head_subset = S`` every component must
return ``full[S]`` with shape ``(len(S),)``.
"""

import os

import numpy as np
import pytest
import torch
from hidten.hmm import HMMConfig as HidtenHMMConfig

from learnMSA.config import Configuration, PHMMConfig, StructureConfig
from learnMSA.hmm.torch.embedding_emitter import TorchEmbeddingEmitter
from learnMSA.hmm.torch.joint_profile_emitter import TorchJointProfileEmitter
from learnMSA.hmm.torch.util import load_dirichlet, make_mvn_prior
from learnMSA.hmm.util.value_set import PHMMValueSet
from learnMSA.hmm.util.value_set_emb import PHMMEmbeddingValueSet
from learnMSA.model.context import LearnMSAContext
from learnMSA.model.torch.model import TorchLearnMSAModel
from learnMSA.util import SequenceDataset

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")
SUBSETS = [[0], [1], [1, 0]]


def _assert_subset_scores(module, full: torch.Tensor, subset) -> None:
    module.head_subset = subset
    try:
        with torch.no_grad():
            scores = module.prior_scores()
    finally:
        module.head_subset = None
    assert scores.shape == (len(subset),)
    np.testing.assert_allclose(
        scores.cpu().numpy(), full[subset].cpu().numpy(), rtol=1e-5
    )


@pytest.mark.parametrize("lengths", [[5, 4], [4, 5, 6], [5, 5]])
@pytest.mark.parametrize("subset", SUBSETS)
def test_layer_prior_scores(lengths: list[int], subset: list[int]) -> None:
    config = Configuration()
    config.training.no_sequence_weights = True
    config.training.length_init = lengths
    data = SequenceDataset(os.path.join(DATA, "felix.fa"))
    model = TorchLearnMSAModel(LearnMSAContext(config, data))
    model.build()
    layer = model.phmm_layer
    components = [layer.hmm.transitioner, *layer.hmm.emitter]
    with torch.no_grad():
        full = layer.prior_scores()
        full_components = [c.prior_scores() for c in components]
    _assert_subset_scores(layer, full, subset)
    for component, component_full in zip(components, full_components):
        if isinstance(component_full, torch.Tensor):
            _assert_subset_scores(component, component_full, subset)


@pytest.mark.parametrize("low_rank", [0, 2])
@pytest.mark.parametrize("subset", SUBSETS)
def test_joint_emitter_prior_scores(low_rank: int, subset: list[int]) -> None:
    lengths = [4, 3]
    states = [2 * L + 2 for L in lengths]
    config = Configuration(hmm=PHMMConfig(), structure=StructureConfig())
    emitter = TorchJointProfileEmitter(
        marginal_values=[
            [PHMMValueSet.from_config(L, h, config.hmm)
             for h, L in enumerate(lengths)],
            [PHMMValueSet.from_structural_config(L, h, config.structure)
             for h, L in enumerate(lengths)],
        ],
        low_rank=low_rank,
        conditional=True,
    )
    emitter.hmm_config = HidtenHMMConfig(states=states)
    emitter.build(((None, None, 20), (None, None, 20)))
    emitter.prior = load_dirichlet(
        "pfam_35_3Di_1.weights", dim=20, components=1, states=states,
    )
    with torch.no_grad():
        full = emitter.prior_scores()
    _assert_subset_scores(emitter, full, subset)


@pytest.mark.parametrize("subset", SUBSETS)
def test_embedding_emitter_prior_scores(subset: list[int]) -> None:
    lengths, dim = [5, 3], 4
    states = [2 * L + 2 for L in lengths]
    rng = np.random.default_rng(0)
    values = [
        PHMMEmbeddingValueSet(
            L=L,
            match_expectations=rng.normal(size=(L, dim)).astype(np.float32),
            match_variance=rng.uniform(.5, 2, (L, dim)).astype(np.float32),
            insert_expectation=rng.normal(size=dim).astype(np.float32),
            insert_variance=rng.uniform(.5, 2, dim).astype(np.float32),
        )
        for L in lengths
    ]
    emitter = TorchEmbeddingEmitter(values)
    emitter.hmm_config = HidtenHMMConfig(states=states)
    emitter.build((None, None, dim))
    emitter.prior = make_mvn_prior(dim=dim, states=states)
    with torch.no_grad():
        full = emitter.prior_scores()
    # distinct heads, so that a wrong head order is detected
    assert not torch.allclose(full[0], full[1])
    _assert_subset_scores(emitter, full, subset)
