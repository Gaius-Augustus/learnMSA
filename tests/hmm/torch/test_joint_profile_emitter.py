"""Priors on the conditional joint emitter (``conditional=True``).

The emitter models ``P(struct | aa, state)``, i.e. one categorical
distribution per ``(state, aa)`` row, so the structural Dirichlet is applied
to every row rather than to a marginal.
"""

import numpy as np
import pytest
import torch
from hidten.hmm import HMMConfig as HidtenHMMConfig

from learnMSA.config import Configuration, PHMMConfig, StructureConfig
from learnMSA.hmm.torch.joint_profile_emitter import TorchJointProfileEmitter
from learnMSA.hmm.torch.util import load_dirichlet
from learnMSA.hmm.util.value_set import PHMMValueSet

LENGTHS = [4, 3]
STATES = [2 * L + 2 for L in LENGTHS]  # [10, 8]
D1 = D2 = 20


@pytest.fixture
def hidten_config() -> HidtenHMMConfig:
    return HidtenHMMConfig(states=STATES)


@pytest.fixture
def config() -> Configuration:
    """Default (background) emissions: every state gets the same, non-degenerate
    distribution, which keeps the Dirichlet densities finite."""
    return Configuration(hmm=PHMMConfig(), structure=StructureConfig())


def make_conditional_emitter(
    config: Configuration,
    hidten_config: HidtenHMMConfig,
    low_rank: int = 0,
    components: int = 1,
    with_prior: bool = True,
) -> TorchJointProfileEmitter:
    aa_values = [
        PHMMValueSet.from_config(L, h, config.hmm)
        for h, L in enumerate(LENGTHS)
    ]
    struct_values = [
        PHMMValueSet.from_structural_config(L, h, config.structure)
        for h, L in enumerate(LENGTHS)
    ]
    emitter = TorchJointProfileEmitter(
        marginal_values=[aa_values, struct_values],
        low_rank=low_rank,
        conditional=True,
    )
    emitter.hmm_config = hidten_config
    emitter.build(((None, None, D1), (None, None, D2)))
    if with_prior:
        emitter.prior = load_dirichlet(
            f"pfam_35_3Di_{components}.weights",
            dim=D2, components=components, states=STATES,
        )
    return emitter


def conditional_rows(emitter: TorchJointProfileEmitter) -> torch.Tensor:
    matrix = emitter.matrix()
    return matrix.reshape(matrix.shape[0], matrix.shape[1], D1, D2)


@pytest.mark.parametrize("low_rank", [0, 2])
@pytest.mark.parametrize("components", [1, 9])
def test_conditional_prior_scores_sum_over_rows(
    low_rank: int, components: int,
    config: Configuration, hidten_config: HidtenHMMConfig,
) -> None:
    """The prior is applied to each of the D1 conditional rows and summed."""
    emitter = make_conditional_emitter(
        config, hidten_config, low_rank=low_rank, components=components
    )
    rows = conditional_rows(emitter)
    expected = sum(
        emitter._prior(rows[:, :, i]) for i in range(D1)
    )
    scores = emitter.prior_scores()
    assert scores.shape == (2,)
    assert torch.all(torch.isfinite(scores))
    np.testing.assert_allclose(
        scores.detach().numpy(), expected.detach().numpy(),
        rtol=1e-4,  # float32 accumulation over the 20 rows
    )


@pytest.mark.parametrize("low_rank", [0, 2])
def test_conditional_prior_scores_at_init(
    low_rank: int, config: Configuration, hidten_config: HidtenHMMConfig,
) -> None:
    """At initialisation every row equals the structural marginal, so the
    score is exactly D1 times the score of a single row."""
    emitter = make_conditional_emitter(
        config, hidten_config, low_rank=low_rank
    )
    rows = conditional_rows(emitter)
    single_row = emitter._prior(rows[:, :, 0])
    np.testing.assert_allclose(
        emitter.prior_scores().detach().numpy(),
        (D1 * single_row).detach().numpy(),
        rtol=1e-4,  # float32 accumulation over the 20 rows
    )


def test_conditional_prior_ignores_padding_states(
    config: Configuration, hidten_config: HidtenHMMConfig,
) -> None:
    """Head 1 has 8 of the 10 padded states. With identical emissions in every
    state the two heads' scores must be in a 10:8 ratio -- padded rows sum to
    zero and are masked out by the prior."""
    emitter = make_conditional_emitter(config, hidten_config)
    scores = emitter.prior_scores().detach().numpy()
    np.testing.assert_allclose(
        scores[0] / scores[1], STATES[0] / STATES[1], rtol=1e-5
    )


def test_marginal_prior_rejected_when_conditional(
    config: Configuration, hidten_config: HidtenHMMConfig,
) -> None:
    """Marginal priors need the joint parameterisation; the conditional table
    has no marginals to score."""
    emitter = make_conditional_emitter(
        config, hidten_config, with_prior=False
    )
    prior = load_dirichlet(
        "pfam_35_3Di_1.weights", dim=D2, components=1, states=STATES
    )
    with pytest.raises(AssertionError):
        emitter.add_marginal_prior(1, prior)
