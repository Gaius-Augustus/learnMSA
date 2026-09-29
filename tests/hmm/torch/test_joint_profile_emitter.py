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
from learnMSA.hmm.torch.profile_emitter import TorchProfileEmitter
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
def test_per_conditional_prior_scores_sum_over_rows(
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
def test_per_conditional_prior_scores_at_init(
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


# Hierarchical row prior (the default of --joint_emissions): the structural
# Dirichlet scores the implied marginal m = sum_a P(a|s) P(.|a,s) once per
# state and every row is shrunk toward m with Dir(c * m + 1).

CONCENTRATION = 20.0


def make_hierarchical_emitter(
    config: Configuration,
    hidten_config: HidtenHMMConfig,
    prior_name: str = "pfam_35_3Di_1.weights",
    low_rank: int = 0,
) -> tuple[TorchJointProfileEmitter, TorchProfileEmitter]:
    aa_values = [
        PHMMValueSet.from_config(L, h, config.hmm)
        for h, L in enumerate(LENGTHS)
    ]
    aa_emitter = TorchProfileEmitter(values=aa_values)
    aa_emitter.hmm_config = hidten_config
    aa_emitter.build((None, None, D1))
    emitter = make_conditional_emitter(
        config, hidten_config, low_rank=low_rank, with_prior=False
    )
    emitter.prior = load_dirichlet(
        prior_name, dim=D2, components=1, states=STATES
    )
    emitter.set_row_prior("hierarchical", CONCENTRATION, aa_emitter)
    return emitter, aa_emitter


def implied_marginal(
    emitter: TorchJointProfileEmitter, aa_emitter: TorchProfileEmitter
) -> torch.Tensor:
    return torch.einsum(
        "hqa,hqab->hqb",
        aa_emitter.matrix().detach(),
        conditional_rows(emitter),
    )


@pytest.mark.parametrize("low_rank", [0, 2])
def test_hierarchical_prior_scores_at_init(
    low_rank: int, config: Configuration, hidten_config: HidtenHMMConfig,
) -> None:
    """At initialisation all rows equal the marginal m, so the score is
    prior(m) + c * sum_a sum_b m_b log m_b."""
    emitter, aa_emitter = make_hierarchical_emitter(
        config, hidten_config, low_rank=low_rank
    )
    marginal = implied_marginal(emitter, aa_emitter)
    rows_term = torch.xlogy(marginal, marginal).sum(dim=(1, 2)) * D1
    expected = emitter._prior(marginal) + CONCENTRATION * rows_term
    scores = emitter.prior_scores()
    assert scores.shape == (2,)
    np.testing.assert_allclose(
        scores.detach().numpy(), expected.detach().numpy(), rtol=1e-4
    )


def test_hierarchical_row_term_has_no_gradient_at_init(
    config: Configuration, hidten_config: HidtenHMMConfig,
) -> None:
    """Rows that equal the marginal are the optimum of the row term, so only
    the marginal prior moves the kernel."""
    emitter, aa_emitter = make_hierarchical_emitter(config, hidten_config)
    row_term = emitter.prior_scores() \
        - emitter._prior(implied_marginal(emitter, aa_emitter))
    row_term.sum().backward()
    assert emitter.kernel.grad is not None
    assert emitter.kernel.grad.abs().max() < 1e-3


def _fit(emitter: TorchJointProfileEmitter, counts: torch.Tensor,
         steps: int, lr: float) -> None:
    optimizer = torch.optim.Adam([emitter.kernel], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        rows = conditional_rows(emitter)
        loglik = torch.xlogy(counts, rows.clamp_min(1e-30)).sum()
        loss = -(loglik + emitter.prior_scores().sum())
        loss.backward()
        optimizer.step()


def _synthetic_counts(seed: int = 0) -> torch.Tensor:
    """Counts on match states only (insert rows are shared), with every
    third amino acid row left without data."""
    rng = np.random.default_rng(seed)
    counts = np.zeros((len(LENGTHS), max(STATES), D1, D2))
    for h, L in enumerate(LENGTHS):
        profile = rng.dirichlet(np.full(D2, 0.3), size=(L, D1))
        counts[h, :L] = rng.poisson(50 * profile)
        counts[h, :L, ::3] = 0
    return torch.as_tensor(counts, dtype=torch.float32)


def test_hierarchical_map_is_backoff_to_marginal(
    config: Configuration, hidten_config: HidtenHMMConfig,
) -> None:
    """With the marginal prior switched off (huge temperature), the fixed
    point is p_ab = (n_ab + c m_b) / (n_a + c) and empty rows equal m."""
    emitter, aa_emitter = make_hierarchical_emitter(config, hidten_config)
    emitter._prior.temperature = 1e8
    counts = _synthetic_counts()
    _fit(emitter, counts, steps=3000, lr=0.05)

    rows = conditional_rows(emitter).detach()
    marginal = implied_marginal(emitter, aa_emitter).detach()
    n_a = counts.sum(-1, keepdim=True)
    expected = (counts + CONCENTRATION * marginal[:, :, None, :]) \
        / (n_a + CONCENTRATION)
    for h, L in enumerate(LENGTHS):
        np.testing.assert_allclose(
            rows[h, :L].numpy(), expected[h, :L].numpy(), atol=2e-3
        )
        np.testing.assert_allclose(
            rows[h, :L, ::3].numpy(),
            marginal[h, :L, None, :].expand(-1, len(range(0, D1, 3)), -1)
            .numpy(),
            atol=2e-3,
        )


def test_hierarchical_prior_keeps_empty_rows_off_zero(
    config: Configuration, hidten_config: HidtenHMMConfig,
) -> None:
    """scop_3Di_3_20 has six concentrations below 1. The per-conditional MAP
    pushes rows without data to zero on those letters; the hierarchical prior
    keeps them at the (non-degenerate) marginal."""
    counts = _synthetic_counts()
    minima = {}
    for mode in ("per_conditional", "hierarchical"):
        emitter, aa_emitter = make_hierarchical_emitter(
            config, hidten_config, prior_name="scop_3Di_3_20_1.weights"
        )
        emitter.set_row_prior(mode, CONCENTRATION, aa_emitter)
        _fit(emitter, counts, steps=500, lr=0.1)
        rows = conditional_rows(emitter).detach()
        minima[mode] = min(
            float(rows[h, :L, ::3].min()) for h, L in enumerate(LENGTHS)
        )
    assert minima["per_conditional"] < 1e-6
    assert minima["hierarchical"] > 1e-4


def test_hierarchical_prior_ignores_padding_states(
    config: Configuration, hidten_config: HidtenHMMConfig,
) -> None:
    """Identical emissions in every state: the two heads' scores are in the
    ratio of their real state counts, so padding states contribute nothing."""
    emitter, _ = make_hierarchical_emitter(config, hidten_config)
    scores = emitter.prior_scores().detach().numpy()
    np.testing.assert_allclose(
        scores[0] / scores[1], STATES[0] / STATES[1], rtol=1e-5
    )


def test_set_row_prior_rejects_unknown_mode(
    config: Configuration, hidten_config: HidtenHMMConfig,
) -> None:
    emitter, aa_emitter = make_hierarchical_emitter(config, hidten_config)
    with pytest.raises(ValueError):
        emitter.set_row_prior("global", CONCENTRATION, aa_emitter)


@pytest.mark.parametrize("row_prior", ["hierarchical", "per_conditional"])
def test_layer_joint_row_prior(row_prior: str) -> None:
    """The layer wires the row prior; the state dict is the same in both
    modes, i.e. the amino acid emitter is not registered twice."""
    from learnMSA.hmm.torch.layer import TorchPHMMLayer

    struct_config = StructureConfig(
        use_structure=True, joint_emissions=True, joint_row_prior=row_prior
    )
    layer = TorchPHMMLayer(
        lengths=LENGTHS, config=PHMMConfig(), struct_config=struct_config
    )
    shapes = (
        (None, None, len(LENGTHS), 20),
        (None, None, len(LENGTHS), struct_config.alphabet_size),
        (None, None, len(LENGTHS), 1),
    )
    layer.build(shapes)
    assert layer.joint_emitter.row_prior == row_prior
    scores = layer.prior_scores()
    assert scores.shape == (len(LENGTHS),)
    assert torch.all(torch.isfinite(scores))

    reference = TorchPHMMLayer(
        lengths=LENGTHS, config=PHMMConfig(),
        struct_config=StructureConfig(
            use_structure=True, joint_emissions=True,
            joint_row_prior="per_conditional",
        ),
    )
    reference.build(shapes)
    assert list(layer.state_dict()) == list(reference.state_dict())


def test_hierarchical_prior_without_row_term(
    config: Configuration, hidten_config: HidtenHMMConfig,
) -> None:
    """Concentration 0 keeps only the structural prior on the marginal."""
    emitter, aa_emitter = make_hierarchical_emitter(config, hidten_config)
    emitter.set_row_prior("hierarchical", 0.0, aa_emitter)
    expected = emitter._prior(implied_marginal(emitter, aa_emitter))
    np.testing.assert_allclose(
        emitter.prior_scores().detach().numpy(),
        expected.detach().numpy(), rtol=1e-6,
    )
