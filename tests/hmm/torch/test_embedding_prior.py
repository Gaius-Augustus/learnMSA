"""The prior on the embedding emitter, on both sides of ``reduce_online``.

The emitter's matrix is ``[means | variances]`` and the two halves are scored by
two independent priors summed in a ``CombinedPrior``: the means by a multivariate
normal, the variances by an inverse gamma. Only the mean half is language-model
specific, and only when the embeddings come out of the frozen projection the
shipped mixture was fitted for. Under ``reduce_online`` the projection is
trainable, so that mixture describes nothing and a generic standard normal takes
its place; the variance half is unaffected either way.
"""

import math

import numpy as np
import pytest
import torch

from learnMSA.config import LanguageModelConfig, PHMMConfig, PHMMPriorConfig
from learnMSA.hmm.torch.layer import TorchPHMMLayer

#: One head, so the prior score is a clean multiple of the per-state density.
LENGTHS = [4]
STATES = 2 * LENGTHS[0] + 2

#: Alphabet width the amino acid emitter is built for.
AA_DIM = 20

#: A width no scoring model and no embedding prior ship weights for.
UNSHIPPED_DIM = 17


def _layer(dim: int, reduce_online: bool) -> TorchPHMMLayer:
    layer = TorchPHMMLayer(
        lengths=LENGTHS,
        config=PHMMConfig(),
        prior_config=PHMMPriorConfig(),
        plm_config=LanguageModelConfig(
            use_language_model=True,
            scoring_model_dim=dim,
            reduce_online=reduce_online,
        ),
        use_prior=True,
    )
    layer.build(((None, None, AA_DIM), (None, None, dim)))
    return layer


def _mean_prior(layer: TorchPHMMLayer):
    """The first summand of the combined prior, i.e. the one over the means."""
    return layer.embedding_emitter.prior.priors[0]


def test_online_prior_is_a_standard_normal() -> None:
    """Zero means and unit scales -- not the -inf scales a zero variance gives.

    ``make_mvn_prior`` reads its initializer in natural space and maps the
    variance half through ``inverse_softplus(sqrt(v))``, so leaving it at the
    default of zeros would produce -inf.
    """
    prior = _mean_prior(_layer(UNSHIPPED_DIM, reduce_online=True))

    assert prior.config.components == 1
    mean = prior.mean()
    scale = prior.sqrt_variance()
    assert mean.shape == (1, STATES, 1, UNSHIPPED_DIM)
    torch.testing.assert_close(mean, torch.zeros_like(mean))
    torch.testing.assert_close(scale, torch.ones_like(scale))


def test_online_prior_scores_the_closed_form_density() -> None:
    """At the zero-mean, unit-variance init the score is exactly N(0, I)."""
    layer = _layer(UNSHIPPED_DIM, reduce_online=True)
    emitter = layer.embedding_emitter
    matrix = emitter.matrix()

    # The emitter starts at means of zero, where the standard normal density is
    # its maximum, -0.5 * D * log(2*pi) per state.
    per_state = -0.5 * UNSHIPPED_DIM * math.log(2 * math.pi)
    score = _mean_prior(layer).prior_scores(matrix)
    assert score.item() == pytest.approx(STATES * per_state, rel=1e-5)

    total = layer.prior_scores()
    assert torch.all(torch.isfinite(total))


def test_online_emitter_means_start_at_zero() -> None:
    """The generic prior's mean is zero, which is where the emitter starts."""
    layer = _layer(UNSHIPPED_DIM, reduce_online=True)
    means = layer.embedding_emitter.matrix()[..., :UNSHIPPED_DIM]
    torch.testing.assert_close(means, torch.zeros_like(means))
    # Surgery reads this back to fill freshly inserted match states.
    assert layer.emb_mean is not None
    np.testing.assert_allclose(layer.emb_mean, np.zeros(UNSHIPPED_DIM))


def test_the_variance_prior_is_unaffected_by_reduce_online() -> None:
    """It comes from the config, not from disk, so it is the same either way."""
    online = _layer(UNSHIPPED_DIM, reduce_online=True)
    offline = _layer(16, reduce_online=False)

    for layer in (online, offline):
        priors = layer.embedding_emitter.prior.priors
        assert len(priors) == 2
        alpha_beta = priors[1].matrix()
        # alpha and beta are shared across states and come from the config.
        torch.testing.assert_close(
            alpha_beta[0, 0],
            torch.tensor([
                LanguageModelConfig().inverse_gamma_alpha,
                LanguageModelConfig().inverse_gamma_beta,
            ]),
        )


def test_offline_still_loads_the_shipped_mixture() -> None:
    """A regression guard on the path that legitimately reads from disk."""
    prior = _mean_prior(_layer(16, reduce_online=False))

    assert prior.config.components == \
        LanguageModelConfig().embedding_prior_components
    # The fitted mixture is nothing like a standard normal.
    assert prior.mean().abs().max() > 0.0
    assert not torch.allclose(
        prior.sqrt_variance(), torch.ones_like(prior.sqrt_variance())
    )


def test_offline_needs_shipped_weights() -> None:
    """The offline path is only valid where the fitted mixture exists.

    This is the failure ``reduce_online`` must not inherit; it is what
    ``tests/model/torch/test_reduce_online.py`` pins from the other side.
    """
    with pytest.raises(FileNotFoundError):
        _layer(UNSHIPPED_DIM, reduce_online=False)
