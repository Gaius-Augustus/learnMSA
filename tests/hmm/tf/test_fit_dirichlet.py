"""Lower-bounded Dirichlet fitting (``fit_dirichlet --min-alpha``)."""

import numpy as np
import pytest

from learnMSA.hmm.priors import KERNEL_KEY
from learnMSA.hmm.tf.fit_dirichlet import (build_trainable_model,
                                           make_initializer, save,
                                           standard_prior_model, train)
from learnMSA.hmm.tf.util import make_dirichlet_prior

DIM = 5
#: Two concentrations below 1, like the shipped 3Di prior.
TRUE_ALPHA = np.array([0.4, 0.7, 2.0, 3.0, 5.0])


@pytest.fixture(scope="module")
def columns() -> np.ndarray:
    """Count columns drawn from Dir(TRUE_ALPHA)-multinomial."""
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(TRUE_ALPHA, size=3000)
    counts = np.stack([rng.multinomial(20, p) for p in probs])
    return counts.astype(np.float64)


def fit(columns: np.ndarray, min_alpha: float):
    model = build_trainable_model(
        make_initializer(columns, 1, seed=0), DIM, 1,
        use_map_prior=False, score_counts=True, min_alpha=min_alpha,
    )
    train(model, columns, None, lr=0.05, epochs=300, batch_size=512,
          patience=3, seed=0)
    return standard_prior_model(model.prior.get_weights(), DIM, 1, min_alpha)


def alphas(model) -> np.ndarray:
    return model.layers[1].matrix()[0, 0].numpy()


def test_unbounded_fit_recovers_small_alphas(columns: np.ndarray) -> None:
    fitted = alphas(fit(columns, 0.0))
    np.testing.assert_allclose(fitted, TRUE_ALPHA, rtol=0.25)
    assert np.all(fitted[:2] < 1.0)


def test_bounded_fit_keeps_alphas_above_bound(columns: np.ndarray) -> None:
    fitted = alphas(fit(columns, 1.0))
    assert np.all(fitted > 1.0)
    # The formerly small concentrations sit at the bound; the order is kept.
    assert np.all(fitted[:2] < 1.5)
    assert np.all(np.diff(fitted) > 0)


def test_bounded_fit_saves_plain_kernel(columns: np.ndarray, tmp_path) -> None:
    """The saved kernel is a plain softplus kernel of the bounded alphas."""
    model = fit(columns, 1.0)
    path = tmp_path / "bounded_1.npz"
    save(model, path, DIM, 1)
    reloaded = make_dirichlet_prior(dim=DIM)
    with np.load(path) as data:
        reloaded.kernel.assign(data[KERNEL_KEY].astype(np.float32))
    np.testing.assert_allclose(
        reloaded.matrix()[0, 0].numpy(), alphas(model), rtol=1e-5
    )
    assert np.all(reloaded.matrix().numpy() > 1.0)
