"""Gate for the ``.weights.h5`` -> ``.npz`` migration of the shipped priors.

Prior parameters used to be deserialized by building a throwaway
``tf.keras.Model`` and calling ``load_weights``. They are now read as plain
numpy and assigned to the built layer. This asserts that the two paths produce
bit-identical parameters for every shipped prior, so the migration cannot
silently shift the priors.
"""

from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from learnMSA.hmm.priors import (KERNEL_KEY, prior_path, read_h5_kernel,
                                 warn_if_degenerate)

WEIGHTS_DIR = Path(
    str(__import__("importlib.resources", fromlist=["resources"])
        .files("learnMSA.hmm.weights"))
)

H5_FILES = sorted(WEIGHTS_DIR.glob("*.weights.h5"))


def _npz_for(h5_path: Path) -> Path:
    return h5_path.with_suffix("").with_suffix(".npz")


def test_weight_files_exist() -> None:
    """Guards against the glob silently matching nothing."""
    assert H5_FILES, f"no .weights.h5 files found in {WEIGHTS_DIR}"


@pytest.mark.parametrize("h5_path", H5_FILES, ids=lambda p: p.name)
def test_npz_matches_h5_bit_for_bit(h5_path: Path) -> None:
    """Every shipped prior has an .npz holding exactly the .h5 kernel."""
    npz_path = _npz_for(h5_path)
    assert npz_path.exists(), f"missing converted file {npz_path.name}"

    expected = read_h5_kernel(h5_path)
    with np.load(npz_path) as data:
        actual = data[KERNEL_KEY]

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert np.array_equal(actual, expected), (
        f"{npz_path.name} differs from {h5_path.name}"
    )


def test_loaded_prior_matches_legacy_keras_path() -> None:
    """The new loader reproduces what ``model.load_weights`` produced.

    Uses the amino acid prior, whose exact concentrations feed straight into
    training, and checks the assembled ``matrix()`` rather than just the kernel.
    """
    from learnMSA.hmm.tf.util import load_dirichlet, make_dirichlet_model

    name, dim, components = "amino_acid_dirichlet_1.weights", 23, 1

    # New path: numpy kernel assigned to a built layer.
    new_prior = load_dirichlet(name, dim=dim, components=components)

    # Legacy path: throwaway keras model + load_weights from the .h5.
    legacy_model = make_dirichlet_model(dim=dim, components=components)
    legacy_model.load_weights(str(prior_path(name, ".weights.h5")))
    legacy_prior = legacy_model.layers[1]

    np.testing.assert_array_equal(
        new_prior.kernel.numpy(), legacy_prior.kernel.numpy()
    )
    np.testing.assert_array_equal(
        new_prior.matrix().numpy(), legacy_prior.matrix().numpy()
    )


def test_npz_is_the_primary_path_not_a_fallback() -> None:
    """Loading must not touch the legacy .h5 files.

    ``load_prior_kernel`` keeps an .h5 fallback for freshly fitted priors, so
    without this the whole migration could pass while every load still went
    through h5py.
    """
    import learnMSA.hmm.priors as priors

    def fail(path):  # pragma: no cover - only runs on regression
        raise AssertionError(f"fell back to the legacy .h5 reader for {path}")

    original = priors.read_h5_kernel
    priors.read_h5_kernel = fail
    try:
        for h5_path in H5_FILES:
            priors.load_prior_kernel(h5_path.name[: -len(".weights.h5")])
    finally:
        priors.read_h5_kernel = original


def test_load_prior_kernel_accepts_both_name_spellings() -> None:
    from learnMSA.hmm.priors import load_prior_kernel

    with_suffix = load_prior_kernel("amino_acid_dirichlet_1.weights")
    without_suffix = load_prior_kernel("amino_acid_dirichlet_1")
    np.testing.assert_array_equal(with_suffix, without_suffix)


def test_warn_if_degenerate_flags_collapsed_dimension(caplog) -> None:
    """A near-zero concentration next to an informative one warns."""
    alpha = np.array([[1e-6, 5.0, 2.0]])
    with caplog.at_level("WARNING"):
        warn_if_degenerate(alpha, "unit-test-prior")
    assert "collapsed" in caplog.text


def test_warn_if_degenerate_ignores_uniformly_flat_prior(caplog) -> None:
    """An all-small component is a legitimate flat prior, not a defect."""
    alpha = np.array([[1e-6, 1e-6, 1e-6]])
    with caplog.at_level("WARNING"):
        warn_if_degenerate(alpha, "unit-test-prior")
    assert "collapsed" not in caplog.text
