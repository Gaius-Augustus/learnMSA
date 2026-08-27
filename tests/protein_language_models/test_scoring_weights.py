from pathlib import Path

import numpy as np
import pytest

from learnMSA.protein_language_models.common import ScoringModelConfig, dims
from learnMSA.protein_language_models.scoring_weights import (
    KERNEL_KEYS, load_scoring_weights, read_h5_scoring_weights,
    save_scoring_weights, scoring_weights_path)

#: Every scoring model that ships with learnMSA.
SHIPPED = [
    ScoringModelConfig(lm_name=lm, dim=dim, activation=activation)
    for lm in ("protT5", "esm2", "proteinBERT")
    for dim in (16, 32, 64, 128)
    for activation in ("sigmoid", "softmax")
]


@pytest.mark.parametrize(
    "config", SHIPPED, ids=lambda c: f"{c.lm_name}_{c.dim}_{c.activation}"
)
def test_shipped_weights_have_the_expected_shapes(
    config: ScoringModelConfig,
) -> None:
    weights = load_scoring_weights(config)
    assert set(weights) == set(KERNEL_KEYS)
    assert weights["R"].shape == (dims[config.lm_name], config.dim)
    assert weights["b"].shape == (1,)
    assert np.isfinite(weights["R"]).all()
    assert np.isfinite(weights["b"]).all()


@pytest.mark.parametrize(
    "config", SHIPPED, ids=lambda c: f"{c.lm_name}_{c.dim}_{c.activation}"
)
def test_npz_matches_the_legacy_h5(config: ScoringModelConfig) -> None:
    """The conversion must be exact, not merely close."""
    # h5py is only needed to read the legacy files and is not a hard
    # dependency of either backend.
    pytest.importorskip("h5py")

    h5_path = scoring_weights_path(config, ".h5")
    if not h5_path.exists():
        pytest.skip(f"{h5_path.name} was dropped from the repository")

    from_h5 = read_h5_scoring_weights(h5_path)
    from_npz = load_scoring_weights(config)
    for key in KERNEL_KEYS:
        np.testing.assert_array_equal(from_npz[key], from_h5[key])


def test_round_trip(tmp_path: Path) -> None:
    weights = {
        "R": np.arange(12, dtype=np.float32).reshape(4, 3),
        "b": np.array([-3.0], dtype=np.float32),
    }
    path = tmp_path / "sub" / "toy.npz"
    save_scoring_weights(weights, path)

    with np.load(path) as data:
        for key in KERNEL_KEYS:
            np.testing.assert_array_equal(data[key], weights[key])


def test_saving_incomplete_weights_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Missing"):
        save_scoring_weights({"R": np.zeros((2, 2))}, tmp_path / "toy.npz")


def test_missing_scoring_model_raises() -> None:
    config = ScoringModelConfig(lm_name="protT5", suffix="_does_not_exist")
    with pytest.raises(FileNotFoundError, match="No parameters found"):
        load_scoring_weights(config)
