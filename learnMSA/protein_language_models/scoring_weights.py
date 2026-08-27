"""Backend-neutral loading of the shipped scoring model parameters.

A bilinear scoring model is just two arrays: a projection ``R`` of shape
``(embedding_dim, reduced_dim)`` and a scalar bias ``b``. They used to be
stored as Keras ``.h5`` files, which could only be read by rebuilding a
throwaway
``tf.keras.Model``; they are stored as ``.npz`` now so that any backend can
read them with numpy alone. Same move, and the same shape, as
:mod:`learnMSA.hmm.priors`.

The framework-specific ``make_reduction_layer`` in
``learnMSA/protein_language_models/<backend>/bilinear_symmetric.py`` takes the
arrays returned here and assigns them to a built reduction layer.
"""

import importlib.resources as resources
from pathlib import Path

import numpy as np

from learnMSA.protein_language_models.common import (SCORING_MODEL_PATH,
                                                     ScoringModelConfig)

#: Package that holds the shipped scoring model parameter files.
WEIGHTS_PACKAGE = f"learnMSA.protein_language_models.{SCORING_MODEL_PATH}"

#: Keys under which the parameters are stored inside an ``.npz``.
KERNEL_KEYS: tuple[str, ...] = ("R", "b")


def scoring_weights_basename(config: ScoringModelConfig) -> str:
    """The extension-less file name identifying a scoring model."""
    return (
        f"{config.lm_name}_{config.dim}_{config.activation}{config.suffix}"
    )


def scoring_weights_path(
    config: ScoringModelConfig, suffix: str = ".npz"
) -> Path:
    """Path of a shipped scoring model parameter file."""
    resource = (
        resources.files(WEIGHTS_PACKAGE)
        / f"{scoring_weights_basename(config)}{suffix}"
    )
    return Path(str(resource))


def load_scoring_weights(
    config: ScoringModelConfig
) -> dict[str, np.ndarray]:
    """Load the parameters of a shipped scoring model.

    Args:
        config: Identifies the scoring model.

    Returns:
        A mapping with an ``"R"`` and a ``"b"`` entry.

    Raises:
        FileNotFoundError: If neither an ``.npz`` nor a legacy ``.h5`` file
            exists for this scoring model.
    """
    npz_path = scoring_weights_path(config, ".npz")
    if npz_path.exists():
        with np.load(npz_path) as data:
            return {key: data[key] for key in KERNEL_KEYS}

    # Fall back to the legacy Keras file, so that a scoring model that was just
    # fitted with the (still .h5-writing) pretraining tooling, or a partially
    # converted checkout, keeps working.
    h5_path = scoring_weights_path(config, ".h5")
    if h5_path.exists():
        return read_h5_scoring_weights(h5_path)

    raise FileNotFoundError(
        f"No parameters found for scoring model "
        f"'{scoring_weights_basename(config)}'. Looked for {npz_path} and "
        f"{h5_path}."
    )


def save_scoring_weights(
    weights: dict[str, np.ndarray], path: Path
) -> None:
    """Write scoring model parameters to an ``.npz`` file."""
    missing = [key for key in KERNEL_KEYS if key not in weights]
    if missing:
        raise ValueError(f"Missing scoring model parameters: {missing}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, **{key: np.asarray(weights[key]) for key in KERNEL_KEYS}
    )


def read_h5_scoring_weights(path: Path) -> dict[str, np.ndarray]:
    """Read scoring model parameters out of a legacy Keras ``.h5`` file.

    These are Keras 2 whole-model weight files: the arrays sit at
    ``<layer>/<layer>/R:0`` and ``<layer>/<layer>/b:0``, keyed by the
    TensorFlow variable name rather than by a plain index. That is a different
    layout from
    the ``.weights.h5`` files :func:`learnMSA.hmm.priors.read_h5_kernel`
    reads, which is why this has its own reader.

    Args:
        path: The ``.h5`` file.

    Returns:
        A mapping with an ``"R"`` and a ``"b"`` entry.

    Raises:
        ValueError: If the archive does not hold exactly the expected arrays.
    """
    import h5py  # optional: only needed for legacy files

    found: dict[str, np.ndarray] = {}

    def visit(name: str, obj) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        key = name.rsplit("/", 1)[-1].split(":", 1)[0]
        if key in KERNEL_KEYS:
            found[key] = np.asarray(obj[()])

    with h5py.File(str(path), "r") as archive:
        archive.visititems(visit)

    missing = [key for key in KERNEL_KEYS if key not in found]
    if missing:
        raise ValueError(
            f"Expected {list(KERNEL_KEYS)} in {path}, missing {missing}."
        )
    return found
