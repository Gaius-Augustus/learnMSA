"""Backend-neutral access to the shipped prior parameters.

The prior parameters that learnMSA ships are a single flat kernel array per
prior. They used to be stored as Keras ``.weights.h5`` files, which could only
be read by building a throwaway ``tf.keras.Model``; they are stored as ``.npz``
now so that any backend can read them with numpy alone.

The framework-specific loaders in ``learnMSA/hmm/<backend>/util.py`` take the
array returned here and assign it to a built prior layer.
"""

import importlib.resources as resources
import logging
from pathlib import Path

import numpy as np

#: Package that holds the shipped prior parameter files.
WEIGHTS_PACKAGE = "learnMSA.hmm.weights"

#: Key under which the flat parameter kernel is stored inside an ``.npz``.
KERNEL_KEY = "kernel"

#: A concentration below this is treated as a collapsed dimension.
DEGENERATE_ALPHA: float = 1e-3

#: ...but only within a component that is otherwise informative.
INFORMATIVE_ALPHA: float = 1.0


def prior_basename(name: str) -> str:
    """Normalise a prior name to its file basename.

    Callers build names such as ``"amino_acid_dirichlet_9.weights"`` because the
    legacy files were called ``<name>.weights.h5``. The ``.weights`` part is
    dropped here, in one place, so the stored files are simply ``<name>.npz``.
    """
    return name[: -len(".weights")] if name.endswith(".weights") else name


def prior_path(name: str, suffix: str = ".npz") -> Path:
    """Path of a shipped prior parameter file."""
    resource = resources.files(WEIGHTS_PACKAGE) / f"{prior_basename(name)}{suffix}"
    return Path(str(resource))


def load_prior_kernel(name: str) -> np.ndarray:
    """Load the flat parameter kernel of a shipped prior.

    Args:
        name: Prior name, with or without a trailing ``.weights``.

    Returns:
        The parameter kernel as a 1-D numpy array.

    Raises:
        FileNotFoundError: If neither an ``.npz`` nor a legacy ``.h5`` file
            exists for this prior.
    """
    npz_path = prior_path(name, ".npz")
    if npz_path.exists():
        with np.load(npz_path) as data:
            return data[KERNEL_KEY]

    # Fall back to the legacy Keras file so that a prior that was just fitted
    # with the (still .h5-writing) tooling, or a partially converted checkout,
    # keeps working.
    h5_path = prior_path(name, ".weights.h5")
    if h5_path.exists():
        return read_h5_kernel(h5_path)

    raise FileNotFoundError(
        f"No parameters found for prior '{name}'. Looked for {npz_path} and "
        f"{h5_path}."
    )


def save_prior_kernel(kernel: np.ndarray, path: Path) -> None:
    """Write a flat parameter kernel to an ``.npz`` file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{KERNEL_KEY: np.asarray(kernel)})


def read_h5_kernel(path: Path) -> np.ndarray:
    """Read the parameter kernel out of a legacy Keras ``.weights.h5`` file.

    The archive holds exactly one weight, at ``layers/<layer>/vars/0``, so it can
    be read with h5py alone -- no Keras model has to be reconstructed.
    """
    import h5py  # optional: only needed for legacy files

    with h5py.File(str(path), "r") as archive:
        found: list[np.ndarray] = []

        def visit(_name, obj) -> None:
            if isinstance(obj, h5py.Dataset):
                found.append(obj[()])

        archive["layers"].visititems(visit)

    if len(found) != 1:
        raise ValueError(
            f"Expected exactly one weight array in {path}, found {len(found)}."
        )
    return np.asarray(found[0])


def warn_if_degenerate(alpha: np.ndarray, name: str) -> None:
    """Warn about Dirichlet components with a collapsed dimension.

    A concentration below one makes the density diverge at the simplex
    boundary, which is fine on its own: the flat 3Di priors consist entirely
    of such values. What is not fine is a concentration of ~0 sitting *inside*
    an otherwise informative component. It asserts that this letter is never
    emitted, which is a fitting artifact rather than a modelled belief, and it
    is what the prior behind the reported NaN run looked like.

    Judged per component, so a dead component of a mixture -- all
    concentrations tiny, and carrying a near-zero mixture weight -- does not
    trigger the warning.

    Args:
        alpha: Concentrations of shape ``(..., D)``, mixture coefficients
            already removed.
        name: Prior name, used in the warning message.
    """
    # Padding states carry all-zero rows and are not concentrations.
    alpha = alpha[alpha.sum(axis=-1) > 0]
    if alpha.size == 0:
        return
    degenerate = (
        (alpha.min(axis=-1) < DEGENERATE_ALPHA)
        & (alpha.max(axis=-1) > INFORMATIVE_ALPHA)
    )
    if not degenerate.any():
        return
    bad = alpha[degenerate]
    logging.warning(
        "Dirichlet prior '%s' has %d component(s) with a collapsed "
        "dimension: a concentration of %.3g next to a maximum of %.4g. That "
        "letter is effectively forbidden, which is usually a fitting "
        "artifact; consider refitting the prior.",
        name, int(degenerate.sum()), float(bad.min()), float(bad.max()),
    )
