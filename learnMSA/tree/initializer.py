"""Backend-neutral descriptions of how model parameters are initialized.

An :class:`InitSpec` says *what* a parameter should start at without committing
to a framework. The backend packages turn a spec into the thing their layers
need: ``learnMSA.tree.tf.initializer.to_tf`` builds a
``keras.initializers.Initializer``, and a torch backend would build a plain
callable.

This lets :class:`~learnMSA.model.context.LearnMSAContext` describe the whole
initialization of a run without importing a tensor framework.
"""

from dataclasses import dataclass

import numpy as np

from evoten.substitution_models import AF_3Di, LG, foldseek_3Di

from learnMSA.util.sequence_dataset import SequenceDataset


@dataclass(frozen=True)
class InitSpec:
    """Base class of the initialization specifications."""


@dataclass(frozen=True)
class Constant(InitSpec):
    """Initialize with fixed values."""

    value: np.ndarray | float


@dataclass(frozen=True)
class RandomNormal(InitSpec):
    """Initialize by drawing from a normal distribution."""

    stddev: float
    mean: float = 0.0


@dataclass(frozen=True)
class Zeros(InitSpec):
    """Initialize with zeros."""


def inverse_softplus(x: np.ndarray | float) -> np.ndarray:
    """The inverse of ``softplus``, i.e. ``log(exp(x) - 1)``.

    Computed in float64 to prevent overflow of large entries and cast back to
    the input dtype, matching the tensor implementation in
    :mod:`learnMSA.tree.tf.util` element for element.
    """
    features = np.asarray(x)
    result = np.log(np.expm1(features.astype(np.float64)))
    if features.dtype.kind == "f":
        return result.astype(features.dtype)
    return result


def make_substitution_model_init(
    num_models: int,
    type: str = "LG",
    num_components: int = 1,
    shared_equilibrium: bool = True,
    shared_exchangeabilities: bool = False,
    alphabet: str = SequenceDataset._default_alphabet[:20],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Constructs initializers for exchangeabilities and equilibrium frequencies
    based on an existing substitution model.
    """

    # (D, D), (D,)
    if type == "LG":
        R, p = LG(alphabet)
    elif type == "foldseek_3Di":
        R, p = foldseek_3Di(alphabet)
    elif type == "AF_3Di":
        R, p = AF_3Di(alphabet)
    else:
        raise ValueError(f"Unknown substitution model type: {type}")

    # Build exchangeability initializer: (H, 1, K_R, D, D)
    R_init = inverse_softplus(R + 1e-32)
    exchangeability_stack = np.tile(
        R_init[None, None, None], [num_models, 1, 1, 1, 1]
    )  # (H, 1, 1, D, D)
    if not shared_exchangeabilities and num_components > 1:
        exchangeability_stack = np.tile(
            exchangeability_stack, [1, 1, num_components, 1, 1]
        )  # (H, 1, K, D, D)

    # Build equilibrium initializer: (H, 1, K_p, D)
    log_p_stack = np.tile(
        np.log(p)[None, None, None], [num_models, 1, 1, 1]
    )  # (H, 1, 1, D)
    if not shared_equilibrium and num_components > 1:
        log_p_stack = np.tile(
            log_p_stack, [1, 1, num_components, 1]
        )  # (H, 1, K, D)

    return exchangeability_stack, log_p_stack
