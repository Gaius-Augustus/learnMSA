"""Backend-neutral helpers for joint amino-acid/structure emissions.

Initialisation and reshaping of the low-rank joint emission parameterisation.
All pure numpy, so model surgery can use them without a tensor framework; the
framework-specific emitter imports them as well.
"""

from typing import Sequence

import numpy as np

from learnMSA.hmm.util.value_set import PHMMValueSet


def AB_init(
    n1: int,
    n2: int,
    low_rank: int,
    batch_shape: tuple[int, ...] = (),
    noise_std: float = 1e-2,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Initialises A and B matrices near zero for low-rank parameterisation.

    A is filled with small Gaussian noise; B is zeros.  Because B = 0,
    AB^T = 0 exactly at initialisation, so the initial joint distribution
    is determined entirely by the constant log-joint bias C.

    Args:
        n1 (int): Size of the first marginal alphabet.
        n2 (int): Size of the second marginal alphabet.
        low_rank (int): Rank of the approximation.  Must be >= 1.
        batch_shape (tuple[int, ...]): Optional leading batch dimensions.
        noise_std (float): Standard deviation for A's Gaussian noise.
            Pass 0.0 for exact zeros (e.g. for surgery-inserted positions).
        seed (int | None): Random seed for reproducibility.

    Returns:
        tuple[np.ndarray, np.ndarray]: A of shape
            ``batch_shape + (n1, low_rank)`` and B of shape
            ``batch_shape + (n2, low_rank)``.
    """
    assert low_rank >= 1, "low_rank must be at least 1"
    rng = np.random.default_rng(seed)
    A = rng.normal(scale=noise_std,
                   size=batch_shape + (n1, low_rank)).astype(np.float64)
    B = np.zeros(batch_shape + (n2, low_rank), dtype=np.float64)
    return A, B


def flatten_AB(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Flattens the A and B matrices into a single array for use as an
    initializer.

    Args:
        A (np.ndarray): The A matrix of shape (..., n1, low_rank).
        B (np.ndarray): The B matrix of shape (..., n2, low_rank).

    Returns:
        np.ndarray: The flattened array of shape
            (..., n1 * low_rank + n2 * low_rank).
    """
    batch_shape = A.shape[:-2]
    A = np.reshape(A, batch_shape + (-1,))
    B = np.reshape(B, batch_shape + (-1,))
    return np.concatenate([A, B], axis=-1)


def tile_conditional(marginal: np.ndarray, n1: int) -> np.ndarray:
    """Tiles the second marginal along the first marginal dimension to create
    a conditional distribution P(x2 | x1, s) = P(x2 | s).

    Args:
        marginal (Tensor): The second marginal of shape ``(..., D2)``.
        n1 (int): The size of the first marginal.

    Returns:
        Tensor: The tiled conditional distribution of shape
        ``(..., D1 * D2)``.
    """
    return np.tile(marginal, n1)


def outer_product_flat_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Outer product of two arrays with the multiplied dimensions flattened.

    The numpy twin of the emitter's tensor version, for callers that work on
    host arrays anyway (model surgery).

    Args:
        x (np.ndarray): The first array of shape ``(..., D1)``.
        y (np.ndarray): The second array of shape ``(..., D2)``.

    Returns:
        np.ndarray: The outer product of shape ``(..., D1 * D2)``.
    """
    z = np.einsum("...u,...v->...uv", x, y)
    return np.reshape(z, z.shape[:-2] + (z.shape[-2] * z.shape[-1],))


def assert_value_sets(
    marginal_values: Sequence[Sequence[PHMMValueSet]],
) -> None:
    """Validates that the marginal value sets agree in shape."""
    assert len(marginal_values) > 1,\
        "At least two marginal value sets are required."
    assert all(len(marginal_values[0]) == len(mv) for mv in marginal_values),\
        "All marginal value sets must have the same number of heads."
    for h in range(len(marginal_values[0])):
        assert all(marginal_values[0][h].L == mv[h].L for mv in marginal_values),\
            "All marginal value sets must have the same length for each head."
