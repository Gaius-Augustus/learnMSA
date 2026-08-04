"""Constructing and loading the shipped priors as PyTorch modules.

The counterpart of :mod:`learnMSA.hmm.tf.util`. Both build the same prior with
the same shared-parameter layout and then write a plain numpy kernel into it;
the parameter files themselves are backend-neutral ``.npz`` archives read by
:mod:`learnMSA.hmm.priors`.
"""

from collections.abc import Sequence

import numpy as np
import torch
from hidten.hmm import HMMConfig as HidtenHMMConfig
from hidten.torch.prior.dirichlet import TorchDirichletPrior
from hidten.torch.prior.multivariate_normal import TorchMVNormalPrior

from learnMSA.hmm.priors import load_prior_kernel, warn_if_degenerate
from learnMSA.util.tensor import to_numpy


def sequence_mask(
    lengths: np.ndarray,
    maxlen: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """The equivalent of ``tf.sequence_mask``, as a ``(H, maxlen)`` tensor."""
    positions = torch.arange(maxlen, device=device)
    lengths_t = torch.as_tensor(
        np.asarray(lengths), dtype=torch.int64, device=device
    )
    return (positions[None, :] < lengths_t[:, None]).to(dtype)


def insertion_expansion_indices(lengths: np.ndarray) -> np.ndarray:
    """Indices that expand one insertion score per head into one per state.

    An emitter that skips the full matrix multiplication scores only
    ``L`` match states plus a *single* insertion state per head. The HMM
    however expects a score for every state, i.e. ``L`` matches followed by
    ``L + 2`` insertions, plus padding up to the longest head. Gathering with
    these indices performs that expansion in one op.

    Args:
        lengths: The number of match states per head.

    Returns:
        A flat index array into the ``(H * Q)`` reshaped score tensor.
    """
    indices: list[int] = []
    max_length = lengths.max()
    offset = 0
    for length in lengths:
        # Match states: copy once each
        indices.extend(range(offset, offset + length))
        # Insertion state: repeat L+2 times
        indices.extend([offset + length] * (length + 2))
        offset += length + 1  # Move to next head (L matches + 1 insert)
        # Padding states
        if length < max_length:
            padding = max_length - length
            indices.extend([offset] * (padding + 1))  # repeat first padding
            indices.extend(range(offset + 1, offset + padding))  # the rest
            offset += padding
    return np.asarray(indices, dtype=np.int64)


def make_dirichlet_prior(
    initializer: np.ndarray | None = None,
    dim: int | None = None,
    components: int = 1,
    states: Sequence[int] = [1],
) -> TorchDirichletPrior:
    """Create and build a :class:`TorchDirichletPrior` for the amino acid
    prior. If an initializer is provided, it is used to initialize the prior
    distribution.

    For multi-component priors, ``dim`` must be provided explicitly since
    the initializer length encodes both components and categories.
    """
    assert initializer is not None or dim is not None, \
        "Either initializer or dim must be provided."
    if dim is not None:
        n_dim = int(dim)
    elif components == 1:
        assert initializer is not None
        n_dim = int(initializer.shape[0])
    else:
        raise ValueError(
            "dim must be provided for multi-component Dirichlet priors."
        )
    prior = TorchDirichletPrior(components=components)
    prior.hmm_config = HidtenHMMConfig(states=states)

    n_param = components * n_dim + components if components > 1 else n_dim

    # Configure parameters
    if initializer is not None:
        prior.initializer = initializer
    else:
        prior.initializer = np.ones((n_param,))

    # Share concentrations across all states
    prior.share = np.tile(np.arange(n_param), reps=sum(states))

    prior.build((None, None, None, n_dim))
    return prior


def load_dirichlet(
    name: str, dim: int, components: int = 1, states: Sequence[int] = [1]
) -> TorchDirichletPrior:
    """Load a shipped Dirichlet prior.

    Args:
        name: The name of the parameter resource (without extension).
        dim: The dimension of the Dirichlet prior.
        components: The number of mixture components.
        states: The number of states in each head.
    """
    prior = make_dirichlet_prior(
        dim=dim, components=components, states=states
    )
    _assign_kernel(prior, load_prior_kernel(name))
    _warn_if_degenerate(prior, name)
    return prior


def _assign_kernel(prior, kernel: np.ndarray) -> None:
    """Write a numpy parameter kernel into a built prior module."""
    with torch.no_grad():
        prior.kernel.copy_(
            torch.as_tensor(kernel, dtype=prior.kernel.dtype)
        )


def _warn_if_degenerate(prior: TorchDirichletPrior, name: str) -> None:
    """Extract the concentrations and hand them to the neutral check."""
    with torch.no_grad():
        matrix = prior.matrix()
        if prior.config.components > 1:
            # Drop the trailing mixture coefficients.
            matrix = prior._slice_concentrations(matrix)
    alpha = to_numpy(matrix).reshape(-1, prior.input_dim)
    warn_if_degenerate(alpha, name)


def make_mvn_prior(
    dim: int,
    initializer: np.ndarray | None = None,
    components: int = 1,
    states: Sequence[int] = [1],
) -> TorchMVNormalPrior:
    """Create and build a :class:`TorchMVNormalPrior`. If an initializer is
    provided, it is used to initialize the prior distribution.

    Args:
        dim: The dimension of the observations (means + variances, so 2*D).
        initializer: Optional initial parameter values.
        components: The number of mixture components.
        states: The number of states in each head.
    """
    prior = TorchMVNormalPrior(components=components)
    prior.hmm_config = HidtenHMMConfig(states=states)

    n_param = components * 2 * dim + components if components > 1 else 2 * dim

    # Configure parameters
    if initializer is not None:
        prior.initializer = initializer
    else:
        prior.initializer = np.zeros((n_param,))

    # Share parameters across all states
    prior.share = np.tile(np.arange(n_param), reps=sum(states))

    prior.build((None, None, 2 * dim))
    return prior


def load_mvn(
    name: str, dim: int, components: int = 1, states: Sequence[int] = [1]
) -> TorchMVNormalPrior:
    """Load a shipped multivariate normal prior.

    Args:
        name: The name of the parameter resource (without extension).
        dim: The dimension of the multivariate normal prior.
        components: The number of mixture components.
        states: The number of states in each head.
    """
    prior = make_mvn_prior(dim=dim, components=components, states=states)
    _assign_kernel(prior, load_prior_kernel(name))
    return prior
