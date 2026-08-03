from typing import Sequence

import numpy as np
import tensorflow as tf
from hidten.hmm import HMMConfig as HidtenHMMConfig
from hidten.tf.prior.dirichlet import TFDirichletPrior
from hidten.tf.prior.multivariate_normal import TFMVNormalPrior

from learnMSA.hmm.priors import load_prior_kernel, warn_if_degenerate
from learnMSA.util.tensor import to_numpy


def make_model(dim: int, layer : tf.keras.layers.Layer) -> tf.keras.Model:
    """Utility function that constructs a keras model over a layer for
    serialization.
    """
    input = tf.keras.Input(
        shape=(None, dim,), dtype=layer.dtype
    )
    loglik = layer(input)
    model = tf.keras.Model(inputs=[input], outputs=[loglik])
    return model

def make_dirichlet_prior(
    initializer : np.ndarray | None = None,
    dim: int | None = None,
    components: int = 1,
    states: Sequence[int] = [1],
) -> TFDirichletPrior:
    """Create and build a :class:`TFDirichletPrior` for the amino acid prior.
    If an initializer is provided, it is used to initialize the prior
    distribution.

    For multi-component priors, ``dim`` must be provided explicitly since
    the initializer length encodes both components and categories.
    """
    assert initializer is not None or dim is not None,\
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
    prior = TFDirichletPrior(components=components)
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

def make_dirichlet_model(
    initializer : np.ndarray | None = None,
    dim: int | None = None,
    components: int = 1,
    states: Sequence[int] = [1],
) -> tf.keras.Model:
    """Wrap a Dirichlet prior in a keras model, for the prior fitting tooling."""
    prior = make_dirichlet_prior(
        initializer=initializer, dim=dim, components=components, states=states
    )
    return make_model(prior.input_dim, prior)

def load_dirichlet(
    name: str, dim: int, components: int = 1, states: Sequence[int] = [1]
) -> TFDirichletPrior:
    """Load a shipped Dirichlet prior.

    The parameters are read as a plain numpy array (see
    :func:`learnMSA.hmm.priors.load_prior_kernel`) and assigned to the built
    layer, so no keras model has to be constructed to deserialize them.

    Args:
        name (str): The name of the parameter resource (without extension).
        dim (int): The dimension of the Dirichlet prior.
        components (int): The number of mixture components.
        states (Sequence[int]): The number of states in each head.
    """
    prior = make_dirichlet_prior(
        dim = dim, components = components, states = states
    )
    _assign_kernel(prior, load_prior_kernel(name))
    _warn_if_degenerate(prior, name)
    return prior


def _assign_kernel(prior, kernel: np.ndarray) -> None:
    """Write a numpy parameter kernel into a built prior layer."""
    prior.kernel.assign(np.asarray(kernel, dtype=prior.kernel.dtype))


def _warn_if_degenerate(prior: TFDirichletPrior, name: str) -> None:
    """Extract the concentrations and hand them to the neutral check."""
    matrix = prior.matrix()
    if prior.config.components > 1:
        # Drop the trailing mixture coefficients.
        matrix = prior._slice_concentrations(matrix)
    alpha = to_numpy(matrix).reshape(-1, prior.input_dim)
    warn_if_degenerate(alpha, name)

def make_mvn_prior(
    dim: int,
    initializer : np.ndarray | None = None,
    components: int = 1,
    states: Sequence[int] = [1],
) -> TFMVNormalPrior:
    """Create and build a :class:`TFMVNormalPrior`. If an initializer is
    provided, it is used to initialize the prior distribution.

    Args:
        initializer: Optional initial parameter values.
        dim: The dimension of the observations (means + variances, so 2*D).
        components: The number of mixture components.
        states: The number of states in each head.
    """
    prior = TFMVNormalPrior(components=components)
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

def make_mvn_model(
    dim: int,
    initializer : np.ndarray | None = None,
    components: int = 1,
    states: Sequence[int] = [1],
) -> tf.keras.Model:
    """Wrap a multivariate normal prior in a keras model."""
    prior = make_mvn_prior(
        dim=dim, initializer=initializer, components=components, states=states
    )
    return make_model(dim, prior)

def load_mvn(
    name: str, dim: int, components: int = 1, states: Sequence[int] = [1]
) -> TFMVNormalPrior:
    """Load a shipped multivariate normal prior.

    Args:
        name (str): The name of the parameter resource (without extension).
        dim (int): The dimension of the multivariate normal prior.
        components (int): The number of mixture components.
        states (Sequence[int]): The number of states in each head.
    """
    prior = make_mvn_prior(dim=dim, components=components, states=states)
    _assign_kernel(prior, load_prior_kernel(name))
    return prior
