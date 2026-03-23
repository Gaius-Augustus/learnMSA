import importlib.resources as resources
from typing import cast

import numpy as np
import tensorflow as tf
from hidten.hmm import HMMConfig as HidtenHMMConfig
from hidten.tf.prior.dirichlet import TFDirichletPrior
from hidten.tf.prior.multivariate_normal import TFMVNormalPrior


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

def make_dirichlet_model(
    initializer : np.ndarray | None = None,
    dim: int | None = None,
    components: int = 1,
) -> tf.keras.Model:
    """Create a keras model with a TFDirichletPrior layer for the amino acid
    prior. If an initializer is provided, it is used to initialize the prior
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
    prior.hmm_config = HidtenHMMConfig(states=[1])
    if initializer is not None:
        prior.initializer = initializer
    else:
        if components == 1:
            prior.initializer = np.ones((n_dim,))
        else:
            prior.initializer = np.ones((components * n_dim + components,))
    prior.build((None, None, None, n_dim))
    return make_model(n_dim, prior)

def load_dirichlet(name: str, dim: int, components: int = 1) -> TFDirichletPrior:
    """Load weights from a resource file into the given model.

    Args:
        name (str): The name of the weight resource file (without extension).
        dim (int): The dimension of the Dirichlet prior.
        components (int): The number of mixture components.
    """
    model = make_dirichlet_model(dim = dim, components = components)
    resource = resources.files("learnMSA.hmm.weights") / f"{name}.h5"
    model.load_weights(str(resource))
    return cast(TFDirichletPrior, model.layers[1])

def make_mvn_model(
    dim: int,
    initializer : np.ndarray | None = None,
    components: int = 1,
) -> tf.keras.Model:
    """Create a keras model with a TFMVNormalPrior layer for the multivariate
    normal prior. If an initializer is provided, it is used to initialize the
    prior distribution.

    Args:
        initializer: Optional initial parameter values.
        dim: The dimension of the observations (means + variances, so 2*D).
        components: The number of mixture components.
    """
    prior = TFMVNormalPrior(components=components)
    prior.hmm_config = HidtenHMMConfig(states=[1])
    if initializer is not None:
        prior.initializer = initializer
    else:
        if components == 1:
            prior.initializer = np.zeros((2 * dim,))
        else:
            prior.initializer = np.zeros((components * 2 * dim + components,))
    prior.build((None, None, 2 * dim))
    return make_model(dim, prior)

def load_mvn(name: str, dim: int, components: int = 1) -> TFMVNormalPrior:
    """Load weights from a resource file into the given model.

    Args:
        name (str): The name of the weight resource file (without extension).
        dim (int): The dimension of the multivariate normal prior.
        components (int): The number of mixture components.
    """
    model = make_mvn_model(dim=dim, components=components)
    resource = resources.files("learnMSA.hmm.weights") / f"{name}.h5"
    model.load_weights(str(resource))
    return cast(TFMVNormalPrior, model.layers[1])
