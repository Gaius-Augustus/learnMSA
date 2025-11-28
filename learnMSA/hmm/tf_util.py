import importlib.resources as resources
from typing import cast

import numpy as np
import tensorflow as tf
from hidten.hmm import HMMConfig as HidtenHMMConfig
from hidten.tf.prior.dirichlet import TFDirichletPrior


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
    initializer : np.ndarray | None = None, dim: int | None = None
) -> tf.keras.Model:
    """Create a keras model with a TFDirichletPrior layer for the amino acid
    prior. If an initializer is provided, it is used to initialize the prior
    distribution."""
    assert (initializer is not None) != (dim is not None),\
        "Either initializer or dim must be provided."
    if initializer is not None:
        n_dim: int = int(initializer.shape[0])
    else:
        assert dim is not None
        n_dim = int(dim)
    prior = TFDirichletPrior()
    prior.hmm_config = HidtenHMMConfig(states=[1])
    prior.initializer = initializer if initializer is not None else np.ones((n_dim,))
    prior.build((None, None, None, n_dim))
    return make_model(n_dim, prior)

def load_dirichlet(name: str, dim: int) -> TFDirichletPrior:
    """Load weights from a resource file into the given model.

    Args:
        name (str): The name of the weight resource file (without extension).
        dim (int): The dimension of the Dirichlet prior.
    """
    model = make_dirichlet_model(dim = dim)
    resource = resources.files("learnMSA.hmm.weights") / f"{name}.h5"
    model.load_weights(str(resource))
    return cast(TFDirichletPrior, model.layers[1])
