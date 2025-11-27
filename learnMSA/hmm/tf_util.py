import importlib.resources as resources
import tempfile

import numpy as np
import tensorflow as tf
from hidten.hmm import HMMConfig as HidtenHMMConfig
from hidten.tf.prior.dirichlet import TFDirichletPrior


def make_model(layer : tf.keras.layers.Layer) -> tf.keras.Model:
    """Utility function that constructs a keras model over a layer for
    serialization.
    """
    input = tf.keras.Input(
        shape=(None, 23,), dtype=layer.dtype
    )
    loglik = layer(input)
    model = tf.keras.Model(inputs=[input], outputs=[loglik])
    return model

def make_dirichlet_model(initializer : np.ndarray | None = None) -> tf.keras.Model:
    prior = TFDirichletPrior()
    prior.hmm_config = HidtenHMMConfig(states=[1])
    prior.initializer = initializer if initializer is not None else np.ones((23,))
    prior.build((None, None, None, 23))
    return make_model(prior)


def load_weight_resource(model: tf.keras.Model, name: str) -> None:
    resource = resources.files("learnMSA.hmm.weights") / f"{name}.h5"
    model.load_weights(str(resource))
