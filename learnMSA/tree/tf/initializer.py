"""TensorFlow initializers and the bridge from backend-neutral init specs.

The neutral :class:`~learnMSA.tree.initializer.InitSpec` family describes what a
parameter starts at; :func:`to_tf` turns such a description into the keras
initializer that the TF layers need.
"""

import numpy as np
import tensorflow as tf

from learnMSA.tree import initializer as spec
# Re-exported for callers that still build the numeric initial values directly.
from learnMSA.tree.initializer import make_substitution_model_init  # noqa: F401


class ConstantInitializer(tf.keras.initializers.Constant):

    def __init__(self, value):
        super(ConstantInitializer, self).__init__(value)

    def __repr__(self):
        if np.isscalar(self.value):
            return f"Const({self.value})"
        elif isinstance(self.value, list):
            return f"Const(size={len(self.value)})"
        else:
            return f"Const(shape={self.value.shape})"

    def get_config(self):  # To support serialization
        if isinstance(self.value, np.ndarray):
            value = self.value.tolist()
        else:
            value = self.value
        return {"value": value}

    @classmethod
    def from_config(cls, config):
        return cls(np.array(config["value"]))


def to_tf(
    initializer: "spec.InitSpec | tf.keras.initializers.Initializer",
) -> tf.keras.initializers.Initializer:
    """Materialize a neutral init spec as a keras initializer.

    Keras initializers are passed through unchanged, so layers can accept either
    form.
    """
    if isinstance(initializer, spec.Constant):
        return ConstantInitializer(initializer.value)
    if isinstance(initializer, spec.RandomNormal):
        return tf.keras.initializers.RandomNormal(
            mean=initializer.mean, stddev=initializer.stddev
        )
    if isinstance(initializer, spec.Zeros):
        return tf.keras.initializers.Zeros()
    if isinstance(initializer, spec.InitSpec):
        raise TypeError(
            f"No TensorFlow initializer is defined for {type(initializer).__name__}."
        )
    return initializer
