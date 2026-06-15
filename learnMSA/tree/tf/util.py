import tensorflow as tf
from packaging import version


def inverse_softplus(features):
    # Cast to float64 to prevent overflow of large entries
    features64 = tf.cast(features, tf.float64)
    result = tf.math.log(tf.math.expm1(features64))
    # Cast back to the original data type of `features`
    return tf.cast(result, features.dtype)


def deserialize(obj):
    if version.parse(tf.__version__) < version.parse("2.11.0"):
        return obj
    else:
        return tf.keras.utils.deserialize_keras_object(obj)
