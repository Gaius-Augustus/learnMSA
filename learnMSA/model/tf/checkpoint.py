"""Reading and writing learnMSA models as keras archives."""

import warnings
from pathlib import Path

import tensorflow as tf

# Importing the model module registers TFLearnMSAModel as a custom object,
# which deserialization needs.
from learnMSA.model.tf.model import TFLearnMSAModel

#: Suffix of a saved TensorFlow model.
SUFFIX = ".keras"


def save_model(model: TFLearnMSAModel, filepath: str | Path) -> None:
    """Write the model as a single-file keras archive."""
    model.save(str(filepath) + SUFFIX)


def load_model(filepath: str | Path) -> TFLearnMSAModel:
    """Read a model from a keras archive and compile it."""
    with warnings.catch_warnings():
        # Suppress the compile warning since we manually compile right after
        warnings.filterwarnings(
            'ignore',
            message=".*compile.*was not called as part of model loading.*",
            category=UserWarning
        )
        model = tf.keras.models.load_model(str(filepath) + SUFFIX)

    # Manually compile the model after loading
    model.compile()
    return model
