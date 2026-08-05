"""Reading and writing learnMSA models as keras archives."""

import warnings
from pathlib import Path

import tensorflow as tf

from learnMSA.config import Configuration
from learnMSA.model.checkpoint import apply_runtime_config
# Importing the model module registers TFLearnMSAModel as a custom object,
# which deserialization needs.
from learnMSA.model.tf.model import TFLearnMSAModel

#: Suffix of a saved TensorFlow model.
SUFFIX = ".keras"


def save_model(model: TFLearnMSAModel, filepath: str | Path) -> None:
    """Write the model as a single-file keras archive."""
    model.save(str(filepath) + SUFFIX)


def load_model(
    filepath: str | Path,
    config: Configuration | None = None,
) -> TFLearnMSAModel:
    """Read a model from a keras archive and compile it.

    Args:
        filepath: Path of the archive, without :data:`SUFFIX`.
        config: Configuration of the current run, whose runtime settings
            override the archive's (see :func:`apply_runtime_config`).
    """
    with warnings.catch_warnings():
        # Suppress the compile warning since we manually compile right after
        warnings.filterwarnings(
            'ignore',
            message=".*compile.*was not called as part of model loading.*",
            category=UserWarning
        )
        model = tf.keras.models.load_model(str(filepath) + SUFFIX)

    # The graph mode is decided in compile(), so overriding here still takes
    # effect for this run.
    apply_runtime_config(model.context.config, config)

    # Manually compile the model after loading
    model.compile()
    return model
