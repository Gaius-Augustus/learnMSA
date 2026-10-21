from typing import Callable

import numpy as np
import tensorflow as tf

from learnMSA.protein_language_models.tf.language_model import TFLanguageModel


def make_embedding_fn(
    language_model: TFLanguageModel,
    reduction_layer=None,
) -> Callable[[tuple[np.ndarray, ...]], np.ndarray]:
    """Build the compiled embed-and-reduce call.

    Args:
        language_model: The wrapped language model. Its ``INPUT_SIGNATURE``
            is what the traced function is specialized on.
        reduction_layer: Projects the embeddings onto the scoring model's
            reduced dimension. ``None`` leaves them unreduced.

    Returns:
        A callable mapping the encoder's output to a numpy array of shape
        ``(batch, max_len, dim)``.
    """

    @tf.function(input_signature=(language_model.INPUT_SIGNATURE,))
    def _call(lm_inputs):
        emb = language_model(lm_inputs)
        if reduction_layer is None:
            return emb
        return reduction_layer.reduce(emb, training=False)

    def embed(lm_inputs: tuple[np.ndarray, ...]) -> np.ndarray:
        return _call(tuple(lm_inputs)).numpy()

    return embed
