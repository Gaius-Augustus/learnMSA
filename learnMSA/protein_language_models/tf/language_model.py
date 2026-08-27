import sys

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import tensorflow as tf

from learnMSA.protein_language_models.common import LanguageModel


class TFLanguageModel(tf.keras.layers.Layer, LanguageModel[tf.Tensor]):
    """Base class for the TensorFlow language model wrappers.

    Unlike PyTorch, TensorFlow needs a static input signature to trace a
    ``tf.function`` once instead of once per batch shape. Each concrete wrapper
    therefore declares what its encoder feeds it; nothing about that reaches
    the backend-neutral encoders, which only promise numpy arrays.
    """

    #: Signature of the encoder output this model is traced against, in the
    #: order :meth:`call` unpacks it. ``tf/embed.py`` builds its
    #: ``tf.function`` from it. Annotation without a value: every concrete
    #: wrapper must define it.
    INPUT_SIGNATURE: tuple[tf.TensorSpec, ...]

    @override
    def eliminate_start_stop_tokens(
        self, embeddings: tf.Tensor, crop: tf.Tensor, mask: tf.Tensor
    ) -> tf.Tensor:
        mask = tf.cast(mask, embeddings.dtype)
        mask_crop_1 = tf.concat([mask[:, 1:], tf.zeros_like(mask[:, :1])], 1)
        mask_crop_2 = tf.concat([mask[:, 2:], tf.zeros_like(mask[:, :2])], 1)
        # both tokens
        mask_no_start_stop = (
            mask_crop_2 * (1 - crop[:, :1]) * (1 - crop[:, 1:])
        )
        # only start token
        mask_no_start_stop += mask_crop_1 * crop[:, :1] * (1 - crop[:, 1:])
        # only end token
        mask_no_start_stop += mask_crop_1 * (1 - crop[:, :1]) * crop[:, 1:]
        # no start- or end-token
        mask_no_start_stop += mask * crop[:, :1] * crop[:, 1:]
        # shift sequences with a start token by 1
        embeddings_no_start = tf.concat(
            [embeddings[:, 1:], tf.zeros_like(embeddings[:, :1])], 1
        )
        embeddings_no_start_stop = (
            embeddings_no_start * crop[:, :1, tf.newaxis]
            + embeddings * (1 - crop[:, :1, tf.newaxis])
        )
        embeddings_no_start_stop *= mask_no_start_stop[:, :, tf.newaxis]
        # crop all padding-only columns
        max_len = tf.reduce_max(
            tf.reduce_sum(tf.cast(mask_no_start_stop, tf.int32), -1)
        )
        return embeddings_no_start_stop[:, :max_len]
