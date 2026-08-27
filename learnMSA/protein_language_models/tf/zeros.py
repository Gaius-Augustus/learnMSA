import sys
from typing import Sequence

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import tensorflow as tf

from learnMSA.protein_language_models.tf.language_model import TFLanguageModel


class TFZerosLanguageModel(TFLanguageModel):
    """Emits constant zero embeddings of a configurable width."""

    #: Token ids and padding mask, as
    #: :class:`~learnMSA.protein_language_models.zeros.ZerosInputEncoder`
    #: emits them. Only the mask is read.
    INPUT_SIGNATURE = (
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    )

    def __init__(
        self, embedding_dim: int, dtype: tf.DType = tf.float32
    ) -> None:
        """
        Args:
            embedding_dim: Width of the emitted embeddings.
            dtype: Dtype of the emitted embeddings.
        """
        super().__init__(dtype=dtype)
        self.dim = int(embedding_dim)

    @override
    def call(self, inputs: Sequence[tf.Tensor]) -> tf.Tensor:
        _, mask = inputs
        seq_lens = tf.reduce_sum(tf.cast(mask, tf.int32), axis=-1)
        max_len = tf.reduce_max(seq_lens)
        return tf.zeros(
            (tf.shape(mask)[0], max_len, self.dim), dtype=self.dtype
        )
