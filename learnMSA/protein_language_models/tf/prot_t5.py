import sys
from typing import Sequence

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import tensorflow as tf

from learnMSA.protein_language_models import prot_t5
from learnMSA.protein_language_models.common import make_cache_dir
from learnMSA.protein_language_models.tf.language_model import TFLanguageModel


class TFProtT5LanguageModel(TFLanguageModel):
    """Embeds proteins with the TensorFlow build of the ProtT5 encoder."""

    #: Token ids and attention mask, as
    #: :class:`~learnMSA.protein_language_models.prot_t5.ProtT5InputEncoder`
    #: emits them. No crop flags: ProtT5 uses a relative position embedding.
    INPUT_SIGNATURE = (
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    )

    def __init__(
        self,
        trainable: bool = False,
        dtype: tf.DType = tf.float16,
        cache_dir: str | None = None,
    ) -> None:
        """
        Args:
            trainable: Whether the ProtT5 weights are trainable.
            dtype: Compute dtype. The shipped checkpoint is half precision.
            cache_dir: Where to cache the downloaded checkpoint.
        """
        super().__init__(dtype=dtype)
        from transformers import TFT5EncoderModel, logging

        logging.set_verbosity_error()
        self.model = TFT5EncoderModel.from_pretrained(
            prot_t5.MODEL_CHECKPOINT,
            from_pt=True,
            cache_dir=make_cache_dir(cache_dir, prot_t5.CACHE_ID),
        )
        self.model.trainable = trainable
        self.inputs = self.model.inputs
        self.dim = prot_t5.DIM

    @override
    def call(self, inputs: Sequence[tf.Tensor]) -> tf.Tensor:
        ids, mask = inputs[0], inputs[1]
        protT5_output = self.model(ids, mask)
        # ProtT5 appends a single end-token; drop it, and do not count it in
        # the mask either. There is no start-token, so the shared
        # eliminate_start_stop_tokens is not needed here.
        embeddings = tf.cast(
            protT5_output.last_hidden_state[:, :-1], self.dtype
        )
        mask = mask[:, 1:]
        max_len = tf.reduce_max(tf.reduce_sum(mask, -1))
        mask = tf.cast(tf.expand_dims(mask, -1), self.dtype)
        return (embeddings * mask)[:, :max_len]
