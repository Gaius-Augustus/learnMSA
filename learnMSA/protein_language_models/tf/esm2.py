import sys
from typing import Sequence

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import tensorflow as tf

from learnMSA.protein_language_models import esm2
from learnMSA.protein_language_models.common import make_cache_dir
from learnMSA.protein_language_models.tf.language_model import TFLanguageModel


class TFESM2LanguageModel(TFLanguageModel):
    """Embeds proteins with the TensorFlow build of ESM-2."""

    #: Token ids, attention mask and crop flags, as
    #: :class:`~learnMSA.protein_language_models.esm2.ESM2InputEncoder` emits
    #: them.
    INPUT_SIGNATURE = (
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
    )

    def __init__(
        self,
        trainable: bool = False,
        small: bool = False,
        cache_dir: str | None = None,
    ) -> None:
        """
        Args:
            trainable: Whether the ESM-2 weights are trainable.
            small: Use the 650M checkpoint instead of the 3B one.
            cache_dir: Where to cache the downloaded checkpoint.
        """
        super().__init__()
        from transformers import TFEsmModel, logging

        logging.set_verbosity_error()
        checkpoint = esm2.checkpoint(small)
        self.model = TFEsmModel.from_pretrained(
            checkpoint,
            cache_dir=make_cache_dir(cache_dir, esm2.CACHE_ID),
        )
        self.model.trainable = trainable
        self.inputs = self.model.inputs
        self.dim = esm2.DIMS[checkpoint]

    @override
    def call(self, inputs: Sequence[tf.Tensor]) -> tf.Tensor:
        ids, mask, crop = inputs
        esm2_output = self.model(ids, mask)
        embeddings = tf.cast(esm2_output.last_hidden_state, tf.float32)
        return self.eliminate_start_stop_tokens(embeddings, crop, mask)
