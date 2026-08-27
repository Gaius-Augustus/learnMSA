import os
import sys
from typing import Sequence

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import numpy as np
import tensorflow as tf

from learnMSA.protein_language_models.common import (InputEncoder,
                                                     make_cache_dir)
from learnMSA.protein_language_models.tf.language_model import TFLanguageModel

#: Name of the download cache subdirectory.
CACHE_ID = "proteinbert"

#: Embedding width of the ProteinBERT hidden layers.
DIM = 1562

#: Token ids below this mark real residues; the rest are special tokens.
FIRST_SPECIAL_TOKEN = 25


class TFProteinBERTLanguageModel(TFLanguageModel):
    """Embeds proteins with ProteinBERT.

    Unlike the other wrappers this one does not construct its own model: the
    pretrained generator has to be loaded first and needs the maximum sequence
    length. Use :func:`make_protein_bert`.
    """

    #: Token ids, global annotations and crop flags, as
    #: :class:`ProteinBERTInputEncoder` emits them.
    INPUT_SIGNATURE = (
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
    )

    def __init__(self, model, trainable: bool = False) -> None:
        """
        Args:
            model: The keras model produced by :func:`make_protein_bert`.
            trainable: Whether the ProteinBERT weights are trainable.
        """
        super().__init__()
        self.model = model
        self.model.trainable = trainable
        self.inputs = model.inputs
        self.dim = DIM

    @override
    def call(self, inputs: Sequence[tf.Tensor]) -> tf.Tensor:
        proteinbert_output = self.model(inputs[:2])
        crop = inputs[2]
        # drop the global annotations, keep the per-residue embeddings
        proteinbert_seq_input, embeddings = inputs[0], proteinbert_output[0]
        # mask start-, end- and padding markers
        mask = tf.cast(
            proteinbert_seq_input < FIRST_SPECIAL_TOKEN, embeddings.dtype
        )
        return self.eliminate_start_stop_tokens(embeddings, crop, mask)


class ProteinBERTInputEncoder(InputEncoder):
    """Tokenizes proteins for ProteinBERT.

    Like ESM-2, ProteinBERT brackets a full protein with a start- and an
    end-token, which are removed again for cropped sequences.
    """

    def __init__(self, input_encoder, max_len: int) -> None:
        """
        Args:
            input_encoder: The ``proteinbert`` encoder produced alongside the
                pretrained model generator.
            max_len: Maximum sequence length the model was built for.
        """
        self.input_encoder = input_encoder
        self.max_len = max_len

    @override
    def __call__(
        self, str_seq: Sequence[str], crop: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        from proteinbert.tokenization import additional_token_to_index

        seq, glob = self.input_encoder.encode_X(str_seq, self.max_len)
        self.modify_cropped(
            seq,
            crop,
            [len(s) for s in str_seq],
            additional_token_to_index["<PAD>"],
        )
        return (
            np.asarray(seq, dtype=np.int32),
            np.asarray(glob, dtype=np.float32),
            np.asarray(crop, dtype=np.float32),
        )


def make_protein_bert(
    max_len: int,
    trainable: bool = False,
    cache_dir: str | None = None,
) -> tuple[TFProteinBERTLanguageModel, ProteinBERTInputEncoder]:
    """Download ProteinBERT and build the wrapper and its encoder.

    Args:
        max_len: Maximum sequence length, including the special tokens.
        trainable: Whether the ProteinBERT weights are trainable.
        cache_dir: Where to cache the downloaded model dump.

    Returns:
        The language model and its input encoder.
    """
    from proteinbert import load_pretrained_model
    from proteinbert.conv_and_global_attention_model import \
        get_model_with_hidden_layers_as_outputs
    from proteinbert.existing_model_loading import \
        DEFAULT_REMOTE_MODEL_DUMP_URL

    generator, input_encoder = load_pretrained_model(
        local_model_dump_dir=make_cache_dir(cache_dir, CACHE_ID),
        local_model_dump_file_name=os.path.basename(
            DEFAULT_REMOTE_MODEL_DUMP_URL
        ),
        download_model_dump_if_not_exists=True,
        validate_downloading=False,
    )
    model = generator.create_model(max_len, compile=False)
    model = get_model_with_hidden_layers_as_outputs(model)
    return (
        TFProteinBERTLanguageModel(model, trainable),
        ProteinBERTInputEncoder(input_encoder, max_len),
    )
