import gc
import os
from functools import partial

import numpy as np
import tensorflow as tf

from learnMSA.model.context import LearnMSAContext
from learnMSA.protein_language_models import common
from learnMSA.protein_language_models.bilinear_symmetric import \
    make_scoring_model
from learnMSA.protein_language_models.common import (InputEncoder,
                                                     LanguageModel,
                                                     get_language_model,
                                                     get_scoring_model_path)
from learnMSA.run.util import get_avail_memory_bytes
from learnMSA.util import EmbeddingCache, SequenceDataset
from learnMSA.config import LanguageModelConfig


def compute_embeddings(
    data: SequenceDataset,
    language_model_config : LanguageModelConfig,
    verbose: bool=False,
) -> EmbeddingCache:
    """
    Computes off-the-shelf embeddings for alignments with learnMSA.

    Returns an EmbeddingCache object that can turned into an
    EmbeddingDataset object or it can be used to retrieve the
    embeddings for each sequence in the dataset.
    """
    # TODO: remove the ScoringModelConfig entirely; it's only here for legacy
    # reasons
    scoring_model_config = _get_scoring_model_config(language_model_config)

    # load the language model and the scoring model
    # initialize the weights correctly and make sure they are not trainable
    language_model, encoder = get_language_model(
        language_model_config.language_model,
        max_len = data.max_len+2,
        trainable=False,
        cache_dir=language_model_config.plm_cache_dir,
        embedding_dim=scoring_model_config.dim,
    )

    # Load the scoring model and its weights.
    # The scoring model is used to reduce the embedding dimension.
    # TODO: remove scoring model config and make the whole codebase use
    # the language model config instead
    if language_model_config.language_model == "zeros":
        scoring_layer = None
    else:
        scoring_model = make_scoring_model(
            scoring_model_config, dropout=0.0, trainable=False
        )
        scoring_model_path = get_scoring_model_path(scoring_model_config)
        scoring_model.load_weights(
            os.path.dirname(__file__)
            + f"/../protein_language_models/"
            + scoring_model_path
        )
        scoring_layer = scoring_model.layers[-1]
        scoring_layer.trainable = False #don't forget to freeze the scoring model!

    cache = EmbeddingCache(data.seq_lens, language_model_config.scoring_model_dim)
    lm_scoring_call = _make_lm_scoring_call(language_model, encoder, scoring_layer)
    compute_emb_func = partial(
        _compute_reduced_embeddings,
        data = data,
        encoder = encoder,
        lm_scoring_call = lm_scoring_call,
    )
    impl_factor = 1000.0 # TODO: ad hoc; make pLM-dependent and fine tune
    batch_size_callback = partial(
        get_adaptive_batch_size, impl_factor=impl_factor
    )

    if verbose:
        print(
            f"Computing embeddings for {len(data)} sequences." \
            "This may take a moment...")

    cache.fill_cache(compute_emb_func, batch_size_callback, verbose=verbose)

    # cleanup to erase the LM from memory
    tf.keras.backend.clear_session()
    gc.collect()

    return cache


def _make_lm_scoring_call(language_model, encoder, scoring_layer):
    @tf.function(input_signature=(encoder.get_signature(),))
    def _call_lm_scoring_model(lm_inputs):
        emb = language_model(lm_inputs)
        if scoring_layer is None:
            return emb
        return scoring_layer._reduce(emb, training=False)

    return _call_lm_scoring_model


def _compute_reduced_embeddings(
    indices: np.ndarray,
    data: SequenceDataset,
    encoder: InputEncoder,
    lm_scoring_call,
) -> np.ndarray:
    seq_batch = [data.get_standardized_seq(i) for i in indices]
    lm_inputs = encoder(
        seq_batch, np.repeat([[False, False]], len(seq_batch), axis=0)
    )
    return lm_scoring_call(lm_inputs).numpy()


def get_adaptive_batch_size(
    seq_len: int, impl_factor: float = 1.0, safety_margin: float = 0.8
) -> int:
    """
    Computes an adaptive batch size.
    """
    mem_avail = get_avail_memory_bytes()
    denominator = float(seq_len) ** 2 # accounts for quadratic scaling of pLMs
    denominator *= impl_factor
    if denominator <= 0.0:
        return 1
    batch_size = int(np.floor(safety_margin * mem_avail / denominator))
    # cap batch size to avoid OOM from edge cases
    return min(max(batch_size, 1), 1024)


def _get_scoring_model_config(
    language_model_config : LanguageModelConfig
) -> common.ScoringModelConfig:
    scoring_model_config = common.ScoringModelConfig(
        lm_name=language_model_config.language_model,
        dim=language_model_config.scoring_model_dim,
        activation=language_model_config.scoring_model_activation,
        suffix=language_model_config.scoring_model_suffix,
        scaled=False
    )
    return scoring_model_config
