import gc
import os
from functools import partial

import numpy as np
import tensorflow as tf

from learnMSA.model.context import LearnMSAContext
from learnMSA.protein_language_models.bilinear_symmetric import \
    make_scoring_model
from learnMSA.protein_language_models.common import (InputEncoder,
                                                     LanguageModel,
                                                     get_language_model,
                                                     get_scoring_model_path)
from learnMSA.run.util import get_avail_memory_bytes
from learnMSA.util import EmbeddingCache, SequenceDataset


def compute_embeddings(
    data: SequenceDataset,
    context: LearnMSAContext,
    verbose: bool=False,
) -> EmbeddingCache:
    """
    Computes off-the-shelf embeddings for alignments with learnMSA.
    """
    # load the language model and the scoring model
    # initialize the weights correctly and make sure they are not trainable
    language_model, encoder = get_language_model(
        context.config.language_model.language_model,
        max_len = data.max_len+2,
        trainable=False,
        cache_dir=context.config.language_model.plm_cache_dir,
    )

    # Load the scoring model and its weights.
    # The scoring model is used to reduce the embedding dimension.
    # TODO: remove scoring model config and make the whole codebase use
    # the language model config instead
    scoring_model = make_scoring_model(
        context.scoring_model_config, dropout=0.0, trainable=False
    )
    scoring_model_path = get_scoring_model_path(context.scoring_model_config)
    scoring_model.load_weights(
        os.path.dirname(__file__)
        + f"/../protein_language_models/"
        + scoring_model_path
    )
    scoring_layer = scoring_model.layers[-1]
    scoring_layer.trainable = False #don't forget to freeze the scoring model!

    cache = EmbeddingCache(data.seq_lens, context.scoring_model_config.dim)
    compute_emb_func = partial(
        _compute_reduced_embeddings,
        data = data,
        language_model = language_model,
        encoder = encoder,
        scoring_layer = scoring_layer,
    )
    impl_factor = 50.0 # TODO: ad hoc; make pLM-dependent and fine tune
    batch_size_callback = partial(
        get_adaptive_batch_size, impl_factor=impl_factor
    )

    cache.fill_cache(compute_emb_func, batch_size_callback, verbose=verbose)
    # once we have cached the embeddings do a cleanup to erase the LM from memory
    tf.keras.backend.clear_session()
    gc.collect()

    return cache


@tf.function
def _call_lm_scoring_model(lm_inputs, language_model, scoring_layer):
    emb = language_model(lm_inputs)
    reduced_emb = scoring_layer._reduce(emb, training=False)
    return reduced_emb


def _compute_reduced_embeddings(
    indices: np.ndarray,
    data: SequenceDataset,
    language_model: LanguageModel,
    encoder: InputEncoder,
    scoring_layer,
) -> np.ndarray:
    seq_batch = [data.get_standardized_seq(i) for i in indices]
    lm_inputs = encoder(
        seq_batch, np.repeat([[False, False]], len(seq_batch), axis=0)
    )
    return _call_lm_scoring_model(
        lm_inputs, language_model, scoring_layer
    ).numpy()


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