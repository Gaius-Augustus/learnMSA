import gc
from functools import partial
from typing import Callable

import numpy as np

from learnMSA import backend
from learnMSA.config import LanguageModelConfig
from learnMSA.protein_language_models.common import (InputEncoder,
                                                     ScoringModelConfig,
                                                     get_language_model)
from learnMSA.run.util import get_avail_memory_bytes
from learnMSA.util import EmbeddingCache, SequenceDataset

#: Ad hoc scaling constant of the adaptive batch size.
# TODO: make pLM-dependent and fine tune
IMPL_FACTOR = 1000.0


def compute_embeddings(
    data: SequenceDataset,
    language_model_config: LanguageModelConfig,
    verbose: bool = False,
) -> EmbeddingCache:
    """Compute off-the-shelf embeddings for alignments with learnMSA.

    Args:
        data: The sequences to embed. Must have ``remove_gaps=True``.
        language_model_config: Selects the language model and the scoring model
            that reduces its embedding dimension.
        verbose: Print progress.

    Returns:
        An :class:`~learnMSA.util.embedding_cache.EmbeddingCache` that can be
        turned into an ``EmbeddingDataset`` or queried per sequence.
    """
    # TODO: remove the ScoringModelConfig entirely; it's only here for legacy
    # reasons
    scoring_model_config = _get_scoring_model_config(language_model_config)

    # Load the language model and its encoder. The weights are initialized
    # correctly and frozen -- this is inference only.
    language_model, encoder = get_language_model(
        language_model_config.language_model,
        max_len=data.max_len + 2,
        trainable=False,
        cache_dir=language_model_config.plm_cache_dir,
        embedding_dim=scoring_model_config.dim,
    )

    # The scoring model reduces the language model's embedding dimension. The
    # "zeros" stand-in already emits the reduced width, so it needs none.
    # TODO: remove scoring model config and make the whole codebase use
    # the language model config instead
    if language_model_config.language_model == "zeros":
        reduction_layer = None
    else:
        make_reduction_layer = backend.resolve(
            "protein_language_models.bilinear_symmetric",
            "make_reduction_layer",
        )
        reduction_layer = make_reduction_layer(scoring_model_config)

    make_embedding_fn = backend.resolve(
        "protein_language_models.embed", "make_embedding_fn"
    )
    embedding_fn = make_embedding_fn(language_model, reduction_layer)

    cache = EmbeddingCache(
        data.seq_lens, language_model_config.scoring_model_dim
    )
    compute_emb_func = partial(
        _compute_reduced_embeddings,
        data=data,
        encoder=encoder,
        embedding_fn=embedding_fn,
    )
    batch_size_callback = partial(
        get_adaptive_batch_size, impl_factor=IMPL_FACTOR
    )

    if verbose:
        print(
            f"Computing embeddings for {len(data)} sequences. "
            "This may take a moment..."
        )

    cache.fill_cache(compute_emb_func, batch_size_callback, verbose=verbose)

    # Erase the language model from memory again; it is by far the largest
    # thing this process holds and nothing downstream needs it.
    del language_model
    del encoder
    del reduction_layer
    del embedding_fn
    backend.clear_session()
    gc.collect()

    return cache


def _compute_reduced_embeddings(
    indices: np.ndarray,
    data: SequenceDataset,
    encoder: InputEncoder,
    embedding_fn: Callable[[tuple[np.ndarray, ...]], np.ndarray],
) -> np.ndarray:
    """Embed the sequences at ``indices`` and reduce their dimension."""
    assert data.remove_gaps, \
        "Embeddings can only be computed for datasets with remove_gaps=True"
    seq_batch = [data.get_standardized_seq(i) for i in indices]
    lm_inputs = encoder(
        seq_batch, np.repeat([[False, False]], len(seq_batch), axis=0)
    )
    return embedding_fn(lm_inputs)


def get_adaptive_batch_size(
    seq_len: int, impl_factor: float = 1.0, safety_margin: float = 0.75
) -> int:
    """Compute a batch size that fits into the available memory.

    Args:
        seq_len: Length of the longest sequence in the batch.
        impl_factor: Implementation-dependent memory scaling constant.
        safety_margin: Fraction of the available memory to actually use.

    Returns:
        A batch size in ``[1, 1024]``.
    """
    mem_avail = get_avail_memory_bytes()
    denominator = float(seq_len) ** 2  # pLMs scale quadratically in length
    denominator *= impl_factor
    if denominator <= 0.0:
        return 1
    batch_size = int(np.floor(safety_margin * mem_avail / denominator))
    # cap batch size to avoid OOM from edge cases
    return min(max(batch_size, 1), 1024)


def _get_scoring_model_config(
    language_model_config: LanguageModelConfig,
) -> ScoringModelConfig:
    """Derive the legacy scoring model config from the user-facing config."""
    return ScoringModelConfig(
        lm_name=language_model_config.language_model,
        dim=language_model_config.scoring_model_dim,
        activation=language_model_config.scoring_model_activation,
        suffix=language_model_config.scoring_model_suffix,
        scaled=False,
    )
