"""Small paired amino-acid / embedding datasets used by more than one suite.

``tests/util/test_embedding_dataset.py`` exercises the dataset itself and the
two backend ``test_training.py`` files feed it through ``make_dataset``, so the
builders live here rather than being copied three times. Plain functions rather
than fixtures, because the callers sit in different directories and would
otherwise need a conftest each.
"""

import numpy as np

from learnMSA.util import EmbeddingCache, EmbeddingDataset, SequenceDataset

#: Sequence lengths of :func:`make_aa_dataset`, in order.
SEQ_LENS = np.array([5, 11, 17, 4, 5])

#: Width of the embedding vectors :func:`make_embedding_dataset` produces.
EMBEDDING_DIM = 8


def make_embedding_dataset() -> EmbeddingDataset:
    """An embedding dataset whose i-th sequence is filled with ``i + 1``.

    The constant per-sequence value makes a mis-routed row obvious.
    """
    rows = [
        (i + 1) * np.ones((length, EMBEDDING_DIM), dtype=np.float32)
        for i, length in enumerate(SEQ_LENS)
    ]
    cache = EmbeddingCache(
        SEQ_LENS, EMBEDDING_DIM, cache=np.concatenate(rows, axis=0)
    )
    return EmbeddingDataset(
        embedding_cache=cache,
        seq_ids=[f"seq_{i}" for i in range(len(SEQ_LENS))],
    )


def make_aa_dataset() -> SequenceDataset:
    """The amino acid dataset matching :data:`SEQ_LENS`."""
    return SequenceDataset(
        sequences=[
            ("seq1", "ACDEG"),
            ("seq2", "ACDEFGHIKLM"),
            ("seq3", "ACDEFGHIKLMNPQRSM"),
            ("seq4", "ACDE"),
            ("seq5", "ACDEF"),
        ]
    )
