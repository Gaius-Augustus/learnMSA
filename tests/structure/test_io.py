"""Telling 3Di logit files apart from other npz files (numpy only)."""

import numpy as np

from learnMSA.structure.io import KIND, is_logits_file
from learnMSA.util import EmbeddingCache, EmbeddingDataset


def _dataset() -> EmbeddingDataset:
    seq_lens = np.array([3, 5, 2])
    cache = np.zeros((seq_lens.sum(), 4), dtype=np.float16)
    return EmbeddingDataset(
        embedding_cache=EmbeddingCache(seq_lens, 4, cache=cache),
        seq_ids=["a", "b", "c"],
    )


def test_logits_files_are_recognized(tmp_path) -> None:
    path = _dataset().write(tmp_path / "x", metadata={"kind": KIND})
    assert path.name == "x.npz" and is_logits_file(path)


def test_embedding_files_are_not_logits(tmp_path) -> None:
    assert not is_logits_file(_dataset().write(tmp_path / "emb.npz"))
    path = tmp_path / "other.npz"
    np.savez(path, cache=np.zeros((2, 4)))
    assert not is_logits_file(path)
    assert not is_logits_file(tmp_path / "missing.npz")
