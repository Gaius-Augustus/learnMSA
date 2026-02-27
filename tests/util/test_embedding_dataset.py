import numpy as np
import pytest
from pathlib import Path

from learnMSA.util.embedding_cache import EmbeddingCache
from learnMSA.util.embedding_dataset import EmbeddingDataset


def _make_dataset():
    """Helper: build a small EmbeddingDataset from scratch."""
    seq_lens = np.array([5, 11, 17, 4, 5])
    dim = 8
    rows = []
    for i, L in enumerate(seq_lens):
        rows.append((i + 1) * np.ones((L, dim), dtype=np.float32))
    cache_array = np.concatenate(rows, axis=0)
    cache = EmbeddingCache(seq_lens, dim, cache=cache_array)
    seq_ids = [f"seq_{i}" for i in range(len(seq_lens))]
    ds = EmbeddingDataset(embedding_cache=cache, seq_ids=seq_ids)
    return ds


def test_write_and_read_roundtrip(tmp_path: Path) -> None:
    """Write an EmbeddingDataset to .emb and read it back."""
    ds = _make_dataset()
    emb_path = tmp_path / "test.emb"
    ds.write(emb_path)

    loaded = EmbeddingDataset(filepath=emb_path)
    assert loaded.parsing_ok
    assert loaded.num_seq == ds.num_seq
    assert loaded.max_len == ds.max_len
    assert loaded.seq_ids == ds.seq_ids
    np.testing.assert_array_equal(loaded.seq_lens, ds.seq_lens)
    np.testing.assert_array_equal(loaded._permutation, ds._permutation)
    np.testing.assert_array_equal(
        loaded._embedding_cache.cache, ds._embedding_cache.cache
    )
    for i in range(ds.num_seq):
        np.testing.assert_array_equal(
            loaded.get_encoded_seq(i, dtype=np.int16),
            ds.get_encoded_seq(i, dtype=np.int16),
        )


def test_roundtrip_preserves_permutation(tmp_path: Path) -> None:
    """Reorder, write, read back – permutation must survive."""
    ds = _make_dataset()
    perm = [4, 3, 2, 1, 0]
    ds.reorder(perm)
    emb_path = tmp_path / "perm.emb"
    ds.write(emb_path)

    loaded = EmbeddingDataset(filepath=emb_path)
    np.testing.assert_array_equal(loaded._permutation, perm)
    for i in range(ds.num_seq):
        np.testing.assert_array_equal(
            loaded.get_encoded_seq(i, dtype=np.int16),
            ds.get_encoded_seq(i, dtype=np.int16),
        )


def test_roundtrip_preserves_dtype(tmp_path: Path) -> None:
    """The embedding cache dtype should survive the round-trip."""
    seq_lens = np.array([3, 4])
    dim = 4
    cache_array = np.ones((7, dim), dtype=np.float16)
    cache = EmbeddingCache(seq_lens, dim, cache=cache_array)
    ds = EmbeddingDataset(
        embedding_cache=cache, seq_ids=["a", "b"]
    )
    emb_path = tmp_path / "dtype.emb"
    ds.write(emb_path)

    loaded = EmbeddingDataset(filepath=emb_path)
    assert loaded._embedding_cache.cache.dtype == np.float16


def test_npz_suffix_fallback(tmp_path: Path) -> None:
    """np.savez appends .npz; loading with the original name should still work."""
    ds = _make_dataset()
    emb_path = tmp_path / "fallback.emb"
    ds.write(emb_path)
    # np.savez may create fallback.emb.npz – verify the loader handles it
    loaded = EmbeddingDataset(filepath=emb_path)
    assert loaded.parsing_ok
    assert loaded.num_seq == ds.num_seq
