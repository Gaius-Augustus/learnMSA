from pathlib import Path

import numpy as np
import pytest

from learnMSA.util import EmbeddingCache, EmbeddingDataset, SequenceDataset
from tests.embedding_data import make_aa_dataset, make_embedding_dataset


@pytest.fixture
def embedding_dataset() -> EmbeddingDataset:
    return make_embedding_dataset()

@pytest.fixture
def aa_dataset() -> SequenceDataset:
    return make_aa_dataset()


def test_write_and_read_roundtrip(
    embedding_dataset: EmbeddingDataset,
    tmp_path: Path
) -> None:
    """Write an EmbeddingDataset to .emb and read it back."""
    emb_path = tmp_path / "test.emb"
    embedding_dataset.write(emb_path)

    loaded = EmbeddingDataset(filepath=emb_path)
    assert loaded.parsing_ok
    assert loaded.num_seq == embedding_dataset.num_seq
    assert loaded.max_len == embedding_dataset.max_len
    assert loaded.seq_ids == embedding_dataset.seq_ids
    np.testing.assert_array_equal(
        loaded.seq_lens, embedding_dataset.seq_lens
    )
    np.testing.assert_array_equal(
        loaded._permutation, embedding_dataset._permutation
    )
    np.testing.assert_array_equal(
        loaded._embedding_cache.cache, embedding_dataset._embedding_cache.cache
    )
    for i in range(embedding_dataset.num_seq):
        np.testing.assert_array_equal(
            loaded.get_encoded_seq(i, dtype=np.int16),
            embedding_dataset.get_encoded_seq(i, dtype=np.int16),
        )


def test_roundtrip_preserves_permutation(
    embedding_dataset: EmbeddingDataset,
    tmp_path: Path,
) -> None:
    """Reorder, write, read back – permutation must survive."""
    perm = [4, 3, 2, 1, 0]
    embedding_dataset.reorder(perm)
    emb_path = tmp_path / "perm.emb"
    embedding_dataset.write(emb_path)

    loaded = EmbeddingDataset(filepath=emb_path)
    np.testing.assert_array_equal(loaded._permutation, perm)
    for i in range(embedding_dataset.num_seq):
        np.testing.assert_array_equal(
            loaded.get_encoded_seq(i, dtype=np.int16),
            embedding_dataset.get_encoded_seq(i, dtype=np.int16),
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


def test_npz_suffix_fallback(
    embedding_dataset: EmbeddingDataset,
    tmp_path: Path,
) -> None:
    """np.savez appends .npz; loading with the original name should still
    work."""
    emb_path = tmp_path / "fallback.emb"
    embedding_dataset.write(emb_path)
    # np.savez may create fallback.emb.npz – verify the loader handles it
    loaded = EmbeddingDataset(filepath=emb_path)
    assert loaded.parsing_ok
    assert loaded.num_seq == embedding_dataset.num_seq


def test_reorder_embedding_dataset(embedding_dataset: EmbeddingDataset) -> None:
    """reorder must update seq_ids, seq_lens and route get_encoded_seq correctly."""
    perm = [2, 0, 4, 1, 3]
    original_ids = list(embedding_dataset.seq_ids)
    original_lens = embedding_dataset.seq_lens.copy()

    embedding_dataset.reorder(perm)

    assert embedding_dataset.seq_ids == [original_ids[i] for i in perm]
    np.testing.assert_array_equal(embedding_dataset.seq_lens, original_lens[perm])
    # Sequence i originally had a constant embedding value of (i + 1)
    for new_i, old_i in enumerate(perm):
        emb = embedding_dataset.get_encoded_seq(new_i)
        np.testing.assert_array_equal(emb, np.full_like(emb, old_i + 1))


def test_adapt_order_sequence_dataset() -> None:
    """adapt_order must produce matching seq_ids, seq_lens and encoded sequences."""
    seqs = [("a", "ACDE"), ("b", "GHIL"), ("c", "MNPQ")]
    ds_ref = SequenceDataset(sequences=seqs)
    ds = SequenceDataset(sequences=[seqs[1], seqs[2], seqs[0]])

    ds.adapt_order(ds_ref)

    assert ds.seq_ids == ds_ref.seq_ids
    np.testing.assert_array_equal(ds.seq_lens, ds_ref.seq_lens)
    for i in range(len(seqs)):
        np.testing.assert_array_equal(ds.get_encoded_seq(i), ds_ref.get_encoded_seq(i))


def test_adapt_order_embedding_dataset(embedding_dataset: EmbeddingDataset) -> None:
    """adapt_order must align seq_ids, seq_lens and embeddings with a reference."""
    # Build a reference dataset with the sequences in reversed order
    n = embedding_dataset.num_seq
    rev_ids = list(reversed(embedding_dataset.seq_ids))
    rev_lens = embedding_dataset.seq_lens[::-1].copy()
    dim = embedding_dataset._embedding_cache.dim
    # Original seq_i has constant embedding value (i + 1); reversed index k
    # maps to original index (n - 1 - k), so value = (n - k).
    rows = [np.full((rev_lens[k], dim), n - k, dtype=np.float32) for k in range(n)]
    ref_cache = EmbeddingCache(rev_lens, dim, cache=np.concatenate(rows))
    ref = EmbeddingDataset(embedding_cache=ref_cache, seq_ids=rev_ids)

    embedding_dataset.adapt_order(ref)

    assert embedding_dataset.seq_ids == ref.seq_ids
    np.testing.assert_array_equal(embedding_dataset.seq_lens, ref.seq_lens)
    for i in range(n):
        np.testing.assert_array_equal(
            embedding_dataset.get_encoded_seq(i), ref.get_encoded_seq(i)
        )


def test_the_dtype_follows_the_cache() -> None:
    """A batch is assembled in the precision the embeddings are cached in.

    Converting in the dataset would cost a copy per sequence and per model,
    which for full-width language model embeddings is the most expensive part
    of a training step. The model widens the track instead.
    """
    lens = np.array([4, 6])
    cache = EmbeddingCache(
        lens, 8, cache=np.ones((int(lens.sum()), 8), dtype=np.float16)
    )
    data = EmbeddingDataset(embedding_cache=cache, seq_ids=["a", "b"])

    assert data.get_dtype() == np.float16
    assert data.empty((2, 3)).dtype == np.float16

    seq = data.get_encoded_seq(0)
    assert seq.dtype == np.float16
    # A view of the cache, not a converted copy of it.
    assert seq.base is cache.cache

    # An explicit dtype still wins; the file roundtrip tests rely on it.
    assert data.get_encoded_seq(0, dtype=np.float32).dtype == np.float32
