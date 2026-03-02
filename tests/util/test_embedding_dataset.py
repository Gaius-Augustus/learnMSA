from pathlib import Path

import numpy as np
import pytest

from learnMSA.config.config import Configuration
from learnMSA.model.context import LearnMSAContext
from learnMSA.model.tf.training import BatchGenerator, make_dataset
from learnMSA.util import EmbeddingCache, EmbeddingDataset, SequenceDataset


@pytest.fixture
def embedding_dataset() -> EmbeddingDataset:
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

@pytest.fixture
def aa_dataset() -> SequenceDataset:
    """Helper: build a small SequenceDataset from scratch."""
    seqs = [
        ("seq1", "ACDEG"),
        ("seq2", "ACDEFGHIKLM"),
        ("seq3", "ACDEFGHIKLMNPQRSM"),
        ("seq4", "ACDE"),
        ("seq5", "ACDEF"),
    ]
    ds = SequenceDataset(sequences=seqs)
    return ds


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


def test_make_dataset_aa_plus_embedding(
    aa_dataset: SequenceDataset,
    embedding_dataset: EmbeddingDataset,
) -> None:
    """Test that we can create an EmbeddingDataset from a SequenceDataset and
    an EmbeddingCache."""
    config = Configuration()
    context = LearnMSAContext(config, aa_dataset)
    batch_gen = BatchGenerator()
    batch_gen.configure((aa_dataset, embedding_dataset), context)
    dataset,_ = make_dataset(np.array([0, 2, 3]), batch_gen, batch_size=3)
    for (s,e,i),_ in dataset:
        break
    assert s.shape == (3, 4, 18)
    assert e.shape == (3, 4, 18, 8)
    assert i.shape == (3, 4)
