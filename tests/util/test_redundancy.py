"""Tests for duplicate filtering (``Dataset.filter_duplicates``) and the
representatives that language models embed (``util/redundancy.py``)."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from learnMSA import Configuration
from learnMSA.run import util
from learnMSA.util import (AlignedDataset, EmbeddingCache, EmbeddingDataset,
                           SequenceDataset)
from learnMSA.util.redundancy import (RedundancyMap, find_duplicates,
                                      find_near_identical)

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def _random_seq(rng: np.random.Generator, n: int) -> str:
    return "".join(rng.choice(list(ALPHABET), n))


def _mutate(seq: str, positions: list[int]) -> str:
    s = list(seq)
    for p in positions:
        s[p] = "W" if s[p] != "W" else "Y"
    return "".join(s)


@pytest.fixture
def dup_fasta(tmp_path: Path) -> Path:
    """Duplicates of seq a, case and gap variants included."""
    path = tmp_path / "dups.fasta"
    path.write_text(
        ">a first a\nFELIK\n"
        ">b\nAHCFELIK\n"
        ">a2 copy of a\nfe-lik\n"
        ">c\nFELIKAHCFELIKA\n"
        ">b2\nAHCFELIK\n"
        ">a3\nFELIK\n"
    )
    return path


def test_find_duplicates(dup_fasta: Path) -> None:
    with SequenceDataset(dup_fasta) as data:
        red = find_duplicates(data)
    np.testing.assert_equal(red.rep_full_index, [0, 1, 3])
    np.testing.assert_equal(red.rep_index, [0, 1, 0, 2, 1, 0])
    assert red.num_full == 6 and red.num_rep == 3
    assert not red.is_trivial


def test_from_rep_of_requires_self_representation() -> None:
    with pytest.raises(AssertionError):
        RedundancyMap.from_rep_of(np.array([1, 0]))


def test_filter_duplicates(dup_fasta: Path) -> None:
    with SequenceDataset(dup_fasta) as data:
        red = data.filter_duplicates()
        assert data.seq_ids == ["a", "b", "c"]
        assert data.num_seq == 3
        assert data.max_len == 14
        np.testing.assert_equal(data.seq_lens, [5, 8, 14])
        assert data.get_header(1) == "b"
        # idempotent
        assert data.filter_duplicates() is red
        with pytest.raises(ValueError):
            data.filter_duplicates(find_duplicates(data.unfiltered()))

        full = data.unfiltered()
        assert full.num_seq == 6
        assert full.seq_ids == ["a", "b", "a2", "c", "b2", "a3"]
        assert full.get_header(2) == "a2 copy of a"
        assert full.redundancy is None
        # the filtered dataset is unchanged by the view
        assert data.num_seq == 3


def test_filter_without_duplicates_is_trivial() -> None:
    with SequenceDataset("tests/data/felix.fa") as data:
        red = data.filter_duplicates()
        assert red.is_trivial
        assert data.num_seq == 8
        assert data.unfiltered() is data


def test_write_after_filter_keeps_headers(
    dup_fasta: Path, tmp_path: Path
) -> None:
    with SequenceDataset(dup_fasta) as data:
        data.filter_duplicates()
        out = tmp_path / "out.fasta"
        data.write(out, "fasta")
        assert data.get_header(0) == "a first a"
        with SequenceDataset(out) as written:
            assert written.seq_ids == ["a", "b", "c"]


def test_subset_view(dup_fasta: Path) -> None:
    with SequenceDataset(dup_fasta) as data:
        sub = data.subset(np.array([3, 0]))
        assert sub.seq_ids == ["c", "a"]
        assert sub.max_len == 14
        assert data.num_seq == 6


def test_aligned_dataset_select(tmp_path: Path) -> None:
    path = tmp_path / "msa.fasta"
    path.write_text(">x\nFE-LIK\n>y\nFEALIK\n>z\nFE-LIK\n")
    with AlignedDataset(path) as msa:
        msa.filter_duplicates()
        assert msa.seq_ids == ["x", "y"]
        assert msa.msa_matrix.shape == (2, 6)
        np.testing.assert_equal(msa.get_column_map(1), np.arange(6))


def test_embedding_dataset_follows_map(dup_fasta: Path) -> None:
    with SequenceDataset(dup_fasta) as data:
        lens = data.seq_lens
        cache = EmbeddingCache(lens, 2, cache=np.repeat(
            np.arange(6, dtype=np.float32), lens
        )[:, None].repeat(2, axis=1))
        emb = EmbeddingDataset(embedding_cache=cache, seq_ids=data.seq_ids)
        with pytest.raises(NotImplementedError):
            emb.subset(np.arange(6)).filter_duplicates()
        red = data.filter_duplicates()
        emb.filter_duplicates(red)
        assert emb.seq_ids == data.seq_ids
        assert emb.num_seq == 3
        assert emb.get_encoded_seq(2)[0, 0] == 3  # c is the 4th sequence
        assert emb.unfiltered().get_encoded_seq(4)[0, 0] == 4


def test_cache_indices_share_rows_and_roundtrip(tmp_path: Path) -> None:
    lens = np.array([3, 4])
    cache = EmbeddingCache(lens, 2, cache=np.repeat(
        np.array([1, 2], dtype=np.float32), lens
    )[:, None].repeat(2, axis=1))
    emb = EmbeddingDataset(
        embedding_cache=cache,
        seq_ids=["r0", "m0", "r1", "m1"],
        cache_indices=np.array([0, 0, 1, 1]),
    )
    np.testing.assert_equal(emb.seq_lens, [3, 3, 4, 4])
    assert emb.max_len == 4
    np.testing.assert_equal(emb.get_encoded_seq(1), emb.get_encoded_seq(0))
    path = emb.write(tmp_path / "emb.npz")
    loaded = EmbeddingDataset(path)
    assert loaded.seq_ids == emb.seq_ids
    np.testing.assert_equal(loaded.seq_lens, emb.seq_lens)
    for i in range(4):
        np.testing.assert_equal(
            loaded.get_encoded_seq(i), emb.get_encoded_seq(i)
        )


@pytest.fixture
def near_identical_fasta(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    base = _random_seq(rng, 100)
    other = _random_seq(rng, 120)
    records = [
        ("base", base),
        ("one_sub", _mutate(base, [10])),            # 99% identical
        ("three_subs", _mutate(base, [10, 40, 70])),  # 97% identical
        ("shorter", base[:99]),                        # different length
        ("other", other),
        ("base_copy", base),
    ]
    path = tmp_path / "near.fasta"
    path.write_text("".join(f">{i}\n{s}\n" for i, s in records))
    return path


@pytest.mark.skipif(shutil.which("mmseqs") is None,
                    reason="mmseqs2 is not installed")
def test_find_near_identical(near_identical_fasta: Path, tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    with SequenceDataset(near_identical_fasta) as data:
        red = find_near_identical(data, 0.98, work_dir)
        # base, three_subs, shorter and other represent themselves; one_sub
        # and base_copy share the embeddings of base
        np.testing.assert_equal(red.rep_full_index, [0, 2, 3, 4])
        np.testing.assert_equal(red.rep_index, [0, 0, 1, 2, 3, 0])

        red = find_near_identical(data, 0.97, work_dir)
        np.testing.assert_equal(red.rep_full_index, [0, 3, 4])

        assert find_near_identical(data, 1.0, work_dir).num_rep == 5
    assert list(work_dir.iterdir()) == []


def test_find_near_identical_without_mmseqs(
    near_identical_fasta: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(shutil, "which", lambda _: None)
    with SequenceDataset(near_identical_fasta) as data:
        red = find_near_identical(data, 0.9, tmp_path)
    np.testing.assert_equal(red.rep_full_index, [0, 1, 2, 3, 4])


def test_embed_representatives(dup_fasta: Path, tmp_path: Path) -> None:
    config = Configuration()
    config.input_output.work_dir = str(tmp_path)
    config.input_output.verbose = False
    embedded = []

    def embed(reps: SequenceDataset) -> EmbeddingCache:
        embedded.extend(reps.seq_ids)
        values = np.repeat(
            np.arange(reps.num_seq, dtype=np.float32), reps.seq_lens
        )
        return EmbeddingCache(reps.seq_lens, 1, cache=values[:, None])

    with SequenceDataset(dup_fasta) as data:
        reuse = util.find_plm_representatives(config, data)
        emb = util.embed_representatives(data, reuse, embed)
        assert embedded == ["a", "b", "c"]
        assert emb.seq_ids == data.seq_ids
        np.testing.assert_equal(emb.seq_lens, data.seq_lens)
        values = [emb.get_encoded_seq(i)[0, 0] for i in range(6)]
        assert values == [0, 1, 0, 2, 1, 0]
