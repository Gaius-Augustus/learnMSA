"""Reading and writing per-residue 3Di logit files (numpy only)."""

import numpy as np
import pytest

from learnMSA.config.structure import StructureConfig
from learnMSA.structure.io import is_logits_file, read_logits, write_logits

ALPHABET = StructureConfig().structural_alphabet


def _example(rng):
    seq_lens = np.array([3, 5, 2])
    logits = rng.normal(size=(seq_lens.sum(), 20)).astype(np.float16)
    return logits, seq_lens, ["a", "b", "c"]


def test_round_trip(tmp_path) -> None:
    logits, seq_lens, ids = _example(np.random.default_rng(0))
    path = write_logits(tmp_path / "x", logits, seq_lens, ids, ALPHABET)
    assert path.name == "x.npz" and is_logits_file(path)
    data = read_logits(path, ALPHABET)
    assert data.seq_ids == ids
    np.testing.assert_array_equal(data.seq_lens, seq_lens)
    np.testing.assert_array_equal(data.get_encoded_seq(1), logits[3:8])
    np.testing.assert_array_equal(data.get_encoded_seq(2, 1, 2),
                                  logits[9:10])
    assert data.empty((2, 4)).shape == (2, 4, 20)


def test_permuted_alphabet_is_reordered(tmp_path) -> None:
    logits, seq_lens, ids = _example(np.random.default_rng(1))
    stored = ALPHABET[::-1]
    path = write_logits(tmp_path / "x.npz", logits, seq_lens, ids, stored)
    data = read_logits(path, ALPHABET)
    got = data.get_encoded_seq(0)
    for j, letter in enumerate(ALPHABET):
        np.testing.assert_array_equal(got[:, j],
                                      logits[:3, stored.index(letter)])


def test_foreign_alphabet_is_rejected(tmp_path) -> None:
    logits, seq_lens, ids = _example(np.random.default_rng(2))
    path = write_logits(tmp_path / "x.npz", logits, seq_lens, ids,
                        "ACDEFGHIKLMNPQRSTVWX")
    with pytest.raises(ValueError, match="do not match"):
        read_logits(path, ALPHABET)


def test_reorder_follows_ids(tmp_path) -> None:
    logits, seq_lens, ids = _example(np.random.default_rng(3))
    data = read_logits(
        write_logits(tmp_path / "x.npz", logits, seq_lens, ids, ALPHABET),
        ALPHABET,
    )
    data.reorder([2, 0, 1])
    assert data.seq_ids == ["c", "a", "b"]
    np.testing.assert_array_equal(data.seq_lens, [2, 3, 5])
    np.testing.assert_array_equal(data.get_encoded_seq(0), logits[8:10])


def test_embedding_files_are_not_logits(tmp_path) -> None:
    path = tmp_path / "emb.npz"
    np.savez(path, cache=np.zeros((2, 4)))
    assert not is_logits_file(path)
    assert not is_logits_file(tmp_path / "missing.npz")
