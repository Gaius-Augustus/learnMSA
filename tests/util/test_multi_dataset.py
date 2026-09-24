"""Tests for ``learnMSA/util/multi_dataset.py``."""

import os

import numpy as np
import pytest

from learnMSA.util import (MultiDataset, MultiSequenceDataset,
                           SequenceDataset)
from tests.embedding_data import (EMBEDDING_DIM, SEQ_LENS,
                                  make_embedding_dataset)

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
FILES = [os.path.join(DATA, "felix.fa"), os.path.join(DATA, "simple.fa")]


def test_global_index_space() -> None:
    parts = [SequenceDataset(f) for f in FILES]
    multi = MultiSequenceDataset(parts)

    assert multi.num_datasets == 2
    assert multi.num_seq == parts[0].num_seq + parts[1].num_seq
    np.testing.assert_equal(multi.offsets, [0, 8, 10])
    np.testing.assert_equal(
        multi.seq_lens, np.concatenate([p.seq_lens for p in parts])
    )
    assert multi.max_len == max(p.max_len for p in parts)
    assert multi.seq_ids == parts[0].seq_ids + parts[1].seq_ids
    np.testing.assert_equal(multi.global_indices(1), [8, 9])
    np.testing.assert_equal(multi.dataset_of(np.array([0, 7, 8, 9])),
                            [0, 0, 1, 1])
    for k, part in enumerate(parts):
        for j in range(part.num_seq):
            i = multi.to_global(k, j)
            assert multi.to_local(i) == (k, j)
            np.testing.assert_equal(
                multi.get_encoded_seq(i), part.get_encoded_seq(j)
            )
            np.testing.assert_equal(
                multi.get_encoded_seq(i, 1, 3, remap=False),
                part.get_encoded_seq(j, 1, 3, remap=False),
            )
            assert multi.get_header(i) == part.get_header(j)
    with pytest.raises(IndexError):
        multi.to_local(multi.num_seq)


def test_construction_from_files_and_sequences() -> None:
    from_files = MultiSequenceDataset(filepaths=FILES)
    from_sequences = MultiSequenceDataset(sequences=[
        [("a", "FELIK"), ("b", "AHC")],
        [("c", "ACGT")],
    ])
    from_files.validate_dataset()

    assert from_files.num_seq == 10
    assert from_sequences.num_seq == 3
    np.testing.assert_equal(from_sequences.offsets, [0, 2, 3])
    assert from_sequences.get_standardized_seq(2) == "ACGT"
    assert from_sequences.index("c") == 2
    assert from_sequences.output_alphabet == \
        from_sequences.datasets[0].output_alphabet
    with pytest.raises(ValueError):
        MultiSequenceDataset()
    with pytest.raises(ValueError):
        MultiSequenceDataset(filepaths=FILES, sequences=[[("a", "A")]])


def test_parts_must_share_the_encoding() -> None:
    aa = SequenceDataset(sequences=[("a", "ACDE")])
    other = SequenceDataset(sequences=[("b", "acde")], alphabet="acde")
    with pytest.raises(ValueError):
        MultiSequenceDataset([aa, other])


def test_multi_dataset_of_embeddings() -> None:
    multi = MultiDataset([make_embedding_dataset(), make_embedding_dataset()])
    n = len(SEQ_LENS)

    assert multi.num_seq == 2 * n
    assert multi.empty((2, 3)).shape == (2, 3, EMBEDDING_DIM)
    # The second part's sequence j is filled with j + 1, see the builder.
    np.testing.assert_equal(
        multi.get_encoded_seq(n + 2), np.full((SEQ_LENS[2], EMBEDDING_DIM), 3.)
    )
    np.testing.assert_equal(multi.get_encoded_seq(n + 2, 1, 3).shape,
                            (2, EMBEDDING_DIM))
    with pytest.raises(NotImplementedError):
        multi.reorder(np.arange(2 * n))
