"""Tests for ``learnMSA/model/tf/training.py``.

The mirror of ``tests/model/torch/test_training.py``. The two differ only in
the element contract: a ``tf.data`` pipeline yields ``(inputs, dummy_target)``
because keras expects a target, while the PyTorch ``DataLoader`` yields the
inputs alone.
"""

import numpy as np

from learnMSA.config.config import Configuration
from learnMSA.model.batch_generator import BatchGenerator
from learnMSA.model.context import LearnMSAContext
from learnMSA.model.tf.training import make_dataset
from tests.embedding_data import make_aa_dataset, make_embedding_dataset


def test_make_dataset_aa_plus_embedding() -> None:
    """A two-track batch keeps the amino acid and embedding tracks aligned."""
    aa_dataset = make_aa_dataset()
    embedding_dataset = make_embedding_dataset()

    context = LearnMSAContext(Configuration(), aa_dataset)
    batch_gen = BatchGenerator()
    batch_gen.configure((aa_dataset, embedding_dataset), context)

    # shuffle=False: with the per-model permutation on, the batch holds four
    # random 3-subsets of the dataset rather than sequences 0/2/3, so the
    # padded length -- and the routing checked below -- would be random.
    dataset, _ = make_dataset(
        np.array([0, 2, 3]), batch_gen, batch_size=3, shuffle=False
    )
    for (s, e, i), _ in dataset:
        break

    assert s.shape == (3, 18, 4, 20)  # aa track: per-residue distributions
    assert e.shape == (3, 18, 4, 8)
    assert i.shape == (3, 4)
    # The tracks are aligned: make_embedding_dataset fills sequence j with
    # j + 1, and every model column holds the same sequence here.
    assert np.all(i.numpy() == np.array([[0], [2], [3]]))
    assert np.all(e[:, 0].numpy() == np.array([1.0, 3.0, 4.0])[:, None, None])


def test_make_dataset_shared_batch() -> None:
    aa_dataset = make_aa_dataset()
    embedding_dataset = make_embedding_dataset()

    config = Configuration()
    config.training.share_batch = True
    context = LearnMSAContext(config, aa_dataset)
    batch_gen = BatchGenerator()
    batch_gen.configure((aa_dataset, embedding_dataset), context)

    dataset, _ = make_dataset(
        np.array([0, 2, 3]), batch_gen, batch_size=3, shuffle=False
    )
    for (s, e, i), _ in dataset:
        break

    assert s.shape == (3, 18, 1, 20)
    assert e.shape == (3, 18, 1, 8)
    assert i.shape == (3, 1)
    assert np.all(i.numpy() == np.array([[0], [2], [3]]))
    # The tracks stay aligned: sequence j is filled with j + 1.
    assert np.all(e[:, 0].numpy() == np.array([1.0, 3.0, 4.0])[:, None, None])
