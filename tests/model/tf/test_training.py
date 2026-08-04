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

    dataset, _ = make_dataset(np.array([0, 2, 3]), batch_gen, batch_size=3)
    for (s, e, i), _ in dataset:
        break

    assert s.shape == (3, 18, 4, 20)  # aa track: per-residue distributions
    assert e.shape == (3, 18, 4, 8)
    assert i.shape == (3, 4)
