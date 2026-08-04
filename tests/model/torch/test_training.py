"""Tests for ``learnMSA/model/torch/training.py``.

The mirror of ``tests/model/tf/test_training.py``. The two differ only in the
element contract: a ``tf.data`` pipeline yields ``(inputs, dummy_target)``
because keras expects a target, while the ``DataLoader``'s collate function
returns the inputs as a flat tuple.
"""

import numpy as np

from learnMSA.config.config import Configuration
from learnMSA.model.batch_generator import BatchGenerator
from learnMSA.model.context import LearnMSAContext
from learnMSA.model.torch.training import make_dataset
from tests.embedding_data import make_aa_dataset, make_embedding_dataset


def test_make_dataset_aa_plus_embedding() -> None:
    """A two-track batch keeps the amino acid and embedding tracks aligned."""
    aa_dataset = make_aa_dataset()
    embedding_dataset = make_embedding_dataset()

    context = LearnMSAContext(Configuration(), aa_dataset)
    batch_gen = BatchGenerator()
    batch_gen.configure((aa_dataset, embedding_dataset), context)

    loader, _ = make_dataset(np.array([0, 2, 3]), batch_gen, batch_size=3)
    for s, e, i in loader:
        break

    assert tuple(s.shape) == (3, 18, 4, 20)  # aa track: distributions
    assert tuple(e.shape) == (3, 18, 4, 8)
    assert tuple(i.shape) == (3, 4)
