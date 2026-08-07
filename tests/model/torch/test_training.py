"""Tests for ``learnMSA/model/torch/training.py``.

The mirror of ``tests/model/tf/test_training.py``. The two differ only in the
element contract: a ``tf.data`` pipeline yields ``(inputs, dummy_target)``
because keras expects a target, while the ``DataLoader``'s collate function
returns the inputs as a flat tuple.
"""

from collections import Counter
from itertools import islice

import numpy as np

from learnMSA.config.config import Configuration
from learnMSA.model.batch_generator import BatchGenerator
from learnMSA.model.context import LearnMSAContext
from learnMSA.model.torch.training import (_RepeatingShuffleSampler,
                                           make_dataset)
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


def test_repeating_sampler_never_yields_a_short_batch() -> None:
    """Every batch is exactly ``batch_size``.

    A ``size % batch_size`` remainder would be a second input shape, and under
    ``torch.compile(dynamic=False)`` a second shape means a full extra compile
    of the training graph.
    """
    size, batch_size = 7774, 512  # 7774 % 512 == 94
    sampler = _RepeatingShuffleSampler(size, batch_size)

    batches = list(islice(iter(sampler), 40))

    assert len(batches) == 40
    assert all(len(b) == batch_size for b in batches)
    assert all(0 <= i < size for b in batches for i in b)


def test_repeating_sampler_visits_every_index_equally_often() -> None:
    """The stream is ``shuffle -> repeat -> batch``, like the TF pipeline.

    Batching each permutation on its own would drop or duplicate the tail;
    carrying the remainder into the next permutation must leave the stream an
    exact concatenation of whole permutations.
    """
    size, batch_size, passes = 100, 16, 8
    sampler = _RepeatingShuffleSampler(size, batch_size)

    stream: list[int] = []
    it = iter(sampler)
    while len(stream) < passes * size:
        stream.extend(next(it))

    counts = Counter(stream[:passes * size])
    assert len(counts) == size
    assert set(counts.values()) == {passes}
