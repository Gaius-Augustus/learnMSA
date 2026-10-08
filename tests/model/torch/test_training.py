"""Tests for ``learnMSA/model/torch/training.py``.

The mirror of ``tests/model/tf/test_training.py``. The two differ only in the
element contract: a ``tf.data`` pipeline yields ``(inputs, dummy_target)``
because keras expects a target, while the ``DataLoader``'s collate function
returns the inputs as a flat tuple.
"""

import os
from collections import Counter
from itertools import islice

import numpy as np

from learnMSA.backend import set_random_seed
from learnMSA.config.config import Configuration
from learnMSA.model.batch_generator import BatchGenerator
from learnMSA.model.context import LearnMSAContext
from learnMSA.model.torch.training import (_RepeatingShuffleSampler,
                                           make_dataset)
from learnMSA.util.sequence_dataset import SequenceDataset
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
    loader, _ = make_dataset(
        np.array([0, 2, 3]), batch_gen, batch_size=3, shuffle=False
    )
    for s, e, i in loader:
        break

    assert tuple(s.shape) == (3, 18, 1, 20)  # aa track: distributions
    assert tuple(e.shape) == (3, 18, 1, 8)
    assert tuple(i.shape) == (3, 1)
    # The tracks are aligned: make_embedding_dataset fills sequence j with
    # j + 1, and every model column holds the same sequence here.
    assert np.all(i.numpy() == np.array([[0], [2], [3]]))
    assert np.all(e[:, 0].numpy() == np.array([1.0, 3.0, 4.0])[:, None, None])


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


def test_make_dataset_shared_batch() -> None:
    """A shared batch collapses the model axis of every track at once."""
    aa_dataset = make_aa_dataset()
    embedding_dataset = make_embedding_dataset()

    config = Configuration()
    config.training.share_batch = True
    context = LearnMSAContext(config, aa_dataset)
    batch_gen = BatchGenerator()
    batch_gen.configure((aa_dataset, embedding_dataset), context)

    loader, _ = make_dataset(
        np.array([0, 2, 3]), batch_gen, batch_size=3, shuffle=False
    )
    for s, e, i in loader:
        break

    assert tuple(s.shape) == (3, 18, 1, 20)
    assert tuple(e.shape) == (3, 18, 1, 8)
    assert tuple(i.shape) == (3, 1)
    assert np.all(i.numpy() == np.array([[0], [2], [3]]))
    # The tracks stay aligned: sequence j is filled with j + 1.
    assert np.all(e[:, 0].numpy() == np.array([1.0, 3.0, 4.0])[:, None, None])


def _first_shuffled_batches(seed: int) -> list[np.ndarray]:
    """The first training batches after seeding learnMSA with ``seed``."""
    set_random_seed(seed)
    filename = os.path.dirname(__file__) + "/../../data/felix_insert_delete.fa"
    with SequenceDataset(filename) as data:
        config = Configuration()
        config.training.num_model = 2
        config.training.no_sequence_weights = True
        config.training.auto_crop = False
        config.training.crop = 5
        batch_gen = BatchGenerator()
        batch_gen.configure(data, LearnMSAContext(config, data))
        dataset, _ = make_dataset(
            np.arange(data.num_seq), batch_gen, batch_size=4, shuffle=True
        )
        batches = []
        for element in islice(dataset, 6):
            s, i = element
            batches += [s.numpy(), i.numpy()]
    return batches


def test_shuffled_batches_are_reproducible() -> None:
    """Seeding fixes the batch order, the permutations and the crops."""
    first = _first_shuffled_batches(3)
    for a, b in zip(first, _first_shuffled_batches(3)):
        np.testing.assert_equal(a, b)
    other = _first_shuffled_batches(4)
    assert any(not np.array_equal(a, b) for a, b in zip(first, other))
