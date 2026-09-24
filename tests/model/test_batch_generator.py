import os

import numpy as np
import pytest

from learnMSA import Configuration
from learnMSA.model import batch_generator
from learnMSA.model.bucketing import make_default_bucket_scheme
from learnMSA.model.context import LearnMSAContext
from learnMSA.util.multi_dataset import MultiSequenceDataset
from learnMSA.util.sequence_dataset import SequenceDataset


def test_default_batch_gen() -> None:
    filename = os.path.dirname(__file__) + "/../data/felix_insert_delete.fa"
    with SequenceDataset(filename) as data:
        batch_gen = batch_generator.BatchGenerator(shuffle=False)
        config = Configuration()
        config.training.num_model = 1
        config.training.no_sequence_weights = True
        batch_gen.configure(data, LearnMSAContext(config, data))
        test_batches = [[0], [1], [4], [0, 2], [0, 1, 2, 3, 4], [2, 3, 4]]
        alphabet = np.array(list(SequenceDataset._default_alphabet))
        for ind in test_batches:
            ind = np.array(ind)
            ref = [str(data.get_record(i).seq).upper() for i in ind]
            s, i = batch_gen(ind)
            np.testing.assert_equal(i[:, 0], ind)
            for k, j in enumerate(ind):
                # The batch holds per-residue distributions; compare to the
                # dataset's own encoding.
                expected = data.get_encoded_seq(j)  # (L, D)
                np.testing.assert_allclose(
                    s[k, :data.seq_lens[j], 0], expected, atol=1e-6
                )


def test_static_shape_batch_gen() -> None:
    """Test BatchGenerator with static_shape_mode enabled."""
    filename = os.path.dirname(__file__) + "/../data/felix_insert_delete.fa"
    with SequenceDataset(filename) as data:
        # Set up batch generator with static shape mode
        batch_gen = batch_generator.BatchGenerator(shuffle=False, static_shape_mode=True)
        config = Configuration()
        config.training.num_model = 1
        config.training.no_sequence_weights = True

        batch_gen.configure(data, LearnMSAContext(config, data))

        # Test that all batches have the same shape
        test_batches = [[0], [1], [4], [0, 2], [0, 1, 2, 3, 4], [2, 3, 4]]
        expected_seq_len = data.max_len + 1

        for ind in test_batches:
            ind = np.array(ind)
            s, i = batch_gen(ind)

            # Check shape is static
            assert s.shape[0] == len(ind)  # batch size
            assert s.shape[1] == expected_seq_len  # static sequence length
            assert s.shape[2] == 1  # num_models

            # Verify indices
            np.testing.assert_equal(i[:, 0], ind)

            # Verify sequences are correctly padded/cropped
            for batch_idx, seq_idx in enumerate(ind):
                seq_len = min(int(data.seq_lens[seq_idx]), config.training.crop)
                # Real positions are valid distributions over the amino acids
                actual = s[batch_idx, :seq_len, 0]  # (seq_len, D)
                np.testing.assert_allclose(actual.sum(axis=-1), 1.0, atol=1e-5)
                # Padding positions are all-zero vectors (terminal)
                padding = s[batch_idx, seq_len:, 0]
                assert np.all(padding == 0.0), \
                    f"Expected padding to be all-zero, got {padding}"


def test_multi_dataset_batch_gen_returns_multiple_batches() -> None:
    fn = (os.path.dirname(__file__)
            + "/../data/felix_insert_delete.fa")
    with SequenceDataset(fn) as data_a, SequenceDataset(fn) as data_b:
        batch_gen = batch_generator.BatchGenerator(shuffle=False)
        config = Configuration()
        config.training.num_model = 1
        config.training.no_sequence_weights = True
        batch_gen.configure((data_a, data_b), LearnMSAContext(config, data_a))

        indices = np.array([0, 2, 4])
        s_a, s_b, ind = batch_gen(indices) # type: ignore

        assert s_a.shape[0] == indices.shape[0]
        assert s_b.shape[0] == indices.shape[0]
        assert s_a.shape[2] == 1
        assert s_b.shape[2] == 1
        np.testing.assert_equal(ind[:, 0], indices)

        for row_idx, seq_idx in enumerate(indices):
            seq_len = data_a.seq_lens[seq_idx]
            expected = data_a.get_encoded_seq(seq_idx)  # (L, D) distributions
            np.testing.assert_allclose(
                s_a[row_idx, :seq_len, 0], expected, atol=1e-6
            )
            np.testing.assert_allclose(
                s_b[row_idx, :seq_len, 0], expected, atol=1e-6
            )


def test_multi_model_default() -> None:
    filename = os.path.dirname(__file__) + "/../data/felix_insert_delete.fa"
    with SequenceDataset(filename) as data:
        config = Configuration()
        config.training.num_model = 4
        config.training.no_sequence_weights = True
        batch_gen = batch_generator.BatchGenerator(shuffle=True)
        batch_gen.configure(data, LearnMSAContext(config, data))

        assert batch_gen.generated_num_models == 1
        s, i = batch_gen(np.array([0, 1, 2, 3, 4]))
        assert s.shape[2] == 1
        assert i.shape == (5, 1)


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("share_batch", [False, True])
def test_shared_batch(shuffle: bool, share_batch: bool) -> None:
    filename = os.path.dirname(__file__) + "/../data/felix_insert_delete.fa"
    with SequenceDataset(filename) as data:
        config = Configuration()
        config.training.num_model = 4
        config.training.no_sequence_weights = True
        config.training.share_batch = share_batch
        batch_gen = batch_generator.BatchGenerator(shuffle=shuffle)
        batch_gen.configure(data, LearnMSAContext(config, data))

        shared_dim = 1 if share_batch else 4

        assert batch_gen.num_models == 4
        assert batch_gen.generated_num_models == shared_dim
        assert len(batch_gen.permutations) == shared_dim

        ind = np.array([0, 1, 2, 3, 4])
        s, i = batch_gen(ind)
        assert s.shape[0] == ind.size
        assert s.shape[2] == shared_dim
        assert i.shape == (ind.size, shared_dim)
        if not shuffle:
            np.testing.assert_equal(i[:, 0], ind)


def test_shared_batch_leaves_the_bucket_scheme_alone() -> None:
    filename = os.path.dirname(__file__) + "/../data/felix_insert_delete.fa"
    schemes = []
    for share in (False, True):
        with SequenceDataset(filename) as data:
            config = Configuration()
            config.training.num_model = 4
            config.training.no_sequence_weights = True
            config.training.share_batch = share
            batch_gen = batch_generator.BatchGenerator(shuffle=False)
            batch_gen.configure(data, LearnMSAContext(config, data))
            schemes.append(make_default_bucket_scheme(
                indices=np.arange(data.num_seq),
                batch_generator=batch_gen,
                model_lengths=[20, 20, 20, 20],
            ))
    assert schemes[0] == schemes[1]


def test_shuffle_stays_within_training_indices() -> None:
    filename = os.path.dirname(__file__) + "/../data/felix_insert_delete.fa"
    with SequenceDataset(filename) as data:
        config = Configuration()
        config.training.num_model = 4
        config.training.no_sequence_weights = True
        config.training.share_batch = False
        train = np.array([4, 1, 3, 1])
        batch_gen = batch_generator.BatchGenerator(shuffle=True)
        batch_gen.configure(data, LearnMSAContext(config, data), train)

        # Each column is a bijection of the distinct training indices.
        ind = np.array([1, 3, 4])
        _, i = batch_gen(ind)
        for k in range(i.shape[1]):
            np.testing.assert_equal(np.sort(i[:, k]), ind)
        # The other sequences are never touched.
        for p in batch_gen.permutations:
            np.testing.assert_equal(p[[0, 2, 5]], [0, 2, 5])


def test_full_training_set_keeps_the_permutations() -> None:
    filename = os.path.dirname(__file__) + "/../data/felix_insert_delete.fa"
    with SequenceDataset(filename) as data:
        config = Configuration()
        config.training.num_model = 4
        config.training.no_sequence_weights = True
        config.training.share_batch = False
        context = LearnMSAContext(config, data)

        # The permutations drawn before training indices were supported.
        np.random.seed(7)
        expected = [np.arange(data.num_seq) for _ in range(4)]
        for p in expected:
            np.random.shuffle(p)

        for indices in (None, np.arange(data.num_seq)):
            batch_gen = batch_generator.BatchGenerator(shuffle=True)
            np.random.seed(7)
            batch_gen.configure(data, context, indices)
            for p, q in zip(batch_gen.permutations, expected):
                np.testing.assert_equal(p, q)


def test_index_tables() -> None:
    seq_lens = np.array([3, 7, 5, 2, 9])
    np.testing.assert_equal(
        batch_generator.get_lengths(seq_lens, np.array([4, 0])), [9, 3]
    )
    # Every model gets its own column, padded with -1.
    table = batch_generator.get_index_table(
        [np.array([0, 1, 2]), np.array([3, 4])]
    )
    np.testing.assert_equal(table, [[0, 3], [1, 4], [2, -1]])
    np.testing.assert_equal(
        batch_generator.get_lengths(seq_lens, table), [3, 9, 5]
    )
    # Every column is sorted by decreasing length, empty cells stay last.
    sorted_table, positions = batch_generator.sort_index_table(
        table, seq_lens
    )
    np.testing.assert_equal(sorted_table, [[1, 4], [2, 3], [0, -1]])
    np.testing.assert_equal(positions, [[1, 1], [2, 0], [0, -1]])
    np.testing.assert_equal(
        batch_generator.get_lengths(seq_lens, sorted_table), [9, 5, 3]
    )


def _multi_context(auto_crop: bool = True):
    """Two datasets over disjoint residues, so every cell shows its origin."""
    multi = MultiSequenceDataset(sequences=[
        [("a1", "A" * 4), ("a2", "A" * 4), ("a3", "A" * 40)],
        [("w1", "W" * 10), ("w2", "W" * 10)],
    ])
    config = Configuration()
    config.training.no_sequence_weights = True
    config.training.auto_crop = auto_crop
    return multi, LearnMSAContext(config, multi)


def test_multi_batch_gen_fills_each_column_from_its_dataset() -> None:
    multi, context = _multi_context(auto_crop=False)
    batch_gen = context.batch_gen
    assert isinstance(batch_gen, batch_generator.MultiBatchGenerator)
    batch_gen.configure(multi, context, np.arange(multi.num_seq))
    assert batch_gen.groups == [(0, 3), (3, 5)]

    seqs, ind = batch_gen(np.array([[0, 3], [1, 4], [2, -1]]))
    assert seqs.shape == (3, 41, 2, 20)
    alphabet = SequenceDataset._default_alphabet
    a, w = alphabet.index("A"), alphabet.index("W")
    assert seqs[:, :, 0, a].sum() == 48 and seqs[:, :, 0].sum() == 48
    assert seqs[:, :, 1, w].sum() == 20 and seqs[:, :, 1].sum() == 20
    # The empty cell holds no residues and reports index 0.
    assert seqs[2, :, 1].sum() == 0
    np.testing.assert_equal(ind, [[0, 3], [1, 4], [2, 0]])

    # 1-D indices give every column the same sequences.
    _, ind = batch_gen(np.array([0, 3]))
    np.testing.assert_equal(ind, [[0, 0], [3, 3]])


def test_multi_batch_gen_crops_per_dataset() -> None:
    multi, context = _multi_context()
    # ceil(2 * mean length) of each dataset
    np.testing.assert_equal(context.head_crops, [32, 20])
    assert context.config.training.crop == 32
    batch_gen = context.batch_gen
    batch_gen.configure(multi, context, np.arange(multi.num_seq))

    seqs, _ = batch_gen(np.array([[2, 3]]))
    assert seqs.shape[1] == 33
    assert seqs[0, :, 0].sum() == 32

    # Prediction disables cropping.
    batch_gen.crop_long_seqs = np.inf
    seqs, _ = batch_gen(np.array([[2, 3]]))
    assert seqs[0, :, 0].sum() == 40

    batch_gen.crop_long_seqs = 32
    batch_gen.static_shape_mode = True
    seqs, _ = batch_gen(np.array([[0, 3]]))
    assert seqs.shape[1] == 33


def test_multi_batch_gen_validates_its_inputs() -> None:
    multi, context = _multi_context()
    batch_gen = context.batch_gen
    with pytest.raises(ValueError):
        batch_gen.configure(multi, context, np.array([3, 0, 1]))
    with pytest.raises(ValueError):
        batch_gen.configure(multi, context, np.array([0, 1]))
    with pytest.raises(ValueError):
        batch_gen.configure(multi.datasets[0], context)
