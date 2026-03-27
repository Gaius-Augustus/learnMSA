import os

import numpy as np
import pytest
import tensorflow as tf

from learnMSA import Configuration
from learnMSA.model.tf import training
from learnMSA.model.context import LearnMSAContext
from learnMSA.util.sequence_dataset import SequenceDataset


def test_default_batch_gen() -> None:
    filename = os.path.dirname(__file__) + "/../tests/data/felix_insert_delete.fa"
    with SequenceDataset(filename) as data:
        batch_gen = training.BatchGenerator(shuffle=False)
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
            for i, (r, j) in enumerate(zip(ref, ind)):
                assert "".join(alphabet[s[i, :data.seq_lens[j], 0]]) == r


def test_static_shape_batch_gen() -> None:
    """Test BatchGenerator with static_shape_mode enabled."""
    filename = os.path.dirname(__file__) + "/../tests/data/felix_insert_delete.fa"
    with SequenceDataset(filename) as data:
        # Set up batch generator with static shape mode
        batch_gen = training.BatchGenerator(shuffle=False, static_shape_mode=True)
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
            alphabet = np.array(list(SequenceDataset._default_alphabet))
            for batch_idx, seq_idx in enumerate(ind):
                seq_len = min(data.seq_lens[seq_idx], config.training.crop)
                ref_seq = str(data.get_record(seq_idx).seq).upper()[:config.training.crop]
                actual_seq = "".join(alphabet[s[batch_idx, :seq_len, 0]])
                assert actual_seq == ref_seq

                # Check padding (should be terminal symbols)
                padding = s[batch_idx, seq_len:, 0]
                assert np.all(padding == len(alphabet) - 1), \
                    f"Expected padding to be {len(alphabet) - 1}, got {padding}"


def test_multi_dataset_batch_gen_returns_multiple_batches() -> None:
    fn = (os.path.dirname(__file__)
            + "/../tests/data/felix_insert_delete.fa")
    with SequenceDataset(fn) as data_a, SequenceDataset(fn) as data_b:
        batch_gen = training.BatchGenerator(shuffle=False)
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

        alphabet = np.array(list(SequenceDataset._default_alphabet))
        for row_idx, seq_idx in enumerate(indices):
            ref = str(data_a.get_record(seq_idx).seq).upper()
            seq_len = data_a.seq_lens[seq_idx]
            dec_a = "".join(alphabet[s_a[row_idx, :seq_len, 0]])
            dec_b = "".join(alphabet[s_b[row_idx, :seq_len, 0]])
            assert dec_a == ref
            assert dec_b == ref


def test_make_dataset() -> None:
    """Test that make_dataset returns a usable dataset with correct step count."""
    filename = os.path.dirname(__file__) + "/../tests/data/felix_insert_delete.fa"
    with SequenceDataset(filename) as data:
        batch_gen = training.BatchGenerator(shuffle=False)
        config = Configuration()
        config.training.num_model = 1
        config.training.no_sequence_weights = True
        batch_gen.configure(data, LearnMSAContext(config, data))

        indices = np.arange(data.num_seq)
        batch_size = 2

        ds, steps = training.make_dataset(
            indices=indices,
            batch_generator=batch_gen,
            batch_size=batch_size,
            shuffle=False,
        )

        # Non-shuffled step count should be ceil(num_seq / batch_size)
        expected_steps = int(np.ceil(data.num_seq / batch_size))
        assert steps == expected_steps

        # Iterate one batch and check shapes
        for batch_x, _batch_y in ds.take(1):
            sequences = batch_x[0]
            perm_indices = batch_x[1]
            assert sequences.shape[0] == batch_size
            assert sequences.shape[2] == 1  # num_models
            assert perm_indices.shape[0] == batch_size
            assert perm_indices.shape[1] == 1  # num_models


def test_make_dataset_with_additional_data() -> None:
    """Test that make_dataset forwards additional_data alongside batches."""
    filename = os.path.dirname(__file__) + "/../tests/data/felix_insert_delete.fa"
    with SequenceDataset(filename) as data:
        batch_gen = training.BatchGenerator(shuffle=False)
        config = Configuration()
        config.training.num_model = 1
        config.training.no_sequence_weights = True
        batch_gen.configure(data, LearnMSAContext(config, data))

        indices = np.arange(data.num_seq)
        batch_size = 2
        # Create per-sequence additional data with extra dimensions
        add_data = np.random.default_rng(42).random(
            (data.num_seq, 3, 2), dtype=np.float32
        )

        ds, steps = training.make_dataset(
            indices=indices,
            batch_generator=batch_gen,
            batch_size=batch_size,
            shuffle=False,
            additional_data=add_data,
        )

        expected_steps = int(np.ceil(data.num_seq / batch_size))
        assert steps == expected_steps

        # Iterate one batch and verify the additional data is present
        for batch_x, _batch_y in ds.take(1):
            # With additional_data the tuple is (sequences, perm_indices, add_batch)
            sequences = batch_x[0]
            perm_indices = batch_x[1]
            add_batch = batch_x[2]
            assert sequences.shape[0] == batch_size
            assert sequences.shape[2] == 1
            assert perm_indices.shape[0] == batch_size
            assert add_batch.shape[0] == batch_size
            assert add_batch.shape[1:] == (3, 2)
            # First batch should contain indices 0 and 1
            np.testing.assert_allclose(
                add_batch.numpy(), add_data[:batch_size]
            )

