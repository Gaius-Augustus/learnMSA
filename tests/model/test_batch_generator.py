import os

import numpy as np
import pytest

from learnMSA import Configuration
from learnMSA.model import batch_generator
from learnMSA.model.context import LearnMSAContext
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
