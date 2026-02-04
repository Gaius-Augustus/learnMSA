import os

import numpy as np
import pytest

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
                assert "".join(alphabet[s[i, 0, :data.seq_lens[j]]]) == r


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
            assert s.shape[1] == 1  # num_models
            assert s.shape[2] == expected_seq_len  # static sequence length

            # Verify indices
            np.testing.assert_equal(i[:, 0], ind)

            # Verify sequences are correctly padded/cropped
            alphabet = np.array(list(SequenceDataset._default_alphabet))
            for batch_idx, seq_idx in enumerate(ind):
                seq_len = min(data.seq_lens[seq_idx], config.training.crop)
                ref_seq = str(data.get_record(seq_idx).seq).upper()[:config.training.crop]
                actual_seq = "".join(alphabet[s[batch_idx, 0, :seq_len]])
                assert actual_seq == ref_seq

                # Check padding (should be terminal symbols)
                padding = s[batch_idx, 0, seq_len:]
                assert np.all(padding == len(alphabet) - 1), \
                    f"Expected padding to be {len(alphabet) - 1}, got {padding}"
