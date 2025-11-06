import os

import numpy as np

from learnMSA import Configuration
from learnMSA.msa_hmm.learnmsa_context import LearnMSAContext
from learnMSA.msa_hmm.legacy import make_legacy_config
from learnMSA.msa_hmm import Training
from learnMSA.msa_hmm.SequenceDataset import SequenceDataset


def test_default_batch_gen() -> None:
    filename = os.path.dirname(__file__) + "/../tests/data/felix_insert_delete.fa"
    with SequenceDataset(filename) as data:
        batch_gen = Training.DefaultBatchGenerator(shuffle=False)
        config = Configuration()
        config.training.num_model = 1
        config.training.no_sequence_weights = True
        config = make_legacy_config(config, LearnMSAContext(data, config))
        batch_gen.configure(data, config)
        test_batches = [[0], [1], [4], [0, 2], [0, 1, 2, 3, 4], [2, 3, 4]]
        alphabet = np.array(list(SequenceDataset.alphabet))
        for ind in test_batches:
            ind = np.array(ind)
            ref = [str(data.get_record(i).seq).upper() for i in ind]
            s, i = batch_gen(ind)
            np.testing.assert_equal(i[:, 0], ind)
            for i, (r, j) in enumerate(zip(ref, ind)):
                assert "".join(alphabet[s[i, 0, :data.seq_lens[j]]]) == r
