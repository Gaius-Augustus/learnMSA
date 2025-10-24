import os

import numpy as np

from learnMSA.msa_hmm import Configuration, Training
from learnMSA.msa_hmm.SequenceDataset import SequenceDataset


def test_default_batch_gen() -> None:
    filename = os.path.dirname(__file__) + "/../tests/data/felix_insert_delete.fa"
    with SequenceDataset(filename) as data:
        batch_gen = Training.DefaultBatchGenerator(shuffle=False)
        batch_gen.configure(data, Configuration.make_default(1))
        test_batches = [[0], [1], [4], [0, 2], [0, 1, 2, 3, 4], [2, 3, 4]]
        alphabet = np.array(list(SequenceDataset.alphabet))
        for ind in test_batches:
            ind = np.array(ind)
            ref = [str(data.get_record(i).seq).upper() for i in ind]
            s, i = batch_gen(ind)
            np.testing.assert_equal(i[:, 0], ind)
            for i, (r, j) in enumerate(zip(ref, ind)):
                assert "".join(alphabet[s[i, 0, :data.seq_lens[j]]]) == r
