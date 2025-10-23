import os

import numpy as np
import tensorflow as tf

from learnMSA.msa_hmm import (Align, Configuration, Emitter, Initializers,
                              Training, Transitioner)
from learnMSA.msa_hmm.AlignmentModel import (AlignmentModel,
                                             find_faulty_sequences,
                                             non_homogeneous_mask_func)
from learnMSA.msa_hmm.SequenceDataset import AlignedDataset, SequenceDataset


def string_to_one_hot(s : str) -> tf.Tensor:
    i = [SequenceDataset.alphabet.index(aa) for aa in s]
    return tf.one_hot(i, len(SequenceDataset.alphabet)-1)


def test_subalignment() -> None:
    filename = os.path.dirname(__file__)+"/../tests/data/felix.fa"
    fasta_file = SequenceDataset(filename)
    length = 5
    config = Configuration.make_default(1)
    emission_init = string_to_one_hot("FELIK").numpy()*20
    insert_init = np.squeeze(string_to_one_hot("A") + string_to_one_hot("H") + string_to_one_hot("C"))*20
    config["emitter"] = Emitter.ProfileHMMEmitter(
        emission_init=Initializers.ConstantInitializer(emission_init),
        insertion_init=Initializers.ConstantInitializer(insert_init)
    )
    config["transitioner"] = Transitioner.ProfileHMMTransitioner(
        transition_init=(
            Initializers.make_default_transition_init(
                MM=0, MI=0, MD=0, II=0, IM=0, DM=0, DD=0,
                FC=0, FE=0, R=0, RF=0, T=0, scale=0
            )
        )
    )
    model = Training.default_model_generator(
        num_seq=8,
        effective_num_seq=8,
        model_lengths=[length],
        config=config,
        data=fasta_file
    )
    # subalignment
    subset = np.array([0, 2, 5])
    batch_gen = Training.DefaultBatchGenerator()
    batch_gen.configure(fasta_file, Configuration.make_default(1))
    # create alignment after building model
    sub_am = AlignmentModel(fasta_file, batch_gen, subset, 32, model)
    subalignment_strings = sub_am.to_string(0, add_block_sep=False)
    ref_subalignment = ["FE...LIK...", "FE...LIKhac", "FEahcLIK..."]
    for s, r in zip(subalignment_strings, ref_subalignment):
        assert s == r


def test_alignment_egf() -> None:
    """Test the high level alignment function with real world data"""
    train_filename = os.path.dirname(__file__)+"/../tests/data/egf.fasta"
    ref_filename = os.path.dirname(__file__)+"/../tests/data/egf.ref"
    with SequenceDataset(train_filename) as data:
        with AlignedDataset(ref_filename) as ref_msa:
            ref_subset = np.array([data.seq_ids.index(sid) for sid in ref_msa.seq_ids])
        config = Configuration.make_default(1)
        config["max_surgery_runs"] = 2  # do minimal surgery
        config["epochs"] = [5, 1, 5]
        am = Align.fit_and_align(
            data,
            config=config,
            subset=ref_subset,
            verbose=False
        )
        # some friendly thresholds to check if the alignment makes sense
        assert np.amin(am.compute_loglik()) > -70
        assert am.msa_hmm_layer.cell.length[0] > 25
        am.to_file(os.path.dirname(__file__)+"/../tests/data/egf.out.fasta", 0)
        with AlignedDataset(os.path.dirname(__file__)+"/../tests/data/egf.out.fasta") as pred_msa:
            sp = pred_msa.SP_score(ref_msa)
            # based on experience, any half decent hyperparameter choice should yield at least this score
            assert sp > 0.7


def test_non_homogeneous_mask() -> None:
    """Test non_homogeneous_mask_func"""
    seq_lens = tf.constant([[3, 5, 4]])

    class HmmCellMock:
        def __init__(self):
            self.num_models = 1
            self.length = [4]
            self.max_num_states = 11
            self.dtype = tf.float32

    mask = non_homogeneous_mask_func(2, seq_lens, HmmCellMock()).numpy()
    expected_zero_pos = [
        set([(1,8), (2,8), (3,8), (8,3), (8,4)]),
        set([(1,8), (8,3), (8,4)]),
        set([(1,8), (2,8), (8,3), (8,4)])
    ]
    for k in range(3):
        for u in range(11):
            for v in range(11):
                if (u, v) in expected_zero_pos[k]:
                    assert mask[0, k, u, v] == 0, f"Expected 0 at {u},{v}"
                else:
                    assert mask[0, k, u, v] == 1, f"Expected 1 at {u},{v}"

    # hitting a sequence end is a special case, always allow transitions out of the last match
    mask = non_homogeneous_mask_func(4, seq_lens, HmmCellMock()).numpy()
    expected_zero_pos = [
        set([(1,8), (2,8), (3,8)]),
        set([(1,8), (2,8), (3,8)]),
        set([(1,8), (2,8), (3,8)])
    ]
    for k in range(3):
        for u in range(11):
            for v in range(11):
                if (u, v) in expected_zero_pos[k]:
                    assert mask[0, k, u, v] == 0, f"Expected 0 at {u},{v}"
                else:
                    assert mask[0, k, u, v] == 1, f"Expected 1 at {u},{v}"


def test_find_faulty_sequences() -> None:
    model_length = 4
    C = 2*model_length
    T = 2*model_length+2
    seq_lens = np.array([3, 4, 4, 2,
                        4, 4, 4, 4,
                        2, 5, 5, 5,
                        3])
    state_seqs_max_lik = np.array([[[1,C,2,T,T], [1,C,2,4,T], [1,2,3,4,T], [1,3,T,T,T],
                                    [1,2,C,3,T], [1,2,C,1,T], [1,2,C,4,T], [1,2,C,4,T],
                                    [1,C,T,T,T], [1,2,3,C,5], [1,2,3,4,C], [3,C,C,C,1],
                                    [3,C,1,T,T]]])
    faulty_sequences = find_faulty_sequences(state_seqs_max_lik, model_length, seq_lens)
    np.testing.assert_equal(faulty_sequences, [0, 1, 4, 5, 6, 7, 8])
