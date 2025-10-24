"""Tests for MSA HMM functionality including Viterbi, backward, and posterior probabilities."""
import os
import numpy as np
import tensorflow as tf

from learnMSA.msa_hmm import (
    Align, Emitter, Transitioner, Initializers, MsaHmmCell, MsaHmmLayer,
    Training, Configuration, Viterbi
)
from learnMSA.msa_hmm.SequenceDataset import SequenceDataset
from learnMSA.msa_hmm.AlignmentModel import AlignmentModel
from tests import ref


def string_to_one_hot(s : str) -> tf.Tensor:
    """Convert a string to one-hot encoded tensor."""
    i = [SequenceDataset.alphabet.index(aa) for aa in s]
    return tf.one_hot(i, len(SequenceDataset.alphabet) - 1)


def get_all_seqs(data: SequenceDataset, num_models: int) -> np.ndarray:
    """Get all sequences from a dataset."""
    indices = np.arange(data.num_seq)
    batch_generator = Training.DefaultBatchGenerator()
    config = Configuration.make_default(num_models)
    batch_generator.configure(data, config)
    ds = Training.make_dataset(indices,
                               batch_generator,
                               batch_size=data.num_seq,
                               shuffle=False)
    for (seq, _), _ in ds:
        return seq.numpy()


def assert_vec(x: np.ndarray, y: np.ndarray) -> None:
    """Assert that two arrays are equal."""
    assert x.shape == y.shape
    assert np.all(x == y), str(x) + " not equal to " + str(y)


def test_matrices() -> None:
    """Test that transition and emission matrices sum to 1."""
    length = 32
    hmm_cell = MsaHmmCell.MsaHmmCell(length=length)
    hmm_cell.build((None, None, len(SequenceDataset.alphabet)))
    A = hmm_cell.transitioner.make_A()
    A_sum = np.sum(A, -1)
    for a in A_sum:
        np.testing.assert_almost_equal(a, 1.0, decimal=5)
    B = hmm_cell.emitter[0].make_B()
    B_sum = np.sum(B, -1)
    for b in B_sum:
        np.testing.assert_almost_equal(b, 1.0, decimal=5)


def test_cell() -> None:
    """Test basic HMM cell functionality."""
    length = 4
    emission_init = Initializers.ConstantInitializer(string_to_one_hot("ACGT").numpy() * 10)
    transition_init = Initializers.make_default_transition_init(
        MM=2, MI=0, MD=0, II=0, IM=0, DM=0, DD=0,
        FC=0, FE=3, R=0, RF=-1, T=0, scale=0
    )
    emitter = Emitter.ProfileHMMEmitter(
        emission_init=emission_init,
        insertion_init=tf.keras.initializers.Zeros()
    )
    transitioner = Transitioner.ProfileHMMTransitioner(
        transition_init=transition_init,
        flank_init=tf.keras.initializers.Zeros()
    )
    hmm_cell = MsaHmmCell.MsaHmmCell(length, emitter=emitter, transitioner=transitioner)
    hmm_cell.build((None, None, len(SequenceDataset.alphabet)))
    hmm_cell.recurrent_init()
    filename = os.path.dirname(__file__) + "/../tests/data/simple.fa"
    with SequenceDataset(filename) as data:
        sequences = get_all_seqs(data, 1)
    sequences = tf.one_hot(sequences, len(SequenceDataset.alphabet))
    assert sequences.shape == (2, 1, 5, len(SequenceDataset.alphabet))
    forward, loglik = hmm_cell.get_initial_state(batch_size=2)
    assert loglik[0] == 0
    # Next match state should always yield highest probability
    sequences = tf.transpose(sequences, [1, 0, 2, 3])
    emission_probs = hmm_cell.emission_probs(sequences)
    for i in range(length):
        _, (forward, loglik) = hmm_cell(emission_probs[:, :, i], (forward, loglik))
        assert np.argmax(forward[0]) == i + 1
    last_loglik = loglik
    # Check correct end in match state
    _, (forward, loglik) = hmm_cell(emission_probs[:, :, 4], (forward, loglik))
    assert np.argmax(forward[0]) == 2 * length + 2

    hmm_cell.recurrent_init()
    filename = os.path.dirname(__file__) + "/../tests/data/length_diff.fa"
    with SequenceDataset(filename) as data:
        sequences = get_all_seqs(data, 1)
    sequences = tf.one_hot(sequences, len(SequenceDataset.alphabet))
    assert sequences.shape == (2, 1, 10, len(SequenceDataset.alphabet))
    forward, loglik = hmm_cell.get_initial_state(batch_size=2)
    sequences = tf.transpose(sequences, [1, 0, 2, 3])
    emission_probs = hmm_cell.emission_probs(sequences)
    for i in range(length):
        _, (forward, loglik) = hmm_cell(emission_probs[:, :, i], (forward, loglik))
        assert np.argmax(forward[0]) == i + 1
        assert np.argmax(forward[1]) == i + 1
    _, (forward, loglik) = hmm_cell(emission_probs[:, :, length], (forward, loglik))
    assert np.argmax(forward[0]) == 2 * length + 2
    assert np.argmax(forward[1]) == 2 * length
    for i in range(4):
        old_loglik = loglik
        _, (forward, loglik) = hmm_cell(emission_probs[:, :, length + 1 + i], (forward, loglik))
        # The first sequence is shorter and padded with end-symbols
        # The first end symbol in each sequence affects the likelihood, but this is the
        # same constant for all sequences in the batch
        # Further padding does not affect the likelihood
        assert old_loglik[0] == loglik[0]
        # The second sequence has the motif of the first seq. repeated twice
        # Check whether the model loops correctly
        # Looping must yield larger probabilities than using the right flank state
        assert np.argmax(forward[1]) == i + 1


def test_viterbi() -> None:
    """Test Viterbi algorithm and decoding."""
    length = [5, 3]
    emission_init = [
        Initializers.ConstantInitializer(string_to_one_hot("FELIK").numpy() * 20),
        Initializers.ConstantInitializer(string_to_one_hot("AHC").numpy() * 20)
    ]
    transition_init = [Initializers.make_default_transition_init(
        MM=0, MI=0, MD=0, II=0, IM=0, DM=0, DD=0,
        FC=0, FE=0, R=0, RF=-1, T=0, scale=0
    )] * 2
    emitter = Emitter.ProfileHMMEmitter(
        emission_init=emission_init,
        insertion_init=[tf.keras.initializers.Zeros()] * 2
    )
    transitioner = Transitioner.ProfileHMMTransitioner(
        transition_init=transition_init,
        flank_init=[tf.keras.initializers.Zeros()] * 2
    )
    hmm_cell = MsaHmmCell.MsaHmmCell(length, emitter=emitter, transitioner=transitioner)
    hmm_cell.build((None, None, len(SequenceDataset.alphabet)))
    hmm_cell.recurrent_init()
    with SequenceDataset(os.path.dirname(__file__) + "/../tests/data/felix.fa") as data:
        ref_seqs = np.array([
            # model 1
            [[1, 2, 3, 4, 5, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
             [0, 0, 0, 1, 2, 3, 4, 5, 12, 12, 12, 12, 12, 12, 12],
             [1, 2, 3, 4, 5, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12],
             [1, 2, 3, 4, 5, 10, 10, 10, 1, 2, 3, 4, 5, 11, 12],
             [0, 2, 3, 4, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
             [1, 2, 7, 7, 7, 3, 4, 5, 12, 12, 12, 12, 12, 12, 12],
             [1, 6, 6, 2, 3, 8, 4, 9, 9, 9, 5, 12, 12, 12, 12],
             [1, 2, 3, 8, 8, 8, 4, 5, 11, 11, 11, 12, 12, 12, 12]],
            # model 2
            [[0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
             [1, 2, 3, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8],
             [0, 0, 0, 0, 0, 0, 1, 3, 8, 8, 8, 8, 8, 8, 8],
             [0, 0, 0, 0, 0, 1, 2, 3, 6, 6, 6, 6, 6, 1, 8],
             [1, 4, 4, 4, 2, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
             [0, 0, 1, 2, 3, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8],
             [0, 1, 2, 6, 6, 1, 6, 1, 2, 3, 7, 8, 8, 8, 8],
             [0, 0, 0, 1, 2, 3, 6, 6, 1, 2, 3, 8, 8, 8, 8]]
        ])
        sequences = get_all_seqs(data, 2)
        sequences = np.transpose(sequences, [1, 0, 2])
        state_seqs_max_lik = Viterbi.viterbi(sequences, hmm_cell).numpy()
        # states : [LEFT_FLANK, MATCH x length, INSERT x length-1, UNANNOTATED_SEGMENT, RIGHT_FLANK, END]
        assert_vec(state_seqs_max_lik, ref_seqs)
        # This produces a result identical to above, but runs viterbi batch wise
        # to avoid memory overflow
        batch_generator = Training.DefaultBatchGenerator(return_only_sequences=True)
        batch_generator.configure(data, Configuration.make_default(2))
        state_seqs_max_lik2 = Viterbi.get_state_seqs_max_lik(
            data,
            batch_generator,
            np.arange(data.num_seq),
            batch_size=2,
            model_ids=[0, 1],
            hmm_cell=hmm_cell
        )
        assert_vec(state_seqs_max_lik2, ref_seqs)
        indices = np.array([0, 4, 5])
        state_seqs_max_lik3 = Viterbi.get_state_seqs_max_lik(
            data,
            batch_generator,
            indices,  # try a subset
            batch_size=2,
            model_ids=[0, 1],
            hmm_cell=hmm_cell
        )
        max_len = np.amax(data.seq_lens[indices]) + 1

        for i, j in enumerate(indices):
            assert_vec(state_seqs_max_lik3[:, i], ref_seqs[:, j, :max_len])

        indices = np.array([[0, 3, 0, 0, 1, 0, 0, 0],
                            [5, 0, 6, 5, 0, 2, 1, 3]])  # skip the left flank

        # first domain hit
        ref_consensus = [  # model 1
            np.array([[0, 1, 2, 3, 4],
                      [3, 4, 5, 6, 7],
                      [0, 1, 2, 3, 4],
                      [0, 1, 2, 3, 4],
                      [-1, 1, 2, 3, -1],
                      [0, 1, 5, 6, 7],
                      [0, 3, 4, 6, 10],
                      [0, 1, 2, 6, 7]]),
            # model 2
            np.array([[-1, -1, -1],
                      [0, 1, 2],
                      [6, -1, 7],
                      [5, 6, 7],
                      [0, 4, -1],
                      [2, 3, 4],
                      [1, 2, -1],
                      [3, 4, 5]])]
        ref_insertion_lens = [  # model1
            np.array([[0] * 4,
                      [0] * 4,
                      [0] * 4,
                      [0] * 4,
                      [0] * 4,
                      [0, 3, 0, 0],
                      [2, 0, 1, 3],
                      [0, 0, 3, 0]]),
            # model2
            np.array([[0, 0],
                      [0, 0],
                      [0, 0],
                      [0, 0],
                      [3, 0],
                      [0, 0],
                      [0, 0],
                      [0, 0]])]
        ref_insertion_start = [  # model1
            np.array([[-1] * 4,
                      [-1] * 4,
                      [-1] * 4,
                      [-1] * 4,
                      [-1] * 4,
                      [-1, 2, -1, -1],
                      [1, -1, 5, 7],
                      [-1, -1, 3, -1]]),
            # model2
            np.array([[-1, -1],
                      [-1, -1],
                      [-1, -1],
                      [-1, -1],
                      [1, -1],
                      [-1, -1],
                      [-1, -1],
                      [-1, -1]])]
        ref_finished = np.array([  # model 1
            [True, True, True, False, True, True, True, True],
            # model 2
            [True, True, True, False, True, True, False, False]])
        ref_left_flank_lens = np.array([[0, 3, 0, 0, 1, 0, 0, 0],
                                        [5, 0, 6, 5, 0, 2, 1, 3]])
        ref_segment_lens = np.array([[0, 0, 0, 3, 0, 0, 0, 0],  # model 1
                                     [0, 0, 0, 5, 0, 0, 2, 2]])  # model 2
        ref_segment_start = np.array([[5, 8, 5, 5, 4, 8, 11, 8],  # model 1
                                      [5, 3, 8, 8, 5, 5, 3, 6]])  # model 2
        ref_right_flank_lens = np.array([[0, 0, 3, 1, 1, 0, 0, 3],  # model 1
                                         [0, 5, 0, 0, 0, 3, 1, 0]])  # model 2
        ref_right_flank_start = np.array([[5, 8, 5, 13, 4, 8, 11, 8],  # model 1
                                          [5, 3, 8, 14, 5, 5, 10, 11]])  # model 2

        s = len(SequenceDataset.alphabet)
        A = SequenceDataset.alphabet.index("A")
        H = SequenceDataset.alphabet.index("H")
        C = SequenceDataset.alphabet.index("C")
        a = SequenceDataset.alphabet.index("A") + s
        h = SequenceDataset.alphabet.index("H") + s
        c = SequenceDataset.alphabet.index("C") + s
        F = SequenceDataset.alphabet.index("F")
        E = SequenceDataset.alphabet.index("E")
        L = SequenceDataset.alphabet.index("L")
        I = SequenceDataset.alphabet.index("I")
        X = SequenceDataset.alphabet.index("K")
        f = SequenceDataset.alphabet.index("F") + s
        e = SequenceDataset.alphabet.index("E") + s
        l = SequenceDataset.alphabet.index("L") + s
        i = SequenceDataset.alphabet.index("I") + s
        x = SequenceDataset.alphabet.index("K") + s
        GAP = s - 1
        gap = 2 * s - 1

        ref_left_flank_block = [np.array([[gap] * 3,  # model 1
                                          [a, h, c],
                                          [gap] * 3,
                                          [gap] * 3,
                                          [gap, gap, a],
                                          [gap] * 3,
                                          [gap] * 3,
                                          [gap] * 3]),
                                np.array([[gap, f, e, l, i, x],  # model 2
                                          [gap] * 6,
                                          [f, e, l, i, x, h],
                                          [gap, f, e, l, i, x],
                                          [gap] * 6,
                                          [gap, gap, gap, gap, f, e],
                                          [gap] * 5 + [f],
                                          [gap, gap, gap, f, e, l]])]
        ref_right_flank_block = [np.array([[gap] * 3,  # model 1
                                           [gap] * 3,
                                           [h, a, c],
                                           [a, gap, gap],
                                           [h, gap, gap],
                                           [gap] * 3,
                                           [gap] * 3,
                                           [a, h, c]]),
                                 np.array([[gap] * 5,  # model 2
                                           [f, e, l, i, x],
                                           [gap] * 5,
                                           [gap] * 5,
                                           [gap] * 5,
                                           [l, i, x, gap, gap],
                                           [x] + [gap] * 4,
                                           [gap] * 5])]
        ref_ins_block = [np.array([[gap] * 2,
                                   [gap] * 2,
                                   [gap] * 2,
                                   [gap] * 2,
                                   [gap] * 2,
                                   [gap] * 2,
                                   [a, h],
                                   [gap] * 2]),
                         np.array([[gap] * 3,
                                   [gap] * 3,
                                   [gap] * 3,
                                   [gap] * 3,
                                   [e, l, i],
                                   [gap] * 3,
                                   [gap] * 3,
                                   [gap] * 3])]
        ref_core_blocks = [  # model 1
            [np.array([[F, gap, gap, E, gap, gap, gap, L, gap, gap, gap, I, gap, gap, gap, X],
                       [F, gap, gap, E, gap, gap, gap, L, gap, gap, gap, I, gap, gap, gap, X],
                       [F, gap, gap, E, gap, gap, gap, L, gap, gap, gap, I, gap, gap, gap, X],
                       [F, gap, gap, E, gap, gap, gap, L, gap, gap, gap, I, gap, gap, gap, X],
                       [GAP, gap, gap, E, gap, gap, gap, L, gap, gap, gap, I, gap, gap, gap, GAP],
                       [F, gap, gap, E, a, h, c, L, gap, gap, gap, I, gap, gap, gap, X],
                       [F, a, h, E, gap, gap, gap, L, a, gap, gap, I, a, h, c, X],
                       [F, gap, gap, E, gap, gap, gap, L, a, h, c, I, gap, gap, gap, X]]),
             np.array([[GAP] * 5,
                       [GAP] * 5,
                       [GAP] * 5,
                       [F, E, L, I, X],
                       [GAP] * 5,
                       [GAP] * 5,
                       [GAP] * 5,
                       [GAP] * 5])],
            # model 2
            [np.array([[GAP, gap, gap, gap, GAP, GAP],
                       [A, gap, gap, gap, H, C],
                       [A, gap, gap, gap, GAP, C],
                       [A, gap, gap, gap, H, C],
                       [A, e, l, i, H, GAP],
                       [A, gap, gap, gap, H, C],
                       [A, gap, gap, gap, H, GAP],
                       [A, gap, gap, gap, H, C]]),
             np.array([[GAP] * 3,
                       [GAP] * 3,
                       [GAP] * 3,
                       [A, GAP, GAP],
                       [GAP] * 3,
                       [GAP] * 3,
                       [A, GAP, GAP],
                       [A, H, C]])]]
        ref_num_blocks = [2, 3]
        # second domain hit
        ref_consensus_2 = [  # model 1
            np.array([[-1] * 5] * 3 +
                     [[8, 9, 10, 11, 12]] +
                     [[-1] * 5] * 4),
            # model 2
            np.array([[-1] * 3] * 3 +
                     [[13, -1, -1]] +
                     [[-1] * 3] * 2 +
                     [[5, -1, -1],
                      [8, 9, 10]])]
        ref_insertion_lens_2 = [np.array([[0] * 4] * 8),  # model 1
                                np.array([[0] * 2] * 8)]  # model 2
        ref_insertion_start_2 = [np.array([[-1] * 4] * 8),  # model 1
                                 np.array([[-1] * 2] * 8)]  # model 2
        ref_finished_2 = np.array([[True, True, True, True, True, True, True, True],
                                   [True, True, True, True, True, True, False, True]])
        ref_left_flank_lens_2 = np.array([[0, 3, 0, 0, 1, 0, 0, 0],  # model 1
                                          [5, 0, 6, 5, 0, 2, 1, 3]])  # model 2

        def assert_decoding_core_results(decoded, ref):
            for i in range(data.num_seq):
                for d, r in zip(decoded, ref):
                    assert_vec(d[i], r[i])

        for i in range(len(length)):
            # test decoding
            # test first core block isolated
            decoding_core_results = AlignmentModel.decode_core(length[i], state_seqs_max_lik[i], indices[i])
            assert_decoding_core_results(decoding_core_results, (ref_consensus[i],
                                                                  ref_insertion_lens[i],
                                                                  ref_insertion_start[i],
                                                                  ref_finished[i]))
            # test left flank insertions isolated
            left_flank_lens, left_flank_start = AlignmentModel.decode_flank(state_seqs_max_lik[i],
                                                                             flank_state_id=0,
                                                                             indices=np.array([0, 0, 0, 0, 0, 0, 0, 0]))
            assert_vec(left_flank_lens, ref_left_flank_lens[i])
            assert_vec(left_flank_start, np.array([0, 0, 0, 0, 0, 0, 0, 0]))
            # test whole decoding
            core_blocks, left_flank, right_flank, unannotated_segments = AlignmentModel.decode(length[i], state_seqs_max_lik[i])
            assert len(core_blocks) == ref_num_blocks[i]
            assert_decoding_core_results(core_blocks[0], (ref_consensus[i],
                                                          ref_insertion_lens[i],
                                                          ref_insertion_start[i],
                                                          ref_finished[i]))
            assert_decoding_core_results(core_blocks[1], (ref_consensus_2[i],
                                                          ref_insertion_lens_2[i],
                                                          ref_insertion_start_2[i],
                                                          ref_finished_2[i]))
            assert_vec(left_flank[0], ref_left_flank_lens[i])
            assert_vec(left_flank[1], np.array([0, 0, 0, 0, 0, 0, 0, 0]))
            assert_vec(unannotated_segments[0][0], ref_segment_lens[i])
            assert_vec(unannotated_segments[0][1], ref_segment_start[i])
            assert_vec(right_flank[0], ref_right_flank_lens[i])
            assert_vec(right_flank[1], ref_right_flank_start[i])

            # test conversion of decoded data to an actual alignment in table form
            left_flank_block = AlignmentModel.get_insertion_block(sequences[i],
                                                                  left_flank[0],
                                                                  np.amax(left_flank[0]),
                                                                  left_flank[1],
                                                                  adjust_to_right=True)
            assert_vec(left_flank_block, ref_left_flank_block[i])
            right_flank_block = AlignmentModel.get_insertion_block(sequences[i],
                                                                   right_flank[0],
                                                                   np.amax(right_flank[0]),
                                                                   right_flank[1])
            assert_vec(right_flank_block, ref_right_flank_block[i])
            ins_lens = core_blocks[0][1][:, 0]  # just check the first insert for simplicity
            ins_start = core_blocks[0][2][:, 0]
            ins_block = AlignmentModel.get_insertion_block(sequences[i],
                                                           ins_lens,
                                                           np.amax(ins_lens),
                                                           ins_start)
            assert_vec(ins_block, ref_ins_block[i])
            for (C, IL, IS, f), ref in zip(core_blocks, ref_core_blocks[i]):
                alignment_block = AlignmentModel.get_alignment_block(sequences[i],
                                                                     C, IL, np.amax(IL, axis=0), IS)
                assert_vec(alignment_block, ref)


def test_parallel_viterbi():
    """Test parallel Viterbi algorithm."""
    length = [5, 3]
    emission_init = [
        Initializers.ConstantInitializer(string_to_one_hot("FELIK").numpy() * 20),
        Initializers.ConstantInitializer(string_to_one_hot("AHC").numpy() * 20)
    ]
    transition_init = [Initializers.make_default_transition_init(
        MM=0, MI=0, MD=0, II=0, IM=0, DM=0, DD=0,
        FC=0, FE=0, R=0, RF=-1, T=0, scale=0
    )] * 2
    emitter = Emitter.ProfileHMMEmitter(
        emission_init=emission_init,
        insertion_init=[tf.keras.initializers.Zeros()] * 2
    )
    transitioner = Transitioner.ProfileHMMTransitioner(
        transition_init=transition_init,
        flank_init=[tf.keras.initializers.Zeros()] * 2
    )
    hmm_cell = MsaHmmCell.MsaHmmCell(length, emitter=emitter, transitioner=transitioner)
    hmm_cell.build((None, None, len(SequenceDataset.alphabet)))
    hmm_cell.recurrent_init()
    with SequenceDataset(os.path.dirname(__file__) + "/../tests/data/felix.fa") as data:
        ref_seqs = np.array([
            # model 1
            [[1, 2, 3, 4, 5, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
             [0, 0, 0, 1, 2, 3, 4, 5, 12, 12, 12, 12, 12, 12, 12],
             [1, 2, 3, 4, 5, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12],
             [1, 2, 3, 4, 5, 10, 10, 10, 1, 2, 3, 4, 5, 11, 12],
             [0, 2, 3, 4, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
             [1, 2, 7, 7, 7, 3, 4, 5, 12, 12, 12, 12, 12, 12, 12],
             [1, 6, 6, 2, 3, 8, 4, 9, 9, 9, 5, 12, 12, 12, 12],
             [1, 2, 3, 8, 8, 8, 4, 5, 11, 11, 11, 12, 12, 12, 12]],
            # model 2
            [[0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
             [1, 2, 3, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8],
             [0, 0, 0, 0, 0, 0, 1, 3, 8, 8, 8, 8, 8, 8, 8],
             [0, 0, 0, 0, 0, 1, 2, 3, 6, 6, 6, 6, 6, 1, 8],
             [1, 4, 4, 4, 2, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
             [0, 0, 1, 2, 3, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8],
             [0, 1, 2, 6, 6, 1, 6, 1, 2, 3, 7, 8, 8, 8, 8],
             [0, 0, 0, 1, 2, 3, 6, 6, 1, 2, 3, 8, 8, 8, 8]]
        ])
        sequences = get_all_seqs(data, 2)
        sequences = np.transpose(sequences, [1, 0, 2])
        state_seqs_max_lik_1, gamma_1 = Viterbi.viterbi(sequences, hmm_cell, parallel_factor=1, return_variables=True)
        state_seqs_max_lik_3, gamma_3 = Viterbi.viterbi(sequences, hmm_cell, parallel_factor=3, return_variables=True)
        state_seqs_max_lik_5, gamma_5 = Viterbi.viterbi(sequences, hmm_cell, parallel_factor=5, return_variables=True)
        np.testing.assert_almost_equal(gamma_1[:, :, ::5].numpy(), gamma_3.numpy()[..., 0, :], decimal=4)
        np.testing.assert_almost_equal(gamma_1[:, :, 4::5].numpy(), gamma_3.numpy()[..., 1, :], decimal=4)
        np.testing.assert_almost_equal(gamma_1[:, :, ::3].numpy(), gamma_5.numpy()[..., 0, :], decimal=4)
        np.testing.assert_almost_equal(gamma_1[:, :, 2::3].numpy(), gamma_5.numpy()[..., 1, :], decimal=4)
        assert_vec(state_seqs_max_lik_3.numpy(), ref_seqs)
        assert_vec(state_seqs_max_lik_5.numpy(), ref_seqs)


def test_aligned_insertions() -> None:
    """Test aligned insertion blocks."""
    sequences = np.array([[1, 2, 3, 4, 5],
                          [6, 7, 8, 9, 10],
                          [11, 12, 13, 14, 15]])
    lens = np.array([5, 4, 3])
    starts = np.array([0, 1, 2])
    custom_columns = np.array([[0, 1, 2, 3, 4, -1],
                               [0, 1, 4, 5, -1, -1],
                               [2, 3, 4, -1, -1, -1]])
    block = AlignmentModel.get_insertion_block(sequences, lens, 6, starts, custom_columns=custom_columns)
    expected_block = np.array([[1, 2, 3, 4, 5, 23],
                               [7, 8, 23, 23, 9, 10],
                               [23, 23, 13, 14, 15, 23]])
    assert_vec(block, expected_block + len(SequenceDataset.alphabet))


def test_backward() -> None:
    """Test backward algorithm."""
    length = [4]
    transition_kernel_initializers = ref.make_transition_init_A()
    # alphabet: {A,B}
    emission_kernel_initializer = np.log([[0.5, 0.5], [0.1, 0.9], [0.7, 0.3], [0.9, 0.1]])
    emission_kernel_initializer = Initializers.ConstantInitializer(emission_kernel_initializer)
    insertion_kernel_initializer = np.log([0.5, 0.5])
    insertion_kernel_initializer = Initializers.ConstantInitializer(insertion_kernel_initializer)
    emitter = Emitter.ProfileHMMEmitter(
        emission_init=emission_kernel_initializer,
        insertion_init=insertion_kernel_initializer
    )
    transitioner = Transitioner.ProfileHMMTransitioner(transition_init=transition_kernel_initializers)
    hmm_cell = MsaHmmCell.MsaHmmCell(length, dim=2 + 1, emitter=emitter, transitioner=transitioner)
    seq = tf.one_hot([[[0, 1, 0]]], 3)
    hmm_layer = MsaHmmLayer.MsaHmmLayer(hmm_cell, 1)
    hmm_layer.build(seq.shape)
    backward_seqs = hmm_layer.backward_recursion(seq)
    backward_ref = np.array([[1.] * 11,
                             [0.49724005, 0.11404998, 0.72149999,
                              0.73499997, 0.44999999, 0.3,
                              0.6, 0.7, 0.49931,
                              0.30000002, 0.]])
    for i in range(2):
        actual = np.exp(backward_seqs[0, 0, -(i + 1)])
        r = backward_ref[i] + hmm_cell.epsilon
        np.testing.assert_almost_equal(actual, r, decimal=5)


def test_posterior_state_probabilities() -> None:
    """Test posterior state probabilities."""
    train_filename = os.path.dirname(__file__) + "/../tests/data/egf.fasta"
    with SequenceDataset(train_filename) as data:
        hmm_cell = MsaHmmCell.MsaHmmCell(32)
        hmm_layer = MsaHmmLayer.MsaHmmLayer(hmm_cell, 1)
        hmm_layer.build((1, None, None, len(SequenceDataset.alphabet)))
        batch_gen = Training.DefaultBatchGenerator()
        batch_gen.configure(data, Configuration.make_default(1))
        indices = tf.range(data.num_seq, dtype=tf.int64)
        ds = Training.make_dataset(indices, batch_gen, batch_size=data.num_seq, shuffle=False)
        for x, _ in ds:
            seq = tf.one_hot(x[0], len(SequenceDataset.alphabet))
            seq = tf.transpose(seq, [1, 0, 2, 3])
            p = hmm_layer.state_posterior_log_probs(seq)
        p = np.exp(p)
        np.testing.assert_almost_equal(np.sum(p, -1), 1., decimal=4)


def test_sequence_weights() -> None:
    """Test sequence weighting functionality."""
    sequence_weights = np.array([0.1, 0.2, 0.5, 1, 2, 3])
    hmm_layer = MsaHmmLayer.MsaHmmLayer(MsaHmmCell.MsaHmmCell(32), sequence_weights=sequence_weights)
    loglik = np.array([[1, 2, 3], [4, 5, 6]])
    indices = np.array([[0, 1, 2], [3, 4, 5]])
    weighted_loglik = hmm_layer.apply_sequence_weights(loglik, indices)
    np.testing.assert_equal(np.array([[0.1, 0.4, 1.5], [4., 10., 18.]]), weighted_loglik)
