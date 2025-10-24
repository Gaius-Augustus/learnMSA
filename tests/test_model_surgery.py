"""Tests for model surgery operations (expanding/discarding positions)."""
import os

import numpy as np
import tensorflow as tf

from learnMSA.msa_hmm import (Align, Configuration, Emitter, Initializers,
                              Training, Transitioner)
from learnMSA.msa_hmm.AlignmentModel import AlignmentModel
from learnMSA.msa_hmm.SequenceDataset import SequenceDataset


def assert_vec(x : np.ndarray, y: np.ndarray) -> None:
    """Assert that two arrays are equal."""
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)
    assert x.shape == y.shape, str(x) + " " + str(y)
    assert np.all(x == y), str(x) + " not equal to " + str(y)


def string_to_one_hot(s : str) -> tf.Tensor:
    """Convert a string to one-hot encoded tensor."""
    i = [SequenceDataset.alphabet.index(aa) for aa in s]
    return tf.one_hot(i, len(SequenceDataset.alphabet) - 1)


def make_test_alignment(data: SequenceDataset) -> AlignmentModel:
    """Create a test alignment model with specific parameters."""
    config = Configuration.make_default(1)
    emission_init = string_to_one_hot("FELIC").numpy() * 10
    insert_init = np.squeeze(
        string_to_one_hot("A") + string_to_one_hot("N")
    ) * 10
    transition_init = Initializers.make_default_transition_init(
        MM=0,
        MI=0,
        MD=-1,
        II=1,
        IM=0,
        DM=1,
        DD=0,
        FC=0,
        FE=0,
        R=0,
        RF=0,
        T=0,
        scale=0
    )
    transition_init["match_to_match"] = Initializers.ConstantInitializer(0)
    transition_init["match_to_insert"] = Initializers.ConstantInitializer(0)
    transition_init["match_to_delete"] = Initializers.ConstantInitializer(-1)
    transition_init["begin_to_match"] = Initializers.ConstantInitializer([1, 0, 0, 0, 0])
    transition_init["match_to_end"] = Initializers.ConstantInitializer(0)
    config["emitter"] = Emitter.ProfileHMMEmitter(
        emission_init=Initializers.ConstantInitializer(emission_init),
        insertion_init=Initializers.ConstantInitializer(insert_init)
    )
    config["transitioner"] = Transitioner.ProfileHMMTransitioner(
        transition_init=transition_init
    )
    model = Training.default_model_generator(
        num_seq=10,
        effective_num_seq=10,
        model_lengths=[5],
        config=config,
        data=data
    )
    batch_gen = Training.DefaultBatchGenerator()
    batch_gen.configure(data, config)
    am = AlignmentModel(
        data,
        batch_gen,
        np.arange(data.num_seq),
        32,
        model
    )
    return am


def test_discard_or_expand_positions() -> None:
    """Test detection of positions to discard or expand."""
    filename = os.path.dirname(__file__) + "/../tests/data/felix_insert_delete.fa"
    with SequenceDataset(filename) as data:
        am = make_test_alignment(data)
        # A simple alignment to test detection of
        # too sparse columns and too frequent insertions
        ref_seqs = [
            "..........F.-LnnnI-aaaFELnICnnn",
            "nnnnnnnnnn-.-Lnn.I-aaaF--.ICnnn",
            "..........-.-Lnn.I-...---.--nnn",
            "..........-.-Lnn.I-...---.--nnn",
            "..........-.--...ICaaaF--.I-nnn",
            "..........FnE-...ICaaaF-LnI-nnn"
        ]
        aligned_sequences = am.to_string(model_index=0, add_block_sep=False)
        for s, ref_s in zip(aligned_sequences, ref_seqs):
            assert s == ref_s
        assert 0 in am.metadata
        assert am.metadata[0].alignment_len == len(ref_seqs[0])
        # Shape: [number of domain hits, length]
        deletions = np.sum(am.metadata[0].consensus == -1, axis=1)
        assert_vec(deletions, [[4, 5, 2, 0, 4], [2, 5, 4, 2, 4]])
        # Shape: [number of domain hits, num seq]
        assert_vec(
            am.metadata[0].finished,
            [[False, False, True, True, False, False],
             [True, True, True, True, True, True]]
        )
        # Shape: [number of domain hits, num seq, L-1 inner positions]
        assert_vec(
            am.metadata[0].insertion_lens,
            [[[0, 0, 3, 0],
              [0, 0, 2, 0],
              [0, 0, 2, 0],
              [0, 0, 2, 0],
              [0, 0, 0, 0],
              [1, 0, 0, 0]],

             [[0, 0, 1, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 1, 0]]]
        )
        pos_expand, expansion_lens, pos_discard = Align.get_discard_or_expand_positions(am)
        pos_expand = pos_expand[0]
        expansion_lens = expansion_lens[0]
        pos_discard = pos_discard[0]
        assert_vec(pos_expand, [0, 3, 5])
        assert_vec(expansion_lens, [2, 2, 3])
        assert_vec(pos_discard, [1])


def test_extend_mods() -> None:
    """Test extension of modification positions."""
    pos_expand = np.array([2, 3, 5])
    expansion_lens = np.array([9, 1, 3])
    pos_discard = np.array([4])
    e, l, d = Align.extend_mods(pos_expand, expansion_lens, pos_discard, L=5)
    assert_vec(d, [1, 2, 3])
    assert_vec(e, [1, 2, 4])
    assert_vec(l, [10, 2, 3])
    e, l, d = Align.extend_mods(pos_expand, expansion_lens, pos_discard, L=6, k=1)
    assert_vec(d, [2, 3, 4])
    assert_vec(e, [2, 3, 5])
    assert_vec(l, [10, 2, 3])
    e, l, d = Align.extend_mods(pos_expand, expansion_lens, pos_discard, L=6)
    assert_vec(d, [1, 2, 3, 4])
    assert_vec(e, [1, 2, 3, 4])
    assert_vec(l, [10, 2, 1, 3])


def test_update_kernels() -> None:
    """Test updating model kernels during surgery."""
    filename = os.path.dirname(__file__) + "/../tests/data/felix_insert_delete.fa"
    with SequenceDataset(filename) as data:
        am = make_test_alignment(data)
        pos_expand = np.array([2, 3, 5])
        expansion_lens = np.array([9, 1, 3])
        pos_discard = np.array([4])
        emission_init2 = [Initializers.ConstantInitializer(string_to_one_hot("A").numpy() * 10)]
        transition_init2 = {
            "begin_to_match": Initializers.ConstantInitializer(77),
            "match_to_end": Initializers.ConstantInitializer(77),
            "match_to_match": Initializers.ConstantInitializer(77),
            "match_to_insert": Initializers.ConstantInitializer(77),
            "insert_to_match": Initializers.ConstantInitializer(77),
            "insert_to_insert": Initializers.ConstantInitializer(77),
            "match_to_delete": Initializers.ConstantInitializer(77),
            "delete_to_match": Initializers.ConstantInitializer(77),
            "delete_to_delete": Initializers.ConstantInitializer(77),
            "left_flank_loop": Initializers.ConstantInitializer(77),
            "left_flank_exit": Initializers.ConstantInitializer(77),
            "right_flank_loop": Initializers.ConstantInitializer(77),
            "right_flank_exit": Initializers.ConstantInitializer(77),
            "unannotated_segment_loop": Initializers.ConstantInitializer(77),
            "unannotated_segment_exit": Initializers.ConstantInitializer(77),
            "end_to_unannotated_segment": Initializers.ConstantInitializer(77),
            "end_to_right_flank": Initializers.ConstantInitializer(77),
            "end_to_terminal": Initializers.ConstantInitializer(77)
        }
        transitions_new, emissions_new, _ = Align.update_kernels(
            am, 0,
            pos_expand, expansion_lens, pos_discard,
            emission_init2, transition_init2, Initializers.ConstantInitializer(0.0)
        )
    ref_consensus = "FE" + "A" * 9 + "LAI" + "A" * 3
    assert emissions_new[0].shape[0] == len(ref_consensus)
    assert_vec(emissions_new[0], string_to_one_hot(ref_consensus).numpy() * 10)
    assert_vec(transitions_new["begin_to_match"], [1, 0] + [77] * 9 + [0, 77, 0, 77, 77, 77])
    assert_vec(transitions_new["match_to_end"], [0, 0] + [77] * 9 + [0, 77, 0, 77, 77, 77])
    assert_vec(transitions_new["match_to_match"], [0] + [77] * 15)
    assert_vec(transitions_new["match_to_insert"], [0] + [77] * 15)
    assert_vec(transitions_new["insert_to_match"], [0] + [77] * 15)
    assert_vec(transitions_new["insert_to_insert"], [1] + [77] * 15)
    assert_vec(transitions_new["match_to_delete"], [-1, -1] + [77] * 15)
    assert_vec(transitions_new["delete_to_match"], [1] + [77] * 16)


def test_apply_mods() -> None:
    """Test applying modifications to arrays."""
    x1 = Align.apply_mods(
        x=list(range(10)),
        pos_expand=[0, 4, 7, 10],
        expansion_lens=[2, 1, 2, 1],
        pos_discard=[4, 6, 7, 8, 9],
        insert_value=55
    )
    assert_vec(x1, [55, 55, 0, 1, 2, 3, 55, 5, 55, 55, 55])
    x2 = Align.apply_mods(
        x=[[1, 2, 3], [1, 2, 3]],
        pos_expand=[1],
        expansion_lens=[1],
        pos_discard=[],
        insert_value=[4, 5, 6]
    )
    assert_vec(x2, [[1, 2, 3], [4, 5, 6], [1, 2, 3]])

    # Remark: This was a special solution that failed under practical circumstances
    # I added the scenario to the tests and reduced it to the problem core below
    L = 240
    exp = np.array([0, 1, 10, 25, 26, 27, 30, 31, 36, 66, 71, 89, 95,
                    102, 123, 124, 125, 126, 138, 154, 183, 192, 193, 203, 204,
                    221, 222, 223, 231, 232, 233, 234, 240])
    lens = np.array([2, 1, 1, 1, 1, 1, 1, 5, 1, 1, 5, 10, 1, 1, 3, 1, 1, 1,
                     9, 1, 1, 8, 2, 1, 4, 1, 1, 2, 2, 1, 2, 1, 3])
    dis = np.array([13, 84, 129, 130])
    new_pos_expand, new_expansion_lens, new_pos_discard = Align.extend_mods(exp, lens, dis, L)
    x3 = Align.apply_mods(
        list(range(L - 1)),
        new_pos_expand, new_expansion_lens, new_pos_discard,
        insert_value=-1
    )
    assert x3.size == L - 1 - dis.size + np.sum(lens)

    # Problem core of the above issue, solved by handling as a special case
    L = 5
    exp = np.array([0, 1])
    lens = np.array([1, 1])
    dis = np.array([1])
    new_pos_expand, new_expansion_lens, new_pos_discard = Align.extend_mods(exp, lens, dis, L)
    assert_vec(new_pos_expand, [0])
    assert_vec(new_expansion_lens, [3])
    assert_vec(new_pos_discard, [0, 1])
    x3 = Align.apply_mods(
        list(range(L - 1)),
        new_pos_expand, new_expansion_lens, new_pos_discard,
        insert_value=-1
    )
    assert_vec(x3, [-1, -1, -1, 2, 3])

    L = 5
    exp = np.array([0, 1])
    lens = np.array([9, 1])
    dis = np.array([])
    new_pos_expand, new_expansion_lens, new_pos_discard = Align.extend_mods(exp, lens, dis, L)
    x3 = Align.apply_mods(
        list(range(L - 1)),
        new_pos_expand, new_expansion_lens, new_pos_discard,
        insert_value=-1
    )
    assert x3.size == L - 1 - dis.size + np.sum(lens)

    L = 10
    exp = np.array([0, L - 1])
    lens = np.array([5, 5])
    dis = np.arange(L)
    new_pos_expand, new_expansion_lens, new_pos_discard = Align.extend_mods(exp, lens, dis, L)
    x4 = Align.apply_mods(
        list(range(L - 1)),
        new_pos_expand, new_expansion_lens, new_pos_discard,
        insert_value=-1
    )
    assert x4.size == L - 1 - dis.size + np.sum(lens)


def test_checked_concat() -> None:
    """Test concatenation with checking for edge cases."""
    e, l, d = Align.extend_mods(
        pos_expand=np.array([]),
        expansion_lens=np.array([]),
        pos_discard=np.array([0, 2, 4, 5, 6, 9, 10]),
        L=11
    )
    assert_vec(e, [1, 3])
    assert_vec(l, [1, 1])
    assert_vec(d, [0, 1, 2, 3, 4, 5, 6, 8, 9])
    e, l, d = Align.extend_mods(
        pos_expand=np.array([0, 4, 9, 10, 11]),
        expansion_lens=np.array([2, 1, 2, 3, 1]),
        pos_discard=np.array([]),
        L=11
    )
    assert_vec(e, [0, 3, 8, 9, 10])
    assert_vec(l, [2, 2, 3, 4, 1])
    assert_vec(d, [3, 8, 9])
    e, l, d = Align.extend_mods(
        pos_expand=np.array([]),
        expansion_lens=np.array([]),
        pos_discard=np.array([1]),
        L=11, k=1
    )
    assert_vec(e, [1])
    assert_vec(l, [1])
    assert_vec(d, [1, 2])
    e, l, d = Align.extend_mods(
        pos_expand=np.array([5]),
        expansion_lens=np.array([3]),
        pos_discard=np.array(list(range(8))),
        L=8
    )
    assert_vec(e, [4])
    assert_vec(l, [2])
    assert_vec(d, list(range(7)))
    e, l, d = Align.extend_mods(
        pos_expand=np.array([5]),
        expansion_lens=np.array([3]),
        pos_discard=np.array(list(range(8))),
        L=9, k=1
    )
    assert_vec(e, [5])
    assert_vec(l, [3])
    assert_vec(d, list(range(8)))

    e, l, d = Align.extend_mods(
        pos_expand=np.array([5]),
        expansion_lens=np.array([3]),
        pos_discard=np.array([0, 1, 2, 4, 5, 6, 7]),
        L=8
    )
    assert_vec(e, [4])
    assert_vec(l, [3])
    assert_vec(d, list(range(7)))
    e, l, d = Align.extend_mods(
        pos_expand=np.array([5]),
        expansion_lens=np.array([3]),
        pos_discard=np.array([0, 1, 2, 4, 5, 6, 7]),
        L=9, k=1
    )
    assert_vec(e, [0, 5])
    assert_vec(l, [1, 3])
    assert_vec(d, list(range(8)))
    e, l, d = Align.extend_mods(
        pos_expand=np.array([0, 10]),
        expansion_lens=np.array([5, 5]),
        pos_discard=np.arange(10),
        L=10
    )
    assert_vec(e, [0, 9])
    assert_vec(l, [4, 5])
    assert_vec(d, np.arange(9))


def test_whole_surgery() -> None:
    """Test whole surgery process."""
    pass
