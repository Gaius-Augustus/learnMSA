"""Tests for model surgery operations (expanding/discarding positions)."""
import os

import numpy as np
import pytest
import tensorflow as tf

from learnMSA import Configuration
from learnMSA.config.hmm import PHMMConfig
from learnMSA.hmm.tf.layer import PHMMLayer
from learnMSA.hmm.util.transition_index_set import PHMMTransitionIndexSet
from learnMSA.model.tf.model import LearnMSAModel
from learnMSA.align.alignment_model import AlignmentModel
from learnMSA.model.context import LearnMSAContext
from learnMSA.model.surgery import (apply_mods, extend_mods,
                                            get_discard_or_expand_positions,
                                            update_kernels, model_surgery)
from learnMSA.util.sequence_dataset import SequenceDataset


def string_to_one_hot(s : str) -> tf.Tensor:
    """Convert a string to one-hot encoded tensor."""
    i = [SequenceDataset._default_alphabet.index(aa) for aa in s]
    return tf.one_hot(i, len(SequenceDataset._default_alphabet) - 1)

@pytest.fixture
def data() -> SequenceDataset:
    """Create a test sequence dataset."""
    data = SequenceDataset(sequences=[(str(i), "FELIK") for i in range(10)])
    return data

@pytest.fixture
def model(data: SequenceDataset) -> LearnMSAModel:
    """Create a test alignment model with two heads: FEIK and FDELIK."""
    alphabet = SequenceDataset._default_alphabet

    # First head: "FEIK" (4 states)
    match_emissions_h1 = np.zeros((4, len(alphabet) - 1))
    for i, aa in enumerate("FEIK"):
        match_emissions_h1[i, alphabet.index(aa)] = 1.0

    # Second head: "FDELIK" (6 states)
    match_emissions_h2 = np.zeros((6, len(alphabet) - 1))
    for i, aa in enumerate("FDELIK"):
        match_emissions_h2[i, alphabet.index(aa)] = 1.0

    # Pad the shorter model to match the longer one
    max_len = max(4, 6)
    match_emissions_h1_padded = np.zeros((max_len, len(alphabet) - 1))
    match_emissions_h1_padded[:4] = match_emissions_h1

    # Create 3D array for match emissions
    match_emissions = np.array([match_emissions_h1_padded, match_emissions_h2])

    insert_emissions = np.ones((2, len(alphabet) - 1)) / (len(alphabet) - 1)

    # Create LearnMSAModel
    learnmsa_config = Configuration()
    learnmsa_config.training.num_model = 2
    learnmsa_config.training.no_sequence_weights = True
    learnmsa_config.training.length_init = [4, 6]
    learnmsa_config.hmm.match_emissions = match_emissions
    learnmsa_config.hmm.insert_emissions = insert_emissions
    learnmsa_config.hmm.use_prior_for_emission_init = False
    learnmsa_config.hmm.p_begin_match = 0.9
    learnmsa_config.hmm.p_match_end = 0.2
    learnmsa_config.hmm.p_match_match = 0.65
    learnmsa_config.hmm.p_match_insert = 0.1
    learnmsa_config.hmm.p_insert_insert = 0.01
    learnmsa_config.hmm.p_delete_delete = 0.7
    learnmsa_config.hmm.p_begin_delete = 0.05
    learnmsa_config.hmm.p_left_left = 0.5
    learnmsa_config.hmm.p_right_right = 0.5
    learnmsa_config.hmm.p_unannot_unannot = 0.5
    learnmsa_config.hmm.p_end_unannot = 0.3
    learnmsa_config.hmm.p_end_right = 0.3
    learnmsa_config.hmm.p_start_left_flank = 0.1

    context = LearnMSAContext(learnmsa_config, data)

    model = LearnMSAModel(context)
    model.build()

    return model

@pytest.fixture
def data_insert_delete() -> SequenceDataset:
    """Create a test sequence dataset."""
    test_data_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "felix_insert_delete.fa"
    )
    data = SequenceDataset(test_data_path)
    return data

@pytest.fixture
def model_single_head(data: SequenceDataset) -> LearnMSAModel:
    """Create a single-head test model with FELIC motif for legacy tests."""
    alphabet = SequenceDataset._default_alphabet
    match_emissions = np.zeros((1, 5, len(alphabet) - 1))
    for i, aa in enumerate("FELIC"):
        match_emissions[0, i, alphabet.index(aa)] = 1.0
    insert_emissions = np.zeros((1, len(alphabet) - 1))
    for i, aa in enumerate("AN"):
        insert_emissions[0, alphabet.index(aa)] = 0.5

    # Create LearnMSAModel
    learnmsa_config = Configuration()
    learnmsa_config.training.num_model = 1
    learnmsa_config.training.no_sequence_weights = True
    learnmsa_config.training.length_init = [5]
    learnmsa_config.hmm.match_emissions = match_emissions
    learnmsa_config.hmm.insert_emissions=insert_emissions
    learnmsa_config.hmm.use_prior_for_emission_init = False
    learnmsa_config.hmm.p_begin_match = [
        [0.35, 0.15, 0.15, 0.15, 0.15]
    ]
    learnmsa_config.hmm.p_match_end = 0.3
    learnmsa_config.hmm.p_match_match = 0.3
    learnmsa_config.hmm.p_match_insert = 0.3
    learnmsa_config.hmm.p_match_delete = 0.1
    learnmsa_config.hmm.p_insert_insert = 0.7
    learnmsa_config.hmm.p_delete_delete = 0.3
    learnmsa_config.hmm.p_begin_delete = 0.05
    learnmsa_config.hmm.p_left_left = 0.5
    learnmsa_config.hmm.p_right_right = 0.5
    learnmsa_config.hmm.p_unannot_unannot = 0.5
    learnmsa_config.hmm.p_end_unannot = 0.3
    learnmsa_config.hmm.p_end_right = 0.3
    learnmsa_config.hmm.p_start_left_flank = 0.5

    context = LearnMSAContext(learnmsa_config, data)

    model = LearnMSAModel(context)
    model.build()

    return model


def test_discard_or_expand_positions(
    model_single_head: LearnMSAModel, data_insert_delete: SequenceDataset
) -> None:
    """Test detection of positions to discard or expand."""
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

    am = AlignmentModel(data_insert_delete, model_single_head)

    # Decode the alignment depending on the sequences and model parameters
    aligned_sequences = am.to_string(model_index=0, add_block_sep=False)

    # Ensure the alignment is as expected
    for s, ref_s in zip(aligned_sequences, ref_seqs):
        assert s == ref_s

    assert 0 in am.metadata
    assert am.metadata[0].alignment_len == len(ref_seqs[0])
    # Shape: [number of domain hits, length]
    deletions = np.sum(am.metadata[0].consensus == -1, axis=1)
    np.testing.assert_equal(deletions, [[4, 5, 2, 0, 4], [2, 5, 4, 2, 4]])
    # Shape: [number of domain hits, num seq]
    np.testing.assert_equal(
        am.metadata[0].finished,
        [[False, False, True, True, False, False],
            [True, True, True, True, True, True]]
    )
    # Shape: [number of domain hits, num seq, L-1 inner positions]
    np.testing.assert_equal(
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
    pos_expand, expansion_lens, pos_discard = get_discard_or_expand_positions(
        model_single_head, data_insert_delete, del_t=0.5, ins_t=0.5
    )
    pos_expand = pos_expand[0]
    expansion_lens = expansion_lens[0]
    pos_discard = pos_discard[0]
    np.testing.assert_equal(pos_expand, [0, 3, 5])
    np.testing.assert_equal(expansion_lens, [2, 2, 3])
    np.testing.assert_equal(pos_discard, [1])

def test_update_kernels(model_single_head: LearnMSAModel) -> None:
    """Test updating model kernels during surgery."""
    pos_expand = np.array([2, 3, 5])
    expansion_lens = np.array([2, 1, 3])
    pos_discard = np.array([4])

    # Expanded positions will emit "A"
    alphabet = SequenceDataset._default_alphabet
    dummy_emission = np.zeros((len(alphabet) - 1,))
    dummy_emission[alphabet.index("A")] = 1.0

    config = model_single_head.context.config.hmm.model_copy(deep=True)
    # Override some parameters to check if they are correctly used as updates
    config.background_distribution = dummy_emission
    config.p_begin_match = 0.5
    config.p_match_end = 0.2
    config.p_match_match = 0.1
    config.p_match_insert = 0.5
    config.p_match_delete = 0.2
    config.p_insert_insert = 0.4
    config.p_delete_delete = 0.2

    # Update the pHMM with the specified modifications and obtains a
    # new config
    result = update_kernels(
        model_single_head.phmm_layer,
        0,
        pos_expand,
        expansion_lens,
        pos_discard,
        config,
    )

    # Create a new HMM layer from the config
    updated_phmm_layer = PHMMLayer([result.length], result.config)
    updated_phmm_layer.build(((None, None, None, 23), (None, None, None, 1)))

    emissions_new = updated_phmm_layer.hmm.emitter[0].matrix().numpy()[0]
    transitions_new = updated_phmm_layer.hmm.transitioner\
        .explicit_transitioner.matrix().numpy()[0]

    ref_consensus = "FE" + "A" * 2 + "LAI" + "A" * 3
    assert result.length == len(ref_consensus)

    # Create expected emissions for the reference consensus
    expected_emissions = np.zeros((len(ref_consensus), len(alphabet) - 1))
    for i, aa in enumerate(ref_consensus):
        expected_emissions[i, alphabet.index(aa)] = 1.0

    np.testing.assert_equal(emissions_new[:result.length], expected_emissions)

    ind = PHMMTransitionIndexSet(result.length)

    # Begin to match transitions must have been reset to defaults
    np.testing.assert_almost_equal(
        transitions_new[ind.begin_to_match[:,0], ind.begin_to_match[:,1]],
        [0.5] + [(0.5 - 0.05) / (result.length - 1)] * (result.length - 1)
    )
    # Same for match to end transitions
    np.testing.assert_almost_equal(
        transitions_new[ind.match_to_end[:,0], ind.match_to_end[:,1]],
        [0.2] * (result.length-1) + [1.0]
    )
    np.testing.assert_almost_equal(
        transitions_new[ind.match_to_match[:,0], ind.match_to_match[:,1]],
        [.3] + [.1] * 8
    )
    np.testing.assert_almost_equal(
        transitions_new[ind.match_to_insert[:,0], ind.match_to_insert[:,1]],
        [.3] + [.5] * 8
    )
    np.testing.assert_almost_equal(
        transitions_new[ind.insert_to_insert[:,0], ind.insert_to_insert[:,1]],
        [.7] + [.4] * 8
    )
    np.testing.assert_almost_equal(
        transitions_new[ind.insert_to_match[:,0], ind.insert_to_match[:,1]],
        [.3] + [.6] * 8
    )
    np.testing.assert_almost_equal(
        transitions_new[ind.delete_to_delete[:,0], ind.delete_to_delete[:,1]],
        [.3] + [.2] * 8
    )

def test_apply_mods() -> None:
    """Test applying modifications to arrays."""
    x1 = apply_mods(
        x=np.array(range(10)),
        pos_expand=np.array([0, 4, 7, 10]),
        expansion_lens=np.array([2, 1, 2, 1]),
        pos_discard=np.array([4, 6, 7, 8, 9]),
        insert_value=55
    )
    np.testing.assert_equal(x1, [55, 55, 0, 1, 2, 3, 55, 5, 55, 55, 55])
    x2 = apply_mods(
        x=np.array([[1, 2, 3], [1, 2, 3]]),
        pos_expand=np.array([1]),
        expansion_lens=np.array([1]),
        pos_discard=np.array([], dtype=np.int32),
        insert_value=np.array([4, 5, 6])
    )
    np.testing.assert_equal(x2, [[1, 2, 3], [4, 5, 6], [1, 2, 3]])

def test_extend_mods() -> None:
    """Test extension of modification positions."""
    pos_expand = np.array([2, 3, 5])
    expansion_lens = np.array([9, 1, 3])
    pos_discard = np.array([4])
    e, l, d = extend_mods(pos_expand, expansion_lens, pos_discard, L=5)
    np.testing.assert_equal(d, [1, 2, 3])
    np.testing.assert_equal(e, [1, 2, 4])
    np.testing.assert_equal(l, [10, 2, 3])
    e, l, d = extend_mods(pos_expand, expansion_lens, pos_discard, L=6, k=1)
    np.testing.assert_equal(d, [2, 3, 4])
    np.testing.assert_equal(e, [2, 3, 5])
    np.testing.assert_equal(l, [10, 2, 3])
    e, l, d = extend_mods(pos_expand, expansion_lens, pos_discard, L=6)
    np.testing.assert_equal(d, [1, 2, 3, 4])
    np.testing.assert_equal(e, [1, 2, 3, 4])
    np.testing.assert_equal(l, [10, 2, 1, 3])

def test_apply_and_extend_mods() -> None:
    """This was a special thing that failed when testing alignments of
    real data, due to the combination of many expansions and discards.
    Simplified test case to reproduce the problem."""
    L = 240
    exp = np.array([0, 1, 10, 25, 26, 27, 30, 31, 36, 66, 71, 89, 95,
                    102, 123, 124, 125, 126, 138, 154, 183, 192, 193, 203, 204,
                    221, 222, 223, 231, 232, 233, 234, 240])
    lens = np.array([2, 1, 1, 1, 1, 1, 1, 5, 1, 1, 5, 10, 1, 1, 3, 1, 1, 1,
                     9, 1, 1, 8, 2, 1, 4, 1, 1, 2, 2, 1, 2, 1, 3])
    dis = np.array([13, 84, 129, 130])
    new_pos_expand, new_expansion_lens, new_pos_discard = extend_mods(
        exp, lens, dis, L
    )
    x3 = apply_mods(
        np.array(list(range(L - 1))),
        new_pos_expand, new_expansion_lens, new_pos_discard,
        insert_value=-1
    )
    assert x3.size == L - 1 - dis.size + np.sum(lens)

    # Problem core of the above issue, solved by handling as a special case
    L = 5
    exp = np.array([0, 1])
    lens = np.array([1, 1])
    dis = np.array([1])
    new_pos_expand, new_expansion_lens, new_pos_discard = extend_mods(
        exp, lens, dis, L
    )
    np.testing.assert_equal(new_pos_expand, [0])
    np.testing.assert_equal(new_expansion_lens, [3])
    np.testing.assert_equal(new_pos_discard, [0, 1])
    x3 = apply_mods(
        np.array(list(range(L - 1))),
        new_pos_expand, new_expansion_lens, new_pos_discard,
        insert_value=-1
    )
    np.testing.assert_equal(x3, [-1, -1, -1, 2, 3])

    L = 5
    exp = np.array([0, 1])
    lens = np.array([9, 1])
    dis = np.array([], dtype=np.int32)
    new_pos_expand, new_expansion_lens, new_pos_discard = extend_mods(
        exp, lens, dis, L
    )
    x3 = apply_mods(
        np.array(list(range(L - 1))),
        new_pos_expand, new_expansion_lens, new_pos_discard,
        insert_value=-1
    )
    assert x3.size == L - 1 - dis.size + np.sum(lens)

    L = 10
    exp = np.array([0, L - 1])
    lens = np.array([5, 5])
    dis = np.arange(L)
    new_pos_expand, new_expansion_lens, new_pos_discard = extend_mods(
        exp, lens, dis, L
    )
    x4 = apply_mods(
        np.array(list(range(L - 1))),
        new_pos_expand, new_expansion_lens, new_pos_discard,
        insert_value=-1
    )
    assert x4.size == L - 1 - dis.size + np.sum(lens)

def test_checked_concat() -> None:
    """Test concatenation with checking for edge cases."""
    e, l, d = extend_mods(
        pos_expand=np.array([], dtype=np.int32),
        expansion_lens=np.array([], dtype=np.int32),
        pos_discard=np.array([0, 2, 4, 5, 6, 9, 10]),
        L=11
    )
    np.testing.assert_equal(e, [1, 3])
    np.testing.assert_equal(l, [1, 1])
    np.testing.assert_equal(d, [0, 1, 2, 3, 4, 5, 6, 8, 9])
    e, l, d = extend_mods(
        pos_expand=np.array([0, 4, 9, 10, 11]),
        expansion_lens=np.array([2, 1, 2, 3, 1]),
        pos_discard=np.array([], dtype=np.int32),
        L=11
    )
    np.testing.assert_equal(e, [0, 3, 8, 9, 10])
    np.testing.assert_equal(l, [2, 2, 3, 4, 1])
    np.testing.assert_equal(d, [3, 8, 9])
    e, l, d = extend_mods(
        pos_expand=np.array([], dtype=np.int32),
        expansion_lens=np.array([], dtype=np.int32),
        pos_discard=np.array([1]),
        L=11, k=1
    )
    np.testing.assert_equal(e, [1])
    np.testing.assert_equal(l, [1])
    np.testing.assert_equal(d, [1, 2])
    e, l, d = extend_mods(
        pos_expand=np.array([5]),
        expansion_lens=np.array([3]),
        pos_discard=np.array(list(range(8))),
        L=8
    )
    np.testing.assert_equal(e, [4])
    np.testing.assert_equal(l, [2])
    np.testing.assert_equal(d, list(range(7)))
    e, l, d = extend_mods(
        pos_expand=np.array([5]),
        expansion_lens=np.array([3]),
        pos_discard=np.array(list(range(8))),
        L=9, k=1
    )
    np.testing.assert_equal(e, [5])
    np.testing.assert_equal(l, [3])
    np.testing.assert_equal(d, list(range(8)))

    e, l, d = extend_mods(
        pos_expand=np.array([5]),
        expansion_lens=np.array([3]),
        pos_discard=np.array([0, 1, 2, 4, 5, 6, 7]),
        L=8
    )
    np.testing.assert_equal(e, [4])
    np.testing.assert_equal(l, [3])
    np.testing.assert_equal(d, list(range(7)))
    e, l, d = extend_mods(
        pos_expand=np.array([5]),
        expansion_lens=np.array([3]),
        pos_discard=np.array([0, 1, 2, 4, 5, 6, 7]),
        L=9, k=1
    )
    np.testing.assert_equal(e, [0, 5])
    np.testing.assert_equal(l, [1, 3])
    np.testing.assert_equal(d, list(range(8)))
    e, l, d = extend_mods(
        pos_expand=np.array([0, 10]),
        expansion_lens=np.array([5, 5]),
        pos_discard=np.arange(10),
        L=10
    )
    np.testing.assert_equal(e, [0, 9])
    np.testing.assert_equal(l, [4, 5])
    np.testing.assert_equal(d, np.arange(9))


def test_model_surgery(data: SequenceDataset, model: LearnMSAModel) -> None:
    """Test model surgery with two heads: FEIK should expand at position 2,
    FDELIK should discard position 1."""
    # Run model surgery
    result = model_surgery(
        model=model,
        data=data,
        surgery_del=0.5,
        surgery_ins=0.5,
    )

    # Verify we got results for both heads
    assert len(result.model_lengths) == 2

    # Original: F E I K (4 states)
    # Expected after expansion: F E L I K (5 states)
    assert result.model_lengths[0] == 5,\
        f"Expected first head to have 6 states, got {result.model_lengths[0]}"

    # Original: F D E L I K (6 states)
    # Expected after discard: F E L I K (5 states)
    assert result.model_lengths[1] == 5,\
        f"Expected second head to have 5 states, got {result.model_lengths[1]}"

    # Surgery should not have converged (modifications were applied)
    assert not result.surgery_converged
