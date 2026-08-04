"""Fixtures shared by the neutral and the TensorFlow alignment tests.

Built with :func:`make_learnmsa_model` rather than a backend class, so this
conftest imports no tensor framework -- which is what lets ``tests/align/tf``
inherit the same fixtures without the neutral tests inheriting TensorFlow.

``simple_model`` is a hand-parameterised two-head pHMM (FELIK, length 5, and
AHC, length 3) over ``tests/data/felix.fa``; every transition and emission
probability is pinned in ``simple_config`` so the Viterbi paths the tests
assert are exactly reproducible.
"""

import os

import numpy as np
import pytest

from learnMSA import Configuration
from learnMSA.model.context import LearnMSAContext
from learnMSA.model.model import LearnMSAModel, make_learnmsa_model
from learnMSA.util.sequence_dataset import SequenceDataset

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


@pytest.fixture
def simple_data() -> SequenceDataset:
    return SequenceDataset(os.path.join(DATA, "felix.fa"))

@pytest.fixture
def multi_hit_data() -> SequenceDataset:
    return SequenceDataset(os.path.join(DATA, "felix_multi_hit.fa"))

@pytest.fixture
def simple_config() -> Configuration:
    config = Configuration()
    config.training.num_model = 1
    config.training.no_sequence_weights = True
    config.training.length_init = [5, 3]
    alphabet = SequenceDataset._default_alphabet

    # Create FELIK model (length 5)
    felik_indices = [alphabet.index(aa) for aa in "FELIK"]
    ahc_indices = [alphabet.index(aa) for aa in "AHC"]
    match_emissions = np.zeros((2, 5, len(alphabet)))
    for i, aa_idx in enumerate(felik_indices):
        match_emissions[0, i, aa_idx] = 1.0
    for i, aa_idx in enumerate(ahc_indices):
        match_emissions[1, i, aa_idx] = 1.0
    config.hmm.match_emissions = match_emissions
    config.hmm.insert_emissions = [1/len(alphabet)]*len(alphabet)
    config.hmm.use_prior_for_emission_init = False
    config.hmm.p_end_right = 0.2
    config.hmm.p_end_unannot = 0.3
    config.hmm.p_match_match = 0.5
    config.hmm.p_unannot_unannot = 0.5
    config.hmm.p_left_left = 0.5
    config.hmm.p_right_right = 0.4
    config.hmm.p_begin_match = 0.6
    config.hmm.p_begin_delete = 0.2
    config.hmm.shared_flank_transitions = False

    return config

@pytest.fixture
def simple_context(
    simple_data: SequenceDataset,
    simple_config: Configuration,
) -> LearnMSAContext:
    """Fixture for a LearnMSAContext for FELIK model (single head)."""
    # Create context and set phmm_config
    context = LearnMSAContext(simple_config, simple_data)
    return context

@pytest.fixture
def simple_model(
    simple_context: LearnMSAContext
) -> LearnMSAModel:
    """Fixture for a LearnMSAModel with FELIK model."""
    model = make_learnmsa_model(simple_context)
    model.build()
    return model
@pytest.fixture
def viterbi_seqs() -> np.ndarray:
    # Reference Viterbi state sequences (pre-computed)
    # States: [MATCH x length, INSERT x length-1, LEFT_FLANK, UNANNOTATED_SEGMENT, RIGHT_FLANK, END]
    viterbi_seqs = np.array([
        # model 1 (FELIK - length 5)
        [[0, 1, 2, 3, 4, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
         [9, 9, 9, 0, 1, 2, 3, 4, 12, 12, 12, 12, 12, 12, 12],
         [0, 1, 2, 3, 4, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12],
         [0, 1, 2, 3, 4, 10, 10, 10, 0, 1, 2, 3, 4, 11, 12],
         [9, 1, 2, 3, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
         [0, 1, 6, 6, 6, 2, 3, 4, 12, 12, 12, 12, 12, 12, 12],
         [0, 5, 5, 1, 2, 7, 3, 8, 8, 8, 4, 12, 12, 12, 12],
         [0, 1, 2, 7, 7, 7, 3, 4, 11, 11, 11, 12, 12, 12, 12]],
        # model 2 (AHC - length 3)
        [[5, 5, 5, 5, 5, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
         [0, 1, 2, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8],
         [5, 5, 5, 5, 5, 5, 0, 2, 8, 8, 8, 8, 8, 8, 8],
         [5, 5, 5, 5, 5, 0, 1, 2, 6, 6, 6, 6, 6, 0, 8],
         [0, 3, 3, 3, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
         [5, 5, 0, 1, 2, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8],
         [5, 0, 1, 6, 6, 0, 6, 0, 1, 2, 7, 8, 8, 8, 8],
         [5, 5, 5, 0, 1, 2, 6, 6, 0, 1, 2, 8, 8, 8, 8]]
    ])
    return viterbi_seqs