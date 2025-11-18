import numpy as np
import pytest
from hidten.hmm import HMMConfig as HidtenHMMConfig

import tests.hmm.ref as ref
from learnMSA.config.hmm import HMMConfig
from learnMSA.hmm.transition_index_set import PHMMTransitionIndexSet
from learnMSA.hmm.transitioner import (PHMMExplicitTransitioner,
                                       PHMMTransitioner)
from learnMSA.hmm.value_set import PHMMValueSet


@pytest.fixture(autouse=True)
def config() -> HMMConfig:
    # Create a configuration with some example probabilities
    # Some parameters are head-specific
   return HMMConfig(
        p_begin_match=0.5,
        p_match_match=[0.7, 0.5], # Different for two heads
        p_match_insert=0.1,
        p_match_end=0.05,
        p_insert_insert=0.4,
        p_delete_delete=0.3,
        p_begin_delete=0.2,
        p_left_left=0.8,
        p_right_right=0.8,
        p_unannot_unannot=0.7,
        p_end_unannot=1e-4,
        p_end_right=0.6,
        p_start_left_flank=[0.2, 0.3], # Different for two heads
    )

def test_explicit_transitioner_matrix(config: HMMConfig) -> None:
    lengths = [4, 3]

    # Create value sets for different heads
    values = [
        PHMMValueSet.from_config(L, h, config) for h, L in enumerate(lengths)
    ]

    # We need to manually create a Hidten HMMConfig because the transitioner is
    # not added to an HMM here.
    states= [
        PHMMTransitionIndexSet(L=L, folded=False).num_states
        for L in lengths
    ]
    hidten_hmm_config = HidtenHMMConfig(states=states)

    # Construct a transitioner with two heads from the initial values
    transitioner = PHMMExplicitTransitioner(values, hidten_hmm_config)
    transitioner.build()

    S = transitioner.start_dist()
    A = transitioner.matrix()

    # Check start distribution
    # Head 0
    np.testing.assert_allclose(S[0,:3*lengths[0]-1], 0.0)
    np.testing.assert_allclose(S[0,3*lengths[0]-1:3*lengths[0]+1], [0.2, 0.8])  # L, B
    np.testing.assert_allclose(S[0,3*lengths[0]+1:], 0.0)
    # Head 1
    np.testing.assert_allclose(S[1,:3*lengths[1]-1], 0.0)
    np.testing.assert_allclose(S[1,3*lengths[1]-1:3*lengths[1]+1], [0.3, 0.7])  # L, B
    np.testing.assert_allclose(S[1,3*lengths[1]+1:], 0.0)

    # Check the transition probabilities
    # Head 0
    np.testing.assert_allclose(
        np.sum(A[0, :states[0]], axis=-1), 1.0, atol=1e-6
    )
    # Head 1
    np.testing.assert_allclose(
        np.sum(A[1, :states[1]], axis=-1), 1.0, atol=1e-6
    )

def test_folded_transitioner_matrix(config: HMMConfig) -> None:
    lengths = [4, 3]

    # Create value sets for different heads
    values = [
        PHMMValueSet.from_config(L, h, config) for h, L in enumerate(lengths)
    ]

    # We need to manually create a Hidten HMMConfig because the transitioner is
    # not added to an HMM here.
    states= [
        PHMMTransitionIndexSet(L=L, folded=False).num_states
        for L in lengths
    ]
    hidten_hmm_config = HidtenHMMConfig(states=states)
    transitioner = PHMMTransitioner(
        values=values,
        hidten_hmm_config=hidten_hmm_config
    )