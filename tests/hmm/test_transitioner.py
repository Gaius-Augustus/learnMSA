import numpy as np
from hidten.hmm import HMMConfig as HidtenHMMConfig

import tests.hmm.ref as ref
from learnMSA.config.hmm import HMMConfig
from learnMSA.hmm.transition_index_set import PHMMTransitionIndexSet
from learnMSA.hmm.transitioner import (PHMMExplicitTransitioner,
                                       PHMMTransitioner)
from learnMSA.hmm.value_set import PHMMValueSet


def test_explicit_transitioner_matrix() -> None:
    lengths = [4, 3]

    # Create value sets for different heads
    values = [
        PHMMValueSet.from_config(L, h, ref.config)
        for h, L in enumerate(lengths)
    ]

    # We need to manually create a Hidten HMMConfig because the transitioner is
    # not added to an HMM here.
    states = [PHMMTransitionIndexSet.num_states_unfolded(L=L) for L in lengths]
    hidten_hmm_config = HidtenHMMConfig(states=states)

    # Construct a transitioner with two heads from the initial values
    transitioner = PHMMExplicitTransitioner(values, hidten_hmm_config)
    transitioner.build()

    S = transitioner.start_dist()
    A = transitioner.matrix()

    # Check start distribution
    # Head 0
    np.testing.assert_allclose(S[0,:3*lengths[0]-1], 0.0)
    np.testing.assert_allclose(S[0,3*lengths[0]-1:3*lengths[0]+1], [0.5, 0.5])  # L, B
    np.testing.assert_allclose(S[0,3*lengths[0]+1:], 0.0)
    # Head 1
    np.testing.assert_allclose(S[1,:3*lengths[1]-1], 0.0)
    np.testing.assert_allclose(S[1,3*lengths[1]-1:3*lengths[1]+1], [0.5, 0.5])  # L, B
    np.testing.assert_allclose(S[1,3*lengths[1]+1:], 0.0)

    # Check the transition probabilities
    # Head 0 is probabilistic
    np.testing.assert_allclose(
        np.sum(A[0, :states[0]], axis=-1), 1.0, atol=1e-6
    )
    # M1 ... ML I1 ... IL-1 D1 ... DL L B E C R T
    np.testing.assert_allclose(A[0], ref.unfolded_transitions_a)
    # Head 1 is probabilistic
    np.testing.assert_allclose(
        np.sum(A[1, :states[1]], axis=-1), 1.0, atol=1e-6
    )

def test_folded_transitioner_matrix() -> None:
    lengths = [4, 3]

    # Create value sets for different heads
    values = [
        PHMMValueSet.from_config(L, h, ref.config)
        for h, L in enumerate(lengths)
    ]

    # We need to manually create a Hidten HMMConfig because the transitioner is
    # not added to an HMM here.
    states= [
        PHMMTransitionIndexSet(L=L, folded=True).num_states
        for L in lengths
    ]
    hidten_hmm_config = HidtenHMMConfig(states=states)
    transitioner = PHMMTransitioner(
        values=values,
        hidten_hmm_config=hidten_hmm_config
    )
    transitioner.build()

    S = transitioner.start_dist()
    A = transitioner.matrix()

    assert S.shape == (2, max(states))
    assert A.shape == (2, max(states), max(states))

    np.testing.assert_allclose(S[0], ref.start_a, atol=1e-6)
    np.testing.assert_allclose(S[1, :states[1]], ref.start_b, atol=1e-6)
    np.testing.assert_allclose(A[0], ref.transitions_a, atol=1e-6)
    np.testing.assert_allclose(
        A[1, :states[1], :states[1]], ref.transitions_b, atol=1e-6
    )

def test_construct_big_transitioner() -> None:
    import time

    lengths = [500]*10
    config = HMMConfig()
    values = [
        PHMMValueSet.from_config(L, h, config) for h, L in enumerate(lengths)
    ]
    states= [
        PHMMTransitionIndexSet(L=L, folded=True).num_states
        for L in lengths
    ]
    hidten_hmm_config = HidtenHMMConfig(states=states)

    t0 = time.perf_counter()
    transitioner = PHMMTransitioner(
        values=values,
        hidten_hmm_config=hidten_hmm_config
    )
    t1 = time.perf_counter()
    print(f"Constructor time: {t1-t0:.4f}s")

    t0 = time.perf_counter()
    transitioner.build()
    t1 = time.perf_counter()
    print(f"Build time: {t1-t0:.4f}s")

    t0 = time.perf_counter()
    M = transitioner.matrix()
    t1 = time.perf_counter()
    print(f"Matrix time: {t1-t0:.4f}s")

    t0 = time.perf_counter()
    S = transitioner.start_dist()
    t1 = time.perf_counter()
    print(f"Start dist time: {t1-t0:.4f}s")
