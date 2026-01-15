import numpy as np
import pytest
import tensorflow as tf
from hidten.hmm import HMMConfig as HidtenHMMConfig

import tests.hmm.ref as ref
from learnMSA.config.hmm import PHMMConfig
from learnMSA.hmm.util.transition_index_set import PHMMTransitionIndexSet
from learnMSA.hmm.tf.transitioner import (PHMMExplicitTransitioner,
                                       PHMMTransitioner)
from learnMSA.hmm.util.value_set import PHMMValueSet


@pytest.fixture
def hmm_config() -> HidtenHMMConfig:
    lengths = [4, 3]
    return HidtenHMMConfig(states=[2*L+2 for L in lengths])

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

    # Construct a transitioner with two heads from the initial values
    transitioner = PHMMExplicitTransitioner(values)
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
    np.testing.assert_allclose(A[0], ref.unfolded_transitions_a, rtol=1e-6, atol=1e-7)
    # Head 1 is probabilistic
    np.testing.assert_allclose(
        np.sum(A[1, :states[1]-1], axis=-1), 1.0, atol=1e-6
    )
    np.testing.assert_allclose(np.sum(A[1, -1], axis=-1), 1.0, atol=1e-6)

def test_folded_transitioner_matrix(hmm_config: HidtenHMMConfig) -> None:
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
    transitioner = PHMMTransitioner(values=values)
    transitioner.hmm_config = hmm_config
    transitioner.build()

    S = transitioner.start_dist()
    A = transitioner.matrix()

    assert S.shape == (2, max(states))
    assert A.shape == (2, max(states), max(states))

    np.testing.assert_allclose(S[0], ref.start_a, atol=1e-6)
    np.testing.assert_allclose(S[1, :states[1]-1], ref.start_b[:-1], atol=1e-6)
    np.testing.assert_allclose(S[1, -1], ref.start_b[-1], atol=1e-6)
    np.testing.assert_allclose(A[0], ref.transitions_a, atol=1e-6)
    actual_non_terminal = np.concatenate(
        [A[1, :states[1]-1, :states[1]-1], A[1, :states[1]-1, -1:]], axis=1
    )
    actual_terminal = np.concatenate(
        [A[1, -1:, :states[1]-1], A[1, -1:, -1:]], axis=1
    )
    actual_transitions_b = np.concatenate(
        [actual_non_terminal, actual_terminal], axis=0
    )
    np.testing.assert_allclose(
        actual_transitions_b, ref.transitions_b, atol=1e-6
    )

def test_prior() -> None:
    pass

def test_construct_big_transitioner(hmm_config: HidtenHMMConfig) -> None:
    import time

    lengths = [500]*10
    config = PHMMConfig()
    values = [
        PHMMValueSet.from_config(L, h, config) for h, L in enumerate(lengths)
    ]

    t0 = time.perf_counter()
    transitioner = PHMMTransitioner(values=values)
    transitioner.hmm_config = HidtenHMMConfig(states=[2*L+2 for L in lengths])
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

def test_head_subset(hmm_config: HidtenHMMConfig) -> None:
    lengths = [4, 3]

    # Create value sets for different heads
    values = [
        PHMMValueSet.from_config(L, h, ref.config)
        for h, L in enumerate(lengths)
    ]

    transitioner = PHMMTransitioner(values=values)
    transitioner.hmm_config = hmm_config
    transitioner.build()

    assert transitioner.matrix().shape[0] == 2  # two heads

    # Set head subset to only head 1
    transitioner.head_subset = [1]

    S = transitioner.start_dist()
    A = transitioner.matrix()

    assert S.shape == (1, hmm_config.states[1] + 1)
    assert A.shape == (1, hmm_config.states[1] + 1, hmm_config.states[1] + 1)

    np.testing.assert_allclose(
        S[0, :hmm_config.states[1]], ref.start_b[:-1], atol=1e-6
    )
    np.testing.assert_allclose(
        S[0, -1], ref.start_b[-1], atol=1e-6
    )

    np.testing.assert_allclose(
        A[0], ref.transitions_b, atol=1e-6
    )

def test_gradient(hmm_config: HidtenHMMConfig) -> None:
    """Test that gradients flow correctly through the PHMMTransitioner."""
    lengths = [4, 3]

    # Create value sets for different heads
    values = [
        PHMMValueSet.from_config(L, h, ref.config)
        for h, L in enumerate(lengths)
    ]

    transitioner = PHMMTransitioner(values=values)
    transitioner.hmm_config = hmm_config
    transitioner.build()

    # Perform forward pass and compute gradients
    with tf.GradientTape() as tape:
        transitioner._launch()
        x = transitioner.start_dist()
        x = tf.expand_dims(x, axis=0) # Add batch dimension
        y = transitioner(x)
        # Create a scalar loss for gradient computation
        loss = tf.reduce_sum(y)

    grads = tape.gradient(loss, transitioner.trainable_variables)

    for i, grad in enumerate(grads):
        assert grad is not None, f"Gradient {i} is None!"

    assert not tf.reduce_any(tf.math.is_nan(grad.values)).numpy(), \
        f"Gradient {i} contains NaN!"
