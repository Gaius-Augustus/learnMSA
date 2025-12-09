import numpy as np
import tensorflow as tf

import tests.hmm.ref as ref
from learnMSA.hmm.layer import PHMMLayer


def test_loglik_call() -> None:
    """Test forward algorithm for a single-head PHMMLayer.

    This test verifies that the PHMMLayer correctly computes forward
    probabilities for a single head using reference values.
    """
    # Test sequence: [0, 1, 0, 2] with alphabet size 3
    # This is binary (0/1) emissions with an extra padding symbol
    seq = tf.constant(
        [[[1, 0], [0, 1], [1, 0], [0, 0]]], dtype=tf.float32
    ) # (B, T, S)

    # One padding position at the end
    padding = tf.constant([[[1], [1], [1], [0]]], dtype=tf.float32)
    # (B, T, 1)

    # Test models A & B
    lengths = [4, 3]
    config = ref.config.model_copy(deep=True)
    config.use_prior_for_emission_init = False
    layer = PHMMLayer(lengths=lengths, config=config)

    # Build the layer by providing shapes for observations and padding
    layer.build(input_shape=((None, None, 2), (None, None, 1)))

    # Compute log-likelihood
    loglik = layer(seq, padding)

    # Check whether parameters are built correctly
    A = layer.hmm.transitioner.matrix().numpy()
    S = layer.hmm.transitioner.start_dist().numpy()
    B = layer.hmm.emitter[0].matrix().numpy()
    np.testing.assert_allclose(
        A[0],
        ref.transitions_a,
        atol=1e-6,
        err_msg="Transition matrix does not match reference for model A"
    )
    np.testing.assert_allclose(
        A[1, :layer.hmm.config.states[1], :layer.hmm.config.states[1]],
        ref.transitions_b[:-1, :-1],
        atol=1e-6,
        err_msg="Transition matrix does not match reference for model B"
    )
    np.testing.assert_allclose(
        A[1, -1, -1],
        ref.transitions_b[-1, -1],
        atol=1e-6,
        err_msg="Transition matrix does not match reference for model B"
    )
    np.testing.assert_allclose(
        S[0],
        ref.start_a,
        atol=1e-6,
        err_msg="Start distribution does not match reference for model A"
    )
    np.testing.assert_allclose(
        S[1, :layer.hmm.config.states[1]],
        ref.start_b[:-1],
        atol=1e-6,
        err_msg="Start distribution does not match reference for model B"
    )
    np.testing.assert_allclose(
        S[1, -1],
        ref.start_b[-1],
        atol=1e-6,
        err_msg="Start distribution does not match reference for model B"
    )
    np.testing.assert_allclose(
        B[0],
        ref.emissions_a,
        atol=1e-6,
        err_msg="Emission matrix does not match reference for model A"
    )
    np.testing.assert_allclose(
        B[1, :lengths[1]+1, :],
        ref.emissions_b,
        atol=1e-6,
        err_msg="Emission matrix does not match reference for model B"
    )

    assert layer.hmm.use_padding()

    # Check that likelihood matches reference
    np.testing.assert_allclose(
        tf.exp(loglik[0,:,0]).numpy(),
        ref.likelihoods,
        rtol=1e-4,
        atol=1e-4,
        err_msg="Log-likelihood does not match reference for model A"
    )
