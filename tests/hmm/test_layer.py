import numpy as np
import pytest
import tensorflow as tf

import tests.hmm.ref as ref
from learnMSA.hmm.layer import PHMMLayer


@pytest.fixture
def layer() -> PHMMLayer:
    """Fixture for a PHMMLayer with two heads of lengths 4 and 3."""
    # Test models A & B
    lengths = [4, 3]
    config = ref.config.model_copy(deep=True)
    config.use_prior_for_emission_init = False
    layer = PHMMLayer(lengths=lengths, config=config)

    # Build the layer by providing shapes for observations and padding
    layer.build(input_shape=((None, None, 2), (None, None, 1)))

    return layer

@pytest.fixture
def seq() -> tf.Tensor:
    """Fixture for a test sequence tensor."""
    # Test sequence: [0, 1, 0, 2] with alphabet size 3
    # This is binary (0/1) emissions with an extra padding symbol
    return tf.constant(
        [[[1, 0], [0, 1], [1, 0], [0, 0]]], dtype=tf.float32
    ) # (B, T, S)

@pytest.fixture
def padding() -> tf.Tensor:
    """Fixture for a test padding tensor."""
    return tf.constant([[[1], [1], [1], [0]]], dtype=tf.float32) # (B, T, 1)

def test_matrices(
    layer: PHMMLayer, seq: tf.Tensor, padding: tf.Tensor
) -> None:
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
        B[1, :layer.lengths[1]+1, :],
        ref.emissions_b,
        atol=1e-6,
        err_msg="Emission matrix does not match reference for model B"
    )

    assert layer.hmm.use_padding()

def test_loglik_call(
    layer: PHMMLayer, seq: tf.Tensor, padding: tf.Tensor
) -> None:
    # Compute log-likelihood
    loglik = layer(seq, padding).numpy()

    # Check that likelihood matches reference
    np.testing.assert_allclose(
        np.exp(loglik[0,:,0]),
        ref.likelihoods,
        rtol=1e-4,
        atol=1e-5,
        err_msg="Log-likelihood does not match reference for model A"
    )

def test_forward(
    layer: PHMMLayer, seq: tf.Tensor, padding: tf.Tensor
) -> None:
    # For forward, access the hmm attribute directly
    forward_log = layer.hmm.forward_log(seq, padding)
    forward_scaled, _ = layer.hmm.forward_scaled(seq, padding)
    forward_log = forward_log.numpy()
    forward_scaled = forward_scaled.numpy()

    RTOL, ATOL = 1e-4, 1e-4

    # Check that forward probabilities match reference
    np.testing.assert_allclose(
        np.exp(forward_log[0,:,0,:]),
        ref.forward_a,
        rtol=RTOL,
        atol=ATOL,
        err_msg="Forward probabilities do not match reference for model A"
    )
    np.testing.assert_allclose(
        forward_scaled[0,:,0,:],
        ref.forward_scaled_a,
        rtol=RTOL,
        atol=ATOL,
        err_msg="Scaled forward probabilities do not match reference for model A"
    )
    np.testing.assert_allclose(
        np.exp(forward_log[0,:,1,:layer.hmm.config.states[1]]),
        ref.forward_b[..., :-1],
        rtol=RTOL,
        atol=ATOL,
        err_msg="Forward probabilities do not match reference for model B"
    )
    np.testing.assert_allclose(
        np.exp(forward_log[0,:,1,-1]),
        ref.forward_b[..., -1],
        rtol=RTOL,
        atol=ATOL,
        err_msg="Padding state forward probabilities do not match reference "\
            "for model B"
    )
    np.testing.assert_allclose(
        forward_scaled[0,:,1,:layer.hmm.config.states[1]],
        ref.forward_scaled_b[..., :-1],
        rtol=RTOL,
        atol=ATOL,
        err_msg="Scaled forward probabilities do not match reference for model B"
    )
    np.testing.assert_allclose(
        forward_scaled[0,:,1,-1],
        ref.forward_scaled_b[..., -1],
        rtol=RTOL,
        atol=ATOL,
        err_msg="Padding state scaled forward probabilities do not match "\
            "reference for model B"
    )

def test_viterbi(
    layer: PHMMLayer, seq: tf.Tensor, padding: tf.Tensor
) -> None:
    # For Viterbi, access the hmm attribute directly
    viterbi_paths, gamma = layer.hmm.viterbi(seq, padding, return_variables=True)
    viterbi_paths = viterbi_paths.numpy()
    gamma = gamma.numpy()

    # Check that Viterbi paths match reference
    np.testing.assert_equal(
        viterbi_paths[0,:,0],
        ref.viterbi_a,
        err_msg="Viterbi states do not match reference for model A"
    )
    np.testing.assert_equal(
        viterbi_paths[0,:,1],
        ref.viterbi_b,
        err_msg="Viterbi states do not match reference for model B"
    )

    # Check that viterbi variables match reference
    np.testing.assert_allclose(
        np.exp(gamma[0,:,0,:]),
        ref.viterbi_variables_a,
        rtol=1e-4,
        atol=1e-4,
        err_msg="Viterbi variables do not match reference for model A"
    )
    np.testing.assert_allclose(
        np.exp(gamma[0,:,1,:layer.hmm.config.states[1]]),
        ref.viterbi_variables_b[..., :-1],
        rtol=1e-4,
        atol=1e-4,
        err_msg="Viterbi variables do not match reference for model B"
    )
    np.testing.assert_allclose(
        np.exp(gamma[0,:,1,-1]),
        ref.viterbi_variables_b[..., -1],
        rtol=1e-4,
        atol=1e-4,
        err_msg="Padding state Viterbi variables do not match reference for "\
            "model B"
    )

def test_backward(
    layer: PHMMLayer, seq: tf.Tensor, padding: tf.Tensor
) -> None:
    # For backward, access the hmm attribute directly
    backward_log = layer.hmm.backward_log(seq, padding)
    backward_log = backward_log.numpy()

    # Check that backward probabilities match reference
    # TODO: first backward states (i.e. last in true order) don't handle the
    # post-emission reversion correctly yet
    # Should not be a problem for learnMSA
    np.testing.assert_allclose(
        np.exp(backward_log[0,:,0,:]),
        ref.backward_a,
        rtol=1e-4,
        atol=1e-4,
        err_msg="Backward probabilities do not match reference for model A"
    )
    np.testing.assert_allclose(
        np.exp(backward_log[0,:,1,:layer.hmm.config.states[1]]),
        ref.backward_b[..., :-1],
        rtol=1e-4,
        atol=1e-4,
        err_msg="Backward probabilities do not match reference for model B"
    )
    np.testing.assert_allclose(
        np.exp(backward_log[0,:,1,-1]),
        ref.backward_b[..., -1],
        rtol=1e-4,
        atol=1e-4,
        err_msg="Padding state backward probabilities do not match reference "\
            "for model B"
    )

def test_posterior(
    layer: PHMMLayer, seq: tf.Tensor, padding: tf.Tensor
) -> None:
    # For posteriors, access the hmm attribute directly
    posterior = layer.hmm.posterior(seq, padding)
    assert isinstance(posterior, tf.Tensor)
    posterior = posterior.numpy()

    # Check that posterior probabilities match reference
    np.testing.assert_allclose(
        posterior[0,:,0,:],
        ref.posterior_a,
        rtol=1e-3,
        atol=1e-4,
        err_msg="Posterior probabilities do not match reference for model A"
    )
    np.testing.assert_allclose(
        posterior[0,:,1,:layer.hmm.config.states[1]],
        ref.posterior_b[..., :-1],
        rtol=1e-3,
        atol=1e-4,
        err_msg="Posterior probabilities do not match reference for model B"
    )
    np.testing.assert_allclose(
        posterior[0,:,1,-1],
        ref.posterior_b[..., -1],
        rtol=1e-3,
        atol=1e-4,
        err_msg="Padding state posterior probabilities do not match reference "\
            "for model B"
    )