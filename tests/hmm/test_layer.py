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

    # TODO: legacy code tested this with assert_almost_equal and decimal=6
    # Look into that.
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

def test_tf_model(
    layer: PHMMLayer, seq: tf.Tensor, padding: tf.Tensor
) -> None:
    # Test the hmm layer in a (jit-)compiled model and use model.predict
    seq_input = tf.keras.Input(
        shape=(None, 2), name="sequences", dtype=tf.float32
    )
    padding_input = tf.keras.Input(
        shape=(None, 1), name="padding", dtype=tf.float32
    )
    output = layer(seq_input, padding_input)
    hmm_tf_model = tf.keras.Model(
        inputs=[seq_input, padding_input], outputs=[output]
    )
    hmm_tf_model.compile(jit_compile=True)
    loglik = hmm_tf_model.predict([seq, padding])

    # Check that likelihood matches reference
    np.testing.assert_allclose(
        np.exp(loglik[0,:,0]),
        ref.likelihoods,
        rtol=1e-4,
        atol=1e-5,
        err_msg="Log-likelihood does not match reference for model A"
    )

    # Test if switching to inference mode works
    layer.viterbi_mode()
    hmm_tf_model.compile(jit_compile=True)
    viterbi_paths = hmm_tf_model.predict([seq, padding])

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

    # Test if switching to posterior mode works
    layer.posterior_mode()
    hmm_tf_model.compile(jit_compile=True)
    posterior = hmm_tf_model.predict([seq, padding])

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

def test_parallel_factor(
    layer: PHMMLayer, seq: tf.Tensor, padding: tf.Tensor
) -> None:
    loglik = layer.hmm.likelihood_log(seq, padding, parallel=2)
    viterbi_paths = layer.hmm.viterbi(seq, padding, parallel=2)
    posterior = layer.hmm.posterior(seq, padding, parallel=2)
    loglik = loglik.numpy()
    viterbi_paths = viterbi_paths.numpy()
    assert isinstance(posterior, tf.Tensor)
    posterior = posterior.numpy()

    # Check that likelihood matches reference
    np.testing.assert_allclose(
        np.exp(loglik[0,:,0]),
        ref.likelihoods,
        rtol=1e-4,
        atol=1e-5,
        err_msg="Log-likelihood does not match reference for model A"
    )

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

def test_parallel_longer_seq_batch(layer: PHMMLayer) -> None:
    """Test that parallel and non-parallel processing give the same results
    on longer sequences with multiple batch items."""
    # Create a longer sequence (16 timesteps, 2 batch items)
    # One sequence without padding, one with padding (last symbols are 0,0)
    seq = tf.constant(
        [
            # First batch item - no padding
            [[1, 0], [0, 1], [1, 0], [0, 1], [0, 1], [1, 0], [0, 1], [1, 0],
             [1, 0], [0, 1], [1, 0], [0, 1], [0, 1], [1, 0], [0, 1], [1, 0]],
            # Second batch item - with padding at the end
            [[1, 0], [0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [0, 1], [0, 1],
             [0, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
        ],
        dtype=tf.float32
    )  # (B=2, T=16, S=2)

    padding = tf.constant(
        [
            [[1], [1], [1], [1], [1], [1], [1], [1],
             [1], [1], [1], [1], [1], [1], [1], [1]],
            [[1], [1], [1], [1], [1], [1], [1], [1],
             [1], [0], [0], [0], [0], [0], [0], [0]]
        ],
        dtype=tf.float32
    )  # (B=2, T=16, 1)

    # Run without parallel processing
    forward_log = layer.hmm.forward_log(seq, padding)
    loglik = layer.hmm.likelihood_log(seq, padding)
    posterior = layer.hmm.posterior(seq, padding)

    # Run with parallel processing (parallel_factor=4, so 16/4 = 4 chunks)
    forward_log_parallel = layer.hmm.forward_log(seq, padding, parallel=4)
    loglik_parallel = layer.hmm.likelihood_log(seq, padding, parallel=4)
    posterior_parallel = layer.hmm.posterior(seq, padding, parallel=4)

    # Convert to numpy for comparison
    forward_log = forward_log.numpy()
    loglik = loglik.numpy()
    assert isinstance(posterior, tf.Tensor)
    posterior = posterior.numpy()
    forward_log_parallel = forward_log_parallel.numpy()
    loglik_parallel = loglik_parallel.numpy()
    assert isinstance(posterior_parallel, tf.Tensor)
    posterior_parallel = posterior_parallel.numpy()

    # Compare results for both heads
    for head_idx in range(layer.heads):
        q = layer.hmm.config.states[head_idx]

        # Check log-likelihood
        np.testing.assert_allclose(
            np.exp(loglik[:, head_idx]),
            np.exp(loglik_parallel[:, head_idx]),
            rtol=1e-5,
            atol=1e-4,
            err_msg=f"Log-likelihood mismatch for head {head_idx}"
        )

        # Check forward probabilities
        np.testing.assert_allclose(
            np.exp(forward_log[:, :, head_idx, :q]),
            np.exp(forward_log_parallel[:, :, head_idx, :q]),
            rtol=1e-5,
            atol=1e-4,
            err_msg=f"Forward probabilities mismatch for head {head_idx}"
        )

        # Check posterior probabilities
        np.testing.assert_allclose(
            np.exp(posterior[:, :, head_idx, :q]),
            np.exp(posterior_parallel[:, :, head_idx, :q]),
            rtol=2e-3,
            atol=1e-4,
            err_msg=f"Posterior probabilities mismatch for head {head_idx}"
        )

def test_parallel_viterbi_long(layer: PHMMLayer) -> None:
    """Test that Viterbi with different parallel factors gives identical
    results on very long sequences (10000 timesteps)."""
    # Generate a long random sequence
    np.random.seed(57235782)
    seq = np.random.randint(2, size=(3, 10000))  # (B=3, T=10000), values 0 or 1
    seq = tf.one_hot(seq, 2)  # (B=3, T=10000, S=2) - alphabet size matches "AB"
    seq = tf.cast(seq, tf.float32)

    # Create padding tensor (no padding for this test)
    padding = tf.ones((3, 10000, 1), dtype=tf.float32)

    # Run Viterbi with parallel_factor=1 (sequential)
    viterbi_path_1 = layer.hmm.viterbi(seq, padding, parallel=1)

    # Run Viterbi with parallel_factor=100 (highly parallel)
    viterbi_path_100 = layer.hmm.viterbi(seq, padding, parallel=100)

    # Convert to numpy for comparison
    assert isinstance(viterbi_path_1, tf.Tensor)
    assert isinstance(viterbi_path_100, tf.Tensor)
    viterbi_path_1 = viterbi_path_1.numpy()
    viterbi_path_100 = viterbi_path_100.numpy()

    # Verify that both parallel factors produce identical paths
    np.testing.assert_equal(
        viterbi_path_1,
        viterbi_path_100,
        err_msg="Viterbi paths differ between parallel=1 and parallel=100"
    )
