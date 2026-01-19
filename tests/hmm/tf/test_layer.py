import numpy as np
import pytest
import tensorflow as tf

from learnMSA.util.sequence_dataset import SequenceDataset
import tests.hmm.ref as ref
from learnMSA.config.hmm import PHMMConfig, PHMMPriorConfig
from learnMSA.hmm.tf import prior
from learnMSA.hmm.tf.layer import PHMMLayer


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
    # Test sequence: [0, 1, 0, P] with alphabet size 2 and padding P
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
        B[1, :2*layer.lengths[1]+2, :],
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

def test_prior_values() -> None:
    """
    Test that the priors are correctly set in the layer.
    """
    config = ref.config.model_copy(deep=True)
    config.use_prior_for_emission_init = True

    # Test model A
    lengths = [4]
    prior_config = PHMMPriorConfig()
    prior_config.alpha_flank = 7000.
    prior_config.alpha_single = 1e9
    prior_config.alpha_global = 1e4
    prior_config.alpha_flank_compl = 1
    prior_config.alpha_single_compl = 1
    prior_config.alpha_global_compl = 1
    layer = PHMMLayer(
        lengths=lengths, config=config, prior_config=prior_config
    )

    # Build the layer by providing shapes for observations and padding
    layer.build(input_shape=((None, None, 23), (None, None, 1)))

    # Test whether computing the prior scores of the layer does not error
    prior_scores = layer.prior_scores()

    # The the emitter's Dirichlet priors
    # Battle-tested gradients computed with legacy learnMSA of dirichlet(p)
    # where p is an amino acid background distribution
    # The new implementation should give the same gradients, but will differ
    # by a constant offset, because it's normalized differently.
    p = tf.constant([
        8.3433352e-02, 5.1926684e-02, 4.9351085e-02, 4.6587169e-02, 2.2493618e-02,
        5.0682485e-02, 6.2964454e-02, 4.7214236e-02, 3.3491924e-02, 5.2677721e-02,
        7.3317297e-02, 6.3507542e-02, 3.5261709e-02, 3.6099266e-02, 3.4606576e-02,
        7.2123714e-02, 6.5257192e-02, 1.7763134e-02, 3.3940718e-02, 6.6508673e-02,
        7.9144974e-04, 5.8379495e-08, 9.9920245e-33
    ])
    expected_grads = np.array([
        16.196707, 8.924403, 7.9193473, 6.7171874, -16.274733,
        8.451642, 12.300347, 7.002275, -1.6756237, 9.198967,
        14.542977, 12.436159, -0.17705068, 0.48093507, -0.71392107,
        14.317257, 12.858342, -28.114046, -1.2808125, 13.146688,
        0., 0., 0.
    ])

    # Compute gradients with the new implementation
    assert layer.hmm.emitter[0].prior is not None
    with tf.GradientTape() as tape:
        tape.watch(p)
        prior_score = layer.hmm.emitter[0].prior(p) # type: ignore
    grads = tape.gradient(prior_score, p)
    assert isinstance(grads, tf.Tensor)
    np.testing.assert_allclose(grads, expected_grads, atol=1e-6, rtol=1e-6)



    # TEMPORARY, only to get refererence values for transition priors

    # Test the transitioners priors
    # Get the legacy density values
    from learnMSA.msa_hmm.Transitioner import ProfileHMMTransitioner

    transitioner = ProfileHMMTransitioner()
    transitioner.set_lengths(lengths)
    transitioner.build()

    # transition_probs = transitioner.make_probs()
    # for k,v in transition_probs[0].items():
    #     transition_probs[0][k] = v.numpy().tolist()
    #     if len(transition_probs[0][k]) > 1:
    #         # wrap for first head
    #         transition_probs[0][k] = [transition_probs[0][k]]
    #     print(k, transition_probs[0][k])

    # from learnMSA.hmm.value_set import PHMMValueSet
    # from learnMSA.config.hmm import HMMConfig

    # # Convert flank_init_prob to scalar to avoid deprecation warning
    # flank_init_prob = transitioner.make_flank_init_prob()
    # if hasattr(flank_init_prob, 'numpy'):
    #     flank_init_prob = float(flank_init_prob.numpy())
    # elif isinstance(flank_init_prob, np.ndarray):
    #     flank_init_prob = float(flank_init_prob)

    # transfer_config = HMMConfig(
    #     p_begin_match = transition_probs[0]["begin_to_match"],
    #     p_match_end = transition_probs[0]["match_to_end"],
    #     p_match_match = transition_probs[0]["match_to_match"],
    #     p_match_insert = transition_probs[0]["match_to_insert"],
    #     p_insert_insert = transition_probs[0]["insert_to_insert"],
    #     p_delete_delete = transition_probs[0]["delete_to_delete"],
    #     p_begin_delete = transition_probs[0]["match_to_delete"][0],
    #     p_left_left = transition_probs[0]["left_flank_loop"],
    #     p_right_right = transition_probs[0]["right_flank_loop"],
    #     p_unannot_unannot = transition_probs[0]["unannotated_segment_loop"],
    #     p_end_unannot = transition_probs[0]["end_to_unannotated_segment"],
    #     p_end_right = transition_probs[0]["end_to_right_flank"],
    #     p_start_left_flank = flank_init_prob
    # )

    # value_set = PHMMValueSet.from_config(4, 0, transfer_config)
    # A = value_set.transitions
    # A = tf.constant(A)[None]  # (1, Q, Q)

    # print("A", A[0])

    legacy_priors = transitioner.get_prior_log_densities()
    for k,v in legacy_priors.items():
        print(f"Legacy prior {k} = {v}")

    # TEMPORARY ENDS

    A = [[
        [0.0000000e+00, 6.3772297e-01, 0.0000000e+00, 0.0000000e+00, 8.6306415e-02,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 9.5895998e-02, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.8007462e-01, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00],
        [0.0000000e+00, 0.0000000e+00, 6.3772297e-01, 0.0000000e+00, 0.0000000e+00,
        8.6306415e-02, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 9.5895998e-02,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.8007462e-01, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00],
        [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 6.3772297e-01, 0.0000000e+00,
        0.0000000e+00, 8.6306415e-02, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        9.5895998e-02, 0.0000000e+00, 0.0000000e+00, 1.8007462e-01, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00],
        [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00],
        [0.0000000e+00, 5.7180440e-01, 0.0000000e+00, 0.0000000e+00, 4.2819563e-01,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00],
        [0.0000000e+00, 0.0000000e+00, 6.1526734e-01, 0.0000000e+00, 0.0000000e+00,
        3.8473266e-01, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00],
        [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 6.2823641e-01, 0.0000000e+00,
        0.0000000e+00, 3.7176356e-01, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00],
        [0.0000000e+00, 6.6576588e-01, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 3.3423415e-01, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00],
        [0.0000000e+00, 0.0000000e+00, 6.7859656e-01, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 3.2140344e-01,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00],
        [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 6.3755649e-01, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        3.6244351e-01, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00],
        [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00],
        [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 7.0727462e-01, 2.9272538e-01, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00],
        [4.7875389e-01, 1.5958464e-01, 1.5958464e-01, 1.5958464e-01, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 4.2492181e-02, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00],
        [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 7.1023889e-05,
        5.3933340e-01, 4.6059558e-01],
        [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 2.6544642e-01, 0.0000000e+00, 7.3455358e-01,
        0.0000000e+00, 0.0000000e+00],
        [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        7.0727462e-01, 2.9272538e-01],
        [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 1.0000000e+00]
    ]]
    A = tf.constant(A, dtype=tf.float32)  # (1, Q, Q)

    legacy_match_prior = -10.251389
    legacy_insert_prior = 3.5873108
    legacy_delete_prior = 4.169487
    legacy_flank_prior = -16179.822
    legacy_hit_prior = -71051.26
    legacy_global_prior = -9213.436

    # Run the new prior
    trans_prior = layer.hmm.transitioner.explicit_transitioner.prior
    assert isinstance(trans_prior, prior.TFPHMMTransitionPrior)
    match_prior_scores = trans_prior.compute_transition_prior(
        A, prior.TFPHMMTransitionPrior.TransitionType.MATCH
    )
    insert_prior_scores = trans_prior.compute_transition_prior(
        A, prior.TFPHMMTransitionPrior.TransitionType.INSERT
    )
    delete_prior_scores = trans_prior.compute_transition_prior(
        A, prior.TFPHMMTransitionPrior.TransitionType.DELETE
    )

    np.testing.assert_allclose(
        match_prior_scores,
        legacy_match_prior,
        atol=1e-4,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        insert_prior_scores,
        legacy_insert_prior,
        atol=1e-4,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        delete_prior_scores,
        legacy_delete_prior,
        atol=1e-4,
        rtol=1e-5,
    )

    # Test flank, hit, and global priors
    # Get flank_init_prob from config
    if isinstance(ref.config.p_start_left_flank, float):
        flank_init_prob = tf.constant(
            [ref.config.p_start_left_flank], dtype=tf.float32
        )
    else:
        flank_init_prob = tf.constant(
            ref.config.p_start_left_flank, dtype=tf.float32
        )

    flank_prior_scores = trans_prior.compute_flank_prior(A, flank_init_prob)
    hit_prior_scores = trans_prior.compute_hit_prior(A)
    global_prior_scores = trans_prior.compute_global_prior(A)

    np.testing.assert_allclose(
        flank_prior_scores,
        legacy_flank_prior,
        atol=1e-4,
        rtol=1e-5,
        err_msg="Flank prior does not match legacy implementation"
    )
    np.testing.assert_allclose(
        hit_prior_scores,
        legacy_hit_prior,
        atol=1e-4,
        rtol=1e-5,
        err_msg="Hit prior does not match legacy implementation"
    )
    np.testing.assert_allclose(
        global_prior_scores,
        legacy_global_prior,
        atol=1e-4,
        rtol=1e-5,
        err_msg="Global prior does not match legacy implementation"
    )

    # Test the full prior call (sum of all components)
    full_prior_scores = trans_prior(A, flank_init_prob)
    expected_full_prior = (
        legacy_match_prior + legacy_insert_prior + legacy_delete_prior +
        legacy_flank_prior + legacy_hit_prior + legacy_global_prior
    )

    np.testing.assert_allclose(
        full_prior_scores,
        expected_full_prior,
        atol=1e-3,
        rtol=1e-5,
        err_msg="Full prior (sum of all components) does not match expected "
        "value"
    )

def test_head_subset(
    layer: PHMMLayer, seq: tf.Tensor, padding: tf.Tensor
) -> None:
    layer.head_subset = [1]  # Only use head 1

    loglik = layer(seq, padding).numpy()
    layer.posterior_mode()
    posterior = layer(seq, padding).numpy()
    layer.viterbi_mode()
    viterbi_paths = layer(seq, padding).numpy()

    np.testing.assert_allclose(loglik, np.log(ref.likelihoods[1]))
    np.testing.assert_allclose(
        posterior[0,:,0,:], ref.posterior_b, rtol=1e-3, atol=1e-4
    )
    np.testing.assert_equal(viterbi_paths[0,:,0], ref.viterbi_b)


def test_basic_forward_simple_seq() -> None:
    """Test basic forward algorithm functionality with simple sequences.
    """

    # Create a model with deterministic emissions matching "ACGT"
    alphabet = SequenceDataset._default_alphabet
    acgt = [alphabet.index(s) for s in "ACGT"]
    match_emissions = np.zeros((1, 4, len(alphabet)-1))
    match_emissions[0, 0, acgt[0]] = 1
    match_emissions[0, 1, acgt[1]] = 1
    match_emissions[0, 2, acgt[2]] = 1
    match_emissions[0, 3, acgt[3]] = 1

    # Convert to list for Pydantic validation
    config = PHMMConfig(
        match_emissions=match_emissions,
        use_prior_for_emission_init=False,
    )

    layer = PHMMLayer(lengths=[4], config=config)

    # Create one-hot encoded sequences for "ACGT"
    seqs = np.eye(len(alphabet)-1)[[acgt, acgt]]  # (2, 4, 23)

    # No padding
    padding = np.ones((2, 4, 1), dtype=np.float32)

    # Build the layer
    layer.build(input_shape=((None, None, 23), (None, None, 1)))

    # Run forward algorithm
    seqs = tf.constant(seqs, dtype=tf.float32)
    padding = tf.constant(padding, dtype=tf.float32)
    forward_log = layer.hmm.forward_log(seqs, padding)
    forward_log_argmax = np.argmax(forward_log, axis=-1)

    # Check that forward probabilities progress through match states
    # The highest probability state at each timestep should be the
    # corresponding match state
    # Match states are at indices 0-3 in the state space
    for t in range(4):
        for b in range(2):
            # Get the state with highest forward probability at time t
            max_state = forward_log_argmax[b, t, 0]
            # Should be in match state t (accounting for state ordering)
            # In PHMMLayer, match states M1-M4 are typically at indices 0-3
            assert max_state == t or max_state == t + 1, \
                f"At time {t}, expected match state {t}, got {max_state}"


def test_variable_length_sequences() -> None:
    """Test handling of sequences with different lengths using padding.

    This is analogous to the second part of legacy test_cell.
    Tests that padding is correctly handled and doesn't affect likelihood.
    """
    # Create a model with deterministic emissions matching "ACGT"
    alphabet = SequenceDataset._default_alphabet
    acgt = [alphabet.index(s) for s in "ACGT"]
    x = alphabet.index("X")

    match_emissions = np.zeros((1, 4, len(alphabet)-1))
    match_emissions[0, 0, acgt[0]] = 1  # A
    match_emissions[0, 1, acgt[1]] = 1  # C
    match_emissions[0, 2, acgt[2]] = 1  # G
    match_emissions[0, 3, acgt[3]] = 1  # T

    config = PHMMConfig(
        match_emissions=match_emissions,
        p_end_unannot=0.5,
        use_prior_for_emission_init=False,
    )

    layer = PHMMLayer(lengths=[4], config=config)

    # Create two sequences: one short (ACGT=4), one long (ACGTXACGT=9)
    max_len = 10
    seq = np.zeros((2, max_len, len(alphabet)-1))

    # First sequence: ACGT (4 symbols) + padding
    seq[0, :4] = np.eye(len(alphabet)-1)[acgt]

    # Second sequence: ACGTXACGT (9 symbols) + padding
    seq[1, :4] = np.eye(len(alphabet)-1)[acgt]
    seq[1, 4, x] = 1.0
    seq[1, 5:9] = np.eye(len(alphabet)-1)[acgt]

    # Create padding mask
    padding = np.zeros((2, max_len, 1))
    padding[0, :4] = 1.0   # First 4 positions are valid
    padding[1, :9] = 1.0   # First 9 positions are valid

    # Build the layer
    layer.build(((None, None, len(alphabet)-1), (None, None, 1)))

    # Convert to tensors
    seq = tf.constant(seq, dtype=tf.float32)
    padding = tf.constant(padding, dtype=tf.float32)

    # Compute likelihoods
    loglik = layer(seq, padding).numpy()

    # Run forward algorithm to check state progression
    forward_log = layer.hmm.forward_log(seq, padding)
    forward_log_argmax = np.argmax(forward_log, axis=-1)

    # First sequence should have most probability mass on the path ACGT
    np.testing.assert_array_equal(
        forward_log_argmax[0, :4, 0],
        np.array([0, 1, 2, 3]),
        err_msg="First sequence did not progress through ACGT correctly"
    )

    # Second sequence should progress twice through ACGT with X treated as
    # unannotated region
    np.testing.assert_array_equal(
        forward_log_argmax[1, :9, 0],
        np.array([0, 1, 2, 3, 8, 0, 1, 2, 3]),
        err_msg="Second sequence did not progress through ACGTXACGT correctly"
    )

    # Ensure that the likelihood of ACGTP equals that of ACGTP^5,
    # i.e. the number of padding tokens after the first padding token does
    # not affect the likelihood
    loglik_short = layer(seq[:1,:5], padding[:1,:5]).numpy()
    np.testing.assert_allclose(
        loglik[0],
        loglik_short[0],
        rtol=1e-5,
        atol=1e-4,
        err_msg="Likelihoods differ for sequences with different padding"
    )


def test_viterbi_two_heads() -> None:
    """Test Viterbi algorithm with two heads on real sequence data.

    This is analogous to the legacy test_viterbi but uses PHMMLayer.
    Tests Viterbi decoding with two models of different lengths.
    """
    alphabet = SequenceDataset._default_alphabet

    # Create two models: one for "FELIK" (5 states), one for "AHC" (3 states)
    felik_indices = [alphabet.index(aa) for aa in "FELIK"]
    ahc_indices = [alphabet.index(aa) for aa in "AHC"]

    # Model 1: FELIK (length 5)
    match_emissions_1 = np.zeros((5, len(alphabet)-1))
    for i, aa_idx in enumerate(felik_indices):
        match_emissions_1[i, aa_idx] = 1.0

    # Model 2: AHC (length 3) - pad to length 5 to match model 1
    match_emissions_2 = np.zeros((5, len(alphabet)-1))
    for i, aa_idx in enumerate(ahc_indices):
        match_emissions_2[i, aa_idx] = 1.0
    # Positions 3 and 4 remain zero (padding for unused match states)

    # Combine into multi-head configuration
    match_emissions = np.array([match_emissions_1, match_emissions_2])

    config = PHMMConfig(
        match_emissions=match_emissions,
        insert_emissions=[1/23]*23,
        use_prior_for_emission_init=False,
        # First head (length 5) and second head (length 3) values
        p_begin_match=[
            [0.43636364, 0.10909091, 0.10909091, 0.10909091, 0.10909091],
            [0.44444445, 0.22222222, 0.22222222]
        ],
        p_match_end=[
            [0.12857144, 0.12857144, 0.12857144, 0.12857144, 1.0],
            [0.3, 0.3, 1.0]
        ],
        p_match_match=[
            [0.2857143, 0.2857143, 0.2857143, 0.2857143],
            [0.19999999, 0.19999999]
        ],
        p_match_insert=[
            [0.2857143, 0.2857143, 0.2857143, 0.2857143],
            [0.19999999, 0.19999999]
        ],
        p_insert_insert=[
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5]
        ],
        p_delete_delete=[
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5]
        ],
        p_begin_delete=[0.12727274, 0.11111111],
        p_left_left=0.5,
        p_right_right=0.5,
        p_unannot_unannot=0.5,
        p_end_unannot=0.42231882,
        p_end_right=0.15536243,
        p_start_left_flank=0.5,
    )

    layer = PHMMLayer(lengths=[5, 3], config=config)

    # Load test data from felix.fa
    import os
    test_data_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "felix.fa"
    )

    with SequenceDataset(test_data_path) as data:
        # Get sequences and convert to one-hot
        max_len = np.max(data.seq_lens)
        num_seqs = data.num_seq

        # Create one-hot encoded sequences
        seq = np.zeros((num_seqs, max_len+1))+alphabet.index("-")
        for i in range(num_seqs):
           seq[i, :data.seq_lens[i]] = data.get_encoded_seq(i)
        seq = np.eye(len(alphabet))[seq.astype(int)]

    padding = 1-tf.constant(seq[..., -1:], dtype=tf.float32)  # (N, L, 1)
    seq = tf.constant(seq[..., :-1], dtype=tf.float32) # (N, L, 23)

    # Build layer
    layer.build(((None, None, len(alphabet)-1), (None, None, 1)))

    # Run Viterbi
    layer.viterbi_mode()
    viterbi_paths = layer(seq, padding).numpy()

    # Check basic properties of Viterbi paths
    assert viterbi_paths.shape == (num_seqs, max_len+1, 2), \
        f"Expected shape ({num_seqs}, {max_len+1}, 2), got {viterbi_paths.shape}"

    def remap_states(old_seq, length):
        """Remap state numbers from legacy to new PHMMLayer ordering."""
        # For length L:
        # Old: [LEFT_FLANK=0, M1=1...ML=L, I1=L+1...IL-1=2L-1,
        # UNANNOT=2L, RIGHT=2L+1, END=2L+2]
        # New: [M1=0...ML=L-1, I1=L...IL-1=2L-2, LEFT_FLANK=2L-1,
        # UNANNOT=2L, RIGHT=2L+1, END=-1]
        new_seq = np.zeros_like(old_seq)
        for i, old_state in enumerate(old_seq):
            if old_state == 0:
                # LEFT_FLANK
                new_seq[i] = 2 * length - 1
            elif 1 <= old_state <= length:
                # MATCH states
                new_seq[i] = old_state - 1
            elif length + 1 <= old_state <= 2 * length - 1:
                # INSERT states
                new_seq[i] = old_state - 1
            elif old_state in [2 * length, 2 * length + 1]:
                # UNANNOTATED, RIGHT_FLANK (unchanged)
                new_seq[i] = old_state
            elif old_state == 2 * length + 2:
                # END state
                new_seq[i] = -1
        return new_seq

    # Legacy reference sequences (using old state numbering)
    ref_seqs_legacy = np.array([
        # model 1 (FELIK, length 5)
        [[1, 2, 3, 4, 5, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
         [0, 0, 0, 1, 2, 3, 4, 5, 12, 12, 12, 12, 12, 12, 12],
         [1, 2, 3, 4, 5, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12],
         [1, 2, 3, 4, 5, 10, 10, 10, 1, 2, 3, 4, 5, 11, 12],
         [0, 2, 3, 4, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
         [1, 2, 7, 7, 7, 3, 4, 5, 12, 12, 12, 12, 12, 12, 12],
         [1, 6, 6, 2, 3, 8, 4, 9, 9, 9, 5, 12, 12, 12, 12],
         [1, 2, 3, 8, 8, 8, 4, 5, 11, 11, 11, 12, 12, 12, 12]],
        # model 2 (AHC, length 3)
        [[0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
         [1, 2, 3, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8],
         [0, 0, 0, 0, 0, 0, 1, 3, 8, 8, 8, 8, 8, 8, 8],
         [0, 0, 0, 0, 0, 1, 2, 3, 6, 6, 6, 6, 6, 1, 8],
         [1, 4, 4, 4, 2, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
         [0, 0, 1, 2, 3, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8],
         [0, 1, 2, 6, 6, 1, 6, 1, 2, 3, 7, 8, 8, 8, 8],
         [0, 0, 0, 1, 2, 3, 6, 6, 1, 2, 3, 8, 8, 8, 8]]
    ])

    # Remap to new state numbering
    ref_seqs = np.array([
        # model 1, length 5
        np.array([remap_states(seq, 5) for seq in ref_seqs_legacy[0]]),
        # model 2, length 3
        np.array([remap_states(seq, 3) for seq in ref_seqs_legacy[1]])
    ])

    # Check that Viterbi paths match reference for both models
    np.testing.assert_equal(
        viterbi_paths[:, :15, 0],
        ref_seqs[0],
        err_msg="Viterbi paths do not match reference for model 1 (FELIK)"
    )
    np.testing.assert_equal(
        viterbi_paths[:, :15, 1],
        ref_seqs[1],
        err_msg="Viterbi paths do not match reference for model 2 (AHC)"
    )

    # Test with parallel Viterbi
    viterbi_paths_parallel = layer.hmm.viterbi(seq, padding, parallel=3)
    np.testing.assert_array_equal(
        viterbi_paths,
        viterbi_paths_parallel.numpy(), # type: ignore
        err_msg="Parallel Viterbi gives different results"
    )

