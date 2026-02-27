import numpy as np
import pytest
import tensorflow as tf

import learnMSA.model.training_util as training_util
import tests.hmm.ref as ref
from learnMSA.config import Configuration, TrainingConfig
from learnMSA.config.hmm import PHMMPriorConfig
from learnMSA.model.tf.model import LearnMSAModel
from learnMSA.model.context import LearnMSAContext
from learnMSA.util.sequence_dataset import SequenceDataset


@pytest.fixture
def config_binary() -> Configuration:
    """A basic configuration to set up context and model.
    For a pair of pHMM heads over a binary alphabet.
    """
    hmm_config = ref.config.model_copy(deep=True)
    hmm_config.use_prior_for_emission_init = False
    return Configuration(
        training=TrainingConfig(
            length_init=[4, 3],
            use_anc_probs=False,
        ),
        hmm=hmm_config,
        hmm_prior=PHMMPriorConfig(
            use_amino_acid_prior=False
        )
    )

@pytest.fixture
def config_amino_acid() -> Configuration:
    """A basic configuration to set up context and model.
    """
    return Configuration(training=TrainingConfig(length_init=[20, 10]))

@pytest.fixture
def config_amino_acid_no_prior() -> Configuration:
    """A basic configuration to set up context and model.
    """
    return Configuration(
        training=TrainingConfig(
            length_init=[20, 10],
            use_prior=False,
        ),
    )

@pytest.fixture
def context_binary(config_binary: Configuration) -> LearnMSAContext:
    """The context for setting up the model."""
    return LearnMSAContext(
        config=config_binary,
        num_seq=10,
        sequence_weights=np.arange(10, dtype=float),
    )

@pytest.fixture
def context_amino_acid(config_amino_acid: Configuration) -> LearnMSAContext:
    """The context for setting up the model."""
    return LearnMSAContext(
        config=config_amino_acid,
        num_seq=50,
        sequence_weights=np.arange(50, dtype=float),
    )

@pytest.fixture
def context_amino_acid_no_prior(
    config_amino_acid_no_prior: Configuration
) -> LearnMSAContext:
    """The context for setting up the model."""
    return LearnMSAContext(
        config=config_amino_acid_no_prior,
        num_seq=50,
        sequence_weights=np.arange(50, dtype=float),
    )

# Transitioner Tests
def test_create_and_call(context_amino_acid: LearnMSAContext) -> None:
    # Test that the model can be created without errors.
    model = LearnMSAModel(context_amino_acid)
    assert model is not None

    # Test a forward pass of the model without embeddings.
    batch_size = 4
    seq_length = 15
    seqs = np.random.randint(
        low=0, high=20, size=(batch_size, 1, seq_length), dtype=np.int32
    )
    indices = np.array([22, 7, 13, 3])[..., np.newaxis]
    output = model((seqs, indices))
    assert output.shape == (batch_size, 2)

def test_compute_loss_amino_acid(context_amino_acid: LearnMSAContext) -> None:
    # Test that the compute_loss method runs without errors.
    model = LearnMSAModel(context_amino_acid)

    batch_size = 4
    seq_length = 15
    seqs = np.random.randint(
        low=0, high=20, size=(batch_size, 1, seq_length), dtype=np.int32
    )
    indices = np.array([22, 7, 13, 3])[..., np.newaxis]

    # Compute a forward pass and the loss
    y_pred = model((seqs, indices))
    loss = model.compute_loss((seqs, indices), 0., y_pred)

    assert isinstance(loss, tf.Tensor)

    # Assume that hidten computes likelihoods correctly
    encoded_seqs = model.encode_batch((seqs, indices))

    padding = 1 - encoded_seqs[:, :, :, -1:]
    encoded_seqs = encoded_seqs[:, :, :, :-1]

    # Compute the log-likelihoods manually and apply weights
    manual_likelihoods = model.phmm_layer.hmm.likelihood_log(
        encoded_seqs, padding
    )
    manual_likelihoods = manual_likelihoods.numpy()[..., 0]
    weights = indices.astype(np.float32)
    loss_log_lik_per_head = -(manual_likelihoods * weights).sum(axis=0)\
        / weights.sum()
    loss_log_lik = loss_log_lik_per_head.mean()

    assert context_amino_acid.sequence_weights is not None
    loss_log_prior = -model.phmm_layer.prior_scores().numpy().mean()\
        / context_amino_acid.sequence_weights.sum()

    np.testing.assert_allclose(loss.numpy(), loss_log_lik + loss_log_prior)

def test_compute_loss_amino_acid_no_prior(
    context_amino_acid_no_prior: LearnMSAContext
) -> None:
    # Test that the compute_loss method runs without errors
    # when no prior is used.
    model = LearnMSAModel(context_amino_acid_no_prior)

    batch_size = 4
    seq_length = 15
    seqs = np.random.randint(
        low=0, high=20, size=(batch_size, 1, seq_length), dtype=np.int32
    )
    indices = np.array([22, 7, 13, 3])[..., np.newaxis]

    # Compute a forward pass and the loss
    y_pred = model((seqs, indices))
    loss = model.compute_loss((seqs, indices), 0., y_pred)

    assert isinstance(loss, tf.Tensor)

    # Assume that hidten computes likelihoods correctly
    encoded_seqs = model.encode_batch((seqs, indices))

    padding = 1 - encoded_seqs[:, :, :, -1:]
    encoded_seqs = encoded_seqs[:, :, :, :-1]

    # Compute the log-likelihoods manually and apply weights
    manual_likelihoods = model.phmm_layer.hmm.likelihood_log(
        encoded_seqs, padding
    )
    manual_likelihoods = manual_likelihoods.numpy()[..., 0]
    weights = indices.astype(np.float32)
    loss_log_lik_per_head = -(manual_likelihoods * weights).sum(axis=0)\
        / weights.sum()
    loss_log_lik = loss_log_lik_per_head.mean()

    # When use_prior=False, the loss should only be the log-likelihood
    # without any prior component
    np.testing.assert_allclose(loss.numpy(), loss_log_lik, rtol=1e-6)

def test_compute_loss_binary(context_binary: LearnMSAContext) -> None:
    # Test that the compute_loss method runs without errors.
    model = LearnMSAModel(context_binary)

    # Test sequence: [0, 1, 0, P] with alphabet size 2 and padding P
    # This test sequence is replicated multiple times with different weights
    seqs = tf.constant(
        [[0, 1, 0, 2]], dtype=tf.int32
    ) # (1, T)
    seqs = tf.repeat(seqs, repeats=4, axis=0) # replicate to batch size 4
    seqs = tf.expand_dims(seqs, axis=2)  # add head dimension
    indices = np.array([2, 7, 1, 3])[..., np.newaxis]

    # TODO learnMSA currently expects shape (B, H, T, S) --- FIX LATER ---
    # so we need to swap axes here
    seqs = tf.keras.ops.swapaxes(seqs, 1, 2)

    y_pred = model((seqs, indices))

    loss = model.compute_loss((seqs, indices), 0., y_pred)
    assert isinstance(loss, tf.Tensor)

    # Reference log-likelihoods for a single sequence in both heads
    # The loss per head is a weighted average and all batch elements have
    # the same likelihood
    loss_log_lik = -np.log(ref.likelihoods).mean()
    assert context_binary.sequence_weights is not None
    loss_log_prior = -model.phmm_layer.prior_scores().numpy().mean()\
        / context_binary.sequence_weights.sum()

    np.testing.assert_allclose(loss.numpy(), loss_log_lik + loss_log_prior)

def test_viterbi_on_batch(context_binary: LearnMSAContext) -> None:
    # Test that the viterbi_on_batch method runs without errors.
    model = LearnMSAModel(context_binary)
    model.viterbi_mode()

    seq = tf.constant(
        [[0, 1, 0, 2]], dtype=tf.int32
    ) # (1, T)
    seq = tf.expand_dims(seq, axis=2)  # add head dimension

    # TODO learnMSA currently expects shape (B, H, T, S) --- FIX LATER ---
    # so we need to swap axes here
    seq = tf.keras.ops.swapaxes(seq, 1, 2)

    viterbi_seq = model((seq, tf.constant([[0]]))).numpy()

    # Replace padding -1
    viterbi_seq[:,:,0][viterbi_seq[:,:,0] == -1] = 10
    viterbi_seq[:,:,1][viterbi_seq[:,:,1] == -1] = 8

    np.testing.assert_equal(viterbi_seq[0, :, 0], ref.viterbi_a)
    np.testing.assert_equal(viterbi_seq[0, :, 1], ref.viterbi_b)

def test_posterior_on_batch(context_binary: LearnMSAContext) -> None:
    # Test that the posterior_on_batch method runs without errors.
    model = LearnMSAModel(context_binary)
    model.posterior_mode()

    seq = tf.constant(
        [[0, 1, 0, 2]], dtype=tf.int32
    ) # (1, T)
    seq = tf.expand_dims(seq, axis=2)  # add head dimension

    # TODO learnMSA currently expects shape (B, H, T, S) --- FIX LATER ---
    # so we need to swap axes here
    seq = tf.keras.ops.swapaxes(seq, 1, 2)

    posterior = model((seq, tf.constant([[0]])))

    np.testing.assert_allclose(
        posterior[0,:,0,:],
        ref.posterior_a,
        rtol=1e-3,
        atol=1e-4,
        err_msg="Posterior probabilities do not match reference for model A"
    )
    np.testing.assert_allclose(
        posterior[0,:,1,:model.phmm_layer.hmm.config.states[1]],
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

def test_fit(context_amino_acid: LearnMSAContext) -> None:
    # Test that the fit method runs without errors and that parameters are
    # updated depending on the data.
    model = LearnMSAModel(context_amino_acid)
    model.build()
    model.compile()

    # Feed artifical data that contains only a few sequences with a single
    # amino acid type 'A' (index 0)
    data = SequenceDataset(sequences=[
            ("1", "AAAAAAAAAAAAAAAAAAAA"),
            ("2", "AAAAAAAAAAAAAAAAAA"),
            ("3", "AAAAAAAAAAAAA"),
            ("4", "AAAAAAAAAAAAAAAAAA"),
            ("5", "AAAAAAAAAA"),
            ("6", "AAAAAAAAAAAAAAAAAA"),
            ("7", "AAAAAAAAAAAAAAAAAAA"),
            ("8", "AAAAAAAAAAAAAAA"),
    ])

    # Get average emission probability of A before training
    matrix_before_training = model.phmm_layer.hmm.emitter[0].matrix().numpy()
    prob_A_before_training = matrix_before_training[:, :10, 0].mean()

    # Fit for a few epochs
    model.fit(data, batch_size=4, epochs=1, steps_per_epoch=10)

    # Get average emission probability of A after training
    matrix_after_training = model.phmm_layer.hmm.emitter[0].matrix().numpy()
    prob_A_after_training = matrix_after_training[:, :10, 0].mean()

    assert prob_A_after_training > prob_A_before_training, \
        "Emission probability for amino acid A did not increase after training"

def test_predict(context_binary: LearnMSAContext) -> None:
    # Test that the predict method correctly computes log-likelihoods
    # for sequences in a binary alphabet
    model = LearnMSAModel(context_binary)
    model.build()
    model.loglik_mode()
    model.compile()

    # Create a dataset with the test sequence "ABA"
    # and longer "BBBBB" sequences to test if the output ordering is correct
    # after bucketing internally
    data = SequenceDataset(
        sequences=[
            ("1", "ABA"),
            ("2", "ABA"),
            ("3", "BBBBBBBBBBBBBBBBBB"),
            ("4", "ABA"),
            ("5", "BBBBBBBBBB"),
            ("6", "ABA"),
            ("7", "BBBBBBBBBBBBBBBB"),
            ("8", "ABA"),
            ("9", "ABA"),
            ("10", "ABA"),
        ],
        alphabet="AB-",
        replace_with_x="",
    )

    # Manually set an adaptive batch size function for testing
    batch_cb = training_util.get_adaptive_batch_size(
        context_binary.model_lengths.max(),
        len(context_binary.model_lengths),
        20,
    )
    context_binary.config.training.batch_size = batch_cb

    # Predict log-likelihoods for the sequence
    # Note: When JIT is enabled, pad_to_bucket_boundary=True requires
    # all buckets to have explicit upper bounds.
    bucket_boundaries = [4, 20]
    bucket_batch_sizes = [2, 3, 3]
    predictions = model.predict(
        data,
        bucket_boundaries=bucket_boundaries,
        bucket_batch_sizes=bucket_batch_sizes,
    )

    # The predictions should match the reference log-likelihoods
    # ref.likelihoods contains the likelihoods for both model heads
    expected_loglik = np.log(ref.likelihoods).reshape((1,2)).repeat(7, axis=0)

        # predictions shape should be (1, 2) for 1 sequence and 2 model heads
    assert predictions.shape == (10, 2)
    np.testing.assert_allclose(
        predictions[[0,1,3,5,7,8,9]],
        expected_loglik,
        rtol=1e-3,
        atol=1e-4,
        err_msg="Predicted log-likelihoods do not match reference values"
    )
    assert np.all(predictions[[2,4,6]] != expected_loglik[0])

    # Viterbi predictions
    model.viterbi_mode()
    model.compile()
    viterbi_seqs = model.predict(
        data,
        bucket_boundaries=bucket_boundaries,
        bucket_batch_sizes=bucket_batch_sizes,
    )
    # Replace padding -1
    viterbi_seqs[:,:,0][viterbi_seqs[:,:,0] == -1] = 10
    viterbi_seqs[:,:,1][viterbi_seqs[:,:,1] == -1] = 8
    assert viterbi_seqs.shape == (10, 19, 2)
    np.testing.assert_equal(
        viterbi_seqs[[0,1,3,5,7,8,9], :4, 0],
        ref.viterbi_a[np.newaxis, :].repeat(7, axis=0),
        err_msg="Predicted Viterbi sequences do not match reference for model A"
    )
    np.testing.assert_equal(
        viterbi_seqs[[0,1,3,5,7,8,9], :4, 1],
        ref.viterbi_b[np.newaxis, :].repeat(7, axis=0),
        err_msg="Predicted Viterbi sequences do not match reference for model B"
    )

    # Posterior predictions
    model.posterior_mode()
    posterior = model.predict(
        data,
        bucket_boundaries=bucket_boundaries,
        bucket_batch_sizes=bucket_batch_sizes,
    )
    assert posterior.shape == (10, 19, 2, model.phmm_layer.hmm.config.max_states+1)
    np.testing.assert_allclose(
        posterior[[0,1,3,5,7,8,9], :4, 0, :],
        ref.posterior_a[np.newaxis, :, :].repeat(7, axis=0),
        rtol=1e-3,
        atol=1e-4,
        err_msg="Predicted posterior probabilities do not match reference for "\
            "model A"
    )
    np.testing.assert_allclose(
        posterior[[0,1,3,5,7,8,9], :4, 1, :model.phmm_layer.hmm.config.states[1]],
        ref.posterior_b[np.newaxis, :, :-1].repeat(7, axis=0),
        rtol=1e-3,
        atol=1e-4,
        err_msg="Predicted posterior probabilities do not match reference for "\
            "model B"
    )
    np.testing.assert_allclose(
        posterior[[0,1,3,5,7,8,9], :4, 1, -1],
        ref.posterior_b[np.newaxis, :, -1].repeat(7, axis=0),
        rtol=1e-3,
        atol=1e-4,
        err_msg="Predicted padding state posterior probabilities do not match "\
            "reference for model B"
    )

def test_evaluate(context_binary: LearnMSAContext) -> None:
    # Test that the evaluate method correctly computes metrics
    # for sequences in a binary alphabet
    context_binary.sequence_weights = np.arange(1, 401, dtype=float) / 200.0
    model = LearnMSAModel(context_binary)
    model.build()

    # Create a dataset with the test sequence "ABA"
    data = SequenceDataset(
        sequences=[(str(i), "ABA") for i in range(400)],
        alphabet="AB-",
        replace_with_x="",
    )

    # Evaluate on the dataset
    metrics = model.evaluate(data)

    loss = metrics["loss"]
    loglik = metrics["loglik"]
    prior = metrics["prior"]

    # metrics should be an array with [loss, loglik, prior]
    assert isinstance(loss, np.ndarray)
    assert isinstance(loglik, np.ndarray)
    assert isinstance(prior, np.ndarray)
    assert loss.shape == ()
    assert loglik.shape == (2,)
    assert prior.shape == (2,)

    expected_loglik = np.log(ref.likelihoods)

    assert context_binary.sequence_weights is not None
    expected_prior = model.phmm_layer.prior_scores().numpy() \
        / context_binary.sequence_weights.sum()

    expected_loss = -expected_loglik - expected_prior

    # Check that metrics match expected values
    np.testing.assert_allclose(
        loss, expected_loss, rtol=1e-4,
        err_msg="Loss does not match expected value"
    )
    np.testing.assert_allclose(
        loglik, expected_loglik, rtol=1e-4,
        err_msg="Loglik does not match expected value"
    )
    np.testing.assert_allclose(
        prior, expected_prior, rtol=1e-4,
        err_msg="Prior does not match expected value"
    )

def test_estimate_loglik(context_amino_acid: LearnMSAContext) -> None:
    # Test that the estimate_loglik method correctly computes
    # log-likelihoods for sequences in an amino acid alphabet
    model = LearnMSAModel(context_amino_acid)
    model.build()

    # Create a dataset with random sequences
    np.random.seed(42)
    sequences = []
    for i in range(50):
        seq_length = np.random.randint(10, 50)
        seq = ''.join(
            np.random.choice(
                list('ACDEFGHIKLMNPQRSTVWY'),
                size=seq_length
            )
        )
        sequences.append((str(i), seq))
    data = SequenceDataset(sequences=sequences)

    # Estimate log-likelihoods
    loglik_reduced = model.estimate_loglik(data, reduce=True)
    loglik_full = model.estimate_loglik(data, reduce=False)

    # Check shapes of the outputs
    assert loglik_reduced.shape == (2,), \
        "Reduced loglik shape is incorrect"
    assert loglik_full.shape == (50, 2), \
        "Full loglik shape is incorrect"

def test_null_model_log_probs(context_amino_acid: LearnMSAContext) -> None:
    # Test that compute_null_model_log_probs correctly computes
    # log probabilities under a null model
    model = LearnMSAModel(context_amino_acid)
    model.build()

    # Create a dataset with sequences of varying lengths
    sequences = [
        ("1", "ACDEFGHIKLMNPQRSTVWY"),
        ("2", "AAAAAAAAA"),
        ("3", "ACDEFG"),
        ("4", "MGKLPQRSTVWY"),
    ]
    data = SequenceDataset(sequences=sequences)

    # Use a uniform distribution for easy reference value computation
    uniform_dist = np.ones(23) / 23.0

    # Compute null model log probabilities with uniform background
    log_probs = model.compute_null_model_log_probs(
        data, background_dist=uniform_dist, transition_prob=0.5
    )

    # Log probabilities
    p = np.log(uniform_dist[0])

    # Compute expected log probs for each sequence
    expected_log_probs = np.zeros(4)
    for i, (_, seq) in enumerate(sequences):
        # Emissions
        expected_log_probs[i] = p * len(seq)
        # Transitions
        expected_log_probs[i] += np.log(0.5) * (len(seq) - 1)

    # Check that output has correct shape
    assert log_probs.shape == (4,), \
        f"Expected shape (4,), got {log_probs.shape}"

    # Check that computed values match expected values
    np.testing.assert_allclose(
        log_probs, expected_log_probs, rtol=1e-5,
        err_msg="Computed log probabilities do not match expected values"
    )

def test_predict_posterior_reduce(context_amino_acid: LearnMSAContext) -> None:
    """Test that predict with reduce=True matches manual reduction."""
    model = LearnMSAModel(context_amino_acid)
    model.build()
    model.posterior_mode()

    # Create a small dataset
    sequences = [
        ("1", "ACDEFGHIKLMNPQRSTVWY"),
        ("2", "MGKLPQRSTVWY"),
        ("3", "ACDEFG"),
    ]
    data = SequenceDataset(sequences=sequences)
    indices = np.arange(len(sequences))

    # Get reduced predictions
    reduced = model.predict(data, indices=indices, reduce=True)

    # Get full predictions and manually reduce
    full = model.predict(data, indices=indices, reduce=False)  # (B, T, H, Q)
    full = full[:, :, :, :-1]  # Drop terminal state
    manual_reduced = np.sum(full, axis=1)  # Sum over time/sequence positions
    manual_reduced = np.mean(manual_reduced, axis=0)  # Average over batch

    # Check shapes
    assert reduced.shape == manual_reduced.shape, \
        f"Shape mismatch: reduced={reduced.shape}, manual={manual_reduced.shape}"

    # Check values match
    np.testing.assert_allclose(
        reduced, manual_reduced, rtol=1e-5,
        err_msg="Reduced predictions do not match manually reduced values"
    )
