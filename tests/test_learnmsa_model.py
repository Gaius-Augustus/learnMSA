import numpy as np
import pytest
import tensorflow as tf

import learnMSA.msa_hmm.training_util as training_util
import tests.hmm.ref as ref
from learnMSA.config import Configuration, TrainingConfig
from learnMSA.config.hmm import PHMMPriorConfig
from learnMSA.model.model import LearnMSAModel
from learnMSA.msa_hmm.learnmsa_context import LearnMSAContext
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

    # TODO: transpose is currently needed --- FIX LATER ---
    encoded_seqs = tf.transpose(encoded_seqs, [1, 2, 0, 3])
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

    # TODO: transpose is currently needed --- FIX LATER ---
    encoded_seqs = tf.transpose(encoded_seqs, [1, 2, 0, 3])
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
    np.testing.assert_allclose(loss.numpy(), loss_log_lik)

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

    viterbi_seq = model((seq, tf.constant([[0]])))

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

def test_predict_binary(context_binary: LearnMSAContext) -> None:
    # Test that the predict method correctly computes log-likelihoods
    # for sequences in a binary alphabet
    model = LearnMSAModel(context_binary)
    model.loglik_mode()

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
    )

    # Manually set an adaptive batch size function for testing
    batch_cb = training_util.get_adaptive_batch_size(
        context_binary.model_lengths.tolist(), 20, False
    )
    context_binary.config.training.batch_size = batch_cb

    # Predict log-likelihoods for the sequence
    bucket_boundaries = [4]
    bucket_batch_sizes = [2, 3]
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
