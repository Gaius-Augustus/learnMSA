import numpy as np
import pytest
import tensorflow as tf

import tests.hmm.ref as ref
from learnMSA.config import Configuration, TrainingConfig
from learnMSA.config.hmm import PHMMPriorConfig
from learnMSA.msa_hmm.learnmsa_context import LearnMSAContext
from learnMSA.msa_hmm.model import LearnMSAModel


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
def context_amino_acid_no_prior(config_amino_acid_no_prior: Configuration) -> LearnMSAContext:
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
    # Test that the compute_loss method runs without errors when no prior is used.
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
    tws = context_binary.sequence_weights.sum()
    loss_log_prior = -model.phmm_layer.prior_scores().numpy().mean()\
        / context_binary.sequence_weights.sum()

    np.testing.assert_allclose(loss.numpy(), loss_log_lik + loss_log_prior)
