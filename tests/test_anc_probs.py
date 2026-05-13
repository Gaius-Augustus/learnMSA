"""Tests for ancestral probability calculations and rate matrices."""
import os

import numpy as np
import pytest
import tensorflow as tf

from learnMSA import Configuration
from learnMSA.model import LearnMSAModel, LearnMSAContext
from learnMSA.model.tf import training
import learnMSA.tree.tf.initializer as Initializers
from learnMSA.tree.tf.anc_probs_layer import AncProbsLayer
from learnMSA.util import SequenceDataset


def assert_vec(x : np.ndarray, y: np.ndarray, almost : bool=False) -> None:
    """Assert that two arrays are equal in shape and values."""
    for i, (a, b) in enumerate(zip(x.shape, y.shape)):
        assert a == b or a == 1 or b == 1, f"{a} {b} (dim {i})"
    assert x.dtype == y.dtype
    if almost:
        np.testing.assert_allclose(x, y, rtol=5e-3, atol=5e-3)
    else:
        assert np.all(x == y), str(x) + " not equal to " + str(y)


def parse_a(string : str) -> np.ndarray:
    """Parse a string of floats into a numpy array."""
    return np.array([float(x) for x in string.split()], dtype=np.float32)


def assert_equilibrium(p: np.ndarray) -> None:
    """Assert that equilibrium frequencies sum to 1."""
    np.testing.assert_almost_equal(np.sum(p), 1., decimal=5)


def assert_symmetric(matrix: np.ndarray) -> None:
    """Assert that a matrix is symmetric."""
    assert matrix.shape[-1] == matrix.shape[-2]
    n = matrix.shape[-1]
    for i in range(n):
        assert matrix[i, i] == 0.
        for j in range(i + 1, n):
            assert matrix[i, j] == matrix[j, i]


def assert_rate_matrix(Q: np.ndarray, p: np.ndarray) -> None:
    """Assert that a rate matrix satisfies detailed balance."""
    for i in range(Q.shape[0] - 1):
        for j in range(Q.shape[0] - 1):
            np.testing.assert_almost_equal(Q[i, j] * p[i], Q[j, i] * p[j])


def assert_anc_probs(
        anc_prob_seqs: np.ndarray,
        expected_sum: np.ndarray,
        expected_anc_probs: np.ndarray|None = None
) -> None:
    """Assert properties of ancestral probability sequences."""
    assert_vec(
        np.sum(anc_prob_seqs, -1, keepdims=True), expected_sum, almost=True
    )
    if expected_anc_probs is not None:
        assert_vec(anc_prob_seqs, expected_anc_probs, almost=True)


def assert_anc_probs_layer(
    anc_probs_layer : AncProbsLayer, config: Configuration
) -> None:
    """Assert properties of an ancestral probabilities layer."""
    anc_probs_layer.build()
    p = anc_probs_layer.make_p().numpy()
    R = anc_probs_layer.make_R().numpy()
    Q = anc_probs_layer.make_Q().numpy()
    assert p.shape[0] == config.training.num_model
    assert R.shape[0] == config.training.num_model
    assert Q.shape[0] == config.training.num_model
    # TODO: test with multiple tracks
    assert p.shape[1] == 1
    assert R.shape[1] == 1
    assert Q.shape[1] == 1
    for model_equi in p:
        for equi in model_equi:
            assert_equilibrium(equi)
    for model_exchange in R:
        for exchange in model_exchange:
            assert_symmetric(exchange)
    for model_rate, model_equi in zip(Q, p):
        for rate, equi in zip(model_rate, model_equi):
            assert_rate_matrix(rate, equi)


def get_test_configs(sequences : np.ndarray) -> list[dict]:
    """Generate test configurations for ancestral probabilities.

    Args:
        sequences: Shape (b, L, num_model) with integer indices
    """
    # Assuming sequences only contain the 20 standard AAs
    oh_sequences = tf.one_hot(sequences, 20)
    anc_probs_init = Initializers.make_default_anc_probs_init(1)
    inv_sp_R = anc_probs_init[1]((1, 1, 20, 20))
    log_p = anc_probs_init[2]((1, 1, 20))
    p = tf.nn.softmax(log_p)
    cases = []
    for rate_init in [-100., -3., 100.]:
        for n in [1, 3]:
            case = {}

            config = Configuration()
            config.training.length_init = [10]*n
            config.training.no_sequence_weights = True
            case["config"] = config

            encoder_initializer = Initializers.make_default_anc_probs_init(n)
            encoder_initializer = (
                [Initializers.ConstantInitializer(rate_init)] +
                encoder_initializer[1:]
            )
            case["encoder_initializer"] = encoder_initializer

            if rate_init == -100.:
                expected_anc_probs = tf.one_hot(sequences, 20).numpy()
                # Shape: (B, L, H, S)
                case["expected_anc_probs"] = expected_anc_probs
            elif rate_init == 100.:
                b, L, num_model = sequences.shape
                anc = np.concatenate([p] * b * L * num_model, axis=1)
                anc = np.reshape(anc, (b, L, num_model, 20))
                # Shape: (B, L, H, S)
                case["expected_anc_probs"] = anc

            case["expected_freq"] = np.ones((), dtype=np.float32)

            if "expected_anc_probs" in case:
                case["expected_anc_probs"] = np.stack(
                    [case["expected_anc_probs"]] * n, axis=2
                )
            cases.append(case)
    return cases


def make_anc_probs_layer(
    config : Configuration,
    enc_init : list[tf.keras.initializers.Initializer],
    num_rates: int,
    time_reversed: bool=False
) -> AncProbsLayer:
    return AncProbsLayer(
        heads = config.training.num_model,
        rates = num_rates,
        input_tracks = 1,
        equilibrium_init=enc_init[2],
        rate_init=enc_init[0],
        exchangeability_init=enc_init[1],
        trainable_distances=config.training.trainable_distances,
        alphabet_size=20,
        time_reversed=time_reversed,
    )


def get_simple_seq(data: SequenceDataset) -> np.ndarray:
    """Get simple sequence data for testing."""
    from learnMSA.model.tf import training
    indices = np.arange(data.num_seq)
    batch_generator = training.BatchGenerator()
    config = Configuration()
    config.training.num_model = 1
    config.training.no_sequence_weights = True
    batch_generator.configure(data, context=LearnMSAContext(config, data))
    ds, steps = training.make_dataset(
        indices,
        batch_generator,
        batch_size=data.num_seq,
        shuffle=False,
    )
    for (seq, _), _ in ds.take(1):
        sequences = seq.numpy()[:, :-1, :]
        break
    return sequences


def test_anc_probs_layer() -> None:
    """Test ancestral probability calculations."""
    filename = os.path.dirname(__file__) + "/../tests/data/simple.fa"
    with SequenceDataset(filename) as data:
        sequences = get_simple_seq(data)  # Shape: (b, L, num_model)
    n = sequences.shape[0]
    # Convert to one-hot: (b, L, num_model, 20)
    sequences_onehot = tf.one_hot(sequences, depth=20, dtype=tf.float32).numpy()
    # rate_indices should be (b, num_model)
    rate_indices = np.arange(n)[:, np.newaxis]

    for case in get_test_configs(sequences):
        anc_probs_layer = make_anc_probs_layer(
            case["config"], case["encoder_initializer"], n
        )
        assert_anc_probs_layer(anc_probs_layer, case["config"])
        anc_prob_seqs = anc_probs_layer(
            sequences_onehot, rate_indices=rate_indices # type: ignore
        ).numpy()
        shape = (
            n,
            sequences.shape[1],  # L is at index 1
            case["config"].training.num_model,
            1,
            20
        )
        anc_prob_seqs = np.reshape(anc_prob_seqs, shape)
        if "expected_anc_probs" in case:
            assert_anc_probs(
                anc_prob_seqs,
                case["expected_freq"],
                case["expected_anc_probs"],
            )
        else:
            assert_anc_probs(anc_prob_seqs, case["expected_freq"])


def test_encoder_model() -> None:
    """Test encoder model with ancestral probabilities layer."""
    filename = os.path.dirname(__file__) + "/../tests/data/simple.fa"
    with SequenceDataset(filename) as data:
        sequences = get_simple_seq(data)  # Shape: (b, L, num_model)
        n = sequences.shape[0]
        ind = np.arange(n)
        model_length = 10
        # this test is currently a bit messy, although it works...
        # this sets up data once instead of within the loop,
        # but this requires resetting a dummy config and context here
        config = Configuration()
        config.training.num_model = 1
        config.training.no_sequence_weights = True
        batch_gen = training.BatchGenerator()
        batch_gen.configure(data, context=LearnMSAContext(config, data))
        for case in get_test_configs(sequences):
            # The default emitter initializers expect 25 as last dimension
            # which is not compatible with num_matrix=3
            config: Configuration = case["config"]
            config.input_output.verbose = False
            context = LearnMSAContext(config, data)
            context.encoder_initializer = case["encoder_initializer"]
            context.effective_num_seq = n
            model = LearnMSAModel(context)
            assert_anc_probs_layer(model.anc_probs_layer, config)
            ds, steps = training.make_dataset(
                ind,
                batch_gen,
                batch_size=n,
                shuffle=False,
                bucket_boundaries=[],
                bucket_batch_sizes=[],
            )
            for x, _ in ds.take(1):
                # encode_batch now returns (b, L, num_model, features)
                anc_prob_seqs = model.encode_batch(x).numpy()

                # Trim to the original sequence length (remove padding positions)
                L = sequences.shape[1]  # L is at index 1
                anc_prob_seqs = anc_prob_seqs[:, :L, :, :]

                shape = (
                    n,
                    L,  # Use original sequence length
                    config.training.num_model,
                    1,
                    len(SequenceDataset._default_alphabet)
                )
                anc_prob_seqs = np.reshape(anc_prob_seqs, shape)
                break
            if "expected_anc_probs" in case:
                assert_anc_probs(
                    anc_prob_seqs[:,:,:,:,:20],
                    case["expected_freq"],
                    case["expected_anc_probs"]
                )
            else:
                assert_anc_probs(anc_prob_seqs, case["expected_freq"])

def test_time_reversed() -> None:
    # Set up a layer with default initialization and time_reversed=True
    config = Configuration()
    config.training.num_model = 1
    encoder_initializer = Initializers.make_default_anc_probs_init(1)
    anc_probs_layer = make_anc_probs_layer(
        config,
        Initializers.make_default_anc_probs_init(1),
        num_rates=1,
        time_reversed=True,
    )

    # Get sequence data
    filename = os.path.dirname(__file__) + "/../tests/data/simple.fa"
    with SequenceDataset(filename) as data:
        sequences = get_simple_seq(data)  # Shape: (b, L, num_model)
    n = sequences.shape[0]
    L = sequences.shape[1]
    # Convert to one-hot: (b, L, num_model, 20)
    sequences_onehot = tf.one_hot(sequences, depth=20, dtype=tf.float32).numpy()
    rate_indices = np.arange(n)[:, np.newaxis]

    anc_probs = anc_probs_layer(
        sequences_onehot, rate_indices=rate_indices # type: ignore
    ).numpy()

    # Compute the expected output by gathering columns of P
    # shape P: (n, 1, 1, 20, 20)
    P = anc_probs_layer.make_P(anc_probs_layer.make_tau(rate_indices)).numpy()
    ref_anc_probs = np.empty((n, L, 1, 20), dtype=np.float32)
    for i in range(n):
        for j in range(L):
            for k in range(20):
                ref_anc_probs[i, j, 0, k] = P[i, 0, 0, sequences[i, j, 0], k]

    assert_vec(anc_probs, ref_anc_probs, almost=True)
