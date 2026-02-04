"""Tests for ancestral probability calculations and rate matrices."""
import os

import numpy as np
import pytest
import tensorflow as tf

from learnMSA import Configuration
from learnMSA.hmm.tf.layer import PHMMLayer
from learnMSA.model import LearnMSAModel, LearnMSAContext
from learnMSA.model.tf import training
import learnMSA.tree.util as Utility
import learnMSA.tree.tf.initializer as Initializers
from learnMSA.tree.tf.anc_probs_layer import AncProbsLayer
from learnMSA.util import SequenceDataset


class AncProbsData:
    """Fixture class for ancestral probabilities test data."""

    def __init__(self):
        self.paml_all = [Utility.LG_paml] + Utility.LG4X_paml
        self.A = SequenceDataset._default_alphabet[:20]


@pytest.fixture
def anc_probs_data() -> AncProbsData:
    """Fixture providing test data for ancestral probabilities."""
    return AncProbsData()


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
    assert_vec(np.sum(anc_prob_seqs, -1, keepdims=True), expected_sum, almost=True)
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
    assert p.shape[1] == config.training.num_rate_matrices
    assert R.shape[1] == config.training.num_rate_matrices
    assert Q.shape[1] == config.training.num_rate_matrices
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
    for equilibrium_sample in [True, False]:
        for rate_init in [-100., -3., 100.]:
            for num_matrices in [1, 3]:
                case = {}

                config = Configuration()
                config.training.num_model = 1
                config.training.no_sequence_weights = True
                config.training.equilibrium_sample = equilibrium_sample
                config.training.num_rate_matrices = num_matrices
                case["config"] = config

                encoder_initializer = Initializers.make_default_anc_probs_init(num_matrices)
                if num_matrices > 1:
                    R_stack = np.concatenate([inv_sp_R] * num_matrices, axis=1)
                    p_stack = np.concatenate([log_p] * num_matrices, axis=1)
                    encoder_initializer = (
                        encoder_initializer[:1] +
                        [Initializers.ConstantInitializer(R_stack),
                         Initializers.ConstantInitializer(p_stack)]
                    )
                encoder_initializer = (
                    [Initializers.ConstantInitializer(rate_init)] +
                    encoder_initializer[1:]
                )
                case["encoder_initializer"] = encoder_initializer

                if rate_init == -100.:
                    expected_anc_probs = tf.one_hot(
                        sequences, len(SequenceDataset._default_alphabet)
                    ).numpy()
                    # Shape: (B, L, H, S)
                    case["expected_anc_probs"] = expected_anc_probs
                elif rate_init == 100.:
                    anc = np.concatenate(
                        [p, np.zeros((1, 1, len(SequenceDataset._default_alphabet) - 20), dtype=np.float32)],
                        axis=-1,
                    )
                    b, L, num_model = sequences.shape
                    anc = np.concatenate([anc] * b * L * num_model, axis=1)
                    anc = np.reshape(anc, (b, L, num_model, len(SequenceDataset._default_alphabet)))
                    # Shape: (B, L, H, S)
                    case["expected_anc_probs"] = anc

                if equilibrium_sample:
                    # Shape: (b, L, num_model)
                    expected_freq = tf.linalg.matvec(p, oh_sequences).numpy()
                    case["expected_freq"] = expected_freq
                    if rate_init != -3.:
                        case["expected_anc_probs"] *= expected_freq
                    case["expected_freq"] = np.stack([case["expected_freq"]] * num_matrices, axis=-2)
                else:
                    case["expected_freq"] = np.ones((), dtype=np.float32)
                if "expected_anc_probs" in case:
                    case["expected_anc_probs"] = np.stack([case["expected_anc_probs"]] * num_matrices, axis=-2)
                cases.append(case)
    return cases


def make_anc_probs_layer(config : Configuration, enc_init, num_seq) -> AncProbsLayer:
    return AncProbsLayer(
        config.training.num_model,
        num_seq,
        config.training.num_rate_matrices,
        equilibrium_init=enc_init[2],
        rate_init=enc_init[0],
        exchangeability_init=enc_init[1],
        trainable_rate_matrices=config.training.trainable_rate_matrices,
        trainable_distances=config.training.trainable_distances,
        per_matrix_rate=config.training.per_matrix_rate,
        matrix_rate_l2=config.training.matrix_rate_l2,
        shared_matrix=config.training.shared_rate_matrix,
        equilibrium_sample=config.training.equilibrium_sample,
        transposed=config.training.transposed,
    )


def get_simple_seq(data: SequenceDataset) -> np.ndarray:
    """Get simple sequence data for testing."""
    from learnMSA.model.tf import training
    indices = np.arange(data.num_seq)
    batch_generator = training.BatchGenerator()
    config = Configuration()
    config.training.num_model = 1
    config.training.no_sequence_weights = True
    batch_generator.configure(data, LearnMSAContext(config, data))
    ds, steps = training.make_dataset(
        indices,
        batch_generator,
        batch_size=data.num_seq,
        shuffle=False,
    )
    for (seq, _), _ in ds.take(1):
        sequences = seq.numpy()[:, :, :-1]
        # transpose to shape (b, L, num_model)
        sequences = np.transpose(sequences, [0, 2, 1]) # TODO: fix batch gen
        break
    return sequences


def test_paml_parsing(anc_probs_data : AncProbsData) -> None:
    """Test parsing of PAML format rate matrices."""
    R1, p1 = Utility.parse_paml(Utility.LG4X_paml[0], anc_probs_data.A)
    true_p1_str = """0.147383 0.017579 0.058208 0.017707 0.026331
                    0.041582 0.017494 0.027859 0.011849 0.076971
                    0.147823 0.019535 0.037132 0.029940 0.008059
                    0.088179 0.089653 0.006477 0.032308 0.097931"""
    true_X_row_1 = "0.295719"
    true_X_row_4 = "1.029289 0.576016 0.251987 0.189008"
    true_X_row_19 = """0.916683 0.102065 0.043986 0.080708 0.885230
                        0.072549 0.206603 0.306067 0.205944 5.381403
                        0.561215 0.112593 0.693307 0.400021 0.584622
                        0.089177 0.755865 0.133790 0.154902"""
    assert_vec(p1, parse_a(true_p1_str))
    assert_vec(R1[1, :1], parse_a(true_X_row_1))
    assert_vec(R1[4, :4], parse_a(true_X_row_4))
    assert_vec(R1[19, :19], parse_a(true_X_row_19))
    for R, p in map(Utility.parse_paml, anc_probs_data.paml_all, [anc_probs_data.A] * len(anc_probs_data.paml_all)):
        assert_equilibrium(p)
        assert_symmetric(R)


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
            sequences_onehot, rate_indices=rate_indices
        ).numpy()
        shape = (
            n,
            sequences.shape[1],  # L is at index 1
            case["config"].training.num_model,
            case["config"].training.num_rate_matrices,
            20  # Only 20 amino acids, not full alphabet
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
        batch_gen.configure(data, LearnMSAContext(config, data))
        for case in get_test_configs(sequences):
            # The default emitter initializers expect 25 as last dimension
            # which is not compatible with num_matrix=3
            config: Configuration = case["config"]
            config.training.length_init = [model_length]
            config.input_output.verbose = False
            context = LearnMSAContext(config, data)
            context.encoder_initializer = case["encoder_initializer"]
            context.effective_num_seq = n
            model = LearnMSAModel(context)
            assert_anc_probs_layer(model.anc_probs_layer, config)
            ds, steps = training.make_dataset(ind, batch_gen, batch_size=n, shuffle=False)
            for x, _ in ds.take(1):
                # encode_batch now returns (b, L, num_model, features)
                # features = num_rate_matrices * alphabet_size (24 amino acids, no padding)
                anc_prob_seqs = model.encode_batch(x).numpy()

                # Trim to the original sequence length (remove padding positions)
                L = sequences.shape[1]  # L is at index 1
                anc_prob_seqs = anc_prob_seqs[:, :L, :, :]

                shape = (
                    n,
                    L,  # Use original sequence length
                    config.training.num_model,
                    config.training.num_rate_matrices,
                    len(SequenceDataset._default_alphabet)
                )
                anc_prob_seqs = np.reshape(anc_prob_seqs, shape)
                break
            if "expected_anc_probs" in case:
                assert_anc_probs(
                    anc_prob_seqs,
                    case["expected_freq"],
                    case["expected_anc_probs"]
                )
            else:
                assert_anc_probs(anc_prob_seqs, case["expected_freq"])


def test_transposed() -> None:
    """Test transposed ancestral probabilities."""
    # Load data and prepare sequences
    filename = os.path.dirname(__file__) + "/../tests/data/simple.fa"
    with SequenceDataset(filename) as data:
        sequences = get_simple_seq(data)  # Shape: (b, L, num_model)
    n = sequences.shape[0]
    # Convert to one-hot: (b, L, num_model, 20)
    sequences_onehot = tf.one_hot(sequences, depth=20, dtype=tf.float32).numpy()

    # Create a configuration
    config = Configuration()
    config.training.num_model = 1
    config.training.no_sequence_weights = True
    config.training.transposed = False

    # Create ancestral probabilities layer with/without transpose
    anc_probs_layer = make_anc_probs_layer(
        config, Initializers.make_default_anc_probs_init(1), n
    )
    config.training.transposed = True
    anc_probs_layer_transposed = make_anc_probs_layer(
        config, Initializers.make_default_anc_probs_init(1), n
    )

    context = LearnMSAContext(config, data)

    # Obtain emission matrix
    phmm_layer = PHMMLayer(context.model_lengths, config.hmm)
    phmm_layer.build(input_shape=(
        (None, None, n, context.config.hmm.alphabet_size),
        (None, None, n, 1),
    ))
    B = phmm_layer.hmm.emitter[0].matrix()[0] # (H, Q, S)
    # Add terminal state to make this test work
    # TODO: clean up
    B = tf.concat([B, tf.zeros((B.shape[0], 1), dtype=B.dtype)], axis=1)

    rate_indices = np.arange(n)[:, np.newaxis]

    anc_prob_seqs = anc_probs_layer_transposed(
        sequences_onehot, rate_indices=rate_indices
    ).numpy() # (b, L, num_model, num_matrices*20)
    shape = (
        n,
        sequences.shape[1],  # L is at index 1
        config.training.num_model,
        config.training.num_rate_matrices,
        20  # Only 20 amino acids
    )
    anc_prob_seqs = np.reshape(anc_prob_seqs, shape) # (b, L, num_model, num_matrices, 20)
    anc_prob_seqs = tf.cast(anc_prob_seqs, B.dtype)

    # For B matrix test: create one-hot version of B
    B_onehot = tf.one_hot(tf.range(B.shape[0]), depth=20, dtype=B.dtype)  # (M, 20)
    B_batch_first = B_onehot[tf.newaxis, :, tf.newaxis, :]  # (1, M, 1, 20)
    anc_prob_B = anc_probs_layer(
        B_batch_first, rate_indices=[[0]]
    )
    # Output is (1, M, 1, num_matrices*20), reshape to (M, num_matrices, 20)
    anc_prob_B = tf.squeeze(anc_prob_B, axis=[0, 2])  # (M, num_matrices*20)
    anc_prob_B = tf.reshape(anc_prob_B, (tf.shape(B)[0], config.training.num_rate_matrices, 20))  # (M, num_matrices, 20)
    prob1 = tf.linalg.matvec(B, anc_prob_seqs)
    oh_seqs = tf.one_hot(sequences, 20, dtype=anc_prob_B.dtype)
    oh_seqs = tf.expand_dims(oh_seqs, -2)
    prob2 = tf.linalg.matvec(anc_prob_B, oh_seqs)
    np.testing.assert_allclose(
        prob1.numpy(), prob2.numpy(),
        rtol=1e-5, atol=1e-4,
    )
