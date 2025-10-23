"""Tests for ancestral probability calculations and rate matrices."""
import os

import numpy as np
import pytest
import tensorflow as tf

from learnMSA.msa_hmm import (AncProbsLayer, Configuration, Emitter,
                              Initializers, Training, Utility)
from learnMSA.msa_hmm.AlignmentModel import AlignmentModel
from learnMSA.msa_hmm.SequenceDataset import SequenceDataset


class AncProbsData:
    """Fixture class for ancestral probabilities test data."""

    def __init__(self):
        self.paml_all = [Utility.LG_paml] + Utility.LG4X_paml
        self.A = SequenceDataset.alphabet[:20]


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
        expected_anc_probs: np.ndarray = None
) -> None:
    """Assert properties of ancestral probability sequences."""
    assert_vec(np.sum(anc_prob_seqs, -1, keepdims=True), expected_sum, almost=True)
    if expected_anc_probs is not None:
        assert_vec(anc_prob_seqs, expected_anc_probs, almost=True)


def assert_anc_probs_layer(
    anc_probs_layer : AncProbsLayer, config: dict
) -> None:
    """Assert properties of an ancestral probabilities layer."""
    anc_probs_layer.build()
    p = anc_probs_layer.make_p()
    R = anc_probs_layer.make_R()
    Q = anc_probs_layer.make_Q()
    assert p.shape[0] == config["num_models"]
    assert R.shape[0] == config["num_models"]
    assert Q.shape[0] == config["num_models"]
    assert p.shape[1] == config["num_rate_matrices"]
    assert R.shape[1] == config["num_rate_matrices"]
    assert Q.shape[1] == config["num_rate_matrices"]
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
    """Generate test configurations for ancestral probabilities."""
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
                config = Configuration.make_default(1)
                config["num_models"] = 1
                config["equilibrium_sample"] = equilibrium_sample
                config["num_rate_matrices"] = num_matrices
                if num_matrices > 1:
                    R_stack = np.concatenate([inv_sp_R] * num_matrices, axis=1)
                    p_stack = np.concatenate([log_p] * num_matrices, axis=1)
                    config["encoder_initializer"] = (
                        config["encoder_initializer"][:1] +
                        [Initializers.ConstantInitializer(R_stack),
                         Initializers.ConstantInitializer(p_stack)]
                    )
                config["encoder_initializer"] = (
                    [Initializers.ConstantInitializer(rate_init)] +
                    config["encoder_initializer"][1:]
                )
                case["config"] = config
                if rate_init == -100.:
                    case["expected_anc_probs"] = tf.one_hot(sequences, len(SequenceDataset.alphabet)).numpy()
                elif rate_init == 100.:
                    anc = np.concatenate([p, np.zeros((1, 1, len(SequenceDataset.alphabet) - 20), dtype=np.float32)], axis=-1)
                    anc = np.concatenate([anc] * sequences.shape[0] * sequences.shape[1] * sequences.shape[2], axis=1)
                    anc = np.reshape(anc, (sequences.shape[0], sequences.shape[1], sequences.shape[2], len(SequenceDataset.alphabet)))
                    case["expected_anc_probs"] = anc
                if equilibrium_sample:
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


def get_simple_seq(data: SequenceDataset) -> np.ndarray:
    """Get simple sequence data for testing."""
    from learnMSA.msa_hmm import Configuration, Training
    indices = np.arange(data.num_seq)
    batch_generator = Training.DefaultBatchGenerator()
    config = Configuration.make_default(1)
    batch_generator.configure(data, config)
    ds = Training.make_dataset(indices,
                               batch_generator,
                               batch_size=data.num_seq,
                               shuffle=False)
    for (seq, _), _ in ds:
        sequences = seq.numpy()[:, :, :-1]
        sequences = np.transpose(sequences, [1, 0, 2])
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


def test_rate_matrices(anc_probs_data : AncProbsData) -> None:
    """Test construction of rate matrices from PAML data."""
    for R, p in map(Utility.parse_paml, anc_probs_data.paml_all, [anc_probs_data.A] * len(anc_probs_data.paml_all)):
        Q = AncProbsLayer.make_rate_matrix(R, p)
        assert_rate_matrix(Q, p)


def test_anc_probs() -> None:
    """Test ancestral probability calculations."""
    filename = os.path.dirname(__file__) + "/../tests/data/simple.fa"
    with SequenceDataset(filename) as data:
        sequences = get_simple_seq(data)
    n = sequences.shape[1]
    for case in get_test_configs(sequences):
        anc_probs_layer = Training.make_anc_probs_layer(n, case["config"])
        assert_anc_probs_layer(anc_probs_layer, case["config"])
        anc_prob_seqs = anc_probs_layer(sequences, np.arange(n)[np.newaxis, :]).numpy()
        shape = (case["config"]["num_models"], n, sequences.shape[2], case["config"]["num_rate_matrices"], len(SequenceDataset.alphabet))
        anc_prob_seqs = np.reshape(anc_prob_seqs, shape)
        if "expected_anc_probs" in case:
            assert_anc_probs(anc_prob_seqs, case["expected_freq"], case["expected_anc_probs"])
        else:
            assert_anc_probs(anc_prob_seqs, case["expected_freq"])


def test_encoder_model() -> None:
    """Test encoder model with ancestral probabilities layer."""
    filename = os.path.dirname(__file__) + "/../tests/data/simple.fa"
    with SequenceDataset(filename) as data:
        sequences = get_simple_seq(data)
        n = sequences.shape[1]
        ind = np.arange(n)
        model_length = 10
        batch_gen = Training.DefaultBatchGenerator()
        batch_gen.configure(data, Configuration.make_default(1))
        ds = Training.make_dataset(ind, batch_gen, batch_size=n, shuffle=False)
        for case in get_test_configs(sequences):
            # The default emitter initializers expect 25 as last dimension which is not compatible with num_matrix=3
            config = dict(case["config"])
            config["emitter"] = Emitter.ProfileHMMEmitter(
                emission_init=Initializers.ConstantInitializer(0.),
                insertion_init=Initializers.ConstantInitializer(0.)
            )
            model = Training.default_model_generator(
                num_seq=n,
                effective_num_seq=n,
                model_lengths=[model_length],
                config=config,
                data=data
            )
            am = AlignmentModel(
                data,
                batch_gen,
                ind,
                batch_size=n,
                model=model
            )
            assert_anc_probs_layer(am.encoder_model.layers[-1], case["config"])
            for x, _ in ds:
                anc_prob_seqs = am.encoder_model(x).numpy()[:, :, :-1]
                shape = (case["config"]["num_models"], n, sequences.shape[2], case["config"]["num_rate_matrices"], len(SequenceDataset.alphabet))
                anc_prob_seqs = np.reshape(anc_prob_seqs, shape)
            if "expected_anc_probs" in case:
                assert_anc_probs(anc_prob_seqs, case["expected_freq"], case["expected_anc_probs"])
            else:
                assert_anc_probs(anc_prob_seqs, case["expected_freq"])


def test_transposed() -> None:
    """Test transposed ancestral probabilities."""
    filename = os.path.dirname(__file__) + "/../tests/data/simple.fa"
    with SequenceDataset(filename) as data:
        sequences = get_simple_seq(data)
    n = sequences.shape[1]
    config = Configuration.make_default(1)
    anc_probs_layer = Training.make_anc_probs_layer(1, config)
    msa_hmm_layer = Training.make_msa_hmm_layer(n, 10, config)
    msa_hmm_layer.build((1, None, None, len(SequenceDataset.alphabet)))
    B = msa_hmm_layer.cell.emitter[0].make_B()[0]
    config["transposed"] = True
    anc_probs_layer_transposed = Training.make_anc_probs_layer(n, config)
    anc_prob_seqs = anc_probs_layer_transposed(sequences, np.arange(n)[np.newaxis, :]).numpy()
    shape = (config["num_models"], n, sequences.shape[2], config["num_rate_matrices"], len(SequenceDataset.alphabet))
    anc_prob_seqs = np.reshape(anc_prob_seqs, shape)
    anc_prob_seqs = tf.cast(anc_prob_seqs, B.dtype)
    anc_prob_B = anc_probs_layer(B[tf.newaxis, tf.newaxis, :, :20], rate_indices=[[0]])
    anc_prob_B = tf.squeeze(anc_prob_B)
    prob1 = tf.linalg.matvec(B, anc_prob_seqs)
    oh_seqs = tf.one_hot(sequences, 20, dtype=anc_prob_B.dtype)
    oh_seqs = tf.expand_dims(oh_seqs, -2)
    prob2 = tf.linalg.matvec(anc_prob_B, oh_seqs)
    np.testing.assert_allclose(
        prob1.numpy(), prob2.numpy(),
        rtol=1e-5, atol=1e-4,
    )
