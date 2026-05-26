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
        for equi_tracks in model_equi:
            for equi in equi_tracks:
                assert_equilibrium(equi)
    for model_exchange in R:
        for exchange_tracks in model_exchange:
            for exchange in exchange_tracks:
                assert_symmetric(exchange)
    for model_rate, model_equi in zip(Q, p):
        for rate_tracks, equi_tracks in zip(model_rate, model_equi):
            for rate, equi in zip(rate_tracks, equi_tracks):
                assert_rate_matrix(rate, equi)


def get_test_configs(sequences : np.ndarray) -> list[dict]:
    """Generate test configurations for ancestral probabilities.

    Args:
        sequences: Shape (b, L, num_model) with integer indices
    """
    # Assuming sequences only contain the 20 standard AAs
    oh_sequences = tf.one_hot(sequences, 20)
    inv_sp_R, log_p = Initializers.make_substitution_model_init(1)
    p = tf.nn.softmax(log_p)
    cases = []
    for rate_init in [-100., -3., 100.]:
        for n in [1, 3]:
            case = {}

            config = Configuration()
            config.training.length_init = [10]*n
            config.training.no_sequence_weights = True
            case["config"] = config

            R_init, p_init = Initializers.make_substitution_model_init(n)
            case["R_init"] = Initializers.ConstantInitializer(R_init)
            case["p_init"] = Initializers.ConstantInitializer(p_init)
            case["t_init"] = Initializers.ConstantInitializer(rate_init)

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
    R_init : tf.keras.initializers.Initializer,
    p_init : tf.keras.initializers.Initializer,
    t_init : tf.keras.initializers.Initializer,
    num_rates: int,
    time_reversed: bool=False
) -> AncProbsLayer:
    return AncProbsLayer(
        heads = config.training.num_model,
        rates = num_rates,
        input_tracks = 1,
        equilibrium_init=p_init,
        rate_init=t_init,
        exchangeability_init=R_init,
        trainable_rates=config.tree.trainable_rates,
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
            case["config"], case["R_init"], case["p_init"], case["t_init"], n
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
            context.R_init = case["R_init"]
            context.p_init = case["p_init"]
            context.t_init = case["t_init"]
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
                anc_prob_seqs = model.encode_batch(x)[0].numpy()

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
    R_init, p_init = Initializers.make_substitution_model_init(1)
    anc_probs_layer = make_anc_probs_layer(
        config,
        R_init=Initializers.ConstantInitializer(R_init),
        p_init=Initializers.ConstantInitializer(p_init),
        t_init=Initializers.ConstantInitializer(-3.0),
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


def test_mixture_model() -> None:
    """Test AncProbsLayer with num_components > 1 (mixture of GTR models)."""
    config = Configuration()
    config.training.num_model = 2
    num_components = 3
    R_init, p_init = Initializers.make_substitution_model_init(
        config.training.num_model, num_components=num_components
    )

    # Get sequence data
    filename = os.path.dirname(__file__) + "/../tests/data/simple.fa"
    with SequenceDataset(filename) as data:
        sequences = get_simple_seq(data)
    n = sequences.shape[0]
    rate_indices = np.arange(n)[:, np.newaxis]
    sequences_onehot = tf.one_hot(sequences, depth=20, dtype=tf.float32).numpy()

    layer = AncProbsLayer(
        heads=config.training.num_model,
        rates=n,
        input_tracks=1,
        equilibrium_init=Initializers.ConstantInitializer(p_init),
        rate_init=Initializers.ConstantInitializer(0.0),
        exchangeability_init=Initializers.ConstantInitializer(R_init),
        num_components=num_components,
    )
    layer.build()

    # Verify mixture weights sum to 1
    w = layer.make_w(rate_indices).numpy()
    assert w.shape == (n, rate_indices.shape[1], 1, num_components)
    np.testing.assert_allclose(w.sum(axis=-1), 1.0, atol=1e-6)

    # Verify P is a valid stochastic matrix (rows sum to 1)
    tau = layer.make_tau(rate_indices)
    P = layer.make_P(tau, subset=rate_indices).numpy()
    assert P.shape == (n, config.training.num_model, 1, 20, 20)
    np.testing.assert_allclose(P.sum(axis=-1), 1.0, atol=1e-5)

    # Verify Q and p shapes include K
    assert layer.make_Q().shape == (config.training.num_model, 1, num_components, 20, 20)
    assert layer.make_p().shape == (config.training.num_model, 1, num_components, 20)

    # Verify call output shape is unchanged (same as K=1)
    out = layer(sequences_onehot, rate_indices=rate_indices)
    assert out.shape == (n, sequences.shape[1], config.training.num_model, 20)


def test_shared_equilibrium() -> None:
    """Test AncProbsLayer with shared_equilibrium=True.

    All mixture components should share a single equilibrium distribution,
    so the equilibrium kernel has shape (H, I, 1, D) while make_p() tiles
    it to (H, I, K, D).
    """
    config = Configuration()
    config.training.num_model = 2
    num_components = 3
    R_init, p_init = Initializers.make_substitution_model_init(
        config.training.num_model, num_components=num_components
    )

    filename = os.path.dirname(__file__) + "/../tests/data/simple.fa"
    with SequenceDataset(filename) as data:
        sequences = get_simple_seq(data)
    n = sequences.shape[0]
    rate_indices = np.arange(n)[:, np.newaxis]
    sequences_onehot = tf.one_hot(sequences, depth=20, dtype=tf.float32).numpy()

    layer = AncProbsLayer(
        heads=config.training.num_model,
        rates=n,
        input_tracks=1,
        equilibrium_init=Initializers.ConstantInitializer(p_init),
        rate_init=Initializers.ConstantInitializer(0.0),
        exchangeability_init=Initializers.ConstantInitializer(R_init),
        num_components=num_components,
        shared_equilibrium=True,
    )
    layer.build()

    # Equilibrium kernel stores only one set of frequencies per (H, I)
    assert layer.equilibrium_kernel.shape == (config.training.num_model, 1, 1, 20)

    # make_p() tiles to K components; all components must be identical
    p = layer.make_p().numpy()
    assert p.shape == (config.training.num_model, 1, num_components, 20)
    for k in range(1, num_components):
        np.testing.assert_array_equal(p[:, :, 0, :], p[:, :, k, :])

    # Each equilibrium distribution must sum to 1
    np.testing.assert_allclose(p.sum(axis=-1), 1.0, atol=1e-6)

    # P must be a valid stochastic matrix
    tau = layer.make_tau(rate_indices)
    P = layer.make_P(tau, subset=rate_indices).numpy()
    assert P.shape == (n, config.training.num_model, 1, 20, 20)
    np.testing.assert_allclose(P.sum(axis=-1), 1.0, atol=1e-5)

    # Call output shape must match the K=1 case
    out = layer(sequences_onehot, rate_indices=rate_indices)
    assert out.shape == (n, sequences.shape[1], config.training.num_model, 20)


def test_two_input_tracks() -> None:
    """Test AncProbsLayer with input_tracks=2.

    Passes original sequences as track 0 and a batch-permuted copy as track 1.
    With a single rate cluster (rates=1) all sequences share the same branch
    length, and the default neutral tau_track_kernel (softplus → 1.0) makes
    both tracks use identical effective branch lengths. Under these conditions
    the output of track 1 must equal the output of track 0 reindexed by the
    same permutation.
    """
    filename = os.path.dirname(__file__) + "/../tests/data/simple.fa"
    with SequenceDataset(filename) as data:
        sequences = get_simple_seq(data)  # (B, L, 1) integer indices
    n = sequences.shape[0]
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)

    H = 1
    R_init, p_init = Initializers.make_substitution_model_init(H)

    layer = AncProbsLayer(
        heads=H,
        rates=1,           # single branch-length cluster shared by all sequences
        input_tracks=2,
        equilibrium_init=Initializers.ConstantInitializer(p_init),
        rate_init=Initializers.ConstantInitializer(-3.0),
        exchangeability_init=Initializers.ConstantInitializer(R_init),
        alphabet_size=20,
    )
    layer.build()

    # Verify the new kernel shapes introduced by the refactor
    assert layer.tau_kernel.shape == (1, H)
    assert layer.tau_track_kernel.shape == (H, 2)

    # All sequences map to the single rate cluster
    rate_indices = np.zeros((n, H), dtype=np.int32)

    sequences_onehot = tf.one_hot(sequences, depth=20, dtype=tf.float32).numpy()
    sequences_perm_onehot = sequences_onehot[perm]  # batch-permuted copy

    out_0, out_1 = layer(
        sequences_onehot, sequences_perm_onehot,
        rate_indices=rate_indices,
    )  # each (B, L, H=1, 20)
    out_0 = out_0.numpy()
    out_1 = out_1.numpy()

    # Both outputs must be valid probability distributions
    np.testing.assert_allclose(out_0.sum(axis=-1), 1.0, atol=1e-5)
    np.testing.assert_allclose(out_1.sum(axis=-1), 1.0, atol=1e-5)

    # With equal per-track conversion rates and a shared branch length,
    # out_1[i] = anc_probs(sequences[perm[i]], tau_0) = out_0[perm[i]]
    np.testing.assert_allclose(out_1, out_0[perm], rtol=1e-5, atol=1e-5)
