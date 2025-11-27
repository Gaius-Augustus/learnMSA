import numpy as np
import pytest
import tensorflow as tf

import tests.hmm.ref as ref
from learnMSA.config.hmm import HMMConfig
from learnMSA.hmm.profile_emitter import ProfileEmitter
from learnMSA.hmm.value_set import PHMMValueSet

@pytest.fixture
def emitter() -> ProfileEmitter:
    lengths = [4, 3]

    # Create value sets for different heads
    values = [
        PHMMValueSet.from_config(L, h, ref.config)
        for h, L in enumerate(lengths)
    ]

    # Construct an emitter with two heads from the initial values
    emitter = ProfileEmitter(values)
    emitter.build()
    return emitter

def test_matrix(emitter: ProfileEmitter) -> None:
    B = emitter.matrix()

    # Check basic matrix properties
    assert B.shape == (2, 5, 23)
    np.testing.assert_allclose(np.sum(B[0], axis=-1), 1.0)
    np.testing.assert_allclose(np.sum(B[1, :4], axis=-1), 1.0)


def test_call_shapes(emitter: ProfileEmitter) -> None:
    # Create dummy emissions
    B, T, H, S = 2, 10, 2, 23
    inputs = np.random.randint(0, S, size=(B, T))
    inputs_with_heads = np.random.randint(0, S, size=(B, T, H))

    inputs_oh = tf.one_hot(inputs, depth=S)
    inputs_oh_heads = tf.one_hot(inputs_with_heads, depth=S)

    emission_scores = emitter.call(inputs_oh)
    emission_scores_heads = emitter.call(inputs_oh_heads)
    assert emission_scores.shape == (B, T, H, 2*4+3+1)  # max length is 4
    assert emission_scores_heads.shape == (B, T, H, 2*4+3+1)


def test_call(emitter: ProfileEmitter) -> None:
    # Create two value sets for different heads
    values_1 = PHMMValueSet(
        L=4,
        match_emissions=np.array([
            [0.9]+[0.1/22]*22,
            [0.1/22] + [0.9] + [0.1/22]*21,
            [0.1/22]*2 + [0.9] + [0.1/22]*20,
            [0.1/22]*3 + [0.9] + [0.1/22]*19,
        ]),
        insert_emissions=np.array([1./23]*23),
        transitions=np.array([]), # not used
        start=np.array([]), # not used
    )
    values_2 = PHMMValueSet(
        L=3,
        match_emissions=np.array([
            [0.8]+[0.2/22]*22,
            [0.2/22] + [0.8] + [0.2/22]*21,
            [0.2/22]*2 + [0.8] + [0.2/22]*20,
        ]),
        insert_emissions=np.array([1./23]*23),
        transitions=np.array([]), # not used
        start=np.array([]), # not used
    )

    # Construct an emitter with two heads from the initial values
    emitter = ProfileEmitter([values_1, values_2], use_prior_aa_dist=False)
    emitter.build()

    inputs_1_list = [0, 1, 2, 3, 9]
    inputs_2_list = [0, 1, 2, 16]
    inputs_1 = tf.one_hot(inputs_1_list, depth=23)
    inputs_2 = tf.one_hot(inputs_2_list, depth=23)
    # Add padding
    inputs_2 = tf.pad(inputs_2, [[0, 1], [0, 0]], constant_values=0.0)
    inputs = tf.stack([inputs_1, inputs_2], axis=0)  # (B, T, S)
    padding = np.sum(inputs.numpy(), axis=-1)

    emission_scores = emitter.call(inputs).numpy()

    # Head 1, sequence 1
    np.testing.assert_allclose(
        emission_scores[0, range(4), 0, range(4)], 0.9
    )
    np.testing.assert_allclose(
        emission_scores[0, 4, 0, :4], 0.1/22, atol=1e-6
    )
    np.testing.assert_allclose(
        emission_scores[0, :, 0, 4:-1], 1.0/23, atol=1e-6
    )
    # Head 1, sequence 2
    np.testing.assert_allclose(
        emission_scores[1, range(3), 0, range(3)], 0.9
    )
    np.testing.assert_allclose(
        emission_scores[1, 3, 0, :3], 0.1/22, atol=1e-6
    )
    np.testing.assert_allclose(
        emission_scores[1, 4, 0, :3], 0.0, atol=1e-6
    )
    np.testing.assert_allclose(
        emission_scores[1, :-1, 0, 4:-1], 1.0/23, atol=1e-6
    )
    np.testing.assert_allclose(
        emission_scores[1, -1, 0, -2], 0.0, atol=1e-6
    )

    # Test padding
    # See HidTen docs on how padding works
    # We expect the emitter to add a new, rightmost padding state
    np.testing.assert_allclose(emission_scores[..., -1], 1.0)


def test_dirichlet_prior(emitter: ProfileEmitter) -> None:
    print(emitter.matrix().numpy()[:,0])
    print(emitter.prior_scores().numpy())


def test_construct_big_emitter() -> None:
    import time

    lengths = [500]*10
    config = HMMConfig()
    values = [
        PHMMValueSet.from_config(L, h, config) for h, L in enumerate(lengths)
    ]

    # Construct an emitter with two heads from the initial values
    t0 = time.perf_counter()
    emitter = ProfileEmitter(values)
    t1 = time.perf_counter()
    print(f"Constructor time: {t1-t0:.4f}s")

    t0 = time.perf_counter()
    emitter.build()
    t1 = time.perf_counter()
    print(f"Build time: {t1-t0:.4f}s")

    t0 = time.perf_counter()
    M = emitter.matrix()
    t1 = time.perf_counter()
    print(f"Matrix time: {t1-t0:.4f}s")
