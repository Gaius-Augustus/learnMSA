from collections.abc import Callable

import numpy as np
import pytest
import tensorflow as tf
from hidten.hmm import HMMConfig as HidtenHMMConfig
from hidten.tf.prior.dirichlet import TFDirichletPrior

from learnMSA.config import Configuration, PHMMConfig, StructureConfig
from learnMSA.hmm.tf.joint_profile_emitter import JointProfileEmitter
from learnMSA.hmm.tf.util import load_dirichlet
from learnMSA.hmm.util.value_set import PHMMValueSet
from learnMSA.util.sequence_dataset import SequenceDataset


@pytest.fixture
def hidten_config() -> HidtenHMMConfig:
    lengths = [4, 3]
    return HidtenHMMConfig(states=[2*L+2 for L in lengths])

@pytest.fixture
def config() -> Configuration:
    hmm = PHMMConfig()
    hmm.match_emissions = np.eye(20)[[[0,1,2,3], [0,1,2,-1]]]
    hmm.insert_emissions = np.eye(20)[7]
    structure = StructureConfig()
    structure.match_emissions = np.eye(20)[[[10,11,12,13], [10,11,12,-1]]]
    structure.insert_emissions = np.eye(20)[18]
    return Configuration(
        hmm=hmm,
        structure=structure,
    )

def make_joint_emitter_from_marginals(
    config: Configuration,
    hidten_config: HidtenHMMConfig
) -> JointProfileEmitter:
    lengths = [4, 3]

    # Create value sets
    aa_values = [
        PHMMValueSet.from_config(L, h, config.hmm)
        for h, L in enumerate(lengths)
    ]
    struct_values = [
        PHMMValueSet.from_structural_config(L, h, config.structure)
        for h, L in enumerate(lengths)
    ]

    # Construct an emitter with two heads from the marginal values,
    # inferring the initial joint distribution as the product of marginals
    emitter = JointProfileEmitter(
        marginal_values=[aa_values, struct_values],
        low_rank=config.structure.joint_emission_low_rank,
        conditional=False,
    )

    emitter.hmm_config = hidten_config
    input_shapes = ((None, None, 20), (None, None, 20))
    emitter.build(input_shapes)

    # add marginal priors for tests that need them
    aa_dirichlet = load_dirichlet(
        "pfam_aa_neff_conc_3_20_1.weights",
        dim=20, components=1, states=[10, 8]
    )
    struct_dirichlet = load_dirichlet(
        "pfam_35_3Di_1.weights",
        dim=20, components=1, states=[10, 8]
    )
    emitter.add_marginal_prior(0, aa_dirichlet)
    emitter.add_marginal_prior(1, struct_dirichlet)

    return emitter

def make_joint_emitter_from_values(
    config: Configuration,
    hidten_config: HidtenHMMConfig
) -> JointProfileEmitter:
    lengths = [4, 3]

    aa_head_1 = [19, 11, 3, 6]
    struct_head_1 = [10, 11, 12, 13]
    aa_head_2 = [13, 4, 15]
    struct_head_2 = [2, 3, 4]
    aa_insert_head_1 = 7
    struct_insert_head_1 = 18
    aa_insert_head_2 = 17
    struct_insert_head_2 = 3
    values_1 = PHMMValueSet(
        L=lengths[0],
        match_emissions=np.eye(20*20)[
            [aa_head_1[i]*20 + struct_head_1[i] for i in range(4)]
        ],
        insert_emissions=np.eye(20*20)[
            aa_insert_head_1 * 20 + struct_insert_head_1
        ]
    )
    values_2 = PHMMValueSet(
        L=lengths[1],
        match_emissions=np.eye(20*20)[
            [aa_head_2[i]*20 + struct_head_2[i] for i in range(3)]
        ],
        insert_emissions=np.eye(20*20)[
            aa_insert_head_2 * 20 + struct_insert_head_2
        ]
    )
    joint_values = [values_1, values_2]

    # Construct an emitter with two heads from the initial values
    emitter = JointProfileEmitter(
        values=joint_values,
        low_rank=config.structure.joint_emission_low_rank,
        conditional=False,
    )

    emitter.hmm_config = hidten_config
    input_shapes = ((None, None, 20), (None, None, 20))
    emitter.build(input_shapes)

    return emitter

def test_matrix_from_marginals(
    config: Configuration,
    hidten_config: HidtenHMMConfig
) -> None:
    joint_emitter_from_marginals = make_joint_emitter_from_marginals(
        config, hidten_config
    )
    B = joint_emitter_from_marginals.matrix()

    # Check basic matrix properties
    assert B.shape == (2, 10, 20 * 20)
    np.testing.assert_allclose(np.sum(B[0], axis=-1), 1.0, rtol=1e-6)
    np.testing.assert_allclose(np.sum(B[1, :8], axis=-1), 1.0, rtol=1e-6)

    # Check match emissions of head 1
    for i in range(4):
        expected_aa = i
        expected_struct = i + 10
        expected_index = expected_aa * 20 + expected_struct
        np.testing.assert_allclose(B[0, i, expected_index], 1.0, rtol=1e-6)

    # Check match emissions of head 2
    for i in range(3):
        expected_aa = i
        expected_struct = i + 10
        expected_index = expected_aa * 20 + expected_struct
        np.testing.assert_allclose(B[1, i, expected_index], 1.0, rtol=1e-6)

    # Check insertions
    expected_aa = 7
    expected_struct = 18
    expected_index = expected_aa * 20 + expected_struct
    np.testing.assert_allclose(B[0, 4:, expected_index], 1.0, rtol=1e-6)
    np.testing.assert_allclose(B[1, 3:8, expected_index], 1.0, rtol=1e-6)

def test_matrix_from_values(
    config: Configuration,
    hidten_config: HidtenHMMConfig
) -> None:
    joint_emitter_from_values = make_joint_emitter_from_values(
        config, hidten_config
    )
    B = joint_emitter_from_values.matrix()

    # Check basic matrix properties
    assert B.shape == (2, 10, 20 * 20)
    np.testing.assert_allclose(np.sum(B[0], axis=-1), 1.0, rtol=1e-6)
    np.testing.assert_allclose(np.sum(B[1, :8], axis=-1), 1.0, rtol=1e-6)

    # Check match emissions of head 1
    expected_indices_head_1 = [19*20 + 10, 11*20 + 11, 3*20 + 12, 6*20 + 13]
    for i in range(4):
        expected_index = expected_indices_head_1[i]
        np.testing.assert_allclose(B[0, i, expected_index], 1.0, rtol=1e-6)

    # Check match emissions of head 2
    expected_indices_head_2 = [13*20 + 2, 4*20 + 3, 15*20 + 4]
    for i in range(3):
        expected_index = expected_indices_head_2[i]
        np.testing.assert_allclose(B[1, i, expected_index], 1.0, rtol=1e-6)

    # Check insertions
    expected_index_head_1 = 7*20 + 18
    expected_index_head_2 = 17*20 + 3
    np.testing.assert_allclose(B[0, 4:, expected_index_head_1], 1.0, rtol=1e-6)
    np.testing.assert_allclose(B[1, 3:8, expected_index_head_2], 1.0, rtol=1e-6)

def test_marginal_matrix_and_priors(
    config: Configuration,
    hidten_config: HidtenHMMConfig
) -> None:
    joint_emitter_from_marginals = make_joint_emitter_from_marginals(
        config, hidten_config
    )
    # The the marginal matrices
    aa_matrix, struct_matrix = joint_emitter_from_marginals.marginal_matrices()
    assert aa_matrix.shape == (2, 10, 20)
    assert struct_matrix.shape == (2, 10, 20)

    aa_prior = joint_emitter_from_marginals.get_marginal_prior(0)
    struct_prior = joint_emitter_from_marginals.get_marginal_prior(1)
    assert isinstance(aa_prior, TFDirichletPrior)
    assert isinstance(struct_prior, TFDirichletPrior)

    prior_scores = joint_emitter_from_marginals.prior_scores()

    # Prior scores (log scale) should be the sum of the individual marginal
    # priors scores for each head
    aa_scores = aa_prior(aa_matrix)
    struct_scores = struct_prior(struct_matrix)
    expected_scores = aa_scores + struct_scores
    np.testing.assert_allclose(prior_scores, expected_scores)
    assert all(prior_scores.numpy() != 0)

def test_call(
    config: Configuration,
    hidten_config: HidtenHMMConfig
) -> None:
    joint_emitter_from_values = make_joint_emitter_from_values(
        config, hidten_config
    )

    aa_head_1 = [19, 11, 3, 6]
    struct_head_1 = [10, 11, 12, 13]
    aa_head_2 = [13, 4, 15]
    struct_head_2 = [2, 3, 4]
    aa_insert_head_1 = 7
    struct_insert_head_1 = 18
    aa_insert_head_2 = 17
    struct_insert_head_2 = 3

    aa_inputs_1 = np.eye(20)[aa_head_1 + [aa_insert_head_1]]
    struct_inputs_1 = np.eye(20)[struct_head_1 + [struct_insert_head_1]]
    aa_inputs_2 = np.eye(20)[aa_head_2 + [aa_insert_head_2]]
    struct_inputs_2 = np.eye(20)[struct_head_2 + [struct_insert_head_2]]

    # Add batch dimensions
    aa_inputs_1 = aa_inputs_1[np.newaxis]
    struct_inputs_1 = struct_inputs_1[np.newaxis]
    aa_inputs_2 = aa_inputs_2[np.newaxis]
    struct_inputs_2 = struct_inputs_2[np.newaxis]

    aa_inputs_2_padded = np.pad(
        aa_inputs_2, [[0, 0], [0, 1], [0, 0]], constant_values=0.0
    )
    struct_inputs_2_padded = np.pad(
        struct_inputs_2, [[0, 0], [0, 1], [0, 0]], constant_values=0.0
    )
    stacked_aa_inputs = np.stack(
        [aa_inputs_1, aa_inputs_2_padded], axis=2
    )
    stacked_struct_inputs = np.stack(
        [struct_inputs_1, struct_inputs_2_padded], axis=2
    )

    E_1 = joint_emitter_from_values(
        aa_inputs_1, struct_inputs_1 # type: ignore
    )[:,:,0]
    E_2 = joint_emitter_from_values(
        aa_inputs_2, struct_inputs_2 # type: ignore
    )[:,:,1]
    E = joint_emitter_from_values(
        stacked_aa_inputs, stacked_struct_inputs # type: ignore
    )

    assert np.allclose(E_1[:,:4,:4], np.eye(4))
    assert np.allclose(E_1[:,4,:], [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    assert np.allclose(E_2[:,:3,:3], np.eye(3))
    assert np.allclose(E_2[:,3,:], [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1])
    assert np.allclose(E[:,:4,0,:4], np.eye(4))
    assert np.allclose(E[:,:3,1,:3], np.eye(3))

@pytest.mark.parametrize("low_rank", [1, 2, 4])
def test_matrix_low_rank(
    low_rank: int,
    config: Configuration,
    hidten_config: HidtenHMMConfig
) -> None:
    """Test that the low-rank C + AB^T parameterization recovers the product
    of marginals at initialization.

    C encodes the log-outer-product of the marginals; A and B are initialised
    near zero so AB^T \u2248 0.  At init, softmax(C) = P(aa) \u2297 P(struct),
    recovering the independent joint for any low_rank >= 1.
    """
    config.structure.joint_emission_low_rank = low_rank
    emitter = make_joint_emitter_from_marginals(config, hidten_config)
    B = emitter.matrix()

    # Check if the kernel has the correct size
    assert emitter.low_rank == low_rank
    assert emitter.AB_matrix().shape == (2, 10, (20 + 20) * low_rank)

    # Check basic matrix properties
    assert B.shape == (2, 10, 20 * 20)
    np.testing.assert_allclose(np.sum(B[0], axis=-1), 1.0, rtol=1e-5)
    np.testing.assert_allclose(np.sum(B[1, :8], axis=-1), 1.0, rtol=1e-5)

    # The config uses one-hot marginals: aa peaks at position i, struct peaks
    # at i+10. The joint product is therefore a one-hot at i*20 + (i+10).

    # Check match emissions of head 1
    for i in range(4):
        expected_index = i * 20 + (i + 10)
        np.testing.assert_allclose(B[0, i, expected_index], 1.0, rtol=1e-5)

    # Check match emissions of head 2
    for i in range(3):
        expected_index = i * 20 + (i + 10)
        np.testing.assert_allclose(B[1, i, expected_index], 1.0, rtol=1e-5)

    # Check insertions: aa insert=7, struct insert=18
    expected_insert_index = 7 * 20 + 18
    np.testing.assert_allclose(B[0, 4:, expected_insert_index], 1.0, rtol=1e-5)
    np.testing.assert_allclose(B[1, 3:8, expected_insert_index], 1.0, rtol=1e-5)


def test_C_weight(
    config: Configuration,
    hidten_config: HidtenHMMConfig
) -> None:
    """Test that C_weight is non-trainable and equals log(p1) + log(p2)."""
    from learnMSA.hmm.tf.joint_profile_emitter import compute_C_from_marginals
    config.structure.joint_emission_low_rank = 2
    emitter = make_joint_emitter_from_marginals(config, hidten_config)

    assert emitter.C_weight.trainable is False

    lengths = [4, 3]
    aa_values = [
        PHMMValueSet.from_config(L, h, config.hmm)
        for h, L in enumerate(lengths)
    ]
    struct_values = [
        PHMMValueSet.from_structural_config(L, h, config.structure)
        for h, L in enumerate(lengths)
    ]
    # Verify C_weight values for head 0 match states
    for i in range(4):
        expected_C = compute_C_from_marginals(
            aa_values[0].match_emissions[i],
            struct_values[0].match_emissions[i],
        )  # (n1, n2)
        actual_C = emitter.C_weight.numpy()[0, i, :].reshape(20, 20)
        np.testing.assert_allclose(actual_C, expected_C, rtol=1e-5)

    # Verify insert state C
    expected_C_ins = compute_C_from_marginals(
        aa_values[0].insert_emissions,
        struct_values[0].insert_emissions,
    )
    actual_C_ins = emitter.C_weight.numpy()[0, 4, :].reshape(20, 20)
    np.testing.assert_allclose(actual_C_ins, expected_C_ins, rtol=1e-5)


def test_AB_matrix_requires_low_rank(
    config: Configuration,
    hidten_config: HidtenHMMConfig
) -> None:
    """Test that AB_matrix() raises AssertionError when low_rank == 0."""
    emitter = make_joint_emitter_from_marginals(config, hidten_config)
    assert emitter.low_rank == 0
    with pytest.raises(AssertionError):
        emitter.AB_matrix()


def make_conditional_emitter(
    config: Configuration,
    hidten_config: HidtenHMMConfig,
    low_rank: int = 0,
) -> JointProfileEmitter:
    """Builds a JointProfileEmitter with conditional=True from marginal values."""
    lengths = [4, 3]
    aa_values = [
        PHMMValueSet.from_config(L, h, config.hmm)
        for h, L in enumerate(lengths)
    ]
    struct_values = [
        PHMMValueSet.from_structural_config(L, h, config.structure)
        for h, L in enumerate(lengths)
    ]
    emitter = JointProfileEmitter(
        marginal_values=[aa_values, struct_values],
        low_rank=low_rank,
        conditional=True,
    )
    emitter.hmm_config = hidten_config
    emitter.build(((None, None, 20), (None, None, 20)))
    return emitter


def test_conditional_matrix_normalization(
    config: Configuration,
    hidten_config: HidtenHMMConfig,
) -> None:
    """Full distribution (low_rank=0): each conditional row P(x2 | x1=i, s)
    must sum to 1 for valid states."""
    emitter = make_conditional_emitter(config, hidten_config, low_rank=0)
    B = emitter.matrix()
    assert B.shape == (2, 10, 20 * 20)

    D1, D2 = 20, 20
    B_reshaped = B.numpy().reshape(2, 10, D1, D2)

    # Head 0: all 10 states are valid
    np.testing.assert_allclose(
        B_reshaped[0, :, :, :].sum(axis=-1),
        np.ones((10, D1)),
        rtol=1e-6,
    )
    # Head 1: only 8 states are valid (L=3 → 2*3+2=8)
    np.testing.assert_allclose(
        B_reshaped[1, :8, :, :].sum(axis=-1),
        np.ones((8, D1)),
        rtol=1e-6,
    )


def test_conditional_initializes_independent_of_x1(
    config: Configuration,
    hidten_config: HidtenHMMConfig,
) -> None:
    """When initialized from marginals with conditional=True, the emitter uses
    only the x2 marginal so that P(x2 | x1=i, s) = P(x2 | s) for every x1.
    Because the struct marginals are one-hot (position i+10 for match state i),
    every conditional row must be one-hot at x2 = i+10, regardless of x1.
    """
    emitter = make_conditional_emitter(config, hidten_config, low_rank=0)
    B = emitter.matrix()

    D1, D2 = 20, 20
    B_reshaped = B.numpy().reshape(2, 10, D1, D2)

    # Head 0 has 4 match states. The struct marginal for state i is one-hot
    # at position i+10, so every conditional row must be one-hot there.
    for i in range(4):
        for d1 in range(D1):
            np.testing.assert_allclose(
                B_reshaped[0, i, d1, :],
                np.eye(D2)[i + 10],
                rtol=1e-5,
            )


def test_marginal_matrix_from_conditional(
    config: Configuration,
    hidten_config: HidtenHMMConfig,
) -> None:
    """marginal_matrix_from_conditional returns the correct weighted average
    of conditional rows."""
    emitter = make_conditional_emitter(config, hidten_config, low_rank=0)
    B = emitter.matrix()

    D1, D2 = 20, 20
    B_reshaped = B.numpy().reshape(2, 10, D1, D2)

    # One-hot prior selects a single row of the conditional
    for d1_idx in [0, 5, 19]:
        prior_onehot = tf.one_hot(d1_idx, D1)
        marginal = emitter.marginal_matrix_from_conditional(prior_onehot, B)
        assert marginal.shape == (2, 10, D2)
        np.testing.assert_allclose(
            marginal.numpy(), B_reshaped[:, :, d1_idx, :], rtol=1e-5
        )

    # Uniform prior: marginal equals the mean over x1 rows
    prior_uniform = tf.ones([D1], dtype=tf.float32) / D1
    marginal_uniform = emitter.marginal_matrix_from_conditional(prior_uniform, B)
    expected_uniform = B_reshaped.mean(axis=2)  # (H, Q, D2)
    np.testing.assert_allclose(marginal_uniform.numpy(), expected_uniform, rtol=1e-5)


@pytest.mark.parametrize("low_rank", [2, 4])
def test_conditional_low_rank_normalization(
    low_rank: int,
    config: Configuration,
    hidten_config: HidtenHMMConfig,
) -> None:
    """Low-rank conditional: each conditional row P(x2 | x1=i, s) must sum
    to 1 for valid states."""
    config.structure.joint_emission_low_rank = low_rank
    emitter = make_conditional_emitter(config, hidten_config, low_rank=low_rank)
    B = emitter.matrix()
    assert B.shape == (2, 10, 20 * 20)

    D1, D2 = 20, 20
    B_reshaped = B.numpy().reshape(2, 10, D1, D2)

    np.testing.assert_allclose(
        B_reshaped[0, :, :, :].sum(axis=-1),
        np.ones((10, D1)),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        B_reshaped[1, :8, :, :].sum(axis=-1),
        np.ones((8, D1)),
        rtol=1e-5,
    )


def make_conditional_prior_emitter(
    config: Configuration,
    hidten_config: HidtenHMMConfig,
    low_rank: int = 0,
    components: int = 1,
    with_prior: bool = True,
) -> JointProfileEmitter:
    """A conditional emitter with background (non-degenerate) emissions and a
    structural Dirichlet prior on its conditional rows."""
    lengths = [4, 3]
    plain = Configuration(hmm=PHMMConfig(), structure=StructureConfig())
    aa_values = [
        PHMMValueSet.from_config(L, h, plain.hmm)
        for h, L in enumerate(lengths)
    ]
    struct_values = [
        PHMMValueSet.from_structural_config(L, h, plain.structure)
        for h, L in enumerate(lengths)
    ]
    emitter = JointProfileEmitter(
        marginal_values=[aa_values, struct_values],
        low_rank=low_rank,
        conditional=True,
    )
    emitter.hmm_config = hidten_config
    if with_prior:
        # Keras refuses new sub-layers on a built layer, so the prior is
        # attached first -- the same order PHMMLayer uses.
        emitter.prior = load_dirichlet(
            f"pfam_35_3Di_{components}.weights",
            dim=20, components=components, states=[10, 8],
        )
    emitter.build(((None, None, 20), (None, None, 20)))
    return emitter


@pytest.mark.parametrize("low_rank", [0, 2])
@pytest.mark.parametrize("components", [1, 9])
def test_conditional_prior_scores_sum_over_rows(
    low_rank: int,
    components: int,
    config: Configuration,
    hidten_config: HidtenHMMConfig,
) -> None:
    """The prior is applied to each of the D1 conditional rows and summed."""
    emitter = make_conditional_prior_emitter(
        config, hidten_config, low_rank=low_rank, components=components
    )
    B = emitter.matrix()
    rows = tf.reshape(B, [2, 10, 20, 20])
    expected = sum(emitter._prior(rows[:, :, i]) for i in range(20))

    scores = emitter.prior_scores()
    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores.numpy()))
    np.testing.assert_allclose(
        scores.numpy(), expected.numpy(),
        rtol=1e-4,  # float32 accumulation over the 20 rows
    )


@pytest.mark.parametrize("low_rank", [0, 2])
def test_conditional_prior_scores_at_init(
    low_rank: int,
    config: Configuration,
    hidten_config: HidtenHMMConfig,
) -> None:
    """At initialisation every row equals the structural marginal, so the score
    is exactly D1 times the score of a single row."""
    emitter = make_conditional_prior_emitter(
        config, hidten_config, low_rank=low_rank
    )
    B = emitter.matrix()
    rows = tf.reshape(B, [2, 10, 20, 20])
    single_row = emitter._prior(rows[:, :, 0])
    np.testing.assert_allclose(
        emitter.prior_scores().numpy(),
        (20 * single_row).numpy(),
        rtol=1e-4,  # float32 accumulation over the 20 rows
    )


def test_conditional_prior_ignores_padding_states(
    config: Configuration,
    hidten_config: HidtenHMMConfig,
) -> None:
    """Head 1 has 8 of the 10 padded states. With identical emissions in every
    state the two heads' scores must be in a 10:8 ratio -- padded rows sum to
    zero and are masked out by the prior."""
    emitter = make_conditional_prior_emitter(config, hidten_config)
    scores = emitter.prior_scores().numpy()
    np.testing.assert_allclose(scores[0] / scores[1], 10 / 8, rtol=1e-5)


def test_marginal_prior_rejected_when_conditional(
    config: Configuration,
    hidten_config: HidtenHMMConfig,
) -> None:
    """Marginal priors need the joint parameterisation; the conditional table
    has no marginals to score."""
    emitter = make_conditional_prior_emitter(
        config, hidten_config, with_prior=False
    )
    prior = load_dirichlet(
        "pfam_35_3Di_1.weights", dim=20, components=1, states=[10, 8]
    )
    with pytest.raises(AssertionError):
        emitter.add_marginal_prior(1, prior)
