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
    hmm.match_emissions = np.eye(23)[[[0,1,2,3], [0,1,2,-1]]]
    hmm.insert_emissions = np.eye(23)[7]
    structure = StructureConfig()
    structure.match_emissions = np.eye(20)[[[10,11,12,13], [10,11,12,-1]]]
    structure.insert_emissions = np.eye(20)[18]
    return Configuration(
        hmm=hmm,
        structure=structure,
    )

@pytest.fixture
def joint_emitter_from_marginals(
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
    emitter = JointProfileEmitter(marginal_values=[aa_values, struct_values])

    emitter.hmm_config = hidten_config
    input_shapes = ((None, None, 23), (None, None, 20))
    emitter.build(input_shapes)

    # add marginal priors for tests that need them
    aa_dirichlet = load_dirichlet(
        "amino_acid_dirichlet_1.weights",
        dim=23, components=1, states=[10, 8]
    )
    struct_dirichlet = load_dirichlet(
        "pfam_35_3Di_1.weights",
        dim=20, components=1, states=[10, 8]
    )
    emitter.add_marginal_prior(0, aa_dirichlet)
    emitter.add_marginal_prior(1, struct_dirichlet)

    return emitter

@pytest.fixture
def joint_emitter_from_values(
    config: Configuration,
    hidten_config: HidtenHMMConfig
) -> JointProfileEmitter:
    lengths = [4, 3]

    aa_head_1 = [21, 11, 3, 6]
    struct_head_1 = [10, 11, 12, 13]
    aa_head_2 = [13, 4, 15]
    struct_head_2 = [2, 3, 4]
    aa_insert_head_1 = 7
    struct_insert_head_1 = 18
    aa_insert_head_2 = 17
    struct_insert_head_2 = 3
    values_1 = PHMMValueSet(
        L=lengths[0],
        match_emissions=np.eye(20*23)[
            [aa_head_1[i]*20 + struct_head_1[i] for i in range(4)]
        ],
        insert_emissions=np.eye(20*23)[
            aa_insert_head_1 * 20 + struct_insert_head_1
        ]
    )
    values_2 = PHMMValueSet(
        L=lengths[1],
        match_emissions=np.eye(20*23)[
            [aa_head_2[i]*20 + struct_head_2[i] for i in range(3)]
        ],
        insert_emissions=np.eye(20*23)[
            aa_insert_head_2 * 20 + struct_insert_head_2
        ]
    )
    joint_values = [values_1, values_2]

    # Construct an emitter with two heads from the initial values
    emitter = JointProfileEmitter(values = joint_values)

    emitter.hmm_config = hidten_config
    input_shapes = ((None, None, 23), (None, None, 20))
    emitter.build(input_shapes)

    return emitter

def test_matrix_from_marginals(
    joint_emitter_from_marginals: JointProfileEmitter
) -> None:
    B = joint_emitter_from_marginals.matrix()

    # Check basic matrix properties
    assert B.shape == (2, 10, 23 * 20)
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
    joint_emitter_from_values: JointProfileEmitter
) -> None:
    B = joint_emitter_from_values.matrix()

    # Check basic matrix properties
    assert B.shape == (2, 10, 23 * 20)
    np.testing.assert_allclose(np.sum(B[0], axis=-1), 1.0, rtol=1e-6)
    np.testing.assert_allclose(np.sum(B[1, :8], axis=-1), 1.0, rtol=1e-6)

    # Check match emissions of head 1
    expected_indices_head_1 = [21*20 + 10, 11*20 + 11, 3*20 + 12, 6*20 + 13]
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
    joint_emitter_from_marginals: JointProfileEmitter,
) -> None:
    # The the marginal matrices
    aa_matrix, struct_matrix = joint_emitter_from_marginals.marginal_matrices()
    assert aa_matrix.shape == (2, 10, 23)
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
    joint_emitter_from_values: JointProfileEmitter,
) -> None:

    aa_head_1 = [21, 11, 3, 6]
    struct_head_1 = [10, 11, 12, 13]
    aa_head_2 = [13, 4, 15]
    struct_head_2 = [2, 3, 4]
    aa_insert_head_1 = 7
    struct_insert_head_1 = 18
    aa_insert_head_2 = 17
    struct_insert_head_2 = 3

    aa_inputs_1 = np.eye(23)[aa_head_1 + [aa_insert_head_1]]
    struct_inputs_1 = np.eye(20)[struct_head_1 + [struct_insert_head_1]]
    aa_inputs_2 = np.eye(23)[aa_head_2 + [aa_insert_head_2]]
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
