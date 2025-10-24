import numpy as np
import pytest
import tensorflow as tf

from learnMSA.msa_hmm import Initializers, Viterbi
from learnMSA.msa_hmm.Emitter import ProfileHMMEmitter
from learnMSA.msa_hmm.MsaHmmCell import MsaHmmCell
from learnMSA.msa_hmm.MsaHmmLayer import MsaHmmLayer
from learnMSA.msa_hmm.Transitioner import ProfileHMMTransitioner
from tests import ref, supervised_training


# Fixture to set up test data (replaces __init__ in unittest)
class MsaHmmCellData:
    """Helper class to store test reference data"""
    def __init__(self) -> None:
        self.length = [4, 3]
        self.emission_init = [
            ref.make_emission_init_A(), ref.make_emission_init_B()
        ]
        self.insertion_init = [ref.make_insertion_init() for i in range(2)]
        self.transition_init = [
            ref.make_transition_init_A(), ref.make_transition_init_B()
        ]
        A1, B1, I1 = ref.get_ref_model_A()
        A2, B2, I2 = ref.get_ref_model_B()
        self.A_ref = [A1, A2]
        self.B_ref = [B1, B2]
        self.init_ref = [I1, I2]
        self.ref_alpha = [ref.get_ref_forward_A(), ref.get_ref_forward_B()]
        self.ref_beta = [ref.get_ref_backward_A(), ref.get_ref_backward_B()]
        self.ref_lik = [ref.get_ref_lik_A(), ref.get_ref_lik_B()]
        self.ref_scaled_alpha = [
            ref.get_ref_scaled_forward_A(), ref.get_ref_scaled_forward_B()
        ]
        self.ref_posterior_probs = [
            ref.get_ref_posterior_probs_A(), ref.get_ref_posterior_probs_B()
        ]
        self.ref_gamma = [
            ref.get_ref_viterbi_variables_A(), ref.get_ref_viterbi_variables_B()
        ]
        self.ref_viterbi_path = [
            ref.get_ref_viterbi_path_A(), ref.get_ref_viterbi_path_B()
        ]


@pytest.fixture
def test_data() -> MsaHmmCellData:
    """Fixture to provide test reference data"""
    return MsaHmmCellData()


def make_test_cell(
        test_data, models
) -> tuple[MsaHmmCell, list[int]]:
    """Helper function to create test cell"""
    if not hasattr(models, '__iter__'):
        models = [models]

    # Get the lengths and initializations for the selected models
    length = [test_data.length[i] for i in models]
    e = [test_data.emission_init[i] for i in models]
    i = [test_data.insertion_init[i] for i in models]
    t = [test_data.transition_init[i] for i in models]
    f = [Initializers.make_default_flank_init() for _ in models]

    # Create the HMM cell and sublayers
    emitter = ProfileHMMEmitter(emission_init=e, insertion_init=i)
    transitioner = ProfileHMMTransitioner(transition_init=t, flank_init=f)
    hmm_cell = MsaHmmCell(
        length, dim=3, emitter=emitter,
        transitioner=transitioner, use_step_counter=True
    )
    hmm_cell.build((None, None, 3))

    return hmm_cell, length


def test_single_models(test_data : MsaHmmCellData) -> None:
    # Test setting up a single model
    for i in range(2):
        # Create the cell
        hmm_cell, length = make_test_cell(test_data, i)
        hmm_cell.recurrent_init()

        # Compute matrices
        A = hmm_cell.transitioner.make_A()
        init = hmm_cell.transitioner.make_initial_distribution()
        B = hmm_cell.emitter[0].make_B()

        # Compare to reference
        np.testing.assert_almost_equal(A[0], test_data.A_ref[i], decimal=5)
        np.testing.assert_almost_equal(B[0], test_data.B_ref[i], decimal=5)
        np.testing.assert_almost_equal(init[0, 0], test_data.init_ref[i], decimal=5)
        imp_log_probs = hmm_cell.transitioner.make_implicit_log_probs()[0][0]

        # Assert that all expected implicit parts are present and correct length
        for part_name in imp_log_probs.keys():
            parts = [
                part[0]
                for part in hmm_cell.transitioner.implicit_transition_parts[0]
            ]
            assert part_name in parts, \
                part_name + " is in the kernel but not under the expected "\
                    "kernel parts. Wrong spelling?"
        for part_name, l in hmm_cell.transitioner.implicit_transition_parts[0]:
            if part_name in imp_log_probs:
                kernel_length = tf.size(imp_log_probs[part_name]).numpy()
                assert kernel_length == l, \
                    "\"" + part_name + "\" implicit probs array has length "\
                          + str(kernel_length) + " but kernel length is " + str(l)


def test_multi_models(test_data : MsaHmmCellData) -> None:
    # Test setting up multiple models at once
    # Setup
    models = [0, 1]
    hmm_cell, length = make_test_cell(test_data, models)
    hmm_cell.recurrent_init()
    A = hmm_cell.transitioner.make_A()
    init = hmm_cell.transitioner.make_initial_distribution()
    B = hmm_cell.emitter[0].make_B()

    # Compare to reference
    for i in models:
        q = hmm_cell.num_states[i]
        np.testing.assert_almost_equal(
            A[i, :q, :q], test_data.A_ref[i], decimal=5
        )
        np.testing.assert_almost_equal(
            B[i, :q], test_data.B_ref[i], decimal=5
        )
        np.testing.assert_almost_equal(
            init[0, i, :q], test_data.init_ref[i], decimal=5
        )


def test_single_model_forward(test_data : MsaHmmCellData) -> None:
    # Test correctness of a single model
    seq = tf.one_hot([[0, 1, 0, 2]], 3)
    for i in range(2):
        # Create the cell
        hmm_cell, length = make_test_cell(test_data, i)
        hmm_cell.recurrent_init()
        scaled_forward, loglik = hmm_cell.get_initial_state(batch_size=1)

        # Run a half-manual forward pass and test against reference
        init = True
        for j in range(4):
            col = seq[:, j]
            emission_probs = hmm_cell.emission_probs(col)
            log_forward, (scaled_forward, loglik) = hmm_cell(
                emission_probs, (scaled_forward, loglik), init=init
            )
            init = False
            ref_forward = test_data.ref_alpha[i][j]
            ref_scaled_forward = test_data.ref_scaled_alpha[i][j]
            np.testing.assert_almost_equal(
                np.exp(log_forward[..., :-1] + log_forward[..., -1:])[0],
                ref_forward,
                decimal=4,
            )
            np.testing.assert_almost_equal(
                scaled_forward[0],
                ref_scaled_forward,
                decimal=4,
            )
        np.testing.assert_almost_equal(
            np.exp(loglik),
            test_data.ref_lik[i],
            decimal=4,
        )


def test_multi_model_forward(test_data : MsaHmmCellData) -> None:
    # Test correctness of a multi-model forward pass
    models = [0, 1]
    hmm_cell, length = make_test_cell(test_data, models)
    scaled_forward, loglik = hmm_cell.get_initial_state(batch_size=1)
    seq = tf.one_hot([[0, 1, 0, 2]], 3)
    init = True
    for j in range(4):
        col = np.repeat(seq[:, j], len(models), axis=0)
        emission_probs = hmm_cell.emission_probs(col)
        log_forward, (scaled_forward, loglik) = hmm_cell(
            emission_probs, (scaled_forward, loglik), init=init
        )
        init = False
        for i in range(2):
            q = hmm_cell.num_states[i]
            ref_forward = test_data.ref_alpha[i][j]
            ref_scaled_forward = test_data.ref_scaled_alpha[i][j]
            np.testing.assert_almost_equal(
                np.exp(log_forward[..., :-1] + log_forward[..., -1:])[i, :q],
                ref_forward,
                decimal=4,
            )
            np.testing.assert_almost_equal(
                scaled_forward[i, :q],
                ref_scaled_forward,
                decimal=4,
            )
    for i in range(2):
        np.testing.assert_almost_equal(
            np.exp(loglik[i]),
            test_data.ref_lik[i],
            decimal=4,
        )


def test_multi_model_layer(test_data : MsaHmmCellData) -> None:
    # Test the MsaHmmLayer with multiple models
    models = [0, 1]
    hmm_cell, length = make_test_cell(test_data, models)
    hmm_layer = MsaHmmLayer(hmm_cell, use_prior=False)
    seq = tf.one_hot([[0, 1, 0, 2]], 3)
    # we have to expand the seq dimension
    # we have 2 identical inputs for 2 models respectively
    # the batch size is still 1
    seq = np.repeat(seq[np.newaxis], len(models), axis=0)
    loglik = hmm_layer(seq)[0]
    log_forward, _ = hmm_layer.forward_recursion(seq)
    assert hmm_layer.cell.step_counter.numpy() == 4
    log_backward = hmm_layer.backward_recursion(seq)
    state_posterior_log_probs = hmm_layer.state_posterior_log_probs(seq)
    for i in range(2):
        q = hmm_cell.num_states[i]
        np.testing.assert_almost_equal(
            np.exp(loglik[i]), test_data.ref_lik[i]
        )
        np.testing.assert_almost_equal(
            np.exp(log_forward)[i, 0, :, :q],
            test_data.ref_alpha[i],
            decimal=6,
        )
        np.testing.assert_almost_equal(
            np.exp(log_backward)[i, 0, :, :q],
            test_data.ref_beta[i],
            decimal=6,
        )
        np.testing.assert_almost_equal(
            np.exp(state_posterior_log_probs[i, 0, :, :q]),
            test_data.ref_posterior_probs[i],
            decimal=6,
        )


def test_multi_model_tf_model(test_data : MsaHmmCellData) -> None:
    # also test the hmm layer in a compiled model and use model.predict
    # the cell call is traced once in this case which can cause trouble with
    # the initial step
    models = [0, 1]
    hmm_cell, length = make_test_cell(test_data, models)
    hmm_layer = MsaHmmLayer(hmm_cell, use_prior=False)
    sequences = tf.keras.Input(
        shape=(None, None, 3), name="sequences", dtype=tf.float32
    )
    loglik = hmm_layer(sequences)[0]
    hmm_tf_model = tf.keras.Model(inputs=[sequences], outputs=[loglik])
    hmm_tf_model.compile(jit_compile=False)
    seq = tf.one_hot([[0, 1, 0, 2]], 3)
    # we have to expand the seq dimension
    # we have 2 identical inputs for 2 models respectively
    # the batch size is still 1
    seq = np.repeat(seq[np.newaxis], len(models), axis=0)
    loglik = hmm_tf_model.predict(seq)
    for i in range(2):
        np.testing.assert_almost_equal(np.exp(loglik[i]), test_data.ref_lik[i])


def test_duplication(test_data : MsaHmmCellData) -> None:
    models = [0, 1]
    hmm_cell, length = make_test_cell(test_data, models)
    test_shape = [None, None, 3]
    hmm_cell.build(test_shape)

    def test_copied_cell(hmm_cell_copy, model_indices):
        emitter_copy = hmm_cell_copy.emitter
        transitioner_copy = hmm_cell_copy.transitioner
        for i, j in enumerate(model_indices):
            # match emissions
            ref_kernel = hmm_cell.emitter[0].emission_kernel[j].numpy()
            kernel_copy = emitter_copy[0].emission_init[i](ref_kernel.shape)
            np.testing.assert_almost_equal(kernel_copy, ref_kernel)
            # insertions
            ref_ins_kernel = hmm_cell.emitter[0].insertion_kernel[j].numpy()
            ins_kernel_copy = emitter_copy[0].insertion_init[i](ref_ins_kernel.shape)
            np.testing.assert_almost_equal(ins_kernel_copy, ref_ins_kernel)
            # transitioners
            for key, ref_kernel in hmm_cell.transitioner.transition_kernel[j].items():
                ref_kernel = ref_kernel.numpy()
                kernel_copy = transitioner_copy.transition_init[i][key](ref_kernel.shape)
                np.testing.assert_almost_equal(kernel_copy, ref_kernel)

    # clone both models
    test_copied_cell(hmm_cell.duplicate(), [0, 1])

    # clone single model
    for i in range(2):
        test_copied_cell(hmm_cell.duplicate([i]), [i])


def test_parallel_forward(test_data : MsaHmmCellData) -> None:
    models = [0, 1]
    n = len(models)
    hmm_cell, length = make_test_cell(test_data, models)
    hmm_layer = MsaHmmLayer(hmm_cell, use_prior=False, parallel_factor=2)
    seq = tf.one_hot([[0, 1, 0, 2]], 3)
    seq = np.stack([seq] * n)
    hmm_layer.build(seq.shape)
    log_forward, loglik = hmm_layer.forward_recursion(seq)
    assert hmm_layer.cell.step_counter.numpy() == 2
    for i in range(n):
        q = hmm_cell.num_states[i]
        np.testing.assert_allclose(
            np.exp(loglik[i]),
            test_data.ref_lik[i],
            rtol=1e-5, atol=1e-4,
        )
        np.testing.assert_allclose(
            np.exp(log_forward)[i, 0, :, :q],
            test_data.ref_alpha[i],
            rtol=1e-5, atol=1e-4,
        )


def test_parallel_backward(test_data : MsaHmmCellData) -> None:
    models = [0, 1]
    n = len(models)
    hmm_cell, length = make_test_cell(test_data, models)
    hmm_layer = MsaHmmLayer(hmm_cell, use_prior=False, parallel_factor=2)
    seq = tf.one_hot([[0, 1, 0, 2]], 3)
    seq = np.stack([seq] * n)
    hmm_layer.build(seq.shape)
    log_backward = hmm_layer.backward_recursion(seq)
    for i in range(n):
        q = hmm_cell.num_states[i]
        np.testing.assert_allclose(
            np.exp(log_backward)[i, 0, :, :q],
            test_data.ref_beta[i],
            rtol=1e-5, atol=1e-4,
        )


def test_parallel_posterior(test_data : MsaHmmCellData) -> None:
    models = [0, 1]
    n = len(models)
    hmm_cell, length = make_test_cell(test_data, models)
    hmm_layer = MsaHmmLayer(hmm_cell, use_prior=False, parallel_factor=2)
    seq = tf.one_hot([[0, 1, 0, 2]], 3)
    seq = np.stack([seq] * n)
    hmm_layer.build(seq.shape)
    state_posterior_log_probs = hmm_layer.state_posterior_log_probs(seq)
    assert hmm_layer.cell.step_counter.numpy() == 2
    for i in range(2):
        q = hmm_cell.num_states[i]
        np.testing.assert_allclose(
            np.exp(state_posterior_log_probs[i, 0, :, :q]),
            test_data.ref_posterior_probs[i],
            rtol=1e-5, atol=1e-4,
        )


def test_parallel_longer_seq_batch(test_data : MsaHmmCellData) -> None:
    models = [0, 1]
    n = len(models)
    hmm_cell, length = make_test_cell(test_data, models)
    hmm_layer = MsaHmmLayer(
        hmm_cell, use_prior=False, parallel_factor=1
    )
    hmm_layer_parallel = MsaHmmLayer(
        hmm_cell, use_prior=False, parallel_factor=4)
    # set sequence length to 16 so that we have 4 chunks of size 4
    # try one sequence with padding and one without
    seq = tf.one_hot([
        [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0],
        [0, 1, 1, 1, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]
    ], 3)
    seq = np.stack([seq] * n)
    hmm_layer.build(seq.shape)
    hmm_layer_parallel.build(seq.shape)
    # non parallel
    log_forward, loglik = hmm_layer.forward_recursion(seq)
    log_backward = hmm_layer.backward_recursion(seq)
    state_posterior_log_probs = hmm_layer.state_posterior_log_probs(seq)
    # parallel
    log_forward_parallel, loglik_parallel = hmm_layer_parallel.forward_recursion(seq)
    log_backward_parallel = hmm_layer_parallel.backward_recursion(seq)
    state_posterior_log_probs_parallel = hmm_layer_parallel.state_posterior_log_probs(seq)
    assert hmm_layer.cell.step_counter.numpy() == 4
    for i in range(2):
        q = hmm_cell.num_states[i]
        np.testing.assert_allclose(
            np.exp(loglik[i]),
            np.exp(loglik_parallel[i]),
            rtol=1e-5, atol=1e-4,
        )
        np.testing.assert_allclose(
            np.exp(log_forward)[i, 0, :, :q],
            np.exp(log_forward_parallel)[i, 0, :, :q],
            rtol=1e-5, atol=1e-4,
        )
        np.testing.assert_allclose(
            np.exp(log_backward)[i, 0, :, :q],
            np.exp(log_backward_parallel)[i, 0, :, :q],
            rtol=2e-4, atol=2e-4,
        )
        np.testing.assert_allclose(
            np.exp(state_posterior_log_probs[i, 0, :, :q]),
            np.exp(state_posterior_log_probs_parallel[i, 0, :, :q]),
            rtol=2e-3, atol=1e-4,
        )


def test_parallel_posterior_casino() -> None:
    y1 = supervised_training.get_prediction(1).numpy()
    y2 = supervised_training.get_prediction(1).numpy()
    y3 = supervised_training.get_prediction(10).numpy()
    np.testing.assert_almost_equal(y1, y2)  # if this fails, check if the batches are non-random
    np.testing.assert_almost_equal(y2, y3, decimal=4)  # if this fails, parallel != non-parallel


def test_parallel_viterbi(test_data : MsaHmmCellData) -> None:
    models = [0, 1]
    n = len(models)
    hmm_cell, length = make_test_cell(test_data, models)
    seq = tf.one_hot([[0, 1, 0, 2], [1, 0, 0, 0], [1, 1, 1, 1]], 3)
    seq = np.stack([seq] * n)
    viterbi_path_1, gamma_1 = Viterbi.viterbi(
        seq, hmm_cell, parallel_factor=1, return_variables=True
    )
    viterbi_path_2, gamma_2 = Viterbi.viterbi(
        seq, hmm_cell, parallel_factor=2, return_variables=True
    )
    for i in range(2):
        q = hmm_cell.num_states[i]
        np.testing.assert_almost_equal(
            np.exp(gamma_1[i, 0, :, :q]), test_data.ref_gamma[i]
        )
        np.testing.assert_almost_equal(
            np.exp(gamma_2[i, 0, :, 0, :q]), test_data.ref_gamma[i][0::2]
        )
        np.testing.assert_almost_equal(
            np.exp(gamma_2[i, 0, :, 1, :q]), test_data.ref_gamma[i][1::2]
        )
        np.testing.assert_almost_equal(
            viterbi_path_1[i, 0], test_data.ref_viterbi_path[i]
        )
        np.testing.assert_almost_equal(
            viterbi_path_2[i, 0], test_data.ref_viterbi_path[i]
        )


def test_parallel_viterbi_long(test_data):
    models = [0, 1]
    n = len(models)
    hmm_cell, length = make_test_cell(test_data, models)
    np.random.seed(57235782)
    seq = np.random.randint(2, size=(3, 10000))
    seq = tf.one_hot(seq, 3)
    seq = np.stack([seq] * n)
    viterbi_path_1 = Viterbi.viterbi(seq, hmm_cell, parallel_factor=1)
    viterbi_path_100 = Viterbi.viterbi(seq, hmm_cell, parallel_factor=100)
    np.testing.assert_equal(viterbi_path_1.numpy(), viterbi_path_100.numpy())
