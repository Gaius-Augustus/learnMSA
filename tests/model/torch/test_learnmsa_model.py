"""Behaviour tests for the PyTorch learnMSA model.

These mirror ``tests/model/tf/test_learnmsa_model.py``. They are not
fixture-based parity tests: the expectations come from ``tests/hmm/ref.py``,
which holds analytically derived likelihoods and Viterbi paths for two small
reference pHMMs. Both backends are checked against the same numbers, which is a
stronger statement than either backend agreeing with the other.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import learnMSA.model.training_util as training_util  # noqa: E402
import tests.hmm.ref as ref  # noqa: E402
from learnMSA.config import (Configuration, TrainingConfig,  # noqa: E402
                             TreeConfig)
from learnMSA.config.hmm import PHMMPriorConfig  # noqa: E402
from learnMSA.model.context import LearnMSAContext  # noqa: E402
from learnMSA.model.torch.model import \
    TorchLearnMSAModel as LearnMSAModel  # noqa: E402
from learnMSA.util.sequence_dataset import SequenceDataset  # noqa: E402

pytestmark = pytest.mark.torch


def _to_onehot(seqs, depth: int) -> np.ndarray:
    """Convert an integer token array to one-hot float vectors along a new last
    axis. Tokens outside [0, depth) (padding/terminal) become all-zero vectors,
    matching the model's per-residue distribution input contract.
    """
    seqs = np.asarray(seqs)
    oh = np.zeros(seqs.shape + (depth,), dtype=np.float32)
    for idx in np.ndindex(seqs.shape):
        t = int(seqs[idx])
        if 0 <= t < depth:
            oh[idx + (t,)] = 1.0
    return oh


@pytest.fixture
def config_binary() -> Configuration:
    """A basic configuration for a pair of pHMM heads over a binary alphabet."""
    hmm_config = ref.config.model_copy(deep=True)
    hmm_config.use_prior_for_emission_init = False
    return Configuration(
        training=TrainingConfig(length_init=[4, 3]),
        tree=TreeConfig(use_anc_probs=False),
        hmm=hmm_config,
        hmm_prior=PHMMPriorConfig(use_amino_acid_prior=False),
    )


@pytest.fixture
def config_amino_acid() -> Configuration:
    return Configuration(training=TrainingConfig(length_init=[20, 10]))


@pytest.fixture
def context_binary(config_binary: Configuration) -> LearnMSAContext:
    return LearnMSAContext(
        config=config_binary,
        num_seq=10,
        sequence_weights=np.arange(10, dtype=float),
    )


@pytest.fixture
def context_amino_acid(config_amino_acid: Configuration) -> LearnMSAContext:
    return LearnMSAContext(
        config=config_amino_acid,
        num_seq=50,
        sequence_weights=np.arange(50, dtype=float),
    )


def test_create_and_call(context_amino_acid: LearnMSAContext) -> None:
    model = LearnMSAModel(context_amino_acid)
    assert model is not None

    batch_size = 4
    seq_length = 15
    seqs = _to_onehot(np.random.randint(
        low=0, high=20, size=(batch_size, seq_length, 1), dtype=np.int32
    ), 20)
    indices = np.array([22, 7, 13, 3])[..., np.newaxis]
    model.build()
    output = model((
        torch.as_tensor(seqs).to(model.device),
        torch.as_tensor(indices).to(model.device),
    ))
    assert output.shape == (batch_size, 2)


def test_predict_loglik_matches_reference(
    context_binary: LearnMSAContext
) -> None:
    """Log-likelihoods must match the analytic reference, and the bucketed
    prediction loop must restore the original sequence order."""
    model = LearnMSAModel(context_binary)
    model.build()
    model.loglik_mode()
    model.compile()

    # "ABA" sequences interleaved with longer "B" runs, so that bucketing
    # actually splits them and a reordering bug would show up.
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
        alphabet="AB",
    )

    batch_cb = training_util.get_adaptive_batch_size(
        context_binary.model_lengths.max(),
        len(context_binary.model_lengths),
        20,
    )
    context_binary.config.training.batch_size = batch_cb

    bucket_boundaries = [4, 20]
    bucket_batch_sizes = [2, 3, 3]
    predictions = model.predict(
        data,
        bucket_boundaries=bucket_boundaries,
        bucket_batch_sizes=bucket_batch_sizes,
    )

    expected_loglik = np.log(ref.likelihoods).reshape((1, 2)).repeat(7, axis=0)
    assert predictions.shape == (10, 2)
    aba = [0, 1, 3, 5, 7, 8, 9]
    np.testing.assert_allclose(
        predictions[aba],
        expected_loglik,
        rtol=1e-3,
        atol=1e-4,
        err_msg="Predicted log-likelihoods do not match reference values",
    )
    assert np.all(predictions[[2, 4, 6]] != expected_loglik[0])


def test_predict_viterbi_matches_reference(
    context_binary: LearnMSAContext
) -> None:
    model = LearnMSAModel(context_binary)
    model.build()
    model.viterbi_mode()
    model.compile()

    data = SequenceDataset(
        sequences=[(str(i), "ABA") for i in range(4)], alphabet="AB"
    )
    viterbi_seqs = model.predict(
        data, bucket_boundaries=[4], bucket_batch_sizes=[2, 2]
    )
    assert viterbi_seqs.shape[0] == 4
    np.testing.assert_equal(
        viterbi_seqs[0, :4, 0],
        ref.viterbi_a,
        err_msg="Viterbi path does not match the reference for model A",
    )


def test_fit_increases_target_emission(
    context_amino_acid: LearnMSAContext
) -> None:
    """Training on sequences of a single residue must raise its emission
    probability -- the end-to-end check that gradients flow."""
    model = LearnMSAModel(context_amino_acid)
    model.build()
    model.compile()

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

    assert model.phmm_layer.profile_emitter is not None
    with torch.no_grad():
        before = model.phmm_layer.profile_emitter.matrix().cpu().numpy()
    prob_A_before = before[:, :10, 0].mean()

    model.fit(data, batch_size=4, epochs=1, steps_per_epoch=10)

    with torch.no_grad():
        after = model.phmm_layer.profile_emitter.matrix().cpu().numpy()
    prob_A_after = after[:, :10, 0].mean()

    assert prob_A_after > prob_A_before, \
        "Emission probability for amino acid A did not increase after training"


def test_fit_returns_history(context_amino_acid: LearnMSAContext) -> None:
    """``_check_training_complete`` reads history.history["loss"], so the
    torch loop has to produce it."""
    model = LearnMSAModel(context_amino_acid)
    model.build()
    model.compile()
    data = SequenceDataset(
        sequences=[(str(i), "ACDEFGHIKLMNPQRSTVWY") for i in range(4)]
    )

    history = model.fit(data, batch_size=2, epochs=2, steps_per_epoch=2)

    assert set(history.history) >= {"loss", "loglik", "prior"}
    assert len(history.history["loss"]) == 2
    assert all(np.isfinite(history.history["loss"]))


def test_evaluate_reports_per_model_metrics(
    context_binary: LearnMSAContext
) -> None:
    model = LearnMSAModel(context_binary)
    model.build()
    model.compile()
    data = SequenceDataset(
        sequences=[(str(i), "ABA") for i in range(6)], alphabet="AB"
    )

    result = model.evaluate(data)

    assert set(result) == {"loss", "loglik", "prior"}
    assert result["loglik"].shape == (2,)
    assert result["prior"].shape == (2,)
    # Every sequence is "ABA", so the loglik must match the reference.
    np.testing.assert_allclose(
        result["loglik"] / np.array([1.0, 1.0]),
        np.log(ref.likelihoods),
        rtol=1e-3,
        atol=1e-3,
    )


def test_estimate_loglik(context_binary: LearnMSAContext) -> None:
    model = LearnMSAModel(context_binary)
    model.build()
    model.compile()
    data = SequenceDataset(
        sequences=[(str(i), "ABA") for i in range(6)], alphabet="AB"
    )

    loglik = model.estimate_loglik(data)

    assert loglik.shape == (2,)
    np.testing.assert_allclose(
        loglik, np.log(ref.likelihoods), rtol=1e-3, atol=1e-3
    )


def test_predict_posterior_reduce(context_binary: LearnMSAContext) -> None:
    model = LearnMSAModel(context_binary)
    model.build()
    model.posterior_mode()
    model.compile()
    data = SequenceDataset(
        sequences=[(str(i), "ABA") for i in range(6)], alphabet="AB"
    )

    reduced = model.predict(data, reduce=True)

    num_states = max(model.phmm_layer.states)
    assert reduced.shape == (2, num_states)
    assert np.all(reduced >= 0)


def test_null_model_log_probs(context_amino_acid: LearnMSAContext) -> None:
    model = LearnMSAModel(context_amino_acid)
    model.build()
    model.compile()
    data = SequenceDataset(sequences=[
        ("1", "ACDEFGHIKLMNPQRSTVWY"),
        ("2", "ACDEFGHIKL"),
        ("3", "ACDEFGHIKLMNPQRSTVWYACDEFGHIKL"),
    ])

    log_probs = model.compute_null_model_log_probs(data)

    assert log_probs.shape == (3,)
    assert np.all(log_probs < 0)
    # A longer sequence is less likely under the null model.
    assert log_probs[2] < log_probs[0] < log_probs[1]


def test_decode_modes_are_refused(context_binary: LearnMSAContext) -> None:
    """The fused GPU decoder is TensorFlow-only; the refusal must be explicit
    rather than a wrong result."""
    model = LearnMSAModel(context_binary)
    model.build()
    model.viterbi_decode_mode()
    model.compile()
    data = SequenceDataset(
        sequences=[(str(i), "ABA") for i in range(4)], alphabet="AB"
    )

    with pytest.raises(NotImplementedError, match="TensorFlow"):
        model.predict(data)
