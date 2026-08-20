import pytest

from learnMSA.model import training_util
from learnMSA.model.training_util import (
    IMPL_FACTORS,
    MODE_FALLBACK,
    MAX_BATCH_SIZE,
    MAX_TOKENS_PER_BATCH,
    get_adaptive_batch_size,
    get_impl_factors,
    tokens_per_batch_to_batch_size,
)

# A shape whose batch size is memory-bound rather than clamped by
# MAX_BATCH_SIZE or MAX_TOKENS_PER_BATCH, so the scaling below is observable.
# Both this shape and the halved variants stay under the caps.
MODEL_LEN = 220
SEQ_LEN = 1_000
NUM_MODEL = 4
FACTOR = 26.0


@pytest.fixture(autouse=True)
def fixed_memory(monkeypatch):
    """Pin the device memory so the tests do not depend on the host GPU."""
    monkeypatch.setattr(
        training_util, "get_avail_memory_bytes", lambda: 20 * 1024 ** 3
    )


def batch_size(model_len=MODEL_LEN, seq_len=SEQ_LEN, **kwargs):
    kwargs.setdefault("impl_factor", FACTOR)
    return get_adaptive_batch_size(model_len, NUM_MODEL, seq_len, **kwargs)


def test_batch_size_is_linear_in_model_len():
    """The property this formula exists to get right.

    Peak memory grows like model_len, not model_len**2, so halving the model
    length must double the batch size.
    """
    assert batch_size(model_len=MODEL_LEN // 2) == pytest.approx(
        2 * batch_size(), rel=0.02
    )


def test_batch_size_is_linear_in_seq_len():
    assert batch_size(seq_len=SEQ_LEN // 2) == pytest.approx(
        2 * batch_size(), rel=0.02
    )


def test_batch_size_is_inverse_in_impl_factor():
    assert batch_size(impl_factor=2 * FACTOR) == pytest.approx(
        batch_size() / 2, rel=0.02
    )


def test_batch_size_shrinks_with_more_models():
    many = get_adaptive_batch_size(
        MODEL_LEN, 16, SEQ_LEN, impl_factor=FACTOR
    )
    assert many < batch_size()


def test_batch_size_respects_max_batch_size():
    """A tiny workload is clamped, not unbounded."""
    assert get_adaptive_batch_size(3, 1, 10, impl_factor=FACTOR) \
        <= MAX_BATCH_SIZE


def test_batch_size_respects_max_tokens_per_batch():
    seq_len = 5_000
    assert get_adaptive_batch_size(3, 1, seq_len, impl_factor=FACTOR) \
        <= MAX_TOKENS_PER_BATCH // seq_len


def test_batch_size_stays_positive_for_absurd_workloads():
    """Never return 0: the caller would divide by it."""
    assert get_adaptive_batch_size(10 ** 6, 64, 10 ** 6) >= 1


def test_tokens_per_batch_is_a_plain_division():
    assert tokens_per_batch_to_batch_size(20_000, 575) == 34


def test_tokens_per_batch_is_clamped():
    assert tokens_per_batch_to_batch_size(10 ** 9, 10) == MAX_BATCH_SIZE
    assert tokens_per_batch_to_batch_size(1, 10 ** 6) == 1


def test_tokens_per_batch_never_exceeds_the_token_budget():
    seq_len = 137
    tokens = 50_000
    assert tokens_per_batch_to_batch_size(tokens, seq_len) * seq_len <= tokens


#: The IMPL_FACTORS key prefix of every input configuration, "" being the
#: amino-acid-only one.
CONFIGURATIONS = (
    "", "language_model", "structure", "language_model_and_structure",
)


#: Every workload a factor may be keyed by. "inference" is not a workload but
#: the aggregate fallback for inference modes without a key of their own.
WORKLOADS = ("train", "inference", "viterbi", "posterior", "loglik")


def _key(prefix, workload):
    return f"{prefix}_{workload}" if prefix else workload


def _mode_keys(factors, prefix):
    """The per-mode keys shipped for one input configuration, if any."""
    return [
        factors[_key(prefix, mode)]
        for mode in ("viterbi", "posterior", "loglik")
        if _key(prefix, mode) in factors
    ]


@pytest.mark.parametrize("backend", sorted(IMPL_FACTORS))
def test_impl_factors_are_complete_and_positive(backend):
    """Every configuration needs train and the inference fallback at minimum.

    Per-mode keys are optional: a mode without one falls back to "inference",
    so the table may ship only the modes that were worth separating.
    """
    factors = get_impl_factors(backend)
    required = {
        _key(prefix, workload)
        for prefix in CONFIGURATIONS
        for workload in ("train", "inference")
    }
    permitted = {
        _key(prefix, workload)
        for prefix in CONFIGURATIONS
        for workload in WORKLOADS
    }
    assert required <= set(factors) <= permitted
    assert all(v > 0 for v in factors.values())


@pytest.mark.parametrize("backend", sorted(IMPL_FACTORS))
@pytest.mark.parametrize("prefix", CONFIGURATIONS)
def test_inference_never_costs_more_than_training(backend, prefix):
    """Inference holds no gradients, so it must not ask for *more* memory.

    Equality is permitted, and the pytorch posterior actually reaches it. Since
    the log-likelihood forward became a fold that keeps only its final carry,
    training no longer materialises the state history, while a posterior runs
    both the forward and the backward sweep -- so the two are within 0.1% of
    each other on the amino acid track and round to the same integer.
    """
    factors = get_impl_factors(backend)
    train = factors[_key(prefix, "train")]
    for mode in ("inference", "viterbi", "posterior", "loglik"):
        if _key(prefix, mode) in factors:
            assert factors[_key(prefix, mode)] <= train


@pytest.mark.parametrize("backend", sorted(IMPL_FACTORS))
@pytest.mark.parametrize("prefix", CONFIGURATIONS)
def test_inference_fallback_covers_every_shipped_mode(backend, prefix):
    """"inference" is what an uncalibrated mode falls back to.

    It has to be at least as expensive as every mode that was calibrated, or
    the fallback would hand out a batch size that does not fit.
    """
    factors = get_impl_factors(backend)
    fallback = factors[_key(prefix, "inference")]
    for measured in _mode_keys(factors, prefix):
        assert measured <= fallback


def test_mea_borrows_the_posterior_factor():
    """MEA computes posteriors and then decodes them, so it is not measured."""
    assert MODE_FALLBACK == {"mea": "posterior"}


def test_unknown_backend_falls_back_to_tensorflow():
    assert get_impl_factors("nonexistent") is IMPL_FACTORS["tensorflow"]


def test_calibrated_factor_predicts_the_measured_peak():
    """Guards the units of IMPL_FACTORS.

    A factor counts model_len x seq_len values held per model and per sequence,
    so multiplying back out must reproduce a real measurement. This is the
    pytorch/train probe at L=1000, S=3000, B=32 from
    util/impl_factor_calibration_rtx3090_torch.json, which peaked at 23.20 GiB
    and is the shape the shipped train factor was derived from.
    """
    factor = IMPL_FACTORS["pytorch"]["train"]
    # dtype_size * num_model_factor * model_len * seq_len * batch_size
    predicted = factor * 4 * 4 * 1000 * 3000 * 32
    assert predicted / 1024 ** 3 == pytest.approx(23.20, rel=0.06)


# ---------------------------------------------------------------------------
# The lookup that turns a pHMM call mode into one of the keys above.
# ---------------------------------------------------------------------------

#: A table with per-mode keys, so the fallback chain is observable regardless
#: of which modes the shipped table happens to separate.
FAKE_FACTORS = {
    "pytorch": {
        "train": 30.0,
        "inference": 14.0,
        "viterbi": 11.0,
        "posterior": 14.0,
        # no "loglik" key: it must fall back to "inference"
        "structure_train": 33.0,
        "structure_inference": 15.0,
        "language_model_train": 100.0,
        "language_model_inference": 40.0,
        "language_model_and_structure_train": 110.0,
        "language_model_and_structure_inference": 45.0,
    },
}
# get_impl_factors falls back to the TensorFlow column for unknown backends,
# so the table must always carry one.
FAKE_FACTORS["tensorflow"] = FAKE_FACTORS["pytorch"]


def _context(**config_kwargs):
    """A context carrying only the config that _get_impl_factor reads."""
    from learnMSA import Configuration
    from learnMSA.model.context import LearnMSAContext

    context = LearnMSAContext.__new__(LearnMSAContext)
    context.config = Configuration(**config_kwargs)
    return context


@pytest.fixture
def fake_factors(monkeypatch):
    monkeypatch.setattr(training_util, "IMPL_FACTORS", FAKE_FACTORS)
    monkeypatch.setattr("learnMSA.backend.get_backend", lambda: "pytorch")


def test_impl_factor_uses_the_mode_specific_key(fake_factors):
    context = _context()
    assert context._get_impl_factor("viterbi") == 11.0
    assert context._get_impl_factor("posterior") == 14.0
    assert context._get_impl_factor("train") == 30.0
    assert context._get_impl_factor() == 30.0


def test_impl_factor_falls_back_for_an_uncalibrated_mode(fake_factors):
    """loglik has no key here, so it takes the conservative aggregate."""
    context = _context()
    assert context._get_impl_factor("loglik") == 14.0


def test_impl_factor_maps_mea_onto_the_posterior_key(fake_factors):
    context = _context()
    assert context._get_impl_factor("mea") == 14.0
    assert context._get_impl_factor("mea") == \
        context._get_impl_factor("posterior")


def test_impl_factor_falls_back_twice_when_the_borrowed_key_is_absent(
    monkeypatch,
):
    """mea -> posterior -> inference, when posterior was never calibrated."""
    table = {"pytorch": dict(FAKE_FACTORS["pytorch"])}
    del table["pytorch"]["posterior"]
    table["tensorflow"] = table["pytorch"]
    monkeypatch.setattr(training_util, "IMPL_FACTORS", table)
    monkeypatch.setattr("learnMSA.backend.get_backend", lambda: "pytorch")
    assert _context()._get_impl_factor("mea") == 14.0


def test_impl_factor_still_selects_by_input_track(fake_factors):
    """The track prefix and the mode are independent choices."""
    struct = _context(structure={"use_structure": True})
    assert struct._get_impl_factor("viterbi") == 15.0  # no structure_viterbi
    assert struct._get_impl_factor("train") == 33.0

    lm = _context(language_model={"use_language_model": True})
    assert lm._get_impl_factor("posterior") == 40.0

    both = _context(
        structure={"use_structure": True},
        language_model={"use_language_model": True},
    )
    assert both._get_impl_factor("loglik") == 45.0
