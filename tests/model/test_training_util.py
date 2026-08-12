import pytest

from learnMSA.model import training_util
from learnMSA.model.training_util import (
    IMPL_FACTORS,
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


@pytest.mark.parametrize("backend", sorted(IMPL_FACTORS))
def test_impl_factors_are_complete_and_positive(backend):
    factors = get_impl_factors(backend)
    assert set(factors) == {
        "train", "inference",
        "language_model_train", "language_model_inference",
        "structure_train", "structure_inference",
    }
    assert all(v > 0 for v in factors.values())


@pytest.mark.parametrize("backend", sorted(IMPL_FACTORS))
def test_inference_is_cheaper_than_training(backend):
    """Inference holds no gradients, so it must not ask for more memory."""
    factors = get_impl_factors(backend)
    assert factors["inference"] < factors["train"]


def test_unknown_backend_falls_back_to_tensorflow():
    assert get_impl_factors("nonexistent") is IMPL_FACTORS["tensorflow"]


def test_calibrated_factor_predicts_the_measured_peak():
    """Guards the units of IMPL_FACTORS.

    A factor counts model_len x seq_len values held per model and per sequence,
    so multiplying back out must reproduce a real measurement. This is the
    pytorch/train probe at L=700, S=2100, B=32, which peaked at 18.18 GiB.
    """
    factor = IMPL_FACTORS["pytorch"]["train"]
    # dtype_size * num_model_factor * model_len * seq_len * batch_size
    predicted = factor * 4 * 4 * 700 * 2100 * 32
    assert predicted / 1024 ** 3 == pytest.approx(18.2, rel=0.05)
