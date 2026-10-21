import numpy as np
import pytest
import torch

from learnMSA.protein_language_models.torch import embed
from learnMSA.protein_language_models.torch.language_model import \
    TorchLanguageModel


class _LanguageModel(TorchLanguageModel):
    """Embeds token ids with a linear map, standing in for a transformer."""

    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Embedding(8, 4)

    def forward(self, inputs):
        ids, mask = inputs
        return self.model(ids) * mask.unsqueeze(-1)


@pytest.fixture
def gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)


def test_short_datasets_run_eagerly(gpu) -> None:
    assert not embed.should_compile(np.full(10000, 100))


def test_long_datasets_are_compiled(gpu) -> None:
    assert embed.should_compile(np.full(10000, 1000))


def test_cpu_runs_eagerly(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert not embed.should_compile(np.full(10000, 1000))


def test_single_sequence_batch_through_compiled_model(monkeypatch) -> None:
    monkeypatch.setattr(embed, "should_compile", lambda seq_lens: True)
    language_model = _LanguageModel()
    fn = embed.make_embedding_fn(language_model, seq_lens=np.array([3]))
    ids = np.array([[1, 2, 3]], dtype=np.int64)
    mask = np.ones_like(ids)
    out = fn((ids, mask))
    assert out.shape == (1, 3, 4)
    with torch.no_grad():
        expected = language_model(
            (torch.as_tensor(ids), torch.as_tensor(mask))
        )
    np.testing.assert_allclose(out, expected.numpy(), rtol=1e-6)
