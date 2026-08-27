"""Checks the PyTorch embedding pipeline against recorded TensorFlow outputs.

The reference arrays in ``tests/fixtures/plm_parity.npz`` were produced by
``tests/fixtures/generate_plm_parity_fixtures.py`` under the TensorFlow
environment; see that module for why the comparison is split in two.

These two pieces -- start/end-token elimination and the bilinear reduction --
are the only parts of the embedding pipeline that exist twice, so they are the
only parts that can drift. Everything else is either shared neutral code or a
HuggingFace checkpoint whose weights are the same in both frameworks.
"""

import numpy as np
import pytest
import torch

from learnMSA.protein_language_models.torch.bilinear_symmetric import \
    make_reduction_layer
from learnMSA.protein_language_models.torch.language_model import \
    TorchLanguageModel
from tests.fixtures.generate_plm_parity_fixtures import (FIXTURE,
                                                         scoring_model_config)

#: Relative tolerance for the reduced embeddings. The reduction is one matmul
#: over 1024 float32 terms, which the two backends are free to associate
#: differently.
RTOL = 1e-5

#: Absolute floor, so entries that are zero on one backend and denormal on the
#: other do not fail the relative check.
ATOL = 1e-5


class _LanguageModel(TorchLanguageModel):
    """Only the inherited token elimination is under test."""

    def forward(self, inputs):
        raise NotImplementedError


@pytest.fixture(scope="module")
def fixture_data():
    assert FIXTURE.exists(), (
        f"{FIXTURE.name} is committed to the repository but is missing here. "
        "Regenerate it with 'python tests/fixtures/"
        "generate_plm_parity_fixtures.py' under the TensorFlow environment."
    )
    with np.load(FIXTURE) as archive:
        yield {key: archive[key] for key in archive.files}


def test_eliminate_start_stop_tokens_matches_tensorflow(fixture_data) -> None:
    """Pure masking and shifting, so this must agree bit for bit.

    The recorded batch covers all four combinations of cropped-at-the-start and
    cropped-at-the-end, each of which takes its own branch.
    """
    eliminated = _LanguageModel().eliminate_start_stop_tokens(
        torch.from_numpy(fixture_data["emb"]),
        torch.from_numpy(fixture_data["crop"]),
        torch.from_numpy(fixture_data["mask"]),
    ).numpy()

    assert eliminated.shape == fixture_data["eliminated"].shape
    np.testing.assert_array_equal(eliminated, fixture_data["eliminated"])


def test_reduction_matches_tensorflow(fixture_data) -> None:
    """The shipped scoring model must reduce to the same embeddings."""
    layer = make_reduction_layer(scoring_model_config())
    reduced = layer.reduce(torch.from_numpy(fixture_data["emb"])).numpy()

    assert reduced.shape == fixture_data["reduced"].shape
    np.testing.assert_allclose(
        reduced, fixture_data["reduced"], rtol=RTOL, atol=ATOL
    )


def test_reduction_layer_is_frozen() -> None:
    """The scoring model must never pick up gradients during alignment."""
    layer = make_reduction_layer(scoring_model_config())
    assert not layer.training
    assert not any(p.requires_grad for p in layer.parameters())
