"""The trainable embedding bottleneck used by ``language_model.reduce_online``.

The point of seeding the encoder with the shipped bilinear matrix is that the
pHMM's embedding prior was fitted in exactly the space that matrix projects to.
:func:`test_reduce_matches_the_frozen_projection` is what keeps that true; if it
breaks, the prior silently stops describing the bottleneck's output.
"""

import numpy as np
import pytest
import torch

from learnMSA.protein_language_models.common import ScoringModelConfig
from learnMSA.protein_language_models.torch.bilinear_symmetric import \
    make_reduction_layer
from learnMSA.protein_language_models.torch.embedding_encoder import (
    TorchEmbeddingEncoder, make_embedding_encoder)

#: The shipped protT5 scoring model, the one production uses.
SCORING_CONFIG = ScoringModelConfig(
    lm_name="protT5", dim=16, activation="sigmoid", scaled=False
)

#: protT5's embedding width.
PROT_T5_DIM = 1024


def test_reduce_matches_the_frozen_projection() -> None:
    """At init the bottleneck reproduces the frozen bilinear reduction."""
    encoder = make_embedding_encoder(SCORING_CONFIG, input_dim=PROT_T5_DIM)
    frozen = make_reduction_layer(SCORING_CONFIG)
    embeddings = torch.randn(2, 7, PROT_T5_DIM)
    torch.testing.assert_close(
        encoder.reduce(embeddings), frozen.reduce(embeddings)
    )


def test_encoder_and_decoder_are_trainable() -> None:
    encoder = make_embedding_encoder(SCORING_CONFIG, input_dim=PROT_T5_DIM)
    assert all(p.requires_grad for p in encoder.parameters())
    encoder.reconstruction_loss(torch.randn(2, 3, PROT_T5_DIM)).backward()
    assert encoder.encoder.weight.grad is not None
    assert encoder.decoder.weight.grad is not None


def test_decoder_starts_at_the_pseudo_inverse() -> None:
    """The seeded decoder is the least-squares inverse of the encoder.

    Vectors that lie in the projection's row space come back unchanged, which
    is the best any linear decoder can do at initialization.
    """
    encoder = make_embedding_encoder(SCORING_CONFIG, input_dim=PROT_T5_DIM)
    R = encoder.encoder.weight.detach().T  # (D, R)
    # Anything of the form R @ v is in the row space by construction.
    in_row_space = (R @ torch.randn(SCORING_CONFIG.dim, 4)).T  # (4, D)
    round_tripped = encoder.reconstruct(encoder.reduce(in_row_space))
    torch.testing.assert_close(round_tripped, in_row_space, rtol=1e-3, atol=1e-3)


def test_missing_scoring_weights_fall_back_to_default_init() -> None:
    """``zeros`` ships no scoring model, so the seeding is simply skipped."""
    config = ScoringModelConfig(lm_name="zeros", dim=4, activation="sigmoid")
    encoder = make_embedding_encoder(config, input_dim=16)
    assert encoder.reduce(torch.randn(2, 16)).shape == (2, 4)


def test_bottleneck_must_be_narrower_than_the_input() -> None:
    with pytest.raises(ValueError, match="narrower"):
        TorchEmbeddingEncoder(reduced_dim=16, input_dim=16)


def test_loss_weight_scales_the_loss() -> None:
    torch.manual_seed(0)
    embeddings = torch.randn(2, 3, 32)
    unweighted = TorchEmbeddingEncoder(8, 32, loss_weight=1.0)
    weighted = TorchEmbeddingEncoder(8, 32, loss_weight=0.25)
    weighted.load_state_dict(unweighted.state_dict())
    torch.testing.assert_close(
        weighted.reconstruction_loss(embeddings),
        0.25 * unweighted.reconstruction_loss(embeddings),
    )


def test_mask_excludes_padding_rows() -> None:
    """Masked loss equals the loss over the real rows alone.

    Padding arrives as all-zero embeddings. Left in, they are rows the decoder
    can fit almost exactly, which deflates the reported error for free.
    """
    encoder = TorchEmbeddingEncoder(8, 32)
    embeddings = torch.randn(2, 5, 32)
    embeddings[:, 3:] = 0.0
    mask = TorchEmbeddingEncoder.padding_mask(embeddings)
    assert mask.sum() == 2 * 3

    masked = encoder.reconstruction_loss(embeddings, mask=mask)
    torch.testing.assert_close(
        masked, encoder.reconstruction_loss(embeddings[mask])
    )
    assert not torch.isclose(masked, encoder.reconstruction_loss(embeddings))


def test_all_padding_batch_does_not_divide_by_zero() -> None:
    encoder = TorchEmbeddingEncoder(8, 32)
    embeddings = torch.zeros(2, 3, 32)
    mask = TorchEmbeddingEncoder.padding_mask(embeddings)
    assert torch.isfinite(encoder.reconstruction_loss(embeddings, mask=mask))


def test_forward_can_return_the_loss_alongside_the_bottleneck() -> None:
    encoder = TorchEmbeddingEncoder(8, 32, loss_weight=2.0)
    embeddings = torch.randn(2, 3, 32)
    reduced, loss = encoder(embeddings, return_loss=True)
    assert reduced.shape == (2, 3, 8)
    torch.testing.assert_close(loss, encoder.reconstruction_loss(embeddings))


def test_forward_takes_a_mask_for_the_loss() -> None:
    """The masked loss is what the model wants; forward must pass it through."""
    encoder = TorchEmbeddingEncoder(8, 32)
    embeddings = torch.randn(2, 5, 32)
    embeddings[:, 3:] = 0.0
    mask = TorchEmbeddingEncoder.padding_mask(embeddings)

    _reduced, loss = encoder(embeddings, return_loss=True, mask=mask)
    torch.testing.assert_close(
        loss, encoder.reconstruction_loss(embeddings, mask=mask)
    )
    assert not torch.isclose(loss, encoder.reconstruction_loss(embeddings))


def test_custom_encoder_and_decoder_are_used() -> None:
    encoder = TorchEmbeddingEncoder(
        4, 8,
        encoder=torch.nn.Sequential(
            torch.nn.Linear(8, 6), torch.nn.ReLU(), torch.nn.Linear(6, 4)
        ),
        decoder=torch.nn.Sequential(
            torch.nn.Linear(4, 6), torch.nn.ReLU(), torch.nn.Linear(6, 8)
        ),
        reconstruction_loss=torch.nn.functional.l1_loss,
        trainable=False,
    )
    assert encoder(torch.randn(2, 8)).shape == (2, 4)
    assert not any(p.requires_grad for p in encoder.parameters())
