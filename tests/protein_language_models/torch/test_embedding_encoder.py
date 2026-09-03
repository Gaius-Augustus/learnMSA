"""The trainable embedding bottleneck used by ``language_model.reduce_online``.

The point of that mode is to start from scratch and adapt to embeddings from any
language model of any width, so the bottleneck must not depend on anything
shipped on disk. :func:`test_initialization_is_random_not_a_shipped_matrix` and
:func:`test_works_for_a_width_no_scoring_model_ships` are what keep that true;
the matching half of the contract -- the pHMM falling back to a generic prior --
lives in ``tests/hmm/torch/test_embedding_prior.py``.
"""

import math

import pytest
import torch

from learnMSA.protein_language_models.torch.embedding_encoder import (
    TorchEmbeddingEncoder, make_embedding_encoder)

#: protT5's embedding width, and the bottleneck production reduces it to.
PROT_T5_DIM = 1024
REDUCED_DIM = 16


def test_initialization_is_random_not_a_shipped_matrix() -> None:
    """Two encoders differ, so nothing fixed is being loaded and copied in."""
    torch.manual_seed(0)
    first = make_embedding_encoder(REDUCED_DIM, input_dim=PROT_T5_DIM)
    torch.manual_seed(1)
    second = make_embedding_encoder(REDUCED_DIM, input_dim=PROT_T5_DIM)
    assert not torch.allclose(first.encoder.weight, second.encoder.weight)
    assert not torch.allclose(first.decoder.weight, second.decoder.weight)


def test_weights_follow_the_glorot_scale() -> None:
    """Both weights are Glorot-uniform over their own fan in/out."""
    torch.manual_seed(0)
    encoder = make_embedding_encoder(REDUCED_DIM, input_dim=PROT_T5_DIM)
    # xavier_uniform_ draws from U(-a, a) with a = sqrt(6/(fan_in+fan_out)),
    # whose standard deviation is a/sqrt(3) = sqrt(2/(fan_in+fan_out)).
    expected = math.sqrt(2.0 / (PROT_T5_DIM + REDUCED_DIM))
    for weight in (encoder.encoder.weight, encoder.decoder.weight):
        assert weight.std().item() == pytest.approx(expected, rel=0.05)
        assert weight.mean().item() == pytest.approx(0.0, abs=0.1 * expected)


def test_works_for_a_width_no_scoring_model_ships() -> None:
    """No pretrained scoring model exists for these dimensions."""
    encoder = make_embedding_encoder(7, input_dim=999)
    assert encoder.reduce(torch.randn(2, 3, 999)).shape == (2, 3, 7)
    assert encoder.reconstruct(torch.randn(2, 3, 7)).shape == (2, 3, 999)


def test_the_bottleneck_has_no_bias() -> None:
    """A bias-free linear map keeps ``reduce`` a pure projection."""
    encoder = make_embedding_encoder(REDUCED_DIM, input_dim=PROT_T5_DIM)
    assert encoder.encoder.bias is None
    assert encoder.decoder.bias is None


def test_encoder_and_decoder_are_trainable() -> None:
    encoder = make_embedding_encoder(REDUCED_DIM, input_dim=PROT_T5_DIM)
    assert all(p.requires_grad for p in encoder.parameters())
    encoder.reconstruction_loss(torch.randn(2, 3, PROT_T5_DIM)).backward()
    assert encoder.encoder.weight.grad is not None
    assert encoder.decoder.weight.grad is not None


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
