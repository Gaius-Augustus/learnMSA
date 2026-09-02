import sys
from typing import Callable

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import numpy as np
import torch

from learnMSA.protein_language_models.common import ScoringModelConfig
from learnMSA.protein_language_models.scoring_weights import \
    load_scoring_weights


class TorchEmbeddingEncoder(torch.nn.Module):
    """Compresses embeddings through a bottleneck and reconstructs them.
    The encoder maps ``(..., input_dim)`` to the bottleneck
    ``(..., reduced_dim)`` and the decoder maps back.
    """

    def __init__(
        self,
        reduced_dim: int,
        input_dim: int,
        encoder: torch.nn.Module | None = None,
        decoder: torch.nn.Module | None = None,
        reconstruction_loss: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = torch.nn.functional.mse_loss,
        loss_weight: float = 1.0,
        trainable: bool = True,
    ) -> None:
        """
        Args:
            reduced_dim: Width of the bottleneck. Meant to be smaller than
                ``input_dim``.
            input_dim: Width of the incoming embeddings.
            encoder: Maps ``(..., input_dim)`` to ``(..., reduced_dim)``.
                Defaults to a single fully connected layer.
            decoder: Maps ``(..., reduced_dim)`` back to ``(..., input_dim)``.
                Defaults to a single fully connected layer.
            reconstruction_loss: Called as ``loss(reconstructed, embeddings)``
                and expected to return a scalar. Defaults to the mean squared
                error.
            loss_weight: Factor the reconstruction loss is scaled by before it
                is added to the total loss.
            trainable: Whether the encoder and decoder receive gradients.
        """
        super().__init__()
        if reduced_dim >= input_dim:
            raise ValueError(
                f"The bottleneck must be narrower than the input, but got "
                f"reduced_dim={reduced_dim} >= input_dim={input_dim}."
            )
        self.reduced_dim = reduced_dim
        self.input_dim = input_dim
        self.encoder = (
            torch.nn.Linear(input_dim, reduced_dim)
            if encoder is None
            else encoder
        )
        self.decoder = (
            torch.nn.Linear(reduced_dim, input_dim)
            if decoder is None
            else decoder
        )
        self.reconstruction_loss_fn = reconstruction_loss
        self.loss_weight = loss_weight
        self.requires_grad_(trainable)

    def reduce(
        self, embeddings: torch.Tensor, training: bool = False
    ) -> torch.Tensor:
        """Project embeddings onto the bottleneck.

        This is the method downstream code uses; it never needs the decoder.

        Args:
            embeddings: Shape ``(..., input_dim)``.
            training: Accepted for signature parity with
                :class:`~learnMSA.protein_language_models.torch.bilinear_symmetric.TorchSymmetricBilinearReduction`.
                Dropout inside a custom ``encoder`` follows the module's own
                train/eval mode instead.

        Returns:
            Shape ``(..., reduced_dim)``.
        """
        del training
        return self.encoder(embeddings.to(self._dtype))

    def reconstruct(self, reduced_embeddings: torch.Tensor) -> torch.Tensor:
        """Map the bottleneck back to the input space.

        Args:
            reduced_embeddings: Shape ``(..., reduced_dim)``.

        Returns:
            Shape ``(..., input_dim)``.
        """
        return self.decoder(reduced_embeddings)

    def reconstruction_loss(
        self,
        embeddings: torch.Tensor,
        reconstructed: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The weighted reconstruction loss, ready to be added to the total.

        Args:
            embeddings: The original embeddings, shape ``(..., input_dim)``.
            reconstructed: The decoder output for those embeddings. Computed
                here if omitted, which runs encoder and decoder again.
            mask: Optional ``(...)`` of booleans, ``True`` at the positions
                that should count. Padding positions arrive as all-zero
                embeddings and would otherwise drag the error down. Masking
                calls :attr:`reconstruction_loss_fn` with ``reduction="none"``,
                so a custom callable has to accept that keyword.

        Returns:
            Scalar.
        """
        embeddings = embeddings.to(self._dtype)
        if reconstructed is None:
            reconstructed = self.reconstruct(self.reduce(embeddings))
        if mask is None:
            return self.loss_weight * self.reconstruction_loss_fn(
                reconstructed, embeddings
            )
        elementwise = self.reconstruction_loss_fn(
            reconstructed, embeddings, reduction="none"
        )
        mask = mask.to(elementwise.dtype).unsqueeze(-1)
        # Average over the unmasked entries only. The clamp keeps an
        # all-padding batch from dividing by zero.
        total = (elementwise * mask).sum()
        count = mask.sum() * elementwise.shape[-1]
        return self.loss_weight * total / count.clamp(min=1.0)

    @staticmethod
    def padding_mask(embeddings: torch.Tensor) -> torch.Tensor:
        """``True`` wherever an embedding is not an all-zero padding vector."""
        return (embeddings != 0).any(dim=-1)

    @override
    def forward(
        self,
        embeddings: torch.Tensor,
        return_loss: bool = False,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode embeddings, and optionally score the reconstruction.

        Args:
            embeddings: Shape ``(..., input_dim)``.
            return_loss: Also run the decoder and return the weighted
                reconstruction loss. Encoding happens once either way.
            mask: Passed on to :meth:`reconstruction_loss`; only read when
                ``return_loss``.

        Returns:
            The bottleneck of shape ``(..., reduced_dim)``, or that tensor
            paired with the scalar loss when ``return_loss``.
        """
        embeddings = embeddings.to(self._dtype)
        reduced_emb = self.reduce(embeddings)
        if not return_loss:
            return reduced_emb
        reconstructed = self.reconstruct(reduced_emb)
        loss = self.reconstruction_loss(embeddings, reconstructed, mask=mask)
        return reduced_emb, loss

    @property
    def _dtype(self) -> torch.dtype:
        """The dtype the incoming embeddings are cast to."""
        return next(self.parameters()).dtype


def make_embedding_encoder(
    config: ScoringModelConfig,
    input_dim: int,
    loss_weight: float = 1.0,
) -> TorchEmbeddingEncoder:
    """Build a trainable bottleneck seeded with the shipped bilinear matrix.

    The pHMM's embedding emitter carries a multivariate normal prior that was
    fitted in the space the frozen bilinear model projects to (see
    ``_add_emb_emitter`` in :mod:`learnMSA.hmm.layer`). Seeding the encoder
    with that same matrix makes :meth:`TorchEmbeddingEncoder.reduce` reproduce
    the frozen projection exactly at step 0, so the prior is meaningful from
    the start and training only fine-tunes away from it.

    Args:
        config: Identifies the scoring model whose ``R`` seeds the encoder.
        input_dim: Width of the language model's embeddings.
        loss_weight: Weight of the reconstruction loss.

    Returns:
        A trainable :class:`TorchEmbeddingEncoder`.
    """
    layer = TorchEmbeddingEncoder(
        reduced_dim=config.dim,
        input_dim=input_dim,
        encoder=torch.nn.Linear(input_dim, config.dim, bias=False),
        decoder=torch.nn.Linear(config.dim, input_dim, bias=False),
        loss_weight=loss_weight,
        trainable=True,
    )
    R = _load_reduction_matrix(config, input_dim)
    if R is not None:
        with torch.no_grad():
            # torch.nn.Linear computes x @ weight.T, so the weight is R
            # transposed. The decoder starts at the projection's pseudo-inverse,
            # which is the least-squares optimal reconstruction of that encoder.
            layer.encoder.weight.copy_(torch.as_tensor(R.T))
            layer.decoder.weight.copy_(torch.as_tensor(np.linalg.pinv(R).T))
    return layer


def _load_reduction_matrix(
    config: ScoringModelConfig, input_dim: int
) -> np.ndarray | None:
    """The shipped ``R`` of shape ``(input_dim, dim)``, or None if unusable.

    Falls back to the default initialization rather than failing: the ``zeros``
    stand-in ships no scoring model at all, and a mismatching width means the
    weights belong to a different language model.
    """
    if config.lm_name == "zeros":
        return None
    try:
        R = np.asarray(load_scoring_weights(config)["R"], dtype=np.float32)
    except (FileNotFoundError, KeyError):
        return None
    if R.shape != (input_dim, config.dim):
        return None
    return R
