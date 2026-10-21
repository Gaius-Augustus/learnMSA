import sys
from typing import Callable

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import torch

from learnMSA.protein_language_models.common import ScoringModelConfig
from learnMSA.protein_language_models.scoring_weights import \
    load_scoring_weights

#: Additive penalty that pushes padding positions out of the attention
#: distribution. Matches the TensorFlow implementation.
MASK_PENALTY = 1e9


class TorchSymmetricBilinearReduction(torch.nn.Module):
    """Scores pairs of embeddings for homology through a shared projection.

    Both embeddings are projected with the same matrix ``R``, which makes the
    resulting score matrix symmetric. In learnMSA's embedding pipeline only
    :meth:`reduce` is used; :meth:`forward` exists for parity with the
    TensorFlow layer and for the pretraining tooling.
    """

    def __init__(
        self,
        reduced_dim: int,
        input_dim: int,
        activation: Callable[[torch.Tensor], torch.Tensor] = torch.sigmoid,
        scaled: bool = True,
        trainable: bool = False,
    ) -> None:
        """
        Args:
            reduced_dim: Width the embeddings are projected to.
            input_dim: Width of the incoming language model embeddings.
            activation: Applied to the scores when ``activate_output``.
            scaled: Rescale so that scores have roughly unit variance.
            trainable: Whether ``R`` and ``b`` receive gradients.
        """
        super().__init__()
        self.reduced_dim = reduced_dim
        self.input_dim = input_dim
        self.activation = activation
        self.scaled = scaled
        self.R = torch.nn.Parameter(
            torch.empty(input_dim, reduced_dim), requires_grad=trainable
        )
        self.b = torch.nn.Parameter(
            torch.empty(1), requires_grad=trainable
        )

    def reduce(
        self, embeddings: torch.Tensor, training: bool = False
    ) -> torch.Tensor:
        """Project embeddings onto the reduced space.

        Args:
            embeddings: Shape ``(..., k, input_dim)``.
            training: Accepted for signature parity with the TensorFlow layer,
                whose ``reduce`` applies dropout. This layer has none, since it
                is only ever used frozen, for inference.

        Returns:
            Shape ``(..., k, reduced_dim)``.
        """
        del training
        reduced_emb = torch.matmul(embeddings.to(self.R.dtype), self.R)
        if self.scaled:
            # scores should have roughly variance 1
            reduced_emb = reduced_emb / self.R.shape[0] ** 0.5
        return reduced_emb

    @override
    def forward(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor,
        a_is_reduced: bool = False,
        b_is_reduced: bool = False,
        activate_output: bool = True,
        use_bias: bool = True,
    ) -> torch.Tensor:
        """Compute the probability that two embeddings are homologous.

        Args:
            embeddings_a: Shape ``(..., k1, input_dim)``.
            embeddings_b: Shape ``(..., k2, input_dim)``.
            a_is_reduced: ``embeddings_a`` is already of reduced dimension.
            b_is_reduced: ``embeddings_b`` is already of reduced dimension.
            activate_output: Apply :attr:`activation` to the scores.
            use_bias: Add the learned bias to the scores.

        Returns:
            Scores for all pairs, of shape ``(..., k1, k2)``.
        """
        reduced_a = (
            embeddings_a if a_is_reduced else self.reduce(embeddings_a)
        )
        reduced_b = (
            embeddings_b if b_is_reduced else self.reduce(embeddings_b)
        )
        scores = torch.matmul(reduced_a, reduced_b.transpose(-1, -2))
        if self.scaled:
            # a reduced neuron also has roughly variance 1, since it was scaled
            scores = scores / self.reduced_dim ** 0.5
        if use_bias:
            scores = scores + self.b
        # make non-padding positions not contribute to the distribution
        mask = (embeddings_b == 0).all(dim=-1).unsqueeze(-2)
        scores = scores - MASK_PENALTY * mask.to(scores.dtype)
        if activate_output:
            return self.activation(scores)
        return scores


def make_reduction_layer(
    config: ScoringModelConfig,
) -> TorchSymmetricBilinearReduction:
    """Build a frozen reduction layer with the shipped parameters loaded.

    This is what :mod:`learnMSA.protein_language_models.compute_embeddings`
    calls.

    Args:
        config: Identifies the scoring model.

    Returns:
        A non-trainable :class:`TorchSymmetricBilinearReduction` in eval mode.
    """
    weights = load_scoring_weights(config)
    layer = TorchSymmetricBilinearReduction(
        reduced_dim=config.dim,
        input_dim=int(weights["R"].shape[0]),
        activation=_get_activation(config.activation),
        scaled=config.scaled,
        trainable=False,
    )
    with torch.no_grad():
        layer.R.copy_(torch.as_tensor(weights["R"], dtype=layer.R.dtype))
        layer.b.copy_(torch.as_tensor(weights["b"], dtype=layer.b.dtype))
    layer.eval()
    layer.requires_grad_(False)  # don't forget to freeze the scoring model!
    return layer


def _get_activation(
    activation: str | Callable[[torch.Tensor], torch.Tensor],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Map an activation name to a callable, passing callables through."""
    if activation == "softmax":
        return lambda x: torch.softmax(x, dim=-1)
    if activation == "sigmoid":
        return torch.sigmoid
    return activation
