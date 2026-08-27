import sys
from typing import Sequence

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import torch

from learnMSA.protein_language_models.torch.language_model import \
    TorchLanguageModel


class TorchZerosLanguageModel(TorchLanguageModel):
    """Emits constant zero embeddings of a configurable width."""

    def __init__(
        self, embedding_dim: int, dtype: torch.dtype = torch.float32
    ) -> None:
        """
        Args:
            embedding_dim: Width of the emitted embeddings.
            dtype: Dtype of the emitted embeddings.
        """
        super().__init__()
        self.dim = int(embedding_dim)
        self.compute_dtype = dtype

    @override
    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        _, mask = inputs
        max_len = int(mask.to(torch.int32).sum(-1).amax())
        return torch.zeros(
            (mask.shape[0], max_len, self.dim),
            dtype=self.compute_dtype,
            device=mask.device,
        )
