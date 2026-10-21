import sys
from typing import Any, Sequence

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import torch

from learnMSA.protein_language_models.common import LanguageModel


class TorchLanguageModel(torch.nn.Module, LanguageModel[torch.Tensor]):
    """Base class for the PyTorch language model wrappers."""

    @override
    def eliminate_start_stop_tokens(
        self,
        embeddings: torch.Tensor,
        crop: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = mask.to(embeddings.dtype)
        zeros_1 = torch.zeros_like(mask[:, :1])
        mask_crop_1 = torch.cat([mask[:, 1:], zeros_1], 1)
        zeros_2 = torch.zeros_like(mask[:, :2])
        mask_crop_2 = torch.cat([mask[:, 2:], zeros_2], 1)
        crop = crop.to(embeddings.dtype)
        crop_start, crop_end = crop[:, :1], crop[:, 1:]
        # both tokens
        mask_no_start_stop = mask_crop_2 * (1 - crop_start) * (1 - crop_end)
        # only start token
        mask_no_start_stop = mask_no_start_stop + (
            mask_crop_1 * crop_start * (1 - crop_end)
        )
        # only end token
        mask_no_start_stop = mask_no_start_stop + (
            mask_crop_1 * (1 - crop_start) * crop_end
        )
        # no start- or end-token
        mask_no_start_stop = mask_no_start_stop + mask * crop_start * crop_end
        # shift sequences with a start token by 1
        embeddings_no_start = torch.cat(
            [embeddings[:, 1:], torch.zeros_like(embeddings[:, :1])], 1
        )
        start = crop_start.unsqueeze(-1)
        embeddings_no_start_stop = (
            embeddings_no_start * start + embeddings * (1 - start)
        )
        embeddings_no_start_stop = (
            embeddings_no_start_stop * mask_no_start_stop.unsqueeze(-1)
        )
        # crop all padding-only columns
        max_len = int(mask_no_start_stop.to(torch.int32).sum(-1).amax())
        return embeddings_no_start_stop[:, :max_len]

    @override
    def call(self, inputs: Sequence[Any]) -> torch.Tensor:
        """Alias of ``forward``.

        The neutral base and its callers use ``call``; torch modules use
        ``forward``. Subclasses implement ``forward``, so that hooks and
        ``torch.compile`` see a normal module.
        """
        return self(inputs)
