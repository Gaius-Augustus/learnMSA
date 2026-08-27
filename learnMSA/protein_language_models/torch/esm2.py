import sys
from typing import Sequence

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import torch

from learnMSA.protein_language_models import esm2
from learnMSA.protein_language_models.common import make_cache_dir
from learnMSA.protein_language_models.torch.language_model import \
    TorchLanguageModel


class TorchESM2LanguageModel(TorchLanguageModel):
    """Embeds proteins with the PyTorch build of ESM-2."""

    def __init__(
        self,
        trainable: bool = False,
        small: bool = False,
        cache_dir: str | None = None,
    ) -> None:
        """
        Args:
            trainable: Whether the ESM-2 weights are trainable.
            small: Use the 650M checkpoint instead of the 3B one.
            cache_dir: Where to cache the downloaded checkpoint.
        """
        super().__init__()
        from transformers import EsmModel, logging

        logging.set_verbosity_error()
        checkpoint = esm2.checkpoint(small)
        self.model = EsmModel.from_pretrained(
            checkpoint,
            cache_dir=make_cache_dir(cache_dir, esm2.CACHE_ID),
            add_pooling_layer=False,
        )
        self.model.requires_grad_(trainable)
        if not trainable:
            self.model.eval()
        self.dim = esm2.DIMS[checkpoint]

    @override
    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        ids, mask, crop = inputs
        esm2_output = self.model(input_ids=ids, attention_mask=mask)
        embeddings = esm2_output.last_hidden_state.to(torch.float32)
        return self.eliminate_start_stop_tokens(embeddings, crop, mask)
