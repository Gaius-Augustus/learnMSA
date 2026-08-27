import sys
from typing import Sequence

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import torch

from learnMSA.protein_language_models import prot_t5
from learnMSA.protein_language_models.common import make_cache_dir
from learnMSA.protein_language_models.torch.language_model import \
    TorchLanguageModel


class TorchProtT5LanguageModel(TorchLanguageModel):
    """Embeds proteins with the PyTorch build of the ProtT5 encoder."""

    def __init__(
        self,
        trainable: bool = False,
        dtype: torch.dtype = torch.float16,
        cache_dir: str | None = None,
    ) -> None:
        """
        Args:
            trainable: Whether the ProtT5 weights are trainable.
            dtype: Compute dtype. The shipped checkpoint is half precision.
                Half precision is not supported on CPU, so float32 is used
                there instead.
            cache_dir: Where to cache the downloaded checkpoint.
        """
        super().__init__()
        from transformers import T5EncoderModel, logging

        logging.set_verbosity_error()
        if dtype == torch.float16 and not torch.cuda.is_available():
            dtype = torch.float32
        self.compute_dtype = dtype
        self.model = T5EncoderModel.from_pretrained(
            prot_t5.MODEL_CHECKPOINT,
            dtype=dtype,
            cache_dir=make_cache_dir(cache_dir, prot_t5.CACHE_ID),
        )
        self.model.requires_grad_(trainable)
        if not trainable:
            self.model.eval()
        self.dim = prot_t5.DIM

    @override
    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        ids, mask = inputs[0], inputs[1]
        protT5_output = self.model(input_ids=ids, attention_mask=mask)
        # ProtT5 appends a single end-token; drop it, and do not count it in
        # the mask either. There is no start-token, so the shared
        # eliminate_start_stop_tokens is not needed here.
        embeddings = protT5_output.last_hidden_state[:, :-1]
        embeddings = embeddings.to(self.compute_dtype)
        mask = mask[:, 1:]
        max_len = int(mask.sum(-1).amax())
        embeddings = embeddings * mask.unsqueeze(-1).to(embeddings.dtype)
        return embeddings[:, :max_len]
