from typing import Callable

import numpy as np
import torch

from learnMSA.protein_language_models.common import LanguageModel

#: Numpy dtypes the encoders emit, mapped to their torch equivalents.
_DTYPES: dict[str, torch.dtype] = {
    "int32": torch.int32,
    "int64": torch.int64,
    "float16": torch.float16,
    "float32": torch.float32,
}


def make_embedding_fn(
    language_model: LanguageModel,
    reduction_layer: torch.nn.Module | None = None,
) -> Callable[[tuple[np.ndarray, ...]], np.ndarray]:
    """Build the embed-and-reduce call.

    Runs eagerly: every batch has a different shape and each sequence passes
    the language model only once, so compiling would cost more than it saves.

    Args:
        language_model: The wrapped language model.
        reduction_layer: Projects the embeddings onto the scoring model's
            reduced dimension. ``None`` leaves them unreduced.

    Returns:
        A callable mapping the encoder's output to a numpy array of shape
        ``(batch, max_len, dim)``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    language_model.to(device)
    if reduction_layer is not None:
        reduction_layer.to(device)

    def embed(lm_inputs: tuple[np.ndarray, ...]) -> np.ndarray:
        tensors = tuple(_to_tensor(x, device) for x in lm_inputs)
        with torch.inference_mode():
            emb = language_model(tensors)
            if reduction_layer is not None:
                emb = reduction_layer.reduce(emb, training=False)
            emb = emb.to(torch.float32).cpu().numpy()
        if device.type == "cuda":
            # The adaptive batch size is derived from the free memory that
            # nvidia-smi reports, which counts memory held by the caching
            # allocator as used.
            torch.cuda.empty_cache()
        return emb

    return embed


def _to_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    """Move one encoder output onto the compute device."""
    array = np.asarray(array)
    dtype = _DTYPES.get(array.dtype.name)
    return torch.as_tensor(array, dtype=dtype, device=device)
