from typing import Callable

import numpy as np
import torch

from learnMSA.protein_language_models.common import LanguageModel


def _enable_scalar_capture() -> None:
    """Let dynamo trace the data-dependent crop instead of breaking on it.

    Both ``eliminate_start_stop_tokens`` and the ProtT5 wrapper end by cropping
    all-padding columns off the batch, which needs the padded length as a
    Python int. Without this, ``Tensor.item()`` splits the traced region in two
    right at the end of the language model's forward pass. The resulting shape
    is unbacked, which is exactly what ``dynamic=True`` is already set up for.
    """
    torch._dynamo.config.capture_scalar_outputs = True


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
    """Build the compiled embed-and-reduce call.

    Args:
        language_model: The wrapped language model.
        reduction_layer: Projects the embeddings onto the scoring model's
            reduced dimension. ``None`` leaves them unreduced.

    Returns:
        A callable mapping the encoder's output to a numpy array of shape
        ``(batch, max_len, dim)``.
    """
    _enable_scalar_capture()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    language_model.to(device)
    if reduction_layer is not None:
        reduction_layer.to(device)

    def _call(lm_inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        emb = language_model(lm_inputs)
        if reduction_layer is None:
            return emb
        return reduction_layer.reduce(emb, training=False)

    compiled = torch.compile(_call, dynamic=True)

    def embed(lm_inputs: tuple[np.ndarray, ...]) -> np.ndarray:
        tensors = tuple(_to_tensor(x, device) for x in lm_inputs)
        with torch.inference_mode():
            emb = compiled(tensors)
        return emb.to(torch.float32).cpu().numpy()

    return embed


def _to_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    """Move one encoder output onto the compute device."""
    array = np.asarray(array)
    dtype = _DTYPES.get(array.dtype.name)
    return torch.as_tensor(array, dtype=dtype, device=device)
