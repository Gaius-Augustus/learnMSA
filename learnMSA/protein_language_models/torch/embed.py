from typing import Callable

import numpy as np
import torch

from learnMSA.protein_language_models.common import LanguageModel
from learnMSA.protein_language_models.t5_compile import compile_pays_off

#: Numpy dtypes the encoders emit, mapped to their torch equivalents.
_DTYPES: dict[str, torch.dtype] = {
    "int32": torch.int32,
    "int64": torch.int64,
    "float16": torch.float16,
    "float32": torch.float32,
}

def should_compile(seq_lens: np.ndarray) -> bool:
    """Whether compiling the language model pays off for these sequences.

    Args:
        seq_lens: Lengths of all sequences that will be embedded.

    Returns:
        True on a GPU if the estimated saving exceeds the compile overhead.
    """
    return torch.cuda.is_available() and compile_pays_off(seq_lens)


def make_embedding_fn(
    language_model: LanguageModel,
    reduction_layer: torch.nn.Module | None = None,
    seq_lens: np.ndarray | None = None,
) -> Callable[[tuple[np.ndarray, ...]], np.ndarray]:
    """Build the embed-and-reduce call.

    The transformer inside the language model is compiled with dynamic shapes
    if :func:`should_compile` expects that to be faster for ``seq_lens``;
    otherwise everything runs eagerly.

    Args:
        language_model: The wrapped language model.
        reduction_layer: Projects the embeddings onto the scoring model's
            reduced dimension. ``None`` leaves them unreduced.
        seq_lens: Lengths of all sequences that will be embedded. ``None``
            runs eagerly.

    Returns:
        A callable mapping the encoder's output to a numpy array of shape
        ``(batch, max_len, dim)``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    language_model.to(device)
    if reduction_layer is not None:
        reduction_layer.to(device)

    # Only the wrapped transformer is compiled. The wrappers end by cropping
    # padding columns, which needs a Python int and would break the graph.
    compiled = (
        seq_lens is not None
        and hasattr(language_model, "model")
        and should_compile(seq_lens)
    )
    if compiled:
        language_model.model = torch.compile(
            language_model.model, dynamic=True
        )

    def embed(lm_inputs: tuple[np.ndarray, ...]) -> np.ndarray:
        batch_size = len(lm_inputs[0])
        if compiled and batch_size == 1:
            # Dynamo specializes on size 1, so a single sequence (e.g. the
            # last batch) would trigger a full recompile. Duplicate it.
            lm_inputs = tuple(np.repeat(x, 2, axis=0) for x in lm_inputs)
        tensors = tuple(_to_tensor(x, device) for x in lm_inputs)
        with torch.inference_mode():
            emb = language_model(tensors)
            if reduction_layer is not None:
                emb = reduction_layer.reduce(emb, training=False)
            return emb[:batch_size].to(torch.float32).cpu().numpy()

    return embed


def _to_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    """Move one encoder output onto the compute device."""
    array = np.asarray(array)
    dtype = _DTYPES.get(array.dtype.name)
    return torch.as_tensor(array, dtype=dtype, device=device)
