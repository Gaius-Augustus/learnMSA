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

#: Seconds that compiling the language model costs up front. Measured for
#: ProtT5 on an RTX 3090 with a cold inductor cache (~60 s; ~25 s warm).
COMPILE_OVERHEAD = 60.0

#: Seconds that compiling saves per residue and per residue of sequence
#: length: the compiled model is faster mostly in attention, so the saving per
#: residue grows linearly with the sequence length. Measured for ProtT5 on an
#: RTX 3090 (6% faster at length 100, 13% at 300, 27% at 600, 30% at 1000;
#: 4,000 proteins of length ~1000 took 505 s compiled from a cold cache and
#: 557 s eagerly).
COMPILE_SAVING = 3e-8


def should_compile(seq_lens: np.ndarray) -> bool:
    """Whether compiling the language model pays off for these sequences.

    Args:
        seq_lens: Lengths of all sequences that will be embedded.

    Returns:
        True if the estimated saving exceeds the compile overhead.
    """
    if not torch.cuda.is_available():
        return False
    seq_lens = np.asarray(seq_lens, dtype=np.float64)
    return COMPILE_SAVING * float(np.sum(seq_lens**2)) > COMPILE_OVERHEAD


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
