"""Files of per-residue 3Di logits (numpy only).

A logits file is an :class:`~learnMSA.util.EmbeddingDataset` file whose
metadata has

- ``alphabet``: the 3Di letter of each logit column, in column order,
- ``kind``: ``"3di_logits"``, to tell these files apart from embeddings,
- ``source``: free text naming the predictor that wrote them.
"""

from pathlib import Path

import numpy as np

KIND = "3di_logits"

TOKEN_LOGIT_MARGIN = 20.0
"""Log-odds of a known 3Di letter over the others in ``tokens_as_logits``."""


def is_logits_file(filepath: str | Path) -> bool:
    """Whether ``filepath`` is a 3Di logits file."""
    filepath = Path(filepath)
    if filepath.suffix != ".npz" or not filepath.is_file():
        return False
    with np.load(filepath, allow_pickle=False) as data:
        return "kind" in data and str(data["kind"]) == KIND


def tokens_as_logits(
    observations: np.ndarray,
    background: np.ndarray,
    temperature: float,
    margin: float = TOKEN_LOGIT_MARGIN,
) -> np.ndarray:
    """Encodes 3Di letters as logits that the structural observation layer
    maps back to the letters' observation vectors, so that letters from
    structures and ProstT5 logits can share one track.
    """
    pi = np.asarray(background, dtype=np.float64)
    x = np.asarray(observations, dtype=np.float64)
    # All-ones rows carry no letter.
    x = x - x.min(axis=-1, keepdims=True)
    z = temperature * (np.log(pi / pi.sum()) + margin * x)
    return z.astype(np.float16)
