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


def is_logits_file(filepath: str | Path) -> bool:
    """Whether ``filepath`` is a 3Di logits file."""
    filepath = Path(filepath)
    if filepath.suffix != ".npz" or not filepath.is_file():
        return False
    with np.load(filepath, allow_pickle=False) as data:
        return "kind" in data and str(data["kind"]) == KIND
