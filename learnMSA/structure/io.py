"""Files of per-residue 3Di logits (numpy only).

A logits file is an :class:`~learnMSA.util.EmbeddingDataset` file whose
metadata has

- ``alphabet``: the 3Di letter of each logit column, in column order,
- ``kind``: ``"3di_logits"``, to tell these files apart from embeddings,
- ``source``: free text naming the predictor that wrote them,
- ``logit_source``: how the logits were produced, which sets how learnMSA
  applies the temperature (see :data:`LOGIT_SOURCES`); files without it
  are ProstT5 predictions.
"""

from pathlib import Path

import numpy as np

KIND = "3di_logits"

#: Metadata key of the logit source.
SOURCE_KEY = "logit_source"
#: Logits predicted from sequence by ProstT5: a posterior over the letters.
PROSTT5 = "prostt5"
#: Logits translated from a structure (``learnMSA-3di-translate``):
#: ``-d^2``, with ``d`` the distance of the VQ-VAE embedding to each
#: letter's centroid.
PDB = "pdb"
LOGIT_SOURCES = (PROSTT5, PDB)


def logit_source(metadata: dict) -> str:
    """The logit source recorded in a file's metadata."""
    source = str(metadata.get(SOURCE_KEY, PROSTT5))
    if source not in LOGIT_SOURCES:
        raise ValueError(f"Unknown 3Di logit source '{source}', expected "
                         f"one of {LOGIT_SOURCES}.")
    return source


def is_logits_file(filepath: str | Path) -> bool:
    """Whether ``filepath`` is a 3Di logits file."""
    filepath = Path(filepath)
    if filepath.suffix != ".npz" or not filepath.is_file():
        return False
    with np.load(filepath, allow_pickle=False) as data:
        return "kind" in data and str(data["kind"]) == KIND
