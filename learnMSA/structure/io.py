"""Files of per-residue 3Di logits (numpy only).

The layout is the one of :class:`~learnMSA.util.EmbeddingDataset` (``cache``,
``seq_lens``, ``seq_ids``, ``permutation``, ``dim``) plus

- ``alphabet``: the 3Di letter of each logit column, in column order,
- ``kind``: ``"3di_logits"``, to tell these files apart from embeddings,
- ``source``: free text naming the predictor that wrote them.
"""

from pathlib import Path

import numpy as np

from learnMSA.util.embedding_cache import EmbeddingCache
from learnMSA.util.embedding_dataset import EmbeddingDataset

KIND = "3di_logits"


def write_logits(
    filepath: str | Path,
    logits: np.ndarray,
    seq_lens: np.ndarray,
    seq_ids: list[str],
    alphabet: str,
    source: str = "",
) -> Path:
    """Writes concatenated per-residue logits of shape (sum(seq_lens), A).

    Returns the path written (``np.savez`` appends ``.npz`` if missing).
    """
    seq_lens = np.asarray(seq_lens, dtype=np.int64)
    if logits.shape != (int(seq_lens.sum()), len(alphabet)):
        raise ValueError(
            f"Expected logits of shape ({int(seq_lens.sum())}, "
            f"{len(alphabet)}), got {logits.shape}."
        )
    if len(seq_ids) != seq_lens.size:
        raise ValueError("seq_ids and seq_lens differ in length.")
    filepath = Path(filepath)
    if filepath.suffix != ".npz":
        filepath = filepath.with_suffix(filepath.suffix + ".npz")
    np.savez(
        filepath,
        cache=logits,
        seq_lens=seq_lens,
        seq_ids=np.array(seq_ids, dtype=str),
        permutation=np.arange(seq_lens.size),
        dim=np.array([len(alphabet)]),
        alphabet=np.array(alphabet),
        kind=np.array(KIND),
        source=np.array(source),
    )
    return filepath


def is_logits_file(filepath: str | Path) -> bool:
    """Whether ``filepath`` is a 3Di logits file written by
    :func:`write_logits`."""
    filepath = Path(filepath)
    if filepath.suffix != ".npz" or not filepath.is_file():
        return False
    with np.load(filepath, allow_pickle=False) as data:
        return "kind" in data and str(data["kind"]) == KIND


def read_logits(filepath: str | Path, alphabet: str) -> EmbeddingDataset:
    """Reads a 3Di logits file as a dataset whose columns follow
    ``alphabet``.

    The columns are permuted if the file stores the same letters in another
    order. A file over a different set of letters is rejected.
    """
    with np.load(filepath, allow_pickle=False) as data:
        if "kind" not in data or str(data["kind"]) != KIND:
            raise ValueError(f"{filepath} is not a 3Di logits file.")
        stored = str(data["alphabet"])
        cache = data["cache"]
        seq_lens = data["seq_lens"]
        seq_ids = data["seq_ids"].tolist()
        permutation = data["permutation"]
    if sorted(stored) != sorted(alphabet) or len(set(stored)) != len(stored):
        raise ValueError(
            f"The 3Di letters of {filepath} ({stored}) do not match the "
            f"structural alphabet ({alphabet})."
        )
    if stored != alphabet:
        cache = cache[:, [stored.index(c) for c in alphabet]]
    # As in EmbeddingDataset files, seq_ids[i] names cache row
    # permutation[i]. Build the dataset in cache order, then apply the
    # stored permutation.
    cache_ids = [""] * len(seq_ids)
    for i, row in enumerate(permutation):
        cache_ids[row] = seq_ids[i]
    dataset = EmbeddingDataset(
        embedding_cache=EmbeddingCache(seq_lens, len(alphabet), cache=cache),
        seq_ids=cache_ids,
    )
    dataset.reorder(permutation)
    dataset.filepath = Path(filepath)
    return dataset
