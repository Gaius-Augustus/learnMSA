from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .dataset import Dataset
from .sequence_dataset import SequenceDataset


class MultiDataset(Dataset):
    """
    Concatenates datasets into one global index space.

    Dataset ``k`` owns the global indices ``offsets[k]:offsets[k+1]``. All
    per-sequence attributes (``seq_ids``, ``seq_lens``) are concatenated in
    that order, and :meth:`get_encoded_seq` accepts a global index. This is
    the type of the auxiliary tracks (e.g. 3Di strings or embeddings) that
    accompany a :class:`MultiSequenceDataset`.

    Attributes:
        datasets (list[Dataset]): The parts, in index order.
        offsets (np.ndarray): Global index of the first sequence of each part,
            followed by the total number of sequences. Shape: (K+1,)
    """

    def __init__(self, datasets: Sequence[Dataset]) -> None:
        """
        Args:
            datasets: The datasets to concatenate. All of them must encode
                sequences the same way, i.e. agree on the shape and dtype of
                :meth:`empty`.
        """
        if len(datasets) == 0:
            raise ValueError("A MultiDataset needs at least one dataset.")
        self.datasets = list(datasets)
        reference = self.datasets[0]
        for d in self.datasets[1:]:
            if type(d) is not type(reference) \
                    or d.get_dtype() != reference.get_dtype() \
                    or d.empty(()).shape != reference.empty(()).shape:
                raise ValueError(
                    "All datasets of a MultiDataset must encode sequences "
                    "the same way."
                )
        sizes = [d.num_seq for d in self.datasets]
        self.offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        self.filepath = Path()
        self.fmt = reference.fmt
        self.seq_ids = [sid for d in self.datasets for sid in d.seq_ids]
        self.num_seq = int(self.offsets[-1])
        self.seq_lens = np.concatenate([d.seq_lens for d in self.datasets])
        self.max_len = int(np.amax(self.seq_lens)) if self.num_seq else 0
        self.indexed = any(d.indexed for d in self.datasets)
        self.parsing_ok = all(d.parsing_ok for d in self.datasets)

    @property
    def num_datasets(self) -> int:
        """The number of concatenated datasets."""
        return len(self.datasets)

    def dataset_of(self, i: int | np.ndarray) -> Any:
        """The index of the dataset that owns global index (or indices) i."""
        return np.searchsorted(self.offsets, i, side="right") - 1

    def to_local(self, i: int) -> tuple[int, int]:
        """Maps a global index to (dataset index, index within the dataset).
        """
        if i < 0 or i >= self.num_seq:
            raise IndexError(
                f"Index {i} is out of range for {self.num_seq} sequences."
            )
        k = int(self.dataset_of(i))
        return k, int(i - self.offsets[k])

    def to_global(self, k: int, i: int | np.ndarray) -> Any:
        """Maps index (or indices) i of dataset k to global indices."""
        return self.offsets[k] + i

    def global_indices(self, k: int) -> np.ndarray:
        """All global indices of dataset k."""
        return np.arange(self.offsets[k], self.offsets[k + 1])

    def close(self) -> None:
        for d in self.datasets:
            d.close()

    def get_encoded_seq(
        self,
        i: int,
        crop_start: int | None = None,
        crop_end: int | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Returns the encoded sequence with global index i. Keyword
        arguments are passed on to the dataset that owns it."""
        k, j = self.to_local(i)
        return self.datasets[k].get_encoded_seq(
            j, crop_start, crop_end, **kwargs
        )

    def empty(
        self,
        shape: tuple[int, ...],
        dtype: type[np.integer | np.floating] = np.float32,
    ) -> np.ndarray:
        return self.datasets[0].empty(shape, dtype)

    def get_dtype(self) -> type[np.integer | np.floating]:
        return self.datasets[0].get_dtype()

    def write(
        self,
        filepath: Path | str,
        fmt: str = "fasta",
        standardize_sequences: bool = False,
    ) -> None:
        raise NotImplementedError(
            "A MultiDataset can not be written as a whole. Write its parts."
        )

    def reorder(self, permutation: list[int] | np.ndarray) -> None:
        raise NotImplementedError(
            "A MultiDataset can not be reordered. Reorder its parts before "
            "combining them."
        )


class MultiSequenceDataset(MultiDataset, SequenceDataset):
    """
    Holds the sequences of several SequenceDatasets in one global index space.

    Behaves like a SequenceDataset whose sequences are the concatenation of
    its parts: ``get_record``, ``get_header``, ``get_encoded_seq`` and the
    other accessors take a global index. Use :meth:`to_local`,
    :meth:`to_global` and :meth:`global_indices` to convert between global
    indices and (dataset, local index) coordinates.
    """

    #: Encoding settings that all parts must share.
    _ENCODING_ATTRIBUTES = (
        "alphabet",
        "output_alphabet",
        "remap",
        "model_uo",
        "remove_gaps",
        "gap_symbols",
        "ignore_symbols",
        "validate_alphabet",
    )

    def __init__(
        self,
        datasets: Sequence[SequenceDataset] | None = None,
        *,
        filepaths: Sequence[Path | str] | None = None,
        sequences: Sequence[list[tuple[str, str]]] | None = None,
        fmt: str = "fasta",
        **kwargs,
    ) -> None:
        """
        Exactly one of datasets, filepaths or sequences must be given.

        Args:
            datasets: Existing SequenceDatasets.
            filepaths: Sequence files, one per dataset.
            sequences: In-memory id/sequence pairs, one list per dataset.
            fmt: Format of the files in filepaths.
            **kwargs: Passed on to SequenceDataset when parsing filepaths or
                sequences (e.g. alphabet, indexed, model_uo).
        """
        sources = [x is not None for x in (datasets, filepaths, sequences)]
        if sum(sources) != 1:
            raise ValueError(
                "Provide exactly one of datasets, filepaths or sequences."
            )
        if filepaths is not None:
            datasets = [SequenceDataset(p, fmt, **kwargs) for p in filepaths]
        elif sequences is not None:
            datasets = [
                SequenceDataset(sequences=list(s), **kwargs)
                for s in sequences
            ]
        assert datasets is not None
        for d in datasets:
            if not isinstance(d, SequenceDataset):
                raise ValueError(
                    "All parts of a MultiSequenceDataset must be "
                    "SequenceDatasets."
                )
        MultiDataset.__init__(self, datasets)
        reference = self.datasets[0]
        for attr in self._ENCODING_ATTRIBUTES:
            value = getattr(reference, attr)
            if any(getattr(d, attr) != value for d in self.datasets[1:]):
                raise ValueError(
                    "All parts of a MultiSequenceDataset must share the same "
                    f"encoding, but they differ in {attr}."
                )
            setattr(self, attr, value)
        # The inherited SequenceDataset methods encode with these.
        self._remap_matrix = reference._remap_matrix
        self._invalid_char_pattern = reference._invalid_char_pattern
        self.record_dict = {}
        self._released = False

    def get_record(self, i: int):
        k, j = self.to_local(i)
        return self.datasets[k].get_record(j)

    def validate_dataset(self, **kwargs) -> None:
        """Validates every part. Sequence ids only need to be unique within
        a part."""
        for d in self.datasets:
            d.validate_dataset(**kwargs)

    def release_records(self) -> None:
        for d in self.datasets:
            d.release_records()
