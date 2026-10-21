from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from pathlib import Path
from types import TracebackType
from typing import Self

import numpy as np

from learnMSA.util.redundancy import RedundancyMap


class Dataset(ABC):
    """Abstract base class for sequence-like datasets."""

    filepath: Path
    """Path to the underlying dataset file."""

    fmt: str
    """Format of the underlying dataset file."""

    seq_ids: list[str]
    """Ordered sequence IDs."""

    num_seq: int
    """Number of sequences in the dataset."""

    seq_lens: np.ndarray
    """Per-sequence lengths."""

    max_len: int
    """Maximum sequence length."""

    indexed: bool = False
    """Whether the dataset will be streamed from disk without fully loading it.
    Only supported for SequenceDataset and fasta files."""

    parsing_ok: bool
    """Whether parsing the source was successful."""

    redundancy: RedundancyMap | None = None
    """Set by filter_duplicates: maps the original sequences to the
    representatives that the dataset exposes."""

    _index_attrs: tuple[str, ...] = ("seq_ids", "seq_lens")
    """Per-sequence attributes that select and filter_duplicates index."""

    _full_index_state: dict | None = None

    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return self.num_seq

    @abstractmethod
    def close(self) -> None:
        """Release any resources associated with the dataset."""

    @abstractmethod
    def get_encoded_seq(
        self,
        i: int,
        crop_start: int | None = None,
        crop_end: int | None = None,
    ) -> np.ndarray:
        """Return sequence i encoded as a numpy float array (one-hot /
        per-position distribution), never as integer indices."""

    @abstractmethod
    def empty(
        self,
        shape: tuple[int, ...],
        dtype: type[np.integer | np.floating] = np.int16,
    ) -> np.ndarray:
        """Return a tensor of "null" values of the given shape and dtype.
        This is useful to initialize a tensor that should contain encoded
        sequences of different lengths."""

    @abstractmethod
    def write(
        self,
        filepath: Path | str,
        fmt: str,
        standardize_sequences: bool = False,
    ) -> None:
        """
        Write the dataset to a file.

        Args:
            filepath (Path): Path to the output file.
            fmt (str): Format of the output file. Can be any format supported
                by Biopython's SeqIO.
            standardize_sequences (bool): If True, sequences are converted to
                a standardized format (e.g. for amino acids uppercase and
                non-standard amino acids are replaced with 'X'. Dots are
                replaced with dashes).
        """

    @abstractmethod
    def reorder(self, permutation: list[int] | np.ndarray) -> None:
        """Reorder sequences in-place using a permutation of sequence indices."""

    def adapt_order(self, other_dataset: 'Dataset') -> None:
        """
        Reorder the sequences in-place to match the order of another dataset.

        Args:
            other_dataset: The dataset to match the order of. Must contain the
                same sequence IDs as this dataset.
        """
        if set(self.seq_ids) != set(other_dataset.seq_ids):
            raise ValueError(
                "The sequence IDs in the two datasets do not match."
            )
        index = {seq_id: i for i, seq_id in enumerate(self.seq_ids)}
        perm = [index[seq_id] for seq_id in other_dataset.seq_ids]
        self.reorder(perm)

    @abstractmethod
    def get_dtype(self) -> type[np.integer | np.floating]:
        """Return the dtype of the encoded sequences."""

    def filter_duplicates(
        self, redundancy: RedundancyMap | None = None
    ) -> RedundancyMap:
        """
        Hides all but one copy of identical sequences in-place. Afterwards,
        the dataset only shows the representatives (in input order) and
        ``unfiltered()`` shows all sequences. Calling it again returns the
        existing map.

        Args:
            redundancy: Use this map instead of finding the duplicates in
                this dataset, e.g. the map of the amino acid dataset for an
                auxiliary dataset in the same order.

        Returns:
            The map from the original sequences to the representatives.
        """
        if self.redundancy is not None:
            if redundancy is not None and redundancy is not self.redundancy:
                raise ValueError(
                    "The dataset was already filtered with another map."
                )
            return self.redundancy
        if redundancy is None:
            redundancy = self._find_duplicates()
        if redundancy.num_full != self.num_seq:
            raise ValueError(
                f"The redundancy map covers {redundancy.num_full} sequences, "
                f"but the dataset has {self.num_seq}."
            )
        if not redundancy.is_trivial:
            self._full_index_state = {
                a: getattr(self, a) for a in self._index_attrs
            }
            self._select(redundancy.rep_full_index)
        self.redundancy = redundancy
        return redundancy

    def unfiltered(self) -> Self:
        """A view of the dataset with all sequences, including those hidden
        by filter_duplicates. The dataset itself if nothing is hidden."""
        if self._full_index_state is None:
            return self
        view = copy.copy(self)
        for a, value in self._full_index_state.items():
            setattr(view, a, value)
        view.redundancy = None
        view._full_index_state = None
        view._update_counts()
        return view

    def subset(self, indices: np.ndarray) -> Self:
        """A view of the dataset with the sequences at ``indices``."""
        view = copy.copy(self)
        view.redundancy = None
        view._full_index_state = None
        view._select(np.asarray(indices))
        return view

    def _find_duplicates(self) -> RedundancyMap:
        raise NotImplementedError(
            f"{type(self).__name__} can not find duplicates itself. Pass the "
            "redundancy map of the corresponding sequence dataset."
        )

    def _select(self, indices: np.ndarray) -> None:
        """Keeps the sequences at ``indices`` in-place."""
        for a in self._index_attrs:
            value = getattr(self, a)
            if isinstance(value, list):
                value = [value[i] for i in indices]
            else:
                value = value[indices]
            setattr(self, a, value)
        self._update_counts()

    def _update_counts(self) -> None:
        self.num_seq = len(self.seq_ids)
        self.max_len = int(np.amax(self.seq_lens)) if self.num_seq > 0 else 0

    def __enter__(self) -> Self:
        """Enter context manager and return the dataset."""
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None
    ) -> None:
        """Exit context manager and clean up resources."""
        self.close()