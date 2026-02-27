from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from types import TracebackType
from typing import Self

import numpy as np


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

    parsing_ok: bool
    """Whether parsing the source was successful."""

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
        dtype: type[np.integer] = np.int16,
    ) -> np.ndarray:
        """Return sequence i encoded as a numpy integer array."""

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