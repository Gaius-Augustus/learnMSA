from dataclasses import dataclass
import numpy as np


# utility class used in AlignmentModel storing data to construct a full MSA
@dataclass
class AlignmentMetaData:

    num_rows: int
    """Number of sequences in the alignment."""

    num_match: int
    """Total number of match states in pHMM."""

    num_repeats: int
    """Maximum number of repeats across all sequences."""

    domain_hit: np.ndarray
    """An array of shape (num_repeats, num_rows, num_match) with the domain hits.
    """

    domain_loc: np.ndarray
    """An array of shape (num_repeats, num_rows, 2) with the starting and
    ending positions of the domain hits in the sequences.
    """

    insertion_lens: np.ndarray
    """An array of shape (num_repeats, num_rows, num_match - 1)
    containing the lengths of insertions.
    """

    insertion_start: np.ndarray
    """An array of shape (num_repeats, num_rows, num_match - 1) containing the
    starting positions of insertions.
    """

    skip: np.ndarray
    """A binary array of shape (num_repeats, num_rows) indicating with 1 which
    rows skip domain hits.
    """

    left_flank_len: np.ndarray
    """An array of shape (num_rows,) containing the lengths of the left flanks.
    """

    left_flank_start: np.ndarray
    """An array of shape (num_rows,) containing the starting positions of the
    left flanks.
    """

    right_flank_len: np.ndarray
    """An array of shape (num_rows,) containing the lengths of the right flanks.
    """

    right_flank_start: np.ndarray
    """An array of shape (num_rows,) containing the starting positions of the
    right flanks.
    """

    unannotated_segments_len: np.ndarray
    """An array of shape (num_repeat-1, num_rows,) containing the lengths of the
    unannotated segments.
    """

    unannotated_segments_start: np.ndarray
    """An array of shape (num_repeat-1, num_rows,) containing the starting
    positions of the unannotated segments.
    """

    @property
    def num_repeats_per_row(self) -> np.ndarray:
        """An array of shape (num_rows,) containing the number of repeats for
        each row.
        """
        return np.sum(self.skip == 0, axis=0) + 1

    @property
    def insertion_lens_total(self) -> np.ndarray:
        """An array of shape (num_repeats, num_match - 1) containing the
        maximum total length of insertions across all sequences for each repeat.
        """
        return np.amax(self.insertion_lens, axis=1).astype(np.int32)

    @property
    def left_flank_len_total(self) -> np.int32:
        """The maximum length of the left flanks across all sequences."""
        return np.amax(self.left_flank_len).astype(np.int32)

    @property
    def right_flank_len_total(self) -> np.int32:
        """The maximum length of the right flanks across all sequences."""
        return np.amax(self.right_flank_len).astype(np.int32)

    @property
    def unannotated_segment_lens_total(self) -> np.ndarray:
        """An array of shape (num_repeats-1,) containing the maximum total length
        of the unannotated segments across all sequences for each repeat.
        """
        return np.amax(self.unannotated_segments_len, axis=1).astype(np.int32)

    @property
    def alignment_len(self) -> int:
        """The total length of the alignment, including flanks, core blocks,
        insertions, and unannotated segments. Might contain empty columns
        (which will be removed later).
        """
        return (
            self.left_flank_len_total +
            self.num_match * self.num_repeats +
            np.sum(self.insertion_lens_total) +
            np.sum(self.unannotated_segment_lens_total) +
            self.right_flank_len_total
        )

    def shift(self, shift: np.ndarray) -> None:
        """Shifts the domain hits according to the given shift vector. Can be
        used to align domain hits. Note that this only supports shifting all
        domain hits to the right as a continuous block. Gaps are currently not
        supported.
        Will affect alignment length when a row is shifted beyond the current
        alignment length.

        Args:
            shift: np.ndarray of shape (num_rows,) containing the shift values
                for each row.
        """
        self.num_repeats = np.amax(self.num_repeats_per_row + shift)
        self.domain_hit = self._ashift(self.domain_hit, shift, -1)
        self.domain_loc = self._ashift(self.domain_loc, shift, -1)
        self.insertion_lens = self._ashift(self.insertion_lens, shift, 0)
        self.insertion_start = self._ashift(self.insertion_start, shift, -1)
        self.skip = self._ashift(self.skip, shift, 1)
        self.unannotated_segments_len = self._ashift(
            self.unannotated_segments_len, shift, 0, margin=-1
        )
        self.unannotated_segments_start = self._ashift(
            self.unannotated_segments_start, shift, -1, margin=-1
        )

    def _ashift(
        self,
        arr: np.ndarray,
        shift: np.ndarray,
        padding_value: int | float,
        margin: int = 0,
    ) -> np.ndarray:
        """Shifts the given array according to the given shift vector. This is
        a helper function for shift().

        Args:
            arr: np.ndarray of shape (num_repeats, N, ...)
            shift: np.ndarray of shape (N,) containing the shift values
                for each row.
            padding_value: The value to use for padding.
            margin: Margin added to the number of repeats.

        Returns:
            shifted_arr: np.ndarray of shape (new_num_repeats, N, ...)
            where in each row i the values of `arr` are shifted by shift[i]
            positions to the right.
        """
        assert margin <= 0, "Margin must be non-positive."
        new_num_repeats = self.num_repeats + margin
        new_shape = (new_num_repeats,) + arr.shape[1:]
        result = np.full(new_shape, padding_value, dtype=arr.dtype)

        if arr.shape[0] == 0 or new_num_repeats == 0:
            return result

        r_idx = np.arange(arr.shape[0])[:, np.newaxis]      # (old_R, 1)
        row_idx = np.arange(arr.shape[1])[np.newaxis, :]    # (1, N)
        dest_r = r_idx + shift[np.newaxis, :]               # (old_R, N)

        mask = dest_r < new_num_repeats                     # (old_R, N)
        row_idx_exp = np.broadcast_to(row_idx, dest_r.shape)
        result[dest_r[mask], row_idx_exp[mask]] = arr[mask]
        return result
