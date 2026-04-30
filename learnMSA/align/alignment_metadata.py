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
    """An array of shape (num_repeat, num_rows,) containing the lengths of the
    unannotated segments.
    """

    unannotated_segments_start: np.ndarray
    """An array of shape (num_repeat, num_rows,) containing the starting
    positions of the unannotated segments.
    """

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
        """An array of shape (num_repeats,) containing the maximum total length
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
