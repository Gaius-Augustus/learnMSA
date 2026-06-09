from enum import Enum

import numpy as np

from learnMSA.align.alignment_metadata import AlignmentMetaData


class HitAlignmentMode(Enum):
    LEFT_ALIGN = "left"
    """Aligns the domain hits by index (starting from the left)."""
    RIGHT_ALIGN = "right"
    """Aligns the domain hits by negative index (starting from the right)."""
    GREEDY_SCORES = "greedy_scores"
    """Aligns the domain hits such that the domain hits that use the most
    matches assemble into the same columns."""
    GREEDY_SINGLE = "greedy_single"
    """Aligns the domain hits such that the single best domain hits
    assemble into the same columns. The best domain hit is determined by
    comparing the hit intervals in a multi-hit model with the hit interval
    resulting from a single-hit model."""

    @staticmethod
    def from_str(label: str) -> "HitAlignmentMode":
        if label.lower() == "left":
            return HitAlignmentMode.LEFT_ALIGN
        elif label.lower() == "right":
            return HitAlignmentMode.RIGHT_ALIGN
        elif label.lower() == "greedy_scores":
            return HitAlignmentMode.GREEDY_SCORES
        elif label.lower() == "greedy_single":
            return HitAlignmentMode.GREEDY_SINGLE
        else:
            raise ValueError(f"Unsupported decoding mode: {label}")


def hit_alignment(
    data: AlignmentMetaData,
    mode: HitAlignmentMode,
    scores: np.ndarray | None = None,
) -> AlignmentMetaData:
    """Aligns the domain hits according to the specified mode.

    Args:
        alignmnent_data : AlignmentMetaData
            Alignment metadata containing the domain hits and their locations.
        mode : HitAlignmentMode
            Mode for aligning the domain hits.
        scores : np.ndarray, shape (num_repeats, num_rows)
            Score matrix required by `GREEDY_SCORES` mode. The scores should
            be non-negative, except for -1 which is allowed to indicate empty
            hits.

    Returns:
        Updated AlignmentMetaData with aligned domain hits.
    """
    if mode == HitAlignmentMode.LEFT_ALIGN:
        return data

    elif mode == HitAlignmentMode.RIGHT_ALIGN:
        # Build (num_repeats, num_rows) scores: 0 for actual hits, -1 for empty,
        # 1 for the last actual repeat of each row.
        occ = data.occupancy_matrix()  # (R, N), -1 for empty
        scores = np.where(occ == -1, -1.0, 0.0)
        last_virt = data._repeat_offset + data.num_repeats_per_row - 1
        scores[last_virt, np.arange(data.num_rows)] = 1.0
        shift = greedy_scores_hit_alignment(scores)
        data.shift(shift)
        return data

    elif mode == HitAlignmentMode.GREEDY_SCORES:
        assert scores is not None,\
            "Scores are required for greedy consensus hit alignment."
        shift = greedy_scores_hit_alignment(scores)
        data.shift(shift)
        return data

    else:
        raise ValueError(f"Unknown hit alignment mode: {mode}")


def greedy_scores_hit_alignment(
    Z: np.ndarray, prevent_extend: bool = True
) -> np.ndarray:
    """Aligns the domain hits such that the highest-scoring
    domain hits assemble into the same column. This is greedy
    heuristic that quickly constructs a representative, high-scoring
    hit column. Hits are shifted as a continuous block. Non-best hits
    are not directly aligned and their scores do not contribute.

    Args:
        Z : np.ndarray, shape (num_repeats, num_rows)
            Score matrix.
        prevent_extend: If True, prevents shifting beyond the existing
            number of repeats, i.e. prevents extending the alignment.

    Returns:
        shifts : np.ndarray of bool, shape (N,)
    """
    # Find the best hit per row
    Zm = np.argmax(Z, axis=0)
    # Compute how much the domains per row must be shifted
    shifts = np.amax(Zm) - Zm
    if prevent_extend:
        original_num_repeats = np.sum(Z != -1, axis=0)
        total_num_repeats = Z.shape[0]
        # Prevent shifting beyond the existing number of repeats
        shifts = np.minimum(shifts, total_num_repeats - original_num_repeats)
    return shifts
