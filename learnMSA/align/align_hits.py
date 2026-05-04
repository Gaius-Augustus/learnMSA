from enum import Enum

import numpy as np

from learnMSA.align.alignment_metadata import AlignmentMetaData


class HitAlignmentMode(Enum):
    LEFT_ALIGN = "left_align"
    """Aligns the domain hits by index (starting from the left)."""
    RIGHT_ALIGN = "right_align"
    """Aligns the domain hits by negative index (starting from the right)."""
    GREEDY_CONSENSUS = "greedy_consensus"
    """Aligns the domain hits such that the highest-scoring
    domain hits assemble into the same column."""

    @staticmethod
    def from_str(label: str) -> "HitAlignmentMode":
        if label.lower() == "left":
            return HitAlignmentMode.LEFT_ALIGN
        elif label.lower() == "right":
            return HitAlignmentMode.RIGHT_ALIGN
        elif label.lower() == "greedy":
            return HitAlignmentMode.GREEDY_CONSENSUS
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
            Score matrix required by `GREEDY_CONSENSUS` mode.

    Returns:
        Updated AlignmentMetaData with aligned domain hits.
    """
    if mode == HitAlignmentMode.LEFT_ALIGN:
        print("Aligning hits by left index...")
        return data

    elif mode == HitAlignmentMode.RIGHT_ALIGN:
        print("Aligning hits by right index...")
        scores = np.zeros((data.num_repeats, data.num_rows))
        scores[data.num_repeats_per_row - 1, np.arange(data.num_rows)] = 1
        shift = greedy_consensus_hit_alignment(scores)
        data.shift(shift)
        return data

    elif mode == HitAlignmentMode.GREEDY_CONSENSUS:
        print("Aligning hits by greedy consensus...")
        assert scores is not None,\
            "Scores are required for greedy consensus hit alignment."
        shift = greedy_consensus_hit_alignment(scores)
        data.shift(shift)
        return data

    else:
        raise ValueError(f"Unknown hit alignment mode: {mode}")


def greedy_consensus_hit_alignment(Z: np.ndarray) -> np.ndarray:
    """Aligns the domain hits such that the highest-scoring
    domain hits assemble into the same column. This is greedy
    heuristic that quickly constructs a representative, high-scoring
    hit column. Hits are shifted as a continuous block. Non-best hits
    are not directly aligned and their scores do not contribute.

    Args:
        Z : np.ndarray, shape (num_repeats, num_rows)
            Score matrix.

    Returns:
        shifts : np.ndarray of bool, shape (N,)
    """
    K = Z.shape[0]
    # Find the best hit per row
    Zm = np.argmax(Z, axis=0)
    # Compute how much the domains per row must be shifted
    shifts = np.amax(Zm) - Zm
    return shifts
