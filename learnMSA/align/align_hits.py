import numpy as np


def greedy_hit_alignment(Z: np.ndarray) -> np.ndarray:
    """Aligns the domain hits such that the highest-scoring
    domain hits assemble into the same column. This is greedy
    heuristic that quickly constructs a representative, high-scoring
    hit column. The remaining hits are not directly aligned and their
    scores do not contribute. Hits are just shifted as a continuous block.

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
