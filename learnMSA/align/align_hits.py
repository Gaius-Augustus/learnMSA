import numpy as np


def greedy_bucket_assignment(Z: np.ndarray) -> np.ndarray:
    """Assigns domain hits to buckets such that the highest-scoring
    domain hits per row align.

    Args:
        Z : np.ndarray, shape (N, K)
            Score matrix.

    Returns:
        shifts : np.ndarray of bool, shape (N,)
    """
    K = Z.shape[1]
    # Find the best hit per row
    Zm = np.argmax(Z, axis=-1)
    # Compute how much the domains per row must be shifted
    shifts = np.amax(Zm) - Zm
    return shifts
