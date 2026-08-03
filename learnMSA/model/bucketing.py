"""Backend-neutral sequence-length bucketing.

Groups sequences of similar length into buckets with adaptive batch sizes, so
that padding is kept small and the framework only has to compile one shape per
bucket. Pure numpy: the framework-specific pipeline consumes the boundaries and
batch sizes computed here.
"""

import math
from functools import partial
from typing import Sequence

import numpy as np

import learnMSA.model.training_util as training_util
from learnMSA.model.batch_generator import BatchGenerator


def compute_dataset_steps(
    indices: np.ndarray,
    batch_generator: BatchGenerator,
    bucket_boundaries: Sequence[int | float],
    bucket_batch_sizes: Sequence[int],
) -> int:
    """
    Compute the number of steps needed to iterate through a bucketed dataset.

    Args:
        indices: The indices of the sequences to include in the dataset.
        batch_generator: The batch generator (must be configured).
        bucket_boundaries: Sequence length boundaries for bucketing.
        bucket_batch_sizes: Batch sizes for each bucket.

    Returns:
        Number of steps to iterate through the bucketed dataset.
    """
    # Compute number of steps for bucketed dataset
    seq_lengths = batch_generator.data[0].seq_lens[indices]
    total_steps = 0
    boundaries = list(bucket_boundaries) + [math.inf]

    for i, (lower, upper, bsize) in enumerate(
        zip([0] + boundaries[:-1], boundaries, bucket_batch_sizes)
    ):
        # Count sequences in this bucket
        # Match TensorFlow's bucket_by_sequence_length boundary conditions:
        # - Bucket 0: length < boundary[0]
        # - Bucket i (i>0): boundary[i-1] <= length < boundary[i]
        # - Last bucket: length >= boundary[-1]
        if i == 0:
            count = np.sum(seq_lengths < upper)
        elif i == len(bucket_batch_sizes) - 1:
            # Last bucket: inclusive lower bound, no upper bound
            count = np.sum(seq_lengths >= lower)
        else:
            count = np.sum((seq_lengths >= lower) & (seq_lengths < upper))

        # Compute number of batches for this bucket
        if count > 0:
            total_steps += int(np.ceil(count / bsize))

    return total_steps


def make_default_bucket_scheme(
    indices: np.ndarray,
    batch_generator: BatchGenerator,
    model_lengths: Sequence[int],
    batch_size_impl_factor: float = 1.0,
    max_batch_size_override: int | None = None,
) -> tuple[list[int], list[int]]:
    """
    Create inferred bucketing boundaries and adaptive bucket batch sizes.

    Args:
        indices: The indices of the sequences to include in the dataset.
        batch_generator: The batch generator (must be configured).
        model_lengths: List of model lengths for adaptive batching.
        batch_size_impl_factor: Implementation factor passed to adaptive
            batch-size computation.
        max_batch_size_override: Optional override for the maximum batch size
            used in adaptive batch-size computation. If None, uses the default
            MAX_BATCH_SIZE.

    Returns:
        A tuple of (bucket_boundaries, bucket_batch_sizes).
    """
    seq_lens = batch_generator.data[0].seq_lens[indices]

    max_num_buckets = min(indices.size // 10000 + 1, 7)
    if max_num_buckets > 1:
        # Use uniform percentile tiles only up to the 95th percentile and
        # always keep [0.95, 1.0] as a dedicated tail bucket.
        q = np.linspace(0.0, 0.95, max_num_buckets, endpoint=True)[1:]
    else:
        q = np.array([], dtype=float)
    quantiles = np.quantile(seq_lens, q=q).astype(int)

    min_seq_len = int(np.min(seq_lens))
    max_seq_len = int(np.max(seq_lens))

    bucket_boundaries = np.unique(quantiles).tolist()

    # Inlude a final bucket boundary above the maximum sequence length
    # (pad_to_bucket_boundary=True requires every sequence length to be
    # strictly smaller than max(quantiles))
    if len(bucket_boundaries) == 0 or bucket_boundaries[-1] <= max_seq_len:
        bucket_boundaries.append(max_seq_len + 1)

    # Compute an adaptive minimum gap between boundaries. Otherwise bucket
    # boundaries can be very close together if the sequences are homogeneous
    # in length.
    length_range = max_seq_len - min_seq_len
    min_gap = max(10, length_range // (max_num_buckets + 1))

    # Greedy filter: keep a quantile boundary only if it is more than min_gap
    # away from the previous kept boundary.
    filtered: list[int] = [bucket_boundaries[-1]]
    prev = filtered[0]
    for v in bucket_boundaries[:-1][::-1]:  # iterate from largest to smallest
        if prev - v > min_gap:
            filtered.append(v)
            prev = v
    bucket_boundaries = filtered[::-1]

    adaptive_batch = partial(
        training_util.get_adaptive_batch_size,
        impl_factor=batch_size_impl_factor,
        max_batch_size=max_batch_size_override or training_util.MAX_BATCH_SIZE,
    )
    max_model_len = max(model_lengths)
    num_model = batch_generator.num_models
    boundary_batch_sizes = [
        adaptive_batch(max_model_len, num_model, int(boundary))
        for boundary in bucket_boundaries
    ]

    if boundary_batch_sizes:
        # Keep the last boundary for each repeated batch size.
        _, rev_idx = np.unique(boundary_batch_sizes[::-1], return_index=True)
        keep_idx = np.sort(len(boundary_batch_sizes) - 1 - rev_idx)
        bucket_boundaries = [bucket_boundaries[i] for i in keep_idx]
        boundary_batch_sizes = [boundary_batch_sizes[i] for i in keep_idx]

    bucket_batch_sizes = boundary_batch_sizes + [
        adaptive_batch(max_model_len, num_model, int(1e6))
    ]

    return bucket_boundaries, bucket_batch_sizes
