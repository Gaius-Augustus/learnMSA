import math
from collections.abc import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from learnMSA.model.batch_generator import BatchGenerator
from learnMSA.model.bucketing import compute_dataset_steps


class _PositionDataset(Dataset):
    """Yields positions into the caller's index array.

    All the real work happens in the collate function, which needs the whole
    batch of positions at once -- the batch generator reads and pads a batch as
    a unit.
    """

    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, position: int) -> int:
        return position


class _RepeatingShuffleSampler(Sampler[list[int]]):
    """Endlessly reshuffles the positions and hands out fixed-size batches.
    """

    def __init__(self, size: int, batch_size: int) -> None:
        self.size = size
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[list[int]]:
        carry: list[int] = []
        while True:
            order = carry + np.random.permutation(self.size).tolist()
            n_full = len(order) // self.batch_size
            for i in range(n_full):
                yield order[i * self.batch_size:(i + 1) * self.batch_size]
            # short of a batch: hand these to the head of the next permutation
            carry = order[n_full * self.batch_size:]

    def __len__(self) -> int:
        return math.ceil(self.size / self.batch_size)


class _SequentialBatchSampler(Sampler[list[int]]):
    """Hands out the positions in order, in fixed-size batches."""

    def __init__(self, size: int, batch_size: int) -> None:
        self.size = size
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[list[int]]:
        for start in range(0, self.size, self.batch_size):
            yield list(range(start, min(start + self.batch_size, self.size)))

    def __len__(self) -> int:
        return math.ceil(self.size / self.batch_size)


class _BucketBatchSampler(Sampler[list[int]]):
    """Groups positions by sequence length before batching them.

    Rules:

    - bucket 0: ``length < boundaries[0]``
    - bucket i: ``boundaries[i-1] <= length < boundaries[i]``
    - last bucket: ``length >= boundaries[-1]``

    Each bucket is emitted in ``bucket_batch_sizes[i]``-sized batches.
    """

    def __init__(
        self,
        sequence_lengths: np.ndarray,
        bucket_boundaries: Sequence[int | float],
        bucket_batch_sizes: Sequence[int],
    ) -> None:
        if len(bucket_boundaries) == 0 or len(bucket_batch_sizes) == 0:
            raise ValueError(
                "bucket_boundaries and bucket_batch_sizes must be provided "
                "when bucket_by_seq_length=True."
            )
        if len(bucket_batch_sizes) != len(bucket_boundaries) + 1:
            raise ValueError(
                "bucket_batch_sizes must have exactly one more entry than "
                f"bucket_boundaries, got {len(bucket_batch_sizes)} and "
                f"{len(bucket_boundaries)}."
            )
        # np.searchsorted with side="right" maps a length to the index of the
        # first boundary strictly greater than it, which is TensorFlow's rule.
        self.bucket_of = np.searchsorted(
            np.asarray(bucket_boundaries, dtype=np.float64),
            np.asarray(sequence_lengths, dtype=np.float64),
            side="right",
        )
        self.bucket_batch_sizes = list(bucket_batch_sizes)
        self.num_buckets = len(bucket_batch_sizes)

    def __iter__(self) -> Iterator[list[int]]:
        for bucket in range(self.num_buckets):
            positions = np.flatnonzero(self.bucket_of == bucket)
            batch_size = self.bucket_batch_sizes[bucket]
            for start in range(0, positions.size, batch_size):
                yield positions[start:start + batch_size].tolist()

    def __len__(self) -> int:
        total = 0
        for bucket in range(self.num_buckets):
            count = int(np.count_nonzero(self.bucket_of == bucket))
            if count:
                total += math.ceil(count / self.bucket_batch_sizes[bucket])
        return total


def _seed_worker(worker_id: int) -> None:
    """Without this all workers would share state any might produce
    identical batches.
    """
    np.random.seed(torch.initial_seed() % 2**32)


def _make_collate(
    indices: np.ndarray,
    batch_generator: BatchGenerator,
    with_position: bool,
):
    """Builds the collate function that runs the batch generator."""

    def collate(positions: list[int]) -> tuple[torch.Tensor, ...]:
        position_array = np.asarray(positions)
        output = batch_generator(indices[position_array])
        arrays = output if isinstance(output, tuple) else (output,)
        tensors = tuple(torch.as_tensor(np.asarray(a)) for a in arrays)
        if with_position:
            tensors += (torch.as_tensor(position_array),)
        return tensors

    return collate


def make_dataset(
    indices: np.ndarray,
    batch_generator: BatchGenerator,
    batch_size: int = 512,
    shuffle: bool = True,
    bucket_by_seq_length: bool = False,
    bucket_boundaries: Sequence[int] = [],
    bucket_batch_sizes: Sequence[int] = [],
) -> tuple[DataLoader, int]:
    """
    Creates a data loader for training and inference.

    Args:
        indices: The indices of the sequences to include in the dataset.
        batch_generator: The batch generator that consumes sequence indices
            and produces batches.
        batch_size: The batch size to use. Ignored when bucketing, which takes
            its sizes from ``bucket_batch_sizes``.
        shuffle: Whether to shuffle the dataset.
        bucket_by_seq_length: Whether to use bucketing by sequence length.
        bucket_boundaries: Sequence length boundaries for bucketing.
        bucket_batch_sizes: Batch sizes for each bucket.

    Returns:
        A tuple of (loader, steps) where steps is the number of steps needed
        to iterate through the entire dataset, or -1 for a repeating
        (infinite) loader.
    """
    shuffle = shuffle and not bucket_by_seq_length
    batch_generator.shuffle = shuffle

    sampler: Sampler[list[int]]
    if bucket_by_seq_length:
        boundaries = list(bucket_boundaries)
        # Padding to a bucket boundary keeps the shapes within a bucket
        # identical, which is what makes the batches interchangeable.
        batch_generator.bucket_boundaries = boundaries
        sequence_lengths = batch_generator.data[0].seq_lens[indices]
        sampler = _BucketBatchSampler(
            sequence_lengths, boundaries, list(bucket_batch_sizes)
        )
        total_steps = compute_dataset_steps(
            indices=indices,
            batch_generator=batch_generator,
            bucket_boundaries=boundaries,
            bucket_batch_sizes=list(bucket_batch_sizes),
        )
    else:
        batch_generator.bucket_boundaries = None
        if shuffle:
            sampler = _RepeatingShuffleSampler(indices.size, batch_size)
            total_steps = -1  # Repeated dataset - infinite steps
        else:
            sampler = _SequentialBatchSampler(indices.size, batch_size)
            total_steps = int(np.ceil(indices.size / batch_size))

    # mutable state -- configure(), static_shape_mode, crop_long_seqs,
    # bucket_boundaries, num_models -- must be set BEFORE creating the
    # interator, otherwise forking across multiple workers is not safe
    # normal pipeline works; indexed datasets do not work
    num_workers = 0 if batch_generator.data[0].indexed else 2

    loader = DataLoader(
        _PositionDataset(indices.size),
        batch_sampler=sampler,
        collate_fn=_make_collate(
            indices, batch_generator, with_position=bucket_by_seq_length
        ),
        num_workers=num_workers,
        worker_init_fn=_seed_worker if num_workers else None,
        prefetch_factor=2 if num_workers else None,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, total_steps
