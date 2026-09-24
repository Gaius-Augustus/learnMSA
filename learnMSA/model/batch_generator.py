from typing import TYPE_CHECKING, Any, Sequence, cast

import numpy as np

from learnMSA.util.multi_dataset import MultiDataset, MultiSequenceDataset
from learnMSA.util.sequence_dataset import Dataset

if TYPE_CHECKING:
    from learnMSA.model.context import LearnMSAContext


def get_lengths(seq_lens: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """The length of each row of an index array.

    Args:
        seq_lens: The sequence lengths of the dataset.
        indices: Either 1-D, one sequence per row, or 2-D ``(N, H)`` with one
            sequence per model and ``-1`` for an empty cell.

    Returns:
        Shape ``(N,)``. For 2-D indices, the longest sequence of the row.
    """
    if indices.ndim == 1:
        return seq_lens[indices]
    lens = np.where(indices >= 0, seq_lens[np.maximum(indices, 0)], 0)
    return lens.max(axis=1)


def get_index_table(indices: Sequence[np.ndarray]) -> np.ndarray:
    """Stacks one index array per model into a 2-D index table. Arrays shorter
    than the longest one are padded with ``-1``.

    Args:
        indices: One index array per model.

    Returns:
        The index table of shape ``(max len(indices), H)``.
    """
    num_rows = max(len(i) for i in indices)
    table = np.full((num_rows, len(indices)), -1, dtype=np.int64)
    for k, model_indices in enumerate(indices):
        table[:len(model_indices), k] = model_indices
    return table


def sort_index_table(
    table: np.ndarray,
    seq_lens: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sorts every column of an index table by decreasing sequence length.

    This pairs sequences of similar length in a row and keeps the padding
    small. Empty cells (-1) stay at the end of their column.

    Args:
        table: An index table of shape ``(N, H)``, see
            :func:`get_index_table`.
        seq_lens: The sequence lengths of the dataset.

    Returns:
        ``(sorted_table, positions)``, both of shape ``(N, H)``.
        ``positions[r, k]`` is the row of ``sorted_table[r, k]`` in ``table``
        (for a table of :func:`get_index_table`, its position in
        ``indices[k]``), or -1 for empty cells.
    """
    lens = np.where(table >= 0, seq_lens[np.maximum(table, 0)], -1)
    order = np.argsort(-lens, axis=0, kind="stable")
    sorted_table = np.take_along_axis(table, order, axis=0)
    positions = np.where(sorted_table >= 0, order, -1)
    return sorted_table, positions


class BatchGenerator():

    """ Builds batches for training and prediction from sequence indices."""
    crop_long_seqs: float
    static_shape_mode: bool
    bucket_boundaries: Sequence[int] | None

    def __init__(
        self,
        return_only_sequences=False,
        shuffle=True,
        static_shape_mode=False,
    ) -> None:
        self.return_only_sequences = return_only_sequences
        self.shuffle = shuffle
        self.static_shape_mode = static_shape_mode
        self.bucket_boundaries = None
        self.share_batch = False
        self.configured = False
        # Position ranges within the configured indices that the columns
        # sample from, one per column (see MultiBatchGenerator). None means
        # every column samples from all indices.
        self.groups: list[tuple[int, int]] | None = None

    def configure(
        self,
        data: Dataset | tuple[Dataset, ...],
        context: "LearnMSAContext",
        indices: np.ndarray | None = None,
    ):
        """
        Args:
            data: The dataset(s) with the sequences to sample from.
            context: LearnMSAContext object with the configuration.
            indices: The sequence indices that will be used to generate batches.
        """
        if isinstance(data, Dataset):
            data = (data,)
        self.data = data
        self.expected_shapes = tuple(d.empty(()).shape for d in self.data)
        self.context = context
        self.config = context.config
        self.num_models = self.config.training.num_model
        self.share_batch = self.config.training.share_batch
        self.crop_long_seqs = float(self.config.training.crop)

        # Validate crop_long_seqs in static shape mode
        if self.static_shape_mode:
            if not float(self.crop_long_seqs).is_integer():
                raise ValueError(
                    f"static_shape_mode requires crop_long_seqs to be an "
                    f"integer, got {type(self.crop_long_seqs).__name__}: "
                    f"{self.crop_long_seqs}"
                )
            if self.crop_long_seqs <= 0:
                raise ValueError(
                    f"static_shape_mode requires crop_long_seqs to be "
                    f"positive, got {self.crop_long_seqs}"
                )
            if not np.isfinite(self.crop_long_seqs):
                raise ValueError(
                    "static_shape_mode requires a finite crop_long_seqs value"
                )

        self.permutations = self._make_permutations(indices)
        self.configured = True

    def _make_permutations(
        self, indices: np.ndarray | None
    ) -> list[np.ndarray]:
        """One permutation per emitted model column: one per trained model
        normally, a single shared one when the batch is shared."""
        num_seq = self.data[0].num_seq
        if indices is None:
            idx = np.arange(num_seq)
        else:
            idx = np.unique(indices)
        permutations = []
        for _ in range(self.generated_num_models):
            # permutation of a subset still needs to be in a full-length array
            p = np.arange(num_seq)
            p[idx] = np.random.permutation(idx)
            permutations.append(p)
        return permutations

    def _columns(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The sequence index of every batch cell and the crop length of
        every model column.

        Returns:
            ``(column_indices, crops)`` of shapes ``(B, C)`` and ``(C,)``.
        """
        # Use a different permutation of the sequences per trained model,
        # unless a single sample is shared across all of them.
        num_gen = self.generated_num_models
        if self.shuffle:
            column_indices = np.stack(
                [perm[indices] for perm in self.permutations[:num_gen]],
                axis=1,
            )
        else:
            column_indices = np.stack([indices]*num_gen, axis=1)
        crops = np.full(column_indices.shape[1], float(self.crop_long_seqs))
        return column_indices, crops

    def _column_max_lens(self, num_columns: int) -> np.ndarray:
        """The longest sequence each model column can receive, for the static
        shape mode."""
        return np.full(num_columns, self.data[0].max_len)

    @property
    def generated_num_models(self) -> int:
        """The size of the model axis of the arrays returned by ``__call__``.
        """
        return 1 if self.share_batch else self.num_models

    def __call__(
        self, indices: np.ndarray
    ) -> tuple[np.ndarray, ...] | np.ndarray:
        if not self.configured:
            raise ValueError(
                "A batch generator must be configured with the "\
                "configure(data, config) method."
            )
        permutated_indices, crops = self._columns(indices)
        num_gen = permutated_indices.shape[1]
        # Empty cells (-1) hold a zero-length sequence.
        safe_indices = np.maximum(permutated_indices, 0)
        seq_lens = np.where(
            permutated_indices < 0, 0, self.data[0].seq_lens[safe_indices]
        )

        # Assume sequence lengths are identical across datasets.
        if self.static_shape_mode:
            max_len = int(np.max(np.minimum(
                self._column_max_lens(num_gen), crops
            ))) + 1
        else:
            max_len = np.max(np.minimum(seq_lens, crops)) + 1

            # Pad to bucket boundary for consistent shapes (avoids retracing).
            # TF places seq_len into bucket i where boundary[i-1] <= seq_len
            # < boundary[i], so max_raw <= boundary[i]-1 and
            # max_len = max_raw+1 <= boundary[i].  Using <= here ensures every
            # batch in TF bucket i gets the same padded_len = boundary[i].
            if self.bucket_boundaries is not None:
                # Find which bucket this batch belongs to
                for boundary in self.bucket_boundaries:
                    if max_len <= boundary:
                        max_len = boundary
                        break

        max_len = int(max_len)

        batch_dtypes = [dataset.get_dtype() for dataset in self.data]
        batch_size = permutated_indices.shape[0]
        batches = [
            dataset.empty(
                (batch_size, max_len, num_gen),
                dtype=cast(Any, dtype),
            )
            for dataset, dtype in zip(self.data, batch_dtypes)
        ]

        # Compute random crop bounds once per (batch item, model column) and
        # reuse them for all datasets. A shared batch has a single column, so
        # every model then sees the same crop window of a long sequence.
        crop_starts = np.zeros((batch_size, num_gen), dtype=np.int32)
        crop_ends = np.zeros((batch_size, num_gen), dtype=np.int32)
        for i, perm_ind in enumerate(permutated_indices):
            for k, j in enumerate(perm_ind):
                seq_len = int(seq_lens[i, k])
                if np.isfinite(crops[k]):
                    crop_len = int(crops[k])
                    if seq_len > crop_len:
                        crop_start = np.random.randint(
                            0,
                            seq_len - crop_len + 1,
                        )
                        crop_end = crop_start + crop_len
                    else:
                        crop_start = 0
                        crop_end = seq_len
                else:
                    crop_start = 0
                    crop_end = seq_len

                crop_starts[i, k] = crop_start
                crop_ends[i, k] = crop_end

        for i,perm_ind in enumerate(permutated_indices):
            for k,j in enumerate(perm_ind):
                if j < 0:
                    continue
                crop_start = crop_starts[i, k]
                crop_end = crop_ends[i, k]
                for d, dataset in enumerate(self.data):
                    seq = dataset.get_encoded_seq(j, crop_start, crop_end)
                    batches[d][i, :seq.shape[0], k] = seq

        if len(batches) == 1:
            batch_output: tuple[np.ndarray, ...] | np.ndarray = batches[0]
        else:
            batch_output = tuple(batches)

        if self.return_only_sequences:
            return batch_output
        else:
            # Empty cells report index 0; their outputs are discarded.
            if isinstance(batch_output, tuple):
                return *batch_output, safe_indices
            return batch_output, safe_indices

    def get_out_dtypes(self) -> tuple:
        """The numpy dtypes of the arrays returned by ``__call__``.

        The framework-specific pipeline converts these to its own dtypes.
        """
        batch_types = tuple(d.get_dtype() for d in self.data)
        if self.return_only_sequences:
            return batch_types
        else:
            return batch_types + (np.int64,)


class MultiBatchGenerator(BatchGenerator):
    """Builds batches for training and prediction from multiple datasets, by
    filling each head dimension with sequences from one dataset. Must be
    configured with a :class:`MultiSequenceDataset`.

    - Training batches come from a sampler that draws the rows of each
      column from :attr:`groups`, the position range of that dataset
      within the configured indices.
    - Prediction takes rows of an index table built by
      :func:`get_index_table`.

    Each dataset is cropped to its own length (``context.head_crops``) as long
    as ``crop_long_seqs`` is finite; setting it to inf disables cropping, as
    prediction does. 1-D indices give every column the same sequences, like
    the base class without shuffling.
    """

    def configure(
        self,
        data: Dataset | tuple[Dataset, ...],
        context: "LearnMSAContext",
        indices: np.ndarray | None = None,
    ):
        """
        Args:
            data: A MultiSequenceDataset, optionally followed by MultiDataset
                auxiliary tracks with the same parts.
            context: LearnMSAContext object with the configuration.
            indices: The global sequence indices that will be used to
                generate training batches. They must be grouped by dataset
                (sorted indices are) and cover every dataset.
        """
        if isinstance(data, Dataset):
            data = (data,)
        if not isinstance(data[0], MultiSequenceDataset):
            raise ValueError(
                "A MultiBatchGenerator must be configured with a "
                "MultiSequenceDataset."
            )
        multi = data[0]
        for aux in data[1:]:
            if not isinstance(aux, MultiDataset) \
                    or not np.array_equal(aux.offsets, multi.offsets):
                raise ValueError(
                    "Auxiliary tracks must be MultiDatasets with the same "
                    "parts as the MultiSequenceDataset."
                )
        super().configure(data, context, indices)
        if self.num_models != multi.num_datasets:
            raise ValueError(
                f"Expected one model per dataset ({multi.num_datasets}), "
                f"got {self.num_models}."
            )
        if context.head_crops is not None:
            self.head_crops = np.asarray(context.head_crops, dtype=float)
        else:
            self.head_crops = np.full(multi.num_datasets, self.crop_long_seqs)
        self.groups = self._make_groups(multi, indices)

    def _make_groups(
        self,
        multi: MultiSequenceDataset,
        indices: np.ndarray | None,
    ) -> list[tuple[int, int]]:
        if indices is None:
            indices = np.arange(multi.num_seq)
        group = multi.dataset_of(np.asarray(indices))
        if np.any(np.diff(group) < 0):
            raise ValueError("The indices must be grouped by dataset.")
        bounds = np.searchsorted(group, np.arange(multi.num_datasets + 1))
        if np.any(np.diff(bounds) == 0):
            raise ValueError("Every dataset needs at least one index.")
        return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])]

    def _make_permutations(
        self, indices: np.ndarray | None
    ) -> list[np.ndarray]:
        # The sampler chooses the sequences of each column.
        return []

    @property
    def generated_num_models(self) -> int:
        # Every model has its own dataset, so there is no shared batch.
        return self.num_models

    def _columns(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if indices.ndim == 1:
            column_indices = np.stack(
                [indices] * self.generated_num_models, axis=1
            )
            crops = np.full(column_indices.shape[1], float(self.crop_long_seqs))
            return column_indices, crops
        if np.isinf(self.crop_long_seqs):
            crops = np.full(indices.shape[1], np.inf)
        else:
            assert indices.shape[1] == self.head_crops.size, \
                "Per-dataset crops need one column per dataset."
            crops = np.minimum(self.head_crops, float(self.crop_long_seqs))
        return indices, crops

    def _column_max_lens(self, num_columns: int) -> np.ndarray:
        multi = self.data[0]
        assert isinstance(multi, MultiSequenceDataset)
        if num_columns != multi.num_datasets:
            return super()._column_max_lens(num_columns)
        return np.array([d.max_len for d in multi.datasets])
