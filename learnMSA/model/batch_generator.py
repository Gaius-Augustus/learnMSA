"""Backend-neutral assembly of training batches from sequence indices.

:class:`BatchGenerator` turns a batch of sequence indices into padded numpy
arrays, applying the per-model index permutation and random cropping. It knows
nothing about tensor frameworks -- the framework-specific input pipeline
(``tf.data`` or a torch ``DataLoader``) wraps it and only has to adapt dtypes
and shapes.
"""

from typing import TYPE_CHECKING, Any, Sequence, cast

import numpy as np

from learnMSA.util.sequence_dataset import Dataset

if TYPE_CHECKING:
    from learnMSA.model.context import LearnMSAContext


class BatchGenerator():
    crop_long_seqs: float
    static_shape_mode: bool
    bucket_boundaries: Sequence[int] | None

    def __init__(
        self,
        return_only_sequences=False,
        shuffle=True,
        static_shape_mode=False,
    ) -> None:
        # generate a unique permutation of the sequence indices
        # for each model to train
        self.return_only_sequences = return_only_sequences
        self.shuffle = shuffle
        self.static_shape_mode = static_shape_mode
        self.bucket_boundaries = None
        self.configured = False

    def configure(
        self,
        data: Dataset | tuple[Dataset, ...],
        context: "LearnMSAContext",
    ):
        if isinstance(data, Dataset):
            data = (data,)
        self.data = data
        self.expected_shapes = tuple(d.empty(()).shape for d in self.data)
        self.context = context
        self.config = context.config
        self.num_models = self.config.training.num_model
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

        self.permutations = [
            np.arange(data[0].num_seq) for _ in range(self.num_models)
        ]
        for p in self.permutations:
            np.random.shuffle(p)
        self.configured = True

    def __call__(
        self, indices: np.ndarray
    ) -> tuple[np.ndarray, ...] | np.ndarray:
        if not self.configured:
            raise ValueError(
                "A batch generator must be configured with the "\
                "configure(data, config) method."
            )
        # Use a different permutation of the sequences per trained model
        if self.shuffle:
            permutated_indices = np.stack(
                [perm[indices] for perm in self.permutations], axis=1
            )
        else:
            permutated_indices = np.stack([indices]*self.num_models, axis=1)

        # Assume sequence lengths are identical across datasets.
        if self.static_shape_mode:
            max_len = min(self.data[0].max_len, int(self.crop_long_seqs)) + 1
        else:
            max_len = np.max(self.data[0].seq_lens[permutated_indices])
            max_len = min(max_len, self.crop_long_seqs) + 1

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
        batches = [
            dataset.empty(
                (indices.shape[0], max_len, self.num_models),
                dtype=cast(Any, dtype),
            )
            for dataset, dtype in zip(self.data, batch_dtypes)
        ]

        # Compute random crop bounds once per (batch item, model) and reuse
        # them for all datasets.
        crop_starts = np.zeros(
            (indices.shape[0], self.num_models),
            dtype=np.int32,
        )
        crop_ends = np.zeros(
            (indices.shape[0], self.num_models),
            dtype=np.int32,
        )
        for i, perm_ind in enumerate(permutated_indices):
            for k, j in enumerate(perm_ind):
                seq_len = int(self.data[0].seq_lens[j])
                if np.isfinite(self.crop_long_seqs):
                    crop_len = int(self.crop_long_seqs)
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
            if isinstance(batch_output, tuple):
                return *batch_output, permutated_indices
            return batch_output, permutated_indices

    def get_out_dtypes(self) -> tuple:
        """The numpy dtypes of the arrays returned by ``__call__``.

        The framework-specific pipeline converts these to its own dtypes.
        """
        batch_types = tuple(d.get_dtype() for d in self.data)
        if self.return_only_sequences:
            return batch_types
        else:
            return batch_types + (np.int64,)
