import math
import os
from datetime import datetime
from functools import partial
from typing import TYPE_CHECKING, Any, Sequence, cast

import numpy as np
import tensorflow as tf

import learnMSA.model.training_util as training_util
from learnMSA.util.embedding_dataset import EmbeddingDataset
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
            max_len = min(self.data[0].max_len, int(self.crop_long_seqs))
        else:
            max_len = np.max(self.data[0].seq_lens[permutated_indices])
            max_len = min(max_len, self.crop_long_seqs)

            # When JIT compiling with bucketing, pad to bucket boundary for
            # consistent shapes
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
                (indices.shape[0], max_len + 1, self.num_models),
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

    def get_out_types(self):
        batch_types = tuple(d.get_dtype() for d in self.data)
        if self.return_only_sequences:
            return batch_types
        else:
            return batch_types + (tf.int64,)

def make_dataset(
    indices: np.ndarray,
    batch_generator: BatchGenerator,
    batch_size:int = 512,
    shuffle:bool = True,
    bucket_by_seq_length:bool = False,
    bucket_boundaries: Sequence[int] = [],
    bucket_batch_sizes: Sequence[int] = [],
) -> tuple[tf.data.Dataset, int]:
    """
    Creates a dataset for training and inference.

    Args:
        indices: The indices of the sequences to include in the dataset.
        batch_generator: The batch generator that consumes sequence indices
            and produces batches.
        batch_size: The batch size to use.
        shuffle: Whether to shuffle the dataset.
        bucket_by_seq_length: Whether to use bucketing by sequence length.
        model_lengths: List of model lengths for adaptive batching.
        bucket_boundaries: Sequence length boundaries for bucketing.
        bucket_batch_sizes: Batch sizes for each bucket.

    Returns:
        A tuple of (dataset, steps) where steps is the number of steps needed
        to iterate through the entire dataset, or -1 for repeated (infinite)
        datasets.
    """
    def _to_tuple(output):
        if isinstance(output, tuple):
            return output
        return (output,)

    shuffle = shuffle and not bucket_by_seq_length
    batch_generator.shuffle = shuffle
    ds = tf.data.Dataset.from_tensor_slices(indices)
    if bucket_by_seq_length:
        if len(bucket_boundaries) == 0 or len(bucket_batch_sizes) == 0:
            raise ValueError(
                "bucket_boundaries and bucket_batch_sizes must be provided "
                "when bucket_by_seq_length=True."
            )

        # Bucketing only usable if user has not set a fixed batch size
        seq_lens = batch_generator.data[0].seq_lens[indices]
        ds_len = tf.data.Dataset.from_tensor_slices(
            seq_lens.astype(np.int32)
        )
        ds_ind =  tf.data.Dataset.from_tensor_slices(
            np.arange(indices.size)
        )
        ds = tf.data.Dataset.zip((ds, ds_len, ds_ind))
        _bucket_boundaries = list(bucket_boundaries)
        bucket_batch_sizes = list(bucket_batch_sizes)

        # Set bucket boundaries on batch generator for JIT-friendly padding
        batch_generator.bucket_boundaries = _bucket_boundaries

        # Compute steps for bucketed dataset
        total_steps = compute_dataset_steps(
            indices=indices,
            batch_generator=batch_generator,
            bucket_boundaries=_bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
        )

        ds = ds.bucket_by_sequence_length(
            element_length_func=cast(Any, lambda i, L, j: L),
            bucket_boundaries=_bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
            # when jit-compiling, make sure compilation only happens once
            # for each bucket
            pad_to_bucket_boundary=\
                batch_generator.context.config.advanced.jit_compile,
        )

        batch_func_out_types = batch_generator.get_out_types() + (tf.int64,)
        num_batch_outputs = len(batch_generator.data)

        def func(i, j):
            return *_to_tuple(batch_generator(i)), j

        def _bucket_batch_func(i,_,j):
            results = tf.numpy_function(
                func=func, inp=[i,j], Tout=batch_func_out_types
            )
            if not isinstance(results, (tuple, list)):
                results = (results,)

            batches = list(results[:num_batch_outputs])
            extras = list(results[num_batch_outputs:-1])
            j_out = results[-1]

            for batch, exp_shape in zip(batches, batch_generator.expected_shapes):
                batch.set_shape(tf.TensorShape(
                    [None, None, batch_generator.num_models] + list(exp_shape)
                ))

            if extras:
                extras[0].set_shape(
                    tf.TensorShape([None, batch_generator.num_models])
                )
            scoring_model_config = getattr(
                batch_generator, "scoring_model_config", None
            )
            if len(extras) > 1 and scoring_model_config is not None:
                extras[1].set_shape(tf.TensorShape([
                    None,
                    None,
                    batch_generator.num_models,
                    int(scoring_model_config.dim)+1
                ]))

            j_out.set_shape(tf.TensorShape([None]))
            return tuple(batches + extras + [j_out])

        map_func = _bucket_batch_func
    else:
        # Compute steps for non-bucketed dataset
        if shuffle:
            total_steps = -1  # Repeated dataset - infinite steps
        else:
            total_steps = int(np.ceil(indices.size / batch_size))

        if bucket_by_seq_length:
            ds_arange = tf.data.Dataset.from_tensor_slices(
                np.arange(indices.size)
            )
            ds = tf.data.Dataset.zip((ds, ds_arange))
        if shuffle:
            ds = ds.shuffle(indices.size, reshuffle_each_iteration=True)
            ds = ds.repeat()
        ds = ds.batch(batch_size)

        if batch_generator.static_shape_mode:
            seq_dims = [
                min(int(batch_generator.crop_long_seqs), dataset.max_len) + 1
                for dataset in batch_generator.data
            ]
        else:
            seq_dims = [None] * len(batch_generator.data)

        batch_generator.bucket_boundaries = None
        num_batch_outputs = len(batch_generator.data)

        def _batch_func(i):
            results = tf.numpy_function(
                batch_generator, [i], batch_generator.get_out_types()
            )

            if not isinstance(results, (tuple, list)):
                results = (results,)

            batches = list(results[:num_batch_outputs])
            extras = list(results[num_batch_outputs:])

            for batch, seq_dim, exp_shape in zip(
                batches, seq_dims, batch_generator.expected_shapes
            ):
                # explicitly set output shapes or tf 2.17 will complain about
                # unknown shapes
                batch.set_shape(tf.TensorShape(
                    [batch_size, seq_dim, batch_generator.num_models] + list(exp_shape)
                ))

            if extras:
                extras[0].set_shape(
                    tf.TensorShape([batch_size, batch_generator.num_models])
                )
            scoring_model_config = getattr(
                batch_generator, "scoring_model_config", None
            )
            if len(extras) > 1 and scoring_model_config is not None:
                extras[1].set_shape(tf.TensorShape([
                    batch_size,
                    seq_dims[0],
                    batch_generator.num_models,
                    int(scoring_model_config.dim)+1
                ]))

            return tuple(batches + extras)
        if bucket_by_seq_length:
            def batch_func(i,j):
                return *_batch_func(i), j
        else:
            def batch_func(i):
                return _batch_func(i)

        map_func = batch_func


    ds = ds.map(
        map_func,
        # no parallel processing if using an indexed dataset
        num_parallel_calls=None if batch_generator.data[0].indexed else tf.data.AUTOTUNE,
        deterministic=True
    )
    if not batch_generator.data[0].indexed:
        ds = ds.prefetch(2) #preprocessings and training steps in parallel
    # get rid of a warning, see https://github.com/tensorflow/tensorflow/issues/42146
    # in case of multi GPU, we want to split the data dimension accross GPUs
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    ds = ds.with_options(options)
    ds_y = tf.data.Dataset.from_tensor_slices(tf.zeros(1)).batch(batch_size).repeat()
    ds = tf.data.Dataset.zip((ds, ds_y))
    return ds, total_steps

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
) -> tuple[list[int], list[int]]:
    """
    Create inferred bucketing boundaries and adaptive bucket batch sizes.

    Args:
        indices: The indices of the sequences to include in the dataset.
        batch_generator: The batch generator (must be configured).
        model_lengths: List of model lengths for adaptive batching.
        batch_size_impl_factor: Implementation factor passed to adaptive
            batch-size computation.

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
    bucket_boundaries = np.unique(quantiles).tolist()

    # pad_to_bucket_boundary=True requires every sequence length to be
    # strictly smaller than max(bucket_boundaries).
    max_seq_len = int(np.max(seq_lens))
    if len(bucket_boundaries) == 0 or bucket_boundaries[-1] <= max_seq_len:
        bucket_boundaries.append(max_seq_len + 1)

    adaptive_batch = partial(
        training_util.get_adaptive_batch_size,
        impl_factor=batch_size_impl_factor,
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


class TerminateOnNaNWithCheckpoint(tf.keras.callbacks.TerminateOnNaN):
    """Callback that terminates training when a NaN loss is encountered and
    saves a model checkpoint for debugging.
    """

    def __init__(self, model: "tf.keras.Model", work_dir: str):
        super().__init__()
        self.learnmsa_model = model
        self.work_dir = work_dir

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")
        if loss is not None:
            if np.isnan(loss) or np.isinf(loss):
                # Save checkpoint before terminating
                os.makedirs(self.work_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_path = os.path.join(
                    self.work_dir, f"nan_checkpoint_{timestamp}.keras"
                )
                try:
                    self.learnmsa_model.save(checkpoint_path)
                    print(
                        f"\nNaN detected in loss. Model checkpoint saved to: "
                        f"{checkpoint_path}"
                    )
                except Exception as e:
                    print(
                        f"\nNaN detected but failed to save checkpoint: {e}"
                    )
        # Call parent to terminate training
        super().on_batch_end(batch, logs)
