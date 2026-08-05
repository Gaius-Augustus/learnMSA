"""TensorFlow input pipeline and training callbacks.

Wraps the backend-neutral :class:`~learnMSA.model.batch_generator.BatchGenerator`
in a ``tf.data`` pipeline. The batching logic itself lives in
:mod:`learnMSA.model.batch_generator` and the bucketing scheme in
:mod:`learnMSA.model.bucketing`; only the framework wiring is here.
"""

import os
from datetime import datetime
from typing import Any, Sequence, cast

import numpy as np
import tensorflow as tf

from learnMSA.model.batch_generator import BatchGenerator
from learnMSA.model.bucketing import compute_dataset_steps


def _tf_out_types(batch_generator: BatchGenerator) -> tuple:
    """The ``tf`` dtypes of the arrays the batch generator returns."""
    return tuple(
        tf.as_dtype(dtype) for dtype in batch_generator.get_out_dtypes()
    )


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
            # when compiling, make sure compilation only happens once
            # for each bucket
            pad_to_bucket_boundary=\
                batch_generator.context.config.advanced.compile != "off",
        )

        batch_func_out_types = _tf_out_types(batch_generator) + (tf.int64,)
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
                batch_generator, [i], _tf_out_types(batch_generator)
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
