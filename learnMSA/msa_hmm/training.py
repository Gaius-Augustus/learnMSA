import math
from typing import TYPE_CHECKING, Sequence

import numpy as np
import tensorflow as tf
from functools import partial

import learnMSA.msa_hmm.training_util as training_util
from learnMSA.util.sequence_dataset import SequenceDataset

if TYPE_CHECKING:
    from learnMSA.msa_hmm.learnmsa_context import LearnMSAContext


class BatchGenerator():
    crop_long_seqs: float

    def __init__(
        self,
        return_only_sequences=False,
        shuffle=True,
    ) -> None:
        #generate a unique permutation of the sequence indices for each model to train
        self.return_only_sequences = return_only_sequences
        self.shuffle = shuffle
        self.configured = False

    def configure(
        self,
        data : SequenceDataset,
        context: "LearnMSAContext",
    ):
        self.data = data
        self.alphabet_size = len(data.alphabet)-1
        self.context = context
        self.config = context.config
        self.num_models = self.config.training.num_model
        self.crop_long_seqs = self.config.training.crop
        self.permutations = [np.arange(data.num_seq) for _ in range(self.num_models)]
        for p in self.permutations:
            np.random.shuffle(p)
        self.configured = True

    def __call__(self, indices, return_crop_boundaries=False):
        if not self.configured:
            raise ValueError(
                "A batch generator must be configured with the "\
                "configure(data, config) method."
            )
        # Use a different permutation of the sequences per trained model
        if self.shuffle:
            permutated_indices = np.stack([perm[indices] for perm in self.permutations], axis=1)
        else:
            permutated_indices = np.stack([indices]*self.num_models, axis=1)
        max_len = np.max(self.data.seq_lens[permutated_indices])
        max_len = min(max_len, self.crop_long_seqs)
        batch = np.zeros((indices.shape[0], self.num_models, max_len+1), dtype=np.uint8)
        if return_crop_boundaries:
            start = np.zeros((indices.shape[0], self.num_models), dtype=np.int32)
            end = np.zeros((indices.shape[0], self.num_models), dtype=np.int32)
        batch += self.alphabet_size # Initialize with terminal symbols
        for i,perm_ind in enumerate(permutated_indices):
            for k,j in enumerate(perm_ind):
                if self.data.alphabet == SequenceDataset._default_alphabet:
                    # Replace rare amino acids with X if the amino acid
                    # alphabet if used
                    replace_with_x = "BZJ"
                else:
                    replace_with_x = ""
                if return_crop_boundaries:
                    seq, start[i, k], end[i, k] = self.data.get_encoded_seq(
                        j,
                        crop_to_length=self.crop_long_seqs,
                        return_crop_boundaries=True,
                        replace_with_x=replace_with_x,
                    )
                else:
                    seq = self.data.get_encoded_seq(
                        j,
                        crop_to_length=self.crop_long_seqs,
                        return_crop_boundaries=False,
                        replace_with_x=replace_with_x,
                    )
                batch[i, k, :min(self.data.seq_lens[j], self.crop_long_seqs)] = seq
        if self.return_only_sequences:
            if return_crop_boundaries:
                return batch, start, end
            else:
                return batch
        else:
            if return_crop_boundaries:
                return batch, permutated_indices, start, end
            else:
                return batch, permutated_indices

    def get_out_types(self):
        if self.return_only_sequences:
            return (tf.uint8, )
        else:
            return (tf.uint8, tf.int64)

def make_dataset(
    indices: np.ndarray,
    batch_generator: BatchGenerator,
    batch_size:int = 512,
    shuffle:bool = True,
    bucket_by_seq_length:bool = False,
    model_lengths:Sequence[int] = [0],
    bucket_boundaries: Sequence[int | float] | None = None,
    bucket_batch_sizes: Sequence[int] | None = None,
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
        bucket_boundaries: Sequence length boundaries for bucketing. If None,
            uses default boundaries [200, 520, 700, 850, 1200, 2000, 4000, inf].
        bucket_batch_sizes: Batch sizes for each bucket. If None, uses the
            adaptive batch size function from the batch generator.

    Returns:
        A tuple of (dataset, steps) where steps is the number of steps needed
        to iterate through the entire dataset, or -1 for repeated (infinite) datasets.
    """
    shuffle = shuffle and not bucket_by_seq_length
    batch_generator.shuffle = shuffle
    ds = tf.data.Dataset.from_tensor_slices(indices)
    # TODO: clean up
    # we can not use the context's adaptive batch, because it takes in a
    # dataset, here we only have the sequence lengths as a numpy array
    if batch_generator.context.config.language_model.use_language_model:
        adaptive_batch = partial(
            training_util.get_adaptive_batch_size_with_language_model,
            embedding_dim=batch_generator.context.config.language_model.scoring_model_dim,
            small_gpu=batch_generator.context.small_gpu
        )
    else:
        adaptive_batch = partial(
            training_util.get_adaptive_batch_size,
            small_gpu=batch_generator.context.small_gpu
        )
    if bucket_by_seq_length and callable(adaptive_batch):
        # Bucketing only usable if user has not set a fixed batch size
        ds_len = tf.data.Dataset.from_tensor_slices(
            batch_generator.data.seq_lens[indices].astype(np.int32)
        )
        ds_ind =  tf.data.Dataset.from_tensor_slices(
            np.arange(indices.size)
        )
        ds = tf.data.Dataset.zip((ds, ds_len, ds_ind))

        # Use provided bucket boundaries or defaults
        if bucket_boundaries is None:
            bucket_boundaries = [200, 520, 700, 850, 1200, 2000, 4000]
        if bucket_batch_sizes is None:
            bucket_batch_sizes = [
                adaptive_batch(model_lengths, b) for b in bucket_boundaries+[math.inf]
            ]

        # Compute steps for bucketed dataset
        total_steps = compute_dataset_steps(
            indices=indices,
            batch_generator=batch_generator,
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
        )

        ds = ds.bucket_by_sequence_length(
            element_length_func=lambda i,L,j: L,
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes
        )

        batch_func_out_types = batch_generator.get_out_types() + (tf.int64,)
        if len(batch_func_out_types) == 2:
            # return_only_sequences=True case: (batch, j)
            func = (lambda i,j: (batch_generator(i), j))
        else:
            # return_only_sequences=False case: (batch, indices, j)
            func = lambda i,j: (*batch_generator(i), j)

        def batch_func(i,_,j):
            results = tf.numpy_function(
                func=func, inp=[i,j], Tout=batch_func_out_types
            )
            # Set shapes explicitly for TensorFlow 2.17+
            if len(batch_func_out_types) == 2:
                batch, j_out = results
                batch.set_shape(
                    tf.TensorShape([None, batch_generator.num_models, None])
                )
                j_out.set_shape(tf.TensorShape([None]))
                return batch, j_out
            else:
                batch, ind, j_out = results
                batch.set_shape(
                    tf.TensorShape([None, batch_generator.num_models, None])
                )
                ind.set_shape(
                    tf.TensorShape([None, batch_generator.num_models])
                )
                j_out.set_shape(tf.TensorShape([None]))
                return batch, ind, j_out
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
        def _batch_func(i):
            if len(batch_generator.get_out_types()) == 2:
                batch, ind = tf.numpy_function(
                    batch_generator, [i], batch_generator.get_out_types()
                )
                # explicitly set output shapes or tf 2.17 will complain about
                # unknown shapes
                batch.set_shape(
                    tf.TensorShape([None, batch_generator.num_models, None])
                )
                ind.set_shape(
                    tf.TensorShape([None, batch_generator.num_models])
                )
                return batch, ind
            else:
                batch, ind, emb = tf.numpy_function(
                    batch_generator, [i], batch_generator.get_out_types()
                )
                # explicitly set output shapes or tf 2.17 will complain about
                # unknown shapes
                batch.set_shape(
                    tf.TensorShape([None, batch_generator.num_models, None])
                )
                ind.set_shape(
                    tf.TensorShape([None, batch_generator.num_models])
                )
                emb.set_shape(tf.TensorShape([
                    None,
                    batch_generator.num_models,
                    None,
                    batch_generator.scoring_model_config.dim+1
                ]))
                return batch, ind, emb
        if bucket_by_seq_length:
            def batch_func(i,j):
                return *_batch_func(i), j
        else:
            batch_func = _batch_func


    ds = ds.map(batch_func,
                # no parallel processing if using an indexed dataset
                num_parallel_calls=None if batch_generator.data.indexed else tf.data.AUTOTUNE,
                deterministic=True)
    if not batch_generator.data.indexed:
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
    seq_lengths = batch_generator.data.seq_lens[indices]
    total_steps = 0
    boundaries = list(bucket_boundaries) + [math.inf]

    for i, (lower, upper, bsize) in enumerate(
        zip([0] + boundaries[:-1], boundaries, bucket_batch_sizes)
    ):
        # Count sequences in this bucket
        if i == 0:
            count = np.sum(seq_lengths <= upper)
        else:
            count = np.sum((seq_lengths > lower) & (seq_lengths <= upper))

        # Compute number of batches for this bucket
        if count > 0:
            total_steps += int(np.ceil(count / bsize))

    return total_steps
