import math
from typing import TYPE_CHECKING

import numpy as np
import tensorflow as tf

from learnMSA.msa_hmm.SequenceDataset import SequenceDataset

if TYPE_CHECKING:
    from learnMSA.msa_hmm.learnmsa_context import LearnMSAContext


class BatchGenerator():
    def __init__(
        self,
        return_only_sequences=False,
        shuffle=True,
        alphabet_size=len(SequenceDataset.alphabet)-1
    ) -> None:
        #generate a unique permutation of the sequence indices for each model to train
        self.return_only_sequences = return_only_sequences
        self.alphabet_size = alphabet_size
        self.shuffle = shuffle
        self.configured = False

    def configure(
        self,
        data : SequenceDataset,
        context: "LearnMSAContext",
    ):
        self.data = data
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
        #use a different permutation of the sequences per trained model
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
        batch += self.alphabet_size #initialize with terminal symbols
        for i,perm_ind in enumerate(permutated_indices):
            for k,j in enumerate(perm_ind):
                if return_crop_boundaries:
                    seq, start[i, k], end[i, k] = self.data.get_encoded_seq(j, crop_to_length=self.crop_long_seqs, return_crop_boundaries=True)
                else:
                    seq = self.data.get_encoded_seq(j, crop_to_length=self.crop_long_seqs, return_crop_boundaries=False)
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


# batch_generator is a callable object that maps a vector of sequence indices to
# inputs compatible with the model
def make_dataset(indices, batch_generator, batch_size=512, shuffle=True, bucket_by_seq_length=False, model_lengths=[0]):
    shuffle = shuffle and not bucket_by_seq_length
    batch_generator.shuffle = shuffle
    ds = tf.data.Dataset.from_tensor_slices(indices)
    adaptive_batch = batch_generator.config.training.batch_size
    if bucket_by_seq_length and callable(adaptive_batch): #bucketing only usable if user has not set a fixed batch size
        ds_len = tf.data.Dataset.from_tensor_slices(batch_generator.data.seq_lens[indices].astype(np.int32))
        ds_ind =  tf.data.Dataset.from_tensor_slices(np.arange(indices.size))
        ds = tf.data.Dataset.zip((ds, ds_len, ds_ind))
        bucket_boundaries = [200, 520, 700, 850, 1200, 2000, 4000, math.inf]
        bucket_batch_sizes = [adaptive_batch(model_lengths, b) for b in bucket_boundaries]
        ds = ds.bucket_by_sequence_length(
                                element_length_func=lambda i,L,j: L,
                                bucket_boundaries=bucket_boundaries[:-1],
                                bucket_batch_sizes=bucket_batch_sizes)

        batch_func_out_types = batch_generator.get_out_types() + (tf.int64,)
        func = (lambda i,j: (batch_generator(i), j)) if len(batch_func_out_types) == 2 else lambda i,j: (*batch_generator(i), j)
        batch_func = lambda i,_,j: tf.numpy_function(func=func, inp=[i,j], Tout=batch_func_out_types)
    else:
        if bucket_by_seq_length:
            ds_arange = tf.data.Dataset.from_tensor_slices(np.arange(indices.size))
            ds = tf.data.Dataset.zip((ds, ds_arange))
        if shuffle:
            ds = ds.shuffle(indices.size, reshuffle_each_iteration=True)
            ds = ds.repeat()
        ds = ds.batch(batch_size)
        def _batch_func(i):
            if len(batch_generator.get_out_types()) == 2:
                batch, ind = tf.numpy_function(batch_generator, [i], batch_generator.get_out_types())
                #explicitly set output shapes or tf 2.17 will complain about unknown shapes
                batch.set_shape(tf.TensorShape([None, batch_generator.num_models, None]))
                ind.set_shape(tf.TensorShape([None, batch_generator.num_models]))
                return batch, ind
            else:
                batch, ind, emb = tf.numpy_function(batch_generator, [i], batch_generator.get_out_types())
                #explicitly set output shapes or tf 2.17 will complain about unknown shapes
                batch.set_shape(tf.TensorShape([None, batch_generator.num_models, None]))
                ind.set_shape(tf.TensorShape([None, batch_generator.num_models]))
                emb.set_shape(tf.TensorShape([None, batch_generator.num_models, None, batch_generator.scoring_model_config.dim+1]))
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
    return ds
