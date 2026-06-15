import numpy as np
import tensorflow as tf

import learnMSA.protein_language_models.common as common


class ZerosLanguageModel(common.LanguageModel):

    def __init__(self, embedding_dim: int, dtype=tf.float32):
        super(ZerosLanguageModel, self).__init__(dtype=dtype)
        self.dim = int(embedding_dim)

    def call(self, inputs):
        _, mask = inputs
        seq_lens = tf.reduce_sum(tf.cast(mask, tf.int32), axis=-1)
        max_len = tf.reduce_max(seq_lens)
        return tf.zeros((tf.shape(mask)[0], max_len, self.dim), dtype=self.dtype)

    def clear_internal_model(self):
        pass


class ZerosInputEncoder(common.InputEncoder):

    def __call__(self, str_seq, crop):
        del crop
        lens = np.asarray([len(sequence) for sequence in str_seq], dtype=np.int32)
        max_len = int(np.max(lens, initial=0))
        ids = np.zeros((len(str_seq), max_len), dtype=np.int32)
        mask = np.zeros((len(str_seq), max_len), dtype=np.int32)
        for index, seq_len in enumerate(lens):
            mask[index, :seq_len] = 1
        return ids, mask

    def get_signature(self):
        return (
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        )