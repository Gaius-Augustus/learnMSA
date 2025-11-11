import math

import numpy as np
import tensorflow as tf

import learnMSA.msa_hmm.training as train
from learnMSA.msa_hmm.SequenceDataset import SequenceDataset


def get_state_expectations(
    data : SequenceDataset,
    batch_generator,
    indices,
    batch_size,
    msa_hmm_layer,
    encoder,
    reduce=True,
    with_plm=False,
    plm_dim=0
) -> tf.Tensor:
    """ Computes the expected number of occurences per model and state.
    Args:
        data: The sequence dataset.
        batch_generator: Batch generator.
        indices: Indices that specify which sequences in the dataset should be decoded.
        batch_size: Specifies how many sequences will be decoded in parallel.
        msa_hmm_layer: MsaHmmLayer object.
        encoder: Encoder model that is applied to the sequences before Viterbi.
        reduce: If true (default), the posterior state probs are summed up over
        the dataset size. Otherwise the posteriors for each sequence are returned.
        with_plm: If true, use protein language model embeddings.
        plm_dim: Dimension of the protein language model embeddings.
    Returns:
        The expected number of occurences per model state. Shape (num_model, max_num_states)
    """
    #does currently not support multi-GPU, scale the batch size to account for that and prevent overflow
    num_gpu = len([x.name for x in tf.config.list_logical_devices() if x.device_type == 'GPU'])
    num_devices = num_gpu + int(num_gpu==0) #account for the CPU-only case
    batch_size = int(batch_size / num_devices)
    #compute an optimized order for decoding that sorts sequences of equal length into the same batch
    indices = np.reshape(indices, (-1))
    num_indices = indices.shape[0]
    sorted_indices = np.array([[i,j] for l,i,j in sorted(zip(data.seq_lens[indices], indices, range(num_indices)))])
    msa_hmm_layer.cell.recurrent_init()
    cell = msa_hmm_layer.cell
    old_crop_long_seqs = batch_generator.crop_long_seqs
    batch_generator.crop_long_seqs = math.inf #do not crop sequences
    ds = train.make_dataset(sorted_indices[:,0],
                            batch_generator,
                            batch_size,
                            shuffle=False,
                            bucket_by_seq_length=True,
                            model_lengths=cell.length)

    if with_plm:
        signature = [[
            tf.TensorSpec((None, None, None), dtype=tf.uint8),
            tf.TensorSpec((None, None), dtype=tf.int64),
            tf.TensorSpec((None, None, None, plm_dim+1), dtype=cell.dtype)
        ]]
    else:
        signature = [[
            tf.TensorSpec((None, None, None), dtype=tf.uint8),
            tf.TensorSpec((None, None), dtype=tf.int64)
        ]]
    @tf.function(input_signature=signature)
    def batch_posterior_state_probs(inputs):
        encoded_seq = encoder(inputs)
        posterior_probs = msa_hmm_layer.state_posterior_log_probs(encoded_seq)
        posterior_probs = tf.math.exp(posterior_probs)
        #compute expected number of visits per hidden state and sum over batch dim
        posterior_probs = tf.reduce_sum(posterior_probs, -2)
        if reduce:
            posterior_probs = tf.reduce_sum(posterior_probs, 1) / num_indices
        return posterior_probs

    if reduce:
        posterior_probs = tf.zeros((cell.num_models, cell.max_num_states), cell.dtype)
        for (*inputs, _), _ in ds:
            posterior_probs += batch_posterior_state_probs(inputs)
    else:
        posterior_probs = np.zeros((cell.num_models, num_indices, cell.max_num_states), cell.dtype)
        for (*inputs, batch_indices), _ in ds:
            posterior_probs[:,batch_indices] = batch_posterior_state_probs(inputs)
    batch_generator.crop_long_seqs = old_crop_long_seqs
    return posterior_probs