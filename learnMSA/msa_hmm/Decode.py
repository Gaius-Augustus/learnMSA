import copy
import sys
from typing import Callable

import numpy as np
import tensorflow as tf

import learnMSA.msa_hmm.Training as train
from learnMSA.msa_hmm.BatchGenerator import BatchGenerator
from learnMSA.msa_hmm.AlignmentModel import AlignmentModel
from learnMSA.msa_hmm.MsaHmmLayer import MsaHmmLayer


def decode(
    indices : np.ndarray,
    batch_generator : BatchGenerator,
    decode_fn : Callable,
    batch_size : int,
    msa_hmm_layer : MsaHmmLayer,
    model_ids: list[int],
    encoder: tf.keras.Model|None=None,
    non_homogeneous_mask_func=None,
    parallel_factor=1
):
    """ Runs a decoding algorithm (e.g. Viterbi) batch-wise on the sequences 
        in the dataset.

    Args:
        indices (np.ndarray): Indices of sequences in the dataset to be 
            decoded. 
        batch_generator (BatchGenerator): A configured batch generator that
            provides the sequences to be decoded. 
        decode_fn (Callable): A function tha
        batch_size (int): Specifies how many sequences will be decoded in 
            parallel.
        msa_hmm_layer (MsaHmmLayer): The MSA HMM layer that is passed to the
            decoding function. 
        model_ids (list[int]): The ids of the models to be decoded.
        encoder (AlignmentModel, optional): An optional encoder that encodes
            the sequences before being passed to the decoding function.
        non_homogeneous_mask_func: Optional function that maps a sequence 
            index i to a num_model x batch x q x q mask that specifies which 
            transitions are allowed.
        parallel_factor: Increasing this number allows computing likelihoods 
            and posteriors chunk-wise in parallel at the cost of memory usage.
            The parallel factor has to be a divisor of the sequence length.

    Returns:
        A dense integer representation of the most likely state sequences. 
        Shape: (num_model, num_seq, max_len(batch))
    """

    assert batch_generator.is_valid(), \
        "Batch generator is not configured. Call configure() first."
    
    # make a copy, as we will change some properties of the batch generator
    batch_generator = copy.copy(batch_generator)
    
    # disable cropping 
    batch_generator.crop_long_seqs = sys.maxsize

    #does currently not support multi-GPU, scale the batch size to account for that and prevent overflow
    num_gpu = len([
        x.name 
        for x in tf.config.list_logical_devices() 
        if x.device_type == 'GPU'
    ]) 
    num_devices = num_gpu + int(num_gpu==0) #account for the CPU-only case 
    batch_size = int(batch_size / num_devices)

    msa_hmm_layer.cell.recurrent_init()
    
    ds = train.make_dataset(
        indices, 
        batch_generator, 
        batch_size,
        shuffle=False,
        bucket_by_seq_length=True,
        model_lengths=msa_hmm_layer.cell.length
    )

    seq_len = np.amax(batch_generator.data.seq_len(indices)+1)

    #initialize with terminal states
    decoded_seqs = np.zeros(
        (msa_hmm_layer.cell.num_models, indices.size, seq_len), 
        dtype=np.uint32
    ) 
    for i,q in enumerate(msa_hmm_layer.cell.num_states):
        decoded_seqs[i] = q-1 #terminal state

    if encoder is None:
        raise ValueError("Not implemented.")
    else:
        @tf.function(
            input_signature=[[
                tf.TensorSpec(x.shape, dtype=x.dtype) 
                for x in encoder.inputs
            ]]
        )
        def _decode(inputs):
            encoded_seq = encoder(inputs)
            # TODO: this can be improved by encoding only for required models
            encoded_seq = tf.gather(encoded_seq, model_ids, axis=0)
            decoded_seq = decode_fn(
                encoded_seq, 
                msa_hmm_layer.cell, 
                parallel_factor=parallel_factor, 
                non_homogeneous_mask_func=non_homogeneous_mask_func
            )
            return decoded_seq

    # decode all batches
    # TODO: make this more performant, a lot of GPU -> CPU transfers
    for (*inputs, batch_indices), _ in ds:
        decoded_batch = _decode(inputs).numpy()
        _,b,l = decoded_batch.shape
        decoded_seqs[:, batch_indices, :l] = decoded_batch

    return decoded_seqs