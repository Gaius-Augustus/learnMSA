import copy
import sys
from enum import Enum
from typing import Callable

import numpy as np
import tensorflow as tf

import learnMSA.msa_hmm.Training as train
from learnMSA.msa_hmm.BatchGenerator import BatchGenerator
from learnMSA.msa_hmm.MsaHmmLayer import MsaHmmLayer


class DecodingAlgorithm(Enum):
    VITERBI = 1
    MEA = 2

def decode(
    indices : np.ndarray,
    batch_generator : BatchGenerator,
    msa_hmm_layer : MsaHmmLayer,
    batch_size : int,
    model_ids: list[int],
    encoder: tf.keras.Model|None=None,
    non_homogeneous_mask_func=None,
    parallel_factor=1,
    decode_algorithm: DecodingAlgorithm = DecodingAlgorithm.VITERBI
):
    """ Runs a decoding algorithm (e.g. Viterbi) batch-wise on the sequences 
        in the dataset.

    Args:
        indices (np.ndarray): Indices of sequences in the dataset to be 
            decoded. 
        batch_generator (BatchGenerator): A configured batch generator that
            provides the sequences to be decoded. 
        msa_hmm_layer (MsaHmmLayer): The MSA HMM layer that is passed to the
            decoding function. 
        batch_size (int): Specifies how many sequences will be decoded in 
            parallel.
        model_ids (list[int]): The ids of the models to be decoded.
        encoder (AlignmentModel, optional): An optional encoder that encodes
            the sequences before being passed to the decoding function.
        non_homogeneous_mask_func: Optional function that maps a sequence 
            index i to a num_model x batch x q x q mask that specifies which 
            transitions are allowed.
        parallel_factor: Increasing this number allows computing likelihoods 
            and posteriors chunk-wise in parallel at the cost of memory usage.
            The parallel factor has to be a divisor of the sequence length.
        decode_algorithm (DecodingAlgorithm): The decoding algorithm to be
            used. Defaults to Viterbi.

    Returns:
        A dense integer representation of the most likely state sequences. 
        Shape: (num_model, num_seq, max_len(batch))
    """

    assert batch_generator.is_valid(), \
        "Batch generator is not valid. Have you assigned a dataset?"
    
    # make a copy, as we will change some properties of the batch generator
    batch_generator = copy.copy(batch_generator)
    
    # disable cropping 
    batch_generator.crop_long_seqs = sys.maxsize

    # TODO: ugly workaround code, should be refactored
    # does currently not support multi-GPU, scale the batch size to account 
    # for that and prevent overflow
    num_gpu = len([
        x.name 
        for x in tf.config.list_logical_devices() 
        if x.device_type == 'GPU'
    ]) 
    num_devices = num_gpu + int(num_gpu==0) #account for the CPU-only case 
    batch_size = int(batch_size / num_devices)

    msa_hmm_layer.parallel_factor = parallel_factor
    
    # initialize HMM matrices once before decoding
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

    # wrap everything in a tf function for speed

    def _decode_template(inputs):
        # TODO: this can be improved by encoding only for required models
        encoded_seq = tf.gather(inputs, model_ids, axis=0)
        match decode_algorithm:
            case DecodingAlgorithm.VITERBI:
                decode_fn = msa_hmm_layer.viterbi
            case DecodingAlgorithm.MEA:
                decode_fn = msa_hmm_layer.maximum_expected_accuracy
            case _:
                raise ValueError(
                    f"Unknown decoding algorithm: {decode_algorithm}"
                )
        decoded_seq = decode_fn(
            encoded_seq, 
            non_homogeneous_mask_func=non_homogeneous_mask_func,
            do_recurrent_init=False,  # already done before
        )
        return decoded_seq

    # wrap everything in a tf function for speed
    if encoder is None:
        @tf.function(
            input_signature=[[
                tf.TensorSpec(
                    shape=[
                        None,  
                        None
                    ], 
                    dtype=batch_generator.get_out_types()[0]
                ),
                tf.TensorSpec(
                    shape=[
                        None
                    ], 
                    dtype=batch_generator.get_out_types()[1]
                ) 
            ]]
        )
        def _decode(inputs):
            seqs = tf.one_hot(
                inputs[0], depth=batch_generator.data.alphabet_size()
            )
            seqs = seqs[tf.newaxis, ...]  # add model dimension
            seqs = tf.repeat(seqs, repeats=msa_hmm_layer.cell.num_models, axis=0)
            return _decode_template(seqs)
    else:
        @tf.function(
            input_signature=[[
                tf.TensorSpec(x.shape, dtype=x.dtype) 
                for x in encoder.inputs
            ]]
        )
        def _decode(inputs):
            encoded_seq = encoder(inputs)
            return _decode_template(encoded_seq)

    # decode all batches
    # TODO: make this more performant, a lot of GPU -> CPU transfers
    # probably no prefetching
    for (*inputs, batch_indices), _ in ds:
        decoded_batch = _decode(inputs).numpy()
        _,b,l = decoded_batch.shape
        decoded_seqs[:, batch_indices, :l] = decoded_batch

    return decoded_seqs