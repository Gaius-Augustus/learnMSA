from logging import config
from typing import Sequence

import numpy as np

from learnMSA.run.util import get_num_gpus


def get_initial_model_lengths(
    seq_lens: np.ndarray,
    quantile: float,
    len_mul: float,
    num_models: int,
    random: bool = True,
) -> np.ndarray:
    """
    Computes initial model lengths based on sequence lengths.

    Args:
        seq_lens: np.ndarray
            1D-array of sequence lengths.
        quantile: float
            Quantile of sequence lengths to use for initial model length.
        random: bool
            Whether to add randomness to the model lengths.

    Returns:
        A list of initial model lengths for each model.
    """
    model_length = np.quantile(seq_lens, q=quantile)
    model_length *= len_mul
    model_length = max(3., model_length)
    if random:
        scale = (1 + model_length/50.)
        lens = np.round(np.random.normal(
            loc=model_length, scale=scale, size=num_models
        )).astype(np.int32)
        lens = np.maximum(lens, 3)
        return lens
    else:
        return np.array([int(model_length)] * num_models, dtype=np.int32)


def get_full_length_estimate(
    seq_lens: np.ndarray,
    quantile: float,
    min_seqs: int,
) -> np.ndarray:
    """
    Returns a subset of the indices [0, ..., data.num_seq-1] corresponding to
    sequences that are likely to be full-length based on a simple heuristic.
    """
    num_seq = len(seq_lens)
    #ignore short sequences for all surgery iterations except the last
    k = int(min(num_seq*quantile, max(0, num_seq-min_seqs)))
    #a rough estimate of a set of only full-length sequences
    sorted_indices = np.array([
        i for l,i in sorted(zip(seq_lens, range(num_seq)))
    ])
    full_length_estimate = sorted_indices[k:]
    return full_length_estimate


def get_low_seq_num_batch_size(num_seq: int) -> int:
    """
    Computes a batch size for datasets with a low number of sequences that is
    not the entire dataset but still large enough for efficient training.
    """
    # Compute the number of computing devices, which is the number of GPUs
    # if there is at least one GPU, else 1 (for CPU-only case)
    num_devices = get_num_gpus() + int(get_num_gpus() == 0)
    batch_size = int(np.ceil(num_seq*0.5))
    batch_size -= batch_size % num_devices
    return max(batch_size, num_devices)


def get_adaptive_batch_size(
       model_lengths : Sequence[int], max_seq_len: int, small_gpu: bool
) -> int:
    """
    Computes an adaptive batch size depending on sequence and model lengths.
    Scales automatically with the number of GPUs.
    """
    # Compute the number of computing devices, which is the number of GPUs
    # if there is at least one GPU, else 1 (for CPU-only case)
    num_devices = get_num_gpus() + int(get_num_gpus() == 0)
    model_length = max(model_lengths) if len(model_lengths) > 0 else 0
    if max_seq_len < 200 and model_length < 180:
        batch_size = 512*num_devices
    elif max_seq_len < 520 and model_length < 230:
        batch_size = 256*num_devices
    elif max_seq_len < 700 and model_length < 420:
        batch_size = 128*num_devices
    elif max_seq_len < 850 and model_length < 550:
        batch_size = 64*num_devices
    elif max_seq_len < 1200 and model_length < 700:
        batch_size = 32*num_devices
    elif max_seq_len < 2000 and model_length < 1000:
        batch_size = 8*num_devices
    elif max_seq_len < 4000 and model_length < 1500:
        batch_size = 4*num_devices
    else:
        batch_size = 2*num_devices
    if small_gpu:
        batch_size = max(1, batch_size//2)
    return max(1, batch_size)

def get_adaptive_batch_size_with_language_model(
    model_lengths: Sequence[int],
    max_seq_len: int,
    embedding_dim: int,
    small_gpu: bool
) -> int:
    """
    Computes an adaptive batch size depending on sequence and model lengths
    for use in language model mode.
    """
    # Compute the number of computing devices, which is the number of GPUs
    # if there is at least one GPU, else 1 (for CPU-only case)
    num_devices = get_num_gpus() + int(get_num_gpus() == 0)
    model_length = max(model_lengths) if len(model_lengths) > 0 else 0
    if max_seq_len < 200 and model_length < 180:
        batch_size = (20 + 180*32//embedding_dim)*num_devices
    elif max_seq_len < 520 and model_length < 230:
        batch_size = (10 + 90*32//embedding_dim)*num_devices
    elif max_seq_len < 700 and model_length < 420:
        batch_size = (5 + 45*32//embedding_dim)*num_devices
    elif max_seq_len < 850 and model_length < 550:
        batch_size = (3 + 22*32//embedding_dim)*num_devices
    elif max_seq_len < 1200 and model_length < 700:
        batch_size = (1 + 9*32//embedding_dim)*num_devices
    elif max_seq_len < 2000 and model_length < 1000:
        batch_size = (1 + 4*32//embedding_dim)*num_devices
    elif max_seq_len < 4000 and model_length < 1500:
        batch_size = (1 + 32//embedding_dim)*num_devices
    else:
        batch_size = 1*num_devices
    if small_gpu:
        batch_size = max(1, batch_size//2)
    return max(1, batch_size)

def tokens_per_batch_to_batch_size(
    model_lengths: Sequence[int], max_seq_len: int, tokens_per_batch: int
) -> int:
    # Convert tokens per batch to batch size
    _model_length = max(model_lengths) if len(model_lengths) > 0 else 0  # unused
    if max_seq_len < 1:
        return 1
    return int(max(1, tokens_per_batch // max_seq_len))
