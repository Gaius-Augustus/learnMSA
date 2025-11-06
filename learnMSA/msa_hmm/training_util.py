from typing import Sequence

from learnMSA.run.util import get_num_gpus


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
    return max(1, tokens_per_batch // max_seq_len)
