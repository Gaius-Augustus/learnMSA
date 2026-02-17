import numpy as np

from learnMSA.run.util import get_batch_multiplicator, get_gpu_memory

DEFAULT_IMPL_FACTOR = 15.0


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
    num_devices = get_batch_multiplicator()
    batch_size = int(np.ceil(num_seq*0.5))
    batch_size -= batch_size % num_devices
    return max(batch_size, num_devices)


def get_adaptive_batch_size(
    model_len: int,
    num_model: int,
    seq_len: int,
    impl_factor: float = DEFAULT_IMPL_FACTOR,
    safety_margin: float = 0.8,
    data_type_size: int = 4,
) -> int:
    """
    Computes an adaptive batch size depending on sequence and model lengths.

    Args:
        model_len: (int) The maximum number of match states.
        num_model: (int) The number of models.
        seq_len: (int) The maximum sequence length.
        impl_factor: (float) An empirical factor to account for
            implementation-specific memory overhead.
        safety_margin: (float) A safety margin to reduce the batch size to
            avoid OOM from edge cases.
        data_type_size: (int) The size of the data type in bytes
            (e.g., 4 for float32).
    """
    mem_avail = float(get_gpu_memory()[0]) * 1e6 # in byte
    batch_size = safety_margin * mem_avail /\
        (impl_factor * model_len * num_model * seq_len * data_type_size)
    batch_size = int(np.floor(batch_size))
    return min(max(batch_size, 1), 4096)

def tokens_per_batch_to_batch_size(
    tokens_per_batch: int,
    seq_len: int,
    impl_factor: float = DEFAULT_IMPL_FACTOR,
) -> int:
    """
    Computes the batch size corresponding to a given number of tokens per batch.

    Args:
        tokens_per_batch: (int) The desired number of tokens per batch.
        seq_len: (int) The maximum sequence length.
        impl_factor: (float) An empirical factor to account for
            implementation-specific memory overhead.
    """
    batch_size = tokens_per_batch /\
        (impl_factor * seq_len)
    batch_size = int(np.floor(batch_size))
    return max(batch_size, 1)
