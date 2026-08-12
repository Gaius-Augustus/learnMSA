import os
import numpy as np

from learnMSA.run.util import get_batch_multiplicator, get_avail_memory_bytes

MAX_BATCH_SIZE = 1_024
MAX_TOKENS_PER_BATCH = 700_000
MEMORY_DAMP = 0.5

# Implementation factors for the adaptive batch size, per backend.
#
# To recalibrate use util/calibrate_impl_factor.py.
IMPL_FACTORS: dict[str, dict[str, float]] = {
    "tensorflow": {
        "train": 27.0,
        "inference": 13.0,
        "language_model_train": 54.0,
        "language_model_inference": 195.0,
        "structure_train": 19.0,
        "structure_inference": 9.0,
    },
    "pytorch": {
        "train": 26.0,
        "inference": 18.0,
        "language_model_train": 173.0,
        "language_model_inference": 386.0,
        "structure_train": 61.0,
        "structure_inference": 18.0,
    },
}


def get_impl_factors(backend_name: str) -> dict[str, float]:
    """
    Returns the implementation factors of a backend.

    Args:
        backend_name: One of the names in IMPL_FACTORS, i.e. "tensorflow" or
            "pytorch".

    Returns:
        The factor dictionary of that backend. Falls back to the TensorFlow
        factors for an unknown backend.
    """
    return IMPL_FACTORS.get(backend_name, IMPL_FACTORS["tensorflow"])


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
        scale = model_length/50.
        lens = np.round(np.random.normal(
            loc=model_length, scale=scale, size=num_models
        )).astype(np.int32)
        lens = np.maximum(lens, 3)
        return lens
    else:
        return np.array([int(model_length)] * num_models, dtype=np.int32)


def get_backbone(
    seq_lens: np.ndarray,
    quantile: float,
    min_seqs: int,
    representatives: np.ndarray | None = None,
) -> np.ndarray:
    """
    Returns a subset of the indices [0, ..., data.num_seq-1] corresponding to
    sequences that sufficiently represent the full family.
    """
    num_seq = len(seq_lens)

    backbone = []

    # If available, include all representatives of clusters
    if representatives is not None:
        backbone.append(representatives)

    k = int(min(num_seq*quantile, max(0, num_seq-min_seqs)))

    # rough estimate of a set of only full-length sequences
    sorted_indices = np.array([
        i for l,i in sorted(zip(seq_lens, range(num_seq)))
    ])
    full_length_estimate = sorted_indices[k:]
    backbone.append(full_length_estimate)

    backbone = np.concatenate(backbone)
    backbone = np.unique(backbone)

    return backbone


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
    impl_factor: float = 1.0,
    safety_margin: float = 0.8,
    data_type_size: int = 4,
    max_batch_size: int = MAX_BATCH_SIZE,
) -> int:
    """
    Computes an adaptive batch size depending on sequence and model lengths.

    Args:
        model_len: (int) The maximum number of match states.
        num_model: (int) The number of models.
        seq_len: (int) The maximum sequence length.
        impl_factor: (float) Specific factor that can vary depending on the
            context.
        safety_margin: (float) A safety margin to reduce the batch size to
            avoid OOM from edge cases.
        data_type_size: (int) The size of the data type in bytes
            (e.g., 4 for float32).
    """
    # dampen the true available VRAM to avoid too aggressive scaling
    # on large GPUS
    mem_avail = get_avail_memory_bytes()
    REFERENCE_MEM = 20 * 1024**3
    scale = mem_avail / REFERENCE_MEM
    if scale > 1:
        scale = scale ** MEMORY_DAMP
    mem_avail = REFERENCE_MEM * scale
    # large model numbers can cause OOM if not damped
    num_model_factor = num_model if num_model <= 4 else num_model ** 1.1
    denominator = num_model_factor * float(model_len) * float(seq_len)
    denominator *= impl_factor * data_type_size
    if denominator <= 0.0:
        return 1
    batch_size = int(np.floor(safety_margin * mem_avail / denominator))
    max_batch_size_tokens_per_batch = max(1, MAX_TOKENS_PER_BATCH // seq_len)
    max_batch_size = min(max_batch_size, max_batch_size_tokens_per_batch)
    return min(max(batch_size, 1), max_batch_size)

def tokens_per_batch_to_batch_size(
    tokens_per_batch: int,
    seq_len: int,
    max_batch_size: int = MAX_BATCH_SIZE,
) -> int:
    """
    Computes the batch size corresponding to a given number of tokens per batch.

    Args:
        tokens_per_batch: (int) The desired number of tokens per batch.
        seq_len: (int) The maximum sequence length.
    """
    tokens_per_batch = min(tokens_per_batch, MAX_TOKENS_PER_BATCH)
    batch_size = int(tokens_per_batch // seq_len)
    return min(max(batch_size, 1), max_batch_size)
