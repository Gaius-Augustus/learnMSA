import os
import numpy as np

from learnMSA.run.util import get_batch_multiplicator, get_avail_memory_bytes

MAX_BATCH_SIZE = 1_024
MAX_TOKENS_PER_BATCH = 700_000
MEMORY_DAMP = 0.5

# Implementation factors for the adaptive batch size, per backend.
#
# Keys are "<prefix>_<workload>", the prefix naming the active input tracks
# ("" for amino acids alone) and the workload being either "train" or one of
# the inference modes. "<prefix>_inference" is the fallback for any inference
# mode that has no key of its own; the calibration derives it as the maximum
# over the measured modes, so falling back never underestimates.
#
# To recalibrate use util/calibrate_impl_factor.py.
IMPL_FACTORS: dict[str, dict[str, float]] = {
    "tensorflow": {
        "train": 27.0,                                  # 26.30
        "inference": 15.0,                              # 14.32
        "viterbi": 13.0,                                # 12.31
        "posterior": 15.0,                              # 14.32
        "loglik": 7.0,                                  # 6.18
        "structure_train": 33.0,                        # 32.47
        "structure_inference": 15.0,                    # 14.52
        "structure_viterbi": 13.0,                      # 12.51
        "structure_posterior": 15.0,                    # 14.52
        "structure_loglik": 9.0,                        # 8.32
        "language_model_train": 83.0,                   # 82.99
        "language_model_inference": 54.0,               # 53.19
        "language_model_viterbi": 54.0,                 # 53.19
        "language_model_posterior": 54.0,               # 53.10
        "language_model_loglik": 54.0,                  # 53.10
        "language_model_and_structure_train": 90.0,     # 89.51
        "language_model_and_structure_inference": 56.0, # 55.54
        "language_model_and_structure_viterbi": 56.0,   # 55.54
        "language_model_and_structure_posterior": 56.0, # 55.53
        "language_model_and_structure_loglik": 56.0,    # 55.53
    },
    "pytorch": {
        "train": 17.0,                                  # 16.22
        "inference": 17.0,                              # 16.24
        "viterbi": 12.0,                                # 11.12
        "posterior": 17.0,                              # 16.24
        "loglik": 9.0,                                  # 8.46
        "structure_train": 25.0,                        # 24.61
        "structure_inference": 17.0,                    # 16.17
        "structure_viterbi": 12.0,                      # 11.34
        "structure_posterior": 17.0,                    # 16.17
        "structure_loglik": 9.0,                        # 8.47
        "language_model_train": 101.0,                  # 100.88
        "language_model_inference": 40.0,               # 39.36
        "language_model_viterbi": 37.0,                 # 36.65
        "language_model_posterior": 40.0,               # 39.36
        "language_model_loglik": 37.0,                  # 36.65
        "language_model_and_structure_train": 106.0,    # 105.02
        "language_model_and_structure_inference": 45.0, # 44.22
        "language_model_and_structure_viterbi": 43.0,   # 42.00
        "language_model_and_structure_posterior": 45.0, # 44.22
        "language_model_and_structure_loglik": 43.0,    # 42.01
    },
}


#: Inference modes that are not calibrated in their own right, and the
#: calibrated mode whose cost they share. MEA computes state posteriors and
#: then decodes them, so it is a posterior workload plus a decoding tail.
MODE_FALLBACK: dict[str, str] = {"mea": "posterior"}


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
