"""Conservative wall time estimate for aligning a FASTA file with learnMSA,
e.g. to configure a cluster job. Run standalone with

    python -m learnMSA.run.runtime proteins.fasta [--scale SCALE]

which prints the estimate as HH:MM:SS, for example:

    sbatch --time=$(python -m learnMSA.run.runtime proteins.fasta) job.sh
"""
import argparse
import math
from pathlib import Path

from learnMSA.util.fasta_stats import FastaStats, read_fasta_stats


# Fitted on 104 runs with default settings on a GPU workstation (2026-09): the
# 94 HomFam families and 10 extHomFam families with 110k to 3M sequences,
# taking 17s to 76min. With N sequences of mean length L and median length M
# (a proxy for the model length, which is initialized from the median):
# - Every sequence has a fixed cost (parsing, batching, output): N.
# - Training runs about 100 * sqrt(N) / batch_size steps per epoch over
#   batches of cropped sequences: sqrt(N) * L * M.
# - Prediction passes (model selection, Viterbi decoding) cover all
#   sequences: N * L * M.
# The fit is scaled by a safety factor that covers all runs used for fitting,
# also when each run is left out of the fit (worst underestimate 1.9x).
# Slower machines, e.g. with few CPU cores or a slow filesystem, need a scale.
_C_OVERHEAD = 24.1  # seconds
_C_SEQUENCE = 2.76e-4  # seconds per sequence
_C_TRAINING = 1.08e-5  # seconds per sqrt(N) * L * M
_C_PREDICTION = 2.68e-8  # seconds per N * L * M
_SAFETY = 2.5


def estimate_runtime_seconds(stats: FastaStats) -> float:
    """Conservative estimate of the wall time in seconds needed to align the
    sequences described by stats.

    Args:
        stats: Statistics of the input file.

    Returns:
        The estimated runtime in seconds.
    """
    n = stats.num_seqs
    lm = stats.avg_len * stats.median_len
    fitted = (
        _C_OVERHEAD
        + _C_SEQUENCE * n
        + _C_TRAINING * math.sqrt(n) * lm
        + _C_PREDICTION * n * lm
    )
    return _SAFETY * fitted


def round_runtime(seconds: float) -> int:
    """Rounds a runtime up to 15 minute steps below one hour and to full hours
    above.

    Args:
        seconds: Runtime in seconds.

    Returns:
        The rounded runtime in minutes (at least 15).
    """
    minutes = max(1, math.ceil(seconds / 60))
    step = 15 if minutes <= 60 else 60
    return math.ceil(minutes / step) * step


def format_runtime(minutes: int) -> str:
    """Formats a runtime as HH:MM:SS, which is accepted by e.g. Slurm's --time
    and Snakemake's runtime resource. Hours are not wrapped into days.
    """
    return f"{minutes // 60:02d}:{minutes % 60:02d}:00"


def estimate_runtime(input_file: str | Path, scale: float = 1.0) -> int:
    """Estimates a conservative wall time limit for aligning a FASTA file with
    learnMSA, e.g. to configure a cluster job.

    The estimate is deliberately loose and only depends on the number of
    sequences and their mean and median lengths. It is calibrated to cover
    runs with default settings on a GPU workstation.

    Args:
        input_file: Path to the FASTA file to align.
        scale: Factor applied to the estimated seconds before rounding, to
            account for different hardware (e.g. > 1 for slower machines).

    Returns:
        The estimated runtime in minutes, rounded up to 15 minute steps below
        one hour and to full hours above.

    Raises:
        ValueError: If scale is not positive or the file is not a FASTA file.
    """
    if not scale > 0:
        raise ValueError(f"scale must be positive, got {scale}.")
    stats = read_fasta_stats(input_file)
    return round_runtime(scale * estimate_runtime_seconds(stats))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m learnMSA.run.runtime",
        description="Print a conservative wall time estimate (HH:MM:SS) for "
            "aligning a FASTA file with learnMSA using default settings. The "
            "estimate is rounded up to 15 minute steps below one hour and to "
            "full hours above.",
    )
    parser.add_argument("input_file", help="FASTA file to align.")
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Factor applied to the estimate to account for different "
            "hardware, e.g. > 1 for slower machines (default: 1).",
    )
    args = parser.parse_args(argv)
    try:
        minutes = estimate_runtime(args.input_file, args.scale)
    except (ValueError, OSError) as e:
        parser.error(str(e))
    print(format_runtime(minutes))


if __name__ == "__main__":
    main()
