from dataclasses import dataclass
from pathlib import Path

import numpy as np


_NEWLINE = ord("\n")
_HEADER = ord(">")
_GAPS = (ord("-"), ord("."))


@dataclass(frozen=True)
class FastaStats:
    """Summary statistics of the sequences in a FASTA file."""

    num_seqs: int
    max_len: int
    avg_len: float
    median_len: float
    total_residues: int


def read_fasta_stats(
    path: str | Path, chunk_size: int = 16 * 1024 * 1024
) -> FastaStats:
    """Computes the number of sequences and their maximum, average and median
    lengths in a single vectorized pass over a FASTA file, without parsing
    records.

    Gaps (- and .) and whitespace do not count towards sequence lengths, so
    aligned FASTA/A2M files yield the lengths of the unaligned sequences.

    Args:
        path: Path to the FASTA file.
        chunk_size: Number of bytes read at once. Memory usage is a small
            multiple of this.

    Returns:
        The statistics of the file.

    Raises:
        ValueError: If the file is empty or not in FASTA format.
    """
    num_headers = 0
    # Residues of the sequence that is still open at the end of a chunk
    open_len = 0
    max_len = 0
    total = 0
    # Histogram of sequence lengths for the median
    hist = np.zeros(0, dtype=np.int64)
    carry = b""

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            eof = not chunk
            buf = carry + chunk
            if eof:
                carry = b""
            else:
                # Only process complete lines, the rest goes to the next chunk
                cut = buf.rfind(b"\n") + 1
                buf, carry = buf[:cut], buf[cut:]
                if not buf:
                    continue
            if buf:
                arr = np.frombuffer(buf, dtype=np.uint8)
                ends = np.flatnonzero(arr == _NEWLINE)
                if arr[-1] != _NEWLINE:
                    ends = np.append(ends, arr.size)
                starts = np.empty_like(ends)
                starts[0] = 0
                starts[1:] = ends[:-1] + 1

                nonempty = starts < ends
                is_header = np.zeros(ends.size, dtype=bool)
                is_header[nonempty] = arr[starts[nonempty]] == _HEADER

                if num_headers == 0:
                    first = np.flatnonzero(is_header)
                    prefix_end = starts[first[0]] if first.size else arr.size
                    if np.any(arr[:prefix_end] > 32):
                        raise ValueError(f"{path} is not a FASTA file.")

                residue = (arr > 32) & (arr != _GAPS[0]) & (arr != _GAPS[1])
                cumsum = np.zeros(arr.size + 1, dtype=np.int32)
                np.cumsum(residue, dtype=np.int32, out=cumsum[1:])
                counts = cumsum[ends] - cumsum[starts]
                counts[is_header] = 0

                # Local sequence index of each line; 0 is the sequence that was
                # open before this chunk (or the empty prefix of the file)
                local = np.cumsum(is_header)
                new_headers = int(local[-1])
                lens = np.bincount(
                    local, weights=counts, minlength=new_headers + 1
                ).astype(np.int64)
                lens[0] += open_len

                # All but the last sequence are complete
                complete = lens[:-1] if num_headers > 0 else lens[1:-1]
                if complete.size:
                    max_len = max(max_len, int(complete.max()))
                    total += int(complete.sum())
                    hist = _add_to_histogram(hist, complete)
                open_len = int(lens[-1])
                num_headers += new_headers

            if eof:
                break

    if num_headers == 0:
        raise ValueError(f"{path} is empty or not a FASTA file.")
    max_len = max(max_len, open_len)
    total += open_len
    hist = _add_to_histogram(hist, np.array([open_len]))

    return FastaStats(
        num_seqs=num_headers,
        max_len=max_len,
        avg_len=total / num_headers,
        median_len=_histogram_median(hist),
        total_residues=total,
    )


def _add_to_histogram(hist: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    counts = np.bincount(lengths)
    if counts.size > hist.size:
        counts[:hist.size] += hist
        return counts
    hist[:counts.size] += counts
    return hist


def _histogram_median(hist: np.ndarray) -> float:
    """Median of the values counted in hist, with the same convention as
    np.median (mean of the two middle values for an even count)."""
    cumsum = np.cumsum(hist)
    n = int(cumsum[-1])
    lower = int(np.searchsorted(cumsum, (n - 1) // 2, side="right"))
    upper = int(np.searchsorted(cumsum, n // 2, side="right"))
    return (lower + upper) / 2
