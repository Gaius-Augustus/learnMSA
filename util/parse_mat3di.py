"""Parse a mat3di-style substitution matrix (.out) and write it in the
LG-style rate model format used by evoten.

The input format (mat3di.out) contains:
  - A comment line with background frequencies
  - A comment line with the lambda scaling factor (in half-bit / bit units)
  - A symmetric scoring matrix with a header row of single-letter labels

The output follows the evoten/LG convention:
  - Line 1: equilibrium (background) frequencies
  - Lines 2..n: lower triangular exchangeabilities (no diagonal)
  - Optional last line: scaling factor (omitted when 1.0)

The conversion from scoring matrix S to exchangeabilities uses the standard
half-bit relation::

    R(i, j) = 2 ** (lambda * S(i, j) / 2)

Usage::

    python parse_mat3di.py <input.out> <output.model>
"""

import sys
from pathlib import Path

import numpy as np

# Allow running from the learnMSA repo without installing evoten
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evoten"))
from evoten.util import write_rate_model  # noqa: E402


def parse_mat3di(path: Path | str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Parse a mat3di .out file.

    Args:
        path: Path to the .out file.

    Returns:
        A tuple ``(exchangeabilities, equilibrium, alphabet)`` where
        ``exchangeabilities`` has shape ``(n, n)``, ``equilibrium`` shape
        ``(n,)``, and ``alphabet`` is a list of *n* single-letter labels (the
        'X' wildcard row/column is dropped).
    """
    with open(path) as f:
        lines = f.readlines()

    background: np.ndarray | None = None
    lambda_val: float | None = None
    alphabet: list[str] | None = None
    matrix_rows: dict[str, list[int]] = {}

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if "Background" in line and ":" in line:
            background = np.array(list(map(float, line.split(":")[1].split())))
            continue

        if "Lambda" in line and ":" in line:
            lambda_val = float(line.split(":")[1].strip())
            continue

        if stripped.startswith("#"):
            continue

        parts = stripped.split()

        # Header row: all tokens are single alphabetic characters
        if alphabet is None and all(len(p) == 1 and p.isalpha() for p in parts):
            alphabet = parts
            continue

        # Matrix row: first token is the row label, rest are integers
        if alphabet is not None and parts[0] in alphabet:
            matrix_rows[parts[0]] = list(map(int, parts[1:]))

    assert background is not None, "Background frequencies not found."
    assert lambda_val is not None, "Lambda not found."
    assert alphabet is not None, "Alphabet not found."

    # Drop the 'X' wildcard
    clean = [(i, aa) for i, aa in enumerate(alphabet) if aa != "X"]
    orig_indices = [i for i, _ in clean]
    alphabet_clean = [aa for _, aa in clean]
    n = len(alphabet_clean)

    # Background frequencies for the clean alphabet, renormalized
    equilibrium = background[orig_indices]
    equilibrium = (equilibrium / equilibrium.sum()).astype(np.float32)

    # Build the clean score matrix
    score_matrix = np.zeros((n, n), dtype=float)
    for i, aa_i in enumerate(alphabet_clean):
        row = matrix_rows[aa_i]
        for j, orig_j in enumerate(orig_indices):
            score_matrix[i, j] = row[orig_j]

    # Convert scores to exchangeabilities: R(i,j) = 2^(lambda * S(i,j) / 2)
    exchangeabilities = np.exp(
        lambda_val * score_matrix * (np.log(2) / 2)
    ).astype(np.float32)
    np.fill_diagonal(exchangeabilities, 0.0)

    return exchangeabilities, equilibrium, alphabet_clean


def main() -> None:
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.out> <output.model>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    exchangeabilities, equilibrium, alphabet = parse_mat3di(input_path)

    print(f"Alphabet ({len(alphabet)}): {''.join(alphabet)}")
    print(f"Equilibrium sum: {equilibrium.sum():.6f}")

    write_rate_model(output_path, exchangeabilities, equilibrium)
    print(f"Written to {output_path}")


if __name__ == "__main__":
    main()
