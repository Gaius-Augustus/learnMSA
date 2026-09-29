"""Estimate a 3Di observation (confusion) matrix from paired sequences.

Pairs true 3Di strings (e.g. Foldseek on PDB structures) with predicted 3Di
strings (e.g. ProstT5) by sequence ID and counts, for every residue, the
true letter y and the predicted letter o. The saved matrix is

    confusion[y, o] = P(observed o | true y),

row-normalised with a symmetric pseudo-count. Rows follow
``StructureConfig.structural_alphabet``.

Example:
    python util/fit_3di_confusion.py \\
        --true-dir ~/src/snakeMSA/data/homstrad/3Di \\
        --pred-dir ~/src/snakeMSA/data/homfam/predicted_3di \\
        --out learnMSA/hmm/weights/prostt5_3Di_confusion_homstrad.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def read_fasta(path: Path) -> dict[str, str]:
    seqs: dict[str, list[str]] = {}
    name = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                name = line[1:].split()[0]
                seqs[name] = []
            elif name is not None:
                seqs[name].append(line)
    return {k: "".join(v).replace("-", "").replace(".", "").upper()
            for k, v in seqs.items()}


def family_counts(
    true_seqs: dict[str, str], pred_seqs: dict[str, str]
) -> tuple[np.ndarray, int, int]:
    """Returns the (20, 20) count matrix, #pairs used, #pairs skipped."""
    index = {c: i for i, c in enumerate(ALPHABET)}
    counts = np.zeros((len(ALPHABET), len(ALPHABET)))
    used = skipped = 0
    for name, t in true_seqs.items():
        p = pred_seqs.get(name)
        if p is None:
            continue
        if len(p) != len(t):
            skipped += 1
            continue
        ti = [index.get(c, -1) for c in t]
        pi = [index.get(c, -1) for c in p]
        for y, o in zip(ti, pi):
            if y >= 0 and o >= 0:
                counts[y, o] += 1
        used += 1
    return counts, used, skipped


def normalise(counts: np.ndarray, pseudocount: float) -> np.ndarray:
    c = counts + pseudocount
    return c / c.sum(axis=1, keepdims=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--true-dir", type=Path, required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*.fasta")
    parser.add_argument("--pseudocount", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=0,
                        help="Family bootstrap replicates for entry SDs.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    families, per_family = [], []
    total_used = total_skipped = 0
    for true_file in sorted(args.true_dir.glob(args.pattern)):
        pred_file = args.pred_dir / true_file.name
        if not pred_file.exists():
            continue
        counts, used, skipped = family_counts(
            read_fasta(true_file), read_fasta(pred_file)
        )
        total_used += used
        total_skipped += skipped
        if used:
            families.append(true_file.stem)
            per_family.append(counts)
    if not per_family:
        sys.exit("No paired sequences found.")
    per_family = np.stack(per_family)
    counts = per_family.sum(0)
    confusion = normalise(counts, args.pseudocount)

    accuracy = np.trace(counts) / counts.sum()
    print(f"families {len(families)}, pairs {total_used} "
          f"(skipped {total_skipped} length mismatches), "
          f"residues {int(counts.sum())}, accuracy {accuracy:.4f}")
    print("letter  n_true  P(o=y|y)  P(y|o=y)  top confusion")
    col = counts.sum(0)
    for i, c in enumerate(ALPHABET):
        row = confusion[i].copy()
        row[i] = 0
        j = int(np.argmax(row))
        print(f"{c:>6}  {int(counts[i].sum()):>6}  {confusion[i, i]:.3f}"
              f"     {counts[i, i] / max(col[i], 1):.3f}"
              f"     {ALPHABET[j]} {confusion[i, j]:.3f}")

    extra = {}
    if args.bootstrap > 0:
        rng = np.random.default_rng(args.seed)
        n = len(families)
        reps = np.stack([
            normalise(per_family[rng.integers(0, n, n)].sum(0),
                      args.pseudocount)
            for _ in range(args.bootstrap)
        ])
        extra["bootstrap_sd"] = reps.std(0)
        print(f"bootstrap ({args.bootstrap}): max entry SD "
              f"{reps.std(0).max():.4f}, max diagonal SD "
              f"{np.diag(reps.std(0)).max():.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        confusion=confusion.astype(np.float32),
        counts=counts,
        alphabet=np.array(ALPHABET),
        families=np.array(families),
        pseudocount=args.pseudocount,
        **extra,
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
