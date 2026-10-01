"""Calibrate ProstT5 3Di logits against true 3Di (descriptive statistics).

1. ``fasta``: collect the amino acid sequences that have a true 3Di string
   (e.g. Homstrad reference chains inside the Homfam families) into one
   FASTA, with IDs ``<family>|<id>``.
2. Run ``learnMSA-3di -i that.fasta -o that.npz``.
3. ``fit``: fit one temperature T minimising the NLL of the true letters
   under softmax(z / T), and report accuracy, NLL, ECE and a reliability
   table before and after.

Example:
    python util/calibrate_prostt5.py fasta \\
        --true-dir ~/src/snakeMSA/data/homstrad/3Di \\
        --aa-dir ~/src/snakeMSA/data/homfam/unaligned --out homstrad.fasta
    learnMSA-3di -i homstrad.fasta -o homstrad.3di.npz
    python util/calibrate_prostt5.py fit \\
        --true-dir ~/src/snakeMSA/data/homstrad/3Di \\
        --logits homstrad.3di.npz
"""

import argparse
from pathlib import Path

import numpy as np

from learnMSA.config.structure import StructureConfig

ALPHABET = StructureConfig().structural_alphabet


def read_fasta(path: Path) -> dict[str, str]:
    """Gap-free, uppercase sequences by ID (first word of the header)."""
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


def cmd_fasta(args) -> None:
    n = 0
    with open(args.out, "w") as out:
        for true_file in sorted(args.true_dir.glob("*.fasta")):
            fam = true_file.stem
            aa_file = args.aa_dir / f"{fam}{args.aa_suffix}"
            if not aa_file.exists():
                continue
            true = read_fasta(true_file)
            aa = read_fasta(aa_file)
            for sid, t in true.items():
                s = aa.get(sid)
                if s is not None and len(s) == len(t):
                    out.write(f">{fam}|{sid}\n{s}\n")
                    n += 1
    print(f"wrote {n} sequences to {args.out}")


def _log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def _ece(prob: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf = prob.max(axis=1)
    correct = prob.argmax(axis=1) == y
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        sel = idx == b
        if sel.any():
            ece += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    return float(ece)


def cmd_fit(args) -> None:
    from scipy.optimize import minimize_scalar

    from learnMSA.structure.io import read_logits

    data = read_logits(args.logits, ALPHABET)
    index = {c: i for i, c in enumerate(ALPHABET)}
    zs, ys, fams = [], [], []
    for i, key in enumerate(data.seq_ids):
        fam, sid = key.split("|", 1)
        true = read_fasta(args.true_dir / f"{fam}.fasta")[sid]
        y = np.array([index.get(c, -1) for c in true])
        z = data.get_encoded_seq(i).astype(np.float64)
        keep = y >= 0
        zs.append(z[keep]); ys.append(y[keep]); fams += [fam] * keep.sum()
    z, y = np.concatenate(zs), np.concatenate(ys)
    fams = np.array(fams)
    nll = lambda t: -_log_softmax(z / t)[np.arange(len(y)), y].mean()
    best = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    t_star = float(best.x)

    # Stability: fit T on each half of the families, score the other half.
    rng = np.random.default_rng(0)
    uniq = np.unique(fams)
    half = set(rng.permutation(uniq)[: len(uniq) // 2])
    in_a = np.array([f in half for f in fams])
    splits = []
    for train in (in_a, ~in_a):
        f = lambda t, m=train: -_log_softmax(z[m] / t)[
            np.arange(m.sum()), y[m]].mean()
        t_half = minimize_scalar(f, bounds=(0.1, 10.0),
                                 method="bounded").x
        test = ~train
        splits.append((t_half, -_log_softmax(z[test] / t_half)[
            np.arange(test.sum()), y[test]].mean()))

    print(f"residues {len(y)}, sequences {len(data.seq_ids)}, "
          f"families {len(uniq)}")
    print(f"accuracy (argmax) {np.mean(z.argmax(1) == y):.4f}")
    for label, t in (("T=1", 1.0), (f"T*={t_star:.3f}", t_star)):
        p = np.exp(_log_softmax(z / t))
        print(f"{label:10s} NLL {nll(t):.4f} nats, ECE {_ece(p, y):.4f}, "
              f"mean max-prob {p.max(1).mean():.4f}")
    print("family half-splits (T fitted on one half, NLL on the other): "
          + ", ".join(f"T={t:.3f} NLL={v:.4f}" for t, v in splits))

    p = np.exp(_log_softmax(z / t_star))
    conf, correct = p.max(1), p.argmax(1) == y
    print("reliability at T*: max-prob bin, share of residues, accuracy")
    for lo in np.arange(0.0, 1.0, 0.1):
        sel = (conf >= lo) & (conf < lo + 0.1)
        if sel.any():
            print(f"  [{lo:.1f}, {lo + 0.1:.1f})  {sel.mean():.3f}  "
                  f"{correct[sel].mean():.3f}")

    print(f"fitted temperature: {t_star:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fasta")
    p.add_argument("--true-dir", type=Path, required=True)
    p.add_argument("--aa-dir", type=Path, required=True)
    p.add_argument("--aa-suffix", default=".vie")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("fit")
    p.add_argument("--true-dir", type=Path, required=True)
    p.add_argument("--logits", type=Path, required=True)
    args = parser.parse_args()
    {"fasta": cmd_fasta, "fit": cmd_fit}[args.cmd](args)


if __name__ == "__main__":
    main()
