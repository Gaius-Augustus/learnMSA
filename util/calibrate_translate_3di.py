# SPDX-License-Identifier: GPL-3.0-or-later
# Imports learnMSA.structure.translate_3di (GPL-3.0-or-later, a port of
# Foldseek code); see LICENSES/GPL-3.0-or-later.txt.
"""Calibrate the temperature of 3Di logits from ``learnMSA-3di-translate``.

The files store ``-d^2`` (squared distances of the VQ-VAE embedding to the
20 centroids); learnMSA reads them as the posterior
``softmax(-d^2 / T + log(pi))`` with the 3Di background ``pi`` and
``T = StructureConfig.input_temperature`` (auto: ``"pdb_logits"`` in
``DEFAULT_INPUT_TEMPERATURES``). The argmax of ``-d^2`` is
the foldseek letter of that very structure, so the structure itself cannot
calibrate T. Instead, T is fitted so that the posterior matches the 3Di
letter distribution of the residue's column in a structural alignment.

1. ``collect``: translate the domain files of every superfamily alignment
   (``<sf>.fasta``, 3Di, gapped, IDs = domain file names), check our
   letters against the alignment's (foldseek's), and store per residue the
   squared distances and its column's letter counts.
2. ``fit``: fit T.

Example:
    python util/calibrate_translate_3di.py collect \\
        --alignments ~/data/SCOP/superfamily_3Di_alignments \\
        --structures ~/data/SCOP/domains --out scop_vae.npz
    python util/calibrate_translate_3di.py fit --data scop_vae.npz
"""

import argparse
import warnings
from pathlib import Path

import numpy as np

from calibrate_prostt5 import _log_softmax


def _log_background() -> np.ndarray:
    from learnMSA.config.structure import StructureConfig

    pi = np.asarray(StructureConfig().background_distribution, np.float64)
    return np.log(pi / pi.sum())


def read_aligned(path: Path) -> dict[str, str]:
    """Aligned, uppercase sequences by ID (first word of the header)."""
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
    return {k: "".join(v).upper() for k, v in seqs.items()}


def cmd_collect(args) -> None:
    from learnMSA.structure.translate_3di import (CLASS_ORDER, Encoder3Di,
                                                  read_chains, states,
                                                  translate)

    encoder = Encoder3Di.from_foldseek()
    index = {c: i for i, c in enumerate(CLASS_ORDER)}
    d2s, counts, fams, others = [], [], [], []
    fam_names = []
    n_used = n_skipped = n_mismatch = 0
    for aln_file in sorted(args.alignments.glob("*.fasta")):
        aln = read_aligned(aln_file)
        width = len(next(iter(aln.values())))
        # Letter (or -1) and squared distances at each column per member.
        rows, dists = [], []
        for sid, aligned in aln.items():
            path = args.structures / sid
            if not path.is_file():
                n_skipped += 1
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                chains = read_chains(path)
            ungapped = aligned.replace("-", "").replace(".", "")
            if (len(chains) != 1 or chains[0].ca_only
                    or len(chains[0]) != len(ungapped)):
                n_skipped += 1
                continue
            d2, mask = translate(chains[0], encoder)
            letters = states(d2, mask)
            if any(a != b for a, b, m in zip(letters, ungapped, mask) if m):
                n_mismatch += 1
                continue
            cols = np.flatnonzero(~np.isin(np.array(list(aligned)), ["-", "."]))
            row = np.full(width, -1, dtype=np.int64)
            row[cols[mask]] = [index[c] for c in np.array(list(letters))[mask]]
            dist = np.zeros((width, 20), dtype=np.float32)
            dist[cols] = d2
            rows.append(row)
            dists.append(dist)
            n_used += 1
        if len(rows) < 2:
            continue
        rows = np.array(rows)
        col_counts = np.stack([(rows == k).sum(0) for k in range(20)], -1)
        member, col = np.nonzero(rows >= 0)
        n_others = col_counts.sum(1)[col] - 1
        keep = n_others >= 1
        member, col = member[keep], col[keep]
        d2s.append(np.array(dists)[member, col])
        counts.append(col_counts[col].astype(np.int16))
        others.append(n_others[keep].astype(np.int16))
        fams.append(np.full(len(col), len(fam_names), dtype=np.int32))
        fam_names.append(aln_file.stem)
    np.savez(args.out, d2=np.concatenate(d2s), counts=np.concatenate(counts),
             others=np.concatenate(others),
             fam=np.concatenate(fams), fam_names=np.array(fam_names))
    print(f"used {n_used} domains from {len(fam_names)} families; skipped "
          f"{n_skipped} (missing, multi-chain, CA-only or length differs), "
          f"{n_mismatch} with 3Di different from the alignment")
    print(f"wrote {sum(len(o) for o in others)} residues to {args.out}")


def _entropy(q: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(-np.where(q > 0, q * np.log(q), 0).sum(1).mean())


def _fit(d2: np.ndarray, q: np.ndarray, bounds: tuple[float, float],
         w: np.ndarray | None = None) -> tuple[float, float]:
    """(T*, mean cross-entropy at T*); optional residue weights w."""
    from scipy.optimize import minimize_scalar

    w = np.full(len(q), 1 / len(q)) if w is None else w / w.sum()
    log_pi = _log_background()

    def ce(log_t: float) -> float:
        logp = _log_softmax(-d2 / np.exp(log_t) + log_pi)
        return float(-(w * (q * logp).sum(1)).sum())

    best = minimize_scalar(ce, bounds=np.log(bounds), method="bounded",
                           options={"xatol": 1e-4})
    return float(np.exp(best.x)), float(best.fun)


def cmd_fit(args) -> None:
    data = np.load(args.data)
    sel = data["others"] >= args.min_others
    d2 = data["d2"][sel].astype(np.float64)
    others, fam = data["others"][sel], data["fam"][sel]
    counts = data["counts"][sel].astype(np.float64)
    q = counts / counts.sum(1, keepdims=True)
    bounds = (args.t_min, args.t_max)
    uniq, fam_idx, fam_count = np.unique(fam, return_inverse=True,
                                         return_counts=True)
    print(f"residues {len(q)} (columns with >= {args.min_others} other "
          f"residues), families {len(uniq)}")

    t_star, ce = _fit(d2, q, bounds)
    floor = _entropy(q)
    logp = _log_softmax(-d2 / t_star + _log_background())
    mode = np.mean(logp.argmax(1) == q.argmax(1))
    t_fam, _ = _fit(d2, q, bounds, w=1.0 / fam_count[fam_idx])
    print(f"T*={t_star:.4f}  CE {ce:.4f} nats, H(q) floor {floor:.4f}, "
          f"KL {ce - floor:.4f}; argmax = column mode {mode:.4f}; "
          f"family-weighted T={t_fam:.4f}")

    rng = np.random.default_rng(0)
    half = np.isin(fam, rng.permutation(uniq)[: len(uniq) // 2])
    splits = []
    for train in (half, ~half):
        t_half, _ = _fit(d2[train], q[train], bounds)
        logp = _log_softmax(-d2[~train] / t_half + _log_background())
        splits.append((t_half, -(q[~train] * logp).sum(1).mean()))
    print("family half-splits (fit on one half, CE on the other): "
          + ", ".join(f"T={t:.4f} CE={v:.4f}" for t, v in splits))

    print("T by column depth (other residues in the column):")
    edges = [args.min_others, 6, 11, 31, np.inf]
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo >= hi:
            continue
        m = (others >= lo) & (others < hi)
        if m.sum() < 1000:
            continue
        label = f"{int(lo)}-{int(hi) - 1}" if np.isfinite(hi) \
            else f">={int(lo)}"
        t_bin, _ = _fit(d2[m], q[m], bounds)
        print(f"  {label:>6s}  residues {m.sum():9d}  T {t_bin:.4f}")
    print(f"fitted temperature: {t_star:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("collect")
    p.add_argument("--alignments", type=Path, required=True)
    p.add_argument("--structures", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("fit")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--min-others", type=int, default=4,
                   help="Minimum number of other residues in a column.")
    p.add_argument("--t-min", type=float, default=1e-3)
    p.add_argument("--t-max", type=float, default=100.0)
    args = parser.parse_args()
    {"collect": cmd_collect, "fit": cmd_fit}[args.cmd](args)


if __name__ == "__main__":
    main()
