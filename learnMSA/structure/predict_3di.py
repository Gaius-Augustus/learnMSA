"""``learnMSA-3di``: predict per-residue 3Di logits with ProstT5.

Writes an ``.npz`` that ``learnMSA --struct`` accepts, and optionally the
argmax 3Di sequences as FASTA (what ``foldseek createdb --prostt5-model``
produces). Requires the PyTorch extras (``torch``, ``transformers<5``).
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="learnMSA-3di",
        description="Predict per-residue 3Di logits with ProstT5.",
    )
    parser.add_argument("-i", "--input", required=True,
                        help="Amino acid sequences (FASTA).")
    parser.add_argument("-o", "--output", required=True,
                        help="Output .npz with per-residue 3Di logits.")
    parser.add_argument("--fasta", default=None,
                        help="Also write the argmax 3Di sequences as FASTA.")
    parser.add_argument("--device", default=None,
                        help="Torch device (default: cuda if available).")
    parser.add_argument("--cache_dir", default=None,
                        help="Cache for the ProstT5 weights "
                        "(default: ~/.cache/learnmsa/prostt5).")
    parser.add_argument("--split_length", type=int, default=1024,
                        help="Predict longer sequences in pieces, as "
                        "foldseek does; 0 disables splitting. "
                        "(default: %(default)s)")
    parser.add_argument("--batch_memory", type=float, default=2.0,
                        help="Activation memory per batch in GiB "
                        "(default: %(default)s).")
    parser.add_argument("--silent", action="store_true",
                        help="Suppress progress messages.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    # Batches of varying shape fragment the CUDA cache; expandable segments
    # keep the reserved memory close to what is allocated. Must be set
    # before torch initializes CUDA.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                          "expandable_segments:True")
    from learnMSA.structure.io import KIND
    from learnMSA.structure.prostt5 import (PROSTT5_CLASS_ORDER,
                                            PROSTT5_REPO, PROSTT5_REVISION,
                                            predict_logits)
    from learnMSA.util.embedding_cache import EmbeddingCache
    from learnMSA.util.embedding_dataset import EmbeddingDataset
    from learnMSA.util.sequence_dataset import SequenceDataset

    t0 = time.time()
    with SequenceDataset(args.input, "fasta") as data:
        seq_ids = list(data.seq_ids)
        seqs = [data.get_standardized_seq(i) for i in range(data.num_seq)]
    lengths = np.array([len(s) for s in seqs], dtype=np.int64)
    logits = predict_logits(
        seqs, device=args.device, cache_dir=args.cache_dir,
        split_length=args.split_length, verbose=not args.silent,
        batch_memory=args.batch_memory * 2**30,
    )
    dataset = EmbeddingDataset(
        embedding_cache=EmbeddingCache(
            lengths, len(PROSTT5_CLASS_ORDER), cache=logits
        ),
        seq_ids=seq_ids,
    )
    path = dataset.write(args.output, metadata={
        "alphabet": PROSTT5_CLASS_ORDER,
        "kind": KIND,
        "source": f"{PROSTT5_REPO}@{PROSTT5_REVISION}",
    })
    if args.fasta:
        letters = np.array(list(PROSTT5_CLASS_ORDER))
        best = letters[np.argmax(logits, axis=1)]
        offsets = np.concatenate([[0], np.cumsum(lengths)])
        with open(args.fasta, "w") as f:
            for i, seq_id in enumerate(seq_ids):
                f.write(f">{seq_id}\n")
                f.write("".join(best[offsets[i]:offsets[i + 1]]) + "\n")
    if not args.silent:
        print(f"Wrote {path}" + (f" and {args.fasta}" if args.fasta else "")
              + f" ({time.time() - t0:.1f} s in total).", file=sys.stderr)


if __name__ == "__main__":
    main()
