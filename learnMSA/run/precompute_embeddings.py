"""Pre-compute protein language model embeddings and store them on disk.
"""

import argparse
import sys
from pathlib import Path

import learnMSA.run.util as util
from learnMSA.backend import set_backend
from learnMSA.config import LanguageModelConfig

EPILOG = """\
The embeddings are held in memory in half precision while they are computed, so
a full-dimensional file needs roughly

    sum(sequence lengths) x dim x 2 bytes

of RAM -- about 2 GiB for a million residues at protT5's 1024 dimensions. The
same applies to the machine that later reads the file back.

Example:

  learnMSA_embed -i seqs.fasta -o seqs.emb --language_model protT5 --full
  learnMSA -i seqs.fasta -o msa.fasta --use_language_model \\
      --reduce_online --load_emb seqs.emb
"""


def make_parser() -> argparse.ArgumentParser:
    """The command line of ``learnMSA_embed``."""
    defaults = LanguageModelConfig()
    parser = argparse.ArgumentParser(
        prog="learnMSA_embed",
        description=(
            "Pre-compute protein language model embeddings and write them to "
            "a file that learnMSA --load_emb can read."
        ),
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input", dest="input_file", type=str, required=True,
        help="Input sequence file."
    )
    parser.add_argument(
        "-o", "--output", dest="output_file", type=str, required=True,
        help="Output embedding file."
    )
    parser.add_argument(
        "--input_format", dest="input_format", type=str, default="fasta",
        help="Format of the input file. (default: %(default)s)"
    )
    parser.add_argument(
        "--language_model", dest="language_model", type=str,
        default=defaults.language_model,
        help="Name of the language model to use. (default: %(default)s)"
    )
    parser.add_argument(
        "--plm_cache_dir", dest="plm_cache_dir", type=str, default=None,
        help="Directory where the protein language model is stored."
    )
    parser.add_argument(
        "--scoring_model_dim", dest="scoring_model_dim", type=int,
        default=defaults.scoring_model_dim,
        help="Reduced embedding dimension of the scoring model. Ignored with "
             "--full. (default: %(default)s)"
    )
    parser.add_argument(
        "--full", dest="full", action="store_true",
        help="Keep the language model's native embedding width instead of "
             "reducing it with the frozen scoring model. Such embeddings can "
             "only be aligned with --reduce_online, which learns the "
             "reduction jointly with the alignment."
    )
    parser.add_argument(
        "--backend", dest="backend", type=str, default="auto",
        choices=["auto", "tensorflow", "pytorch"],
        help="Compute backend. --full requires pytorch. "
             "(default: %(default)s)"
    )
    parser.add_argument(
        "--cuda_visible_devices", dest="cuda_visible_devices", type=str,
        default="default",
        help="Controls the CUDA_VISIBLE_DEVICES environment variable."
    )
    parser.add_argument(
        "--silent", dest="silent", action="store_true",
        help="Suppress progress output."
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = make_parser()
    args = parser.parse_args(argv)
    verbose = not args.silent

    # Select the backend before anything imports a framework, then set up the
    # devices -- the same order run_main uses.
    if args.backend != "auto":
        set_backend(args.backend)
    util.setup_devices(args.cuda_visible_devices, verbose, one_dnn_opts=False)

    from learnMSA.protein_language_models.compute_embeddings import \
        compute_embeddings
    from learnMSA.util import EmbeddingDataset, SequenceDataset

    config = LanguageModelConfig(
        use_language_model=True,
        reduce_online=args.full,
        language_model=args.language_model,
        plm_cache_dir=args.plm_cache_dir,
        scoring_model_dim=args.scoring_model_dim,
    )

    with SequenceDataset(
        args.input_file, args.input_format, remove_gaps=True
    ) as data:
        data.validate_dataset(single_seq_ok=True)
        cache = compute_embeddings(data, config, verbose=verbose)
        dataset = EmbeddingDataset(
            embedding_cache=cache, seq_ids=data.seq_ids
        )
        dataset.write(args.output_file)

        if verbose:
            print(
                f"Wrote {cache.dim}-dimensional embeddings for "
                f"{data.num_seq} sequences to {Path(args.output_file)}"
            )


if __name__ == "__main__":
    sys.exit(main())
