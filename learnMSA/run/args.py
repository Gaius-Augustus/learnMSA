import argparse
import sys
from pathlib import Path


class LearnMSAArgumentParser(argparse.ArgumentParser):
    def error(self, message : str):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

def parse_args(version : str) -> LearnMSAArgumentParser:

    parser = LearnMSAArgumentParser(
        description=f"learnMSA (version {version}) - "
                    "multiple alignment of protein sequences\n"
                    "\n"
                    "Use 'learnMSA help [argument]' to get detailed help on a specific argument.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input/output group
    io_group = parser.add_argument_group("Input/output and general control")
    io_group.add_argument(
        "-i",
        "--in_file",
        dest="input_file",
        type=str,
        required=True,
        help="Input fasta file."
    )
    io_group.add_argument(
        "-o",
        "--out_file",
        dest="output_file",
        type=str,
        required=True,
        help="Output file. Use -f to change format."
    )
    io_group.add_argument(
        "-f",
        "--format",
        dest="format",
        type=str,
        default="a2m",
        help="Format of the output alignment file."
    )
    io_group.add_argument(
        "--input_format",
        dest="input_format",
        type=str,
        default="fasta",
        help="Format of the input alignment file."
    )
    io_group.add_argument(
        "--save_model",
        dest="save_model",
        type=str,
        default="",
        help="Save a trained model for later reuse"
    )
    io_group.add_argument(
        "--load_model",
        dest="load_model",
        type=str,
        default="",
        help="Load a saved model."
    )
    io_group.add_argument(
        "-s",
        "--silent",
        dest="silent",
        action='store_true',
        help="Suppresses all standard output messages."
    )
    io_group.add_argument(
        "-d",
        "--cuda_visible_devices",
        dest="cuda_visible_devices",
        type=str,
        default="default",
        help="GPU device(s) visible to learnMSA. Use -1 for CPU."
    )
    io_group.add_argument(
        "--work_dir",
        dest="work_dir",
        type=str,
        default="tmp",
        help="Working directory. (default: %(default)s)"
    )
    io_group.add_argument(
        "--convert",
        dest="convert",
        action='store_true',
        help="Convert input files to format specific by --format."
    )

    # Training
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "-n",
        "--num_model",
        dest="num_model",
        type=int,
        default=4,
        help="Number of models to train. (default: %(default)s)"
    )
    train_group.add_argument(
        "-b",
        "--batch",
        dest="batch_size",
        type=int,
        default=-1,
        help="Batch size for training. Default: adaptive "
    )
    train_group.add_argument(
        "--learning_rate",
        dest="learning_rate",
        type=float,
        default=0.1,
        help="Learning rate for gradient descent. "\
            "(default: %(default)s)"
    )
    train_group.add_argument(
        "--epochs",
        dest="epochs",
        type=int,
        nargs=3,
        default=[10, 2, 10],
        help="Number of training epochs. (see detailed help)."
    )
    train_group.add_argument(
        "--max_surgery_runs",
        dest="max_surgery_runs",
        type=int,
        default=2,
        help="Maximum number of model surgery iterations. " \
            "(default: %(default)s)"
    )
    train_group.add_argument(
        "--length_init_quantile",
        dest="length_init_quantile",
        type=float,
        default=0.5,
        help="Check learnMSA help length_init_quantile for details."
    )
    train_group.add_argument(
        "--surgery_quantile",
        dest="surgery_quantile",
        type=float,
        default=0.5,
        help="Check learnMSA help surgery_quantile for details."
    )
    train_group.add_argument(
        "--min_surgery_seqs",
        dest="min_surgery_seqs",
        type=int,
        default=100000,
        help="Check learnMSA help min_surgery_seqs for details."
    )
    train_group.add_argument(
        "--len_mul",
        dest="len_mul",
        type=float,
        default=0.8,
        help="Check learnMSA help len_mul for details."
    )
    train_group.add_argument(
        "--surgery_del",
        dest="surgery_del",
        type=float,
        default=0.5,
        help="Discard match states expected less often than this fraction. " \
            "(default: %(default)s)"
    )
    train_group.add_argument(
        "--surgery_ins",
        dest="surgery_ins",
        type=float,
        default=0.5,
        help="Expand insertions expected more often than this fraction. " \
            "(default: %(default)s)"
    )
    train_group.add_argument(
        "--model_criterion",
        dest="model_criterion",
        type=str,
        default="AIC",
        help="Criterion for model selection. (default: %(default)s)"
    )
    train_group.add_argument(
        "--indexed_data",
        dest="indexed_data",
        action='store_true',
        help="Stream training data at the cost of training time."
    )
    train_group.add_argument(
        "--unaligned_insertions",
        dest="unaligned_insertions",
        action='store_true',
        help="Insertions will be left unaligned."
    )
    train_group.add_argument(
        "--crop",
        dest="crop",
        type=str,
        default="auto",
        help="Crop sequences longer than the given value during training."
    )
    train_group.add_argument(
        "--auto_crop_scale",
        dest="auto_crop_scale",
        type=float,
        default=2.,
        help="Automatically crop sequences longer than this factor times the " \
            "average length during training. (default: %(default)s)"
    )
    train_group.add_argument(
        "--frozen_insertions",
        dest="frozen_insertions",
        action='store_true',
        help="Insertions will be frozen during training."
    )
    train_group.add_argument(
        "--no_sequence_weights",
        dest="no_sequence_weights",
        action='store_true',
        help="Do not use sequence weights and strip mmseqs2 from requirements."\
            " In general not recommended."
    )

    plm_group = parser.add_argument_group("Protein language model integration")
    plm_group.add_argument(
        "--use_language_model",
        dest="use_language_model",
        action='store_true',
        help="Uses a large protein lanague model to generate per-token "\
            "embeddings that guide the MSA step. (default: %(default)s)"
    )
    plm_group.add_argument(
        "--plm_cache_dir",
        dest="plm_cache_dir",
        type=str,
        default=None,
        help="Directory where the protein language model is stored. "\
            "(default: learnMSA install dir)"
    )
    plm_group.add_argument(
        "--language_model",
        dest="language_model",
        type=str,
        default="protT5",
        help="Name of the language model to use. (default: %(default)s)"
    )
    plm_group.add_argument(
        "--scoring_model_dim",
        dest="scoring_model_dim",
        type=int,
        default=16,
        help="Reduced embedding dimension of the scoring model. "\
            "(default: %(default)s)"
    )
    plm_group.add_argument(
        "--scoring_model_activation",
        dest="scoring_model_activation",
        type=str,
        default="sigmoid",
        help="Activation function of the scoring model. "\
            "(default: %(default)s)"
    )
    plm_group.add_argument(
        "--scoring_model_suffix",
        dest="scoring_model_suffix",
        type=str,
        default="",
        help="Suffix to identify a specific scoring model. "\
            "(default: %(default)s)"
    )
    plm_group.add_argument(
        "--temperature",
        dest="temperature",
        type=float,
        default=3.,
        help="Temperature of the softmax function. (default: %(default)s)"
    )
    plm_group.add_argument(
        "--temperature_mode",
        dest="temperature_mode",
        type=str,
        default="trainable",
        help=argparse.SUPPRESS
    )
    plm_group.add_argument(
        "--use_L2",
        dest="use_L2",
        action='store_true',
        help=argparse.SUPPRESS
    )
    plm_group.add_argument(
        "--L2_match",
        dest="L2_match",
        type=float,
        default=0.0,
        help=argparse.SUPPRESS
    )
    plm_group.add_argument(
        "--L2_insert",
        dest="L2_insert",
        type=float,
        default=1000.0,
        help=argparse.SUPPRESS
    )
    plm_group.add_argument(
        "--embedding_prior_components",
        dest="embedding_prior_components",
        type=int,
        default=32,
        help=argparse.SUPPRESS
    )

    vis_group = parser.add_argument_group("Visualization")
    vis_group.add_argument(
        "--logo",
        dest="logo",
        type=str,
        default="",
        help="Produces a pdf of the learned sequence logo."
    )
    vis_group.add_argument(
        "--logo_gif",
        dest="logo_gif",
        type=str,
        default="",
        help="Produces a gif that animates the learned sequence logo over " \
            "training time. Slows down training significantly."
    )

    deprecated_group = parser.add_argument_group("Deprecated arguments")
    deprecated_group.add_argument(
        "--noA2M",
        dest="noA2M",
        action='store_true',
        help="Deprecated: Use --format fasta instead."
    )
    deprecated_group.add_argument(
        "--cluster_dir",
        dest="work_dir",
        type=str,
        default="tmp",
        help="Deprecated: Use --work_dir instead."
    )


    # suppressed arguments intended for development but not for users
    parser.add_argument(
        "--dist_out",
        dest="dist_out",
        type=str,
        default="",
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--alpha_flank",
        dest="alpha_flank",
        type=float,
        default=7000,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--alpha_single",
        dest="alpha_single",
        type=float,
        default=1e9,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--alpha_global",
        dest="alpha_global",
        type=float,
        default=1e4,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--alpha_flank_compl",
        dest="alpha_flank_compl",
        type=float,
        default=1,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--alpha_single_compl",
        dest="alpha_single_compl",
        type=float,
        default=1,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--alpha_global_compl",
        dest="alpha_global_compl",
        type=float,
        default=1,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--inverse_gamma_alpha",
        dest="inverse_gamma_alpha",
        type=float,
        default=3.,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--inverse_gamma_beta", dest="inverse_gamma_beta",
        type=float,
        default=0.5,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--frozen_distances",
        dest="frozen_distances",
        action='store_true',
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--initial_distance",
        dest="initial_distance",
        type=float,
        default=0.05,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--trainable_rate_matrices",
        dest="trainable_rate_matrices",
        action='store_true',
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--grow_mem",
        dest="grow_mem",
        action='store_true',
        help=argparse.SUPPRESS
    )

    return parser