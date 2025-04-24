import argparse
import sys


class LearnMsaArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)


def parse_args(version):

    parser = LearnMsaArgumentParser(
        description=f"learnMSA (version {version}) - "
                    "multiple alignment of protein sequences"
    )
    parser.add_argument(
        "-i",
        "--in_file",
        dest="input_file",
        type=str,
        required=True,
        help="Input sequence file in fasta format."
    )
    parser.add_argument(
        "-o",
        "--out_file",
        dest="output_file",
        type=str,
        required=True,
        help="Filepath for the output alignment."
    )
    parser.add_argument(
        "-n",
        "--num_model",
        dest="num_model",
        type=int,
        default=4,
        help="Number of models trained in parallel. (default: %(default)s)"
    )
    parser.add_argument(
        "-s",
        "--silent",
        dest="silent",
        action='store_true',
        help="Prevents output to stdout."
    )
    parser.add_argument(
        "-b",
        "--batch",
        dest="batch_size",
        type=int,
        default=-1,
        help="Should be lowered if memory issues with the default settings " \
            "occur. Default: Adaptive, depending on model length (64-512)."
    )
    parser.add_argument(
        "-d", 
        "--cuda_visible_devices", 
        dest="cuda_visible_devices", 
        type=str, 
        default="default",
        help="Controls the GPU devices visible to learnMSA as a " \
        "comma-separated list of device IDs. The value -1 forces learnMSA to " \
        "run on CPU. Per default, learnMSA attempts to use all available GPUs."
    )
    parser.add_argument(
        "-f", 
        "--format", 
        dest="format", 
        type=str, 
        default="fasta",
        help="Output file format (see Biopython's SeqIO formats suitable for " \
            "aligned data). Default: fasta (A2M)."
    )
    parser.add_argument(
        "--save_model", 
        dest="save_model", 
        type=str,
        default="", 
        help="Optional filepath to store the trained model."
    )
    parser.add_argument(
        "--load_model", 
        dest="load_model", 
        type=str, 
        default="",
        help="A pretrained model can be loaded from a file, skipping the " \
        "training process."
    )
    parser.add_argument(
        "--noA2M", 
        dest="noA2M", 
        action='store_true',
        help="If set, the output will use only upper case letters and \"-\" " \
            "for gaps. Otherwise, lower case letters are used for insertions " \
            "and \".\" denotes an insertion in another sequence."
    )

    parser.add_argument(
        "--max_surgery_runs", 
        dest="max_surgery_runs", 
        type=int, 
        default=2,
        help="Maximum number of model surgery iterations. " \
            "(default: %(default)s)"
    )
    parser.add_argument(
        "--length_init_quantile", 
        dest="length_init_quantile", 
        type=float, 
        default=0.5,
        help="Quantile of the input sequence lengths that defines the initial" \
            " model lengths. (default: %(default)s)"
    )
    parser.add_argument(
        "--surgery_quantile", 
        dest="surgery_quantile", 
        type=float, 
        default=0.5,
        help="learnMSA will not use sequences shorter than this quantile for " \
            "training during all iterations except the last. "\
                "(default: %(default)s)"
    )
    parser.add_argument(
        "--min_surgery_seqs", 
        dest="min_surgery_seqs", 
        type=int, 
        default=100000,
        help="Minimum number of sequences used per iteration. Overshadows the " \
            "effect of --surgery_quantile. (default: %(default)s)"
    )
    parser.add_argument(
        "--len_mul", 
        dest="len_mul", 
        type=float, 
        default=0.8,
        help="Multiplicative constant for the quantile used to define the " \
            "initial model length (see --length_init_quantile). "\
            "(default: %(default)s)"
    )
    parser.add_argument(
        "--learning_rate", 
        dest="learning_rate", 
        type=float, 
        default=0.1,
        help="The learning rate used during gradient descent. "\
            "(default: %(default)s)"
    )
    parser.add_argument(
        "--epochs", 
        dest="epochs", 
        type=int, 
        nargs=3, 
        default=[10, 2, 10],
        help="Scheme for the number of training epochs during the first, an " \
            "intermediate and the last iteration (expects 3 integers in this " \
            "order). (default: %(default)s)"
    )
    parser.add_argument(
        "--surgery_del", 
        dest="surgery_del", 
        type=float, 
        default=0.5,
        help="Will discard match states that are expected less often than " \
            "this fraction. (default: %(default)s)"
    )
    parser.add_argument(
        "--surgery_ins", 
        dest="surgery_ins", 
        type=float, 
        default=0.5,
        help="Will expand insertions that are expected more often than this " \
            "fraction. (default: %(default)s)"
    )
    parser.add_argument(
        "--model_criterion", 
        dest="model_criterion", 
        type=str, 
        default="AIC",
        help="Criterion for model selection. (default: %(default)s)"
    )
    parser.add_argument(
        "--indexed_data", 
        dest="indexed_data", 
        action='store_true',
        help="Don't load all data into memory at once at the cost of training" \
            " time.")

    parser.add_argument(
        "--unaligned_insertions", 
        dest="unaligned_insertions",
        action='store_true', 
        help="Insertions will be left unaligned."
    )
    parser.add_argument(
        "--crop", 
        dest="crop", 
        type=str, 
        default="auto", 
        help="""During training, sequences longer than the given value will " \
            "be cropped randomly. \n" \
            "Reduces training runtime and memory usage, but might produce " \
            "inaccurate results if too much of the sequences is cropped. The " \
            "output alignment will not be cropped. \n" \
            "Can be set to auto in which case sequences longer than 3 times " \
            "the average length are cropped. Can be set to disable. " \
            "(default: %(default)s)"""
    )
    parser.add_argument(
        "--auto_crop_scale", 
        dest="auto_crop_scale", 
        type=float, 
        default=2.,
        help="During training sequences longer than this factor times the " \
            "average length are cropped. (default: %(default)s)"
    )
    parser.add_argument(
        "--frozen_insertions", 
        dest="frozen_insertions",
        action='store_true', 
        help="Insertions will be frozen during training."
    )

    parser.add_argument(
        "--no_sequence_weights", 
        dest="no_sequence_weights", 
        action='store_true',
        help="Do not use sequence weights and strip mmseqs2 from requirements."\
            " In general not recommended."
    )
    parser.add_argument(
        "--cluster_dir", 
        dest="cluster_dir", 
        type=str, 
        default="tmp",
        help="Directory where the sequence clustering is stored. "\
            "(default: %(default)s)"
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
        help=argparse.SUPPRESS)
    parser.add_argument(
        "--alpha_flank_compl", 
        dest="alpha_flank_compl",
        type=float, 
        default=1, 
        help=argparse.SUPPRESS)
    parser.add_argument(
        "--alpha_single_compl", 
        dest="alpha_single_compl",
        type=float, 
        default=1, 
        help=argparse.SUPPRESS)
    parser.add_argument(
        "--alpha_global_compl", 
        dest="alpha_global_compl",
        type=float, 
        default=1, 
        help=argparse.SUPPRESS)

    parser.add_argument(
        "--inverse_gamma_alpha", 
        dest="inverse_gamma_alpha",
        type=float, 
        default=3., 
        help=argparse.SUPPRESS)
    parser.add_argument(
        "--inverse_gamma_beta", dest="inverse_gamma_beta",
        type=float, 
        default=0.5, 
        help=argparse.SUPPRESS)

    parser.add_argument(
        "--frozen_distances", 
        dest="frozen_distances", 
        action='store_true',
        help="Prevents learning evolutionary distances for all sequences."
    )
    parser.add_argument(
        "--initial_distance", 
        dest="initial_distance", 
        type=float, 
        default=0.05,
        help="The initial evolutionary distance for all sequences. "\
            "(default: %(default)s)"
    )
    parser.add_argument(
        "--trainable_rate_matrices", 
        dest="trainable_rate_matrices", 
        action='store_true',
        help="Prevents learning the amino acid rate matrices of the "\
            "evolutionary model."
    )
    parser.add_argument(
        "--dist_out", 
        dest="dist_out", 
        type=str, 
        default="",
        help="Optional output file for the learned evolutionary distances."
    )

    parser.add_argument(
        "--use_language_model", 
        dest="use_language_model", 
        action='store_true',
        help="Uses a large protein lanague model to generate per-token "\
            "embeddings that guide the MSA step. (default: %(default)s)"
    )
    parser.add_argument(
        "--plm_cache_dir", 
        dest="plm_cache_dir", 
        type=str, 
        default=None,
        help="Directory where the protein language model is stored. "\
            "(default: learnMSA install dir)"
    )
    parser.add_argument(
        "--language_model", 
        dest="language_model", 
        type=str,
        default="protT5", 
        help="Name of the language model to use. (default: %(default)s)"
    )
    parser.add_argument(
        "--scoring_model_dim", 
        dest="scoring_model_dim", 
        type=int, 
        default=16,
        help="Reduced embedding dimension of the scoring model. "\
            "(default: %(default)s)"
    )
    parser.add_argument(
        "--scoring_model_activation", 
        dest="scoring_model_activation", 
        type=str, 
        default="sigmoid",
        help="Activation function of the scoring model. "\
            "(default: %(default)s)"
    )
    parser.add_argument(
        "--scoring_model_suffix", 
        dest="scoring_model_suffix", 
        type=str, 
        default="",
        help="Suffix to identify a specific scoring model. "\
            "(default: %(default)s)"
    )
    parser.add_argument(
        "--temperature_mode", 
        dest="temperature_mode", 
        type=str,
        default="trainable",
        help="The annealing scheme used. Possible values are: "\
        "[trainable, length_norm, warm_to_cold, cold_to_warm, constant, none]. "\
            "(default: %(default)s)"
    )
    parser.add_argument(
        "--use_L2", 
        dest="use_L2", 
        action='store_true',
        help="Uses L2 regularization on the match and insertion state "\
            "embeddings."
    )
    parser.add_argument(
        "--L2_match", 
        dest="L2_match", 
        type=float, 
        default=0.0,
        help="Strength of the L2 regularization on the match state embedding "\
            "weights. (default: %(default)s)"
    )
    parser.add_argument(
        "--L2_insert", 
        dest="L2_insert", 
        type=float, 
        default=1000.0,
        help="Strength of the L2 regularization on the insertion state "\
            "embedding weights. (default: %(default)s)"
    )
    parser.add_argument(
        "--embedding_prior_components", 
        dest="embedding_prior_components", 
        type=int, 
        default=32,
        help="Number of components of the multivariate normal prior "\
            "distribution over the embedding weights. (default: %(default)s)"
    )
    parser.add_argument(
        "--temperature", 
        dest="temperature", 
        type=float, 
        default=3.,
        help="Temperature of the softmax function. (default: %(default)s)"
    )

    parser.add_argument(
        "--logo", 
        dest="logo", 
        action='store_true',
        help="Produces a gif that animates the learned sequence logo over "\
            "training time."
    )
    parser.add_argument(
        "--logo_gif", 
        dest="logo_gif", 
        action='store_true',
        help="Produces a gif that animates the learned sequence logo over " \
            "training time. Slows down training significantly."
    )
    parser.add_argument(
        "--logo_path", 
        dest="logo_path", 
        type=str, 
        default="./logo/",
        help="Filepath used to store created logos and logo gifs. " \
            "Directories are created. (default: %(default)s)"
    )

    parser.add_argument(
        "--tree", 
        action='store_true', 
        help="Infers a cluster tree from sequences and uses it to guide " \
            "alignment")
    parser.add_argument(
        "--warmup_epochs", 
        type=int, 
        default=3,
        help="Number of epochs to train a joint model before using the " \
            "cluster tree. (default: %(default)s)")
    parser.add_argument(
        "--use_tree_transitioner", 
        action='store_true',
        help="Use cluster-specific transition parameters " \
            "(default: %(default)s)")

    args = parser.parse_args()

    return args