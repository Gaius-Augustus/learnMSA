import argparse
import sys
from pathlib import Path
from learnMSA.run import util
from learnMSA import Configuration


class LearnMSAArgumentParser(argparse.ArgumentParser):
    def error(self, message : str):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

def parse_args(
    version: str,
    config: Configuration | None = None,
) -> LearnMSAArgumentParser:
    if config is None:
        config = Configuration()

    io = config.input_output
    tr = config.training
    lm = config.language_model
    im = config.init_msa
    vis = config.visualization
    hmm = config.hmm
    hp = config.hmm_prior
    st = config.structure
    adv = config.advanced

    # Encode crop as the string representation expected by --crop
    if tr.auto_crop:
        _crop_default = "auto"
    elif tr.crop == sys.maxsize:
        _crop_default = "disable"
    else:
        _crop_default = str(tr.crop)

    parser = LearnMSAArgumentParser(
        description=f"learnMSA (version {version}) - "
                    "multiple alignment of protein sequences\n"
                    "\n"
                    "Use 'learnMSA help [argument]' to get detailed help on "
                    "a specific argument.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input/output group
    io_group = parser.add_argument_group("Input/output and general control")
    io_group.add_argument(
        "-i",
        "--in_file",
        dest="input_file",
        type=str,
        required=False,
        default=None,
        help="Input fasta file. Optional when --from_msa and --save_model "
            "are both provided."
    )
    io_group.add_argument(
        "-o",
        "--out_file",
        dest="output_file",
        type=str,
        required=False,
        default=None,
        help="Output file. Use -f to change format. Optional when --scores is used."
    )
    io_group.add_argument(
        "-f",
        "--format",
        dest="format",
        type=str,
        default=io.format,
        help="Format of the output alignment file."
    )
    io_group.add_argument(
        "--input_format",
        dest="input_format",
        type=str,
        default=io.input_format,
        help="Format of the input alignment file."
    )
    io_group.add_argument(
        "--config",
        dest="config",
        type=lambda filepath: str(util.validate_filepath(filepath, ".json")),
        default=None,
        help="Path to a JSON configuration file."
    )
    io_group.add_argument(
        "--save_model",
        dest="save_model",
        type=str,
        nargs='?',
        const="<workdir>",
        default=io.save_model,
        help="Save a trained model for later reuse. Optionally provide a path; "
            "if omitted, saves to the working directory."
    )
    io_group.add_argument(
        "--load_model",
        dest="load_model",
        type=str,
        default=io.load_model,
        help="Load a saved model."
    )
    io_group.add_argument(
        "-s",
        "--silent",
        dest="silent",
        action="store_true",
        help="Suppresses all standard output messages."
    )
    io_group.add_argument(
        "-d",
        "--cuda_visible_devices",
        dest="cuda_visible_devices",
        type=str,
        default=io.cuda_visible_devices,
        help="GPU device(s) visible to learnMSA. Use -1 for CPU."
    )
    io_group.add_argument(
        "--work_dir",
        dest="work_dir",
        type=str,
        default=io.work_dir,
        help="Working directory. (default: %(default)s)"
    )
    io_group.add_argument(
        "--convert",
        dest="convert",
        action='store_true',
        help="Convert input files to format specific by --format."
    )
    io_group.add_argument(
        "--scores",
        dest="scores",
        type=str,
        default="" if io.scores == Path() else str(io.scores),
        help="Additional table file containing per-sequence likelihoods.",
    )
    io_group.add_argument(
        "--struct",
        dest="struct_file",
        type=str,
        default=None,
        help="Path to a fasta file containing 3Di states for each sequence."
    )
    io_group.add_argument(
        "--load_emb",
        dest="emb_file",
        type=str,
        default=None,
        help="Path to a file containing embeddings for each sequence."
    )
    io_group.add_argument(
        "--save_emb",
        dest="save_emb",
        type=str,
        default="<workdir>",
        help="Path to save computed embeddings for each sequence. Per default, "\
            "stores embeddings in the working directory. Set to an empty "\
            "string to disable."
    )

    # Training
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "-n",
        "--num_model",
        dest="num_model",
        type=int,
        default=tr.num_model,
        help="Number of models to train. (default: %(default)s)"
    )
    train_group.add_argument(
        "-b",
        "--batch",
        dest="batch_size",
        type=int,
        default=tr.batch_size,
        help="Batch size for training. Prefer --tokens_per_batch unless "\
            "sequences have roughly the same length. Default: adaptive."
    )
    train_group.add_argument(
        "--tokens_per_batch",
        dest="tokens_per_batch",
        type=int,
        default=tr.tokens_per_batch,
        help="Number of tokens per batch for training. Default: adaptive."
    )
    train_group.add_argument(
        "--learning_rate",
        dest="learning_rate",
        type=float,
        default=tr.learning_rate,
        help="Learning rate for gradient descent. "\
            "(default: %(default)s)"
    )
    train_group.add_argument(
        "--no_noise",
        dest="no_noise",
        action="store_true",
        help="Do not perturb the initial HMM parameters with Dirichlet noise." \
        " Default: use noise."
    )
    train_group.add_argument(
        "--noise_concentration",
        dest="noise_concentration",
        type=float,
        default=hmm.noise_concentration,
        help=("Concentration parameter for the Dirichlet noise used during " +
            "training. (default: %(default)s)")
    )
    train_group.add_argument(
        "--no_aa",
        dest="no_aa",
        action="store_true",
        help="Do not use amino acid emissions in the model." \
        "This requires an alternative data source like structure information." \
        "Default: use amino acid emissions."
    )

    class EpochsAction(argparse.Action):
        def __call__(
            self,
            parser: argparse.ArgumentParser,
            namespace: argparse.Namespace,
            values: list[int],
            option_string: str | None = None
        ) -> None:
            if len(values) == 1:
                # Single integer: use for all 3 entries
                setattr(namespace, self.dest, [values[0]] * 3)
            elif len(values) == 3:
                # Three integers: use as provided
                setattr(namespace, self.dest, values)
            else:
                parser.error(f"{option_string} requires either 1 or 3 integers")

    train_group.add_argument(
        "--epochs",
        dest="epochs",
        type=int,
        nargs='+',
        action=EpochsAction,
        default=list(tr.epochs),
        help="Number of training epochs (see detailed help)."
    )
    train_group.add_argument(
        "--max_iterations",
        dest="max_iterations",
        type=int,
        default=tr.max_iterations,
        help="Maximum number of training iterations. If greater than 2, model"
        "surgery will be applied. (default: %(default)s)"
    )
    train_group.add_argument(
        "--length_init",
        dest="length_init",
        type=int,
        nargs='+',
        default=list(tr.length_init) if tr.length_init is not None else None,
        help="Initial lengths for the models. Can be a single integer or a list of integers. "\
            "If a list is provided, the number of models will be set to match the list length. "\
            "(default: determined automatically based on sequence data)"
    )
    train_group.add_argument(
        "--length_init_quantile",
        dest="length_init_quantile",
        type=float,
        default=tr.length_init_quantile,
        help="Check learnMSA help length_init_quantile for details."
    )
    train_group.add_argument(
        "--surgery_quantile",
        dest="surgery_quantile",
        type=float,
        default=tr.surgery_quantile,
        help="Check learnMSA help surgery_quantile for details."
    )
    train_group.add_argument(
        "--min_surgery_seqs",
        dest="min_surgery_seqs",
        type=int,
        default=tr.min_surgery_seqs,
        help="Check learnMSA help min_surgery_seqs for details."
    )
    train_group.add_argument(
        "--len_mul",
        dest="len_mul",
        type=float,
        default=tr.len_mul,
        help="Check learnMSA help len_mul for details."
    )
    train_group.add_argument(
        "--surgery_del",
        dest="surgery_del",
        type=float,
        default=tr.surgery_del,
        help="Discard match states expected less often than this fraction. " \
            "(default: %(default)s)"
    )
    train_group.add_argument(
        "--surgery_ins",
        dest="surgery_ins",
        type=float,
        default=tr.surgery_ins,
        help="Expand insertions expected more often than this fraction. " \
            "(default: %(default)s)"
    )
    train_group.add_argument(
        "--model_criterion",
        dest="model_criterion",
        type=str,
        default=tr.model_criterion,
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
        action="store_true",
        help="Insertions will be left unaligned."
    )
    train_group.add_argument(
        "--crop",
        dest="crop",
        type=str,
        default=_crop_default,
        help="Crop sequences longer than the given value during training."
    )
    train_group.add_argument(
        "--auto_crop_scale",
        dest="auto_crop_scale",
        type=float,
        default=tr.auto_crop_scale,
        help="Automatically crop sequences longer than this factor times the " \
            "average length during training. (default: %(default)s)"
    )
    train_group.add_argument(
        "--trainable_insertions",
        dest="trainable_insertions",
        action="store_true",
        help="Insertions will be trainable during training."
    )
    train_group.add_argument(
        "--no_sequence_weights",
        dest="no_sequence_weights",
        action="store_true",
        help="Do not use sequence weights and strip mmseqs2 from requirements."
            " In general not recommended."
    )
    train_group.add_argument(
        "--skip_training",
        dest="skip_training",
        action="store_true",
        help="Only decode an alignment from the provided model."
    )
    train_group.add_argument(
        "--only_matches",
        dest="only_matches",
        action="store_true",
        help="Omit all insertions in the output and write only those amino "\
            "acids that are assigned to match states."
    )

    init_msa_group = parser.add_argument_group("Initialize with existing MSA")
    init_msa_group.add_argument(
        "--from_msa",
        dest="from_msa",
        type=Path,
        default=im.from_msa,
        help="If set, the initial HMM parameters will inferred from the "
            "provided MSA in FASTA format."
    )
    init_msa_group.add_argument(
        "--match_threshold",
        dest="match_threshold",
        type=float,
        default=im.match_threshold,
        help="When inferring HMM parameters from an MSA, a column is "
            "considered a match state if its occupancy (fraction of non-gap "
            "characters) is at least this value. (default: %(default)s)"
    )
    init_msa_group.add_argument(
        "--global_factor",
        dest="global_factor",
        type=float,
        default=im.global_factor,
        help="A value in [0, 1] that describes the degree to which the MSA"
        " provided with --from_msa is considered a global alignment. This "
        "value is used as a mixing factor and affects how states are counted "
        "when the data contains fragmentary sequences. A global alignment "
        "counts flanks as deletions, while a local alignment counts them as "
        "jumps into the profile using only a single edge. "
        "(default: %(default)s)"
    )
    init_msa_group.add_argument(
        "--pseudocounts",
        dest="pseudocounts",
        action="store_true",
        help="If set, pseudocounts inferred from Dirichlet priors will be "\
            "added on state transition and emissions counted in the MSA "\
            "input via --from_msa."
    )

    plm_group = parser.add_argument_group("Protein language model integration")
    plm_group.add_argument(
        "--use_language_model",
        dest="use_language_model",
        action="store_true",
        help="Uses a large protein lanague model to generate per-token "\
            "embeddings that guide the MSA step. (default: %(default)s)"
    )
    plm_group.add_argument(
        "--plm_cache_dir",
        dest="plm_cache_dir",
        type=str,
        default=lm.plm_cache_dir,
        help="Directory where the protein language model is stored. "\
            "(default: learnMSA install dir)"
    )
    plm_group.add_argument(
        "--language_model",
        dest="language_model",
        type=str,
        default=lm.language_model,
        help="Name of the language model to use. (default: %(default)s)"
    )
    plm_group.add_argument(
        "--scoring_model_dim",
        dest="scoring_model_dim",
        type=int,
        default=lm.scoring_model_dim,
        help=argparse.SUPPRESS
        # help="Reduced embedding dimension of the scoring model. "\
        #     "(default: %(default)s)"
    )
    plm_group.add_argument(
        "--scoring_model_activation",
        dest="scoring_model_activation",
        type=str,
        default=lm.scoring_model_activation,
        help=argparse.SUPPRESS
        # help="Activation function of the scoring model. "\
        #     "(default: %(default)s)"
    )
    plm_group.add_argument(
        "--scoring_model_suffix",
        dest="scoring_model_suffix",
        type=str,
        default=lm.scoring_model_suffix,
        help=argparse.SUPPRESS
        # help="Suffix to identify a specific scoring model. "\
        #     "(default: %(default)s)"
    )
    plm_group.add_argument(
        "--temperature",
        dest="temperature",
        type=float,
        default=lm.temperature,
        help=argparse.SUPPRESS
        # help="Temperature of the softmax function. (default: %(default)s)"
    )
    plm_group.add_argument(
        "--temperature_mode",
        dest="temperature_mode",
        type=str,
        default=lm.temperature_mode,
        help=argparse.SUPPRESS
    )
    plm_group.add_argument(
        "--use_L2",
        dest="use_L2",
        action="store_true",
        help=argparse.SUPPRESS
    )
    plm_group.add_argument(
        "--L2_match",
        dest="L2_match",
        type=float,
        default=lm.L2_match,
        help=argparse.SUPPRESS
    )
    plm_group.add_argument(
        "--L2_insert",
        dest="L2_insert",
        type=float,
        default=lm.L2_insert,
        help=argparse.SUPPRESS
    )
    plm_group.add_argument(
        "--embedding_prior_components",
        dest="embedding_prior_components",
        type=int,
        default=lm.embedding_prior_components,
        help=argparse.SUPPRESS
    )

    struct_group = parser.add_argument_group("Structural information")
    struct_group.add_argument(
        "--struct_prior_name",
        dest="struct_prior_name",
        type=str,
        default=st.prior_name,
        help="Name of a weights file for the structural Dirichlet prior. "\
        "Use empty string for no prior (default: %(default)s)"
    )
    struct_group.add_argument(
        "--struct_prior_components",
        dest="struct_prior_components",
        type=int,
        default=st.prior_components,
        help="The number of mixture components for the structural Dirichlet "\
        "prior. (default: %(default)s)"
    )
    struct_group.add_argument(
        "--struct_prior_temperature",
        dest="struct_prior_temperature",
        type=float,
        default=st.prior_temperature,
        help="The temperature for the structural Dirichlet prior. (default: %(default)s)"
    )
    struct_group.add_argument(
        "--struct_reset_after_surgery",
        dest="struct_reset_after_surgery",
        action="store_true",
        help="Whether to reset the structural information emission parameters "\
        "after model surgery. default: False."
    )

    vis_group = parser.add_argument_group("Visualization")
    vis_group.add_argument(
        "--plot",
        dest="plot",
        type=str,
        default=vis.plot,
        help="Produces a pdf of the learned HMM."
    )
    vis_group.add_argument(
        "--plot_head",
        dest="plot_head",
        type=int,
        default=vis.plot_head,
        help="The HMM head to plot. If not set, the best model based on "\
            "the model selection criterion will be plotted."
    )
    vis_group.add_argument(
        "--logo_gif",
        dest="logo_gif",
        type=str,
        default=vis.logo_gif,
        help="Produces a gif that animates the learned sequence logo over " \
            "training time. Slows down training significantly."
    )

    advanced_group = parser.add_argument_group("Advanced arguments")
    advanced_group.add_argument(
        "--no_jit",
        dest="no_jit",
        action='store_true',
        help="Disable XLA JIT compilation in TensorFlow."
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
        default=io.work_dir,
        help="Deprecated: Use --work_dir instead."
    )
    deprecated_group.add_argument(
        "--logo",
        dest="plot",
        default=vis.plot,
        type=str,
        help="Deprecated: Use --plot instead."
    )


    # suppressed arguments intended for development but not for users
    parser.add_argument(
        "--dist_out",
        dest="dist_out",
        type=str,
        default=adv.dist_out,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--alpha_flank",
        dest="alpha_flank",
        type=float,
        default=hp.alpha_flank,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--alpha_single",
        dest="alpha_single",
        type=float,
        default=hp.alpha_single,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--alpha_global",
        dest="alpha_global",
        type=float,
        default=hp.alpha_global,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--alpha_flank_compl",
        dest="alpha_flank_compl",
        type=float,
        default=hp.alpha_flank_compl,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--alpha_single_compl",
        dest="alpha_single_compl",
        type=float,
        default=hp.alpha_single_compl,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--alpha_global_compl",
        dest="alpha_global_compl",
        type=float,
        default=hp.alpha_global_compl,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--inverse_gamma_alpha",
        dest="inverse_gamma_alpha",
        type=float,
        default=lm.inverse_gamma_alpha,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--inverse_gamma_beta", dest="inverse_gamma_beta",
        type=float,
        default=lm.inverse_gamma_beta,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--frozen_distances",
        dest="frozen_distances",
        action="store_true",
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--initial_distance",
        dest="initial_distance",
        type=float,
        default=adv.initial_distance,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--pre_training_checkpoint",
        dest="pre_training_checkpoint",
        action="store_true",
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--reset_emissions_after_surgery",
        dest="reset_emissions_after_surgery",
        action="store_true",
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--reset_transitions_after_surgery",
        dest="reset_transitions_after_surgery",
        action="store_true",
        help=argparse.SUPPRESS
    )

    return parser