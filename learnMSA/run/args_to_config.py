from pathlib import Path
import sys
from argparse import Namespace

from learnMSA.config import (AdvancedConfig, Configuration, PHMMConfig,
                             InitMSAConfig, InputOutputConfig,
                             LanguageModelConfig, TrainingConfig,
                             VisualizationConfig)
from learnMSA.config.hmm import PHMMPriorConfig
from learnMSA.config.structure import StructureConfig


def args_to_config(args: Namespace) -> Configuration:
    """Convert argparse Namespace to Configuration object.

    Args:
        args: Namespace object from argparse containing command-line arguments.

    Returns:
        Configuration object with values from the command-line arguments.
    """
    # Convert crop to appropriate type (str or int)
    auto_crop = args.crop == "auto"
    if args.crop in {"auto", "disable"}:
        crop = sys.maxsize
    else:
        try:
            crop = int(args.crop)
        except ValueError:
            raise ValueError(
                "Invalid value for --crop. Use 'auto', 'disable' or an integer."
            )

    # Create input/output configuration
    input_output_config = InputOutputConfig(
        input_file=args.input_file,
        output_file=args.output_file if args.output_file is not None else "",
        format=args.format,
        input_format=args.input_format,
        save_model=args.save_model,
        load_model=args.load_model,
        scores=args.scores,
        verbose=not args.silent,
        cuda_visible_devices=args.cuda_visible_devices,
        work_dir=args.work_dir,
        convert=args.convert,
        struct_file=args.struct_file,
        emb_file=args.emb_file,
        save_emb=_get_save_emb(args),
    )

    # Create nested configuration objects
    training_config = TrainingConfig(
        num_model=args.num_model,
        batch_size=args.batch_size,
        tokens_per_batch=args.tokens_per_batch,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        max_iterations=args.max_iterations,
        length_init=args.length_init,
        length_init_quantile=args.length_init_quantile,
        surgery_quantile=args.surgery_quantile,
        min_surgery_seqs=args.min_surgery_seqs,
        len_mul=args.len_mul,
        surgery_del=args.surgery_del,
        surgery_ins=args.surgery_ins,
        model_criterion=args.model_criterion,
        indexed_data=args.indexed_data,
        unaligned_insertions=args.unaligned_insertions,
        crop=crop,
        auto_crop=auto_crop,
        auto_crop_scale=args.auto_crop_scale,
        trainable_insertions=args.trainable_insertions,
        no_sequence_weights=args.no_sequence_weights,
        skip_training=args.skip_training,
        trainable_rate_matrices=args.trainable_rate_matrices,
        trainable_distances=not args.frozen_distances,
        equilibrium_sample=args.trainable_rate_matrices,
        transposed=False,
        only_matches=args.only_matches,
        use_noise=not args.no_noise,
    )

    init_msa_config = InitMSAConfig(
        from_msa=args.from_msa,
        match_threshold=args.match_threshold,
        global_factor=args.global_factor,
        pseudocounts=args.pseudocounts,
    )

    language_model_config = LanguageModelConfig(
        use_language_model=args.use_language_model or args.emb_file is not None,
        only_embeddings=args.save_emb != "<workdir>" and not args.use_language_model,
        plm_cache_dir=args.plm_cache_dir,
        language_model=args.language_model,
        scoring_model_dim=args.scoring_model_dim,
        scoring_model_activation=args.scoring_model_activation,
        scoring_model_suffix=args.scoring_model_suffix,
        temperature=args.temperature,
        temperature_mode=args.temperature_mode,
        use_L2=args.use_L2,
        L2_match=args.L2_match,
        L2_insert=args.L2_insert,
        embedding_prior_components=args.embedding_prior_components,
        inverse_gamma_alpha=args.inverse_gamma_alpha,
        inverse_gamma_beta=args.inverse_gamma_beta,
    )

    visualization_config = VisualizationConfig(
        logo=args.logo,
        logo_gif=args.logo_gif,
    )

    hmm_config = PHMMConfig(
        noise_concentration=args.noise_concentration,
    )

    hmm_prior_config = PHMMPriorConfig(
        alpha_flank=args.alpha_flank,
        alpha_single=args.alpha_single,
        alpha_global=args.alpha_global,
        alpha_flank_compl=args.alpha_flank_compl,
        alpha_single_compl=args.alpha_single_compl,
        alpha_global_compl=args.alpha_global_compl,
    )

    structure_config = StructureConfig(
        use_structure=bool(args.struct_file),
    )

    advanced_config = AdvancedConfig(
        dist_out=args.dist_out,
        initial_distance=args.initial_distance,
        jit_compile=not args.no_jit,
    )

    # Create main Configuration object
    config = Configuration(
        input_output=input_output_config,
        training=training_config,
        hmm=hmm_config,
        hmm_prior=hmm_prior_config,
        init_msa=init_msa_config,
        language_model=language_model_config,
        visualization=visualization_config,
        structure=structure_config,
        advanced=advanced_config,
    )

    # Deprecated checks
    if args.noA2M:
        raise DeprecationWarning(
            "--noA2M is deprecated. Use --format fasta instead."
        )

    return config

def _get_save_emb(args: Namespace) -> str:
    """Determine the save_emb path based on command-line arguments."""
    if args.save_emb == "<workdir>":
        return str(Path(args.work_dir) / (Path(args.input_file).stem + ".emb"))
    else:
        return args.save_emb
