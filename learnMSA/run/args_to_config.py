from argparse import Namespace

from learnMSA.config import (
    AdvancedConfig,
    Configuration,
    InitMSAConfig,
    InputOutputConfig,
    LanguageModelConfig,
    TrainingConfig,
    VisualizationConfig,
)


def args_to_config(args: Namespace) -> Configuration:
    """Convert argparse Namespace to Configuration object.

    Args:
        args: Namespace object from argparse containing command-line arguments.

    Returns:
        Configuration object with values from the command-line arguments.
    """
    # Convert crop to appropriate type (str or int)
    crop = args.crop
    if crop != "auto" and crop != "disable":
        try:
            crop = int(crop)
        except ValueError:
            # Keep as string if not convertible to int
            pass

    # Create input/output configuration
    input_output_config = InputOutputConfig(
        input_file=args.input_file,
        output_file=args.output_file,
        format=args.format,
        input_format=args.input_format,
        save_model=args.save_model,
        load_model=args.load_model,
        silent=args.silent,
        cuda_visible_devices=args.cuda_visible_devices,
        work_dir=args.work_dir,
        convert=args.convert,
    )

    # Create nested configuration objects
    training_config = TrainingConfig(
        batch_size=args.batch_size,
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
        auto_crop_scale=args.auto_crop_scale,
        frozen_insertions=args.frozen_insertions,
        no_sequence_weights=args.no_sequence_weights,
        skip_training=args.skip_training,
    )

    init_msa_config = InitMSAConfig(
        from_msa=args.from_msa,
        match_threshold=args.match_threshold,
        global_factor=args.global_factor,
        random_scale=args.random_scale,
        pseudocounts=args.pseudocounts,
    )

    language_model_config = LanguageModelConfig(
        use_language_model=args.use_language_model,
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
    )

    visualization_config = VisualizationConfig(
        logo=args.logo,
        logo_gif=args.logo_gif,
    )

    advanced_config = AdvancedConfig(
        dist_out=args.dist_out,
        alpha_flank=args.alpha_flank,
        alpha_single=args.alpha_single,
        alpha_global=args.alpha_global,
        alpha_flank_compl=args.alpha_flank_compl,
        alpha_single_compl=args.alpha_single_compl,
        alpha_global_compl=args.alpha_global_compl,
        inverse_gamma_alpha=args.inverse_gamma_alpha,
        inverse_gamma_beta=args.inverse_gamma_beta,
        frozen_distances=args.frozen_distances,
        initial_distance=args.initial_distance,
        trainable_rate_matrices=args.trainable_rate_matrices,
    )

    # Create main Configuration object
    config = Configuration(
        num_model=args.num_model,
        input_output=input_output_config,
        training=training_config,
        init_msa=init_msa_config,
        language_model=language_model_config,
        visualization=visualization_config,
        advanced=advanced_config,
    )

    return config
