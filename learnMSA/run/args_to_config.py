from pathlib import Path
import sys
from argparse import Namespace

from learnMSA.config import Configuration


def args_to_config(args: Namespace, base_config: Configuration | None = None) -> Configuration:
    """Apply parsed command-line arguments on top of base_config.

    Args:
        args: Namespace object from argparse containing command-line arguments.
        base_config: Configuration to use as starting point. If not provided,
            a fresh Configuration() with all defaults is used.

    Returns:
        Validated Configuration with args values placed into base_config.
    """
    if base_config is None:
        base_config = Configuration()

    # Serialize to a plain dict so we can selectively overwrite CLI-relevant
    # fields while preserving any extra fields set in base_config.
    data = base_config.model_dump(mode="json")
    io = data["input_output"]
    tr = data["training"]
    im = data["init_msa"]
    lm = data["language_model"]
    vis = data["visualization"]
    hmm_d = data["hmm"]
    hp = data["hmm_prior"]
    st = data["structure"]
    adv = data["advanced"]

    # --- Input/output ---
    io["input_file"] = args.input_file if args.input_file is not None else ""
    io["output_file"] = args.output_file if args.output_file is not None else ""
    io["format"] = args.format
    io["input_format"] = args.input_format
    io["save_model"] = _get_save_model(args)
    io["load_model"] = args.load_model
    io["scores"] = args.scores
    io["verbose"] = not args.silent
    io["cuda_visible_devices"] = args.cuda_visible_devices
    io["work_dir"] = args.work_dir
    io["convert"] = args.convert
    io["struct_file"] = args.struct_file
    io["emb_file"] = args.emb_file
    io["save_emb"] = _get_save_emb(args)

    # --- Training ---
    tr["num_model"] = args.num_model
    tr["batch_size"] = args.batch_size
    tr["tokens_per_batch"] = args.tokens_per_batch
    tr["learning_rate"] = args.learning_rate
    tr["epochs"] = args.epochs
    tr["max_iterations"] = args.max_iterations
    tr["length_init"] = args.length_init
    tr["length_init_quantile"] = args.length_init_quantile
    tr["surgery_quantile"] = args.surgery_quantile
    tr["min_surgery_seqs"] = args.min_surgery_seqs
    tr["len_mul"] = args.len_mul
    tr["surgery_del"] = args.surgery_del
    tr["surgery_ins"] = args.surgery_ins
    tr["model_criterion"] = args.model_criterion
    tr["indexed_data"] = args.indexed_data
    tr["unaligned_insertions"] = args.unaligned_insertions
    tr["auto_crop_scale"] = args.auto_crop_scale
    tr["trainable_insertions"] = args.trainable_insertions
    tr["no_sequence_weights"] = args.no_sequence_weights
    tr["skip_training"] = args.skip_training
    tr["trainable_distances"] = not args.frozen_distances
    tr["only_matches"] = args.only_matches
    tr["use_noise"] = not args.no_noise
    tr["no_aa"] = args.no_aa
    tr["pre_training_checkpoint"] = args.pre_training_checkpoint
    tr["reset_emissions_after_surgery"] = args.reset_emissions_after_surgery
    tr["reset_transitions_after_surgery"] = args.reset_transitions_after_surgery
    tr["decoding_mode"] = args.decoding_mode

    # Crop: decode the argparser string representation into config fields.
    tr["auto_crop"] = args.crop == "auto"
    if args.crop in {"auto", "disable"}:
        tr["crop"] = sys.maxsize
    else:
        try:
            tr["crop"] = int(args.crop)
        except ValueError:
            raise ValueError(
                "Invalid value for --crop. Use 'auto', 'disable' or an integer."
            )

    # --- Init MSA ---
    im["from_msa"] = str(args.from_msa) if args.from_msa is not None else None
    im["match_threshold"] = args.match_threshold
    im["global_factor"] = args.global_factor
    im["pseudocounts"] = args.pseudocounts

    # --- Language model ---
    lm["use_language_model"] = args.use_language_model or args.emb_file is not None
    lm["only_embeddings"] = args.save_emb != "<workdir>" and not args.use_language_model
    lm["plm_cache_dir"] = args.plm_cache_dir
    lm["language_model"] = args.language_model
    lm["scoring_model_dim"] = args.scoring_model_dim
    lm["scoring_model_activation"] = args.scoring_model_activation
    lm["scoring_model_suffix"] = args.scoring_model_suffix
    lm["temperature"] = args.temperature
    lm["temperature_mode"] = args.temperature_mode
    lm["use_L2"] = args.use_L2
    lm["L2_match"] = args.L2_match
    lm["L2_insert"] = args.L2_insert
    lm["embedding_prior_components"] = args.embedding_prior_components
    lm["inverse_gamma_alpha"] = args.inverse_gamma_alpha
    lm["inverse_gamma_beta"] = args.inverse_gamma_beta

    # --- Visualization ---
    vis["plot"] = args.plot
    vis["plot_head"] = args.plot_head
    vis["logo_gif"] = args.logo_gif or ""

    # --- HMM ---
    hmm_d["noise_concentration"] = args.noise_concentration

    # --- HMM prior ---
    hp["alpha_flank"] = args.alpha_flank
    hp["alpha_single"] = args.alpha_single
    hp["alpha_global"] = args.alpha_global
    hp["alpha_flank_compl"] = args.alpha_flank_compl
    hp["alpha_single_compl"] = args.alpha_single_compl
    hp["alpha_global_compl"] = args.alpha_global_compl

    # --- Structure ---
    st["use_structure"] = bool(args.struct_file)
    st["prior_name"] = args.struct_prior_name
    st["prior_components"] = args.struct_prior_components
    st["prior_temperature"] = args.struct_prior_temperature
    st["reset_after_surgery"] = args.struct_reset_after_surgery

    # --- Advanced ---
    adv["dist_out"] = args.dist_out
    adv["initial_distance"] = args.initial_distance
    adv["jit_compile"] = not args.no_jit

    # Deprecated checks
    if args.noA2M:
        raise DeprecationWarning(
            "--noA2M is deprecated. Use --format fasta instead."
        )

    return Configuration.model_validate(data)


def _get_save_emb(args: Namespace) -> str:
    """Determine the save_emb path based on command-line arguments."""
    if args.save_emb == "<workdir>":
        if args.input_file is None:
            return ""
        else:
            return str(Path(args.work_dir) / (Path(args.input_file).stem + ".emb"))
    else:
        return args.save_emb


def _get_save_model(args: Namespace) -> str:
    """Determine the save_model path based on command-line arguments."""
    if args.save_model == "<workdir>":
        stem = (
            Path(args.input_file).stem
            if args.input_file is not None
            else Path(args.from_msa).stem
        )
        return str(Path(args.work_dir) / (stem + ".model"))
    else:
        return args.save_model if args.save_model is not None else ""
