from learnMSA import Configuration
from learnMSA.msa_hmm.learnmsa_context import LearnMSAContext

def make_legacy_config(
    config : Configuration, context : LearnMSAContext
) -> dict:
    # Temporary to keep all the subsytems untouched
    # Convert the config to a dictionary for now
    legacy_config = {
        "num_models" : config.training.num_model,
        "transitioner" : context.transitioner,
        "emitter" : context.emitter,
        "max_surgery_runs" : config.training.max_iterations,
        "length_init_quantile" : config.training.length_init_quantile,
        "surgery_quantile" : config.training.surgery_quantile,
        "min_surgery_seqs" : config.training.min_surgery_seqs,
        "len_mul" : config.training.len_mul,
        "batch_size" : context.batch_size,
        "learning_rate" : config.training.learning_rate,
        "epochs" : config.training.epochs,
        "crop_long_seqs" : config.training.crop,
        "use_prior" : config.training.use_prior,
        "dirichlet_mix_comp_count" : config.training.dirichlet_mix_comp_count,
        "use_anc_probs" : config.training.use_anc_probs,
        "trainable_rate_matrices" : config.training.trainable_rate_matrices,
        "trainable_distances" : config.training.trainable_distances,
        "surgery_del" : config.training.surgery_del,
        "surgery_ins" : config.training.surgery_ins,
        "num_rate_matrices" : config.training.num_rate_matrices,
        "per_matrix_rate" : config.training.per_matrix_rate,
        "matrix_rate_l2" : config.training.matrix_rate_l2,
        "shared_rate_matrix" : config.training.shared_rate_matrix,
        "equilibrium_sample" : config.training.equilibrium_sample,
        "transposed" : config.training.transposed,
        "encoder_initializer" : context.encoder_initializer,
        "model_criterion" : config.training.model_criterion,
        "encoder_weight_extractor" : context.encoder_weight_extractor,
        "cluster_seq_id" : config.training.cluster_seq_id,
        "use_language_model" : config.language_model.use_language_model,
        "frozen_insertions" : config.training.frozen_insertions,
        "allow_user_keys_in_config" : True
    }
    if config.language_model.use_language_model:
        legacy_config.update({
            "scoring_model_config" : context.scoring_model_config,
            "mvn_prior_components" : config.language_model.embedding_prior_components,
            "use_l2" : config.language_model.use_L2,
            "L2_match" : config.language_model.L2_match,
            "L2_insert" : config.language_model.L2_insert,
            "temperature_mode" : config.language_model.temperature_mode,
            "conditionally_independent" : config.language_model.conditionally_independent,
            "V2_emitter" : True,
            "V2_full_covariance" : False,
            "V2_temperature" : config.language_model.temperature,
            "plm_cache_dir" : config.language_model.plm_cache_dir,
        })
    return legacy_config