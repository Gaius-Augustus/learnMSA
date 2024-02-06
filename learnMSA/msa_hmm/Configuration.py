import math
import numpy as np
import tensorflow as tf
from functools import partial
import learnMSA.msa_hmm.Emitter as emit
import learnMSA.msa_hmm.Transitioner as trans
import learnMSA.msa_hmm.Initializers as initializers
import learnMSA.msa_hmm.Priors as priors
import learnMSA.protein_language_models.Common as plm_common
from learnMSA.protein_language_models.MvnEmitter import MvnEmitter, AminoAcidPlusMvnEmissionInitializer, make_joint_prior


def as_str(config, items_per_line=1, prefix="", sep=""):
    return "\n"+prefix+"{" + sep.join(("\n"+prefix)*(i%items_per_line==0) + key + " : " + str(val) for i,(key,val) in enumerate(config.items())) + "\n"+prefix+"}"

#simple adpative batch size depending on sequence- length and model length
#longer models and sequences require much more memory
#we limit the batch size based on the longest model to train
#the adpative batch size scales automatically with the number of GPUs
def get_adaptive_batch_size(model_lengths, max_seq_len):
    num_gpu = len([x.name for x in tf.config.list_logical_devices() if x.device_type == 'GPU']) 
    num_devices = num_gpu + int(num_gpu==0) #account for the CPU-only case 
    model_length = max(model_lengths)
    if max_seq_len < 200 and model_length < 180:
        return 512*num_devices
    elif max_seq_len < 520 and model_length < 230:
        return 256*num_devices
    elif max_seq_len < 700 and model_length < 420:
        return 128*num_devices
    elif max_seq_len < 850 and model_length < 550:
        return 64*num_devices
    elif max_seq_len < 1200 and model_length < 700:
        return 32*num_devices
    elif max_seq_len < 2000 and model_length < 1000:
        return 8*num_devices
    elif max_seq_len < 4000 and model_length < 1500:
        return 4*num_devices
    else:
        return 2*num_devices
    
def get_adaptive_batch_size_with_language_model(model_lengths, max_seq_len, embedding_dim):
    num_gpu = len([x.name for x in tf.config.list_logical_devices() if x.device_type == 'GPU']) 
    num_devices = num_gpu + int(num_gpu==0) #account for the CPU-only case 
    model_length = max(model_lengths)
    if max_seq_len < 200 and model_length < 180:
        return (20 + 180*32//embedding_dim)*num_devices
    elif max_seq_len < 520 and model_length < 230:
        return (10 + 90*32//embedding_dim)*num_devices
    elif max_seq_len < 700 and model_length < 420:
        return (5 + 45*32//embedding_dim)*num_devices
    elif max_seq_len < 850 and model_length < 550:
        return (3 + 22*32//embedding_dim)*num_devices
    elif max_seq_len < 1200 and model_length < 700:
        return (1 + 9*32//embedding_dim)*num_devices
    elif max_seq_len < 2000 and model_length < 1000:
        return (1 + 4*32//embedding_dim)*num_devices
    elif max_seq_len < 4000 and model_length < 1500:
        return (1 + 32//embedding_dim)*num_devices
    else:
        return 1*num_devices

#the configuration can be changed by experienced users
#proper command line support for these parameters will be added in the future
def make_default(default_num_models=5, 
                 use_language_model=False, 
                 allow_user_keys_in_config=False,
                 use_l2=False,
                 scoring_model_config=plm_common.ScoringModelConfig(),
                 num_prior_components=plm_common.PRIOR_DEFAULT_COMPONENTS,
                 frozen_insertions=True,
                 L2_match=10.,
                 L2_insert=0.,
                 temperature_mode="trainable",
                 conditionally_independent=True,
                 V2_emitter=True,
                 V2_full_covariance=False,
                 V2_temperature=100.):
    if use_language_model:
        if V2_emitter:
            emission_init = [AminoAcidPlusMvnEmissionInitializer(scoring_model_config=scoring_model_config,
                                                                        num_prior_components=num_prior_components) 
                                                                            for _ in range(default_num_models)]
            insertion_init = [AminoAcidPlusMvnEmissionInitializer(scoring_model_config=scoring_model_config,
                                                                        num_prior_components=num_prior_components)  
                                                                            for _ in range(default_num_models)]
            emitter = MvnEmitter(scoring_model_config, 
                                emission_init=emission_init, 
                                insertion_init=insertion_init,
                                num_prior_components=num_prior_components,
                                full_covariance=V2_full_covariance,
                                temperature=V2_temperature,
                                frozen_insertions=frozen_insertions)
        else:
            emission_init = [initializers.EmbeddingEmissionInitializer(scoring_model_config=scoring_model_config,
                                                                        num_prior_components=num_prior_components) 
                                                                            for _ in range(default_num_models)]
            insertion_init = [initializers.EmbeddingEmissionInitializer(scoring_model_config=scoring_model_config,
                                                                        num_prior_components=num_prior_components)  
                                                                            for _ in range(default_num_models)]
            #if num_prior_components == 0:
            prior = priors.L2Regularizer(L2_match=L2_match, L2_insert=L2_insert)
            # else:
            #     prior = priors.MvnEmbeddingPrior(scoring_model_config, num_prior_components, use_l2=use_l2, L2_match=L2_match, L2_insert=L2_insert)
            emitter = emit.EmbeddingEmitter(scoring_model_config, 
                                            emission_init=emission_init, 
                                            insertion_init=insertion_init,
                                            prior=prior,
                                            frozen_insertions=frozen_insertions,
                                            temperature_mode=emit.TemperatureMode.from_string(temperature_mode),
                                            conditionally_independent=conditionally_independent)
    else:
        emitter = emit.ProfileHMMEmitter([initializers.make_default_emission_init()
                                                                         for _ in range(default_num_models)],
                                           [initializers.make_default_insertion_init()
                                                                         for _ in range(default_num_models)])

    transitioner = trans.ProfileHMMTransitioner([initializers.make_default_transition_init() 
                                                                        for _ in range(default_num_models)],
                                                    [initializers.make_default_flank_init() 
                                                                        for _ in range(default_num_models)])
    if use_language_model:                                                                    
        batch_callback = partial(get_adaptive_batch_size_with_language_model, embedding_dim=scoring_model_config.dim)  
    else:
        batch_callback = get_adaptive_batch_size                                                                  
    default = {
        "num_models" : default_num_models,
        "transitioner" : transitioner,
        "emitter" : emitter,
        "max_surgery_runs" : 4,
        "length_init_quantile" : 0.5,
        "surgery_quantile" : 0.5,
        "min_surgery_seqs" : 1e5,
        "len_mul" : 0.8,
        "batch_size" : batch_callback,
        "learning_rate" : 0.05 if use_language_model else 0.1,
        "epochs" : [10, 4, 20] if use_language_model else [10, 2, 10],
        "crop_long_seqs" : math.inf,
        "use_prior" : True,
        "dirichlet_mix_comp_count" : 1,
        "use_anc_probs" : True,
        "trainable_rate_matrices" : False,
        "surgery_del" : 0.5,
        "surgery_ins" : 0.5,
        "num_rate_matrices" : 1,
        "per_matrix_rate" : False,
        "matrix_rate_l2" : 0.0,
        "shared_rate_matrix" : False,
        "equilibrium_sample" : False,
        "transposed" : False,
        "encoder_initializer" : initializers.make_default_anc_probs_init(default_num_models),
        "model_criterion" : "AIC", #AIC is slightly better than loglik on average over multiple benchmarks
        "encoder_weight_extractor" : None,
        "experimental_evolve_upper_half" : False,
        "cluster_seq_id" : 0.5 if use_language_model else 0.9,
        "use_language_model" : use_language_model,
        "frozen_insertions" : frozen_insertions,
        "allow_user_keys_in_config" : allow_user_keys_in_config
    }
    if use_language_model:
        default.update({
            "scoring_model_config" : scoring_model_config,
            "mvn_prior_components" : num_prior_components,
            "use_l2" : use_l2,
            "L2_match" : L2_match,
            "L2_insert" : L2_insert,
            "temperature_mode" : temperature_mode,
            "conditionally_independent" : conditionally_independent,
            "V2_emitter" : V2_emitter,
            "V2_full_covariance" : V2_full_covariance,
            "V2_temperature" : V2_temperature
        })
    return default

def _make_assert_text(message, current_value):
    return message + f" Your input was {current_value}."

def assert_config(config):
    assert config["max_surgery_runs"] > 0, \
        _make_assert_text("Requires as least 1 surgery run.", config["max_surgery_runs"])
    assert config["length_init_quantile"] >= 0. and config["length_init_quantile"] <= 1., \
        _make_assert_text("The given quantile is not in range [0,1].", config["length_init_quantile"])
    assert config["surgery_quantile"] >= 0. and config["surgery_quantile"] <= 1., \
        _make_assert_text("The given quantile is not in range [0,1].", config["surgery_quantile"])
    assert config["len_mul"] >= 0., \
        _make_assert_text("The multiplier must be greater than zero.", config["surgery_quantile"])
    assert "num_models" in config
    if config["use_language_model"]:
        default = make_default(config["num_models"], 
                                scoring_model_config=config["scoring_model_config"],
                                num_prior_components=config["mvn_prior_components"],
                                use_language_model=True)
    else:
        default = make_default(config["num_models"], use_language_model=False)
    for key in default:
        assert key in config, f"User configuration is missing key {key}."
    if not config["allow_user_keys_in_config"]:
        for key in config:
            assert key in default, f"Unrecognized key {key} in user configuration."