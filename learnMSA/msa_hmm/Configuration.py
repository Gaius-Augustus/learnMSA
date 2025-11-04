import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import Initializer
from functools import partial
import learnMSA.msa_hmm.Emitter as emit
import learnMSA.msa_hmm.Transitioner as trans
import learnMSA.msa_hmm.Initializers as initializers
import learnMSA.msa_hmm.Priors as priors
import learnMSA.protein_language_models.Common as plm_common
from learnMSA.protein_language_models.MvnEmitter import MvnEmitter, AminoAcidPlusMvnEmissionInitializer, make_joint_prior
import subprocess as sp
import os


def as_str(config, items_per_line=1, prefix="", sep=""):
    return "\n"+prefix+"{" + sep.join(("\n"+prefix)*(i%items_per_line==0) + key + " : " + str(val) for i,(key,val) in enumerate(config.items())) + "\n"+prefix+"}"

#simple adpative batch size depending on sequence- length and model length
#longer models and sequences require much more memory
#we limit the batch size based on the longest model to train
#the adpative batch size scales automatically with the number of GPUs
def get_adaptive_batch_size(model_lengths, max_seq_len, small_gpu):
    num_gpu = len([x.name for x in tf.config.list_logical_devices() if x.device_type == 'GPU'])
    num_devices = num_gpu + int(num_gpu==0) #account for the CPU-only case
    model_length = max(model_lengths) if len(model_lengths) > 0 else 0
    if max_seq_len < 200 and model_length < 180:
        batch_size = 512*num_devices
    elif max_seq_len < 520 and model_length < 230:
        batch_size = 256*num_devices
    elif max_seq_len < 700 and model_length < 420:
        batch_size = 128*num_devices
    elif max_seq_len < 850 and model_length < 550:
        batch_size = 64*num_devices
    elif max_seq_len < 1200 and model_length < 700:
        batch_size = 32*num_devices
    elif max_seq_len < 2000 and model_length < 1000:
        batch_size = 8*num_devices
    elif max_seq_len < 4000 and model_length < 1500:
        batch_size = 4*num_devices
    else:
        batch_size = 2*num_devices
    if small_gpu:
        batch_size = max(1, batch_size//2)
    return max(1, batch_size)

def get_adaptive_batch_size_with_language_model(model_lengths, max_seq_len, embedding_dim, small_gpu):
    num_gpu = len([x.name for x in tf.config.list_logical_devices() if x.device_type == 'GPU'])
    num_devices = num_gpu + int(num_gpu==0) #account for the CPU-only case
    model_length = max(model_lengths) if len(model_lengths) > 0 else 0
    if max_seq_len < 200 and model_length < 180:
        batch_size = (20 + 180*32//embedding_dim)*num_devices
    elif max_seq_len < 520 and model_length < 230:
        batch_size = (10 + 90*32//embedding_dim)*num_devices
    elif max_seq_len < 700 and model_length < 420:
        batch_size = (5 + 45*32//embedding_dim)*num_devices
    elif max_seq_len < 850 and model_length < 550:
        batch_size = (3 + 22*32//embedding_dim)*num_devices
    elif max_seq_len < 1200 and model_length < 700:
        batch_size = (1 + 9*32//embedding_dim)*num_devices
    elif max_seq_len < 2000 and model_length < 1000:
        batch_size = (1 + 4*32//embedding_dim)*num_devices
    elif max_seq_len < 4000 and model_length < 1500:
        batch_size = (1 + 32//embedding_dim)*num_devices
    else:
        batch_size = 1*num_devices
    if small_gpu:
        batch_size = max(1, batch_size//2)
    return max(1, batch_size)

def tokens_per_batch_to_batch_size(model_lengths, max_seq_len, tokens_per_batch):
    #convert tokens per batch to batch size
    _model_length = max(model_lengths) if len(model_lengths) > 0 else 0 # unused
    if max_seq_len < 1:
        return 1
    return max(1, tokens_per_batch // max_seq_len)

#the configuration can be changed by experienced users
#proper command line support for these parameters will be added in the future
def make_default(
    default_num_models=5,
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
    V2_temperature=3.,
    inv_gamma_alpha=3.,
    inv_gamma_beta=0.5,
    plm_cache_dir=None,
    emission_init : Initializer | list[Initializer] | None = None,
    insertion_init : Initializer | list[Initializer] | None = None,
    transition_init : Initializer | list[Initializer] | None = None,
    flank_init : Initializer | list[Initializer] | None = None,
) -> None:
    if use_language_model:
        if V2_emitter:
            if emission_init is None:
                emission_init = [
                    AminoAcidPlusMvnEmissionInitializer(
                        scoring_model_config=scoring_model_config,
                        num_prior_components=num_prior_components
                    )
                for _ in range(default_num_models)]
            if insertion_init is None:
                insertion_init = [
                    AminoAcidPlusMvnEmissionInitializer(
                        scoring_model_config=scoring_model_config,
                        num_prior_components=num_prior_components
                    )
                    for _ in range(default_num_models)]
            emitter = MvnEmitter(
                scoring_model_config,
                emission_init=emission_init,
                insertion_init=insertion_init,
                num_prior_components=num_prior_components,
                full_covariance=V2_full_covariance,
                temperature=V2_temperature,
                frozen_insertions=frozen_insertions,
                inv_gamma_alpha=inv_gamma_alpha,
                inv_gamma_beta=inv_gamma_beta
            )
        else:
            if emission_init is None:
                emission_init = [
                    initializers.EmbeddingEmissionInitializer(
                        scoring_model_config=scoring_model_config,
                        num_prior_components=num_prior_components
                    )
                    for _ in range(default_num_models)
                ]
            if insertion_init is None:
                insertion_init = [
                    initializers.EmbeddingEmissionInitializer(
                        scoring_model_config=scoring_model_config,
                        num_prior_components=num_prior_components
                    )
                    for _ in range(default_num_models)
                ]
            prior = priors.L2Regularizer(L2_match=L2_match, L2_insert=L2_insert)
            emitter = emit.EmbeddingEmitter(
                scoring_model_config,
                emission_init=emission_init,
                insertion_init=insertion_init,
                prior=prior,
                frozen_insertions=frozen_insertions,
                temperature_mode=emit.TemperatureMode.from_string(temperature_mode),
                conditionally_independent=conditionally_independent
            )
    else:
        if emission_init is None:
            emission_init = [
                initializers.make_default_emission_init()
                for _ in range(default_num_models)
            ]
        if insertion_init is None:
            insertion_init = [
                initializers.make_default_insertion_init()
                for _ in range(default_num_models)
            ]
        emitter = emit.ProfileHMMEmitter(emission_init, insertion_init)

    if transition_init is None:
        transition_init = [initializers.make_default_transition_init()
                for _ in range(default_num_models)]
    if flank_init is None:
        flank_init = [initializers.make_default_flank_init()
                for _ in range(default_num_models)]
    transitioner = trans.ProfileHMMTransitioner(transition_init, flank_init)

    #automaticall scale to a memory friendly version, if the GPU has less than 32GB
    small_gpu = False
    if len([x.name for x in tf.config.list_logical_devices() if x.device_type == 'GPU']) > 0:
        #if there is at least one GPU, check its memory
        gpu_mem = get_gpu_memory()
        small_gpu = gpu_mem[0] < 32000 if len(gpu_mem) > 0 else False
    if use_language_model:
        batch_callback = partial(get_adaptive_batch_size_with_language_model, embedding_dim=scoring_model_config.dim, small_gpu=small_gpu)
    else:
        batch_callback = partial(get_adaptive_batch_size, small_gpu=small_gpu)
    default = {
        "num_models" : default_num_models,
        "transitioner" : transitioner,
        "emitter" : emitter,
        "max_surgery_runs" : 2,
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
        "trainable_distances" : True,
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
            "V2_temperature" : V2_temperature,
            "plm_cache_dir" : plm_cache_dir,
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


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.total --format=csv"
    try:
        memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    except sp.CalledProcessError as e:
        print("Warning: There were GPU(s) detected, but nvidia-smi failed to run. It is used to infer GPU memory and adapt the batch size.")
        print("Please make sure nvidia-smi is installed and working properly. It might also mean that you are not running an NVIDIA GPU.")
        print("learnMSA will continue with default settings and might behave as expected. You can adjust the batch size manually with the -b option.")
        return []
    return memory_free_values