import numpy as np
import tensorflow as tf
import learnMSA.msa_hmm.Emitter as emit
import learnMSA.msa_hmm.Transitioner as trans
import learnMSA.msa_hmm.Initializers as initializers


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
    if max_seq_len < 200 and model_length < 200:
        return 512*num_devices
    elif max_seq_len < 520 and model_length < 290:
        return 256*num_devices
    elif max_seq_len < 800 and model_length < 500:
        return 128*num_devices
    else:
        return 64*num_devices

#the configuration can be changed by experienced users
#proper command line support for these parameters will be added in the future
def make_default(default_num_models = 5):
    default = {

        "num_models" : default_num_models,
        "transitioner" : trans.ProfileHMMTransitioner([initializers.make_default_transition_init() 
                                                                         for _ in range(default_num_models)],
                                                      [initializers.make_default_flank_init() 
                                                                         for _ in range(default_num_models)]),
        "emitter" : emit.ProfileHMMEmitter([initializers.make_default_emission_init()
                                                                         for _ in range(default_num_models)],
                                           [initializers.make_default_insertion_init()
                                                                         for _ in range(default_num_models)]),
        "max_surgery_runs" : 4,
        "length_init_quantile" : 0.5,
        "surgery_quantile" : 0.5,
        "min_surgery_seqs" : 1e4,
        "len_mul" : 0.8,
        "batch_size" : get_adaptive_batch_size,
        "learning_rate" : 0.1,
        "epochs" : [10, 2, 10],
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
        "allow_user_keys_in_config" : False
    }
    return default

def assert_config(config):
    assert "num_models" in config
    default = make_default(config["num_models"])
    for key in default:
        assert key in config, f"User configuration is missing key {key}."
    if not config["allow_user_keys_in_config"]:
        for key in config:
            assert key in default, f"Unrecognized key {key} in user configuration."