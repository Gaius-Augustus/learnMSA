import numpy as np
import tensorflow as tf
import learnMSA.msa_hmm.Emitter as emit
import learnMSA.msa_hmm.Transitioner as trans
import learnMSA.msa_hmm.Initializers as initializers


def as_str(config, items_per_line=3):
    return " , ".join(key + " : " + str(val) + "\n"*((i+1)%items_per_line==0) for i,(key,val) in enumerate(config.items()))

#simple adpative batch size depending on sequence length
#longer models and sequences require much more memory
def get_adaptive_batch_size(model_length, max_len):
    if max_len * model_length < 200 * 200:
        return 512
    if max_len * model_length < 500 * 500:
        return 256
    else:
        return 128

#the configuration can be changed by experienced users
#proper command line support for these parameters will be added in the future
default = {
    
    "transitioner" : trans.ProfileHMMTransitioner(),
    "emitter" : emit.ProfileHMMEmitter(),
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
    "encoder_initializer" : [tf.constant_initializer(-3), 
                             tf.constant_initializer(initializers.exchangeability_init),
                             tf.constant_initializer(np.log(initializers.p))],
    "surgery_del" : 0.5,
    "surgery_ins" : 0.5,
    "num_rate_matrices" : 1,
    "per_matrix_rate" : False,
    "matrix_rate_l2" : 0.0,
    "shared_rate_matrix" : False,
    "equilibrium_sample" : False,
    "transposed" : False
}

def assert_config(config):
    for key in default:
        assert key in config, f"User configuration is missing key {key}."
    for key in config:
        assert key in default, f"Unrecognized key {key} in user configuration."