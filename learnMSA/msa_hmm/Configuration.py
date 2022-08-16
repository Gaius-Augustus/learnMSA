import numpy as np
import tensorflow as tf
import msa_hmm.Utility as ut


def as_str(config, items_per_line=3):
    return " , ".join(key + " : " + str(val) + "\n"*((i+1)%items_per_line==0) for i,(key,val) in enumerate(config.items()))


class EmissionInitializer(tf.keras.initializers.Initializer):

    def __init__(self, dist):
        self.dist = dist

    def __call__(self, shape, dtype=None, **kwargs):
        assert len(shape) == 2, "EmissionInitializer only supports 2D shapes."
        assert shape[-1] == self.dist.size, f"Last dimension of shape must match the size of the initial distribution. Shape={shape} dist.size={self.dist.size}"
        return tf.reshape(tf.tile(self.dist, shape[:1]), shape)
    

class EntryInitializer(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None, **kwargs):
        #choose such that entry[0] will always be ~0.5 independent of model length
        p0 = tf.zeros([1]+[d for d in shape[1:]], dtype=dtype)
        p = tf.cast(tf.repeat(tf.math.log(1/(shape[0]-1)), shape[0]-1), dtype=dtype)
        return tf.concat([p0, p], axis=0)
    
    
class ExitInitializer(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None, **kwargs):
        #choose such that all exit probs equal the probs entry[i] for i > 0 
        return tf.zeros(shape, dtype=dtype) + tf.cast(tf.math.log(0.5/(shape[0]-1)), dtype=dtype)
    

class MatchTransitionInitializer(tf.keras.initializers.Initializer):
    def __init__(self, val, i):
        self.val = val
        self.i = i
    
    def __call__(self, shape, dtype=None, **kwargs):
        p_exit_desired = 0.5 / (shape[0]-1)
        prob = (tf.nn.softmax(np.array(self.val, dtype=float)) * (1-p_exit_desired))[self.i]
        return tf.zeros(shape, dtype=dtype) + np.log(prob)
    
    
    
def make_default_emission_init():
    return EmissionInitializer(np.log(ut.background_distribution))


def make_default_insertion_init():
    return tf.constant_initializer(np.log(ut.background_distribution))


def make_default_flank_init():
    return tf.keras.initializers.Zeros()

    
def make_default_transition_init(MM=1, 
                                 MI=-1, 
                                 MD=-1, 
                                 II=-0.5, 
                                 IM=0, 
                                 DM=0, 
                                 DD=-0.5,
                                 FC=0, 
                                 FE=-1,
                                 R=-9, 
                                 RF=0, 
                                 T=0):
    transition_init_kernel = {
        "begin_to_match" : EntryInitializer(),
        "match_to_end" : ExitInitializer(),
        "match_to_match" : MatchTransitionInitializer([MM, MI, MD], 0),
        "match_to_insert" : MatchTransitionInitializer([MM, MI, MD], 1),
        "insert_to_match" : tf.constant_initializer(IM),
        "insert_to_insert" : tf.constant_initializer(II),
        "match_to_delete" : MatchTransitionInitializer([MM, MI, MD], 2),
        "delete_to_match" : tf.constant_initializer(DM),
        "delete_to_delete" : tf.constant_initializer(DD),
        "left_flank_loop" : tf.constant_initializer(FC),
        "left_flank_exit" : tf.constant_initializer(FE),
        "right_flank_loop" : tf.constant_initializer(FC),
        "right_flank_exit" : tf.constant_initializer(FE),
        "unannotated_segment_loop" : tf.constant_initializer(FC),
        "unannotated_segment_exit" : tf.constant_initializer(FE),
        "end_to_unannotated_segment" : tf.constant_initializer(R),
        "end_to_right_flank" : tf.constant_initializer(RF),
        "end_to_terminal" : tf.constant_initializer(T) }
    return transition_init_kernel


#simple adpative batch size depending on sequence length
#longer models and sequences require much more memory
def get_adaptive_batch_size(model_length, max_len):
    if max_len * model_length < 320 * 350:
        return 512
    if max_len * model_length < 700 * 650:
        return 256
    else:
        return 128


#the configuration can be changed by experienced users
#proper command line support for these parameters will be added in the future
default = {
    
    "transition_init" : make_default_transition_init(),
    "flank_init" : make_default_flank_init(),
    "emission_init" : make_default_emission_init(),
    "insertion_init" : make_default_insertion_init(),
    "alpha_flank" : 7000,
    "alpha_single" : 1e9,
    "alpha_frag" : 1e4,
    "max_surgery_runs" : 4,
    "length_init_quantile" : 0.5,
    "surgery_quantile" : 0.5,
    "min_surgery_seqs" : 1e4,
    "len_mul" : 0.8,
    "batch_size" : get_adaptive_batch_size,
    "learning_rate" : 0.1,
    "epochs" : [2, 1, 3],
    "use_prior" : True,
    "dirichlet_mix_comp_count" : 1,
    "use_anc_probs" : True,
    "keep_encoder" : False,
    "encoder_initializer" : [tf.keras.initializers.Zeros()],
    "frozen_insertions" : True
    
}