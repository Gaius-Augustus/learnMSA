import tensorflow as tf
import numpy as np
import learnMSA.msa_hmm.Fasta as fasta
import learnMSA.msa_hmm.AncProbsLayer as anc_probs
import learnMSA.msa_hmm.DirichletMixture as dm
import os

class EmissionInitializer(tf.keras.initializers.Initializer):

    def __init__(self, dist):
        self.dist = dist

    def __call__(self, shape, dtype=None, **kwargs):
        assert shape[-1] == self.dist.size, f"Last dimension of shape must match the size of the initial distribution. Shape={shape} dist.size={self.dist.size}"
        dist = tf.cast(self.dist, dtype)
        return tf.reshape(tf.tile(dist, tf.math.reduce_prod(shape[:-1], keepdims=True)), shape)
    
    def __repr__(self):
        return f"DefaultEmission()"
    
class ConstantInitializer(tf.keras.initializers.Constant):
    def __repr__(self):
        if np.isscalar(self.value):
            return f"Const({self.value})"
        else:
            return f"Const(shape={self.value.shape})"
    
R, p = anc_probs.parse_paml(anc_probs.LG_paml, fasta.alphabet[:-1])
exchangeability_init = anc_probs.inverse_softplus(R + 1e-32)



prior_path = os.path.dirname(__file__)+"/trained_prior/"
model_path = prior_path+"_".join([str(1), "True", "float32", "_dirichlet/ckpt"])
model = dm.load_mixture_model(model_path, 1, 20, trainable=False, dtype=tf.float32)
dirichlet = model.layers[-1]
background_distribution = dirichlet.expectation()
#the prior was trained on example distributions over the 20 amino acid alphabet
#the additional frequencies for 'B', 'Z',  'X', 'U', 'O' were derived from Pfam
extra = [2.03808244e-05, 1.02731819e-05, 7.92076933e-04, 5.84256792e-08, 1e-32]
background_distribution = np.concatenate([background_distribution, extra], axis=0)
background_distribution /= np.sum(background_distribution)

def make_default_anc_probs_init(num_models):
    exchangeability_stack = np.stack([exchangeability_init]*num_models, axis=0)
    log_p_stack = np.stack([np.log(p)]*num_models, axis=0)
    return [ConstantInitializer(-3), 
            ConstantInitializer(exchangeability_stack), 
            ConstantInitializer(log_p_stack)]
    
def make_default_emission_init():
    return EmissionInitializer(np.log(background_distribution))


def make_default_insertion_init():
    return ConstantInitializer(np.log(background_distribution))


class EntryInitializer(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None, **kwargs):
        #choose such that entry[0] will always be ~0.5 independent of model length
        p0 = tf.zeros([1]+[d for d in shape[1:]], dtype=dtype)
        p = tf.cast(tf.repeat(tf.math.log(1/(shape[0]-1)), shape[0]-1), dtype=dtype)
        return tf.concat([p0, p], axis=0)
    
    def __repr__(self):
        return f"DefaultEntry()"
    
    
class ExitInitializer(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None, **kwargs):
        #choose such that all exit probs equal the probs entry[i] for i > 0 
        return tf.zeros(shape, dtype=dtype) + tf.cast(tf.math.log(0.5/(shape[0]-1)), dtype=dtype)
    
    def __repr__(self):
        return f"DefaultExit()"
    

class MatchTransitionInitializer(tf.keras.initializers.Initializer):
    def __init__(self, val, i, scale):
        self.val = val
        self.i = i
        self.scale = scale
    
    def __call__(self, shape, dtype=None, **kwargs):
        val = tf.constant(self.val, dtype=dtype)[tf.newaxis,:]
        z = tf.random.normal(shape, stddev=self.scale, dtype=dtype)[:,tf.newaxis]
        val_z = val + z
        p_exit_desired = 0.5 / (shape[0]-1)
        prob = (tf.nn.softmax(val_z) * (1-p_exit_desired))[:,self.i]
        return tf.math.log(prob)
    
    def __repr__(self):
        return f"DefaultMatchTransition({self.val[self.i]})"
    
class RandomNormalInitializer(tf.keras.initializers.Initializer):
    def __init__(self, mean=0.0, stddev=0.05):
        self.mean = mean
        self.stddev = stddev
    
    def __call__(self, shape, dtype=None, **kwargs):
        return tf.random.normal(shape, mean=self.mean, stddev=self.stddev, dtype=dtype if dtype != None else tf.float32)
        
    def __repr__(self):
        return f"Norm({self.mean}, {self.stddev})"
    
def make_default_flank_init():
    return ConstantInitializer(0.)

    
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
                                 T=0,
                                 scale=0.1):
    transition_init_kernel = {
        "begin_to_match" : EntryInitializer(),
        "match_to_end" : ExitInitializer(),
        "match_to_match" : MatchTransitionInitializer([MM, MI, MD], 0, scale),
        "match_to_insert" : MatchTransitionInitializer([MM, MI, MD], 1, scale),
        "insert_to_match" : RandomNormalInitializer(IM, scale),
        "insert_to_insert" : RandomNormalInitializer(II, scale),
        "match_to_delete" : MatchTransitionInitializer([MM, MI, MD], 2, scale),
        "delete_to_match" : RandomNormalInitializer(DM, scale),
        "delete_to_delete" : RandomNormalInitializer(DD, scale),
        "left_flank_loop" : RandomNormalInitializer(FC, scale),
        "left_flank_exit" : RandomNormalInitializer(FE, scale),
        "right_flank_loop" : RandomNormalInitializer(FC, scale),
        "right_flank_exit" : RandomNormalInitializer(FE, scale),
        "unannotated_segment_loop" : RandomNormalInitializer(FC, scale),
        "unannotated_segment_exit" : RandomNormalInitializer(FE, scale),
        "end_to_unannotated_segment" : RandomNormalInitializer(R, scale),
        "end_to_right_flank" : RandomNormalInitializer(RF, scale),
        "end_to_terminal" : RandomNormalInitializer(T, scale) }
    return transition_init_kernel