import tensorflow as tf
import numpy as np
from learnMSA.msa_hmm.Utility import inverse_softplus, deserialize
from learnMSA.msa_hmm.SequenceDataset import SequenceDataset
import learnMSA.msa_hmm.DirichletMixture as dm
import os

# tmp include
import sys
sys.path.insert(0, "../TensorTree")
import tensortree



class EmissionInitializer(tf.keras.initializers.Initializer):

    def __init__(self, dist):
        self.dist = dist

    def __call__(self, shape, dtype=None, **kwargs):
        assert shape[-1] == self.dist.size, f"Last dimension of shape must match the size of the initial distribution. Shape={shape} dist.size={self.dist.size}"
        dist = tf.cast(self.dist, dtype)
        return tf.reshape(tf.tile(dist, tf.cast(tf.math.reduce_prod(shape[:-1], keepdims=True), tf.int32)), shape)
    
    def __repr__(self):
        return f"EmissionInitializer()"

    def get_config(self):  # To support serialization
        return {"dist": self.dist.tolist()}

    @classmethod
    def from_config(cls, config):
        return cls(np.array(config["dist"]))
    

class ConstantInitializer(tf.keras.initializers.Constant):

    def __init__(self, value):
        super(ConstantInitializer, self).__init__(value)

    def __repr__(self):
        if np.isscalar(self.value):
            return f"Const({self.value})"
        elif isinstance(self.value, list):
            return f"Const(size={len(self.value)})"
        else:
            return f"Const(shape={self.value.shape})"

    def get_config(self):  # To support serialization
        return {"value": self.value.tolist() if isinstance(self.value, np.ndarray) else self.value}

    @classmethod
    def from_config(cls, config):
        return cls(np.array(config["value"]))
    


prior_path = os.path.dirname(__file__)+"/trained_prior/"
model_path = prior_path+"_".join([str(1), "True", "float32", "_dirichlet.h5"])
model = dm.load_mixture_model(model_path, 1, 20, trainable=False, dtype=tf.float32)
dirichlet = model.layers[-1]
background_distribution = dirichlet.expectation()
#the prior was trained on example distributions over the 20 amino acid alphabet
#the additional frequencies for 'B', 'Z',  'X', 'U', 'O' were derived from Pfam
extra = [7.92076933e-04, 5.84256792e-08, 1e-32]
background_distribution = np.concatenate([background_distribution, extra], axis=0)
background_distribution /= np.sum(background_distribution)



def make_LG_init(num_models):
    R, p = tensortree.substitution_models.LG(alphabet = SequenceDataset.alphabet[:20])
    R_init = np.stack([inverse_softplus(R).numpy()]*num_models, axis=0)
    p_init = np.stack([np.log(p)]*num_models, axis=0)
    return [ConstantInitializer(R_init), 
            ConstantInitializer(p_init)]


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

    def get_config(self):  # To support serialization
        return {"val": self.val, "i": self.i, "scale": self.scale}


    
class RandomNormalInitializer(tf.keras.initializers.Initializer):
    def __init__(self, mean=0.0, stddev=0.05):
        self.mean = mean
        self.stddev = stddev
    
    def __call__(self, shape, dtype=None, **kwargs):
        return tf.random.normal(shape, mean=self.mean, stddev=self.stddev, dtype=dtype if dtype != None else tf.float32)
        
    def __repr__(self):
        return f"Norm({self.mean}, {self.stddev})"

    def get_config(self):  # To support serialization
        return {"mean": self.mean, "stddev": self.stddev}
    
    
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


tf.keras.utils.get_custom_objects()["EmissionInitializer"] = EmissionInitializer
tf.keras.utils.get_custom_objects()["ConstantInitializer"] = ConstantInitializer
tf.keras.utils.get_custom_objects()["EntryInitializer"] = EntryInitializer
tf.keras.utils.get_custom_objects()["ExitInitializer"] = ExitInitializer
tf.keras.utils.get_custom_objects()["MatchTransitionInitializer"] = MatchTransitionInitializer
tf.keras.utils.get_custom_objects()["RandomNormalInitializer"] = RandomNormalInitializer