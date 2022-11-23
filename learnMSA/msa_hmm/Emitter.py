import tensorflow as tf
import numpy as np
import learnMSA.msa_hmm.Initializers as initializers
import learnMSA.msa_hmm.Priors as priors
    

def make_default_emission_matrix(em, ins, length):
    """Constructs an emission matrix from kernels with a shared insertion distribution.
    Args:
       s: Alphabet size.
       em: Emission distribution logits (kernel).
       ins: Shared insertion distribution (kernel).
       length: Model length.
    Returns:
        The emission matrix.
    """
    s = em.shape[-1]
    emissions = tf.concat([tf.expand_dims(ins, 0), 
                           em, 
                           tf.stack([ins]*(length+1))] , axis=0)
    emissions = tf.nn.softmax(emissions)
    emissions = tf.concat([emissions, tf.zeros_like(emissions[:,:1])], axis=-1) 
    end_state_emission = tf.one_hot([s], s+1, dtype=em.dtype) 
    emissions = tf.concat([emissions, end_state_emission], axis=0)
    return emissions

class ProfileHMMEmitter(tf.keras.layers.Layer):
    """ An emitter defines emission distribution and prior for HMM states. This emitter in its default configuration 
        implements multinomial match distributions over the amino acid alphabet with a Dirichlet Prior.
        New emitters may be subclassed from the default emitter or made from scratch following this interface.
        Multiple emitters can be used jointly. 
    Args:
        emission_init: Initializer for the match states (i.e. initializes a (model_len, s) matrix)
        insertion_init: Initializer for the shared insertion states (i.e. initializes a (s) vector)
        emission_func: A function that takes two arguments: Emission matrix (q, s) and inputs (b, s) 
                        and returns emission probabilities (b, q)
        emission_matrix_generator: Function that constructs the emission matrix from pieces (see make_default_emission_matrix).
        emission_prior: A compatible prior (or NullPrior).
        frozen_insertions: If true, insertions will not be trainable.
    """
    def __init__(self, 
                 emission_init = initializers.make_default_emission_init(),
                 insertion_init = initializers.make_default_insertion_init(),
                 emission_func = tf.linalg.matvec, 
                 emission_matrix_generator = make_default_emission_matrix,
                 prior = priors.AminoAcidPrior(),
                 frozen_insertions = True,
                 dtype = tf.float32,
                 **kwargs
                 ):
        super(ProfileHMMEmitter, self).__init__(name="ProfileHMMEmitter", dtype=dtype, **kwargs)
        self.emission_init = emission_init
        self.insertion_init = insertion_init
        self.emission_func = emission_func
        self.emission_matrix_generator = emission_matrix_generator
        self.prior = prior
        self.frozen_insertions = frozen_insertions
        
    def cell_init(self, cell):
        """ Automatically called when the owner cell is created.
        """
        self.length = cell.length
        self.prior.load(self.dtype)
        
    def build(self, input_shape):
        s = input_shape[-1]-1 # substract one for terminal symbol
        self.emission_kernel = self.add_weight(
                                        shape=[self.length, s], 
                                        initializer=self.emission_init, 
                                        name="emission_kernel")
        self.insertion_kernel = self.add_weight(
                                shape=[s],
                                initializer=self.insertion_init,
                                name="insertion_kernel",
                                trainable=not self.frozen_insertions) 
        
    def recurrent_init(self):
        """ Automatically called before each recurrent run. Should be used for setups that
            are only required once per application of the recurrent layer.
        """
        self.B = self.make_B()
        
    def make_B(self):
        return self.emission_matrix_generator(self.emission_kernel, self.insertion_kernel, self.length)
        
    def call(self, inputs):
        return self.emission_func(self.B, inputs)
    
    def get_prior_log_density(self):
        return self.prior(self.B)
    
    def duplicate(self):
        return ProfileHMMEmitter(
                 emission_init = self.emission_init,
                 insertion_init = self.insertion_init,
                 emission_func = self.emission_func, 
                 emission_matrix_generator = self.emission_matrix_generator,
                 prior = self.prior,
                 frozen_insertions = self.frozen_insertions,
                 dtype = self.dtype) 
    
    def __repr__(self):
        return f"ProfileHMMEmitter(emission_init={self.emission_init}, insertion_init={self.insertion_init}, emission_func={self.emission_func}, emission_matrix_generator={self.emission_matrix_generator}, prior={self.prior}, frozen_insertions={self.frozen_insertions}, )"