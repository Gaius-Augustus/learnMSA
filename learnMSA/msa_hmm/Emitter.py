import tensorflow as tf
import numpy as np
import learnMSA.msa_hmm.Initializers as initializers
import learnMSA.msa_hmm.Priors as priors
    
    

class ProfileHMMEmitter(tf.keras.layers.Layer):
    """ An emitter defines emission distribution and prior for HMM states. This emitter in its default configuration 
        implements multinomial match distributions over the amino acid alphabet with a Dirichlet Prior.
        New emitters may be subclassed from the default emitter or made from scratch following this interface.
        Multiple emitters can be used jointly. 
    Args:
        emission_init: List of initializers for the match states (i.e. initializes a (model_len, s) matrix), one per model.
        insertion_init: List of initializers for the shared insertion states (i.e. initializes a (s) vector), one per model.
        prior: A compatible prior (or NullPrior).
        frozen_insertions: If true, insertions will not be trainable.
    """
    def __init__(self, 
                 emission_init = initializers.make_default_emission_init(),
                 insertion_init = initializers.make_default_insertion_init(),
                 prior = priors.AminoAcidPrior(),
                 frozen_insertions = True,
                 **kwargs
                 ):
        super(ProfileHMMEmitter, self).__init__(**kwargs)
        self.emission_init = [emission_init] if not hasattr(emission_init, '__iter__') else emission_init 
        self.insertion_init = [insertion_init] if not hasattr(insertion_init, '__iter__') else insertion_init 
        self.prior = prior
        self.frozen_insertions = frozen_insertions
        
    def cell_init(self, cell):
        """ Automatically called when the owner cell is created.
        """
        self.length = cell.length
        assert len(self.length) == len(self.emission_init), \
            f"The number of emission initializers ({len(self.emission_init)}) should match the number of models ({len(self.length)})."
        assert len(self.length) == len(self.insertion_init), \
            f"The number of insertion initializers ({len(self.insertion_init)}) should match the number of models ({len(self.length)})."
        self.max_num_states = cell.max_num_states
        self.num_models = cell.num_models
        self.prior.load(self.dtype)
        
    def build(self, input_shape):
        s = input_shape[-1]-1 # substract one for terminal symbol
        self.emission_kernel = [self.add_weight(
                                        shape=[length, s], 
                                        initializer=init, 
                                        name="emission_kernel_"+str(i)) 
                                    for i,(length, init) in enumerate(zip(self.length, self.emission_init))]
        self.insertion_kernel = [ self.add_weight(
                                shape=[s],
                                initializer=init,
                                name="insertion_kernel_"+str(i),
                                trainable=not self.frozen_insertions) 
                                    for i,init in enumerate(self.insertion_init)]
        self.built = True
        
    def recurrent_init(self):
        """ Automatically called before each recurrent run. Should be used for setups that
            are only required once per application of the recurrent layer.
        """
        self.B = self.make_B()
        self.B_transposed = tf.transpose(self.B, [0,2,1])
        
    def make_emission_matrix(self, em, ins, length):
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
        
    def make_B(self):
        emission_matrices = []
        for em, ins, length in zip(self.emission_kernel, self.insertion_kernel, self.length):
            em_mat = self.make_emission_matrix(em, ins, length) 
            padding = self.max_num_states - em_mat.shape[0]
            em_mat_pad = tf.pad(em_mat, [[0, padding], [0,0]])
            emission_matrices.append(em_mat_pad)
        B = tf.stack(emission_matrices, axis=0)
        return B
        
    def make_B_amino(self):
        """ A variant of make_B used for plotting the HMM. Can be overridden for more complex emissions. Per default this is equivalent to make_B
        """
        return self.make_B()
        
    def call(self, inputs):
        """ 
        Args: 
                inputs: Shape (k, b, s) (Shape (b, s) works as well if all models should get the same input.)
        Returns:
                Shape (k, b, q)
        """
        # batch matmul of k emission matrices with the b x s input matrix 
        # with broadcasting of the inputs
        return tf.matmul(inputs, self.B_transposed)
    
    def get_prior_log_density(self):
        return self.prior(self.make_B(), self.length)
    
    def duplicate(self, model_indices=None):
        if model_indices is None:
            model_indices = range(len(self.emission_init))
        sub_emission_init = [tf.constant_initializer(self.emission_kernel[i].numpy()) for i in model_indices]
        sub_insertion_init = [tf.constant_initializer(self.insertion_kernel[i].numpy()) for i in model_indices]
        emitter_copy = ProfileHMMEmitter(
                             emission_init = sub_emission_init,
                             insertion_init = sub_insertion_init,
                             prior = self.prior,
                             frozen_insertions = self.frozen_insertions,
                             dtype = self.dtype) 
        return emitter_copy
    
    def get_config(self):
        config = super(ProfileHMMEmitter, self).get_config()
        config.update({
             "emission_init" : [k.numpy() for k in self.emission_kernel],
             "insertion_init" : [k.numpy() for k in self.insertion_kernel],
             "prior" : self.prior,
             "frozen_insertions" : self.frozen_insertions
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        config["emission_init"] = [initializers.ConstantInitializer(k) for k in config["emission_init"]]
        config["insertion_init"] = [initializers.ConstantInitializer(k) for k in config["insertion_init"]]
        return cls(**config)
    
    def __repr__(self):
        return f"ProfileHMMEmitter(\n emission_init={self.emission_init[0]},\n insertion_init={self.insertion_init[0]},\n prior={self.prior},\n frozen_insertions={self.frozen_insertions}, )"