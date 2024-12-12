import tensorflow as tf
import numpy as np
import learnMSA.msa_hmm.Initializers as initializers
from learnMSA.msa_hmm.SequenceDataset import SequenceDataset
from learnMSA.msa_hmm.Utility import inverse_softplus

# tmp include
import sys
sys.path.insert(0, "../TensorTree")
import tensortree 

tensortree.set_backend("tensorflow")



class AncProbsLayer(tf.keras.layers.Layer): 
    """ Computes the probabilities of amino acid exchanges given a sequence 
        using rate matrices and evolutionary times.
        Per default the input is the present and the output is the future.

    Args:
        num_models: The number of independently trained models.
        num_times: The total number of evolutionary times (inputs will index into a tensor of this size).
        equilibrium_init: Initializer for the kernel of the equilibrium distribution.
        exchangeability_init: Initializer for the kernel of the exchangeability matrices.
        time_init: Initializer for the kernel of the evolutionary times.
        trainable_rate_matrices: Makes equilibrium and exchangeabilities learnable.
        trainable_times: Makes the evolutionary times learnable.
        transposed: If True, the input is the past and the output is the present.
                    When set to False, the input is the present and the output is the past (the outputs
                    will no longer sum to one).
    """

    def __init__(self, 
                 num_models,
                 num_times, 
                 equilibrium_init=initializers.ConstantInitializer(0.),
                 exchangeability_init=initializers.ConstantInitializer(0.),
                 time_init=initializers.ConstantInitializer(-3.),
                 trainable_rate_matrices=False,
                 trainable_times=True,
                 transposed=True,
                 **kwargs):
        super(AncProbsLayer, self).__init__(**kwargs)
        self.num_models = num_models
        self.num_times = num_times
        self.equilibrium_init = equilibrium_init
        self.exchangeability_init = exchangeability_init
        self.time_init = time_init
        self.trainable_rate_matrices = trainable_rate_matrices
        self.trainable_times = trainable_times
        self.transposed = transposed
    
    
    def build(self, input_shape=None):
        if self.built:
            return
        self.time_kernel = self.add_weight(shape=[self.num_times, self.num_models], 
                                    name="time_kernel", 
                                    initializer=self.time_init,
                                    trainable=self.trainable_times)
        self.exchangeability_kernel = self.add_weight(shape=[self.num_models, 20, 20], 
                                                        name="exchangeability_kernel", 
                                                        initializer=self.exchangeability_init,
                                                        trainable=self.trainable_rate_matrices)
        self.equilibrium_kernel = self.add_weight(shape=[self.num_models, 20], 
                                                    name="equilibrium_kernel",
                                                    initializer=self.equilibrium_init,
                                                    trainable=self.trainable_rate_matrices)
        self.built = True

    
    """ Returns a stack of positive definite symmetric matrices with zero diagonal
        which are parameterized by the exchangeability kernel.
    Returns:
        A tensor of shape (num_models, 20, 20)
    """
    def make_exchangeability_matrix(self):
        if not self.built:
            self.build()
        return tensortree.backend.make_symmetric_pos_semidefinite(self.exchangeability_kernel)
    

    """ Returns a stack of equilibrium distributions parameterized by the equilibrium kernel.
    Returns:
        A tensor of shape (num_models, 20)
    """
    def make_equilibrium(self):
        if not self.built:
            self.build()
        return tensortree.backend.make_equilibrium(self.equilibrium_kernel)
    

    """ Returns a stack of rate matrices.
    Returns:
        A tensor of shape (num_models, 20, 20)
    """
    def make_rate_matrix(self):
        R, p = self.make_exchangeability_matrix(), self.make_equilibrium()
        return tensortree.backend.make_rate_matrix(R, p)

    
    """ Returns (a subset of) the evolutionary times.
    Args: 
        indices: 2D indices of shape (b, num_models) for the evolutionary times to gather. If None, all times are returned.
    Returns:
        A tensor of shape (b, num_models)
    """
    def make_times(self, indices=None):
        if not self.built:
            self.build()
        if indices is not None:
            t = tf.gather_nd(self.time_kernel, indices)
            if hasattr(indices, "shape"):
                t.set_shape(indices.shape)
        else:
            t = self.time_kernel
        return tensortree.backend.make_branch_lengths(t)
    

    def call(self, inputs, indices, replace_rare_with_equilibrium=True):
        """ Computes probabilities of amino acid exchanges. The input can have any 
            number z of non-standard amino acids. The inputs can be padded. This is
            expected to be indicated by a 1 at the last position.
        Args:
            inputs: One hot encoded input sequences. Shape: (b, num_models, L, 20 + z + 1).
            indices: Indices that map each input sequences to an evolutionary time. Shape: (b, num_model)
            replace_rare_with_equilibrium: If true, replaces non-standard amino acids with the equilibrium distribution.
        Returns:
            Ancestral probabilities. Shape: (b, num_model, L, 20 + 1) if replace_rare_with_equilibrium 
                                    else (b, num_model, L, 20 + z + 1)
        """
        # omit non-standard amino acids and traverse the evolutionary times
        std_inputs = inputs[..., :20]
        Q = self.make_rate_matrix()
        B = self.make_times(indices)
        P = tensortree.model.traverse_branches(std_inputs, Q, B, 
                                               transposed=self.transposed,
                                               logarithmic=False)
        # add non-standard amino acids back
        if inputs.shape[-1] > 20:
            z = inputs.shape[-1] - 21
            if replace_rare_with_equilibrium:
                equilibrium = self.make_equilibrium()
                rest = equilibrium[tf.newaxis,:,tf.newaxis] * tf.reduce_sum(inputs[..., 20:-1], axis=-1, keepdims=True)
                P += rest
                # account for the case that P might be broadcasted but inputs not
                other = tf.zeros_like(P[..., -1:]) + inputs[...,-1:]
                P = tf.concat([P, other], axis=-1)
            else:
                other = tf.zeros(P.shape[:-1] + (z+1), dtype=P.dtype) + inputs[..., 20:]
                P = tf.concat([P, other], axis=-1)
        return P
        
        
    def get_config(self):
        config = super(AncProbsLayer, self).get_config()
        config.update({
                "num_models" : self.num_models,
                "num_times" : self.num_times,
                "equilibrium_init" : initializers.ConstantInitializer(self.equilibrium_kernel.numpy()),
                "exchangeability_init" : initializers.ConstantInitializer(self.exchangeability_kernel.numpy()),
                "time_init" : initializers.ConstantInitializer(self.time_kernel.numpy()),
                "trainable_rate_matrices" : self.trainable_rate_matrices,
                "trainable_times" : self.trainable_times,
                "transposed" : self.transposed
                })
        return config
