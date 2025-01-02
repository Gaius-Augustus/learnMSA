import tensorflow as tf
import numpy as np
import os
import learnMSA.msa_hmm.Initializers as initializers
import learnMSA.msa_hmm.Priors as priors
from learnMSA.msa_hmm.SequenceDataset import SequenceDataset
from learnMSA.msa_hmm.Utility import get_num_states, deserialize
# from learnMSA.protein_language_models.BilinearSymmetric import make_scoring_model
# import learnMSA.protein_language_models.Common as Common
from packaging import version
from tensorflow.python.client import device_lib
from enum import Enum
import inspect
import math


class ProfileHMMEmitter(tf.keras.layers.Layer):
    """ An emitter defines emission distribution and prior. 
        This emitter in its default configuration implements multinomial match distributions over the amino acid alphabet with a Dirichlet Prior.
        New emitters may be subclassed from the default emitter or made from scratch following the same interface.
        Todo: Implement an emitter base class.
        emission_init: List of initializers for the match states (i.e. initializes a (model_len, s) matrix), one per model.
        insertion_init: List of initializers for the shared insertion states (i.e. initializes a (s) vector), one per model.
        prior: A compatible prior (or NullPrior).
        frozen_insertions: If true, insertions will not be trainable.
    """
    def __init__(self, 
                 emission_init = initializers.make_default_emission_init(),
                 insertion_init = initializers.make_default_insertion_init(),
                 prior = None,
                 frozen_insertions = True,
                 **kwargs
                 ):
        super(ProfileHMMEmitter, self).__init__(**kwargs)
        self.emission_init = [emission_init] if not hasattr(emission_init, '__iter__') else emission_init 
        self.insertion_init = [insertion_init] if not hasattr(insertion_init, '__iter__') else insertion_init
        self.prior = priors.AminoAcidPrior(dtype=self.dtype) if prior is None else prior
        self.frozen_insertions = frozen_insertions


    def set_lengths(self, lengths):
        """
        Sets the model lengths.
        Args:
            lengths: A list of model lengths.
        """
        self.lengths = lengths
        self.num_models = len(lengths) 
        # make sure the lengths are valid
        assert len(self.lengths) == len(self.emission_init), \
            f"The number of emission initializers ({len(self.emission_init)}) should match the number of models ({len(self.lengths)})."
        assert len(self.lengths) == len(self.insertion_init), \
            f"The number of insertion initializers ({len(self.insertion_init)}) should match the number of models ({len(self.lengths)})."
        

    def build(self, input_shape):
        if self.built:
            return
        s = input_shape[-1]-1 # substract one for terminal symbol
        self.emission_kernel = [self.add_weight(
                                        shape=(length, s), 
                                        initializer=init, 
                                        name="emission_kernel_"+str(i)) 
                                    for i,(length, init) in enumerate(zip(self.lengths, self.emission_init))]
        self.insertion_kernel = [ self.add_weight(
                                shape=(s,),
                                initializer=init,
                                name="insertion_kernel_"+str(i),
                                trainable=not self.frozen_insertions) 
                                    for i,init in enumerate(self.insertion_init)]
        self.prior.build()
        self.built = True
        

    def recurrent_init(self):
        """ Automatically called before each recurrent run. Should be used for setups that
            are only required once per application of the recurrent layer.
        """
        self.B = self.make_B()
        self.B_transposed = tf.transpose(self.B, [0,2,1])
        

    def make_emission_matrix(self, i):
        """Constructs an emission matrix from kernels with a shared insertion distribution.
        Args:
           i: Model index.
        Returns:
            The emission matrix.
        """
        em, ins = self.emission_kernel[i], self.insertion_kernel[i]
        length = self.lengths[i]
        return self.make_emission_matrix_from_kernels(em, ins, length)
    

    def make_emission_matrix_from_kernels(self, em, ins, length):
        s = em.shape[-1]
        i1 = tf.expand_dims(ins, 0)
        i2 = tf.stack([tf.identity(ins)]*(length+1))
        emissions = tf.concat([i1, em, i2] , axis=0)
        emissions = tf.nn.softmax(emissions)
        emissions = tf.concat([emissions, tf.zeros_like(emissions[:,:1])], axis=-1) 
        end_state_emission = tf.one_hot([s], s+1, dtype=em.dtype) 
        emissions = tf.concat([emissions, end_state_emission], axis=0)
        return emissions
        

    def make_B(self):
        emission_matrices = []
        max_num_states = max(get_num_states(self.lengths))
        for i in range(self.num_models):
            em_mat = self.make_emission_matrix(i)
            padding = max_num_states - em_mat.shape[0]
            em_mat_pad = tf.pad(em_mat, [[0, padding], [0,0]])
            emission_matrices.append(em_mat_pad)
        B = tf.stack(emission_matrices, axis=0)
        return B
        

    def make_B_amino(self):
        """ A variant of make_B used for plotting the HMM. Can be overridden for more complex emissions. Per default this is equivalent to make_B
        """
        return self.make_B()
        

    def call(self, inputs, end_hints=None, training=False):
        """ 
        Args: 
                inputs: A tensor of shape (k, ... , s) 
                end_hints: A tensor of shape (num_models, batch_size, 2, num_states) that contains the correct state for the left and right ends of each chunk.
        Returns:
                A tensor with emission probabilities of shape (k, ... , q) where "..." is identical to inputs.
        """
        input_shape = tf.shape(inputs)
        inputs = tf.reshape(inputs, (tf.shape(inputs)[0], -1, input_shape[-1]))
        B = self.B_transposed[..., :input_shape[-1], :]
        # batch matmul of k emission matrices with the b x s input matrix 
        # with broadcasting of the inputs
        gpu = len([x.physical_device_desc for x in device_lib.list_local_devices() if x.device_type == 'GPU']) > 0
        if version.parse(tf.__version__) < version.parse("2.11.0") or gpu:
            emit = tf.einsum("kbs,ksq->kbq", inputs, B)
        else:
            # something weird happens with batch matmul (or einsam on newer tensorflow versions and CPU) 
            # use this workaround at the cost of some performance
            emit = tf.concat([tf.matmul(inputs[i], B[i]) for i in range(self.num_models)], axis=0)
        emit_shape = tf.concat([tf.shape(B)[:1], input_shape[1:-1], tf.shape(B)[-1:]], 0)
        emit = tf.reshape(emit, emit_shape)
        return emit


    def get_aux_loss(self):
        return tf.constant(0., dtype=self.dtype)
    

    def get_prior_log_density(self):
        return self.prior(self.B, lengths=self.lengths)
    

    def duplicate(self, model_indices=None, share_kernels=False):
        if model_indices is None:
            model_indices = range(len(self.emission_init))
        sub_emission_init = [initializers.ConstantInitializer(self.emission_kernel[i].numpy()) for i in model_indices]
        sub_insertion_init = [initializers.ConstantInitializer(self.insertion_kernel[i].numpy()) for i in model_indices]
        emitter_copy = ProfileHMMEmitter(
                             emission_init = sub_emission_init,
                             insertion_init = sub_insertion_init,
                             prior = self.prior,
                             frozen_insertions = self.frozen_insertions,
                             dtype = self.dtype) 
        if share_kernels:
            emitter_copy.emission_kernel = self.emission_kernel
            emitter_copy.insertion_kernel = self.insertion_kernel
            emitter_copy.built = True
        return emitter_copy
    

    def get_config(self):
        config = super(ProfileHMMEmitter, self).get_config()
        config.update({
        "lengths" : self.lengths.tolist() if isinstance(self.lengths, np.ndarray) else self.lengths,
        "emission_init" : self.emission_init,
        "insertion_init" : self.insertion_init,
        "prior" : self.prior,
        "frozen_insertions" : self.frozen_insertions
        })
        return config
    

    @classmethod
    def from_config(cls, config):
        config["emission_init"] = [deserialize(e) for e in config["emission_init"]]
        config["insertion_init"] = [deserialize(i) for i in config["insertion_init"]]
        config["prior"] = deserialize(config["prior"])
        lengths = config.pop("lengths")
        emitter = cls(**config)
        emitter.set_lengths(lengths)
        return emitter
    
    def __repr__(self):
        return f"ProfileHMMEmitter(\n emission_init={self.emission_init[0]},\n insertion_init={self.insertion_init[0]},\n prior={self.prior},\n frozen_insertions={self.frozen_insertions}, )"
    


class TemperatureMode(Enum):
    TRAINABLE = 1
    LENGTH_NORM = 2
    COLD_TO_WARM = 3
    WARM_TO_COLD = 4
    CONSTANT = 5
    NONE = 6

    @staticmethod
    def from_string(name):
        return {"trainable" : TemperatureMode.TRAINABLE,
                "length_norm" : TemperatureMode.LENGTH_NORM,
                "cold_to_warm" : TemperatureMode.COLD_TO_WARM,
                "warm_to_cold" : TemperatureMode.WARM_TO_COLD,
                "constant" : TemperatureMode.CONSTANT,
                "none" : TemperatureMode.NONE}[name.lower()]


tf.keras.utils.get_custom_objects()["ProfileHMMEmitter"] = ProfileHMMEmitter