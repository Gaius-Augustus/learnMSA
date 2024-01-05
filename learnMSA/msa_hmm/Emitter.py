import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import os
import learnMSA.msa_hmm.Initializers as initializers
import learnMSA.msa_hmm.Priors as priors
from learnMSA.msa_hmm.SequenceDataset import SequenceDataset
# from learnMSA.protein_language_models.BilinearSymmetric import make_scoring_model
# import learnMSA.protein_language_models.Common as Common
from packaging import version
from tensorflow.python.client import device_lib
from enum import Enum
import inspect
import math


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
        self.dim = cell.dim-1
        assert len(self.length) == len(self.emission_init), \
            f"The number of emission initializers ({len(self.emission_init)}) should match the number of models ({len(self.length)})."
        assert len(self.length) == len(self.insertion_init), \
            f"The number of insertion initializers ({len(self.insertion_init)}) should match the number of models ({len(self.length)})."
        self.max_num_states = cell.max_num_states
        self.num_models = cell.num_models
        self.prior.load(self.dtype)
        
    def build(self, input_shape):
        if self.built:
            return
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
        
    def make_emission_matrix(self, i):
        """Constructs an emission matrix from kernels with a shared insertion distribution.
        Args:
           i: Model index.
        Returns:
            The emission matrix.
        """
        em, ins = self.emission_kernel[i], self.insertion_kernel[i]
        length = self.length[i]
        return self.make_emission_matrix_from_kernels(em, ins, length)
    
    def make_emission_matrix_from_kernels(self, em, ins, length):
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
        for i in range(self.num_models):
            em_mat = self.make_emission_matrix(i) 
            padding = self.max_num_states - em_mat.shape[0]
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
    
    def get_prior_log_density(self):
        return self.prior(self.make_B(), self.length)
    
    def duplicate(self, model_indices=None, share_kernels=False):
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
        if share_kernels:
            emitter_copy.emission_kernel = self.emission_kernel
            emitter_copy.insertion_kernel = self.insertion_kernel
            emitter_copy.built = True
        return emitter_copy
    
    def get_config(self):
        config = super(ProfileHMMEmitter, self).get_config()
        config.update({
        "emission_init" : self.emission_init,
        "insertion_init" : self.insertion_init,
        "prior" : self.prior,
        "frozen_insertions" : self.frozen_insertions
        })
        return config
    
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


    
# #have a single emitter that handles both AA inputs and embeddings
# class EmbeddingEmitter(ProfileHMMEmitter):
#     def __init__(self, 
#                  scoring_model_config : Common.ScoringModelConfig,
#                  emission_init=None, 
#                  insertion_init=None,
#                  prior : priors.MvnEmbeddingPrior = None,
#                  temperature_mode : TemperatureMode = TemperatureMode.TRAINABLE,
#                  conditionally_independent = True,
#                  alpha = 1.2,
#                  beta = 1.,
#                  **kwargs):

#         self.scoring_model_config = scoring_model_config
#         self.temperature_mode = temperature_mode
#         self.alpha = alpha
#         self.beta = beta 

#         if prior is None:
#             prior = priors.MvnEmbeddingPrior(scoring_model_config)

#         if emission_init is None:
#             emission_init = initializers.EmbeddingEmissionInitializer(scoring_model_config=scoring_model_config, 
#                                                                         num_prior_components=prior.num_components)
#         if insertion_init is None:
#             insertion_init = initializers.EmbeddingEmissionInitializer(scoring_model_config=scoring_model_config, 
#                                                                         num_prior_components=prior.num_components)
                                                             
#         super(EmbeddingEmitter, self).__init__(emission_init, 
#                                                insertion_init,
#                                                prior, 
#                                                **kwargs)
        
#         # create and load the underlying scoring model
#         self.scoring_model = make_scoring_model(self.scoring_model_config, dropout=0.0, trainable=False)
#         scoring_model_path = Common.get_scoring_model_path(self.scoring_model_config)
#         print("Loading scoring model ", scoring_model_path)
#         self.scoring_model.load_weights(os.path.dirname(__file__)+f"/../protein_language_models/"+scoring_model_path)
#         self.bilinear_symmetric_layer = self.scoring_model.layers[-1]
#         self.bilinear_symmetric_layer.trainable = False 
#         self.conditionally_independent = conditionally_independent
        

#     def build(self, input_shape):
#         num_aa = len(SequenceDataset.alphabet)-1
#         if self.conditionally_independent:
#             s = num_aa + self.scoring_model_config.dim
#         else:
#             s = num_aa + num_aa*self.scoring_model_config.dim
#         shape = (input_shape[0], s + 1)
#         super().build(shape)
#         if self.temperature_mode == TemperatureMode.TRAINABLE:
#             self.temperature = self.add_weight(shape=(len(self.emission_init)), initializer=tf.constant_initializer(0.), name="temperature")
#         else:
#             self.temperature = None
#         self.built = True
#         self.step_counter = tf.Variable(0., trainable=False, dtype=self.dtype, name="step") #step counter for temperature annealing
        
#     def recurrent_init(self):
#         """ Automatically called before each recurrent run. Should be used for setups that
#             are only required once per application of the recurrent layer.
#         """
#         super(EmbeddingEmitter, self).recurrent_init()
        
        
#     def compute_emission_probs(self, states, inputs, aa_inputs, num_models, batch, seq_len):
#         num_states  = tf.shape(states)[-2]
#         #compute emission probabilities
#         if not self.conditionally_independent:
#             #we model a specific embedding per amino acid
#             states = tf.reshape(states[..., :-1], (num_models, -1, self.scoring_model_config.dim))
#         else:
#             states = states[..., :-1]
#         emb_emission_scores = self.bilinear_symmetric_layer(inputs[..., :-1], # (num_models, batch*len, dim)
#                                                            states, # (num_models, num_states * repeats, dim)
#                                                            a_is_reduced=True, 
#                                                            b_is_reduced=True, 
#                                                            training=False, 
#                                                            activate_output=False,
#                                                            use_bias=True) 

#         emb_emission_scores = tf.reshape(emb_emission_scores, (num_models, batch, seq_len, num_states, -1)) #(..., repeats)

#         # we optimized away some states and have to append clones of the first insert to make activations like softmax work correctly
#         rest = tf.repeat(emb_emission_scores[..., :1, :], tf.shape(self.B_transposed)[-1] - max(self.length)-1, axis=-2)
#         # alter padding positions such that the softmax is 0 at these positions
#         rest -= 1e9 * aa_inputs[..., -1:, tf.newaxis]
#         emb_emission_scores = tf.concat([emb_emission_scores, rest], axis=-2)
        
#         act = self.bilinear_symmetric_layer.activation
#         if "axis" in inspect.signature(act).parameters:
#             emb_emission_probs = act(emb_emission_scores, axis=-2) # if activation = softmax, this is over states
#         else:
#             emb_emission_probs = act(emb_emission_scores)

#         if self.conditionally_independent:
#             emb_emission_probs = emb_emission_probs[..., 0] #drop the last dimension, which is 1
#         else:
#             #weighted average based on input amino acid distributions over last dimension
#             reduced_aa_inputs = aa_inputs[..., :len(SequenceDataset.alphabet)-1] # no reduction
#             emb_emission_probs = tf.linalg.matvec(emb_emission_probs, reduced_aa_inputs)

#         #ensure numeric stability during training
#         epsilon = 1e-7
#         emb_emission_probs = tf.clip_by_value(emb_emission_probs, epsilon, 1-epsilon)

#         emb_emission_probs = self.beta_pdf(emb_emission_probs)
#         emb_emission_probs = tf.math.maximum(emb_emission_probs, epsilon)

#         return emb_emission_probs 


#     def beta_pdf(self, probs):
#         tfd = tfp.distributions
#         dist = tfd.Beta(self.alpha, self.beta)
#         return dist.prob(probs)
        
        
#     def call(self, inputs, training=False):
#         """ 
#         Args: 
#             inputs: Shape (num_models, batch, seq_len, d) 
#         Returns:
#             Shape (num_models, batch, seq_len, num_states)
#         """
#         num_models, batch, seq_len, d  = tf.unstack(tf.shape(inputs))
#         num_states = tf.shape(self.B_transposed)[-1]
#         terminal_padding = inputs[..., -1:]
#         orig_inputs = inputs
#         inputs = tf.reshape(inputs, (num_models, -1, d)) 
        
#         #compute amino acid emissions
#         aa_inputs = inputs[..., :len(SequenceDataset.alphabet)]
#         aa_B_transposed = self.B_transposed[:,:len(SequenceDataset.alphabet),:]
#         aa_emission_probs = tf.matmul(aa_inputs, aa_B_transposed)
#         aa_emission_probs = tf.reshape(aa_emission_probs, (num_models, batch, seq_len, num_states))
        
#         #compute embedding emission probs 
#         emb_inputs = inputs[..., len(SequenceDataset.alphabet):]
#         emb_B = self.B[..., len(SequenceDataset.alphabet):]

#         # an optimization to reduce the number of pairs:
#         # since all insertions are mirrored, just compute the pairing for one of them
#         # the state order is START, M1, M2,... ML, I1,...
#         # cut behind the maximum model length + 1 and use the fact that models are usually of similar length
#         # to see that you actually reduce memory by roughly factor 2
#         emb_B = emb_B[:, :max(self.length)+1]

#         #pre sigmoid proto
#         # if self.conditionally_independent:
#         #     emb_B = emb_B[..., :-1]
#         # else:
#         #     emb_B = tf.reshape(emb_B[..., :-1], (num_models, 1, num_states, len(SequenceDataset.alphabet)-1, self.scoring_model_config.dim))
#         #     emb_B = tf.matmul(aa_inputs[:, :, tf.newaxis, :-1], emb_B) #(num_models, batch*len_seq, num_states, dim)

#         #embedding scores that sum to 1 over all valid input sequence positions
#         emb_emission_probs = self.compute_emission_probs(emb_B, emb_inputs, orig_inputs, num_models, batch, seq_len)

#         #optional annealing depending on the configured mode
#         if self.temperature_mode != TemperatureMode.NONE:
#             if self.temperature_mode == TemperatureMode.TRAINABLE:
#                 #use query sequence lengths L_i here? i.e. temperatur.shape = (1,batch,1,1) with temperature[0,i,0,0] = 1 / L_i
#                 temperature = tf.math.sigmoid(self.temperature[:, tf.newaxis, tf.newaxis, tf.newaxis])
#             elif self.temperature_mode == TemperatureMode.LENGTH_NORM:
#                 seq_lens = tf.reduce_sum(1-terminal_padding, axis=-2, keepdims=True)
#                 temperature = 1 / seq_lens #can we ever have sequences of length 0 here?
#             elif self.temperature_mode == TemperatureMode.COLD_TO_WARM or self.temperature_mode == TemperatureMode.WARM_TO_COLD:
#                 # colder, i.e. t closer to 1 means embeddings have the same level of control as amino acids
#                 # t = 0 means the only amino acid emissions are used
#                 examples_seen = self.step_counter * tf.cast(batch, tf.float32)
#                 temperature = tf.math.pow(0.99997, examples_seen)
#                 if self.temperature_mode == TemperatureMode.WARM_TO_COLD:
#                     temperature = 1 - temperature
#                 self.add_metric(temperature, name="temperature")
#             elif self.temperature_mode ==  TemperatureMode.CONSTANT:
#                 temperature = 0.1
#             #emb_emission_probs = tf.math.pow(emb_emission_probs, temperature)

#         #padding/terminal states
#         emb_emission_probs *= 1-terminal_padding # set positions with softmax(0, ..., 0) to 0
#         #set emission probs to 1 where sequence input is terminal and state is also terminal
#         emb_emission_probs += terminal_padding * self.B[:, tf.newaxis, tf.newaxis, :, -1]

#         emission_probs = aa_emission_probs * emb_emission_probs 
#         return emission_probs
    
    
#     def make_emission_matrix(self, i):
#         aa_em = self.emission_kernel[i][:, :len(SequenceDataset.alphabet)-1]
#         emb_em = self.emission_kernel[i][:, len(SequenceDataset.alphabet)-1:]
#         #drop the terminal state probs since we have identical ones for the embeddings
#         aa_emissions = self.make_emission_matrix_from_kernels(aa_em, self.insertion_kernel[i][:len(SequenceDataset.alphabet)-1], self.length[i])
#         """ Construct the emission matrix the same way as usual but leave away the softmax.
#         """
#         s = emb_em.shape[-1]
#         emb_ins = self.insertion_kernel[i][len(SequenceDataset.alphabet)-1:]
#         emb_em = tf.concat([tf.expand_dims(emb_ins, 0), 
#                             emb_em, 
#                             tf.stack([emb_ins]*(self.length[i]+1))] , axis=0)
#         emb_em = tf.concat([emb_em, tf.zeros_like(emb_em[:,:1])], axis=-1) 
#         end_state_emission = tf.one_hot([s], s+1, dtype=emb_em.dtype) 
#         emb_emissions = tf.concat([emb_em, end_state_emission], axis=0)
#         return tf.concat([aa_emissions, emb_emissions], -1)
    
    
#     def duplicate(self, model_indices=None, share_kernels=False):
#         if model_indices is None:
#             model_indices = range(len(self.emission_init))
#         sub_emission_init = [tf.constant_initializer(self.emission_kernel[i].numpy()) for i in model_indices]
#         sub_insertion_init = [tf.constant_initializer(self.insertion_kernel[i].numpy()) for i in model_indices]
#         #todo: this does not dublicate embedding insertion kernels which is probably ok
#         emitter_copy = EmbeddingEmitter(
#                                 scoring_model_config = self.scoring_model_config,  
#                                 emission_init = sub_emission_init,
#                                 insertion_init = sub_insertion_init,
#                                 prior = self.prior,
#                                 temperature_mode = self.temperature_mode,
#                                 conditionally_independent = self.conditionally_independent,
#                                 frozen_insertions = self.frozen_insertions,
#                                 dtype = self.dtype) 
#         if share_kernels:
#             emitter_copy.emission_kernel = self.emission_kernel
#             emitter_copy.insertion_kernel = self.insertion_kernel
#             emitter_copy.temperature = self.temperature
#             emitter_copy.built = True
#         return emitter_copy

        
#     def make_B_amino(self):
#         """ A variant of make_B used for plotting the HMM. Can be overridden for more complex emissions. Per default this is equivalent to make_B
#         """
#         return self.make_B()[:,:,:len(SequenceDataset.alphabet)-1]

    
#     def get_config(self):
#         config = super(EmbeddingEmitter, self).get_config()
#         config.update({"scoring_model_config" : self.scoring_model_config, 
#                         "temperature_mode" : self.temperature_mode,
#                         "conditionally_independent" : self.conditionally_independent})
#         return config


#     def __repr__(self):
#         parent = super(EmbeddingEmitter, self).__repr__()
#         return f"EmbeddingEmitter(scoring_model_config = {self.scoring_model_config}, temperature_mode = {self.temperature_mode}, {parent})"
    
    



# #have a single emitter that handles both AA inputs and embeddings
# class MVNEmitter2(ProfileHMMEmitter):
#     def __init__(self, 
#                  scoring_model_config : Common.ScoringModelConfig,
#                  emission_init=None, 
#                  insertion_init=None,
#                  prior : priors.MvnEmbeddingPrior = None,
#                  diag_init_var = 1,
#                  full_covariance = False,
#                  temperature = 10.,
#                  step_init=0,
#                  **kwargs):

#         self.scoring_model_config = scoring_model_config

#         if prior is None:
#             prior = priors.MvnEmbeddingPrior(scoring_model_config)

#         if emission_init is None:
#             emission_init = initializers.AminoAcidPlusMvnEmissionInitializer(scoring_model_config=scoring_model_config, 
#                                                                             num_prior_components=prior.num_components,
#                                                                             full_covariance=full_covariance)
#         if insertion_init is None:
#             insertion_init = initializers.AminoAcidPlusMvnEmissionInitializer(scoring_model_config=scoring_model_config, 
#                                                                             num_prior_components=prior.num_components,
#                                                                             full_covariance=full_covariance)
                                                             
#         super(MVNEmitter2, self).__init__(emission_init, insertion_init, prior, **kwargs)
        
#         # create and load the underlying scoring model
#         self.scoring_model = make_scoring_model(self.scoring_model_config, dropout=0.0, trainable=False)
#         scoring_model_path = Common.get_scoring_model_path(self.scoring_model_config)
#         print("Loading scoring model ", scoring_model_path)
#         self.scoring_model.load_weights(os.path.dirname(__file__)+f"/../protein_language_models/"+scoring_model_path)
#         self.bilinear_symmetric_layer = self.scoring_model.layers[-1]
#         self.bilinear_symmetric_layer.trainable = False 
#         self.diag_init_var = diag_init_var
#         diag_init_std = np.sqrt(diag_init_var).astype(np.float32)
#         self.scale_diag_init = tfp.math.softplus_inverse(diag_init_std)
#         self.full_covariance = full_covariance
#         self.constant = self.scoring_model_config.dim * tf.math.log(2*math.pi)
#         self.temperature = temperature
#         self.step_init = step_init
        

#     def build(self, input_shape):
#         if self.full_covariance:
#             s = len(SequenceDataset.alphabet) + self.scoring_model_config.dim + self.scoring_model_config.dim * (self.scoring_model_config.dim+1) // 2
#         else:
#             s = len(SequenceDataset.alphabet) + 2*self.scoring_model_config.dim 
#         shape = (input_shape[0], s)
#         super().build(shape)
#         self.scale_tril = tfp.bijectors.FillScaleTriL(diag_bijector=DiagBijector(self.scale_diag_init))
#         self.step_counter = tf.Variable(self.step_init, trainable=False, dtype=self.dtype, name="step") #step counter for temperature annealing
#         self.built = True
    
    
#     def make_emission_matrix(self, i):
#         aa_em = self.emission_kernel[i][:, :len(SequenceDataset.alphabet)-1]
#         emb_em = self.emission_kernel[i][:, len(SequenceDataset.alphabet)-1:]
#         #drop the terminal state probs since we have identical ones for the embeddings
#         aa_emissions = self.make_emission_matrix_from_kernels(aa_em, self.insertion_kernel[i][:len(SequenceDataset.alphabet)-1], self.length[i])
#         """ Construct the emission matrix the same way as usual but leave away the softmax.
#         """
#         emb_ins = self.insertion_kernel[i][len(SequenceDataset.alphabet)-1:]
#         emb_em = tf.concat([tf.expand_dims(emb_ins, 0), 
#                             emb_em, 
#                             tf.stack([emb_ins]*(self.length[i]+1))] , axis=0)
#         end_state_emission = tf.zeros_like(emb_em[:1])
#         emb_emissions = tf.concat([emb_em, end_state_emission], axis=0)
#         return tf.concat([aa_emissions, emb_emissions], -1)


#     def mvn_log_pdf(self, inputs):
#         """
#         Args:
#             inputs: Shape (num_models, batch*seq_len, dim)
#         """
#         mu = self.B[:, :max(self.length)+1, len(SequenceDataset.alphabet):len(SequenceDataset.alphabet)+self.scoring_model_config.dim]
#         scale_kernel = self.B[:, :max(self.length)+1, len(SequenceDataset.alphabet)+self.scoring_model_config.dim:]
#         if self.full_covariance:
#             scale = self.scale_tril(scale_kernel)
#             pinv = tf.linalg.pinv(scale)
#         else:
#             scale_diag = tf.math.softplus(scale_kernel + self.scale_diag_init)[..., tf.newaxis]
#             scale = tf.eye(self.scoring_model_config.dim, batch_shape=tf.shape(scale_kernel)[:-1]) * scale_diag
#             pinv = tf.eye(self.scoring_model_config.dim, batch_shape=tf.shape(scale_kernel)[:-1]) * (1 / (scale_diag+1e-20))
#         log_det = 2*tf.reduce_sum(tf.math.log(tf.linalg.diag_part(scale)), -1, keepdims=True)
#         diff = inputs[:,tf.newaxis] - mu[:,:,tf.newaxis] # (models, states, b, d)
#         y = tf.matmul(diff, pinv) # (models, states, b, d)
#         MD_sq_components = tf.reduce_sum(y * y, -1) 
#         log_pdf = -0.5 * (self.constant + log_det + MD_sq_components)
#         log_pdf = tf.transpose(log_pdf, [0,2,1])
#         return log_pdf
        
        
#     def compute_emission_probs(self, inputs, num_models, batch, seq_len):
#         num_states = max(self.length)+1
#         #compute emission probabilities
#         log_pdf = self.mvn_log_pdf(inputs[..., :-1])
#         log_pdf = tf.reshape(log_pdf, (num_models, batch, seq_len, num_states))

#         # we optimized away some states and have to append clones of the first insert to make activations like softmax work correctly
#         rest = tf.repeat(log_pdf[..., :1], tf.shape(self.B_transposed)[-1] - max(self.length)-1, axis=-1)
#         log_pdf = tf.concat([log_pdf, rest], axis=-1)

#         return tf.exp(log_pdf/self.temperature) + 1e-20
        
        
#     def call(self, inputs, training=False):
#         """ 
#         Args: 
#             inputs: Shape (num_models, batch, seq_len, d) 
#         Returns:
#             Shape (num_models, batch, seq_len, num_states)
#         """
#         num_models, batch, seq_len, d  = tf.unstack(tf.shape(inputs))
#         num_states = tf.shape(self.B_transposed)[-1]
#         terminal_padding = inputs[..., -1:]
#         orig_inputs = inputs
#         inputs = tf.reshape(inputs, (num_models, -1, d)) 
        
#         #compute amino acid emissions
#         aa_inputs = inputs[..., :len(SequenceDataset.alphabet)]
#         aa_B_transposed = self.B_transposed[:,:len(SequenceDataset.alphabet),:]
#         aa_emission_probs = tf.matmul(aa_inputs, aa_B_transposed)
#         aa_emission_probs = tf.reshape(aa_emission_probs, (num_models, batch, seq_len, num_states))
        
#         #compute embedding emission probs 
#         emb_inputs = inputs[..., len(SequenceDataset.alphabet):]

#         #embedding scores that sum to 1 over all valid input sequence positions
#         emb_emission_probs = self.compute_emission_probs(emb_inputs, num_models, batch, seq_len)

#         #padding/terminal states
#         emb_emission_probs *= 1-terminal_padding # set positions with softmax(0, ..., 0) to 0
#         #set emission probs to 1 where sequence input is terminal and state is also terminal
#         emb_emission_probs += terminal_padding * self.B[:, tf.newaxis, tf.newaxis, :, len(SequenceDataset.alphabet)-1]

#         emission_probs = aa_emission_probs * emb_emission_probs 
#         return emission_probs
    
    
#     def duplicate(self, model_indices=None, share_kernels=False):
#         if model_indices is None:
#             model_indices = range(len(self.emission_init))
#         sub_emission_init = [tf.constant_initializer(self.emission_kernel[i].numpy()) for i in model_indices]
#         sub_insertion_init = [tf.constant_initializer(self.insertion_kernel[i].numpy()) for i in model_indices]
#         #todo: this does not dublicate embedding insertion kernels which is probably ok
#         emitter_copy = MVNEmitter(
#                                 scoring_model_config = self.scoring_model_config,  
#                                 emission_init = sub_emission_init,
#                                 insertion_init = sub_insertion_init,
#                                 prior = self.prior,
#                                 diag_init_var = self.diag_init_var,
#                                 full_covariance = self.full_covariance,
#                                 temperature = self.temperature,
#                                 step_init = self.step_counter.numpy(),
#                                 frozen_insertions = self.frozen_insertions,
#                                 dtype = self.dtype) 
#         if share_kernels:
#             emitter_copy.emission_kernel = self.emission_kernel
#             emitter_copy.insertion_kernel = self.insertion_kernel
#             emitter_copy.built = True
#         return emitter_copy

    
#     def get_config(self):
#         config = super(MVNEmitter2, self).get_config()
#         config.update({"scoring_model_config" : self.scoring_model_config,
#                         "diag_init_var" : self.diag_init_var,
#                         "full_covariance" : self.full_covariance,
#                         "temperature" : self.temperature,
#                         "step_init" : self.step_init})
#         return config


#     def __repr__(self):
#         parent = super(MVNEmitter2, self).__repr__()
#         return f"MVNEmitter2(scoring_model_config = {self.scoring_model_config}, {parent})"


# class DiagBijector(tfp.bijectors.Softplus):
#     def __init__(self, diagonal_value):
#         super(DiagBijector, self).__init__()
#         self.diagonal_value = diagonal_value

#     def forward(self, x, name='forward', **kwargs):
#         return super(DiagBijector, self).forward(x + self.diagonal_value, name, **kwargs)



tf.keras.utils.get_custom_objects()["ProfileHMMEmitter"] = ProfileHMMEmitter
# tf.keras.utils.get_custom_objects()["EmbeddingEmitter"] = EmbeddingEmitter
# tf.keras.utils.get_custom_objects()["MVNEmitter2"] = MVNEmitter2