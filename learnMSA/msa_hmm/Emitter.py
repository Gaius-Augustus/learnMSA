import tensorflow as tf
import numpy as np
import os
import learnMSA.msa_hmm.Initializers as initializers
import learnMSA.msa_hmm.Priors as priors
from learnMSA.msa_hmm.SequenceDataset import SequenceDataset
from learnMSA.protein_language_models.BilinearSymmetric import make_scoring_model
import learnMSA.protein_language_models.Common as Common
from packaging import version
from tensorflow.python.client import device_lib
    

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
        
    def call(self, inputs):
        """ 
        Args: 
                inputs: A tensor of shape (k, ... , s) 
        Returns:
                A tensor with emission probabilities of shape (k, ... , q) where "..." is identical to inputs.
        """
        input_shape = tf.shape(inputs)
        inputs = tf.reshape(inputs, (tf.shape(inputs)[0], -1, input_shape[-1]))
        # batch matmul of k emission matrices with the b x s input matrix 
        # with broadcasting of the inputs
        gpu = len([x.physical_device_desc for x in device_lib.list_local_devices() if x.device_type == 'GPU']) > 0
        if version.parse(tf.__version__) < version.parse("2.11.0") or gpu:
            emit = tf.einsum("kbs,ksq->kbq", inputs, self.B_transposed)
        else:
            # something weird happens with batch matmul (or einsam on newer tensorflow versions and CPU) 
            # use this workaround at the cost of some performance
            emit = tf.concat([tf.matmul(inputs[i], self.B_transposed[i]) for i in range(self.num_models)], axis=0)
        emit_shape = tf.concat([tf.shape(self.B_transposed)[:1], input_shape[1:-1], tf.shape(self.B_transposed)[-1:]], 0)
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
    
    
    
#have a single emitter that handles both AA inputs and embeddings
class EmbeddingEmitter(ProfileHMMEmitter):
    def __init__(self, 
                 scoring_model_config : Common.ScoringModelConfig,
                 emission_init=None, 
                 insertion_init=None,
                 prior : priors.MvnEmbeddingPrior = None,
                 **kwargs):

        self.scoring_model_config = scoring_model_config

        if prior is None:
            prior = priors.MvnEmbeddingPrior(scoring_model_config)

        if emission_init is None:
            emission_init = initializers.EmbeddingEmissionInitializer(scoring_model_config=scoring_model_config, 
                                                                        num_prior_components=prior.num_components)
        if insertion_init is None:
            insertion_init = initializers.EmbeddingEmissionInitializer(scoring_model_config=scoring_model_config, 
                                                                        num_prior_components=prior.num_components)
                                                             
        super(EmbeddingEmitter, self).__init__(emission_init, 
                                               insertion_init,
                                               prior, 
                                               **kwargs)
        
        # create and load the underlying scoring model
        self.scoring_model = make_scoring_model(self.scoring_model_config, dropout=0.0, trainable=False)
        scoring_model_path = Common.get_scoring_model_path(self.scoring_model_config)
        print("Loading scoring model ", scoring_model_path)
        self.scoring_model.load_weights(os.path.dirname(__file__)+f"/../protein_language_models/"+scoring_model_path)
        self.bilinear_symmetric_layer = self.scoring_model.layers[-1]
        self.bilinear_symmetric_layer.trainable = False 
        

    def build(self, input_shape):
        s = len(SequenceDataset.alphabet)-1 + self.scoring_model_config.dim
        shape = (input_shape[0], s + 1)
        super().build(shape)
        self.temperature = self.add_weight(shape=(len(self.emission_init)), initializer=tf.constant_initializer(0.), name="temperature")
        self.built = True
        
        
    def compute_emission_probs(self, states, inputs, emit_shape, terminal_padding, temperature=None):
        #compute emission probabilities
        emb_emission_probs = self.bilinear_symmetric_layer(states[..., :-1], inputs[..., :-1], 
                                                           a_is_reduced=True, 
                                                           b_is_reduced=True, 
                                                           training=False, 
                                                           activate_output=False,
                                                           use_bias=True) 
        emb_emission_probs = tf.transpose(emb_emission_probs, [0, 2, 1])
        emb_emission_probs = tf.reshape(emb_emission_probs, emit_shape) # (k, b*l, q) -> (k, b, l, q)
        emb_emission_probs = self.bilinear_symmetric_layer.activation(emb_emission_probs)+ 1e-10
        #optional annealing
        if temperature is not None:
            #use query sequence lengths L_i here? i.e. temperatur.shape = (1,batch,1,1) with temperature[0,i,0,0] = 1 / L_i
            emb_emission_probs = tf.math.pow(emb_emission_probs, tf.math.sigmoid(temperature[:, tf.newaxis, tf.newaxis, tf.newaxis]))
        #padding/terminal states
        emb_emission_probs *= 1-terminal_padding # set positions with softmax(0, ..., 0) to 0
        #set emission probs to 1 where sequence input is terminal and state is also terminal
        emb_emission_probs += terminal_padding * states[:, tf.newaxis, tf.newaxis, :, -1]
        return emb_emission_probs 
            
        
    def call(self, inputs):
        """ 
        Args: 
            inputs: Shape (num_models, batch, seq_len, d) 
        Returns:
            Shape (num_models, batch, seq_len, num_states)
        """
        num_models, batch, seq_len, d  = tf.unstack(tf.shape(inputs))
        num_states = tf.shape(self.B_transposed)[-1]
        emit_shape = (num_models, batch, seq_len, num_states)
        terminal_padding = inputs[..., -1:]
        inputs = tf.reshape(inputs, (num_models, -1, d)) 
        
        #compute amino acid emissions
        aa_inputs = inputs[..., :len(SequenceDataset.alphabet)]
        aa_B_transposed = self.B_transposed[:,:len(SequenceDataset.alphabet),:]
        aa_emission_probs = tf.matmul(aa_inputs, aa_B_transposed)
        aa_emission_probs = tf.reshape(aa_emission_probs, emit_shape)
        
        #compute embedding emission probs
        emb_inputs = inputs[..., len(SequenceDataset.alphabet):]
        emb_B = self.B[..., len(SequenceDataset.alphabet):]
        #embedding scores that sum to 1 over all valid input sequence positions
        emb_emission_probs = self.compute_emission_probs(emb_B, emb_inputs, emit_shape, terminal_padding, self.temperature)
        emission_probs = aa_emission_probs * emb_emission_probs
        return emission_probs
    
    
    def make_emission_matrix(self, i):
        aa_em = self.emission_kernel[i][:, :len(SequenceDataset.alphabet)-1]
        emb_em = self.emission_kernel[i][:, len(SequenceDataset.alphabet)-1:]
        #drop the terminal state probs since we have identical ones for the embeddings
        aa_emissions = self.make_emission_matrix_from_kernels(aa_em, self.insertion_kernel[i][:len(SequenceDataset.alphabet)-1], self.length[i])
        """ Construct the emission matrix the same way as usual but leave away the softmax.
        """
        s = emb_em.shape[-1]
        emb_ins = self.insertion_kernel[i][len(SequenceDataset.alphabet)-1:]
        emb_em = tf.concat([tf.expand_dims(emb_ins, 0), 
                            emb_em, 
                            tf.stack([emb_ins]*(self.length[i]+1))] , axis=0)
        emb_em = tf.concat([emb_em, tf.zeros_like(emb_em[:,:1])], axis=-1) 
        end_state_emission = tf.one_hot([s], s+1, dtype=emb_em.dtype) 
        emb_emissions = tf.concat([emb_em, end_state_emission], axis=0)
        return tf.concat([aa_emissions, emb_emissions], -1)
    
    
    def duplicate(self, model_indices=None, share_kernels=False):
        if model_indices is None:
            model_indices = range(len(self.emission_init))
        sub_emission_init = [tf.constant_initializer(self.emission_kernel[i].numpy()) for i in model_indices]
        sub_insertion_init = [tf.constant_initializer(self.insertion_kernel[i].numpy()) for i in model_indices]
        #todo: this does not dublicate embedding insertion kernels which is probably ok
        emitter_copy = EmbeddingEmitter(
                                scoring_model_config = self.scoring_model_config,  
                                emission_init = sub_emission_init,
                                insertion_init = sub_insertion_init,
                                prior = self.prior,
                                frozen_insertions = self.frozen_insertions,
                                dtype = self.dtype) 
        if share_kernels:
            emitter_copy.emission_kernel = self.emission_kernel
            emitter_copy.insertion_kernel = self.insertion_kernel
            emitter_copy.temperature = self.temperature
            emitter_copy.built = True
        return emitter_copy

        
    def make_B_amino(self):
        """ A variant of make_B used for plotting the HMM. Can be overridden for more complex emissions. Per default this is equivalent to make_B
        """
        return self.make_B()[:,:,:len(SequenceDataset.alphabet)-1]

    
    def get_config(self):
        config = super(EmbeddingEmitter, self).get_config()
        config.update({"scoring_model_config" : self.scoring_model_config})
        return config


    def __repr__(self):
        parent = super(EmbeddingEmitter, self).__repr__()
        return f"EmbeddingEmitter(scoring_model_config = {self.scoring_model_config}, {parent})"
    
    


tf.keras.utils.get_custom_objects()["ProfileHMMEmitter"] = ProfileHMMEmitter
tf.keras.utils.get_custom_objects()["EmbeddingEmitter"] = EmbeddingEmitter