import tensorflow as tf
import numpy as np
import os
import learnMSA.msa_hmm.Initializers as initializers
import learnMSA.msa_hmm.Priors as priors
from learnMSA.msa_hmm.SequenceDataset import SequenceDataset
from learnMSA.protein_language_models.BilinearSymmetric import make_scoring_model
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
    
    
    
#have a single emitter that handles both AA inputs and embeddings
#need a proper implementation of a RNN with nested inputs later
class EmbeddingEmitter(ProfileHMMEmitter):
    def __init__(self, 
                 lm_name,
                 reduced_embedding_dim,
                 L2_match,
                 L2_insert,
                 emission_init=initializers.EmbeddingEmissionInitializer(), 
                 insertion_init=initializers.EmbeddingEmissionInitializer(),
                 use_shared_embedding_insertions=True,
                 frozen_insertions=True,
                 use_finetuned_lm=True):
        super(EmbeddingEmitter, self).__init__(emission_init, 
                                               insertion_init,
                                               priors.L2EmbeddingRegularizer(L2_match, L2_insert, use_shared_embedding_insertions), 
                                               frozen_insertions=frozen_insertions)
        self.lm_name = lm_name
        self.reduced_embedding_dim = reduced_embedding_dim
        self.L2_match = L2_match
        self.L2_insert = L2_insert
        self.use_shared_embedding_insertions = use_shared_embedding_insertions
        self.use_finetuned_lm = use_finetuned_lm
        # only import contextual when lm features are required
        import learnMSA.protein_language_models as plm
        self.scoring_model = make_scoring_model(plm.common.dims[lm_name], reduced_embedding_dim, dropout=0.0)
        if use_finetuned_lm:
            self.scoring_model.load_weights(os.path.dirname(__file__)+f"/../protein_language_models/scoring_models/{lm_name}_{reduced_embedding_dim}/checkpoints")
        else:
            self.scoring_model.load_weights(os.path.dirname(__file__)+f"/../protein_language_models/scoring_models_frozen/{lm_name}_{reduced_embedding_dim}/checkpoints")
        self.scoring_model.layers[-1].trainable = False #don't forget to freeze the scoring model!
        
    def build(self, input_shape):
        s = len(SequenceDataset.alphabet)-1 + self.reduced_embedding_dim
        if self.use_shared_embedding_insertions:
            #the default emitter would construct an emission matrix matching the last input dimension
            #in this case the last input dimension is the full embedding depth whereas the emission matrix
            #should be constructed for the reduced dim as defined in the bilinear symmetric layer
            shape = (input_shape[0], s + 1)
            super().build(shape)
        else:
            #todo: this is a hacky solutions
            #do it clean by rewriting ProfileHMMEmitter
            self.emission_kernel = [self.add_weight(
                                            shape=[length, s], 
                                            initializer=init, 
                                            name="emission_kernel_"+str(i))
                                        for i,(length, init) in enumerate(zip(self.length, self.emission_init))]
            self.insertion_kernel = [ self.add_weight(
                                    shape=[len(SequenceDataset.alphabet)-1],
                                    initializer=init,
                                    name="insertion_kernel_"+str(i),
                                    trainable=not self.frozen_insertions) 
                                        for i,init in enumerate(self.insertion_init)]
            self.embedding_insertion_kernel = [self.add_weight(
                                            shape=[length+2, self.reduced_embedding_dim], 
                                            initializer="zeros", 
                                            name="embedding_emission_kernel_"+str(i),
                                            trainable=not self.frozen_insertions)
                                        for i,length in enumerate(self.length)]
        self.temperature = self.add_weight(shape=(len(self.emission_init)), initializer=tf.constant_initializer(0), name="temperature")
        self.built = True
        
        
    def compute_emission_probs(self, states, inputs, emit_shape, terminal_padding, temperature=None):
        #compute emission probabilities
        emb_emission_probs = self.scoring_model.layers[-1](states[..., :-1], inputs[..., :-1], 
                                                           a_is_reduced=True, b_is_reduced=True, 
                                                           training=False, 
                                                           activate_output=False,
                                                           use_bias=True) 
        emb_emission_probs = tf.transpose(emb_emission_probs, [0, 2, 1])
        emb_emission_probs = tf.reshape(emb_emission_probs, emit_shape) # (k, b*l, q) -> (k, b, l, q)
        emb_emission_probs = tf.nn.softmax(emb_emission_probs)
        #optional annealing
        if temperature is not None:
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
        if self.use_shared_embedding_insertions:
            emb_ins = self.insertion_kernel[i][len(SequenceDataset.alphabet)-1:]
            emb_em = tf.concat([tf.expand_dims(emb_ins, 0), 
                               emb_em, 
                               tf.stack([emb_ins]*(self.length[i]+1))] , axis=0)
        else:
            emb_em = tf.concat([self.embedding_insertion_kernel[i][:1], 
                               emb_em, 
                               self.embedding_insertion_kernel[i][1:]] , axis=0)
        emb_em = tf.concat([emb_em, tf.zeros_like(emb_em[:,:1])], axis=-1) 
        end_state_emission = tf.one_hot([s], s+1, dtype=emb_em.dtype) 
        emb_emissions = tf.concat([emb_em, end_state_emission], axis=0)
        return tf.concat([aa_emissions, emb_emissions], -1)
    
    
    def duplicate(self, model_indices=None):
        if model_indices is None:
            model_indices = range(len(self.emission_init))
        sub_emission_init = [tf.constant_initializer(self.emission_kernel[i].numpy()) for i in model_indices]
        sub_insertion_init = [tf.constant_initializer(self.insertion_kernel[i].numpy()) for i in model_indices]
        #todo: this does not dublicate embedding insertion kernels which is probably ok
        emitter_copy = EmbeddingEmitter(
                             lm_name = self.lm_name,
                             reduced_embedding_dim = self.reduced_embedding_dim,
                             L2_match = self.L2_match,
                             L2_insert = self.L2_insert,
                             emission_init = sub_emission_init,
                             insertion_init = sub_insertion_init,
                             use_shared_embedding_insertions=self.use_shared_embedding_insertions,
                             frozen_insertions=self.frozen_insertions,
                             use_finetuned_lm=self.use_finetuned_lm) 
        return emitter_copy
        
        
    def make_B_amino(self):
        """ A variant of make_B used for plotting the HMM. Can be overridden for more complex emissions. Per default this is equivalent to make_B
        """
        return self.make_B()[:,:,:len(SequenceDataset.alphabet)-1]