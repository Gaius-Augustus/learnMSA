import os
import tensorflow as tf
import numpy as np
import learnMSA.msa_hmm.Emitter as emit
import learnMSA.msa_hmm.Transitioner as trans
from learnMSA.msa_hmm.Utility import get_num_states, get_num_states_implicit, deserialize



class HmmCell(tf.keras.layers.Layer):
    """ A general cell that computes one recursion step of the forward algorithm in its call method and can also compute the backward algorithm.
        It is meant to be used with the generic RNN-layer to compute the likelihood of a batch of sequences. It also wraps a prior and provides 
        functionality (through the injected emitter and transitioner) to construct the emission- and transition-matricies also used elsewhere e.g. during Viterbi.
        Based on https://github.com/mslehre/classify-seqs/blob/main/HMMCell.py.
    Args:
        num_states: A list of the number of states per model.
        dim: The number of dimensions of the input sequence.
        emitter: An object or a list of objects following the emitter interface (see MultinomialAminoAcidEmitter).
        transitioner: An object following the transitioner interface (see ProfileHMMTransitioner).
    """
    def __init__(self,
                 num_states,
                 dim,
                 emitter,
                 transitioner,
                 use_step_counter=False,
                 use_fake_step_counter=False, #only for backwards compatibility in Tiberius, never else set this to True
                 **kwargs
                ):
        super(HmmCell, self).__init__(**kwargs)
        self.num_states = num_states
        self.num_models = len(self.num_states)
        self.max_num_states = max(self.num_states)
        self.dim = dim
        self.emitter = [emitter] if not hasattr(emitter, '__iter__') else emitter 
        self.transitioner = transitioner
        self.state_size = (tf.TensorShape([self.max_num_states]), tf.TensorShape([1]))
        self.epsilon = tf.constant(1e-16, self.dtype)
        self.reverse = False
        self.use_step_counter = use_step_counter
        self.use_fake_step_counter = use_fake_step_counter
            
            
    def build(self, input_shape):
        if self.built:
            return
        for em in self.emitter:
            em.build((None, input_shape[-2], self.dim))
        self.transitioner.build((None, input_shape[-2], self.dim))
        if not self.reverse and self.use_step_counter or self.use_fake_step_counter:
            self.step_counter = self.add_weight(shape=(), initializer=tf.constant_initializer(-1), 
                                                trainable=False, name="step_counter", dtype=tf.int32)
        self.built = True
        self.recurrent_init()

        
    def recurrent_init(self):
        self.transitioner.recurrent_init()
        for em in self.emitter:
            em.recurrent_init()
        self.log_A_dense = self.transitioner.make_log_A()
        self.log_A_dense_t = tf.transpose(self.log_A_dense, [0,2,1])
        self.init_dist = self.make_initial_distribution()
        if not self.reverse and self.use_step_counter:
            self.step_counter.assign(-1)
    
    
    def make_initial_distribution(self):
        """Constructs the initial state distribution which depends on the transition probabilities.
            See ProfileHMMTransitioner.
        Returns:
            A probability distribution of shape: (1, num_model, q)
        """
        return self.transitioner.make_initial_distribution()
        
    
    def emission_probs(self, inputs, end_hints=None, training=False):
        """ Computes the probabilities of emission per state for the given observation. Multiple emitters
            are multiplied.
        Args:
            inputs: A batch of sequence positions.
            end_hints: A tensor of shape (..., 2, num_states) that contains the correct state for the left and right ends of each chunk.
        """
        em_probs = self.emitter[0](inputs, end_hints=end_hints, training=training)
        for em in self.emitter[1:]:
            em_probs *= em(inputs, end_hints=end_hints, training=training)
        return em_probs

    
    def call(self, emission_probs, states, training=None, init=False):
        """ Computes one recurrent step of the Forward DP.
        """
        old_scaled_forward, old_loglik = states
        old_scaled_forward = tf.reshape(old_scaled_forward, (self.num_models, -1, self.max_num_states))
        if init:
            R = old_scaled_forward
        else:
            R = self.transitioner(old_scaled_forward)
        E = tf.reshape(emission_probs, (self.num_models, -1, self.max_num_states))
        #if parallel, allow broadcasting of inputs to forward probs
        q = tf.shape(R)[1] // tf.shape(E)[1] #q == 1 if not parallel else q = num_states
        R = tf.reshape(R, (self.num_models, -1, q, self.max_num_states))
        E = tf.reshape(E, (self.num_models, -1, 1, self.max_num_states))
        old_loglik = tf.reshape(old_loglik, (self.num_models, -1, q, 1))
        E = tf.maximum(E, self.epsilon)
        R = tf.maximum(R, self.epsilon)
        scaled_forward = tf.multiply(E, R, name="scaled_forward")
        S = tf.reduce_sum(scaled_forward, axis=-1, keepdims=True)
        loglik = old_loglik + tf.math.log(S) 
        scaled_forward /= S
        scaled_forward = tf.reshape(scaled_forward, (-1, q*self.max_num_states))
        loglik = tf.reshape(loglik, (-1, q))
        new_state = [scaled_forward, loglik]
        if self.reverse:
            output = tf.math.log(R) 
            output = tf.reshape(output, (-1, q*self.max_num_states))
            old_loglik = tf.reshape(old_loglik, (-1, q))
            output = tf.concat([output, old_loglik], axis=-1) 
        else:
            output = tf.math.log(scaled_forward) 
            output = tf.concat([output, loglik], axis=-1)
        if not self.reverse and self.use_step_counter:
            self.step_counter.assign_add(1)
        return (output, new_state)

    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None, parallel_factor=1):
        """ Returns the initial recurrent state which is a pair of tensors: the scaled 
            forward probabilities of shape (num_models*batch, num_states) 
            and the log likelihood (num_models*batch, 1).
            The return values can be savely reshaped to (num_models, batch, ...).
            If parallel, the returned tensors are of hape (num_models*batch, num_states*num_states)
            and (num_models*batch, num_states).
        """
        if parallel_factor == 1:
            if self.reverse:
                init_dist = tf.ones((self.num_models*batch_size, self.max_num_states), dtype=self.dtype)
            else:
                init_dist = tf.repeat(self.make_initial_distribution(), repeats=batch_size, axis=0)
                init_dist = tf.transpose(init_dist, (1,0,2))
                init_dist = tf.reshape(init_dist, (-1, self.max_num_states))
            loglik = tf.zeros((self.num_models*batch_size, 1), dtype=self.dtype)
            return [init_dist, loglik]
        else:
            indices = tf.range(self.max_num_states, dtype=tf.int32)
            indices = tf.tile(indices, [self.num_models*batch_size])
            init_dist = tf.one_hot(indices, self.max_num_states)
            if self.reverse:
                init_dist_chunk = tf.reshape(init_dist, (self.num_models*batch_size, self.max_num_states, self.max_num_states))
                #inputs shape =  (num_model*b*parallel_factor, chunk_size, q)
                first_emissions = inputs[:, 0, :]
                first_emissions = tf.reshape(first_emissions, (self.num_models, batch_size//parallel_factor, parallel_factor, self.max_num_states))
                first_emissions = tf.roll(first_emissions, shift=-1, axis=2)
                first_emissions = tf.reshape(first_emissions, (self.num_models*batch_size, 1, self.max_num_states))
                init_dist_chunk *= first_emissions
            else:
                init_dist_chunk = init_dist
            init_dist_chunk = tf.reshape(init_dist_chunk, (self.num_models, batch_size*self.max_num_states, self.max_num_states))
            init_dist_trans = self.transitioner(init_dist_chunk) 
            init_dist_trans = tf.reshape(init_dist_trans, (self.num_models, batch_size//parallel_factor, parallel_factor, self.max_num_states*self.max_num_states))
            is_first_chunk = tf.zeros((self.num_models, batch_size//parallel_factor, parallel_factor-1, self.max_num_states*self.max_num_states), dtype=self.dtype)
            if self.reverse:
                is_first_chunk = tf.concat([is_first_chunk, tf.ones_like(is_first_chunk[...,:1,:])], axis=2)
            else:       
                is_first_chunk = tf.concat([tf.ones_like(is_first_chunk[...,:1,:]), is_first_chunk], axis=2)
            init_dist = tf.reshape(init_dist, (self.num_models, batch_size//parallel_factor, parallel_factor, self.max_num_states*self.max_num_states))
            init_dist = is_first_chunk * init_dist + (1-is_first_chunk) * init_dist_trans
            init_dist = tf.reshape(init_dist, (self.num_models*batch_size, self.max_num_states*self.max_num_states))
            loglik = tf.zeros((self.num_models*batch_size, self.max_num_states), dtype=self.dtype)
            return [init_dist, loglik]


    def get_aux_loss(self):
        return sum([em.get_aux_loss() for em in self.emitter])

    
    def get_prior_log_density(self):  
        em_priors = [tf.reduce_sum(em.get_prior_log_density(), 1) for em in self.emitter]
        trans_priors = self.transitioner.get_prior_log_densities()
        prior = sum(em_priors) + sum(trans_priors.values())
        return prior
    
    
    def duplicate(self, model_indices=None, shared_kernels=False):
        """ Returns a new cell by copying the models specified in model_indices from this cell. 
        """
        assert self.built, "Can only duplicate a cell that was built before (i.e. it has kernels)."
        if model_indices is None:
            model_indices = range(self.num_models)
        sub_num_states = [self.num_states[i] for i in model_indices]
        sub_emitter = [e.duplicate(model_indices, shared_kernels) for e in self.emitter]
        sub_transitioner = self.transitioner.duplicate(model_indices, shared_kernels)
        subset_cell = HmmCell(sub_num_states, self.dim, sub_emitter, sub_transitioner)
        return subset_cell
    
    
    def make_reverse_direction_offspring(self):
        """ Returns a cell sharing this cells parameters that is configured for computing the backward recursion.
        """
        reverse_cell = self.duplicate(shared_kernels=True)
        reverse_cell.reverse_direction()
        reverse_cell.built = True
        reverse_cell.recurrent_init()
        return reverse_cell


    def reverse_direction(self, reverse=True):
        self.reverse = reverse
        self.transitioner.reverse = reverse

        
    def get_config(self):
        config = super(HmmCell, self).get_config()
        config.update({
             "num_states" : self.num_states, 
             "dim" : self.dim, 
             "num_emitters" : len(self.emitter),
             "transitioner" : self.transitioner
        })
        for i, em in enumerate(self.emitter):
             config[f"emitter_{i}"] = em
        return config
    
    
    @classmethod
    def from_config(cls, config):
        emitter = []
        for i in range(config["num_emitters"]):
            em = config[f"emitter_{i}"]
            config.pop(f"emitter_{i}")
            emitter.append(deserialize(em))
        config.pop("num_emitters")
        config["emitter"] = emitter
        config["transitioner"] = deserialize(config["transitioner"])
        return cls(**config)




class MsaHmmCell(HmmCell):
    """ A cell for profile HMMs that computes the forward recursion of the forward algorithm.
    Args:
        length: Model length / number of match states or a list of lengths.
        dim: The number of dimensions of the input sequence.
        emitter: An object or a list of objects following the emitter interface (see MultinomialAminoAcidEmitter).
        transitioner: An object following the transitioner interface (see ProfileHMMTransitioner).
    """
    def __init__(self,
                 length, 
                 dim=24,
                 emitter = None,
                 transitioner = None,
                 **kwargs
                ):
        if emitter is None:
            emitter = emit.ProfileHMMEmitter()
        if transitioner is None:
            transitioner = trans.ProfileHMMTransitioner()
        self.length = [length] if not hasattr(length, '__iter__') else length 
        super(MsaHmmCell, self).__init__(get_num_states(self.length), dim, emitter, transitioner, **kwargs)
        for em in self.emitter:
            em.set_lengths(self.length)
        self.transitioner.set_lengths(self.length)
    
    
    def duplicate(self, model_indices=None, shared_kernels=False):
        """ Returns a new cell by copying the models specified in model_indices from this cell. 
        """
        assert self.built, "Can only duplicate a cell that was built before (i.e. it has kernels)."
        if model_indices is None:
            model_indices = range(self.num_models)
        sub_lengths = [self.length[i] for i in model_indices]
        sub_emitter = [e.duplicate(model_indices, shared_kernels) for e in self.emitter]
        sub_transitioner = self.transitioner.duplicate(model_indices, shared_kernels)
        subset_cell = MsaHmmCell(sub_lengths, self.dim, sub_emitter, sub_transitioner)
        return subset_cell

        
    def get_config(self):
        config = super(MsaHmmCell, self).get_config()
        config["length"] = self.length.tolist() if isinstance(self.length, np.ndarray) else self.length
        del config["num_states"]
        return config


tf.keras.utils.get_custom_objects()["MsaHmmCell"] = MsaHmmCell