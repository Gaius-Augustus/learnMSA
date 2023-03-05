import os
import tensorflow as tf
import numpy as np
import learnMSA.msa_hmm.Emitter as emit
import learnMSA.msa_hmm.Transitioner as trans

class MsaHmmCell(tf.keras.layers.Layer):
    """ A general cell for (p)HMM training. It is meant to be used with the generic RNN-layer.
        It computes the likelihood of a batch of sequences, computes a prior value and provides 
        functionality (through the injected emitter and transitioner) to construct the emission- 
        and transition-matricies also used elsewhere e.g. during Viterbi.
        Based on https://github.com/mslehre/classify-seqs/blob/main/HMMCell.py.
    Args:
        length: Model length / number of match states or a list of lengths.
        emitter: An object or a list of objects following the emitter interface (see MultinomialAminoAcidEmitter).
        transitioner: An object following the transitioner interface (see ProfileHMMTransitioner).
        dtype: The datatype of the cell.
    """
    def __init__(self,
                 length, 
                 emitter = None,
                 transitioner = None,
                 **kwargs
                ):
        super(MsaHmmCell, self).__init__(**kwargs)
        if emitter is None:
            emitter = emit.ProfileHMMEmitter()
        if transitioner is None:
            transitioner = trans.ProfileHMMTransitioner()
        self.length = [length] if not hasattr(length, '__iter__') else length 
        self.num_models = len(self.length)
        self.emitter = [emitter] if not hasattr(emitter, '__iter__') else emitter 
        self.transitioner = transitioner
        #number of emitting states, i.e. not counting flanking states and deletions
        self.num_states = [2 * length + 3 for length in self.length]  
        self.num_states_implicit = [num_states + length + 2 
                                    for num_states, length in zip(self.num_states, self.length)]
        self.max_num_states = max(self.num_states)
        self.state_size = (tf.TensorShape([self.max_num_states]), tf.TensorShape([1]))
        self.output_size = tf.TensorShape([self.max_num_states])
        for em in self.emitter:
            em.cell_init(self)
        self.transitioner.cell_init(self)
        self.epsilon = tf.constant(1e-32, self.dtype)
        self.reverse = False
            
            
    def build(self, input_shape):
        self.dim = input_shape[-1]
        for em in self.emitter:
            em.build(input_shape)
        self.transitioner.build(input_shape)
        self.built = True

        
    def recurrent_init(self):
        self.transitioner.recurrent_init()
        for em in self.emitter:
            em.recurrent_init()
        self.log_A_dense = self.transitioner.make_log_A()
        self.log_A_dense_t = tf.transpose(self.log_A_dense, [0,2,1])
        self.init_dist = self.make_initial_distribution()
    
    
    def make_initial_distribution(self):
        """Constructs the initial state distribution which depends on the transition probabilities.
            See ProfileHMMTransitioner.
        Returns:
            A probability distribution of shape: (1, num_model, q)
        """
        return self.transitioner.make_initial_distribution()
        
    
    def emission_probs(self, inputs):
        """ Computes the probabilities of emission per state for the given observation. Multiple emitters
            are multiplied.
        Args:
            inputs: A batch of sequence positions.
        """
        em_probs = self.emitter[0](inputs)
        for em in self.emitter[1:]:
            em_probs *= em(inputs)
        return em_probs

    
    def call(self, inputs, states, training=None, init=False):
        """ Computes one recurrent step of the Forward DP.
        """
        old_scaled_forward, old_loglik = states
        old_scaled_forward = tf.reshape(old_scaled_forward, (self.num_models, -1, self.max_num_states))
        old_loglik = tf.reshape(old_loglik, (self.num_models, -1, 1))
        inputs = tf.reshape(inputs, (self.num_models, -1, self.dim))
        E = self.emission_probs(inputs)
        if init:
            R = old_scaled_forward
        else:
            R = self.transitioner(old_scaled_forward)
        scaled_forward = tf.multiply(E, R, name="scaled_forward")
        S = tf.reduce_sum(scaled_forward, axis=-1, keepdims=True)
        loglik = old_loglik + tf.math.log(S) 
        scaled_forward /= S 
        loglik = tf.reshape(loglik, (-1, 1))
        scaled_forward = tf.reshape(scaled_forward, (-1, self.max_num_states))
        new_state = [scaled_forward, loglik]
        if self.reverse:
            log_unscaled_forward = tf.math.log(R + self.epsilon) + old_loglik
            log_unscaled_forward = tf.reshape(log_unscaled_forward, (-1, self.max_num_states))
        else:
            log_unscaled_forward = tf.math.log(scaled_forward + self.epsilon) + loglik
        return log_unscaled_forward, new_state

    
    def get_initial_state(self, inputs=None, batch_size=None, _dtype=None):
        init_dist = tf.repeat(self.make_initial_distribution(), repeats=batch_size, axis=0)
        init_dist = tf.transpose(init_dist, (1,0,2))
        init_dist = tf.reshape(init_dist, (-1, self.max_num_states))
        loglik = tf.zeros((self.num_models*batch_size, 1), dtype=self.dtype)
        S = [init_dist, loglik]
        return S

    
    def get_initial_backward_state(self, inputs=None, batch_size=None, _dtype=None):
        init_dist = tf.ones((self.num_models*batch_size, self.max_num_states), dtype=self.dtype)
        loglik = tf.zeros((self.num_models*batch_size, 1), dtype=self.dtype)
        S = [init_dist, loglik]
        return S

    
    def get_prior_log_density(self, add_metrics=False):  
        em_priors = [tf.reduce_sum(em.get_prior_log_density(), 1) for em in self.emitter]
        trans_priors = self.transitioner.get_prior_log_densities()
        prior = sum(em_priors) + sum(trans_priors.values())
        if add_metrics:
            for i,d in enumerate(em_priors):
                d = tf.reduce_mean(d)
                self.add_metric(d, "mean_model_em_prior_"+str(i))
            for name, d in trans_priors.items():
                d = tf.reduce_mean(d)
                self.add_metric(d, "mean_model_"+name)
        return prior
    
    
    def duplicate(self, model_indices=None):
        """ Returns a new cell by copying the models specified in model_indices from this cell. 
        """
        assert self.built, "Can only duplicate a cell that was built before (i.e. it has kernels)."
        if model_indices is None:
            model_indices = range(self.num_models)
        sub_lengths = [self.length[i] for i in model_indices]
        sub_emitter = [e.duplicate(model_indices) for e in self.emitter]
        sub_transitioner = self.transitioner.duplicate(model_indices)
        subset_cell = MsaHmmCell(sub_lengths, sub_emitter, sub_transitioner)
        subset_cell.dim = self.dim
        return subset_cell
    
    
    #configures the cell for the backward recursion
    def reverse_direction(self):
        self.transitioner.transpose()
        self.reverse = not self.reverse
        
    def get_config(self):
        config = super(MsaHmmCell, self).get_config()
        config.update({
             "length" : self.length, 
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
            emitter.append(config[f"emitter_{i}"])
            config.pop(f"emitter_{i}")
        config.pop("num_emitters")
        config["emitter"] = emitter
        return cls(**config)

