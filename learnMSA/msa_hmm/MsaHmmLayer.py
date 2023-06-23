import tensorflow as tf
import numpy as np

class MsaHmmLayer(tf.keras.layers.Layer):
    def __init__(self, 
                 cell, 
                 num_seqs=None,
                 use_prior=True,
                 sequence_weights=None,
                 **kwargs
                ):
        super(MsaHmmLayer, self).__init__(**kwargs)
        self.cell = cell
        self.rnn = tf.keras.layers.RNN(self.cell, 
                                       return_sequences=True, 
                                       return_state=True)
        self.rnn_backward = tf.keras.layers.RNN(self.cell, 
                                                return_sequences=True, 
                                                return_state=True,
                                                go_backwards=True)
        self.num_seqs = num_seqs
        self.use_prior = use_prior 
        self.sequence_weights = sequence_weights
        if sequence_weights is not None:
            self.weight_sum = np.sum(sequence_weights)
        
        
    def build(self, input_shape):
        # build the cell 
        self.cell.build((None, input_shape[-2], input_shape[-1]))
        # build the RNN layer with a different input shape
        self.rnn.build((None, input_shape[-2], self.cell.max_num_states))
        self.built = True
        
        
    def forward_recursion(self, inputs, training=False):
        """ Computes the forward recursion for multiple models where each model
            receives a batch of sequences as input.
        Args:
            inputs: Sequences. Shape: (num_model, b, seq_len, s)
        Returns:
            forward variables: Shape: (num_model, b, seq_len, q)
            log-likelihoods: Shape: (num_model, b)
        """
        #initialize transition- and emission-matricies
        self.cell.recurrent_init()
        num_model, b, seq_len, s = tf.unstack(tf.shape(inputs))
        initial_state = self.cell.get_initial_state(batch_size=b)
        #reshape to 3D inputs for RNN (cell will reshape back in each step)
        emission_probs = self.cell.emission_probs(inputs)
        emission_probs = tf.reshape(emission_probs, (num_model*b, seq_len, self.cell.max_num_states))
        #do one initialization step
        #this way, tf will compile two versions of the cell call, one with init=True and one without
        forward_1, step_1_state = self.cell(emission_probs[:,0], initial_state, training, init=True)
        #run forward with the output of the first step as initial state
        forward, _, loglik = self.rnn(emission_probs[:,1:], initial_state=step_1_state, training=training)
        #prepend the separate first step to the other forward steps
        forward = tf.concat([forward_1[:,tf.newaxis], forward], axis=1)
        forward = tf.reshape(forward, (num_model, b, seq_len, -1))
        loglik = tf.reshape(loglik, (num_model, b))
        return forward, loglik
    
    
    def backward_recursion(self, inputs):
        """ Computes the backward recursion for multiple models where each model
            receives a batch of sequences as input.
        Args:
            inputs: Sequences. Shape: (num_model, b, seq_len, s)
        Returns:
            backward variables: Shape: (num_model, b, seq_len, q)
        """
        self.cell.recurrent_init()
        num_model, b, seq_len, s = tf.unstack(tf.shape(inputs))
        initial_state = self.cell.get_initial_backward_state(batch_size=b)
        emission_probs = self.cell.emission_probs(inputs)
        emission_probs = tf.reshape(emission_probs, (num_model*b, seq_len, self.cell.max_num_states))
        self.cell.reverse_direction()
        #note that for backward, we can ignore the initial step like we did it in
        #forward, because we assume that all inputs have terminal tokens
        backward, _, _ = self.rnn_backward(emission_probs, initial_state=initial_state)
        self.cell.reverse_direction()
        backward = tf.reshape(backward, (num_model, b, seq_len, -1))
        backward = tf.reverse(backward, [-2])
        return backward
    
    
    def state_posterior_log_probs(self, inputs):
        """ Computes the log-probability of state q at position i given inputs.
        Args:
            inputs: Sequences. Shape: (num_model, b, seq_len, s)
        Returns:
            state posterior probbabilities: Shape: (num_model, b, seq_len, q)
        """
        forward, loglik = self.forward_recursion(inputs)
        backward = self.backward_recursion(inputs)
        loglik = loglik[:,:,tf.newaxis,tf.newaxis]
        return forward + backward - loglik
    
    
    def apply_sequence_weights(self, loglik, indices, aggregate=False):
        if self.sequence_weights is not None:
            weights = tf.gather(self.sequence_weights, indices)
            loglik *= weights
            if aggregate:
                loglik = tf.reduce_sum(loglik, axis=1) / tf.reduce_sum(weights, axis=1) #mean over batch 
                loglik = tf.reduce_mean(loglik) #mean over models
        elif aggregate:
            loglik = tf.reduce_mean(loglik) #mean over both models and batches
        return loglik
    
    
    #compute the prior, scale it depending on seq weights
    def compute_prior(self):
        prior = self.cell.get_prior_log_density(add_metrics=False)
        prior = tf.reduce_mean(prior)
        if self.sequence_weights is not None:
            prior /= self.weight_sum
        elif self.num_seqs is not None:
            prior /= self.num_seqs
        return prior
        
        
    def call(self, inputs, indices=None, training=False):
        """ Computes log-likelihoods per model and sequence.
        Args:
            inputs: Sequences. Shape: (num_model, b, seq_len, s)
            indices: Optional sequence indices required to assign sequence weights. Shape: (num_model, b)
        Returns:
            log-likelihoods: Sequences. Shape: (num_model, b)
        """
        inputs = tf.cast(inputs, self.dtype)
        _, loglik = self.forward_recursion(inputs, training=training)
        loglik_mean = self.apply_sequence_weights(loglik, indices, aggregate=True)
        if self.use_prior:
            prior = self.compute_prior()
            MAP = loglik_mean + prior
            self.add_loss(tf.squeeze(-MAP))
        else:
            self.add_loss(tf.squeeze(-loglik_mean))
        #tensorflow output summary statistics-
        if training:
            self.add_metric(loglik_mean, "loglik")
            if self.use_prior:
                self.add_metric(prior, "logprior")
        return loglik
        
        
    def get_config(self):
        config = super(MsaHmmLayer, self).get_config()
        config.update({ 
             "cell" : self.cell,
             "num_seqs" : self.num_seqs,
             "use_prior" : self.use_prior, 
             "sequence_weights" : self.sequence_weights
        })
        return config