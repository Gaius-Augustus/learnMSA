import tensorflow as tf


class TotalProbabilityCell(tf.keras.layers.Layer):
    """ A utility RNN cell that computes the total forward probabilities based on the conditional
        forward probabilities of chunked observations given the state at their left border.
    Args:
        cell: HMM cell whose forward recursion is used.
    """
    def __init__(self, cell, **kwargs):
        super(TotalProbabilityCell, self).__init__(**kwargs)
        self.cell = cell
        self.reverse = False
        
    def build(self, input_shape):
        self.built = True

    def recurrent_init(self):
        pass
    
    def make_initial_distribution(self):
        """ 
        Returns:
            A probability distribution over the states per model. Shape: (1, k, q)"""
        return self.cell.transitioner.make_initial_distribution()
    
    def emission_probs(self, inputs, end_hints=None, training=False):
        return inputs

    def call(self, conditional_forward, states, training=None, init=False):
        """ Args:
            conditional_forward: A batch of conditional forward probabilities. Shape: (k*b, z, q)
        """
        old_scaled_forward, old_loglik = states
        old_scaled_forward = tf.reshape(old_scaled_forward, (self.cell.num_models, -1, self.cell.max_num_states))
        old_loglik = tf.reshape(old_loglik, (self.cell.num_models, -1, 1))
        conditional_forward = tf.reshape(conditional_forward, (self.num_models, -1, self.max_num_states, self.max_num_states))
        if init:
            scaled_forward = old_scaled_forward
        else:
            scaled_forward = tf.einsum("kbz,kbzq->kbq", old_scaled_forward, conditional_forward)
        S = tf.reduce_sum(scaled_forward, axis=-1, keepdims=True)
        loglik = old_loglik + tf.math.log(S) 
        scaled_forward /= S
        loglik = tf.reshape(loglik, (-1, 1))
        scaled_forward = tf.reshape(scaled_forward, (-1, self.max_num_states))
        new_state = [scaled_forward, loglik]
        output = tf.math.log(scaled_forward) 
        output = tf.concat([output, loglik], axis=-1)
        return output, new_state
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if self.reverse:
            init_dist = tf.ones((self.num_models*batch_size, self.max_num_states), dtype=self.dtype)
            loglik = tf.zeros((self.num_models*batch_size, 1), dtype=self.dtype)
            S = [init_dist, loglik]
        else:
            init_dist = tf.repeat(self.make_initial_distribution(), repeats=batch_size, axis=0)
            init_dist = tf.transpose(init_dist, (1,0,2))
            init_dist = tf.reshape(init_dist, (-1, self.max_num_states))
            loglik = tf.zeros((self.num_models*batch_size, 1), dtype=self.dtype)
            S = [init_dist, loglik]
        return S

    def get_prior_log_density(self, add_metrics=False):  
        return 0.
    
    def duplicate(self, model_indices=None, shared_kernels=False):
        return TotalProbabilityCell(self.cell)
    
    def make_reverse_direction_offspring(self):
        """ Returns a cell sharing this cells parameters that is configured for computing the backward recursion.
        """
        reverse_cell = self.duplicate()
        reverse_cell.reverse_direction()
        reverse_cell.built = True
        return reverse_cell

    def reverse_direction(self, reverse=True):
        self.reverse = reverse

    def get_config(self):
        config = super(TotalProbabilityCell, self).get_config()
        config.update({"cell" : self.cell})
        return config

tf.keras.utils.get_custom_objects()["TotalProbabilityCell"] = TotalProbabilityCell