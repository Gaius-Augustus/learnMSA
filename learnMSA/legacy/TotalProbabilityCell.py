import tensorflow as tf


class TotalProbabilityCell(tf.keras.layers.Layer):
    """ A utility RNN cell that computes the total forward probabilities based on the conditional
        forward probabilities of chunked observations given the state at their left border 
        in logarithmic scale to avoid underflow.
    Args:
        cell: HMM cell whose forward recursion is used.
        reverse: If True, the cell is configured for computing the backward recursion.
    """
    def __init__(self, cell, reverse=False, **kwargs):
        super(TotalProbabilityCell, self).__init__(**kwargs)
        self.cell = cell
        self.reverse = reverse
        self.state_size = (tf.TensorShape([self.cell.max_num_states]), tf.TensorShape([]))
    
    def make_initial_distribution(self):
        """ 
        Returns:
            A probability distribution over the states per model. Shape: (1, k, q)"""
        return self.cell.transitioner.make_initial_distribution()

    def call(self, conditional_forward, states, training=None, init=False):
        """ Args:
            conditional_forward: A batch of conditional logarithmic forward probabilities. Shape: (b, q*q)
            The rows correspond to the states that are conditioned on.
            Returns: The log of the total forward probabilities. Shape: (b, q)
        """
        old_forward, _ = states
        conditional_forward = tf.reshape(conditional_forward, (-1, self.cell.max_num_states, self.cell.max_num_states))
        forward = old_forward[...,tf.newaxis] + conditional_forward
        forward = tf.math.reduce_logsumexp(forward, axis=-2)
        loglik = tf.math.reduce_logsumexp(forward, axis=-1)
        new_state = [forward, loglik]
        output = forward

        return output, new_state
    
    def get_initial_state(self, batch_size=None, inputs=None, dtype=None):
        if self.reverse:
            init_dist = tf.zeros((batch_size, self.cell.max_num_states), dtype=self.dtype)
            loglik = tf.zeros((batch_size), dtype=self.dtype)
            S = [init_dist, loglik]
        else:
            init_dist = tf.repeat(self.make_initial_distribution(), repeats=batch_size//self.cell.num_models, axis=0)
            init_dist = tf.transpose(init_dist, (1,0,2))
            init_dist = tf.reshape(init_dist, (-1, self.cell.max_num_states))
            loglik = tf.zeros((batch_size), dtype=self.dtype)
            S = [tf.math.log(init_dist), loglik]
        return S

    def get_config(self):
        config = super(TotalProbabilityCell, self).get_config()
        config.update({"cell" : self.cell, "reverse" : self.reverse})
        return config

tf.keras.utils.get_custom_objects()["TotalProbabilityCell"] = TotalProbabilityCell