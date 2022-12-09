import tensorflow as tf
import numpy as np

class MsaHmmLayer(tf.keras.layers.Layer):
    def __init__(self, 
                 cell, 
                 num_seq,
                 use_prior=True,
                 name="MsaHmmLayer",
                 dtype=tf.float32,
                 **kwargs
                ):
        super(MsaHmmLayer, self).__init__(name=name, dtype=dtype, **kwargs)
        self.num_seq = num_seq
        self.cell = cell
        self.rnn = tf.keras.layers.RNN(self.cell, return_sequences=True, return_state=True)
        self.use_prior = use_prior 
        
        
    def build(self, input_shape):
        self.rnn.build((None, input_shape[-2], input_shape[-1])) #also builds the cell
        
        
    def call(self, inputs, training=False):
        self.cell.recurrent_init()
        initial_state = self.cell.get_initial_state(batch_size=tf.shape(inputs)[1])
        inputs = tf.cast(inputs, self.dtype)
        inputs = tf.reshape(inputs, (-1, tf.shape(inputs)[-2], tf.shape(inputs)[-1]))
        _, _, loglik = self.rnn(inputs, initial_state=initial_state, training=training)
        loglik = tf.reshape(loglik, (self.cell.num_models, -1))
        loglik_mean = tf.reduce_mean(loglik) #mean over both models and batches
        if self.use_prior:
            prior = self.cell.get_prior_log_density(add_metrics=False)
            prior = tf.reduce_mean(prior)
            prior /= self.num_seq
            MAP = loglik_mean + prior
            self.add_loss(tf.squeeze(-MAP))
        else:
            self.add_loss(tf.squeeze(-loglik_mean))
        if training:
            self.add_metric(loglik_mean, "loglik")
            if self.use_prior:
                self.add_metric(prior, "logprior")
        return loglik