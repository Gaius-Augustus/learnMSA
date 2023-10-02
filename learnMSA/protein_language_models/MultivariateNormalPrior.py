import tensorflow as tf
import tensorflow_probability as tfp
import math


class MultivariateNormalPrior(tf.keras.layers.Layer):
    """A multivariate normal prior over R^d parameterized by d locations 
    and a dxd lower triangular scale matrix.

    Args:
        comp_count: The number of components in the mixture.
        epsilon: A small constant for numerical stability.
    """
    def __init__(self, 
                dim, 
                mu_init = "zeros", 
                scale_kernel_init = "random_normal", 
                precomputed = False,
                trainable = True,
                **kwargs):
        super(MultivariateNormalPrior, self).__init__(**kwargs)
        self.dim = dim
        self.mu_init = mu_init
        self.scale_kernel_init = scale_kernel_init
        self.precomputed = precomputed
        self.trainable = trainable
        self.constant = self.dim * tf.math.log(2*math.pi)


    def build(self, input_shape):
        self.scale_tril = tfp.bijectors.FillScaleTriL()
        self.mu = self.add_weight(name="mu", shape=(self.dim,), initializer=self.mu_init)
        self.scale_kernel = self.add_weight(name="scale_kernel", 
                                            shape=(self.dim * (self.dim+1) // 2,), 
                                            initializer=self.scale_kernel_init,
                                            trainable=self.trainable)
        if self.precomputed:
            self.compute_values()

    def compute_values(self):
        self.scale = self.scale_tril(self.scale_kernel)
        self.pinv = tf.linalg.pinv(self.scale)
        self.log_det = 2*tf.reduce_sum(tf.math.log(tf.linalg.diag_part(self.scale)), -1)
                                
    def covariance_matrix(self):
        scale_tril = self.scale_tril(self.scale_kernel)
        return tf.matmul(scale_tril, tf.transpose(scale_tril))


    def call(self, inputs):
        """Returns log density values for observations.

        Args:
            inputs: A b x d batch of observations.

        Returns:
            A b-dimensional vector of log density values.
        """
        if not self.precomputed:
            self.compute_values()
        diff = inputs - self.mu
        y = tf.matmul(self.pinv, diff, transpose_b=True)
        MD_sq = tf.reduce_sum(y * y, 0) 
        return -0.5 * (self.constant + self.log_det + MD_sq)



def make_pdf_model(lm_dim, precomputed=False, trainable=True, reduce=True):
    embeddings = tf.keras.Input((None, lm_dim))
    # compute log pdf per observation
    stack = tf.reshape(embeddings, (-1, lm_dim))
    log_pdf = MultivariateNormalPrior(lm_dim, precomputed=precomputed, trainable=trainable)(stack)
    log_pdf = tf.reshape(log_pdf, (tf.shape(embeddings)[0], tf.shape(embeddings)[1]))
    if reduce:
        # compute sequence lengths
        mask = tf.reduce_any(tf.not_equal(embeddings, 0), -1)
        mask = tf.cast(mask, log_pdf.dtype)
        seq_lens = tf.reduce_sum(mask, -1)
        # average per sequence
        log_pdf = tf.reduce_sum(log_pdf * mask, -1) / tf.maximum(seq_lens, 1.)
        # average over batch
        log_pdf = tf.reduce_mean(log_pdf)
    model = tf.keras.Model(inputs=[embeddings], outputs=[log_pdf])
    return model