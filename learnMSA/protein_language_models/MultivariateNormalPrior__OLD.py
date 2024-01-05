import tensorflow as tf
import tensorflow_probability as tfp
import math
import numpy as np


class MultivariateNormalPrior(tf.keras.layers.Layer):
    """A multivariate normal prior over R^d parameterized by d locations 
    and a dxd lower triangular scale matrix.

    Args:
        dim: The dimensionality of the multivariate normal.
        components: The number of components in the mixture.
        mu_init: The initializer for the location parameter.
        scale_kernel_init: The initializer for a kernel matrix closely related to the lower triangular scale matrix.
        diag_init_var: The initial variance (diagonal entries of the covariance matrix).
        precomputed: If True, the scale matrix is not recomputed in each call of the layer.
        trainable: If True, the parameters of the layer are trainable.
    """
    def __init__(self, 
                dim, 
                components = 1,
                mu_init = "zeros", 
                scale_kernel_init = tf.random_normal_initializer(stddev=0.01),
                mixture_kernel_init = tf.random_normal_initializer(stddev=0.01),
                diag_init_var = 1., 
                precomputed = False,
                trainable = True,
                **kwargs):
        super(MultivariateNormalPrior, self).__init__(**kwargs)
        self.dim = dim
        self.components = components
        self.mu_init = mu_init
        self.scale_kernel_init = scale_kernel_init
        diag_init_std = np.sqrt(diag_init_var).astype(np.float32)
        self.scale_diag_init = tfp.math.softplus_inverse(diag_init_std)
        self.mixture_kernel_init = mixture_kernel_init
        self.precomputed = precomputed
        self.trainable = trainable
        self.constant = self.dim * tf.math.log(2*math.pi)


    def build(self, input_shape):
        self.scale_tril = tfp.bijectors.FillScaleTriL(diag_bijector=DiagBijector(self.scale_diag_init))
        self.mu = self.add_weight(name="mu", 
                                  shape=(self.components, self.dim), 
                                  initializer=self.mu_init,
                                  trainable=self.trainable)
        self.scale_kernel = self.add_weight(name="scale_kernel", 
                                            shape=(self.components, self.dim * (self.dim+1) // 2,), 
                                            initializer=self.scale_kernel_init,
                                            trainable=self.trainable)
        if self.components > 1:
            self.mixture_kernel = self.add_weight(name="mixture_kernel",
                                                    shape=(self.components,1),
                                                    initializer=self.mixture_kernel_init,
                                                    trainable=self.trainable)
        if self.precomputed:
            self.compute_values()

    def compute_values(self):
        self.scale = self.scale_tril(self.scale_kernel)
        self.pinv = tf.linalg.pinv(self.scale)
        self.log_det = 2*tf.reduce_sum(tf.math.log(tf.linalg.diag_part(self.scale)), -1, keepdims=True)
        if self.components > 1:
            self.mixture = tf.nn.softmax(self.mixture_kernel, axis=0)
                                
    def covariance_matrix(self):
        scale_tril = self.scale_tril(self.scale_kernel)
        return tf.matmul(scale_tril, tf.transpose(scale_tril, [0,2,1]))

    def mean(self):
        if self.components == 1:
            return self.mu[0]
        else:
            self.compute_values()
            return tf.linalg.matvec(tf.transpose(self.mu), tf.squeeze(self.mixture))

    def component_log_pdf(self, inputs):
        if not self.precomputed:
            self.compute_values()
        diff = tf.expand_dims(inputs, 1) - tf.expand_dims(self.mu, 0) # (b, components, d)
        y = tf.matmul(self.pinv, tf.transpose(diff, [1,2,0])) # (components, d, b)
        MD_sq_components = tf.reduce_sum(y * y, -2) 
        log_pdf_components = -0.5 * (self.constant + self.log_det + MD_sq_components)
        return log_pdf_components

    def call(self, inputs):
        """Returns log density values for observations.

        Args:
            inputs: A b x d batch of observations.

        Returns:
            A b-dimensional vector of log density values.
        """
        log_pdf_components = self.component_log_pdf(inputs)
        if self.components == 1:
            return log_pdf_components[0]
        else:
            return tf.math.reduce_logsumexp(log_pdf_components + tf.math.log(self.mixture), 0)


def make_pdf_model(lm_dim, components=1, precomputed=False, trainable=True, aggregate_result=False):
    embeddings = tf.keras.Input((None, lm_dim))
    # compute log pdf per observation
    stack = tf.reshape(embeddings, (-1, lm_dim))
    log_pdf = MultivariateNormalPrior(lm_dim, components, precomputed=precomputed, trainable=trainable)(stack)
    log_pdf = tf.reshape(log_pdf, (tf.shape(embeddings)[0], tf.shape(embeddings)[1]))
    # zero out pdfs of zero embeddings (assumed padding)
    mask = tf.reduce_any(tf.not_equal(embeddings, 0), -1)
    log_pdf *= tf.cast(mask, log_pdf.dtype)
    if aggregate_result:
        log_pdf = aggregate(log_pdf)
    model = tf.keras.Model(inputs=[embeddings], outputs=[log_pdf])
    return model

def aggregate(log_pdf):
    """ Reduces per-residue embedding log_pdfs to a scalar by averaging over sequences and batch.
    """
    # compute sequence lengths
    mask = tf.not_equal(log_pdf, 0)
    mask = tf.cast(mask, log_pdf.dtype)
    seq_lens = tf.reduce_sum(mask, -1)
    # average per sequence
    log_pdf = tf.reduce_sum(log_pdf * mask, -1) / tf.maximum(seq_lens, 1.)
    # average over batch
    log_pdf = tf.reduce_mean(log_pdf)
    return log_pdf

class DiagBijector(tfp.bijectors.Softplus):
    def __init__(self, diagonal_value):
        super(DiagBijector, self).__init__()
        self.diagonal_value = diagonal_value

    def forward(self, x, name='forward', **kwargs):
        return super(DiagBijector, self).forward(x + self.diagonal_value, name, **kwargs)