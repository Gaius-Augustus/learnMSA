import os
import numpy as np
import copy
import random
import tensorflow as tf
import learnMSA.msa_hmm.DirichletMixture as dm

    
dtype = tf.float64
index_dtype = tf.int16


PRIOR_PATH = os.path.dirname(__file__)+"/trained_prior/"
DIRICHLET_COMP_COUNT = 1
model, DMP = dm.make_model(DIRICHLET_COMP_COUNT, 20, -1, trainable=False)
model.load_weights(PRIOR_PATH+str(DIRICHLET_COMP_COUNT)+"_components_prior_pdf/ckpt").expect_partial()
emission_dirichlet_mix = dm.DirichletMixturePrior(DIRICHLET_COMP_COUNT, 20, -1,
                                    DMP.alpha_kernel.numpy(),
                                    DMP.q_kernel.numpy(),
                                    trainable=False)
background_distribution = emission_dirichlet_mix.expectation()
#the prior was trained on example distributions over the 20 amino acid alphabet
#the additional frequencies for 'B', 'Z',  'X', 'U', 'O' were derived from Pfam
background_distribution = np.concatenate([background_distribution, [2.03808244e-05, 1.02731819e-05, 7.92076933e-04, 5.84256792e-08, 1e-32]], axis=0)
background_distribution /= np.sum(background_distribution)




def inverse_softplus(features):
    #cast to float 64 to prevent overflow of large entries
    features64 = features.astype(np.float64)
    return np.log(np.expm1(features64)).astype(features.dtype)
    

class DefaultDiagBijector():
    def __init__(self, base_variance, epsilon=1e-05):
        """ Args:
                base_variance: The initial variance (diagonal entries of the covariance matrix) if kernel = 0.
        """
        super(DefaultDiagBijector, self).__init__()
        base_std = np.sqrt(base_variance).astype(np.float32)
        self.scale_diag_init = tfp.math.softplus_inverse(base_std)
        self.epsilon = epsilon

    def forward(self, x):
        return tf.math.softplus(x + self.scale_diag_init) + self.epsilon

    def inverse(self, y):
        return inverse_softplus(y - self.epsilon) - self.scale_diag_init


def fill_triangular(x, upper=False, name=None):
    """Creates a (batch of) triangular matrix from a vector of inputs.

    Args:
        x: `Tensor` representing lower (or upper) triangular elements.
        upper: Python `bool` representing whether output matrix should be upper
          triangular (`True`) or lower triangular (`False`, default).
        name: Python `str`. The name to give this op.

    Returns:
        tril: `Tensor` with lower (or upper) triangular elements filled from `x`.

    Raises:
        ValueError: if `x` cannot be mapped to a triangular matrix.
    """

    with tf.name_scope(name or 'fill_triangular'):
        x = tf.convert_to_tensor(x, name='x')

        # Get the last dimension size (m)
        m = x.shape[-1]
        
        # Calculate n from m using the quadratic formula
        if m is not None:
            m = np.int32(m)
            n = np.sqrt(0.25 + 2. * m) - 0.5
            if n != np.floor(n):
                raise ValueError('Input right-most shape ({}) does not '
                                 'correspond to a triangular matrix.'.format(m))
            n = np.int32(n)
            static_final_shape = tf.TensorShape(x.shape[:-1]).concatenate([n, n])
        else:
            m = tf.shape(x)[-1]
            n = tf.cast(
                tf.sqrt(0.25 + tf.cast(2 * m, dtype=tf.float32)), dtype=tf.int32)
            static_final_shape = tf.TensorShape(x.shape[:-1]).concatenate([None, None])

        # Determine the shape of the output tensor
        ndims = tf.rank(x)
        if upper:
            x_list = [x, tf.reverse(x[..., n:], axis=[ndims - 1])]
        else:
            x_list = [x[..., n:], tf.reverse(x, axis=[ndims - 1])]
        
        new_shape = (
            static_final_shape.as_list()
            if static_final_shape.is_fully_defined() else tf.concat(
                [tf.shape(x)[:-1], [n, n]], axis=0))
        
        x = tf.reshape(tf.concat(x_list, axis=-1), new_shape)
        
        # Create a triangular matrix
        x = tf.linalg.band_part(
            x, num_lower=(0 if upper else -1), num_upper=(-1 if upper else 0))
        
        # Set the static shape if it is fully defined
        x.set_shape(static_final_shape)
        return x


def fill_triangular_inverse(x, upper=False, name=None):
    """Creates a vector from a (batch of) triangular matrix.

    Args:
        x: `Tensor` representing lower (or upper) triangular elements.
        upper: Python `bool` representing whether output matrix should be upper
          triangular (`True`) or lower triangular (`False`, default).
        name: Python `str`. The name to give this op.

    Returns:
        flat_tril: (Batch of) vector-shaped `Tensor` representing vectorized lower
          (or upper) triangular elements from `x`.
    """

    with tf.name_scope(name or 'fill_triangular_inverse'):
        x = tf.convert_to_tensor(x, name='x')
        
        # Get the last dimension size (n)
        n = x.shape[-1]
        
        if n is not None:
            n = np.int32(n)
            m = np.int32((n * (n + 1)) // 2)
            static_final_shape = tf.TensorShape(x.shape[:-2]).concatenate([m])
        else:
            n = tf.shape(x)[-1]
            m = (n * (n + 1)) // 2
            static_final_shape = tf.TensorShape(x.shape[:-2]).concatenate([None])
        
        ndims = tf.rank(x)
        if upper:
            initial_elements = x[..., 0, :]
            triangular_portion = x[..., 1:, :]
        else:
            initial_elements = tf.reverse(x[..., -1, :], axis=[ndims - 2])
            triangular_portion = x[..., :-1, :]

        rotated_triangular_portion = tf.reverse(
            tf.reverse(triangular_portion, axis=[ndims - 1]), axis=[ndims - 2])
        
        consolidated_matrix = triangular_portion + rotated_triangular_portion
        
        end_sequence = tf.reshape(
            consolidated_matrix,
            tf.concat([tf.shape(x)[:-2], [n * (n - 1)]], axis=0))
        
        y = tf.concat([initial_elements, end_sequence[..., :m - n]], axis=-1)
        
        y.set_shape(static_final_shape)
        return y



class FillScaleTriL():
    def __init__(diag_bijector):
        self.diag_bijector = diag_bijector

    def forward(self, x):
        y = fill_triangular(x) 
        diag = tf.linalg.diag_part(y)
        transformed_diag = self.diag_bijector.forward(diag)
        return tf.linalg.set_diag(y, transformed_diag)

    def inverse(self, y):
        diag = tf.linalg.diag_part(y)
        transformed_diag = self.diag_bijector.inverse(diag)
        x = tf.linalg.set_diag(y, transformed_diag)
        return fill_triangular_inverse(x)


def make_kernel(mean, scale, diag_bijector=DefaultDiagBijector(1.)):
    """Creates a kernel matrix from mean and scale.
    Args:
        mean: Shape (k1, k2, num_components, dim)
        scale: Shape (k1, k2, num_components, dim, dim) or (k1, k2, num_components, dim)
    Returns:
        Shape (k1, k2, num_components, dim + dim*(dim+1)//2) or (k1, k2, num_components, 2*dim)
    """
    if len(scale.shape) == 4:
        return tf.concat([mean, diag_bijector.inverse(scale)], -1)
    elif len(scale.shape) == 5:
        scale_tril = FillScaleTriL.forward(diag_bijector=diag_bijector)
        return tf.concat([mean, scale_tril.inverse(scale)], -1)
    else:
        raise ValueError(f"Invalid scale shape: {scale.shape}")