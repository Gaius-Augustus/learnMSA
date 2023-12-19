import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import math




class DefaultDiagBijector(tfp.bijectors.Softplus):
    def __init__(self, base_variance):
        """ Args:
                base_variance: The initial variance (diagonal entries of the covariance matrix) if kernel = 0.
        """
        super(DefaultDiagBijector, self).__init__()
        base_std = np.sqrt(base_variance).astype(np.float32)
        self.scale_diag_init = tfp.math.softplus_inverse(base_std)

    def forward(self, x, name='forward', **kwargs):
        return super(DefaultDiagBijector, self).forward(x + self.scale_diag_init, name, **kwargs)

    def inverse(self, y, name='inverse', **kwargs):
        return super(DefaultDiagBijector, self).inverse(y) - self.scale_diag_init



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
        scale_tril = tfp.bijectors.FillScaleTriL(diag_bijector=diag_bijector)
        return tf.concat([mean, scale_tril.inverse(scale)], -1)
    else:
        raise ValueError(f"Invalid scale shape: {scale.shape}")


class MvnMixture():
    def __init__(self, 
                dim,
                kernel,
                mixture_coeff_kernel = None,
                diag_only = True,
                diag_bijector = DefaultDiagBijector(1.),
                precomputed = False,
                **kwargs):
        """A multivariate normal distribution over R^d parameterized by d locations 
            and a dxd lower triangular scale matrix.
            Args:
                dim: The dimensionality of the multivariate normal.
                kernel: A 4D kernel matrix closely related to the model parameters. 
                        Expected to have shape (k1, k2, num_components, num_param).
                        Where k1 is the number of models that have a matching input sequence,
                        k2 is the number of models that will be evaluated against all inputs
                        and num_param = 2*dim if diag_only=True and num_param = dim + dim*(dim+1)//2 otherwise.
                mixture_coeff_kernel: A 3D kernel matrix closely related to the mixture coefficients. Shape (k1, k2, num_components). 
                                    If None, a single component is assumed. And kernel.shape[2] must be 1.
                diag_only: If True, the scale matrix is assumed to be diagonal. Otherwise, a full scale matrix is used.
                diag_bijector: A bijector to project the diagonal entries of the scale matrix to positive values.
                precomputed: If True, the scale matrix is not recomputed in each call of the layer. 
                            Should be False during training and True during inference.
        """
        self.dim = dim
        self.kernel = kernel
        self.mixture_coeff_kernel = mixture_coeff_kernel
        self.num_components = kernel.shape[2]
        self.diag_only = diag_only
        self.diag_bijector = diag_bijector
        self.precomputed = precomputed
        self.scale_tril = tfp.bijectors.FillScaleTriL(diag_bijector=diag_bijector)
        self.constant = self.dim * tf.math.log(2*math.pi)
        self.scale = None
        self.pinv = None
        # validate
        assert len(kernel.shape) == 4
        if diag_only:
            assert kernel.shape[-1] == 2*dim
        else:
            assert kernel.shape[-1] == dim + dim*(dim+1)//2
        if self.mixture_coeff_kernel is not None:
            assert len(mixture_coeff_kernel.shape) == 3
            assert mixture_coeff_kernel.shape == kernel.shape[:3]
        else:
            assert self.num_components == 1


    def component_expectations(self):
        """Computes the expected values of the mixture components.
        Returns:
            Shape (k1, k2, num_components, dim)
        """
        mu = self.kernel[..., :self.dim]
        return mu


    def expectation(self):
        """
        Returns:
            Shape (k1, k2, dim)
        """
        if self.num_components == 1:
            return self.component_expectations()[...,0,:]
        else:
            comp_exp = self.component_expectations()
            mix_coeff = self.mixture_coefficients()
            return tf.reduce_sum(comp_exp * mix_coeff[..., tf.newaxis], -2)


    
    def component_scales(self, return_scale_diag=False, return_inverse=False):
        """Computes the scale matrices of the mixture components. The covariance matrices can be computed as scale * scale^T.
        """
        if not self.precomputed or self.scale is None:
            if self.diag_only:
                scale_diag = self.diag_bijector(self.kernel[..., self.dim:])
                scale_diag += 1e-8
                scale = scale_diag if return_scale_diag else tf.eye(self.dim, batch_shape=tf.shape(scale_diag)[:-1]) * scale_diag[..., tf.newaxis]
                if return_inverse:
                    pinv = 1. / scale_diag
            else:
                scale_kernel = self.kernel[..., self.dim:]
                scale = self.scale_tril(scale_kernel)
                if return_inverse:
                    pinv = tf.linalg.pinv(scale)
                if return_scale_diag:
                    scale = tf.linalg.diag_part(scale)
        return (scale, pinv) if return_inverse else scale


    def component_covariances(self):
        """Computes the covariance matrices of the mixture components.
        Returns:
            Shape (k1, k2, num_components, dim, dim)
        """
        scale = self.component_scales()
        return tf.matmul(scale, scale, transpose_b=True)


    def component_log_pdf(self, inputs):
        """
        Computes the conmponent-wise log probability density function for each mixture distribution. 
        This method performs k1 many all-to-all evaluations between batch many inputs and k2 many models.
        Args:
            inputs: Shape (k1, batch, dim)
        Returns:
            Shape (k1, batch, k2, num_components)
        """
        mu = self.component_expectations()
        scale_diag, pinv = self.component_scales(return_scale_diag=True, return_inverse=True)
        log_det = 2*tf.reduce_sum(tf.math.log(scale_diag), -1) # (k1, k2, c, 1)
        diff = inputs[:,tf.newaxis,tf.newaxis] - mu[...,tf.newaxis,:] # (k1, k2, c, b, d)
        if self.diag_only:
            pinv_sq = tf.math.square(pinv) # (k1, k2, c, d)
            diff_sq = tf.math.square(diff)
            MD_sq_components = tf.linalg.matvec(diff_sq, pinv_sq) # (k1, k2, c, b)
        else:
            y = tf.matmul(diff, pinv, transpose_b=True) # (k1, k2, c, b, d)
            MD_sq_components = tf.reduce_sum(tf.math.square(y), -1) 
        MD_sq_components = tf.transpose(MD_sq_components, [0,3,1,2])
        log_pdf = -0.5 * (self.constant + log_det[:, tf.newaxis] + MD_sq_components)
        return log_pdf

    
    def mixture_coefficients(self):
        """Computes the mixture coefficients.
        Returns:
            Shape (k1, k2, num_components)
        """
        return tf.nn.softmax(self.mixture_coeff_kernel, axis=-1)


    def log_pdf(self, inputs):
        """
        Computes the log probability density function for each mixture distribution. 
        This method performs k1 many all-to-all evaluations between batch many inputs and k2 many models.
        Args:
            inputs: Shape (k1, batch, dim)
        Returns:
            Shape (k1, batch, k2)
        """
        log_pdf_components = self.component_log_pdf(inputs)
        if self.num_components == 1:
            return log_pdf_components[...,0]
        else:
            return tf.math.reduce_logsumexp(log_pdf_components + tf.math.log(self.mixture_coefficients()[:,tf.newaxis]), -1)

    def get_regularization_L2_loss(self):
        """Computes the L2 loss of the variance kernels, preventing too small and too large variances.
        """
        return tf.reduce_mean(tf.reduce_sum(tf.math.square(self.kernel[..., self.dim:]), axis=-1))