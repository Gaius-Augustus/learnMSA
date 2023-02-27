import tensorflow as tf
import numpy as np


background_init_default = np.log([0.08561094, 0.05293611, 0.04151133, 0.05672544, 0.01636436,
       0.03728793, 0.06006028, 0.08408517, 0.0247461 , 0.06263597,
       0.09096471, 0.05326487, 0.02224342, 0.03875183, 0.04393068,
       0.05451456, 0.05665095, 0.01439066, 0.03299227, 0.07033241])

def inverse_softplus(features):
    return np.log(np.expm1(features))

# p: n x s probability distributions over alphabet
def dirichlet_log_pdf(p, alpha, q):
    """ Computes the logarithmic Dirichlet density at p for a mixture given by alpha and q.
    Args:
        p: Probability distributions. Shape: (b, s)
        alpha: Dirichlet component parameters. Shape: (k, s)
        q: Dirichlet mixture parameters. Shape: (k)
    Returns:
        Logarithm of Dirichlet densities. Output shape: (b)
    """
    #logarithms of the normalizing constants for each component
    logZ = tf.math.lbeta(alpha)
    log_p_alpha = tf.math.xlogy(tf.expand_dims(alpha-1, 0), tf.expand_dims(p, 1))
    log_p_alpha = tf.reduce_sum(log_p_alpha, -1) - logZ 
    log_pdf = tf.math.reduce_logsumexp(log_p_alpha + tf.math.log(q), -1)
    return log_pdf
    
class DirichletMixtureLayer(tf.keras.layers.Layer):
    """A dirichlet mixture layer that computes the likelihood of a batch of probability distributions.
    Args:
        num_components: Number of Dirichlet components in the mixture.
        alphabet_size: Size of the discrete categorical probability distributions.
        use_dirichlet_process: If true, when in training mode a Dirichlet Process approximation is used. See 
            Dirichlet Mixtures, the Dirichlet Process, and the Structure of Protein Space, Nguyen et al., 2013
        number_of_examples: The training dataset size which has to be specified if use_dirichlet_process==True.
        alpha_init: Initializer for the component distributions.
        mix_init: Initializer for the mixture distribution.
        trainable: Can be used to freeze the layer when used as a prior.
    """
    def __init__(self, 
                 num_components, 
                 alphabet_size, 
                 use_dirichlet_process=True,
                 number_of_examples=-1,
                 alpha_init = "random_normal",
                 mix_init = "random_normal",
                 background_init = None,
                 trainable=True,
                 **kwargs):
        super(DirichletMixtureLayer, self).__init__(**kwargs)
        self.num_components = num_components
        self.alphabet_size = alphabet_size
        self.use_dirichlet_process = use_dirichlet_process
        self.number_of_examples = number_of_examples
        self.alpha_init = alpha_init
        self.mix_init = mix_init
        self.background_init = background_init
        self.trainable = trainable
        
    def build(self, input_shape=None):
        self.alpha_kernel = self.add_weight(
                                shape = [self.num_components, self.alphabet_size], 
                                initializer = self.alpha_init, 
                                name = 'alpha_kernel', 
                                trainable = self.trainable) 
        self.mix_kernel = self.add_weight(
                                shape = [self.num_components], 
                                initializer = self.mix_init, 
                                name = 'mix_kernel', 
                                trainable = self.trainable)
        if self.use_dirichlet_process:
            self.gamma_kernel = self.add_weight(
                                    shape = [1], 
                                    initializer = tf.constant_initializer([50]), 
                                    name = 'gamma_kernel', 
                                    trainable = self.trainable) 
            self.beta_kernel = self.add_weight(
                                    shape = [1], 
                                    initializer = tf.constant_initializer([100]), 
                                    name = 'beta_kernel', 
                                    trainable = self.trainable) 
            self.lambda_kernel = self.add_weight(
                                    shape = [1], 
                                    initializer = "ones", 
                                    name = 'lambda_kernel', 
                                    trainable = self.trainable) 
            self.background_kernel = self.add_weight(
                                    shape = [20],
                                    initializer = self.background_init, 
                                    name = 'background_kernel', 
                                    trainable = self.trainable)
        
        
    def make_alpha(self):
        return tf.math.softplus(self.alpha_kernel, name="alpha")
    
    def make_mix(self):
        return tf.math.softmax(self.mix_kernel, name="mix")
    
    def make_gamma(self):
        return tf.math.softplus(self.gamma_kernel, name="gamma")
    
    def make_beta(self):
        return tf.math.softplus(self.beta_kernel, name="beta")
    
    def make_lambda(self):
        return tf.math.softplus(self.lambda_kernel, name="lambda")
    
    def make_background(self):
        return tf.nn.softmax(self.background_kernel, name="background")
    
    def log_pdf(self, p):
        return dirichlet_log_pdf(p, self.make_alpha(), self.make_mix())
    
    # out: amino acid distribution of the j's mixture component
    def component_distributions(self):
        alpha = self.make_alpha()
        return alpha / tf.reduce_sum(alpha, axis=-1, keepdims=True)
    
    def expectation(self):
        return tf.reduce_sum(self.component_distributions() * tf.expand_dims(self.make_mix(), -1), 0)
        
    def call(self, p, training=False):
        alpha = self.make_alpha()
        mix = self.make_mix()
        loglik = tf.reduce_mean(dirichlet_log_pdf(p, alpha, mix))
        if training:
            self.add_metric(loglik, name="loglik")
            if self.use_dirichlet_process:
                sum_alpha = tf.reduce_sum(alpha, axis=-1, keepdims=True)
                lamb = self.make_lambda()
                sum_alpha_prior = tf.math.log(lamb) - lamb * sum_alpha #exponential
                sum_alpha_prior = tf.reduce_sum(sum_alpha_prior)
                mix_dist = tf.ones_like(mix, dtype=self.dtype) * self.make_gamma() / self.num_components
                mix_prior = dirichlet_log_pdf(tf.expand_dims(mix, 0), 
                                              tf.expand_dims(mix_dist, 0), 
                                              tf.ones((1), dtype=self.dtype))
                comp_dist = self.make_background() * self.make_beta()
                comp_prior = dirichlet_log_pdf(alpha / sum_alpha, 
                                               tf.expand_dims(comp_dist, 0), 
                                               tf.ones((1), dtype=self.dtype))
                comp_prior = tf.reduce_sum(comp_prior)
                joint_density = loglik + (sum_alpha_prior + mix_prior + comp_prior) / self.number_of_examples
                self.add_loss(-joint_density)
            else:
                self.add_loss(-loglik)
        return loglik
    

def make_model(dirichlet_mixture_layer):
    """Utility function that constructs a keras model over a DirichletMixtureLayer.
    """
    prob_vectors = tf.keras.Input(shape=(dirichlet_mixture_layer.alphabet_size), 
                                  name="prob_vectors", 
                                  dtype=dirichlet_mixture_layer.dtype)
    loglik = dirichlet_mixture_layer(prob_vectors)
    model = tf.keras.Model(inputs=[prob_vectors], outputs=[loglik])
    return model

def load_mixture_model(model_path, num_components, alphabet_size, trainable=False, dtype=tf.float32):
    dm = DirichletMixtureLayer(num_components, alphabet_size, trainable=trainable, dtype=dtype)
    model = make_model(dm)
    model.load_weights(model_path).expect_partial()
    return model
    