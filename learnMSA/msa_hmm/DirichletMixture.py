import tensorflow as tf

dtype = tf.float64


background_distribution = tf.constant([0.08561094, 0.05293611, 0.04151133, 0.05672544, 0.01636436,
       0.03728793, 0.06006028, 0.08408517, 0.0247461 , 0.06263597,
       0.09096471, 0.05326487, 0.02224342, 0.03875183, 0.04393068,
       0.05451456, 0.05665095, 0.01439066, 0.03299227, 0.07033241], dtype)


class DirichletMixturePrior(tf.keras.layers.Layer):
    def __init__(self, k, s, n,
                 alpha_init = "random_normal",
                 q_init = "random_normal",
                 trainable=True):
        super(DirichletMixturePrior, self).__init__(dtype=dtype)
        if not isinstance(alpha_init, str):
            alpha_init = tf.constant_initializer(alpha_init)
        if not isinstance(q_init, str):
            q_init = tf.constant_initializer(q_init)
        self.k = k
        self.s = s
        self.n = n
        self.alpha_kernel = self.add_weight(
            shape=[k, s], 
            initializer=alpha_init, 
            name='alpha_kernel', 
            dtype=dtype, 
            trainable=trainable) 
        self.q_kernel = self.add_weight(
            shape=[k], 
            initializer=q_init, 
            name='q_kernel', 
            dtype=dtype, 
            trainable=trainable)
        self.gamma_kernel = self.add_weight(
            shape=[1], 
            initializer=tf.constant_initializer([50]), 
            name='gamma_kernel', 
            dtype=dtype, 
            trainable=trainable) 
        self.beta_kernel = self.add_weight(
            shape=[1], 
            initializer=tf.constant_initializer([100]), 
            name='beta_kernel', 
            dtype=dtype, 
            trainable=trainable) 
        self.lambda_kernel = self.add_weight(
            shape=[1], 
            initializer="ones", 
            name='lambda_kernel', 
            dtype=dtype, 
            trainable=trainable) 
        
        
    def make_alpha(self):
        return tf.math.softplus(self.alpha_kernel, name="alpha")
    
    def make_q(self):
        return tf.math.softmax(self.q_kernel, name="q")
    
    def make_gamma(self):
        return tf.math.softplus(self.gamma_kernel, name="gamma")
    
    def make_beta(self):
        return tf.math.softplus(self.beta_kernel, name="beta")
    
    def make_lambda(self):
        return tf.math.softplus(self.lambda_kernel, name="lambd")
    
    # p: n x s probability distributions over alphabet
    def log_pdf(self, p, alpha=None, q=None):
        if alpha is None:
            alpha = self.make_alpha()
        if q is None:
            q = self.make_q()
        #logarithms of the normalizing constants for each component
        logZ = tf.math.lbeta(alpha)
        log_p_alpha = tf.math.xlogy(tf.expand_dims(alpha-1, 0), tf.expand_dims(p, 1))
        log_p_alpha = tf.reduce_sum(log_p_alpha, -1) - logZ #n x k
        return tf.math.reduce_logsumexp(log_p_alpha + tf.math.log(q), -1)
    
    
    # out: amino acid distribution of the j's mixture component
    def component_distributions(self):
        alpha = self.make_alpha()
        return alpha / tf.reduce_sum(alpha, axis=-1, keepdims=True)
    
    
    def expectation(self):
        return tf.reduce_sum(self.component_distributions() * tf.expand_dims(self.make_q(), -1), 0)
    
    
    def log_prior(self, alpha, q):
        return 
        
    
    def call(self, x, training=False):
        alpha = self.make_alpha()
        q = self.make_q()
        loglik = tf.reduce_mean(self.log_pdf(x, alpha, q))
        if training:
            sum_alpha = tf.reduce_sum(alpha, axis=-1, keepdims=True)
            lamb = self.make_lambda()
            sum_alpha_prior = tf.math.log(lamb) - lamb * sum_alpha #exponential
            sum_alpha_prior = tf.reduce_sum(sum_alpha_prior)
            mix_dist = tf.ones_like(q, dtype=dtype) * self.make_gamma() / self.k
            mix_prior = self.log_pdf(tf.expand_dims(q, 0), tf.expand_dims(mix_dist, 0), tf.ones((1), dtype=dtype))
            comp_dist = background_distribution * self.make_beta()
            comp_prior = self.log_pdf(alpha / sum_alpha, tf.expand_dims(comp_dist, 0), tf.ones((1), dtype=dtype))
            comp_prior = tf.reduce_sum(comp_prior)
            joint_density = loglik + (sum_alpha_prior + mix_prior + comp_prior) / self.n
            self.add_loss(-joint_density)
            self.add_metric(loglik, name="loglik")
        return loglik
    


def make_model(k, s, n, trainable):
    DMP = DirichletMixturePrior(k, s, n, trainable=trainable)
    prob_vectors = tf.keras.Input(shape=(s), name="prob_vectors", dtype=dtype)
    loglik = DMP(prob_vectors)
    model = tf.keras.Model(inputs=[prob_vectors], outputs=[loglik])
    return model, DMP