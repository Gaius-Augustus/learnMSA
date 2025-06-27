import tensorflow as tf
from learnMSA.msa_hmm.MsaHmmLayer import MsaHmmLayer
from learnMSA.msa_hmm.MsaHmmCell import HmmCell
from learnMSA.msa_hmm.MaximumExpectedAccuracy import maximum_expected_accuracy
from learnMSA.msa_hmm.Viterbi import viterbi


class TestMeaEmitter(tf.keras.layers.Layer):

    def __init__(self, num_states, **kwargs):
        super(TestMeaEmitter, self).__init__(**kwargs)
        self.num_states = num_states
        
    def recurrent_init(self):
        self.B = self.make_B()

    def make_B(self):
        return tf.constant(
            [[1., 0.]] * (self.num_states -1) + [[0., 1.]]
        )[tf.newaxis]
        
    def call(self, inputs, end_hints=None, training=False):
        return tf.einsum("k...s,kqs->k...q", inputs, self.B)

    def duplicate(self, model_indices=None, share_kernels=False):
        emitter_copy = TestMeaEmitter(num_states=self.num_states) 
        return emitter_copy
        
    def get_config(self):
        return {"num_states" : self.num_states}


class TestMeaTransitioner(tf.keras.layers.Layer):
    """ 
    Defines this HMM:
    S -> I1 -> I2 -> ... -> Ik -> T
    |^   |^    |^           |^    |^
    q    p     p            p     1
    """
    def __init__(self, k, p, q, **kwargs):
        super(TestMeaTransitioner, self).__init__(**kwargs)
        self.k = k
        self.p = p # loop probability of inner states
        self.q = q # loop probability of starting state
        self.reverse = False
        
    def recurrent_init(self):
        self.A = self.make_A()
        self.A_transposed = tf.transpose(self.A, (0,2,1))

    def make_A(self):
        p,q = self.p, self.q
        return tf.constant(
            [[q, 1-q] + [0]*self.k]
            + [[0] * (i+1) + [p] + [1-p] + [0] * (self.k - i - 1)
               for i in range(self.k)]
            + [[0]*(self.k+1) + [1]]
        )[tf.newaxis]

    def make_log_A(self):
        return tf.math.log(self.make_A() + 1e-16)

    def make_initial_distribution(self):
        return tf.constant([1.] + [0]*(self.k+1))[tf.newaxis, tf.newaxis]
        
    def call(self, inputs):
        if self.reverse:
            return tf.matmul(inputs, self.A_transposed)
        else:
            return tf.matmul(inputs, self.A)
    
    def get_prior_log_densities(self):
        return 0.
    
    def duplicate(self, model_indices=None, share_kernels=False):
        transitioner_copy = TestMeaTransitioner(
            k=self.k, p=self.p, q=self.q
        )
        return transitioner_copy
        
    def get_config(self):
        return {
            "k" : self.k,
            "p" : self.p,
            "q" : self.q
        }
    

class TestMeaHMMLayer(MsaHmmLayer):

    def __init__(
            self, 
            num_states,
            p,
            q,
            parallel_factor=1, 
            **kwargs
    ):
        super(TestMeaHMMLayer, self).__init__(
            cell=None, 
            use_prior=False,
            num_seqs=1e6, 
            parallel_factor=parallel_factor, 
            **kwargs
        )
        self.num_states = num_states
        self.p = p
        self.q = q

    def build(self, input_shape):
        if self.built:
            return
        emitter = TestMeaEmitter(num_states=self.num_states)
        transitioner = TestMeaTransitioner(
            k=self.num_states - 2, 
            p=self.p, 
            q=self.q
        )
        self.cell = HmmCell(
            num_states=[self.num_states],
            dim=input_shape[-1],
            emitter=emitter, 
            transitioner=transitioner,
            name="test_mea_hmm_cell"
        )
        super(TestMeaHMMLayer, self).build(input_shape)

    def mea(self, inputs, use_loglik=True, training=False):
        log_post = self.state_posterior_log_probs(
            inputs, 
            training=training, 
            no_loglik=not use_loglik
        )
        post = tf.nn.softmax(log_post, axis=-1)
        mea = maximum_expected_accuracy(
            post, 
            self.cell, 
            parallel_factor=self.parallel_factor
        )[0]
        return mea
    
    def viterbi(self, inputs):
        return viterbi(
            inputs, 
            self.cell, 
            parallel_factor=self.parallel_factor
        )[0]
    
    def call(self, inputs, use_loglik=True, training=False):
        return self.mea(inputs, use_loglik=use_loglik, training=training)