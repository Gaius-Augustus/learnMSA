import sys
sys.path.insert(0, "..")
import numpy as np
import tensorflow as tf

from learnMSA.msa_hmm.MsaHmmLayer import MsaHmmLayer
from learnMSA.msa_hmm.MsaHmmCell import HmmCell
from learnMSA.msa_hmm.Initializers import ConstantInitializer
# from learnMSA.msa_hmm import Align, Emitter, Transitioner, Initializers, MsaHmmCell, MsaHmmLayer, Training, Configuration, Viterbi, AncProbsLayer, Priors, DirichletMixture
# from learnMSA.msa_hmm.SequenceDataset import SequenceDataset, AlignedDataset
# from learnMSA.msa_hmm.AlignmentModel import AlignmentModel


# Generate simulated data from the occasionally dishonest casino.
# The HMM has 
#  * state space {0,1} = {F,L}
#  * emission alphabet {1,..., 6}
#  * transition distribution given by matrix A
#  * emission distribution given by matrix B
#  * always starts in state F
 
#  Author: Mario Stanke
#          Felix Becker (generator now outputs x and y) 

import tensorflow as tf
import numpy as np

sigma = [0,1,2,3,4,5]
s = len(sigma)  # emission alphabet 0,..., 5
A = np.array([[19., 1], [3, 17]]) / 20.0
B = np.array([[10.,10,10,10,10,10],[6,6,6,6,6,30]]) / 60.0
n = 2 # number of states
T = 100 # sequence length
batch = 2

def casino_generator():
    x = np.zeros((batch, T, n), dtype=np.float32)
    y = np.zeros((batch, T, s), dtype=np.float32)
    q = 0
    x[:, 0, q] = 1
    for j in range(batch):
        for i in range(T):
            if i > 0:
                q = np.random.choice(range(n), p=A[q])
                x[j, i, q] = 1
            c = np.random.choice(sigma, p=B[q])
            y[j, i,c] = 1    
    yield y,x

def get_casino_dataset():
    dataset = tf.data.Dataset.from_generator(
         casino_generator,
         output_signature=(tf.TensorSpec(shape=(batch, T, s), dtype=tf.float32), 
                           tf.TensorSpec(shape=(batch, T, n), dtype=tf.float32)))
    return dataset.repeat()
    

class CasinoHMMEmitter(tf.keras.layers.Layer):
    def __init__(self, num_states, init="glorot_uniform", **kwargs):
        super(CasinoHMMEmitter, self).__init__(**kwargs)
        self.num_states = num_states
        self.init = init
        
    def build(self, input_shape):
        if self.built:
            return
        self.alphabet_size = input_shape[-1] 
        self.emission_kernel = self.add_weight(
                                        shape=[1, self.num_states, self.alphabet_size], #just 1 HMM 
                                        initializer=self.init,
                                        name="emission_kernel")
        self.built = True
        
    def recurrent_init(self):
        """ Automatically called before each recurrent run. Should be used for setups that
            are only required once per application of the recurrent layer.
        """
        self.B = self.make_B()

    def make_B(self):
        return tf.nn.softmax(self.emission_kernel, name="B")
        
    def call(self, inputs, end_hints=None, training=False):
        """ 
        Args: 
                inputs: A tensor of shape (num_models, batch_size, length, alphabet_size) 
        Returns:
                A tensor with emission probabilities of shape (num_models, batch_size, length, num_states).
        """
        return tf.einsum("k...s,kqs->k...q", inputs, self.B)

    def duplicate(self, model_indices=None, share_kernels=False):
        init = ConstantInitializer(self.emission_kernel.numpy())
        emitter_copy = CasinoHMMEmitter(num_states=self.num_states, init=init) 
        if share_kernels:
            emitter_copy.emission_kernel = self.emission_kernel
            emitter_copy.built = True
        return emitter_copy
        
    def get_config(self):
        return {"num_states" : self.num_states,
                "init": self.init}



class CasinoHMMTransitioner(tf.keras.layers.Layer):
    """ Defines which transitions between HMM states are allowed and how they are initialized.
    """
    def __init__(self, num_states, init="glorot_uniform", start_init="glorot_uniform", **kwargs):
        super(CasinoHMMTransitioner, self).__init__(**kwargs)
        self.num_states = num_states
        self.init = init
        self.start_init = start_init
        self.reverse = False
        

    def build(self, input_shape=None):
        if self.built:
            return
        self.transition_kernel = self.add_weight(shape=[1, self.num_states, self.num_states],  #just 1 HMM
                                                 initializer=self.init,
                                                 name="transition_kernel")
        self.starting_distribution_kernel = self.add_weight(shape=[1,1,self.num_states], 
                                                            initializer=self.start_init,
                                                           name="starting_distribution_kernel")
        self.built = True
        

    def recurrent_init(self):
        """ Automatically called before each recurrent run. Should be used for setups that
            are only required once per application of the recurrent layer.
        """
        self.A = self.make_A()
        self.A_transposed = tf.transpose(self.A, (0,2,1))

    
    def make_A(self):
        return tf.nn.softmax(self.transition_kernel)


    def make_log_A(self):
        return tf.math.log(self.make_A())


    def make_initial_distribution(self):
        return tf.nn.softmax(self.starting_distribution_kernel)
        
        
    def call(self, inputs):
        """ 
        Args: 
                inputs: Shape (k, b, q)
        Returns:
                Shape (k, b, q)
        """
        #batch matmul of k inputs with k matricies
        if self.reverse:
            return tf.matmul(inputs, self.A_transposed)
        else:
            return tf.matmul(inputs, self.A)
    
    def get_prior_log_densities(self):
         # Can be used for regularization in the future.
        return 0.
    
    def duplicate(self, model_indices=None, share_kernels=False):
        init = ConstantInitializer(self.transition_kernel.numpy())
        starting_distribution_init = ConstantInitializer(self.starting_distribution_kernel.numpy())
        transitioner_copy = CasinoHMMTransitioner(num_states=self.num_states, init=init,
                                                  start_init=starting_distribution_init) 
        if share_kernels:
            transitioner_copy.transition_kernel = self.transition_kernel
            transitioner_copy.starting_distribution_kernel = self.starting_distribution_kernel
            transitioner_copy.built = True
        return transitioner_copy
        
    def get_config(self):
        return {"num_states" : self.num_states,
                "init": self.init,
                "start_init": self.start_init}


class CasinoHMMLayer(MsaHmmLayer):

    def __init__(self, parallel_factor, **kwargs):
        super(CasinoHMMLayer, self).__init__(cell=None, use_prior=False, num_seqs=1e6, parallel_factor=parallel_factor, **kwargs)

    def build(self, input_shape):
        if self.built:
            return
        if False:
            emitter = CasinoHMMEmitter(n, init=ConstantInitializer(np.log(B[np.newaxis])))
            transitioner = CasinoHMMTransitioner(n, init=ConstantInitializer(np.log(A[np.newaxis])), 
                                                start_init=ConstantInitializer(np.log(np.array([1-1e-5,1e-5]))[np.newaxis, np.newaxis]))
        if True:
            emitter = CasinoHMMEmitter(n)
            transitioner = CasinoHMMTransitioner(n)
        self.cell = HmmCell(num_states=[n],
                            dim=input_shape[-1],
                            emitter=emitter, 
                            transitioner=transitioner,
                            name="gene_pred_hmm_cell")
        super(CasinoHMMLayer, self).build(input_shape)

    def call(self, inputs, use_loglik=True, training=False):
        """ 
        Computes the state posterior log-probabilities.f
        Args: 
                inputs: Shape (batch, len, alphabet_size)
        Returns:
                State posterior log-probabilities (without loglik if use_loglik is False). Shape (batch, len, number_of_states)
        """ 
        log_post = self.state_posterior_log_probs(tf.expand_dims(inputs, 0), training=training, no_loglik=not use_loglik)[0]
        return log_post

tf.keras.utils.get_custom_objects()["CasinoHMMEmitter"] = CasinoHMMEmitter
tf.keras.utils.get_custom_objects()["CasinoHMMTransitioner"] = CasinoHMMTransitioner
tf.keras.utils.get_custom_objects()["CasinoHMMLayer"] = CasinoHMMLayer

# Supervised HMM training
def make_compiled_model(parallel_factor=1):
    input = tf.keras.layers.Input(shape=(None, s), name="input")
    log_post = CasinoHMMLayer(parallel_factor)(input)
    #post = tf.keras.layers.Softmax()(log_post)
    model = tf.keras.Model(inputs=input, outputs=log_post)
    cce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(loss=cce_loss, optimizer=optimizer, metrics=['accuracy'])
    return model


def test_training():
    model = make_compiled_model()
    data = get_casino_dataset() #generative dataset, train and test data in one

    # test the theoretical supervised estimates
    sums = np.zeros((2))
    rolls = np.zeros((2, 6))
    for i,(y,x) in enumerate(data):
        if i >= 100:
            break
        sums += np.sum(np.sum(x, axis=0), axis=0)
        per_state = y[...,np.newaxis,:] * x[...,:,np.newaxis]
        rolls += np.sum(np.sum(per_state, axis=0), axis=0)
    print(sums / (100 * batch * T))
    print(rolls / np.sum(rolls, axis=1, keepdims=True))
        
    model.fit(data, epochs=5, steps_per_epoch=100)

    print("A", model.layers[-1].cell.transitioner.make_A())
    print("B", model.layers[-1].cell.emitter[0].make_B())
    print("init", model.layers[-1].cell.transitioner.make_initial_distribution())


def get_prediction(parallel_factor=1):
    np.random.seed(2020) # same order of random batches all the time
    tf.keras.utils.set_random_seed(2020)
    model = make_compiled_model(parallel_factor)
    data = get_casino_dataset()
    for x,_ in data:
        break
    return model(x)








