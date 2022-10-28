import tensorflow as tf
import numpy as np

"""Ancestral Probability Layer
Learn one or several rate matrices jointly with a downstream model. Amino acid sequences can be smeared towards expected amino acid distributions after
some amount of evolutionary time has passed under a substitution model. This can help to train models on distantly related sequences.
"""

# the default rate matrix ("LG")
paml_lines = ['0.425093 \n', '0.276818 0.751878 \n', '0.395144 0.123954 5.076149 \n', '2.489084 0.534551 0.528768 0.062556 \n', '0.969894 2.807908 1.695752 0.523386 0.084808 \n', '1.038545 0.363970 0.541712 5.243870 0.003499 4.128591 \n', '2.066040 0.390192 1.437645 0.844926 0.569265 0.267959 0.348847 \n', '0.358858 2.426601 4.509238 0.927114 0.640543 4.813505 0.423881 0.311484 \n', '0.149830 0.126991 0.191503 0.010690 0.320627 0.072854 0.044265 0.008705 0.108882 \n', '0.395337 0.301848 0.068427 0.015076 0.594007 0.582457 0.069673 0.044261 0.366317 4.145067 \n', '0.536518 6.326067 2.145078 0.282959 0.013266 3.234294 1.807177 0.296636 0.697264 0.159069 0.137500 \n', '1.124035 0.484133 0.371004 0.025548 0.893680 1.672569 0.173735 0.139538 0.442472 4.273607 6.312358 0.656604 \n', '0.253701 0.052722 0.089525 0.017416 1.105251 0.035855 0.018811 0.089586 0.682139 1.112727 2.592692 0.023918 1.798853 \n', '1.177651 0.332533 0.161787 0.394456 0.075382 0.624294 0.419409 0.196961 0.508851 0.078281 0.249060 0.390322 0.099849 0.094464 \n', '4.727182 0.858151 4.008358 1.240275 2.784478 1.223828 0.611973 1.739990 0.990012 0.064105 0.182287 0.748683 0.346960 0.361819 1.338132 \n', '2.139501 0.578987 2.000679 0.425860 1.143480 1.080136 0.604545 0.129836 0.584262 1.033739 0.302936 1.136863 2.020366 0.165001 0.571468 6.472279 \n', '0.180717 0.593607 0.045376 0.029890 0.670128 0.236199 0.077852 0.268491 0.597054 0.111660 0.619632 0.049906 0.696175 2.457121 0.095131 0.248862 0.140825 \n', '0.218959 0.314440 0.612025 0.135107 1.165532 0.257336 0.120037 0.054679 5.306834 0.232523 0.299648 0.131932 0.481306 7.803902 0.089613 0.400547 0.245841 3.151815 \n', '2.547870 0.170887 0.083688 0.037967 1.959291 0.210332 0.245034 0.076701 0.119013 10.649107 1.702745 0.185202 1.898718 0.654683 0.296501 0.098369 2.188158 0.189510 0.249313 \n', '\n', '0.079066 0.055941 0.041977 0.053052 0.012937 0.040767 0.071586 0.057337 0.022355 0.062157 0.099081 0.064600 0.022951 0.042302 0.044040 0.061197 0.053287 0.012066 0.034155 0.069147 \n', 'A R N D C Q E G H I L K M F P S T W Y V']

def inverse_softplus(features):
    return np.log(np.expm1(features))

def parse_paml(file_content, desired_alphabet):
    """Parses the content of a paml file.
    Returns:
        A symmetric exchangeability matrix with zero diagonal and a frequency vector.
    """
    paml_alphabet = file_content[-1].split(" ")
    s = len(paml_alphabet)
    R = np.zeros((s, s), dtype=np.float32)
    for i in range(1,s):
        R[i,:i] = R[:i,i] = np.fromstring(file_content[i-1], sep=" ") 
    p = np.fromstring(file_content[s], sep=" ")
    #reorganize to match the amino acid order in desired_alphabet
    perm = [paml_alphabet.index(aa) for aa in desired_alphabet if aa in paml_alphabet]
    p = p[perm]
    R = R[perm, :]
    R = R[:, perm]
    return R, p

def make_rate_matrix(exchangeabilities, frequencies, epsilon=1e-16):
    """Constructs a stack of k rate matrices.
    Args:
        exchangeabilities: Symmetric exchangeability matrices with zero diagonals. Shape: (k, 25, 25)
        frequencies: A vector of relative amino acid frequencies. Shape: (25)
    Returns:
        Normalized rate matrices. Output shape: (k, 25, 25)
    """
    Q = exchangeabilities * tf.reshape(frequencies, (1, 1, -1))
    diag = tf.reduce_sum(Q, axis=-1, keepdims=True)
    eye = tf.eye(tf.shape(diag)[1], batch_shape=tf.shape(diag)[:1], dtype=frequencies.dtype)
    Q -= diag * eye
    #normalize, 1 time unit = 1 expected mutation per site
    mue = tf.reshape(frequencies, (1, -1, 1)) * diag
    mue = tf.reduce_sum(mue, axis=-2, keepdims=True)
    Q /= tf.maximum(mue, epsilon)
    return Q
    
def make_anc_probs(sequences, Q, tau):
    """Computes ancestral probabilities simultaneously for all sites and rate matrices.
    Args:
        sequences: Sequences in one hot format. Shape: (b, L)
        Q: A stack of (normalized) rate matrices. Shape: (k, s, s)
        tau: Evolutionary times for all sequences (1 time unit = 1 expected mutation per site). Shape: (b) or (b,k)
    Returns:
        A tensor with the expected amino acids frequencies after time tau. Output shape: (b, L, k, s)
    """
    while len(tau.shape) < 4:
        tau = tf.expand_dims(tau, -1)
    tauQ = tau * tf.expand_dims(Q, 0)
    P = tf.linalg.expm(tauQ) # P[s,r,i,j] = P(X(tau_s) = j | X(0) = i; Q_r))
    dummy_alphabet = tf.eye(tf.shape(Q)[-1])
    dummy_ancprobs = tf.einsum('zs,bksr->bzkr', dummy_alphabet, P)
    b = tf.shape(dummy_ancprobs)[0]
    s = tf.shape(dummy_ancprobs)[1]
    k = tf.shape(dummy_ancprobs)[2]
    r = tf.shape(dummy_ancprobs)[3]
    dummy_ancprobs = tf.reshape(dummy_ancprobs, (-1, k, r))
    sequences = tf.cast(sequences, tf.int32)
    sequences += tf.expand_dims( tf.range(b) * s, -1)
    ancprobs = tf.nn.embedding_lookup(dummy_ancprobs, sequences)
    return ancprobs
    
class AncProbsLayer(tf.keras.layers.Layer): 
    """A learnable layer for ancestral probabilities.

    Args:
        num_rates: The number of different evolutionary times.
        num_matrices: The number of rate matrices.
        frequencies: Initializer for the exchangeability matrices (default: amino acid background distribution).
        rate_init: Initializer for the rates.
        exchangeability_init: Initializer for the exchangeability matrices. Usually inverse_softplus should be used on the initial matrix by the user.
        trainable_exchangeabilities: Flag that can prevent learning the exchangeabilities.
        per_matrix_rate: Learns an additional evolutionary time per rate matrix.
        shared_matrix: Make all weight matrices internally use the same weights. Only useful in combination with num_matrices > 1 and per_matrix_rate = True. 
        padding: Size of the zero padding expected at the end of the last input dimension.
        name: Layer name.
    """

    def __init__(self, 
                 num_rates, 
                 num_matrices,
                 frequencies,
                 exchangeability_init,
                 rate_init=tf.constant_initializer(-3.),
                 trainable_exchangeabilities=True,
                 per_matrix_rate=False,
                 matrix_rate_init=None,
                 matrix_rate_l2=0.0,
                 shared_matrix=False,
                 name="AncProbsLayer",
                 **kwargs):
        super(AncProbsLayer, self).__init__(name=name, **kwargs)
        self.num_rates = num_rates
        self.num_matrices = num_matrices
        self.rate_init = rate_init
        self.frequencies = tf.cast(frequencies, dtype=self.dtype)
        self.exchangeability_init = exchangeability_init
        self.trainable_exchangeabilities = trainable_exchangeabilities
        self.per_matrix_rate = per_matrix_rate
        self.matrix_rate_init = matrix_rate_init
        self.matrix_rate_l2 = matrix_rate_l2 
        self.shared_matrix = shared_matrix
    
    def build(self, input_shape=None):
        self.tau_kernel = self.add_weight(shape=[self.num_rates], 
                                   name="tau_kernel", 
                                   initializer=self.rate_init)
        if self.shared_matrix:
            self.exchangeability_kernel = self.add_weight(shape=[1,20, 20],
                                                          name="exchangeability_kernel",
                                                           initializer=self.exchangeability_init,
                                                           trainable=self.trainable_exchangeabilities)
        else:
            self.exchangeability_kernel = self.add_weight(shape=[self.num_matrices, 20, 20], 
                                                           name="exchangeability_kernel", 
                                                           initializer=self.exchangeability_init,
                                                           trainable=self.trainable_exchangeabilities)
        if self.per_matrix_rate:
            self.per_matrix_rates_kernel = self.add_weight(shape=[self.num_matrices], 
                                                      name="per_matrix_rates_kernel",
                                                      initializer=self.matrix_rate_init)
       
    def make_R(self, kernel=None):
        if kernel is None:
            kernel = self.exchangeability_kernel
        R = 0.5 * (kernel + tf.transpose(kernel, [0,2,1])) #make symmetric
        R = tf.math.softplus(R)
        R -= tf.linalg.diag(tf.linalg.diag_part(R)) #zero diagonal
        return R
    
    def make_Q(self):
        return make_rate_matrix(self.make_R(), self.frequencies)
        
    def make_tau(self, subset=None):
        tau = self.tau_kernel
        if subset is not None:
            tau = tf.gather(tau, subset)
        return tf.math.softplus(tau)
    
    def make_per_matrix_rate(self):
        return tf.math.softplus(self.per_matrix_rates_kernel)

    def call(self, inputs, rate_indices):
        """ Computes anchestral probabilities of the inputs.
        Args:
            inputs: Input sequences. Shape: (b, L)
            rate_indices: Indices that map each input sequences to an evolutionary times.
        """
        mask = inputs < 20
        only_std_aa_inputs = inputs * tf.cast(mask, inputs.dtype)
        mask = tf.cast(mask, self.dtype)
        mask = tf.expand_dims(mask, -1)
        mask = tf.expand_dims(mask, -1)
        tau_subset = self.make_tau(rate_indices)
        if self.per_matrix_rate:
            per_matrix_rates = self.make_per_matrix_rate()
            per_matrix_rates = tf.expand_dims(per_matrix_rates, 0)
            tau_subset = tf.expand_dims(tau_subset, 1)
            mut_rates = tau_subset * per_matrix_rates
            reg = tf.reduce_mean(tf.square(self.per_matrix_rates_kernel - inverse_softplus(1.)))
            self.add_loss(self.matrix_rate_l2 * reg)
        else:
            mut_rates = tau_subset
        Q = self.make_Q()
        anc_probs = make_anc_probs(only_std_aa_inputs, Q, mut_rates)
        anc_probs *= mask
        anc_probs = tf.pad(anc_probs, [[0,0], [0,0], [0,0], [0,6]])
        rest = tf.expand_dims(tf.one_hot(inputs, 26), -2) * (1-mask)
        anc_probs += rest
        anc_probs = tf.reshape(anc_probs, (tf.shape(inputs)[0], tf.shape(inputs)[1],-1) )
        return anc_probs