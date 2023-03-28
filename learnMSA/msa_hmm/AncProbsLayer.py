import tensorflow as tf
import numpy as np
import learnMSA.msa_hmm.Initializers as initializers

"""Ancestral Probability Layer
Learn one or several rate matrices jointly with a downstream model. Amino acid sequences can be smeared towards expected amino acid distributions after
some amount of evolutionary time has passed under a substitution model. This can help to train models on distantly related sequences.
"""
              
def inverse_softplus(features):
    #cast to float 64 to prevent overflow of large entries
    features64 = features.astype(np.float64)
    return np.log(np.expm1(features64)).astype(features.dtype)

def parse_paml(lines, desired_alphabet):
    """Parses the content of a paml file.
    Returns:
        A symmetric exchangeability matrix with zero diagonal and a frequency vector.
    """
    paml_alphabet = "A R N D C Q E G H I L K M F P S T W Y V".split(" ")
    s = len(paml_alphabet)
    R = np.zeros((s, s), dtype=np.float32)
    for i in range(1,s):
        R[i,:i] = R[:i,i] = np.fromstring(lines[i-1], sep=" ") 
    p = np.fromstring(lines[s-1], sep=" ", dtype=np.float32)
    #reorganize to match the amino acid order in desired_alphabet
    perm = [paml_alphabet.index(aa) for aa in desired_alphabet if aa in paml_alphabet]
    p = p[perm]
    R = R[perm, :]
    R = R[:, perm]
    return R, p

def make_rate_matrix(exchangeabilities, equilibrium, epsilon=1e-16):
    """Constructs a stack of k rate matrices.
    Args:
        exchangeabilities: Symmetric exchangeability matrices with zero diagonals. Shape: (k, 25, 25)
        equilibrium: A vector of relative amino acid frequencies. Shape: (k, 25)
    Returns:
        Normalized rate matrices. Output shape: (k, 25, 25)
    """
    if len(exchangeabilities.shape) == 2:
        exchangeabilities = tf.expand_dims(exchangeabilities, 0)
    if len(equilibrium.shape) == 1:
        equilibrium = tf.expand_dims(equilibrium, 0)
    Q = exchangeabilities *  tf.expand_dims(equilibrium, 1)
    diag = tf.reduce_sum(Q, axis=-1, keepdims=True)
    eye = tf.eye(tf.shape(diag)[1], batch_shape=tf.shape(diag)[:1], dtype=equilibrium.dtype)
    Q -= diag * eye
    #normalize, 1 time unit = 1 expected mutation per site
    mue = tf.expand_dims(equilibrium, -1) * diag
    mue = tf.reduce_sum(mue, axis=-2, keepdims=True)
    Q /= tf.maximum(mue, epsilon)
    return Q
    
def make_anc_probs(sequences, exchangeabilities, equilibrium, tau, equilibrium_sample=False, transposed=False):
    """Computes ancestral probabilities simultaneously for all sites and rate matrices.
    Args:
        sequences: Sequences either as integers (faster embedding lookup) or in vector format. Shape: (num_model, b, L) or (num_model, b, L, s)
        exchangeabilities: A stack of symmetric exchangeability matrices. Shape: (num_model, k, s, s)
        equilibrium: A stack of equilibrium distributions. Shape: (num_model, k, s)
        tau: Evolutionary times for all sequences (1 time unit = 1 expected mutation per site). Shape: (num_model, b) or (num_model,b,k)
        equi_init: If true, a 2-staged process is assumed where an amino acid is first sampled from the equilibirium distribution
                    and the ancestral probabilities are computed afterwards.
    Returns:
        A tensor with the expected amino acids frequencies after time tau. Output shape: (num_model, b, L, k, s)
    """
    while len(tau.shape) < 5:
        tau = tf.expand_dims(tau, -1)
    shape = tf.shape(exchangeabilities)
    exchangeabilities = tf.reshape(exchangeabilities, (-1, 20, 20))
    equilibrium = tf.reshape(equilibrium, (-1, 20))
    Q = make_rate_matrix(exchangeabilities, equilibrium)
    Q = tf.reshape(Q, shape)
    tauQ = tau * tf.expand_dims(Q, 1)
    P = tf.linalg.expm(tauQ) # P[m,b,k,i,j] = P(X(tau_b) = j | X(0) = i; Q_k, model m))
    num_model,b,k,s,s = tf.unstack(tf.shape(P), 5)
    if equilibrium_sample:
        equilibrium = tf.reshape(equilibrium, (num_model,1,k,s,1))
        P *= equilibrium
    if len(sequences.shape) == 3: #assume index format
        if transposed:
            P = tf.transpose(P, [0,1,4,2,3])
        else:
            P = tf.transpose(P, [0,1,3,2,4])
        P = tf.reshape(P, (-1, k, s))
        sequences = tf.cast(sequences, tf.int32)
        sequences = tf.reshape(sequences, (-1, tf.shape(sequences)[-1]))
        sequences += tf.expand_dims( tf.range(num_model*b) * s, -1)
        ancprobs = tf.nn.embedding_lookup(P, sequences)
        ancprobs = tf.reshape(ancprobs, (num_model, b, -1, k, s))
    else: #assume vector format
        if transposed:
            ancprobs = tf.einsum("mbLz,mbksz->mbLks", sequences, P)
        else:
            ancprobs = tf.einsum("mbLz,mbkzs->mbLks", sequences, P)
    return ancprobs
    
class AncProbsLayer(tf.keras.layers.Layer): 
    """A learnable layer for ancestral probabilities.

    Args:
        num_models: The number of independently trained models.
        num_rates: The number of different evolutionary times.
        num_matrices: The number of rate matrices.
        equilibrium_init: Initializer for the equilibrium distribution of the rate matrices
        exchangeability_init: Initializer for the exchangeability matrices. Usually inverse_softplus 
                                should be used on the initial matrix by the user.
        rate_init: Initializer for the rates.
        trainable_exchangeabilities: Flag that can prevent learning the exchangeabilities.
        per_matrix_rate: Learns an additional evolutionary time per rate matrix.
        matrix_rate_init: Initializer for the replacement rate per matrix.
        matrix_rate_l2: L2 regularizer strength that penalizes deviation of the parameters from the initial value.
        shared_matrix: Make all weight matrices internally use the same weights. Only useful in combination with 
                        num_matrices > 1 and per_matrix_rate = True. 
        equilibrium_sample: If true, a 2-staged process is assumed where an amino acid is first sampled from 
                        the equilibirium distribution and the ancestral probabilities are computed afterwards.
        transposed: Transposes the probability matrix P = e^tQ. 
        name: Layer name.
    """

    def __init__(self, 
                 num_models,
                 num_rates, 
                 num_matrices,
                 equilibrium_init,
                 exchangeability_init,
                 rate_init=tf.constant_initializer(-3.),
                 trainable_rate_matrices=True,
                 per_matrix_rate=False,
                 matrix_rate_init=None,
                 matrix_rate_l2=0.0,
                 shared_matrix=False,
                 equilibrium_sample=False,
                 transposed=False,
                 **kwargs):
        super(AncProbsLayer, self).__init__(**kwargs)
        self.num_models = num_models
        self.num_rates = num_rates
        self.num_matrices = num_matrices
        self.rate_init = rate_init
        self.equilibrium_init = equilibrium_init
        self.exchangeability_init = exchangeability_init
        self.trainable_rate_matrices = trainable_rate_matrices
        self.per_matrix_rate = per_matrix_rate
        self.matrix_rate_init = matrix_rate_init
        self.matrix_rate_l2 = matrix_rate_l2 
        self.shared_matrix = shared_matrix
        self.equilibrium_sample = equilibrium_sample
        self.transposed = transposed
    
    def build(self, input_shape=None):
        self.tau_kernel = self.add_weight(shape=[self.num_models, self.num_rates], 
                                   name="tau_kernel", 
                                   initializer=self.rate_init)
        if self.shared_matrix:
            self.exchangeability_kernel = self.add_weight(shape=[self.num_models, 1, 20, 20],
                                                          name="exchangeability_kernel",
                                                           initializer=self.exchangeability_init,
                                                           trainable=self.trainable_rate_matrices)
            
            self.equilibrium_kernel = self.add_weight(shape=[self.num_models, 1, 20], 
                                                      name="equilibrium_kernel",
                                                      initializer=self.equilibrium_init,
                                                      trainable=self.trainable_rate_matrices)
        else:
            self.exchangeability_kernel = self.add_weight(shape=[self.num_models, self.num_matrices, 20, 20], 
                                                           name="exchangeability_kernel", 
                                                           initializer=self.exchangeability_init,
                                                           trainable=self.trainable_rate_matrices)
            
            self.equilibrium_kernel = self.add_weight(shape=[self.num_models, self.num_matrices, 20], 
                                                      name="equilibrium_kernel",
                                                      initializer=self.equilibrium_init,
                                                      trainable=self.trainable_rate_matrices)
            
        if self.per_matrix_rate:
            self.per_matrix_rates_kernel = self.add_weight(shape=[self.num_models, self.num_matrices], 
                                                      name="per_matrix_rates_kernel",
                                                      initializer=self.matrix_rate_init)
       
    def make_R(self, kernel=None):
        if kernel is None:
            kernel = self.exchangeability_kernel
        R = 0.5 * (kernel + tf.transpose(kernel, [0,1,3,2])) #make symmetric
        R = tf.math.softplus(R)
        R -= tf.linalg.diag(tf.linalg.diag_part(R)) #zero diagonal
        return R
    
    def make_p(self):
        return tf.nn.softmax(self.equilibrium_kernel)
    
    def make_Q(self):
        R, p = self.make_R(), self.make_p()
        R = tf.reshape(R, (-1, 20, 20))
        p = tf.reshape(p, (-1, 20))
        Q = make_rate_matrix(R, p)
        Q = tf.reshape(Q, (self.num_models, self.num_matrices, 20, 20))
        return Q
        
    def make_tau(self, subset=None):
        tau = self.tau_kernel
        if subset is not None:
            tau = tf.gather_nd(tau, subset, batch_dims=1)
        return tf.math.softplus(tau)
    
    def make_per_matrix_rate(self):
        return tf.math.softplus(self.per_matrix_rates_kernel)

    def call(self, inputs, rate_indices):
        """ Computes anchestral probabilities of the inputs.
        Args:
            inputs: Input sequences. Shape: (num_model, b, L) or (num_models, b, L, s). The latter format (non index)
                    is only supported for raw amino acid input.
            rate_indices: Indices that map each input sequences to an evolutionary time. Shape: (num_model, b)
        Returns:
            Ancestral probabilities. Shape: (num_model, b, L, num_matrices*s)
        """
        rate_indices = tf.identity(rate_indices) #take care of numpy inputs
        rate_indices.set_shape([self.num_models,None]) #resolves tf 2.12 issues
        input_indices = len(inputs.shape) == 3 
        if input_indices:
            mask = inputs < 20
            only_std_aa_inputs = inputs * tf.cast(mask, inputs.dtype)
            mask = tf.cast(mask, self.dtype)
            mask = tf.expand_dims(mask, -1)
            mask = tf.expand_dims(mask, -1)
        else:
            only_std_aa_inputs = inputs
        rate_indices = tf.expand_dims(rate_indices, -1)
        tau_subset = self.make_tau(rate_indices)
        if self.per_matrix_rate:
            per_matrix_rates = self.make_per_matrix_rate()
            per_matrix_rates = tf.expand_dims(per_matrix_rates, 1)
            tau_subset = tf.expand_dims(tau_subset, 2)
            mut_rates = tau_subset * per_matrix_rates
            reg = tf.reduce_mean(tf.square(self.per_matrix_rates_kernel - inverse_softplus(1.)))
            self.add_loss(self.matrix_rate_l2 * reg)
        else:
            mut_rates = tau_subset
        reg_tau = tf.reduce_sum(tf.square(self.tau_kernel + 3.))
        self.add_loss(self.matrix_rate_l2 * reg_tau)
        anc_probs = make_anc_probs(only_std_aa_inputs, 
                                   self.make_R(), 
                                   self.make_p(), 
                                   mut_rates,
                                   self.equilibrium_sample,
                                   self.transposed)
        if input_indices:
            anc_probs *= mask
            anc_probs = tf.pad(anc_probs, [[0,0], [0,0], [0,0], [0,0], [0,6]])
            rest = tf.expand_dims(tf.one_hot(inputs, 26), -2) * (1-mask)
            anc_probs += rest
            num_model, b, L = tf.unstack(tf.shape(inputs))
            anc_probs = tf.reshape(anc_probs, (num_model, b, L, self.num_matrices * 26) )
        else:
            num_model, b, L, _ = tf.unstack(tf.shape(inputs))
            anc_probs = tf.reshape(anc_probs, (num_model, b, L, self.num_matrices * 20) )
        return anc_probs
        
    def get_config(self):
        config = super(AncProbsLayer, self).get_config()
        config.update({
             "num_models" : self.num_models,
             "num_rates" : self.num_rates,
             "num_matrices" : self.num_matrices,
             "equilibrium_init" : self.equilibrium_kernel.numpy(),
             "exchangeability_init" : self.exchangeability_kernel.numpy(),
             "rate_init" : self.tau_kernel.numpy(),
             "trainable_rate_matrices" : self.trainable_rate_matrices,
             "per_matrix_rate" : self.per_matrix_rate,
             "matrix_rate_init" : self.per_matrix_rates_kernel if self.per_matrix_rate else None,
             "matrix_rate_l2" : self.matrix_rate_l2,
             "shared_matrix" : self.shared_matrix,
             "equilibrium_sample" : self.equilibrium_sample,
             "transposed" : self.transposed
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        #override numpy arrays with initializers
        config["equilibrium_init"] = initializers.ConstantInitializer(config["equilibrium_init"])
        config["exchangeability_init"] = initializers.ConstantInitializer(config["exchangeability_init"])
        config["rate_init"] = initializers.ConstantInitializer(config["rate_init"])
        config["matrix_rate_init"] = initializers.ConstantInitializer(config["matrix_rate_init"]) if config["matrix_rate_init"] is not None else None
        return cls(**config)
        

# the default rate matrix ("LG")
LG_paml = ['0.425093 \n', 
           '0.276818 0.751878 \n', 
           '0.395144 0.123954 5.076149 \n', 
           '2.489084 0.534551 0.528768 0.062556 \n', 
           '0.969894 2.807908 1.695752 0.523386 0.084808 \n', 
           '1.038545 0.363970 0.541712 5.243870 0.003499 4.128591 \n', 
           '2.066040 0.390192 1.437645 0.844926 0.569265 0.267959 0.348847 \n', 
           '0.358858 2.426601 4.509238 0.927114 0.640543 4.813505 0.423881 0.311484 \n', 
           '0.149830 0.126991 0.191503 0.010690 0.320627 0.072854 0.044265 0.008705 0.108882 \n', 
           '0.395337 0.301848 0.068427 0.015076 0.594007 0.582457 0.069673 0.044261 0.366317 4.145067 \n', 
           '0.536518 6.326067 2.145078 0.282959 0.013266 3.234294 1.807177 0.296636 0.697264 0.159069 0.137500 \n',
           '1.124035 0.484133 0.371004 0.025548 0.893680 1.672569 0.173735 0.139538 0.442472 4.273607 6.312358 0.656604 \n', 
           '0.253701 0.052722 0.089525 0.017416 1.105251 0.035855 0.018811 0.089586 0.682139 1.112727 2.592692 0.023918 1.798853 \n', 
           '1.177651 0.332533 0.161787 0.394456 0.075382 0.624294 0.419409 0.196961 0.508851 0.078281 0.249060 0.390322 0.099849 0.094464 \n', 
           '4.727182 0.858151 4.008358 1.240275 2.784478 1.223828 0.611973 1.739990 0.990012 0.064105 0.182287 0.748683 0.346960 0.361819 1.338132 \n', 
           '2.139501 0.578987 2.000679 0.425860 1.143480 1.080136 0.604545 0.129836 0.584262 1.033739 0.302936 1.136863 2.020366 0.165001 0.571468 6.472279 \n', 
           '0.180717 0.593607 0.045376 0.029890 0.670128 0.236199 0.077852 0.268491 0.597054 0.111660 0.619632 0.049906 0.696175 2.457121 0.095131 0.248862 0.140825 \n', 
           '0.218959 0.314440 0.612025 0.135107 1.165532 0.257336 0.120037 0.054679 5.306834 0.232523 0.299648 0.131932 0.481306 7.803902 0.089613 0.400547 0.245841 3.151815 \n', 
           '2.547870 0.170887 0.083688 0.037967 1.959291 0.210332 0.245034 0.076701 0.119013 10.649107 1.702745 0.185202 1.898718 0.654683 0.296501 0.098369 2.188158 0.189510 0.249313 \n', 
           '0.079066 0.055941 0.041977 0.053052 0.012937 0.040767 0.071586 0.057337 0.022355 0.062157 0.099081 0.064600 0.022951 0.042302 0.044040 0.061197 0.053287 0.012066 0.034155 0.069147 \n']

# LG4X: Modeling Protein Evolution with Several Amino-Acid Replacement Matrices Depending on Site Rates
LG4X_paml = [ ['0.295719',
             '0.067388 0.448317',
             '0.253712 0.457483 2.358429',
             '1.029289 0.576016 0.251987 0.189008',
             '0.107964 1.741924 0.216561 0.599450 0.029955',
             '0.514644 0.736017 0.503084 109.901504 0.084794 4.117654',
             '10.868848 0.704334 0.435271 1.070052 1.862626 0.246260 1.202023',
             '0.380498 5.658311 4.873453 5.229858 0.553477 6.508329 1.634845 0.404968',
             '0.084223 0.123387 0.090748 0.052764 0.151733 0.054187 0.060194 0.048984 0.204296',
             '0.086976 0.221777 0.033310 0.021407 0.230320 0.195703 0.069359 0.069963 0.504221 1.495537',
             '0.188789 93.433377 0.746537 0.621146 0.096955 1.669092 2.448827 0.256662 1.991533 0.091940 0.122332',
             '0.286389 0.382175 0.128905 0.081091 0.352526 0.810168 0.232297 0.228519 0.655465 1.994320 3.256485 0.457430',
             '0.155567 0.235965 0.127321 0.205164 0.590018 0.066081 0.064822 0.241077 6.799829 0.754940 2.261319 0.163849 1.559944',
             '1.671061 6.535048 0.904011 5.164456 0.386853 2.437439 3.537387 4.320442 11.291065 0.170343 0.848067 5.260446 0.426508 0.438856',
             '2.132922 0.525521 0.939733 0.747330 1.559564 0.165666 0.435384 3.656545 0.961142 0.050315 0.064441 0.360946 0.132547 0.306683 4.586081',
             '0.529591 0.303537 0.435450 0.308078 0.606648 0.106333 0.290413 0.290216 0.448965 0.372166 0.102493 0.389413 0.498634 0.109129 2.099355 3.634276',
             '0.115551 0.641259 0.046646 0.260889 0.587531 0.093417 0.280695 0.307466 6.227274 0.206332 0.459041 0.033291 0.559069 18.392863 0.411347 0.101797 0.034710',
             '0.102453 0.289466 0.262076 0.185083 0.592318 0.035149 0.105999 0.096556 20.304886 0.097050 0.133091 0.115301 0.264728 66.647302 0.476350 0.148995 0.063603 20.561407',
             '0.916683 0.102065 0.043986 0.080708 0.885230 0.072549 0.206603 0.306067 0.205944 5.381403 0.561215 0.112593 0.693307 0.400021 0.584622 0.089177 0.755865 0.133790 0.154902', 
             '0.147383 0.017579 0.058208 0.017707 0.026331 0.041582 0.017494 0.027859 0.011849 0.076971 0.147823 0.019535 0.037132 0.029940 0.008059 0.088179 0.089653 0.006477 0.032308 0.097931' ],
          ['0.066142',
             '0.590377 0.468325',
             '0.069930 0.013688 2.851667',
             '9.850951 0.302287 3.932151 0.146882',
             '1.101363 1.353957 8.159169 0.249672 0.582670',
             '0.150375 0.028386 0.219934 0.560142 0.005035 3.054085',
             '0.568586 0.037750 0.421974 0.046719 0.275844 0.129551 0.037250',
             '0.051668 0.262130 2.468752 0.106259 0.098208 4.210126 0.029788 0.013513',
             '0.127170 0.016923 0.344765 0.003656 0.445038 0.165753 0.008541 0.002533 0.031779',
             '0.292429 0.064289 0.210724 0.004200 1.217010 1.088704 0.014768 0.005848 0.064558 7.278994',
             '0.071458 0.855973 1.172204 0.014189 0.033969 1.889645 0.125869 0.031390 0.065585 0.029917 0.042762',
             '1.218562 0.079621 0.763553 0.009876 1.988516 3.344809 0.056702 0.021612 0.079927 7.918203 14.799537 0.259400',
             '0.075144 0.011169 0.082464 0.002656 0.681161 0.111063 0.004186 0.004854 0.095591 0.450964 1.506485 0.009457 1.375871',
             '7.169085 0.161937 0.726566 0.040244 0.825960 2.067758 0.110993 0.129497 0.196886 0.169797 0.637893 0.090576 0.457399 0.143327',
             '30.139501 0.276530 11.149790 0.267322 18.762977 3.547017 0.201148 0.976631 0.408834 0.104288 0.123793 0.292108 0.598048 0.328689 3.478333',
             '13.461692 0.161053 4.782635 0.053740 11.949233 2.466507 0.139705 0.053397 0.126088 1.578530 0.641351 0.297913 4.418398 0.125011 2.984862 13.974326',
             '0.021372 0.081472 0.058046 0.006597 0.286794 0.188236 0.009201 0.019475 0.037226 0.015909 0.154810 0.017172 0.239749 0.562720 0.061299 0.154326 0.060703',
             '0.045779 0.036742 0.498072 0.027639 0.534219 0.203493 0.012095 0.004964 0.452302 0.094365 0.140750 0.021976 0.168432 1.414883 0.077470 0.224675 0.123480 0.447011',
             '4.270235 0.030342 0.258487 0.012745 4.336817 0.281953 0.043812 0.015539 0.016212 16.179952 3.416059 0.032578 2.950318 0.227807 1.050562 0.112000 5.294490 0.033381 0.045528',
               '0.063139 0.066357 0.011586 0.066571 0.010800 0.009276 0.053984 0.146986 0.034214 0.088822 0.098196 0.032390 0.021263 0.072697 0.016761 0.020711 0.020797 0.025463 0.045615 0.094372'],
          ['0.733336',
             '0.558955 0.597671',
             '0.503360 0.058964 5.581680',
             '4.149599 2.863355 1.279881 0.225860',
             '1.415369 2.872594 1.335650 0.434096 1.043232',
             '1.367574 0.258365 0.397108 2.292917 0.209978 4.534772',
             '1.263002 0.366868 1.840061 1.024707 0.823594 0.377181 0.496780',
             '0.994098 2.578946 5.739035 0.821921 3.039380 4.877840 0.532488 0.398817',
             '0.517204 0.358350 0.284730 0.027824 1.463390 0.370939 0.232460 0.008940 0.349195',
             '0.775054 0.672023 0.109781 0.021443 1.983693 1.298542 0.169219 0.043707 0.838324 5.102837',
             '0.763094 5.349861 1.612642 0.088850 0.397640 3.509873 0.755219 0.436013 0.888693 0.561690 0.401070',
             '1.890137 0.691594 0.466979 0.060820 2.831098 2.646440 0.379926 0.087640 0.488389 7.010411 8.929538 1.357738',
             '0.540460 0.063347 0.141582 0.018288 4.102068 0.087872 0.020447 0.064863 1.385133 3.054968 5.525874 0.043394 3.135353',
             '0.200122 0.032875 0.019509 0.042687 0.059723 0.072299 0.023282 0.036426 0.050226 0.039318 0.067505 0.023126 0.012695 0.015631',
             '4.972745 0.821562 4.670980 1.199607 5.901348 1.139018 0.503875 1.673207 0.962470 0.204155 0.273372 0.567639 0.570771 0.458799 0.233109',
             '1.825593 0.580847 1.967383 0.420710 2.034980 0.864479 0.577513 0.124068 0.502294 2.653232 0.437116 1.048288 2.319555 0.151684 0.077004 8.113282',
             '0.450842 0.661866 0.088064 0.037642 2.600668 0.390688 0.109318 0.218118 1.065585 0.564368 1.927515 0.120994 1.856122 4.154750 0.011074 0.377578 0.222293',
             '0.526135 0.265730 0.581928 0.141233 5.413080 0.322761 0.153776 0.039217 8.351808 0.854294 0.940458 0.180650 0.975427 11.429924 0.026268 0.429221 0.273138 4.731579',
             '3.839269 0.395134 0.145401 0.090101 4.193725 0.625409 0.696533 0.104335 0.377304 15.559906 2.508169 0.449074 3.404087 1.457957 0.052132 0.260296 2.903836 0.564762 0.681215',
              '0.062457 0.066826 0.049332 0.065270 0.006513 0.041231 0.058965 0.080852 0.028024 0.037024 0.075925 0.064131 0.019620 0.028710 0.104579 0.056388 0.062027 0.008241 0.033124 0.050760'],
          ['0.658412',
             '0.566269 0.540749',
             '0.854111 0.058015 3.060574',
             '0.884454 5.851132 1.279257 0.160296',
             '1.309554 2.294145 1.438430 0.482619 0.992259',
             '1.272639 0.182966 0.431464 2.992763 0.086318 2.130054',
             '1.874713 0.684164 2.075952 1.296206 2.149634 0.571406 0.507160',
             '0.552007 3.192521 4.840271 0.841829 5.103188 4.137385 0.351381 0.679853',
             '0.227683 0.528161 0.644656 0.031467 3.775817 0.437589 0.189152 0.025780 0.665865',
             '0.581512 1.128882 0.266076 0.048542 3.954021 2.071689 0.217780 0.082005 1.266791 8.904999',
             '0.695190 3.010922 2.084975 0.132774 0.190734 2.498630 0.767361 0.326441 0.680174 0.652629 0.440178',
             '0.967985 1.012866 0.720060 0.133055 1.776095 1.763546 0.278392 0.343977 0.717301 10.091413 14.013035 1.082703',
             '0.344015 0.227296 0.291854 0.056045 4.495841 0.116381 0.092075 0.195877 4.001286 2.671718 5.069337 0.091278 4.643214',
             '0.978992 0.156635 0.028961 0.209188 0.264277 0.296578 0.177263 0.217424 0.362942 0.086367 0.539010 0.172734 0.121821 0.161015',
             '3.427163 0.878405 4.071574 0.925172 7.063879 1.033710 0.451893 3.057583 1.189259 0.359932 0.742569 0.693405 0.584083 1.531223 1.287474',
             '2.333253 0.802754 2.258357 0.360522 2.221150 1.283423 0.653836 0.377558 0.964545 4.797423 0.780580 1.422571 4.216178 0.599244 0.444362 5.231362',
             '0.154701 0.830884 0.073037 0.094591 3.017954 0.312579 0.074620 0.401252 1.350568 0.336801 1.331875 0.068958 1.677263 5.832025 0.076328 0.548763 0.208791',
             '0.221089 0.431617 1.238426 0.313945 8.558815 0.305772 0.181992 0.072258 12.869737 1.021885 1.531589 0.163829 1.575754 33.873091 0.079916 0.831890 0.307846 5.910440',
             '2.088785 0.456530 0.199728 0.118104 4.310199 0.681277 0.752277 0.241015 0.531100 23.029406 4.414850 0.481711 5.046403 1.914768 0.466823 0.382271 3.717971 0.282540 0.964421',
             '0.106471 0.074171 0.044513 0.096390 0.002148 0.066733 0.158908 0.037625 0.020691 0.014608 0.028797 0.105352 0.007864 0.007477 0.083595 0.055726 0.047711 0.003975 0.010088 0.027159'] ]