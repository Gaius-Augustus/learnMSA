import tensorflow as tf
import numpy as np
import learnMSA.tree.tf.initializer as initializer
from learnMSA.util.sequence_dataset import SequenceDataset
from learnMSA.tree.util import inverse_softplus, deserialize

from evoten.backend_tf import BackendTF
from evoten.expm_gtr import expm_gtr

# Initialize evoten backend
backend = BackendTF()

"""Ancestral Probability Layer
Learn one or several rate matrices jointly with a downstream model. Amino acid sequences can be smeared towards expected amino acid distributions after
some amount of evolutionary time has passed under a substitution model. This can help to train models on distantly related sequences.
"""

def make_anc_probs(sequences, exchangeabilities, equilibrium, tau, equilibrium_sample=False, transposed=False):
    """Computes ancestral probabilities simultaneously for all sites and rate matrices.
    Args:
        sequences: Sequences either as integers (faster embedding lookup) or in vector format. Shape: (num_model, b, L) or (num_model, b, L, s)
        exchangeabilities: A stack of symmetric exchangeability matrices. Shape: (num_model, k, s, s)
        equilibrium: A stack of equilibrium distributions. Shape: (num_model, k, s)
        tau: Evolutionary times for all sequences (1 time unit = 1 expected mutation per site). Shape: (num_model, b) or (num_model,b,k)
        equilibrium_sample: If true, a 2-staged process is assumed where an amino acid is first sampled from the equilibirium distribution
                    and the ancestral probabilities are computed afterwards.
        transposed: Transposes the probability matrix P = e^tQ.
    Returns:
        A tensor with the expected amino acids frequencies after time tau. Output shape: (num_model, b, L, k, s)
    """
    while len(tau.shape) < 5:
        tau = tf.expand_dims(tau, -1)
    shape = tf.shape(exchangeabilities)
    num_model, k, s, _ = tf.unstack(shape, 4)
    
    # Reshape for evoten backend
    exchangeabilities = tf.reshape(exchangeabilities, (-1, s, s))
    equilibrium_flat = tf.reshape(equilibrium, (-1, s))
    
    # Use evoten backend to create rate matrices
    Q = backend.make_rate_matrix(exchangeabilities, equilibrium_flat)
    Q = tf.reshape(Q, (num_model, k, s, s))
    
    # Extract tau: shape is (num_model, b, k, 1, 1), we need (num_model, b, k)
    tau_extracted = tau[..., 0, 0]  # Shape: (num_model, b, k)
    
    # Use evoten's expm_gtr to create transition probabilities
    # expm_gtr(Q, t, pi) expects:
    #   Q: (..., d, d) - rate matrices
    #   t: (...) - times  
    #   pi: (..., d) - equilibrium distributions
    # It will broadcast and compute exp(t*Q) for each combination
    # 
    # We have:
    #   Q: (num_model, k, s, s)
    #   tau: (num_model, b, k)
    #   equilibrium: (num_model, k, s)
    # We want P[m,b,k,i,j]
    
    # Add b dimension to Q and equilibrium
    Q_exp = tf.expand_dims(Q, 1)  # (num_model, 1, k, s, s)
    equilibrium_exp = tf.expand_dims(equilibrium, 1)  # (num_model, 1, k, s)
    
    # expm_gtr will broadcast to (num_model, b, k, s, s)
    P = expm_gtr(Q_exp, tau_extracted, equilibrium_exp)
    
    if equilibrium_sample:
        equilibrium_reshaped = tf.reshape(equilibrium, (num_model, 1, k, s, 1))
        P *= equilibrium_reshaped
        
    if len(sequences.shape) == 3:  # assume index format
        b_actual = tf.shape(sequences)[1]
        if transposed:
            P = tf.transpose(P, [0, 1, 4, 2, 3])
        else:
            P = tf.transpose(P, [0, 1, 3, 2, 4])
        P = tf.reshape(P, (-1, k, s))
        sequences = tf.cast(sequences, tf.int32)
        sequences = tf.reshape(sequences, (-1, tf.shape(sequences)[-1]))
        sequences += tf.expand_dims(tf.range(num_model * b_actual) * s, -1)
        ancprobs = tf.nn.embedding_lookup(P, sequences)
        ancprobs = tf.reshape(ancprobs, (num_model, b_actual, -1, k, s))
    else:  # assume vector format
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
        trainable_distances: Flag that can prevent learning the evolutionary times.
        per_matrix_rate: Learns an additional evolutionary time per rate matrix.
        matrix_rate_init: Initializer for the replacement rate per matrix.
        matrix_rate_l2: L2 regularizer strength that penalizes deviation of the parameters from the initial value.
        shared_matrix: Make all weight matrices internally use the same weights. Only useful in combination with
                        num_matrices > 1 and per_matrix_rate = True.
        equilibrium_sample: If true, a 2-staged process is assumed where an amino acid is first sampled from
                        the equilibirium distribution and the ancestral probabilities are computed afterwards.
        transposed: Transposes the probability matrix P = e^tQ.
        clusters: An optional vector that assigns each sequence to a cluster. If provided, the evolutionary time
                    is learned per cluster.
        use_lstm: Experimental setting that estimates the evolutionary distance of a sequence with an lstm.
        name: Layer name.
    """

    def __init__(self,
                 num_models,
                 num_rates,
                 num_matrices,
                 equilibrium_init,
                 exchangeability_init,
                 rate_init=initializer.ConstantInitializer(-3.),
                 trainable_rate_matrices=True,
                 trainable_distances=True,
                 per_matrix_rate=False,
                 matrix_rate_init=None,
                 matrix_rate_l2=0.0,
                 shared_matrix=False,
                 equilibrium_sample=False,
                 transposed=False,
                 clusters=None,
                 use_lstm=False,
                 **kwargs):
        super(AncProbsLayer, self).__init__(**kwargs)
        self.num_models = num_models
        self.num_rates = num_rates
        self.num_matrices = num_matrices
        self.rate_init = rate_init
        self.equilibrium_init = equilibrium_init
        self.exchangeability_init = exchangeability_init
        self.trainable_rate_matrices = trainable_rate_matrices
        self.trainable_distances = trainable_distances
        self.per_matrix_rate = per_matrix_rate
        self.matrix_rate_init = matrix_rate_init
        self.matrix_rate_l2 = matrix_rate_l2
        self.shared_matrix = shared_matrix
        self.equilibrium_sample = equilibrium_sample
        self.transposed = transposed
        self.clusters = clusters
        self.num_clusters = np.max(clusters) + 1 if clusters is not None else self.num_rates
        self.use_lstm = use_lstm
        self._head_subset = None

    @property
    def head_subset(self):
        """If set, only these models are used in computations."""
        return self._head_subset

    @head_subset.setter
    def head_subset(self, subset):
        self._head_subset = subset

    def build(self, input_shape=None):
        if self.built:
            return
        if self.use_lstm:
            self.lstm_dim = 64
            self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_dim, return_sequences=True, zero_output_for_mask=True), merge_mode="sum")
            self.dense = tf.keras.layers.Dense(1,
                                            activation="softplus",
                                            #kernel_initializer="zeros",
                                            bias_initializer=self.rate_init)
        else:
            self.tau_kernel = self.add_weight(shape=[self.num_models, self.num_clusters],
                                    name="tau_kernel",
                                    initializer=self.rate_init,
                                    trainable=self.trainable_distances)
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
        self.built = True


    def make_R(self, kernel=None):
        if kernel is None:
            kernel = self.exchangeability_kernel
            if self._head_subset is not None:
                kernel = tf.gather(kernel, self._head_subset, axis=0)
        # Use evoten backend to make symmetric positive semi-definite matrix
        return backend.make_symmetric_pos_semidefinite(kernel)

    def make_p(self):
        kernel = self.equilibrium_kernel
        if self._head_subset is not None:
            kernel = tf.gather(kernel, self._head_subset, axis=0)
        # Use evoten backend to make equilibrium distribution
        return backend.make_equilibrium(kernel)

    def make_Q(self):
        R, p = self.make_R(), self.make_p()
        R = tf.reshape(R, (-1, 20, 20))
        p = tf.reshape(p, (-1, 20))
        # Use evoten backend to create rate matrices
        Q = backend.make_rate_matrix(R, p)
        num_models = len(self._head_subset) if self._head_subset is not None else self.num_models
        Q = tf.reshape(Q, (num_models, self.num_matrices, 20, 20))
        return Q

    def make_tau(self, inputs=None, subset=None):
        if self.use_lstm:
            if len(inputs.shape) == 3:
                num_model, b, L = tf.unstack(tf.shape(inputs))
            else:
                num_model, b, L, _ = tf.unstack(tf.shape(inputs))
            lstm_input = tf.one_hot(inputs, len(SequenceDataset._default_alphabet))
            lstm_input = tf.reshape(lstm_input, (num_model*b, L, len(SequenceDataset._default_alphabet)))
            lstm_mask = tf.reshape(inputs < len(SequenceDataset._default_alphabet)-1, (num_model*b, L))
            seq_lens = tf.reduce_sum(tf.cast(lstm_mask, lstm_input.dtype), axis=-1, keepdims=True)
            lstm_output = self.lstm(lstm_input, mask=lstm_mask)
            lstm_output = tf.reduce_sum(lstm_output, axis=-2) / (seq_lens * self.lstm_dim)
            lstm_output = tf.reshape(lstm_output, (num_model, b, self.lstm_dim))
            return self.dense(lstm_output)[...,0]
        else:
            tau = self.tau_kernel
            if self._head_subset is not None:
                tau = tf.gather(tau, self._head_subset, axis=0)
            if self.clusters is not None:
                tau = tf.gather(tau, self.clusters, axis=-1)
            if subset is not None:
                tau = tf.gather_nd(tau, subset, batch_dims=1)
            # Use evoten backend to convert kernel to branch lengths
            return backend.make_branch_lengths(tau)

    def make_per_matrix_rate(self):
        kernel = self.per_matrix_rates_kernel
        if self._head_subset is not None:
            kernel = tf.gather(kernel, self._head_subset, axis=0)
        # Use evoten backend to convert kernel to branch lengths
        return backend.make_branch_lengths(kernel)

    def call(self, inputs, rate_indices, replace_rare_with_equilibrium=True):
        """ Computes anchestral probabilities of the inputs.
        Args:
            inputs: Input sequences. Shape: (num_model, b, L) or (num_models, b, L, s). The latter format (non index)
                    is only supported for raw amino acid input.
            rate_indices: Indices that map each input sequences to an evolutionary time. Shape: (num_model, b)
            replace_rate_with_equilibrium: If true, replaces non-standard amino acids with the equilibrium distribution.
        Returns:
            Ancestral probabilities. Shape: (num_model, b, L, num_matrices*s)
        """
        input_indices = len(inputs.shape) == 3
        def _make_mask(bools):
            mask = tf.cast(bools, self.dtype)
            mask = tf.expand_dims(mask, -1)
            mask = tf.expand_dims(mask, -1)
            return mask
        if input_indices:
            bool_mask = inputs < 20
            mask = _make_mask(bool_mask)
            only_std_aa_inputs = inputs * tf.cast(bool_mask, inputs.dtype)
        else:
            only_std_aa_inputs = inputs
        tau_subset = self.make_tau(inputs, tf.expand_dims(rate_indices, -1))
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
        equilibrium = self.make_p()
        anc_probs = make_anc_probs(only_std_aa_inputs,
                                   self.make_R(),
                                   equilibrium,
                                   mut_rates,
                                   self.equilibrium_sample,
                                   self.transposed)
        if input_indices:
            anc_probs *= mask
            anc_probs = tf.pad(anc_probs, [[0,0], [0,0], [0,0], [0,0], [0,len(SequenceDataset._default_alphabet)-20]])
            if replace_rare_with_equilibrium:
                rare_mask = _make_mask(tf.math.logical_and(inputs >= 20, inputs < len(SequenceDataset._default_alphabet)-1)) #do not count padding
                padding_mask = _make_mask(inputs == len(SequenceDataset._default_alphabet)-1)
                equilibrium = tf.concat([equilibrium, tf.zeros_like(equilibrium)[..., :tf.shape(anc_probs)[-1]-20]], -1)
                rest = (tf.zeros_like(anc_probs) + equilibrium[:,tf.newaxis,tf.newaxis]) * rare_mask
                rest += tf.expand_dims(tf.one_hot(inputs, len(SequenceDataset._default_alphabet)), -2) * padding_mask
            else:
                rest = tf.expand_dims(tf.one_hot(inputs, len(SequenceDataset._default_alphabet)), -2) * (1-mask)
            anc_probs += rest
            num_model, b, L = tf.unstack(tf.shape(inputs))
            anc_probs = tf.reshape(anc_probs, (num_model, b, L, self.num_matrices * len(SequenceDataset._default_alphabet)) )
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
             "equilibrium_init" : initializer.ConstantInitializer(self.equilibrium_kernel.numpy()),
             "exchangeability_init" : initializer.ConstantInitializer(self.exchangeability_kernel.numpy()),
             "rate_init" : initializer.ConstantInitializer(self.tau_kernel.numpy()),
             "trainable_rate_matrices" : self.trainable_rate_matrices,
            "trainable_distances" : self.trainable_distances,
             "per_matrix_rate" : self.per_matrix_rate,
             "matrix_rate_init" : initializer.ConstantInitializer(self.per_matrix_rates_kernel) if self.per_matrix_rate else None,
             "matrix_rate_l2" : self.matrix_rate_l2,
             "shared_matrix" : self.shared_matrix,
             "equilibrium_sample" : self.equilibrium_sample,
             "transposed" : self.transposed,
                "clusters" : self.clusters,
             "use_lstm" : self.use_lstm
        })
        return config

    @classmethod
    def from_config(cls, config):
        config["clusters"] = deserialize(config["clusters"])
        return cls(**config)
