import tensorflow as tf
import numpy as np
import learnMSA.tree.tf.initializer as initializer
from learnMSA.tree.tf.util import deserialize

from evoten.backend_tf import BackendTF
from evoten.expm_gtr import expm_gtr, precompute_gtr, expm_gtr_from_decomp

# Initialize evoten backend
backend = BackendTF()

class AncProbsLayer(tf.keras.layers.Layer):
    """A learnable layer for ancestral probabilities.
        It models the expected amino acid distributions after some amount of
        evolutionary time has passed under a substitution model.

    Args:
        num_models: The number of independently trained models.
        num_rates: The number of different evolutionary times.
        num_matrices: The number of rate matrices.
        equilibrium_init: Initializer for the equilibrium distribution of the
            rate matrices
        exchangeability_init: Initializer for the exchangeability matrices.
            Usually inverse_softplus should be used on the initial matrix by
            the user.
        rate_init: Initializer for the rates.
        trainable_rate_matrices: Flag that can prevent learning the rate
            matrices.
        trainable_distances: Flag that can prevent learning the evolutionary
            times.
        matrix_rate_l2: L2 regularizer strength that penalizes deviation of
            the parameters from the initial value.
        shared_matrix: Make all weight matrices internally use the same
            weights. Only useful in combination with num_matrices > 1.
        equilibrium_sample: If true, a 2-staged process is assumed where an
            amino acid is first sampled from the equilibirium distribution and
            the ancestral probabilities are computed afterwards.
        transposed: Transposes the probability matrix P = e^tQ.
        clusters: An optional vector that assigns each sequence to a cluster.
            If provided, the evolutionary time is learned per cluster.
        name: Layer name.
    """

    def __init__(
        self,
        num_models: int,
        num_rates: int,
        num_matrices: int,
        equilibrium_init,
        exchangeability_init,
        rate_init=initializer.ConstantInitializer(-3.),
        trainable_rate_matrices=False,
        trainable_distances=True,
        matrix_rate_l2=0.0,
        shared_matrix=False,
        equilibrium_sample=False,
        transposed=False,
        clusters=None,
        **kwargs
    ):
        super(AncProbsLayer, self).__init__(**kwargs)
        self.num_models = num_models
        self.num_rates = num_rates
        self.num_matrices = num_matrices
        self.rate_init = rate_init
        self.equilibrium_init = equilibrium_init
        self.exchangeability_init = exchangeability_init
        self.trainable_rate_matrices = trainable_rate_matrices
        self.trainable_distances = trainable_distances
        self.matrix_rate_l2 = matrix_rate_l2
        self.shared_matrix = shared_matrix
        self.equilibrium_sample = equilibrium_sample
        self.transposed = transposed
        self.clusters = clusters
        if clusters is None:
            self.num_clusters = self.num_rates
        else:
            self.num_clusters = np.max(clusters) + 1
        self._head_subset = None
        self._gtr_decomp = None

    @property
    def head_subset(self):
        """If set, only these models are used in computations."""
        return self._head_subset

    @head_subset.setter
    def head_subset(self, subset):
        self._head_subset = subset
        # Recompute GTR decomposition if needed
        if self.built and not self.trainable_rate_matrices:
            self._precompute_gtr_decomposition()

    def build(self, input_shape=None):
        if self.built:
            return
        self.tau_kernel = self.add_weight(
            shape=[self.num_models, self.num_clusters],
            name="tau_kernel",
            initializer=self.rate_init,
            trainable=self.trainable_distances,
        )
        if self.shared_matrix:
            self.exchangeability_kernel = self.add_weight(
                shape=[self.num_models, 1, 20, 20],
                name="exchangeability_kernel",
                initializer=self.exchangeability_init,
                trainable=self.trainable_rate_matrices
            )
            self.equilibrium_kernel = self.add_weight(
                shape=[self.num_models, 1, 20],
                name="equilibrium_kernel",
                initializer=self.equilibrium_init,
                trainable=self.trainable_rate_matrices
            )
        else:
            self.exchangeability_kernel = self.add_weight(
                shape=[self.num_models, self.num_matrices, 20, 20],
                name="exchangeability_kernel",
                initializer=self.exchangeability_init,
                trainable=self.trainable_rate_matrices
            )
            self.equilibrium_kernel = self.add_weight(
                shape=[self.num_models, self.num_matrices, 20],
                name="equilibrium_kernel",
                initializer=self.equilibrium_init,
                trainable=self.trainable_rate_matrices
            )

        # Precompute GTR decomposition if rate matrices are not trainable
        if not self.trainable_rate_matrices:
            self._precompute_gtr_decomposition()

        self.built = True

    def _precompute_gtr_decomposition(self):
        """Precompute GTR eigendecomposition for non-trainable rate matrices.
        Stores the result as a constant tensor to optimize computation.
        """
        # Compute rate matrices
        R, p = self.make_R(), self.make_p()
        num_models = len(self._head_subset) if self._head_subset is not None else self.num_models
        R_flat = tf.reshape(R, (-1, 20, 20))
        p_flat = tf.reshape(p, (-1, 20))
        Q = backend.make_rate_matrix(R_flat, p_flat)
        Q = tf.reshape(Q, (num_models, self.num_matrices, 20, 20))

        # Add batch dimension for compatibility with _compute_anc_probs
        Q_exp = tf.expand_dims(Q, 1)  # (num_models, 1, num_matrices, 20, 20)
        p_exp = tf.expand_dims(p, 1)  # (num_models, 1, num_matrices, 20)

        # Precompute GTR decomposition
        decomp = precompute_gtr(Q_exp, p_exp)

        # Store as non-trainable constants for use in forward pass
        self._gtr_decomp = type(decomp)(
            eigvals=tf.constant(decomp.eigvals.numpy()),
            eigvecs=tf.constant(decomp.eigvecs.numpy()),
            sqrt_pi=tf.constant(decomp.sqrt_pi.numpy()),
            inv_sqrt_pi=tf.constant(decomp.inv_sqrt_pi.numpy())
        )

    def make_R(self, kernel: tf.Tensor | None = None) -> tf.Tensor:
        """Computes the exchangeability matrices R for all models."""
        if kernel is None:
            kernel = self.exchangeability_kernel
            if self._head_subset is not None:
                kernel = tf.gather(kernel, self._head_subset, axis=0)
        return backend.make_symmetric_pos_semidefinite(kernel)

    def make_p(self) -> tf.Tensor:
        """Computes the equilibrium distributions p for all models."""
        kernel = self.equilibrium_kernel
        if self._head_subset is not None:
            kernel = tf.gather(kernel, self._head_subset, axis=0)
        return backend.make_equilibrium(kernel)

    def make_Q(self) -> tf.Tensor:
        """Computes the rate matrices Q for all models."""
        R, p = self.make_R(), self.make_p()
        R = tf.reshape(R, (-1, 20, 20))
        p = tf.reshape(p, (-1, 20))
        Q = backend.make_rate_matrix(R, p)
        num_models = len(self._head_subset) if self._head_subset is not None else self.num_models
        Q = tf.reshape(Q, (num_models, self.num_matrices, 20, 20))
        return Q

    def make_tau(self, subset: tf.Tensor | None = None) -> tf.Tensor:
        """
        Computes the evolutionary times (tau) for each sequence (in the subset),
        i.e. the length of the branch in the star-shaped tree that connects the
        sequence to the root.

        Args:
            subset: An optional tensor of shape (B, H) that specifies a
                    subset of sequences to compute tau for.
                    If None, computes tau for all sequences.

        Returns:
            A tensor of shape (B, H) containing the evolutionary times for the
            specified subset of sequences.
        """
        tau = self.tau_kernel

        if self._head_subset is not None:
            tau = tf.gather(tau, self._head_subset, axis=0)

        if self.clusters is not None:
            tau = tf.gather(tau, self.clusters, axis=1)

        if subset is None:
            tau = tf.transpose(tau)
        else:
            # Transpose the indices to gather with batch_dims=1
            tau = tf.gather(tau, tf.transpose(subset), batch_dims=1)
            tau = tf.transpose(tau)

        return backend.make_branch_lengths(tau)

    def _compute_anc_probs(self, sequences: tf.Tensor, tau: tf.Tensor) -> tf.Tensor:
        """Computes ancestral probabilities simultaneously for all sites and rate matrices.

        Args:
            sequences: Sequences in one-hot vector format. Shape: (B, L, H, 20)
            tau: Evolutionary times. Shape: (B, H)

        Returns:
            Ancestral probabilities. Shape: (B, L, H, k, 20)
        """
        # Compute equilibrium if needed
        if self.equilibrium_sample or self._gtr_decomp is None:
            equilibrium = self.make_p()

        # Expand tau to proper shape: (B, H) -> (B, H, num_matrices)
        tau_exp = tf.expand_dims(tau, -1)

        # Compute probability matrices
        if self._gtr_decomp is not None:
            # Transpose tau from (B, H, k) to (H, B, k) for expm_gtr_from_decomp
            tau_transposed = tf.transpose(tau_exp, [1, 0, 2])
            P = expm_gtr_from_decomp(self._gtr_decomp, tau_transposed)
            # Transpose P from (H, B, k, s, s) to (B, H, k, s, s)
            P = tf.transpose(P, [1, 0, 2, 3, 4])
        else:
            exchangeabilities = self.make_R()

            # Compute rate matrices on the fly
            shape = tf.shape(exchangeabilities)
            _, k, s, _ = tf.unstack(shape, 4)

            # Reshape for evoten backend
            exchangeabilities_flat = tf.reshape(exchangeabilities, (-1, s, s))
            equilibrium_flat = tf.reshape(equilibrium, (-1, s))

            Q = backend.make_rate_matrix(exchangeabilities_flat, equilibrium_flat)
            Q = tf.reshape(Q, (tf.shape(exchangeabilities)[0], k, s, s))

            # Add batch dimension
            Q_exp = tf.expand_dims(Q, 1)  # (num_model, 1, k, s, s)
            equilibrium_exp = tf.expand_dims(equilibrium, 1)  # (num_model, 1, k, s)

            # Transpose tau from (B, H, k) to (H, B, k) for expm_gtr
            tau_transposed = tf.transpose(tau_exp, [1, 0, 2])
            P = expm_gtr(Q_exp, tau_transposed, equilibrium_exp)
            # Transpose P from (H, B, k, s, s) to (B, H, k, s, s)
            P = tf.transpose(P, [1, 0, 2, 3, 4])

        if self.equilibrium_sample:
            # Reshape equilibrium from (num_model, k, s) to (1, num_model, k, s, 1) for broadcasting
            shape = tf.shape(equilibrium)
            equilibrium_reshaped = tf.reshape(equilibrium, (1, shape[0], shape[1], shape[2], 1))
            P *= equilibrium_reshaped

        # Compute ancestral probabilities using einsum
        if self.transposed:
            ancprobs = tf.einsum("bLmz,bmksz->bLmks", sequences, P)
        else:
            ancprobs = tf.einsum("bLmz,bmkzs->bLmks", sequences, P)

        return ancprobs

    def call(
        self,
        inputs: tf.Tensor,
        rate_indices: tf.Tensor,
    ) -> tf.Tensor:
        """Computes ancestral probabilities of the inputs.

        Args:
            inputs: Input sequences in one-hot vector format.
                Shape: (B, T, H, S).
            rate_indices: Indices that map each input sequences to an
                evolutionary time. Shape: (B, H)

        Returns:
            Ancestral probabilities. Shape: (B, T, H, num_matrices*S)
        """
        tau = self.make_tau(rate_indices)  # (B, H)
        self._add_tau_regularization_loss()

        B, T, H, S = tf.unstack(tf.shape(inputs))

        # Handle special amino acids
        std_aa_only = inputs[:, :, :, :20]
        special = inputs[:, :, :, 20:]

        # Compute ancestral probabilities
        anc_probs = self._compute_anc_probs(std_aa_only, tau) # (B, T, H, k, 20)

        special = tf.broadcast_to(
            tf.expand_dims(special, axis=3),
            (B, T, H, self.num_matrices, special.shape[-1]),
        )
        anc_probs = tf.concat([anc_probs, special], axis=-1)

        anc_probs = tf.reshape(anc_probs, (B, T, H, self.num_matrices * S))
        return anc_probs

    def _add_tau_regularization_loss(self) -> None:
        """Add L2 regularization loss for tau parameters."""
        reg_tau = tf.reduce_sum(tf.square(self.tau_kernel + 3.))
        self.add_loss(self.matrix_rate_l2 * reg_tau)

    def get_config(self):
        config = super(AncProbsLayer, self).get_config()
        config.update({
            "num_models": self.num_models,
            "num_rates": self.num_rates,
            "num_matrices": self.num_matrices,
            "equilibrium_init": initializer.ConstantInitializer(self.equilibrium_kernel.numpy()),
            "exchangeability_init": initializer.ConstantInitializer(self.exchangeability_kernel.numpy()),
            "rate_init": initializer.ConstantInitializer(self.tau_kernel.numpy()),
            "trainable_rate_matrices": self.trainable_rate_matrices,
            "trainable_distances": self.trainable_distances,
            "matrix_rate_l2": self.matrix_rate_l2,
            "shared_matrix": self.shared_matrix,
            "equilibrium_sample": self.equilibrium_sample,
            "transposed": self.transposed,
            "clusters": self.clusters
        })
        return config

    @classmethod
    def from_config(cls, config):
        config["clusters"] = deserialize(config["clusters"])
        return cls(**config)
