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
        It learns per-sequence continuous evolutionary times and computes
        substitution probabilities under a fixed substitution model.

    Args:
        heads: The number of independently trained models. The layer will create
            a separate rate matrix for each head.
        rates: The number of evolutionary times that will be assigned
            using the indices passed to the call method.
        input_tracks: The number of input tracks. The layer will create
            a separate rate matrix for each track.
        equilibrium_init: Initializer for the equilibrium distribution of the
            rate matrices
        exchangeability_init: Initializer for the exchangeability matrices.
            Usually inverse_softplus should be used on the initial matrix by
            the user.
        rate_init: Initializer for the rates.
        trainable_distances: Flag that can prevent learning the evolutionary
            times.
        trainable_rate_matrices: Flag that can prevent learning the rate
            matrices. If False, the GTR eigendecomposition is precomputed and
            stored as a constant tensor to optimize computation. If True,
            time_reversed should also be True.
        clusters: An optional vector that assigns each sequence to a cluster.
            If provided, the evolutionary time is learned per cluster.
        alphabet_size: The size of the alphabet underlying the substitution
            models.
        time_reversed: If False, the layer returns the conditional distribution
            P(S_i^tau | S_i^0; Q) of the amino acid at position i after
            continuous time tau given rate matrix Q. If True, the layer returns
            P(S_i | A; tau, Q, pi) instead (how likely is the observation S_i
            if the substituted amino acid at the root is A after time tau given
            rate matrix Q and equilibrium distribution pi).
    """

    def __init__(
        self,
        heads: int,
        rates: int,
        input_tracks: int,
        equilibrium_init: tf.keras.initializers.Initializer,
        exchangeability_init: tf.keras.initializers.Initializer,
        rate_init: tf.keras.initializers.Initializer,
        trainable_distances: bool=True,
        trainable_rate_matrices: bool=False,
        clusters: np.ndarray|None=None,
        alphabet_size: int=20,
        time_reversed: bool=False,
        **kwargs
    ):
        super(AncProbsLayer, self).__init__(**kwargs)
        self.heads = heads
        self.rates = rates
        self.input_tracks = input_tracks
        self.rate_init = rate_init
        self.equilibrium_init = equilibrium_init
        self.exchangeability_init = exchangeability_init
        self.trainable_distances = trainable_distances
        self.trainable_rate_matrices = trainable_rate_matrices
        self.clusters = clusters
        self.alphabet_size = alphabet_size
        self.time_reversed = time_reversed
        if clusters is None:
            self.num_clusters = self.rates
        else:
            self.num_clusters = np.max(clusters) + 1
        self._head_subset = None
        self._gtr_decomp = None
        if self.trainable_rate_matrices and not self.time_reversed:
            raise ValueError(
                "If trainable_rate_matrices is True, time_reversed must also" \
                "be True. Otherwise no meaningful model can be learned, since " \
                "Q can arbitrarily change residues to maximize HMM likelihood."
            )

    @property
    def head_subset(self):
        """If set, only these models are used in computations."""
        return self._head_subset

    @head_subset.setter
    def head_subset(self, subset):
        self._head_subset = subset
        # Recompute GTR decomposition if needed
        if self.built:
            self._precompute_gtr_decomposition()

    def build(self, input_shape=None):
        if self.built:
            return

        self.tau_kernel = self.add_weight(
            shape=[self.num_clusters, self.heads, self.input_tracks],
            name="tau_kernel",
            initializer=self.rate_init,
            trainable=self.trainable_distances,
        )

        self.exchangeability_kernel = self.add_weight(
            shape=[
                self.heads,
                self.input_tracks,
                self.alphabet_size,
                self.alphabet_size
            ],
            name="exchangeability_kernel",
            initializer=self.exchangeability_init,
            trainable=self.trainable_rate_matrices
        )

        self.equilibrium_kernel = self.add_weight(
            shape=[self.heads, self.input_tracks, self.alphabet_size],
            name="equilibrium_kernel",
            initializer=self.equilibrium_init,
            trainable=self.trainable_rate_matrices
        )

        self._precompute_gtr_decomposition()

        self.built = True

    def _precompute_gtr_decomposition(self) -> None:
        """Precompute GTR eigendecomposition for non-trainable rate matrices.
        Stores the result as a constant tensor to optimize computation.
        """
        if self.trainable_rate_matrices:
            self._gtr_decomp = None
            return
        # Compute rate matrices
        R, p = self.make_R(), self.make_p()

        if self._head_subset is None:
            heads = self.heads
        else:
            heads = len(self._head_subset)

        R_flat = tf.reshape(R, (-1, 20, 20))
        p_flat = tf.reshape(p, (-1, 20))
        Q = backend.make_rate_matrix(R_flat, p_flat)
        Q = tf.reshape(
            Q,
            (heads, self.input_tracks, self.alphabet_size, self.alphabet_size)
        )

        # Add batch dimension for compatibility with _compute_anc_probs
        Q_exp = tf.expand_dims(Q, 0)  # (1, heads, input_tracks, a, a)
        p_exp = tf.expand_dims(p, 0)  # (1, heads, input_tracks, a)

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
        """Computes the exchangeability matrices R for all models.

        Returns:
            A tensor of shape (H, I, D, D).
        """
        if kernel is None:
            kernel = self.exchangeability_kernel
            if self._head_subset is not None:
                kernel = tf.gather(kernel, self._head_subset, axis=0)
        return backend.make_symmetric_pos_semidefinite(kernel)

    def make_p(self) -> tf.Tensor:
        """Computes the equilibrium distributions p for all models.

        Returns:
            A tensor of shape (H, I, D).
        """
        kernel = self.equilibrium_kernel
        if self._head_subset is not None:
            kernel = tf.gather(kernel, self._head_subset, axis=0)
        return backend.make_equilibrium(kernel)

    def make_Q(self) -> tf.Tensor:
        """Computes the rate matrices Q for all models.

        Returns:
            A tensor of shape (H, I, D, D).
        """
        R, p = self.make_R(), self.make_p()
        Q = backend.make_rate_matrix(R, p)
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
            A tensor of shape (B, H, I) containing the evolutionary times for
            the specified subset of sequences.
        """
        tau = self.tau_kernel # (num_clusters, H, I)

        if self._head_subset is not None:
            tau = tf.gather(tau, self._head_subset, axis=1)

        if self.clusters is not None:
            tau = tf.gather(tau, self.clusters, axis=0)

        if subset is not None:
            B, H = tf.unstack(tf.shape(subset))
            h_range = tf.range(H, dtype=subset.dtype)[tf.newaxis, :]
            h_indices = tf.tile(h_range, [B, 1])  # (B, H)
            nd_indices = tf.stack([subset, h_indices], axis=-1)  # (B, H, 2)
            tau = tf.gather_nd(tau, nd_indices)  # (B, H, I)

        # Clamp kernel to prevent NaN during training.
        tau = tf.clip_by_value(tau, -80.0, 80.0)

        return backend.make_branch_lengths(tau)

    def make_P(self, tau: tf.Tensor) -> tf.Tensor:
        """Computes the transition matrices P for all models and
        sequences given the evolutionary times tau. Takes the time_reversed
        argument of the layer into account.

        Args:
            tau: A tensor of shape (B, H, I) containing the evolutionary times.

        Returns:
            A tensor of shape (B, H, I, D, D) containing the transition
            matrices for all models and sequences.
        """
        # Compute probability matrices
        if self._gtr_decomp is not None:
            # (B, H, I, D, D)
            P = expm_gtr_from_decomp(self._gtr_decomp, tau)
        else:
            R = self.make_R()  # (H, I, D, D)
            p = self.make_p()  # (H, I, D)
            Q = backend.make_rate_matrix(R, p)
            Q = tf.expand_dims(Q, 0)  # Add batch dimension (1, H, I, D, D)

            # (B, H, I, D, D)
            P = expm_gtr(Q, tau, p)
        if self.time_reversed:
            # Transpose the last two dimensions to reverse time direction
            P = tf.transpose(P, perm=[0, 1, 2, 4, 3])
        return P

    def _compute_anc_probs(
        self, sequences: tf.Tensor, tau: tf.Tensor
    ) -> tf.Tensor:
        """Computes ancestral probabilities simultaneously for all sites and
        rate matrices.

        Args:
            sequences (Tensor): Sequences in one-hot vector
                format. Shape: (B, L, H*, I, D)
                Currently all sequence must share the same alphabet size D.
            tau: Evolutionary times. Shape: (B, H, I)

        Returns:
            Ancestral probabilities. Shape: (B, L, H, I, D)
        """
        P = self.make_P(tau)  # (B, H, I, D, D)

        # Compute ancestral probabilities using einsum
        # Sequences might require broadcasting to the number of heads
        if sequences.shape[2] == 1:
            ancprobs = tf.einsum(
                "BLID,BHIDZ->BLHIZ", sequences[:, :, 0, :, :], P
            )
        else:
            ancprobs = tf.einsum("BLHID,BHIDZ->BLHIZ", sequences, P)

        return ancprobs

    def call(
        self,
        *sequences: tf.Tensor,
        rate_indices: tf.Tensor,
    ) -> tf.Tensor | tuple[tf.Tensor, ...]:
        """Computes ancestral probabilities of the inputs.

        Args:
            sequences (Tensor or tuple of Tensors): Sequences in one-hot vector
                format. Shape: ``(B, L, H, D_i)`` where ``D_i`` is the dimension of
                the input sequence and must be at least as large as the alphabet
                size.
            rate_indices: Indices that map each input sequences to an
                evolutionary time. Shape: ``(B, H)``

        Returns:
            Ancestral probabilities (tuple of Tensors).
                Shapes: ``(B, L, H, D_i)``
        """
        assert len(sequences) == self.input_tracks,\
            "Expected {} input sequences, got {}."\
            .format(self.input_tracks, len(sequences))

        tau = self.make_tau(rate_indices)  # (B, H, I)

        # Handle any special input tokens beyond the standard alphabet
        inputs = []
        adds = []
        for input in sequences:
            d = input.shape[-1]
            assert d is not None and d >= self.alphabet_size,\
                    "Input sequences must have at least {} channels for the "\
                    "standard alphabet, got {}.".format(self.alphabet_size, d)
            inputs.append(input[:, :, :, :self.alphabet_size])
            adds.append(input[:, :, :, self.alphabet_size:])

        # Compute ancestral probabilities (B, L, H, I, D)
        anc_probs = self._compute_anc_probs(tf.stack(inputs, axis=3), tau=tau)

        extended_anc_probs = []
        for i, add in enumerate(adds):
            anc_probs_i = anc_probs[:, :, :, i, :]
            if add.shape[2] == 1:
                add = tf.repeat(add, tf.shape(anc_probs_i)[2], axis=2)
            extended_anc_probs.append(tf.concat([anc_probs_i, add], axis=-1))

        if self.input_tracks == 1:
            return extended_anc_probs[0]
        else:
            return tuple(extended_anc_probs)

    def get_config(self):
        config = super(AncProbsLayer, self).get_config()
        config.update({
            "heads": self.heads,
            "rates": self.rates,
            "input_tracks": self.input_tracks,
            "equilibrium_init": initializer.ConstantInitializer(self.equilibrium_kernel.numpy()),
            "exchangeability_init": initializer.ConstantInitializer(self.exchangeability_kernel.numpy()),
            "rate_init": initializer.ConstantInitializer(self.tau_kernel.numpy()),
            "trainable_distances": self.trainable_distances,
            "clusters": self.clusters,
            "alphabet_size": self.alphabet_size,
        })
        return config

    @classmethod
    def from_config(cls, config):
        config["clusters"] = deserialize(config["clusters"])
        return cls(**config)
