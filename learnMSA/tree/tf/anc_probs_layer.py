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

        Notes: Provides parameters for detailed control of what parameters
        of the evolutionary model can be trained or not. If nothing except the
        per-sequence rates are trained, the GTR eigendecomposition is
        precomputed for faster training.

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
        rate_init: Initializer for the per-sequence rates.
        mixture_init: Initializer for the mixture logit weights of shape
            (num_clusters, H, I, K) when input_dependent_weights=True, else
            (H, I, K). Defaults to RandomNormal(stddev=0.1) to break symmetry
            while keeping initial weights close to uniform.
        scale_init: Initializer for the per-component rate matrix scale factors
            of shape (H, I, K). Defaults so that initial scale is 1.0.
        tau_track_init: Initializer for the per-head and per-track conversion rate
            kernels of shape (H, I). Only used when input_tracks > 1.
            Defaults so that initial conversion rate is 1.0.
        trainable_equilibrium: Flag that controls whether the equilibrium
            distributions are trainable.
        trainable_exchangeabilities: Flag that controls whether the
            exchangeability matrices are trainable.
        trainable_rates: Flag that can prevent learning the evolutionary
            times.
        trainable_scale: Flag that can prevent learning the per-component rate
            matrix scale factors.
        shared_equilibrium: If True, all mixture components share a single
            equilibrium distribution.
        shared_exchangeabilities: If True, all mixture components share a single
            exchangeability matrix.
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
        num_components: The number of mixture components K. If > 1, the layer
            learns a mixture of K independent GTR models per head and track,
            sharing branch lengths across components.
        input_dependent_weights: If True, the mixture weights depend
            on the sequence cluster/rate index. The mixture_kernel gains a
            leading num_clusters dimension and make_P requires a subset
            argument. If False, all sequences share the same mixture weights.
    """

    def __init__(
        self,
        heads: int,
        rates: int,
        input_tracks: int,
        equilibrium_init: tf.keras.initializers.Initializer,
        exchangeability_init: tf.keras.initializers.Initializer,
        rate_init: tf.keras.initializers.Initializer,
        mixture_init: tf.keras.initializers.Initializer|None=None,
        scale_init: tf.keras.initializers.Initializer|None=None,
        tau_track_init: tf.keras.initializers.Initializer|None=None,
        trainable_equilibrium: bool=False,
        trainable_exchangeabilities: bool=False,
        trainable_rates: bool=True,
        trainable_scale: bool=True,
        shared_equilibrium: bool=True,
        shared_exchangeabilities: bool=True,
        clusters: np.ndarray|None=None,
        alphabet_size: int=20,
        time_reversed: bool=False,
        num_components: int=1,
        input_dependent_weights: bool=False,
        **kwargs
    ):
        super(AncProbsLayer, self).__init__(**kwargs)
        self.heads = heads
        self.rates = rates
        self.input_tracks = input_tracks

        self.rate_init = rate_init
        self.equilibrium_init = equilibrium_init
        self.exchangeability_init = exchangeability_init

        self.trainable_equilibrium = trainable_equilibrium
        self.trainable_exchangeabilities = trainable_exchangeabilities
        self.trainable_rates = trainable_rates
        self.trainable_scale = trainable_scale

        self.shared_equilibrium = shared_equilibrium
        self.shared_exchangeabilities = shared_exchangeabilities
        self.clusters = clusters
        self.alphabet_size = alphabet_size
        self.time_reversed = time_reversed
        self.num_components = num_components
        self.input_dependent_weights = input_dependent_weights
        if mixture_init is None:
            self.mixture_init = tf.keras.initializers.RandomNormal(stddev=0.1)
        else:
            self.mixture_init = mixture_init
        if scale_init is None:
            # inverse_softplus(1.0) = log(exp(1) - 1) ≈ 0.5413 → softplus gives 1.0 (neutral)
            self.scale_init = tf.keras.initializers.Constant(
                np.log(np.exp(1.0) - 1.0)
            )
        else:
            self.scale_init = scale_init
        if tau_track_init is None:
            # inverse_softplus(1.0) → initial conversion rate is 1.0 (neutral)
            self.tau_track_init = tf.keras.initializers.Constant(
                np.log(np.exp(1.0) - 1.0)
            )
        else:
            self.tau_track_init = tau_track_init
        if clusters is None:
            self.num_clusters = self.rates
        else:
            self.num_clusters = np.max(clusters) + 1
        self._head_subset = None
        self._gtr_decomp = None
        if (self.trainable_exchangeabilities or self.trainable_equilibrium) \
                and not self.time_reversed:
            raise ValueError(
                "If trainable_exchangeabilities or trainable_equilibrium is "
                "True, time_reversed must also be True. Otherwise no meaningful "
                "model can be learned, since Q can arbitrarily change residues "
                "to maximize HMM likelihood."
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
            shape=[self.num_clusters, self.heads],
            name="tau_kernel",
            initializer=self.rate_init,
            trainable=self.trainable_rates,
        )

        if self.input_tracks > 1:
            self.tau_track_kernel = self.add_weight(
                shape=[self.heads, self.input_tracks],
                name="tau_track_kernel",
                initializer=self.tau_track_init,
                trainable=self.trainable_rates,
            )

        self.exchangeability_kernel = self.add_weight(
            shape=[
                self.heads,
                self.input_tracks,
                1 if self.shared_exchangeabilities else self.num_components,
                self.alphabet_size,
                self.alphabet_size
            ],
            name="exchangeability_kernel",
            initializer=self.exchangeability_init,
            trainable=self.trainable_exchangeabilities
        )

        self.equilibrium_kernel = self.add_weight(
            shape=[
                self.heads,
                self.input_tracks,
                1 if self.shared_equilibrium else self.num_components,
                self.alphabet_size
            ],
            name="equilibrium_kernel",
            initializer=self.equilibrium_init,
            trainable=self.trainable_equilibrium
        )

        if self.num_components > 1:
            trainable_mixture = (self.trainable_exchangeabilities
                or self.trainable_scale or self.trainable_equilibrium)
            self.mixture_kernel = self.add_weight(
                shape=(
                    [self.num_clusters, self.heads, self.input_tracks, self.num_components]
                    if self.input_dependent_weights
                    else [self.heads, self.input_tracks, self.num_components]
                ),
                name="mixture_kernel",
                initializer=self.mixture_init,
                trainable=trainable_mixture,
                regularizer=tf.keras.regularizers.L2(5e-5 / self.heads)
            )
            self.scale_kernel = self.add_weight(
                shape=[self.heads, self.input_tracks, self.num_components],
                name="scale_kernel",
                initializer=self.scale_init,
                trainable=self.trainable_scale,
            )

        self._precompute_gtr_decomposition()

        self.built = True

    def _precompute_gtr_decomposition(self) -> None:
        """Precompute GTR eigendecomposition for non-trainable rate matrices.
        Stores the result as a constant tensor to optimize computation.
        """
        if self.trainable_exchangeabilities or self.trainable_equilibrium:
            self._gtr_decomp = None
            return
        # Compute rate matrices
        R, p = self.make_R(), self.make_p()

        if self._head_subset is None:
            heads = self.heads
        else:
            heads = len(self._head_subset)

        IK = self.input_tracks * self.num_components
        R_flat = tf.reshape(R, (-1, self.alphabet_size, self.alphabet_size))
        p_flat = tf.reshape(p, (-1, self.alphabet_size))
        Q = backend.make_rate_matrix(R_flat, p_flat)
        Q = tf.reshape(
            Q,
            (heads, IK, self.alphabet_size, self.alphabet_size)
        )
        p_IK = tf.reshape(p, (heads, IK, self.alphabet_size))

        # Add batch dimension for compatibility with _compute_anc_probs
        Q_exp = tf.expand_dims(Q, 0)  # (1, heads, input_tracks*K, a, a)
        p_exp = tf.expand_dims(p_IK, 0)  # (1, heads, input_tracks*K, a)

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
            A tensor of shape (H, I, K, D, D).
        """
        if kernel is None:
            kernel = self.exchangeability_kernel
            if self._head_subset is not None:
                kernel = tf.gather(kernel, self._head_subset, axis=0)
        R = backend.make_symmetric_pos_semidefinite(kernel)
        if self.shared_exchangeabilities and self.num_components > 1:
            R = tf.tile(R, [1, 1, self.num_components, 1, 1])
        return R

    def make_p(self) -> tf.Tensor:
        """Computes the equilibrium distributions p for all models.

        Returns:
            A tensor of shape (H, I, K, D).
        """
        kernel = self.equilibrium_kernel
        if self._head_subset is not None:
            kernel = tf.gather(kernel, self._head_subset, axis=0)
        p = backend.make_equilibrium(kernel)
        if self.shared_equilibrium and self.num_components > 1:
            p = tf.tile(p, [1, 1, self.num_components, 1])
        return p

    def make_w(self, subset: tf.Tensor | None = None) -> tf.Tensor:
        """Computes the mixture weights for all models.

        Args:
            subset: An optional tensor of shape (B, H) that specifies a
                    subset of sequences to compute weights for.
                    If None and input_dependent_weights=True, returns weights
                    indexed by cluster/rate without selecting a batch.

        Returns:
            A tensor of shape (B, H, I, K) when subset is provided and
            input_dependent_weights=True, else (H, I, K).
            All tensors sum to 1 along the K axis.
        """
        if self.num_components == 1:
            if self._head_subset is None:
                H_active = self.heads
            else:
                H_active = len(self._head_subset)
            return tf.ones((H_active, self.input_tracks, 1))
        kernel = self.mixture_kernel
        if not self.input_dependent_weights:
            if self._head_subset is not None:
                kernel = tf.gather(kernel, self._head_subset, axis=0)
            return tf.nn.softmax(kernel, axis=-1)
        # input_dependent_weights=True: kernel shape (num_clusters, H, I, K)
        if self._head_subset is not None:
            kernel = tf.gather(kernel, self._head_subset, axis=1)
        if self.clusters is not None:
            kernel = tf.gather(kernel, self.clusters, axis=0)
        if subset is not None:
            B, H = tf.unstack(tf.shape(subset))
            h_range = tf.range(H, dtype=subset.dtype)[tf.newaxis, :]
            h_indices = tf.tile(h_range, [B, 1])  # (B, H)
            nd_indices = tf.stack([subset, h_indices], axis=-1)  # (B, H, 2)
            kernel = tf.gather_nd(kernel, nd_indices)  # (B, H, I, K)
        return tf.nn.softmax(kernel, axis=-1)

    def make_scale(self) -> tf.Tensor:
        """Computes the per-component rate matrix scale factors for all models.

        Returns:
            A tensor of shape (H, I, K) with positive values (via softplus).
        """
        if self.num_components == 1:
            if self._head_subset is None:
                H_active = self.heads
            else:
                H_active = len(self._head_subset)
            return tf.ones((H_active, self.input_tracks, 1))
        kernel = self.scale_kernel
        if self._head_subset is not None:
            kernel = tf.gather(kernel, self._head_subset, axis=0)
        return backend.make_branch_lengths(kernel)

    def make_Q(self) -> tf.Tensor:
        """Computes the rate matrices Q for all models and mixture components.

        Returns:
            A tensor of shape (H, I, K, D, D).
        """
        R, p = self.make_R(), self.make_p()  # (H, I, K, D, D), (H, I, K, D)
        R_flat = tf.reshape(R, (-1, self.alphabet_size, self.alphabet_size))
        p_flat = tf.reshape(p, (-1, self.alphabet_size))
        Q_flat = backend.make_rate_matrix(R_flat, p_flat)
        return tf.reshape(Q_flat, tf.shape(R))

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
        tau = self.tau_kernel  # (num_clusters, H)

        if self._head_subset is not None:
            tau = tf.gather(tau, self._head_subset, axis=1)

        if self.clusters is not None:
            tau = tf.gather(tau, self.clusters, axis=0)

        if subset is not None:
            B, H = tf.unstack(tf.shape(subset))
            h_range = tf.range(H, dtype=subset.dtype)[tf.newaxis, :]
            h_indices = tf.tile(h_range, [B, 1])  # (B, H)
            nd_indices = tf.stack([subset, h_indices], axis=-1)  # (B, H, 2)
            tau = tf.gather_nd(tau, nd_indices)  # (B, H)

        # Clamp kernel to prevent NaN during training.
        tau = tf.clip_by_value(tau, -80.0, 80.0)
        tau = backend.make_branch_lengths(tau)  # (..., H)

        if self.input_tracks > 1:
            # Apply cluster/data-independent per-head and per-track conversion rates
            track_kernel = self.tau_track_kernel  # (H, I)
            if self._head_subset is not None:
                track_kernel = tf.gather(track_kernel, self._head_subset, axis=0)
            conversion = backend.make_branch_lengths(track_kernel)  # (H_active, I)
            tau = tau[..., tf.newaxis] * conversion[tf.newaxis]  # (..., H, I)
        else:
            tau = tau[..., tf.newaxis]  # (..., H, 1)

        return tau

    def make_P(self, tau: tf.Tensor, subset: tf.Tensor | None = None) -> tf.Tensor:
        """Computes the transition matrices P for all models and
        sequences given the evolutionary times tau. Takes the time_reversed
        argument of the layer into account.

        Args:
            tau: A tensor of shape (B, H, I) containing the evolutionary times.
            subset: An optional tensor of shape (B, H) mapping each sequence
                    to a rate/cluster index. Required when
                    input_dependent_weights=True and num_components > 1.

        Returns:
            A tensor of shape (B, H, I, D, D) containing the transition
            matrices for all models and sequences.
        """
        if self.input_dependent_weights and self.num_components > 1 and subset is None:
            raise ValueError(
                "subset must be provided to make_P when "
                "input_dependent_weights=True and num_components > 1."
            )
        if self._head_subset is None:
            H_active = self.heads
        else:
            H_active = len(self._head_subset)
        IK = self.input_tracks * self.num_components

        # Expand tau from (B, H, I) to (B, H, I*K) by repeating for each component
        tau_expanded = tf.repeat(tau, self.num_components, axis=-1)  # (B, H, I*K)

        # Scale tau per component (equivalent to scaling Q_k by s_k independently of branch lengths)
        s = self.make_scale()  # (H_active, I, K)
        s_flat = tf.reshape(s, [H_active, IK])  # (H_active, I*K)
        tau_expanded = tau_expanded * s_flat[tf.newaxis]  # (B, H_active, I*K)

        # Compute probability matrices for all mixture components
        if self._gtr_decomp is not None:
            P_all = expm_gtr_from_decomp(self._gtr_decomp, tau_expanded)  # (B, H, I*K, D, D)
        else:
            R = self.make_R()  # (H, I, K, D, D)
            p = self.make_p()  # (H, I, K, D)
            R_flat = tf.reshape(R, (-1, self.alphabet_size, self.alphabet_size))
            p_flat = tf.reshape(p, (-1, self.alphabet_size))
            Q = backend.make_rate_matrix(R_flat, p_flat)  # (H*I*K, D, D)
            Q = tf.reshape(Q, (H_active, IK, self.alphabet_size, self.alphabet_size))
            Q = tf.expand_dims(Q, 0)  # (1, H, I*K, D, D)
            p_IK = tf.reshape(p, (H_active, IK, self.alphabet_size))
            p_IK = tf.expand_dims(p_IK, 0)  # (1, H, I*K, D)
            P_all = expm_gtr(Q, tau_expanded, p_IK)  # (B, H, I*K, D, D)

        # Reshape to (B, H, I, K, D, D) and compute the weighted mixture
        P_all = tf.reshape(
            P_all,
            [-1, H_active, self.input_tracks, self.num_components,
             self.alphabet_size, self.alphabet_size]
        )
        w = self.make_w(subset)  # (B, H, I, K) or (H, I, K)
        if self.input_dependent_weights and self.num_components > 1:
            w = w[:, :, :, :, tf.newaxis, tf.newaxis]  # (B, H, I, K, 1, 1)
        else:
            w = w[tf.newaxis, :, :, :, tf.newaxis, tf.newaxis]  # (1, H, I, K, 1, 1)
        P = tf.reduce_sum(w * P_all, axis=3)  # (B, H, I, D, D)

        if self.time_reversed:
            # Transpose the last two dimensions to reverse time direction
            P = tf.transpose(P, perm=[0, 1, 2, 4, 3])
        return P

    def _compute_anc_probs(
        self, sequences: tf.Tensor, tau: tf.Tensor,
        subset: tf.Tensor | None = None
    ) -> tf.Tensor:
        """Computes ancestral probabilities simultaneously for all sites and
        rate matrices.

        Args:
            sequences (Tensor): Sequences in one-hot vector
                format. Shape: (B, L, H*, I, D)
                Currently all sequence must share the same alphabet size D.
            tau: Evolutionary times. Shape: (B, H, I)
            subset: Optional rate/cluster indices of shape (B, H), forwarded
                    to make_P. Required when input_dependent_weights=True
                    and num_components > 1.

        Returns:
            Ancestral probabilities. Shape: (B, L, H, I, D)
        """
        P = self.make_P(tau, subset)  # (B, H, I, D, D)

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
        anc_probs = self._compute_anc_probs(
            tf.stack(inputs, axis=3), tau=tau, subset=rate_indices
        )

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
            "trainable_equilibrium": self.trainable_equilibrium,
            "trainable_exchangeabilities": self.trainable_exchangeabilities,
            "trainable_rates": self.trainable_rates,
            "trainable_scale": self.trainable_scale,
            "shared_equilibrium": self.shared_equilibrium,
            "shared_exchangeabilities": self.shared_exchangeabilities,
            "clusters": self.clusters,
            "alphabet_size": self.alphabet_size,
            "time_reversed": self.time_reversed,
            "num_components": self.num_components,
            "input_dependent_weights": self.input_dependent_weights,
        })
        if self.num_components > 1:
            config["mixture_init"] = initializer.ConstantInitializer(
                self.mixture_kernel.numpy()
            )
            config["scale_init"] = initializer.ConstantInitializer(
                self.scale_kernel.numpy()
            )
        if self.input_tracks > 1:
            config["tau_track_init"] = initializer.ConstantInitializer(
                self.tau_track_kernel.numpy()
            )
        return config

    @classmethod
    def from_config(cls, config):
        config["clusters"] = deserialize(config["clusters"])
        return cls(**config)
