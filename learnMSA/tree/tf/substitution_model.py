
import tensorflow as tf
import numpy as np

from evoten.backend_tf import BackendTF
from evoten.expm_gtr import expm_gtr, precompute_gtr, expm_gtr_from_decomp

backend = BackendTF()


class SubstitutionModel(tf.keras.layers.Layer):
    """
    A utility class for handling parametrization of substitution models for
    sequence evolution.

    Args:
        heads: The number of independently trained models. The layer will create
            a separate rate matrix for each head.
        input_tracks: The number of input tracks. The layer will create
            a separate rate matrix for each track.
        alphabet_size: The size of the alphabet underlying the substitution
            models.
        equilibrium_init: Initializer for the equilibrium distribution of the
            rate matrices
        exchangeability_init: Initializer for the fixed base exchangeability
            matrices (exchangeability_const). These values are stored as a
            non-trainable constant and are never updated during training.
            Usually inverse_softplus should be used on the initial matrix by
            the user.
        exchangeability_delta_init: Initializer for the learnable delta added
            on top of the fixed exchangeability_const. Defaults to zeros so
            that training starts from the fixed base matrix.
        mixture_init: Initializer for the mixture logit weights of shape
            (H, I, K). Defaults to RandomNormal(stddev=0.1) to break symmetry
            while keeping initial weights close to uniform.
        scale_init: Initializer for the per-component rate matrix scale factors
            of shape (H, I, K). Defaults so that initial scale is 1.0.
        trainable_equilibrium: Flag that controls whether the equilibrium
            distributions are trainable.
        trainable_exchangeabilities: Flag that controls whether
            exchangeability_delta_kernel is trainable (i.e. whether the
            learnable delta on top of the fixed base matrix is updated).
        trainable_scale: Flag that can prevent learning the per-component rate
            matrix scale factors.
        shared_equilibrium: If True, all mixture components share a single
            equilibrium distribution.
        shared_exchangeabilities: If True, all mixture components share a single
            exchangeability matrix.
        exchangeability_l2: L2 regularization strength applied to
            exchangeability_delta_kernel. Defaults to 0.0 (no regularization).
        time_reversed: If False, the layer returns the conditional distribution
            P(S_i^tau | S_i^0; Q) of the amino acid at position i after
            continuous time tau given rate matrix Q. If True, the layer returns
            P(S_i | A; tau, Q, pi) instead (how likely is the observation S_i
            if the substituted amino acid at the root is A after time tau given
            rate matrix Q and equilibrium distribution pi).
        num_components: The number of mixture components K. If > 1, the layer
            learns a mixture of K independent GTR models per head and track,
            sharing branch lengths across components.
    """

    def __init__(
        self,
        heads: int,
        input_tracks: int,
        alphabet_size: int,
        equilibrium_init: tf.keras.initializers.Initializer,
        exchangeability_init: tf.keras.initializers.Initializer,
        exchangeability_delta_init: tf.keras.initializers.Initializer | None = None,
        mixture_init: tf.keras.initializers.Initializer | None = None,
        scale_init: tf.keras.initializers.Initializer | None = None,
        trainable_equilibrium: bool = False,
        trainable_exchangeabilities: bool = False,
        trainable_scale: bool = True,
        shared_equilibrium: bool = True,
        shared_exchangeabilities: bool = True,
        exchangeability_l2: float = 0.0,
        time_reversed: bool = True,
        num_components: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.heads = heads
        self.input_tracks = input_tracks
        self.alphabet_size = alphabet_size
        self.equilibrium_init = equilibrium_init
        self.exchangeability_init = exchangeability_init
        self.exchangeability_delta_init = (
            tf.keras.initializers.Zeros()
            if exchangeability_delta_init is None else exchangeability_delta_init
        )
        self.mixture_init = (
            tf.keras.initializers.RandomNormal(stddev=0.1)
            if mixture_init is None else mixture_init
        )
        # inverse_softplus(1.0) = log(exp(1) - 1) ≈ 0.5413 → softplus gives 1.0 (neutral)
        self.scale_init = (
            tf.keras.initializers.Constant(np.log(np.exp(1.0) - 1.0))
            if scale_init is None else scale_init
        )
        self.trainable_equilibrium = trainable_equilibrium
        self.trainable_exchangeabilities = trainable_exchangeabilities
        self.trainable_scale = trainable_scale
        self.shared_equilibrium = shared_equilibrium
        self.shared_exchangeabilities = shared_exchangeabilities
        self.exchangeability_l2 = exchangeability_l2
        self.time_reversed = time_reversed
        self.num_components = num_components
        self._head_subset = None
        self._gtr_decomp = None

    @property
    def head_subset(self):
        """If set, only these models are used in computations."""
        return self._head_subset

    @head_subset.setter
    def head_subset(self, subset):
        self._head_subset = subset
        if self.built:
            self._precompute_gtr_decomposition()

    def build(self, input_shape=None):
        if self.built:
            return

        _exch_shape = [
            self.heads,
            self.input_tracks,
            1 if self.shared_exchangeabilities else self.num_components,
            self.alphabet_size,
            self.alphabet_size,
        ]
        self.exchangeability_const = tf.constant(
            self.exchangeability_init(shape=_exch_shape, dtype=self.dtype),
            name="exchangeability_const",
        )
        self.exchangeability_delta_kernel = self.add_weight(
            shape=_exch_shape,
            name="exchangeability_delta_kernel",
            initializer=self.exchangeability_delta_init,
            trainable=self.trainable_exchangeabilities,
            regularizer=(
                tf.keras.regularizers.L2(self.exchangeability_l2)
                if self.exchangeability_l2 > 0.0 else None
            ),
        )
        self.equilibrium_kernel = self.add_weight(
            shape=[
                self.heads,
                self.input_tracks,
                1 if self.shared_equilibrium else self.num_components,
                self.alphabet_size,
            ],
            name="equilibrium_kernel",
            initializer=self.equilibrium_init,
            trainable=self.trainable_equilibrium,
        )
        if self.num_components > 1:
            trainable_mixture = (
                self.trainable_exchangeabilities
                or self.trainable_scale
                or self.trainable_equilibrium
            )
            self.mixture_kernel = self.add_weight(
                shape=[self.heads, self.input_tracks, self.num_components],
                name="mixture_kernel",
                initializer=self.mixture_init,
                trainable=trainable_mixture,
                regularizer=tf.keras.regularizers.L2(5e-5 / self.heads),
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

        R, p = self.make_R(), self.make_p()
        H_active = self.heads if self._head_subset is None else len(self._head_subset)
        IK = self.input_tracks * self.num_components
        R_flat = tf.reshape(R, (-1, self.alphabet_size, self.alphabet_size))
        p_flat = tf.reshape(p, (-1, self.alphabet_size))
        Q = backend.make_rate_matrix(R_flat, p_flat)
        Q = tf.reshape(Q, (H_active, IK, self.alphabet_size, self.alphabet_size))
        p_IK = tf.reshape(p, (H_active, IK, self.alphabet_size))
        decomp = precompute_gtr(
            tf.expand_dims(Q, 0), tf.expand_dims(p_IK, 0)
        )
        self._gtr_decomp = type(decomp)(
            eigvals=tf.constant(decomp.eigvals.numpy()),
            eigvecs=tf.constant(decomp.eigvecs.numpy()),
            sqrt_pi=tf.constant(decomp.sqrt_pi.numpy()),
            inv_sqrt_pi=tf.constant(decomp.inv_sqrt_pi.numpy()),
        )

    def make_R(self, kernel: tf.Tensor | None = None) -> tf.Tensor:
        """Computes the exchangeability matrices R for all models.

        Returns:
            A tensor of shape (H, I, K, D, D).
        """
        if kernel is None:
            const = self.exchangeability_const
            delta = self.exchangeability_delta_kernel
            if self._head_subset is not None:
                const = tf.gather(const, self._head_subset, axis=0)
                delta = tf.gather(delta, self._head_subset, axis=0)
            kernel = const + delta
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

    def make_w(self) -> tf.Tensor:
        """Computes the mixture weights for all models.

        Returns:
            A tensor of shape (H, I, K) summing to 1 along the K axis.
        """
        H_active = self.heads if self._head_subset is None else len(self._head_subset)
        if self.num_components == 1:
            return tf.ones((H_active, self.input_tracks, 1))
        kernel = self.mixture_kernel
        if self._head_subset is not None:
            kernel = tf.gather(kernel, self._head_subset, axis=0)
        return tf.nn.softmax(kernel, axis=-1)

    def make_scale(self) -> tf.Tensor:
        """Computes the per-component rate matrix scale factors for all models.

        Returns:
            A tensor of shape (H, I, K) with positive values (via softplus).
        """
        H_active = self.heads if self._head_subset is None else len(self._head_subset)
        if self.num_components == 1:
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
        R, p = self.make_R(), self.make_p()
        R_flat = tf.reshape(R, (-1, self.alphabet_size, self.alphabet_size))
        p_flat = tf.reshape(p, (-1, self.alphabet_size))
        Q_flat = backend.make_rate_matrix(R_flat, p_flat)
        return tf.reshape(Q_flat, tf.shape(R))

    def make_P(self, tau: tf.Tensor) -> tf.Tensor:
        """Computes the transition probability matrices P given evolutionary times tau.

        Args:
            tau: A tensor of shape (B, H, I) containing the evolutionary times.

        Returns:
            A tensor of shape (B, H, I, D, D).
        """
        H_active = self.heads if self._head_subset is None else len(self._head_subset)
        IK = self.input_tracks * self.num_components

        # Expand tau from (B, H, I) to (B, H, I*K) and scale per component
        tau_expanded = tf.repeat(tau, self.num_components, axis=-1)  # (B, H, I*K)
        s_flat = tf.reshape(self.make_scale(), [H_active, IK])
        tau_expanded = tau_expanded * s_flat[tf.newaxis]  # (B, H, I*K)

        if self._gtr_decomp is not None:
            P_all = expm_gtr_from_decomp(self._gtr_decomp, tau_expanded)
        else:
            R, p = self.make_R(), self.make_p()
            R_flat = tf.reshape(R, (-1, self.alphabet_size, self.alphabet_size))
            p_flat = tf.reshape(p, (-1, self.alphabet_size))
            Q = backend.make_rate_matrix(R_flat, p_flat)
            Q = tf.expand_dims(
                tf.reshape(Q, (H_active, IK, self.alphabet_size, self.alphabet_size)), 0
            )
            p_IK = tf.expand_dims(
                tf.reshape(p, (H_active, IK, self.alphabet_size)), 0
            )
            P_all = expm_gtr(Q, tau_expanded, p_IK)

        # Reshape, apply mixture weights
        P_all = tf.reshape(
            P_all,
            [-1, H_active, self.input_tracks, self.num_components,
             self.alphabet_size, self.alphabet_size],
        )
        w = self.make_w()[tf.newaxis, :, :, :, tf.newaxis, tf.newaxis]
        P = tf.reduce_sum(w * P_all, axis=3)  # (B, H, I, D, D)

        if self.time_reversed:
            P = tf.transpose(P, perm=[0, 1, 2, 4, 3])
        return P