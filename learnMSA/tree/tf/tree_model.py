
import tensorflow as tf
import numpy as np

from evoten.backend_tf import BackendTF

backend = BackendTF()


class TreeModel(tf.keras.layers.Layer):
    """
    A utility class for handling parametrization of the branches of an
    evolutionary tree.

    Args:
        heads: The number of independently trained models. The layer will create
            a separate rate matrix for each head.
        rates: The number of evolutionary times that will be assigned
            using the indices passed to make_tau.
        input_tracks: The number of input tracks. The layer will create
            a separate rate matrix for each track.
        rate_init: Initializer for the per-sequence rates.
        tau_track_init: Initializer for the per-head and per-track conversion rate
            kernels of shape (H, I). Only used when input_tracks > 1.
            Defaults so that initial conversion rate is 1.0.
        trainable_rates: Flag that can prevent learning the evolutionary
            times.
        clusters: An optional vector that assigns each sequence to a cluster.
            If provided, the evolutionary time is learned per cluster.
    """

    def __init__(
        self,
        heads: int,
        rates: int,
        input_tracks: int,
        rate_init: tf.keras.initializers.Initializer,
        tau_track_init: tf.keras.initializers.Initializer | None = None,
        trainable_rates: bool = True,
        clusters: np.ndarray | None = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.heads = heads
        self.rates = rates
        self.input_tracks = input_tracks
        self.rate_init = rate_init
        # inverse_softplus(1.0) → initial conversion rate is 1.0 (neutral)
        self.tau_track_init = (
            tf.keras.initializers.Constant(np.log(np.exp(1.0) - 1.0))
            if tau_track_init is None else tau_track_init
        )
        self.trainable_rates = trainable_rates
        self.clusters = clusters
        self.num_clusters = rates if clusters is None else np.max(clusters) + 1
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
        self.built = True

    def make_tau(self, subset: tf.Tensor | None = None) -> tf.Tensor:
        """Computes the evolutionary times (tau) for each sequence.

        Args:
            subset: An optional tensor of shape (B, H) selecting a subset of
                sequences. If None, computes tau for all sequences.

        Returns:
            A tensor of shape (B, H, I) containing the evolutionary times.
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