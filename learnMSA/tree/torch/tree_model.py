import numpy as np
import torch
from evoten import backend

from learnMSA.tree.initializer import Constant, InitSpec
from learnMSA.tree.torch.initializer import T_TorchInitializer, to_torch


class TorchTreeModel(torch.nn.Module):
    """
    A utility class for handling parametrization of the branches of an
    evolutionary tree.

    Args:
        heads: The number of independently trained models. The layer will
            create a separate rate matrix for each head.
        rates: The number of evolutionary times that will be assigned
            using the indices passed to make_tau.
        input_tracks: The number of input tracks. The layer will create
            a separate rate matrix for each track.
        rate_init: Initializer for the per-sequence rates.
        tau_track_init: Initializer for the per-head and per-track conversion
            rate kernels of shape (H, I). Only used when input_tracks > 1.
            Defaults so that initial conversion rate is 1.0.
        trainable_rates: Flag that can prevent learning the evolutionary times.
        clusters: An optional vector that assigns each sequence to a cluster.
            If provided, the evolutionary time is learned per cluster.
    """

    def __init__(
        self,
        heads: int,
        rates: int,
        input_tracks: int,
        rate_init: InitSpec | T_TorchInitializer,
        tau_track_init: InitSpec | T_TorchInitializer | None = None,
        trainable_rates: bool = True,
        clusters: np.ndarray | None = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.heads = heads
        self.rates = rates
        self.input_tracks = input_tracks
        self.rate_init = to_torch(rate_init)
        # inverse_softplus(1.0) -> initial conversion rate is 1.0 (neutral)
        self.tau_track_init = to_torch(
            tau_track_init if tau_track_init is not None
            else Constant(float(np.log(np.exp(1.0) - 1.0)))
        )
        self.trainable_rates = trainable_rates
        self.clusters = clusters
        self.num_clusters = (
            rates if clusters is None else int(np.max(clusters)) + 1
        )
        self._head_subset = None
        self.built = False

    @property
    def head_subset(self):
        """If set, only these models are used in computations."""
        return self._head_subset

    @head_subset.setter
    def head_subset(self, subset) -> None:
        self._head_subset = subset

    def build(self, input_shape=None) -> None:
        if self.built:
            return
        self.tau_kernel = torch.nn.Parameter(
            self.rate_init(torch.empty([self.num_clusters, self.heads])),
            requires_grad=self.trainable_rates,
        )
        if self.input_tracks > 1:
            self.tau_track_kernel = torch.nn.Parameter(
                self.tau_track_init(
                    torch.empty([self.heads, self.input_tracks])
                ),
                requires_grad=self.trainable_rates,
            )
        if self.clusters is not None:
            self.register_buffer(
                "_cluster_index",
                torch.as_tensor(np.asarray(self.clusters), dtype=torch.int64),
                persistent=False,
            )
        self.built = True

    def make_tau(self, subset: torch.Tensor | None = None) -> torch.Tensor:
        """Computes the evolutionary times (tau) for each sequence.

        Args:
            subset: An optional tensor of shape (B, H) selecting a subset of
                sequences. If None, computes tau for all sequences.

        Returns:
            A tensor of shape (B, H, I) containing the evolutionary times.
        """
        tau = self.tau_kernel  # (num_clusters, H)
        device = tau.device

        if self._head_subset is not None:
            tau = tau.index_select(1, self._head_index(device))

        if self.clusters is not None:
            tau = tau.index_select(0, self._cluster_index)

        if subset is not None:
            # Advanced indexing over (cluster, head) pairs, the equivalent of
            # gathering with the stacked (B, H, 2) index tensor.
            B, H = subset.shape
            h_indices = torch.arange(H, device=device).expand(B, H)
            tau = tau[subset.to(torch.int64), h_indices]  # (B, H)

        # Clamp kernel to prevent NaN during training.
        tau = tau.clamp(-80.0, 80.0)
        tau = backend.make_branch_lengths(tau)  # (..., H)

        if self.input_tracks > 1:
            # Apply cluster/data-independent per-head and per-track conversion
            # rates
            track_kernel = self.tau_track_kernel  # (H, I)
            if self._head_subset is not None:
                track_kernel = track_kernel.index_select(
                    0, self._head_index(device)
                )
            # (H_active, I)
            conversion = backend.make_branch_lengths(track_kernel)
            tau = tau[..., None] * conversion[None]  # (..., H, I)
        else:
            tau = tau[..., None]  # (..., H, 1)

        return tau

    def _head_index(self, device: torch.device) -> torch.Tensor:
        return torch.as_tensor(
            np.asarray(self._head_subset), dtype=torch.int64, device=device
        )
