import numpy as np
import torch

from learnMSA.tree.initializer import InitSpec
from learnMSA.tree.torch.initializer import T_TorchInitializer
from learnMSA.tree.torch.substitution_model import TorchSubstitutionModel
from learnMSA.tree.torch.tree_model import TorchTreeModel

T_Init = InitSpec | T_TorchInitializer


class TorchAncProbsLayer(torch.nn.Module):
    """A learnable layer for ancestral probabilities.

    It learns per-sequence continuous evolutionary times and computes
    substitution probabilities under a fixed substitution model.

    Notes: Provides parameters for detailed control of what parameters
    of the evolutionary model can be trained or not. If nothing except the
    per-sequence rates are trained, the GTR eigendecomposition is
    precomputed for faster training.

    Args:
        heads: The number of independently trained models. The layer will
            create a separate rate matrix for each head.
        rates: The number of evolutionary times that will be assigned
            using the indices passed to the forward method.
        input_tracks: The number of input tracks. The layer will create
            a separate rate matrix for each track.
        equilibrium_init: Initializer for the equilibrium distribution of the
            rate matrices.
        exchangeability_init: Initializer for the fixed base exchangeability
            matrices (exchangeability_const). These values are stored as a
            non-trainable constant and are never updated during training.
            Usually inverse_softplus should be used on the initial matrix by
            the user.
        rate_init: Initializer for the per-sequence rates.
        exchangeability_delta_init: Initializer for the learnable delta added
            on top of the fixed exchangeability_const. Defaults to zeros so
            that training starts from the fixed base matrix.
        mixture_init: Initializer for the mixture logit weights of shape
            (H, I, K). Defaults to RandomNormal(stddev=0.1) to break symmetry
            while keeping initial weights close to uniform.
        scale_init: Initializer for the per-component rate matrix scale factors
            of shape (H, I, K). Defaults so that initial scale is 1.0.
        tau_track_init: Initializer for the per-head and per-track conversion
            rate kernels of shape (H, I). Only used when input_tracks > 1.
            Defaults so that initial conversion rate is 1.0.
        trainable_equilibrium: Flag that controls whether the equilibrium
            distributions are trainable.
        trainable_exchangeabilities: Flag that controls whether
            exchangeability_delta_kernel is trainable.
        exchangeability_l2: L2 regularization strength applied to
            exchangeability_delta_kernel. Defaults to 0.0 (no regularization).
        trainable_rates: Flag that can prevent learning the evolutionary times.
        trainable_scale: Flag that can prevent learning the per-component rate
            matrix scale factors.
        shared_equilibrium: If True, all mixture components share a single
            equilibrium distribution.
        shared_exchangeabilities: If True, all mixture components share a
            single exchangeability matrix.
        clusters: An optional vector that assigns each sequence to a cluster.
            If provided, the evolutionary time is learned per cluster.
        alphabet_size: The size of the alphabet underlying the substitution
            models.
        time_reversed: If False, the layer returns the conditional distribution
            P(S_i^tau | S_i^0; Q) of the amino acid at position i after
            continuous time tau given rate matrix Q. If True, the layer returns
            P(S_i | A; tau, Q, pi) instead.
        num_components: The number of mixture components K. If > 1, the layer
            learns a mixture of K independent GTR models per head and track,
            sharing branch lengths across components.
        low_rank: If not None, the rank of the low-rank parameterization of the
            exchangeability matrices. If None, full kernels are used.
    """

    substitution_model: TorchSubstitutionModel

    tree_model: TorchTreeModel

    def __init__(
        self,
        heads: int,
        rates: int,
        input_tracks: int,
        equilibrium_init: T_Init,
        exchangeability_init: T_Init,
        rate_init: T_Init,
        exchangeability_delta_init: T_Init | None = None,
        mixture_init: T_Init | None = None,
        scale_init: T_Init | None = None,
        tau_track_init: T_Init | None = None,
        trainable_equilibrium: bool = False,
        trainable_exchangeabilities: bool = False,
        exchangeability_l2: float = 0.0,
        trainable_rates: bool = True,
        trainable_scale: bool = True,
        shared_equilibrium: bool = True,
        shared_exchangeabilities: bool = False,
        clusters: np.ndarray | None = None,
        alphabet_size: int = 20,
        time_reversed: bool = True,
        num_components: int = 1,
        low_rank: int | None = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.heads = heads
        self.rates = rates
        self.input_tracks = input_tracks
        self.alphabet_size = alphabet_size
        self.num_components = num_components
        if (trainable_exchangeabilities or trainable_equilibrium) \
                and not time_reversed:
            raise ValueError(
                "If trainable_exchangeabilities or trainable_equilibrium is "
                "True, time_reversed must also be True. Otherwise no "
                "meaningful model can be learned, since Q can arbitrarily "
                "change residues to maximize HMM likelihood."
            )

        # The sub-models turn backend-neutral init specs into torch
        # initializers themselves, so both forms are accepted here.
        self.substitution_model = TorchSubstitutionModel(
            heads=heads,
            input_tracks=input_tracks,
            alphabet_size=alphabet_size,
            equilibrium_init=equilibrium_init,
            exchangeability_init=exchangeability_init,
            exchangeability_delta_init=exchangeability_delta_init,
            mixture_init=mixture_init,
            scale_init=scale_init,
            trainable_equilibrium=trainable_equilibrium,
            trainable_exchangeabilities=trainable_exchangeabilities,
            trainable_scale=trainable_scale,
            shared_equilibrium=shared_equilibrium,
            shared_exchangeabilities=shared_exchangeabilities,
            exchangeability_l2=exchangeability_l2,
            time_reversed=time_reversed,
            num_components=num_components,
            lora=low_rank,
        )
        self.tree_model = TorchTreeModel(
            heads=heads,
            rates=rates,
            input_tracks=input_tracks,
            rate_init=rate_init,
            tau_track_init=tau_track_init,
            trainable_rates=trainable_rates,
            clusters=clusters,
        )
        self.built = False

    @property
    def head_subset(self):
        """If set, only these models are used in computations."""
        return self.substitution_model.head_subset

    @head_subset.setter
    def head_subset(self, subset) -> None:
        self.substitution_model.head_subset = subset
        self.tree_model.head_subset = subset

    # Forwarding properties for uniform weight access
    @property
    def exchangeability_const(self):
        return self.substitution_model.exchangeability_const

    @property
    def exchangeability_delta_kernel(self):
        return self.substitution_model.exchangeability_delta_kernel

    @property
    def equilibrium_kernel(self):
        return self.substitution_model.equilibrium_kernel

    @property
    def mixture_kernel(self):
        return self.substitution_model.mixture_kernel

    @property
    def scale_kernel(self):
        return self.substitution_model.scale_kernel

    @property
    def tau_kernel(self):
        return self.tree_model.tau_kernel

    @property
    def tau_track_kernel(self):
        return self.tree_model.tau_track_kernel

    def build(self, input_shape=None) -> None:
        if self.built:
            return
        self.substitution_model.build()
        self.tree_model.build()
        self.built = True

    def regularization_loss(self) -> torch.Tensor:
        """The L2 penalties that Keras would collect into ``Model.losses``."""
        return self.substitution_model.regularization_loss()

    def make_R(self, kernel: torch.Tensor | None = None) -> torch.Tensor:
        """The exchangeability matrices R, of shape (H, I, K, D, D)."""
        return self.substitution_model.make_R(kernel)

    def make_p(self) -> torch.Tensor:
        """The equilibrium distributions p, of shape (H, I, K, D)."""
        return self.substitution_model.make_p()

    def make_w(self) -> torch.Tensor:
        """The mixture weights, of shape (H, I, K)."""
        return self.substitution_model.make_w()

    def make_scale(self) -> torch.Tensor:
        """The per-component rate matrix scale factors, of shape (H, I, K)."""
        return self.substitution_model.make_scale()

    def make_Q(self) -> torch.Tensor:
        """The rate matrices Q, of shape (H, I, K, D, D)."""
        return self.substitution_model.make_Q()

    def make_P(self, tau: torch.Tensor) -> torch.Tensor:
        """The transition probability matrices, of shape (B, H, I, D, D)."""
        return self.substitution_model.make_P(tau)

    def make_tau(self, subset: torch.Tensor | None = None) -> torch.Tensor:
        """The evolutionary times, of shape (B, H, I)."""
        return self.tree_model.make_tau(subset)

    def _compute_anc_probs(
        self, sequences: torch.Tensor, tau: torch.Tensor
    ) -> torch.Tensor:
        """Computes ancestral probabilities simultaneously for all sites and
        rate matrices.

        Args:
            sequences: Sequences in one-hot vector format.
                Shape: (B, L, H*, I, D)
            tau: Evolutionary times. Shape: (B, H, I)

        Returns:
            Ancestral probabilities. Shape: (B, L, H, I, D)
        """
        P = self.make_P(tau)  # (B, H, I, D, D)
        if sequences.shape[2] == 1:
            return torch.einsum(
                "BLID,BHIDZ->BLHIZ", sequences[:, :, 0, :, :], P
            )
        return torch.einsum("BLHID,BHIDZ->BLHIZ", sequences, P)

    def forward(
        self,
        *sequences: torch.Tensor,
        rate_indices: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Computes ancestral probabilities of the inputs.

        Args:
            sequences: Sequences in one-hot vector format.
                Shape: ``(B, L, H, D_i)`` where ``D_i`` is the dimension of the
                input sequence and must be at least as large as the alphabet
                size.
            rate_indices: Indices that map each input sequence to an
                evolutionary time. Shape: ``(B, H)``

        Returns:
            Ancestral probabilities (tuple of Tensors).
                Shapes: ``(B, L, H, D_i)``
        """
        assert len(sequences) == self.input_tracks, \
            "Expected {} input sequences, got {}.".format(
                self.input_tracks, len(sequences)
            )

        tau = self.make_tau(rate_indices)  # (B, H, I)

        # Handle any special input tokens beyond the standard alphabet
        inputs, adds = [], []
        for seq in sequences:
            d = seq.shape[-1]
            assert d is not None and d >= self.alphabet_size, \
                "Input sequences must have at least {} channels for the " \
                "standard alphabet, got {}.".format(self.alphabet_size, d)
            inputs.append(seq[:, :, :, :self.alphabet_size])
            adds.append(seq[:, :, :, self.alphabet_size:])

        # Compute ancestral probabilities (B, L, H, I, D)
        anc_probs = self._compute_anc_probs(
            torch.stack(inputs, dim=3), tau=tau
        )

        extended_anc_probs = []
        for i, add in enumerate(adds):
            anc_probs_i = anc_probs[:, :, :, i, :]
            if add.shape[2] == 1:
                add = add.expand(-1, -1, anc_probs_i.shape[2], -1)
            extended_anc_probs.append(
                torch.cat([anc_probs_i, add], dim=-1)
            )

        if self.input_tracks == 1:
            return extended_anc_probs[0]
        return tuple(extended_anc_probs)
