import numpy as np
import torch
from evoten import backend
from evoten.expm_gtr_torch import (GTRDecomp, expm_gtr, expm_gtr_from_decomp,
                                   precompute_gtr)

from learnMSA.tree.initializer import InitSpec
from learnMSA.tree.torch.initializer import T_TorchInitializer, to_torch

#: Buffer names holding the cached GTR eigendecomposition, in the field order
#: of :class:`~evoten.expm_gtr_torch.GTRDecomp`.
_GTR_BUFFERS = (
    "_gtr_eigvals", "_gtr_eigvecs", "_gtr_sqrt_pi", "_gtr_inv_sqrt_pi",
)


class TorchSubstitutionModel(torch.nn.Module):
    """
    A utility class for handling parametrization of substitution models for
    sequence evolution.

    Args:
        heads: The number of independently trained models. The layer will
            create a separate rate matrix for each head.
        input_tracks: The number of input tracks. The layer will create
            a separate rate matrix for each track.
        alphabet_size: The size of the alphabet underlying the substitution
            models.
        equilibrium_init: Initializer for the equilibrium distribution of the
            rate matrices.
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
        shared_exchangeabilities: If True, all mixture components share a
            single exchangeability matrix.
        exchangeability_l2: L2 regularization strength applied to
            exchangeability_delta_kernel. Defaults to 0.0 (no regularization).
            Unlike Keras, torch has no regularizer hook on a parameter, so the
            penalty is returned by :meth:`regularization_loss` and added to the
            training loss by the model.
        time_reversed: If False, the layer returns the conditional distribution
            P(S_i^tau | S_i^0; Q) of the amino acid at position i after
            continuous time tau given rate matrix Q. If True, the layer returns
            P(S_i | A; tau, Q, pi) instead (how likely is the observation S_i
            if the substituted amino acid at the root is A after time tau given
            rate matrix Q and equilibrium distribution pi).
        num_components: The number of mixture components K. If > 1, the layer
            learns a mixture of K independent GTR models per head and track,
            sharing branch lengths across components.
        lora: If not None, the rank of the low-rank parameterization of the
            exchangeability matrices. If None, full kernels are used.
    """

    #: L2 coefficient of the mixture logits, matching the TensorFlow layer's
    #: hard-coded regularizer.
    MIXTURE_L2 = 5e-5

    def __init__(
        self,
        heads: int,
        input_tracks: int,
        alphabet_size: int,
        equilibrium_init: InitSpec | T_TorchInitializer,
        exchangeability_init: InitSpec | T_TorchInitializer,
        exchangeability_delta_init: InitSpec | T_TorchInitializer | None = None,
        mixture_init: InitSpec | T_TorchInitializer | None = None,
        scale_init: InitSpec | T_TorchInitializer | None = None,
        trainable_equilibrium: bool = False,
        trainable_exchangeabilities: bool = False,
        trainable_scale: bool = True,
        shared_equilibrium: bool = True,
        shared_exchangeabilities: bool = False,
        exchangeability_l2: float = 0.0,
        time_reversed: bool = True,
        num_components: int = 1,
        lora: int | None = None,
        **kwargs
    ) -> None:
        super().__init__()

        # assert if configuration is valid
        if lora is not None:
            assert trainable_exchangeabilities, \
                "Low-rank parameterization has no effect if " \
                "exchangeabilities are not trainable."
            assert (lora > 0 and lora < alphabet_size), \
                "lora must be between 0 and alphabet_size"

        self.heads = heads
        self.input_tracks = input_tracks
        self.alphabet_size = alphabet_size
        self.equilibrium_init = to_torch(equilibrium_init)
        self.exchangeability_init = to_torch(exchangeability_init)
        self.exchangeability_delta_init = to_torch(
            spec_or_zeros(exchangeability_delta_init)
        )
        self.mixture_init = to_torch(
            mixture_init if mixture_init is not None
            else _default_mixture_init()
        )
        # inverse_softplus(1.0) = log(exp(1) - 1) ~ 0.5413 -> softplus gives
        # 1.0 (neutral)
        self.scale_init = to_torch(
            scale_init if scale_init is not None
            else _default_scale_init()
        )
        self.trainable_equilibrium = trainable_equilibrium
        self.trainable_exchangeabilities = trainable_exchangeabilities
        self.trainable_scale = trainable_scale
        self.shared_equilibrium = shared_equilibrium
        self.shared_exchangeabilities = shared_exchangeabilities
        self.exchangeability_l2 = exchangeability_l2
        self.time_reversed = time_reversed
        self.num_components = num_components
        self.lora = lora
        self._head_subset = None
        self.built = False
        # The eigendecomposition is derived from the kernels, but it is cached
        # across calls, so it has to follow the module across devices. Plain
        # attributes would not; buffers do. They are non-persistent because
        # they are reproducible from the kernels and depend on head_subset,
        # which a checkpoint does not fix.
        for name in _GTR_BUFFERS:
            self.register_buffer(name, None, persistent=False)

    @property
    def head_subset(self):
        """If set, only these models are used in computations."""
        return self._head_subset

    @head_subset.setter
    def head_subset(self, subset) -> None:
        self._head_subset = subset
        if self.built:
            self._precompute_gtr_decomposition()

    def build(self, input_shape=None) -> None:
        if self.built:
            return

        R_shape = [
            self.heads,
            self.input_tracks,
            1 if self.shared_exchangeabilities else self.num_components,
            self.alphabet_size,
            self.alphabet_size,
        ]
        # A buffer, not a parameter: the base exchangeabilities are fixed, but
        # they still have to follow the module across devices and be captured
        # by state_dict.
        self.register_buffer(
            "exchangeability_const",
            self.exchangeability_init(torch.empty(R_shape)),
        )
        if self.lora is not None:
            R_shape[-1] = self.lora
        self.exchangeability_delta_kernel = torch.nn.Parameter(
            self.exchangeability_delta_init(torch.empty(R_shape)),
            requires_grad=self.trainable_exchangeabilities,
        )
        self.equilibrium_kernel = torch.nn.Parameter(
            self.equilibrium_init(torch.empty([
                self.heads,
                self.input_tracks,
                1 if self.shared_equilibrium else self.num_components,
                self.alphabet_size,
            ])),
            requires_grad=self.trainable_equilibrium,
        )
        if self.num_components > 1:
            trainable_mixture = (
                self.trainable_exchangeabilities
                or self.trainable_scale
                or self.trainable_equilibrium
            )
            self.mixture_kernel = torch.nn.Parameter(
                self.mixture_init(torch.empty(
                    [self.heads, self.input_tracks, self.num_components]
                )),
                requires_grad=trainable_mixture,
            )
            self.scale_kernel = torch.nn.Parameter(
                self.scale_init(torch.empty(
                    [self.heads, self.input_tracks, self.num_components]
                )),
                requires_grad=self.trainable_scale,
            )

        self._precompute_gtr_decomposition()
        self.built = True

    def regularization_loss(self) -> torch.Tensor:
        """The L2 penalties Keras attaches to the kernels via ``regularizer=``.

        TensorFlow collects these into ``Model.losses``; torch has no such
        hook, so the model adds this to the training loss explicitly.
        """
        device = self.exchangeability_delta_kernel.device
        loss = torch.zeros((), device=device)
        if self.exchangeability_l2 > 0.0:
            loss = loss + (self.exchangeability_l2 / self.heads) * torch.sum(
                torch.square(self.exchangeability_delta_kernel)
            )
        if self.num_components > 1:
            loss = loss + (self.MIXTURE_L2 / self.heads) * torch.sum(
                torch.square(self.mixture_kernel)
            )
        return loss

    @property
    def _gtr_decomp(self) -> "GTRDecomp | None":
        """The cached eigendecomposition, reassembled from its buffers."""
        if self._gtr_eigvals is None:
            return None
        return GTRDecomp(
            eigvals=self._gtr_eigvals,
            eigvecs=self._gtr_eigvecs,
            sqrt_pi=self._gtr_sqrt_pi,
            inv_sqrt_pi=self._gtr_inv_sqrt_pi,
        )

    def _precompute_gtr_decomposition(self) -> None:
        """Precompute GTR eigendecomposition for non-trainable rate matrices.
        Stores the result in buffers to optimize computation.
        """
        if self.trainable_exchangeabilities or self.trainable_equilibrium:
            for name in _GTR_BUFFERS:
                setattr(self, name, None)
            return

        with torch.no_grad():
            R, p = self.make_R(), self.make_p()
            H_active = (
                self.heads if self._head_subset is None
                else len(self._head_subset)
            )
            IK = self.input_tracks * self.num_components
            R_flat = R.reshape(-1, self.alphabet_size, self.alphabet_size)
            p_flat = p.reshape(-1, self.alphabet_size)
            Q = backend.make_rate_matrix(R_flat, p_flat)
            Q = Q.reshape(
                H_active, IK, self.alphabet_size, self.alphabet_size
            )
            p_IK = p.reshape(H_active, IK, self.alphabet_size)
            decomp = precompute_gtr(Q.unsqueeze(0), p_IK.unsqueeze(0))
            for name, value in zip(_GTR_BUFFERS, decomp):
                setattr(self, name, value.detach().clone())

    def _gather_heads(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        subset = torch.as_tensor(
            np.asarray(self._head_subset),
            dtype=torch.int64,
            device=tensor.device,
        )
        return tensor.index_select(dim, subset)

    def make_R(self, kernel: torch.Tensor | None = None) -> torch.Tensor:
        """Computes the exchangeability matrices R for all models.

        Returns:
            A tensor of shape (H, I, K, D, D).
        """
        if kernel is None:
            const = self.exchangeability_const
            delta = self.exchangeability_delta_kernel
            if self.lora is not None:
                delta = torch.matmul(delta, delta.transpose(-2, -1))
            if self._head_subset is not None:
                const = self._gather_heads(const, 0)
                delta = self._gather_heads(delta, 0)
            kernel = const + delta
        R = backend.make_symmetric_pos_semidefinite(kernel)
        if self.shared_exchangeabilities and self.num_components > 1:
            R = R.repeat(1, 1, self.num_components, 1, 1)
        return R

    def make_p(self) -> torch.Tensor:
        """Computes the equilibrium distributions p for all models.

        Returns:
            A tensor of shape (H, I, K, D).
        """
        kernel = self.equilibrium_kernel
        if self._head_subset is not None:
            kernel = self._gather_heads(kernel, 0)
        p = backend.make_equilibrium(kernel)
        if self.shared_equilibrium and self.num_components > 1:
            p = p.repeat(1, 1, self.num_components, 1)
        return p

    def make_w(self) -> torch.Tensor:
        """Computes the mixture weights for all models.

        Returns:
            A tensor of shape (H, I, K) summing to 1 along the K axis.
        """
        H_active = (
            self.heads if self._head_subset is None
            else len(self._head_subset)
        )
        if self.num_components == 1:
            return torch.ones(
                (H_active, self.input_tracks, 1),
                device=self.equilibrium_kernel.device,
            )
        kernel = self.mixture_kernel
        if self._head_subset is not None:
            kernel = self._gather_heads(kernel, 0)
        return torch.softmax(kernel, dim=-1)

    def make_scale(self) -> torch.Tensor:
        """Computes the per-component rate matrix scale factors for all models.

        Returns:
            A tensor of shape (H, I, K) with positive values (via softplus).
        """
        H_active = (
            self.heads if self._head_subset is None
            else len(self._head_subset)
        )
        if self.num_components == 1:
            return torch.ones(
                (H_active, self.input_tracks, 1),
                device=self.equilibrium_kernel.device,
            )
        kernel = self.scale_kernel
        if self._head_subset is not None:
            kernel = self._gather_heads(kernel, 0)
        return backend.make_branch_lengths(kernel)

    def make_Q(self) -> torch.Tensor:
        """Computes the rate matrices Q for all models and mixture components.

        Returns:
            A tensor of shape (H, I, K, D, D).
        """
        R, p = self.make_R(), self.make_p()
        R_flat = R.reshape(-1, self.alphabet_size, self.alphabet_size)
        p_flat = p.reshape(-1, self.alphabet_size)
        Q_flat = backend.make_rate_matrix(R_flat, p_flat)
        return Q_flat.reshape(R.shape)

    def make_P(self, tau: torch.Tensor) -> torch.Tensor:
        """Computes the transition probability matrices P given evolutionary
        times tau.

        Args:
            tau: A tensor of shape (B, H, I) containing the evolutionary times.

        Returns:
            A tensor of shape (B, H, I, D, D).
        """
        H_active = (
            self.heads if self._head_subset is None
            else len(self._head_subset)
        )
        IK = self.input_tracks * self.num_components

        # Expand tau from (B, H, I) to (B, H, I*K) and scale per component
        tau_expanded = torch.repeat_interleave(
            tau, self.num_components, dim=-1
        )  # (B, H, I*K)
        s_flat = self.make_scale().reshape(H_active, IK)
        tau_expanded = tau_expanded * s_flat.unsqueeze(0)  # (B, H, I*K)

        if self._gtr_decomp is not None:
            P_all = expm_gtr_from_decomp(self._gtr_decomp, tau_expanded)
        else:
            R, p = self.make_R(), self.make_p()
            R_flat = R.reshape(-1, self.alphabet_size, self.alphabet_size)
            p_flat = p.reshape(-1, self.alphabet_size)
            Q = backend.make_rate_matrix(R_flat, p_flat)
            Q = Q.reshape(
                H_active, IK, self.alphabet_size, self.alphabet_size
            ).unsqueeze(0)
            p_IK = p.reshape(
                H_active, IK, self.alphabet_size
            ).unsqueeze(0)
            P_all = expm_gtr(Q, tau_expanded, p_IK)

        # Reshape, apply mixture weights
        P_all = P_all.reshape(
            -1, H_active, self.input_tracks, self.num_components,
            self.alphabet_size, self.alphabet_size,
        )
        w = self.make_w()[None, :, :, :, None, None]
        P = (w * P_all).sum(dim=3)  # (B, H, I, D, D)

        if self.time_reversed:
            P = P.permute(0, 1, 2, 4, 3)
        return P


def spec_or_zeros(
    initializer: InitSpec | T_TorchInitializer | None,
) -> InitSpec | T_TorchInitializer:
    """The given initializer, or a zeros spec when none was provided."""
    from learnMSA.tree.initializer import Zeros

    return Zeros() if initializer is None else initializer


def _default_mixture_init() -> InitSpec:
    from learnMSA.tree.initializer import RandomNormal

    return RandomNormal(stddev=0.1)


def _default_scale_init() -> InitSpec:
    from learnMSA.tree.initializer import Constant

    return Constant(float(np.log(np.exp(1.0) - 1.0)))
