import sys
from collections.abc import Sequence

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import numpy as np
import torch
from hidten.hmm import HMMConfig
from hidten.torch.base import TorchLayer
from hidten.torch.emitter.categorical import (T_shapelike, T_TorchTensor,
                                              TorchCategoricalEmitter,
                                              n_shared_parameters)
from hidten.torch.prior import TorchPrior
from hidten.torch.util import (T_Initializer, safe_log, setup_initializer,
                               zero_row_softmax)

from learnMSA.hmm.joint_util import (AB_init, assert_value_sets, flatten_AB,
                                     outer_product_flat_np, tile_conditional)
from learnMSA.hmm.torch.profile_emitter import TorchProfileEmitter
from learnMSA.hmm.torch.util import sequence_mask
from learnMSA.hmm.util.value_set import PHMMValueSet


class TorchJointProfileEmitter(TorchProfileEmitter):
    """A profile emitter for the joint distribution of multiple categorical
    variables. Allows to apply individual priors to the marginal distributions.
    Also allows to initialize the joint distribution from the product of
    marginal distributions.
    """

    marginal_dims: list[int]

    @TorchCategoricalEmitter.initializer.setter
    def initializer(self, initializer: T_Initializer) -> None:
        self._initializer = setup_initializer(
            initializer, self.init_transform
        )

    def __init__(
        self,
        values: Sequence[PHMMValueSet] | None = None,
        marginal_values: Sequence[Sequence[PHMMValueSet]] | None = None,
        AB_values: Sequence[PHMMValueSet] | None = None,
        trainable_insertions: bool = True,
        use_full_matmul: bool = True,
        low_rank: int = 0,
        l2_reg: float = 1e-1,
        temperature: float = 1.0,
        conditional: bool = True,
        **kwargs
    ) -> None:
        """
        Args:
            values: A sequence of value sets for the joint distribution, one
                per head, with probabilities. Only valid when low_rank == 0.
            marginal_values: Value sets for the marginal distributions.
                Required when low_rank > 0 (used to compute the constant
                log-joint bias C). When low_rank == 0, an alternative to
                `values`; the joint is initialised as the outer product of the
                marginals.
            AB_values: Pre-computed flat A+B kernel values for the low-rank
                case, e.g. from model surgery. Only valid when low_rank > 0. If
                None, A and B are default-initialised near zero.
            trainable_insertions: Whether insertion emissions are trainable.
                Defaults to True.
            use_full_matmul: Whether to compute emission scores via a full
                matrix multiplication instead of copying insertion emissions.
            low_rank: The rank of the low-rank approximation to parameterise
                the joint distribution when exactly two marginals are provided.
                If 0, no low-rank approximation is used.
            l2_reg: L2 regularization coefficient applied to the kernel (A and
                B matrices) when ``low_rank > 0``.
            temperature: Temperature applied as an exponent (1/temperature).
            conditional: If true, instead of modelling the joint distribution,
                the emission scores are the conditional of the second variable
                given the first, i.e. P(x2 | x1, s).
        """
        _values: Sequence[PHMMValueSet]
        if low_rank <= 0:
            assert AB_values is None, \
                "`AB_values` is only valid when low_rank > 0."
            if values is not None:
                _values = values
            else:
                assert marginal_values is not None, \
                    "Either `values` or `marginal_values` must be provided."
                if conditional:
                    _values = conditional_marginal_values(marginal_values)
                else:
                    _values = product_marginal_values(marginal_values)
            self.init_transform = safe_log
        else:
            assert marginal_values is not None, \
                "`marginal_values` is required when low_rank > 0."
            assert values is None, \
                "`values` is not supported when low_rank > 0; " \
                "use `marginal_values`."
            assert_value_sets(marginal_values)
            # Compute and store the constant log-joint bias C from marginals.
            self._c_bias_init = _compute_c_bias_init(marginal_values)
            _values = AB_values if AB_values is not None \
                else low_rank_marginal_values(marginal_values, low_rank)
            self.init_transform = None

        if len(_values) == 0:
            raise ValueError("At least one value set must be provided.")

        super().__init__(
            values=_values,
            trainable_insertions=trainable_insertions,
            use_full_matmul=use_full_matmul,
            temperature=temperature,
            **kwargs
        )

        self.low_rank = low_rank
        self.l2_reg = l2_reg
        # A ModuleDict so the marginal priors move with .to(device) and are
        # captured by state_dict. They carry requires_grad=False (priors are
        # frozen by default), which is what the TF backend achieves by keeping
        # them out of the Keras tracker.
        self._marginal_priors = torch.nn.ModuleDict()
        self.marginal_dims = []

        self.conditional = conditional

    def add_marginal_prior(
        self, marginal_index: int, prior: TorchPrior
    ) -> None:
        """Adds a prior to the marginal distribution of the joint distribution.

        Args:
            marginal_index: The index of the marginal distribution.
            prior: The prior to add.
        """
        # Set hmm_config if available, otherwise it will be set when hmm_config
        # is set
        if hasattr(self, "_hmm_config"):
            prior.hmm_config = self._hmm_config
        self._marginal_priors[str(marginal_index)] = prior

    def get_marginal_prior(self, marginal_index: int) -> TorchPrior | None:
        """Returns the prior for the marginal distribution of the joint
        distribution.

        Args:
            marginal_index: The index of the marginal distribution.

        Returns:
            The prior for the marginal distribution, or None if no prior has
            been set.
        """
        return self._marginal_priors.get(str(marginal_index), None)

    @TorchProfileEmitter.hmm_config.setter
    def hmm_config(self, hmm_config: HMMConfig) -> None:
        assert TorchProfileEmitter.hmm_config.fset is not None
        TorchProfileEmitter.hmm_config.fset(self, hmm_config)
        # Also set it on the marginal priors if they exist
        for prior in self._marginal_priors.values():
            prior.hmm_config = hmm_config

    def build(self, input_shape: T_shapelike | None = None) -> None:
        if input_shape is None:
            raise ValueError("Input shapes must be provided.")

        assert isinstance(input_shape, tuple) \
            and all(isinstance(s, tuple) for s in input_shape), \
            "Input shape must be a tuple of tuples."

        self.marginal_dims = [shape[-1] for shape in input_shape]
        assert all(isinstance(d, int) for d in self.marginal_dims), \
            "Input shapes must have a known last dimension."

        # Compute the input dimension
        input_dim = np.prod(self.marginal_dims)

        self.input_dim = int(input_dim)
        if self.low_rank > 0:
            self.matrix_dim = self.low_rank * sum(self.marginal_dims)

        self.share = self._build_share()
        # Build the allow indices and the prior without letting the base reset
        # input_dim (and with it matrix_dim), which was just set above.
        TorchLayer.build(self, None)

        self.kernel = torch.nn.Parameter(
            torch.empty(n_shared_parameters(self.allow, self.share)),
            requires_grad=self.trainable,
        )
        self.initializer(self.kernel)

        if self.low_rank > 0:
            n1, n2 = self.marginal_dims[0], self.marginal_dims[1]
            H = len(self._lengths)
            max_states = max(self.hmm_config.states)
            c_init = np.zeros((H, max_states, n1 * n2), dtype=np.float32)
            for h, (c_match, c_insert) in enumerate(self._c_bias_init):
                L_h = int(self._lengths[h])
                q_h = self.hmm_config.states[h]
                c_init[h, :L_h, :] = c_match.reshape(L_h, n1 * n2)
                c_init[h, L_h:q_h, :] = c_insert.reshape(n1 * n2)
            # A buffer rather than a frozen parameter: C is a constant derived
            # from the marginals, but it still has to follow the module across
            # devices and be captured by state_dict.
            self.register_buffer("C_weight", torch.as_tensor(c_init))

    @override
    def matrix(self) -> T_TorchTensor:
        matrix = self._build_matrix(_identity)
        matrix = self._prepare_matrix(matrix)

        if self.low_rank > 0:
            A, B = self._A_B_matrices(matrix)
            logits = self._get_C() + torch.einsum("...ik,...jk->...ij", A, B)
            if self.conditional:
                matrix = torch.softmax(logits, dim=-1)
                matrix = self._anneal_matrix(matrix)
                matrix = matrix.reshape(
                    matrix.shape[0], matrix.shape[1], -1
                )
            else:
                matrix = logits.reshape(
                    logits.shape[0], logits.shape[1], -1
                )
                matrix = zero_row_softmax(matrix)
                matrix = self._anneal_matrix(matrix)
        else:
            if self.conditional:
                H, Q = matrix.shape[0], matrix.shape[1]
                matrix = matrix.reshape([H, Q] + self.marginal_dims)
                matrix = torch.softmax(matrix, dim=-1)
                matrix = self._anneal_matrix(matrix)
                matrix = matrix.reshape(H, Q, -1)
            else:
                matrix = zero_row_softmax(matrix)
                matrix = self._anneal_matrix(matrix)

        # mask out padding states; use only the subset of states if head_subset
        # is active, otherwise self.states would broadcast the mask back to the
        # full number of heads after _prepare_matrix has already filtered them
        matrix = matrix * sequence_mask(
            np.asarray(self._effective_states()),
            matrix.shape[1],
            matrix.dtype,
            matrix.device,
        )[..., None]

        return matrix

    def AB_matrix(self) -> T_TorchTensor:
        """Returns the concatenated A and B matrices for each head and state
        of shape ``(H, Q, n1*k + n2*k)`` where ``n1`` and ``n2`` are the
        marginal dimensions. Padding states are zeroed out. Requires
        ``low_rank > 0``."""
        assert self.low_rank > 0, "AB_matrix() requires low_rank > 0."
        matrix = self._build_matrix(_identity)
        matrix = self._prepare_matrix(matrix)
        A, B = self._A_B_matrices(matrix)
        n1 = self.marginal_dims[0]
        n2 = self.marginal_dims[1]
        k = self.low_rank
        H, Q = matrix.shape[0], matrix.shape[1]
        A = A.reshape(H, Q, n1 * k)
        B = B.reshape(H, Q, n2 * k)
        ab = torch.cat([A, B], dim=-1)
        ab = ab * sequence_mask(
            np.asarray(self._effective_states()),
            ab.shape[1],
            ab.dtype,
            ab.device,
        )[..., None]
        return ab

    def _effective_states(self) -> list[int]:
        if self.head_subset is not None:
            return [self.states[h] for h in self.head_subset]
        return self.states

    def _A_B_matrices(
        self, matrix: T_TorchTensor
    ) -> tuple[T_TorchTensor, T_TorchTensor]:
        """Returns the A and B matrices for the low-rank parameterization."""
        if self.low_rank <= 0:
            raise ValueError("Low-rank parameterization is not used.")
        n1 = self.marginal_dims[0]
        n2 = self.marginal_dims[1]
        k = self.low_rank
        H, Q = matrix.shape[0], matrix.shape[1]
        # Kernel layout: first n1*k entries are A (n1, k),
        # remaining n2*k entries are B (n2, k).
        A = matrix[:, :, :n1 * k].reshape(H, Q, n1, k)
        B = matrix[:, :, n1 * k:].reshape(H, Q, n2, k)
        return A, B

    def _get_C(self) -> T_TorchTensor:
        """Returns the log-joint bias C for the active heads and states,
        shaped ``(H', Q', n1, n2)``."""
        n1, n2 = self.marginal_dims[0], self.marginal_dims[1]
        C = self.C_weight  # (H, max_states, n1*n2)
        if self.head_subset is not None:
            subset = torch.as_tensor(
                np.asarray(self.head_subset),
                dtype=torch.int64,
                device=C.device,
            )
            C = C.index_select(0, subset)
            max_states_subset = max(
                [self.hmm_config.states[h] for h in self.head_subset]
            )
            C = C[:, :max_states_subset, :]
        H, Q = C.shape[0], C.shape[1]
        return C.reshape(H, Q, n1, n2)

    @override
    def forward(
        self, *emissions: T_TorchTensor, use_padding: bool = True,
    ) -> T_TorchTensor:
        observation_product = outer_product_flat(*emissions)
        return super().forward(observation_product, use_padding=use_padding)

    def marginal_matrices(
        self, matrix: T_TorchTensor | None = None
    ) -> tuple[T_TorchTensor, ...]:
        """Computes the marginal matrices for each marginal distribution.
        Requires `conditional=False`."""
        assert not self.conditional, \
            "Marginal matrices can only be computed for joint distributions."
        if matrix is None:
            matrix = self.matrix()
        H, Q = matrix.shape[0], matrix.shape[1]
        matrix = matrix.reshape([H, Q] + self.marginal_dims)
        return tuple(
            marginal_matrix(matrix, i)
            for i in range(len(self.marginal_dims))
        )

    def marginal_matrix_from_conditional(
        self, prior: T_TorchTensor, matrix: T_TorchTensor | None = None
    ) -> T_TorchTensor:
        """Computes the marginal matrix for the second variable from the
        conditional distribution of the second variable given the first and
        the prior distribution of the first variable.
        Requires `conditional=True`.
        """
        assert self.conditional, \
            "Marginal matrix from conditional can only be computed for " \
            "conditional distributions."
        if matrix is None:
            matrix = self.matrix()  # (H, Q, D1 * D2)
        H, Q = matrix.shape[0], matrix.shape[1]
        # (H, Q, D1, D2)
        matrix = matrix.reshape([H, Q] + self.marginal_dims)
        return torch.einsum("...i,...ij->...j", prior, matrix)

    def prior_scores(self) -> T_TorchTensor:
        """Calculates the prior scores for the module's parameters in
        log-scale.

        Returns:
            The prior scores of shape ``(H)``, where ``H`` is the number of
            heads.
        """
        log_prior_scores = torch.zeros(
            self.heads, dtype=self.kernel.dtype, device=self.kernel.device
        )
        matrix = self.matrix()

        # Apply priors to the marginal distributions if they exist
        if len(self._marginal_priors) > 0:
            marginal_matrices = self.marginal_matrices(matrix)
            for i, prior in self._marginal_priors.items():
                log_prior_scores = log_prior_scores + prior(
                    marginal_matrices[int(i)]
                )

        # Apply a prior to the joint distribution if it exists
        if hasattr(self, "_prior"):
            log_prior_scores = log_prior_scores + self._prior(matrix)

        # Apply L2 regularization to the raw kernel (A and B matrices)
        if self.low_rank > 0 and self.l2_reg > 0.0:
            ab = self.AB_matrix()
            # Sum over states, mean over kernel dimension to normalize by rank
            ab_sq_sum = torch.square(ab).sum(dim=(1, 2))
            log_prior_scores = log_prior_scores - self.l2_reg * ab_sq_sum

        return log_prior_scores


def _identity(x: T_TorchTensor) -> T_TorchTensor:
    """``tf.identity`` has no torch equivalent; ``_build_matrix`` needs a
    callable activation that leaves the dense kernel untouched."""
    return x


def product_marginal_values(
    marginal_values: Sequence[Sequence[PHMMValueSet]]
) -> Sequence[PHMMValueSet]:
    """Computes the outer product of the marginal value sets to create a
    joint value set.

    Args:
        marginal_values: Value sets for the marginal distribution.

    Returns:
        The joint value sets.
    """
    assert_value_sets(marginal_values)

    joint_values: list[PHMMValueSet] = []
    for h in range(len(marginal_values[0])):
        match_emission = _outer_product_flat_np(
            *[mv[h].match_emissions for mv in marginal_values]
        )
        insert_emission = _outer_product_flat_np(
            *[mv[h].insert_emissions for mv in marginal_values]
        )
        joint_values.append(
            PHMMValueSet(
                L=marginal_values[0][h].L,
                match_emissions=match_emission,
                insert_emissions=insert_emission,
                transitions=np.empty(()),
                start=np.empty(()),
            )
        )
    return joint_values


def conditional_marginal_values(
    marginal_values: Sequence[Sequence[PHMMValueSet]],
) -> Sequence[PHMMValueSet]:
    """Creates value sets for conditional initialization
    P(x2 | x1, s) = P(x2 | s).

    Only uses the second marginal (index 1). For each state the conditional is
    initialized as the x2 marginal tiled D1 times, so
    P(x2 | x1=i, s) = P(x2 | s) for all x1.

    Args:
        marginal_values: Value sets for the marginal distributions. Must
            contain at least two sequences.

    Returns:
        Value sets whose match/insert emissions contain the tiled second
        marginal, suitable for conditional initialisation.
    """
    assert_value_sets(marginal_values)

    result: list[PHMMValueSet] = []
    for h in range(len(marginal_values[0])):
        n1 = marginal_values[0][h].match_emissions.shape[-1]
        p2_match = marginal_values[1][h].match_emissions   # (L, D2)
        p2_insert = marginal_values[1][h].insert_emissions  # (D2,)

        # Tile p2 along the x1 axis so every conditional row equals p2.
        # Flat layout: [p2[0], ..., p2[D2-1], p2[0], ...] repeated D1 times.
        match_emission = tile_conditional(p2_match, n1)  # (L * D1, D2)
        insert_emission = tile_conditional(p2_insert, n1)  # (D1 * D2,)

        result.append(PHMMValueSet(
            L=marginal_values[0][h].L,
            match_emissions=match_emission,
            insert_emissions=insert_emission,
            transitions=np.empty(()),
            start=np.empty(()),
        ))
    return result


def low_rank_marginal_values(
    marginal_values: Sequence[Sequence[PHMMValueSet]],
    low_rank: int = 1,
    noise_std: float = 1e-2,
    seed: int | None = None,
) -> Sequence[PHMMValueSet]:
    """Creates near-zero A and B kernel value sets for low-rank initialisation.

    A and B are initialised so that AB^T = 0, meaning the initial joint
    distribution is determined entirely by the constant log-joint bias C
    (computed separately from the marginals in ``_compute_c_bias_init``).

    Args:
        marginal_values: Value sets for the marginal distributions. Used for
            shapes only.
        low_rank: The rank of the low-rank approximation.
        noise_std: Standard deviation for A's random noise. B is always zero,
            so AB^T = 0 exactly at initialisation.
        seed: Optional random seed.

    Returns:
        PHMMValueSets whose match/insert emissions contain the flattened
        near-zero A and B kernel values.
    """
    assert_value_sets(marginal_values)
    assert len(marginal_values) == 2, \
        "Low-rank initialisation is only supported for exactly two marginals."

    result: list[PHMMValueSet] = []
    for h in range(len(marginal_values[0])):
        n1 = marginal_values[0][h].match_emissions.shape[-1]
        n2 = marginal_values[1][h].match_emissions.shape[-1]
        L = marginal_values[0][h].L
        A_match, B_match = AB_init(
            n1, n2, low_rank, batch_shape=(L,),
            noise_std=noise_std, seed=seed,
        )
        A_ins, B_ins = AB_init(
            n1, n2, low_rank, noise_std=noise_std, seed=seed
        )
        result.append(
            PHMMValueSet(
                L=L,
                match_emissions=flatten_AB(A_match, B_match),
                insert_emissions=flatten_AB(A_ins, B_ins),
                transitions=np.empty(()),
                start=np.empty(()),
            )
        )
    return result


def outer_product_flat(*emissions: T_TorchTensor) -> T_TorchTensor:
    """Computes the outer product of the emissions in the last dimension
    and returns a tensor with the flattened product dimension.

    Args:
        emissions: The input sequences of shape ``(..., D_i)``.

    Returns:
        The product of the emissions of shape ``(..., prod_i D_i)``.
    """
    assert len(emissions) > 1, "At least two emissions are required."
    x = outer_product_flat_pw(emissions[0], emissions[1])
    for obs in emissions[2:]:
        x = outer_product_flat_pw(x, obs)
    return x


def outer_product_flat_pw(
    x: T_TorchTensor, y: T_TorchTensor
) -> T_TorchTensor:
    """Computes the outer product of two tensors and flattens the multiplied
    dimensions.

    Args:
        x: The first tensor of shape ``(..., D1)``.
        y: The second tensor of shape ``(..., D2)``.

    Returns:
        The outer product of the two tensors of shape ``(..., D1 * D2)``.
    """
    z = torch.einsum("...u,...v->...uv", x, y)
    return z.reshape(*z.shape[:-2], z.shape[-2] * z.shape[-1])


def _outer_product_flat_np(*arrays: np.ndarray) -> np.ndarray:
    """The host-array n-ary form of :func:`outer_product_flat`.

    Value sets are numpy all the way through, so building the joint
    initialisation never has to enter the tensor framework.
    """
    assert len(arrays) > 1, "At least two arrays are required."
    x = outer_product_flat_np(arrays[0], arrays[1])
    for array in arrays[2:]:
        x = outer_product_flat_np(x, array)
    return x


def marginal_matrix(matrix: T_TorchTensor, i: int) -> T_TorchTensor:
    """Computes the marginal matrix for a given marginal index.

    Args:
        matrix: The joint distribution matrix of shape ``(H, Q, D1, ..., Dn)``.
        i: The index of the marginal to compute.

    Returns:
        The marginal matrix of shape ``(H, Q, D{i})``.
    """
    H, Q = matrix.shape[0], matrix.shape[1]
    n = matrix.dim() - 2
    perm = [0, 1] + [j + 2 for j in range(n) if j != i] + [i + 2]
    matrix = matrix.permute(*perm)
    matrix = matrix.reshape(H, Q, -1, matrix.shape[-1])
    return matrix.sum(dim=2)


def compute_C_from_marginals(
    p1: np.ndarray,
    p2: np.ndarray,
    epsilon: float = 1e-16,
) -> np.ndarray:
    """Computes the log-joint bias C for the low-rank parameterisation.

    ``C[..., i, j] = log(p1[i]) + log(p2[j])``, which encodes the independent
    joint distribution as a constant log-probability matrix. When used as the
    sole logit contribution (AB = 0), ``softmax(C) = p1 (x) p2``.

    Args:
        p1: First marginal distribution of shape ``(..., n1)``.
        p2: Second marginal distribution of shape ``(..., n2)``.
        epsilon: Small value to avoid log(0). Defaults to 1e-16.

    Returns:
        C of shape ``(..., n1, n2)``, dtype float32.
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    log_p1 = np.log(p1 + epsilon)
    log_p2 = np.log(p2 + epsilon)
    C = log_p1[..., :, np.newaxis] + log_p2[..., np.newaxis, :]
    return C.astype(np.float32)


def _compute_c_bias_init(
    marginal_values: Sequence[Sequence[PHMMValueSet]],
    epsilon: float = 1e-16,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Computes the constant log-joint bias C for each head.

    Returns a list of ``(c_match, c_insert)`` pairs per head, where
    ``c_match`` has shape ``(L, n1, n2)`` and ``c_insert`` has shape
    ``(n1, n2)``.
    """
    result = []
    for h in range(len(marginal_values[0])):
        c_match = compute_C_from_marginals(
            marginal_values[0][h].match_emissions,
            marginal_values[1][h].match_emissions,
            epsilon=epsilon,
        )  # (L, n1, n2)
        c_insert = compute_C_from_marginals(
            marginal_values[0][h].insert_emissions,
            marginal_values[1][h].insert_emissions,
            epsilon=epsilon,
        )  # (n1, n2)
        result.append((c_match, c_insert))
    return result
