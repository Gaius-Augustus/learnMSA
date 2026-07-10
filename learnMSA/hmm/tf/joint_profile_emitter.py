import sys
from collections.abc import Sequence

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import numpy as np
import tensorflow as tf
from hidten.hmm import HMMConfig
from hidten.tf.emitter.categorical import (T_initializer, T_shapelike,
                                           T_TFTensor, TFCategoricalEmitter,
                                           n_shared_parameters, safe_log,
                                           setup_initializer)
from hidten.tf.prior import TFPrior
from hidten.tf.util import zero_row_softmax

from learnMSA.hmm.util.value_set import PHMMValueSet

from .profile_emitter import ProfileEmitter


class JointProfileEmitter(ProfileEmitter):
    """A profile emitter for the joint distribution of multiple categorical
    variables. Allows to apply individual priors to the marginal distributions.
    Also allows to initialize the joint distribution from the product of
    marginal distributions.
    """

    _marginal_priors: dict[int, TFPrior]

    marginal_dims: list[int]

    @TFCategoricalEmitter.initializer.setter
    def initializer(self, initializer: T_initializer) -> None:
        self._initializer = setup_initializer(initializer, self.init_transform)

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
            values (Sequence[PHMMValueSet]): A sequence of value sets for the
                joint distribution, one per head, with probabilities. Only
                valid when low_rank == 0.
            marginal_values (Sequence[Sequence[PHMMValueSet]]): Value sets for
                the marginal distributions. Required when low_rank > 0 (used
                to compute the constant log-joint bias C). When low_rank == 0,
                an alternative to `values`; the joint is initialised as the
                outer product of the marginals.
            AB_values (Sequence[PHMMValueSet] | None): Pre-computed flat A+B
                kernel values for the low-rank case, e.g. from model surgery.
                Only valid when low_rank > 0. If None, A and B are default-
                initialised near zero.
            trainable_insertions (bool): Whether insertion emissions are
                trainable. Defaults to True.
            use_full_matmul (bool): Whether to compute emission scores via
                a full matrix multiplication instead of copying insertion
                emissions.
            low_rank (int): The rank of the low-rank approximation to
                parameterise the joint distribution when exactly two marginals
                are provided. If 0, no low-rank approximation is used.
            conditional (bool): If true, instead of modelling the joint
                distribution, the emission scores are the conditional of the
                second variable given the first, i.e. P(x2 | x1, s).
            l2_reg (float): L2 regularization coefficient applied to the
                kernel (A and B matrices) when ``low_rank > 0``. Defaults
                to 0.0 (no regularization).
            temperature (float): Temperature applied as an exponent
                (1/temperature).
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
            _assert_value_sets(marginal_values)
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
        # Use object.__setattr__ to store as a plain Python dict, bypassing
        # Keras tracking. This allows priors to be added after build() without
        # triggering the "tracker locked" error.
        object.__setattr__(self, '_marginal_priors', {})
        self.marginal_dims = []

        self.conditional = conditional

    def add_marginal_prior(self, marginal_index: int, prior: TFPrior) -> None:
        """Adds a prior to the marginal distribution of the joint distribution.

        Args:
            marginal_index (int): The index of the marginal distribution.
            prior (TFPrior): The prior to add.
        """
        # Set hmm_config if available, otherwise it will be set when hmm_config
        # is set
        if hasattr(self, "_hmm_config"):
            prior.hmm_config = self._hmm_config
        self._marginal_priors[marginal_index] = prior

    def get_marginal_prior(self, marginal_index: int) -> TFPrior | None:
        """Returns the prior for the marginal distribution of the joint
        distribution.

        Args:
            marginal_index (int): The index of the marginal distribution.

        Returns:
            TFPrior | None: The prior for the marginal distribution, or None if
                no prior has been set.
        """
        return self._marginal_priors.get(marginal_index, None)

    @ProfileEmitter.hmm_config.setter
    def hmm_config(self, hmm_config: HMMConfig) -> None:
        assert ProfileEmitter.hmm_config.fset is not None
        ProfileEmitter.hmm_config.fset(self, hmm_config)
        # Also set it on the marginal priors if they exist
        for prior in self._marginal_priors.values():
            prior.hmm_config = hmm_config


    def build(self, input_shape: T_shapelike | None = None) -> None:
        if input_shape is None:
            raise ValueError("Input shapes must be provided.")
        else:
            assert isinstance(input_shape, tuple)\
                    and all(isinstance(s, tuple) for s in input_shape),\
                "Input shape must be a tuple of tuples."

            self.marginal_dims = [shape[-1] for shape in input_shape] # type: ignore
            assert all(isinstance(d, int) for d in self.marginal_dims),\
                "Input shapes must have a known last dimension."

            # Compute the input dimension
            input_dim = np.prod(self.marginal_dims)
            input_shape = (None, None, int(input_dim))

        if input_shape is not None:
            self.input_dim = input_shape[-1]  # type: ignore
            if self.low_rank > 0:
                self.matrix_dim = self.low_rank * sum(self.marginal_dims)

        self.share = self._build_share()
        self._build_allow()
        self._build_prior()

        self.kernel = self.add_weight(
            shape=(n_shared_parameters(self.allow, self.share), ),
            initializer=self.initializer,
            name="kernel",
        )

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
            self.C_weight = self.add_weight(
                shape=(H, max_states, n1 * n2),
                initializer=tf.constant_initializer(c_init),
                trainable=False,
                name="C_bias",
            )

    @override
    def matrix(self) -> T_TFTensor:
        matrix = self._build_matrix(tf.identity)
        matrix = self._prepare_matrix(matrix)

        if self.low_rank > 0:
            A, B = self._A_B_matrices(matrix)
            logits = self._get_C() + tf.einsum("...ik,...jk->...ij", A, B)
            if self.conditional:
                matrix = tf.nn.softmax(logits, axis=-1)
                matrix = self._anneal_matrix(matrix)
                matrix = tf.reshape(
                    matrix, [tf.shape(matrix)[0], tf.shape(matrix)[1], -1]
                )
            else:
                matrix = tf.reshape(
                    logits, [tf.shape(logits)[0], tf.shape(logits)[1], -1]
                )
                matrix = zero_row_softmax(matrix)
                matrix = self._anneal_matrix(matrix)
        else:
            if self.conditional:
                H = tf.shape(matrix)[0]
                Q = tf.shape(matrix)[1]
                matrix = tf.reshape(matrix, [H, Q] + self.marginal_dims)
                matrix = tf.nn.softmax(matrix, axis=-1)
                matrix = self._anneal_matrix(matrix)
                matrix = tf.reshape(matrix, [H, Q, -1])
            else:
                matrix = zero_row_softmax(matrix)
                matrix = self._anneal_matrix(matrix)

        # mask out padding states; use only the subset of states if head_subset
        # is active, otherwise self.states would broadcast the mask back to the
        # full number of heads after _prepare_matrix has already filtered them
        effective_states = (
            [self.states[h] for h in self.head_subset]
            if self.head_subset is not None
            else self.states
        )
        matrix *= tf.sequence_mask(
            effective_states, dtype=matrix.dtype
        )[..., tf.newaxis]

        return matrix

    def AB_matrix(self) -> T_TFTensor:
        """Returns the concatenated A and B matrices for each head and state
        of shape ``(H, Q, n1*k + n2*k)`` where ``n1`` and ``n2`` are the
        marginal dimensions. Padding states are zeroed out. Requires
        ``low_rank > 0``."""
        assert self.low_rank > 0, "AB_matrix() requires low_rank > 0."
        matrix = self._build_matrix(tf.identity)
        matrix = self._prepare_matrix(matrix)
        A, B = self._A_B_matrices(matrix)
        n1 = self.marginal_dims[0]
        n2 = self.marginal_dims[1]
        k = self.low_rank
        H = tf.shape(matrix)[0]
        Q = tf.shape(matrix)[1]
        A = tf.reshape(A, [H, Q, n1 * k])
        B = tf.reshape(B, [H, Q, n2 * k])
        ab = tf.concat([A, B], axis=-1)
        effective_states = (
            [self.states[h] for h in self.head_subset]
            if self.head_subset is not None
            else self.states
        )
        ab *= tf.sequence_mask(
            effective_states, dtype=ab.dtype
        )[..., tf.newaxis]
        return ab

    def _A_B_matrices(self, matrix: T_TFTensor) -> tuple[T_TFTensor, T_TFTensor]:
        """Returns the A and B matrices for the low-rank parameterization."""
        if self.low_rank <= 0:
            raise ValueError("Low-rank parameterization is not used.")
        n1 = self.marginal_dims[0]
        n2 = self.marginal_dims[1]
        k = self.low_rank
        H = tf.shape(matrix)[0]
        Q = tf.shape(matrix)[1]
        # Kernel layout: first n1*k entries are A (n1, k),
        # remaining n2*k entries are B (n2, k).
        A = tf.reshape(matrix[:, :, :n1 * k], [H, Q, n1, k])
        B = tf.reshape(matrix[:, :, n1 * k:], [H, Q, n2, k])
        return A, B

    def _get_C(self) -> T_TFTensor:
        """Returns the log-joint bias C for the active heads and states,
        shaped ``(H', Q', n1, n2)``."""
        n1, n2 = self.marginal_dims[0], self.marginal_dims[1]
        C = self.C_weight  # (H, max_states, n1*n2)
        if self.head_subset is not None:
            C = tf.gather(C, self.head_subset, axis=0)
            max_states_subset = max(
                [self.hmm_config.states[h] for h in self.head_subset]
            )
            C = C[:, :max_states_subset, :]
        H = tf.shape(C)[0]
        Q = tf.shape(C)[1]
        return tf.reshape(C, [H, Q, n1, n2])

    @override
    def call(
        self, *emissions: T_TFTensor, use_padding: bool = True,
    ) -> T_TFTensor:
        observation_product = outer_product_flat(*emissions)
        return super().call(observation_product, use_padding=use_padding)

    def marginal_matrices(
        self, matrix: T_TFTensor | None = None
    ) -> tuple[T_TFTensor, ...]:
        """Computes the marginal matrices for each marginal distribution.
        Requires `conditional=False`."""
        assert not self.conditional,\
            "Marginal matrices can only be computed for joint distributions."
        if matrix is None:
            matrix = self.matrix()
        H, Q = tf.unstack(tf.shape(matrix)[:2])
        matrix = tf.reshape(
            matrix, [H, Q] + self.marginal_dims
        )
        marginal_matrices = []
        for i in range(len(self.marginal_dims)):
            marginal_matrices.append(marginal_matrix(matrix, i))
        return tuple(marginal_matrices)

    def marginal_matrix_from_conditional(
        self, prior: T_TFTensor, matrix: T_TFTensor | None = None
    ) -> T_TFTensor:
        """Computes the marginal matrix for the second variable from the
        conditional distribution of the second variable given the first and
        the prior distribution of the first variable.
        Requires `conditional=False`.
        """
        assert self.conditional,\
            "Marginal matrix from conditional can only be computed for "\
            "conditional distributions."
        if matrix is None:
            matrix = self.matrix() # (H, Q, D1 * D2)
        H, Q = tf.unstack(tf.shape(matrix)[:2])
        matrix = tf.reshape(matrix, [H, Q] + self.marginal_dims) # (H, Q, D1, D2)
        marginal_matrix = tf.einsum("...i,...ij->...j", prior, matrix)
        return marginal_matrix

    def prior_scores(self) -> T_TFTensor:
        """Calculates the prior scores for the modules' parameters in log-scale.

        Returns:
            Tensor: The prior scores of shape ``(H)``, where ``H``
                is the number of heads.
        """
        log_prior_scores = tf.zeros(shape=[self.heads], dtype=self.dtype)
        matrix = self.matrix()

        # Apply priors to the marginal distributions if they exist
        if len(self._marginal_priors) > 0:
            marginal_matrices = self.marginal_matrices(matrix)
            for i, prior in self._marginal_priors.items():
                log_prior_scores += prior(marginal_matrices[i])

        # Apply a prior to the joint distribution if it exists
        if hasattr(self, "_prior"):
            log_prior_scores += self._prior(matrix)

        # Apply L2 regularization to the raw kernel (A and B matrices)
        if self.low_rank > 0 and self.l2_reg > 0.0:
            ab = self.AB_matrix()
            # Sum over states, mean over kernel dimension to normalize by rank
            ab_sq_sum = tf.reduce_sum(tf.square(ab), axis=[1,2])
            log_prior_scores -= self.l2_reg * ab_sq_sum

        return log_prior_scores


def product_marginal_values(
    marginal_values: Sequence[Sequence[PHMMValueSet]]
) -> Sequence[PHMMValueSet]:
    """Computes the outer product of the marginal value sets to create a
    joint value set.

    Args:
        marginal_values (Sequence[Sequence[PHMMValueSet]]): Value sets for
            the marginal distribution.

    Returns:
        Sequence[PHMMValueSet]: The joint value sets.
    """
    _assert_value_sets(marginal_values)

    joint_values: list[PHMMValueSet] = []
    for h in range(len(marginal_values[0])):
        match_emission = outer_product_flat(
            *[tf.constant(mv[h].match_emissions) for mv in marginal_values]
        ).numpy()
        insert_emission = outer_product_flat(
            *[tf.constant(mv[h].insert_emissions) for mv in marginal_values]
        ).numpy()
        joint_values.append(
            PHMMValueSet(
                L=marginal_values[0][h].L,
                match_emissions=match_emission,
                insert_emissions=insert_emission,
                transitions = np.empty(()),
                start = np.empty(()),
            )
        )
    return joint_values

def conditional_marginal_values(
    marginal_values: Sequence[Sequence[PHMMValueSet]],
) -> Sequence[PHMMValueSet]:
    """Creates value sets for conditional initialization P(x2 | x1, s) = P(x2 | s).

    Only uses the second marginal (index 1). For each state the conditional is
    initialized as the x2 marginal tiled D1 times, so P(x2 | x1=i, s) = P(x2 | s)
    for all x1.

    Args:
        marginal_values (Sequence[Sequence[PHMMValueSet]]): Value sets for the
            marginal distributions. Must contain at least two sequences.

    Returns:
        Sequence[PHMMValueSet]: Value sets whose match/insert emissions contain
            the tiled second marginal, suitable for conditional initialisation.
    """
    _assert_value_sets(marginal_values)

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
        marginal_values (Sequence[Sequence[PHMMValueSet]]): Value sets for
            the marginal distributions. Used for shapes only.
        low_rank (int): The rank of the low-rank approximation.
        noise_std (float): Standard deviation for A's random noise. B is
            always zero, so AB^T = 0 exactly at initialisation.
        seed (int | None): Optional random seed.

    Returns:
        Sequence[PHMMValueSet]: PHMMValueSets whose match/insert emissions
            contain the flattened near-zero A and B kernel values.
    """
    _assert_value_sets(marginal_values)
    assert len(marginal_values) == 2, \
        "Low-rank initialisation is only supported for exactly two marginals."

    result: list[PHMMValueSet] = []
    for h in range(len(marginal_values[0])):
        n1 = marginal_values[0][h].match_emissions.shape[-1]
        n2 = marginal_values[1][h].match_emissions.shape[-1]
        L = marginal_values[0][h].L
        A_match, B_match = AB_init(n1, n2, low_rank,
                                   batch_shape=(L,),
                                   noise_std=noise_std, seed=seed)
        A_ins, B_ins = AB_init(n1, n2, low_rank,
                               noise_std=noise_std, seed=seed)
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

def outer_product_flat(*emissions: T_TFTensor | np.ndarray) -> T_TFTensor:
    """Computes the outer product of the emissions in the last dimension
    and returns a tensor with the flatted product dimension.

    Args:
        emissions (Tensor): The input sequences of shape
        ``(..., D_i)``.

    Returns:
        Tensor: The product of the emissions of shape
        ``(..., prod_i D_i)``.
    """
    assert len(emissions) > 1, "At least two emissions are required."
    x = outer_product_flat_pw(emissions[0], emissions[1])
    for obs in emissions[2:]:
        x = outer_product_flat_pw(x, obs)
    return x

def tile_conditional(marginal: np.ndarray, n1: int) -> np.ndarray:
    """Tiles the second marginal along the first marginal dimension to create
    a conditional distribution P(x2 | x1, s) = P(x2 | s).

    Args:
        marginal (Tensor): The second marginal of shape ``(..., D2)``.
        n1 (int): The size of the first marginal.

    Returns:
        Tensor: The tiled conditional distribution of shape
        ``(..., D1 * D2)``.
    """
    return np.tile(marginal, n1)

def outer_product_flat_pw(x
    : T_TFTensor | np.ndarray, y: T_TFTensor | np.ndarray
) -> T_TFTensor:
    """Computes the outer product of two tensors and flattens the multiplied
    dimensions.

    Args:
        x (Tensor): The first tensor of shape ``(..., D1)``.
        y (Tensor): The second tensor of shape ``(..., D2)``.

    Returns:
        Tensor: The outer product of the two tensors of shape
        ``(..., D1 * D2)``.
    """
    z = tf.einsum("...u,...v->...uv", x, y)
    product_shape = tf.concat(
        [tf.shape(z)[:-2], [tf.shape(z)[-2] * tf.shape(z)[-1]]], axis=0
    )
    return tf.reshape(z, product_shape)

def marginal_matrix(matrix: T_TFTensor, i: int) -> T_TFTensor:
    """Computes the marginal matrix for a given marginal index.

    Args:
        matrix (Tensor): The joint distribution matrix of shape ``(H, Q, D1, ..., Dn)``.
        i (int): The index of the marginal to compute.

    Returns:
        Tensor: The marginal matrix of shape ``(H, Q, D{i})``.
    """
    H, Q = tf.unstack(tf.shape(matrix)[:2])
    n = len(matrix.shape) - 2
    perm = [0, 1] + [j+2 for j in range(n) if j != i] + [i+2]
    matrix = tf.transpose(matrix, perm)
    matrix = tf.reshape(matrix, [H, Q, -1, matrix.shape[-1]])
    marginal_matrix = tf.reduce_sum(matrix, axis=2)
    return marginal_matrix

def compute_C_from_marginals(
    p1: np.ndarray,
    p2: np.ndarray,
    epsilon: float = 1e-16,
) -> np.ndarray:
    """Computes the log-joint bias C for the low-rank parameterisation.

    ``C[..., i, j] = log(p1[i]) + log(p2[j])``, which encodes the independent
    joint distribution as a constant log-probability matrix.  When used as the
    sole logit contribution (AB = 0), ``softmax(C) = p1 ⊗ p2``.

    Args:
        p1 (np.ndarray): First marginal distribution of shape ``(..., n1)``.
        p2 (np.ndarray): Second marginal distribution of shape ``(..., n2)``.
        epsilon (float): Small value to avoid log(0). Defaults to 1e-16.

    Returns:
        np.ndarray: C of shape ``(..., n1, n2)``, dtype float32.
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


def AB_init(
    n1: int,
    n2: int,
    low_rank: int,
    batch_shape: tuple[int, ...] = (),
    noise_std: float = 1e-2,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Initialises A and B matrices near zero for low-rank parameterisation.

    A is filled with small Gaussian noise; B is zeros.  Because B = 0,
    AB^T = 0 exactly at initialisation, so the initial joint distribution
    is determined entirely by the constant log-joint bias C.

    Args:
        n1 (int): Size of the first marginal alphabet.
        n2 (int): Size of the second marginal alphabet.
        low_rank (int): Rank of the approximation.  Must be >= 1.
        batch_shape (tuple[int, ...]): Optional leading batch dimensions.
        noise_std (float): Standard deviation for A's Gaussian noise.
            Pass 0.0 for exact zeros (e.g. for surgery-inserted positions).
        seed (int | None): Random seed for reproducibility.

    Returns:
        tuple[np.ndarray, np.ndarray]: A of shape
            ``batch_shape + (n1, low_rank)`` and B of shape
            ``batch_shape + (n2, low_rank)``.
    """
    assert low_rank >= 1, "low_rank must be at least 1"
    rng = np.random.default_rng(seed)
    A = rng.normal(scale=noise_std,
                   size=batch_shape + (n1, low_rank)).astype(np.float64)
    B = np.zeros(batch_shape + (n2, low_rank), dtype=np.float64)
    return A, B

def flatten_AB(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Flattens the A and B matrices into a single array for use as an
    initializer.

    Args:
        A (np.ndarray): The A matrix of shape (..., n1, low_rank).
        B (np.ndarray): The B matrix of shape (..., n2, low_rank).

    Returns:
        np.ndarray: The flattened array of shape
            (..., n1 * low_rank + n2 * low_rank).
    """
    batch_shape = A.shape[:-2]
    A = np.reshape(A, batch_shape + (-1,))
    B = np.reshape(B, batch_shape + (-1,))
    return np.concatenate([A, B], axis=-1)

def _assert_value_sets(marginal_values: Sequence[Sequence[PHMMValueSet]]) -> None:
    assert len(marginal_values) > 1,\
        "At least two marginal value sets are required."
    assert all(len(marginal_values[0]) == len(mv) for mv in marginal_values),\
        "All marginal value sets must have the same number of heads."
    for h in range(len(marginal_values[0])):
        assert all(marginal_values[0][h].L == mv[h].L for mv in marginal_values),\
            "All marginal value sets must have the same length for each head."