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

    _marginal_priors: dict[int, TFPrior] = {}

    marginal_dims: list[int] = []

    @TFCategoricalEmitter.initializer.setter
    def initializer(self, initializer: T_initializer) -> None:
        self._initializer = setup_initializer(initializer, self.init_transform)

    def __init__(
        self,
        values: Sequence[PHMMValueSet] | None = None,
        marginal_values: Sequence[Sequence[PHMMValueSet]] | None = None,
        trainable_insertions: bool = True,
        use_full_matmul: bool = True,
        temperature: float = 1.0,
        low_rank: int = 0,
        kernel_values: bool = False,
        **kwargs
    ) -> None:
        """
        Args:
            values (Sequence[PHMMValueSet]): A sequence of value sets for the
                joint distribution, one per head, with probabilities.
            marginal_values (Sequence[Sequence[PHMMValueSet]]): Value sets for
                the marginal distribution. An alternative to providing joint
                values. The outer sequence is over the marginals, the inner
                sequence is over the heads. The number of heads must be the
                same for all marginals. If both `values` and `marginal_values`
                are provided, `values` takes precedence.
            trainable_insertions (bool): Whether insertion emissions are
                trainable. Defaults to True.
            use_full_matmul (bool): Whether to compute emission scores via
                a full matrix multiplication instead of copying insertion
                emissions.
            temperature (float): Temperature applied as an exponent
                (1/temperature) to the emission scores. Defaults to 1.0.
            low_rank (int): The rank of the low-rank approximation to
                parameterize the joint distribution when exactly two marginals are
                provided. If 0, no low-rank approximation is used.
            kernel_values (bool): If true, the provided values are treated as
                as kernel values instead of probabilities, i.e. logits unless
                low_rank > 0, in which case they are treated as the concatenated
                and flattened A and B matrices.
        """
        _values : Sequence[PHMMValueSet]
        if values is not None:
            _values = values
            self.init_transform = None if kernel_values else safe_log
        else:
            assert marginal_values is not None,\
                "Either `values` or `marginal_values` must be provided."
            if low_rank <= 0:
                _values = product_marginal_values(marginal_values)
                self.init_transform = None if kernel_values else safe_log
            else:
                assert not kernel_values, "kernel_values is not supported for"\
                    "low-rank initialization with marginals."
                _values = low_rank_marginal_values(marginal_values, low_rank)
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

    @override
    def matrix(self) -> T_TFTensor:
        matrix = self._build_matrix(tf.identity)
        matrix = self._prepare_matrix(matrix)
        if self.low_rank > 0:
            A, B = self._A_B_matrices(matrix)
            matrix = tf.einsum("...ik,...jk->...ij", A, B)
            matrix = tf.reshape(
                matrix, [tf.shape(matrix)[0], tf.shape(matrix)[1], -1]
            )
        matrix = zero_row_softmax(matrix)
        # mask out padding states; use only the subset of states if head_subset
        # is active, otherwise self.states would broadcast the mask back to the
        # full number of heads after _prepare_matrix has already filtered them
        effective_states = (
            [self.states[h] for h in self.head_subset]
            if self.head_subset is not None
            else self.states
        )
        matrix *= tf.sequence_mask(effective_states, dtype=matrix.dtype)[..., tf.newaxis]
        return matrix

    def parameter_matrix(self) -> T_TFTensor:
        """Returns the matrix of raw parameters (before softmax). If low-rank
        is used, this is the concatenated A and B matrices for each head and
        state of shape ``(H, Q, n1*k + n2*k)`` where ``n1`` and ``n2`` are the
        marginal dimensions."""
        matrix = self._build_matrix(tf.identity)
        matrix = self._prepare_matrix(matrix)
        if self.low_rank > 0:
            A, B = self._A_B_matrices(matrix)
            n1 = self.marginal_dims[0]
            n2 = self.marginal_dims[1]
            k = self.low_rank
            H = tf.shape(matrix)[0]
            Q = tf.shape(matrix)[1]
            A = tf.reshape(A, [H, Q, n1 * k])
            B = tf.reshape(B, [H, Q, n2 * k])
            matrix = tf.concat([A, B], axis=-1)
        return matrix

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

    @override
    def call(
        self, *emissions: T_TFTensor, use_padding: bool = True,
    ) -> T_TFTensor:
        observation_product = outer_product_flat(*emissions)
        return super().call(observation_product, use_padding=use_padding)

    def marginal_matrices(
        self, matrix: T_TFTensor | None = None
    ) -> tuple[T_TFTensor, ...]:
        if matrix is None:
            matrix = self.matrix()
        matrix = tf.reshape(
            matrix, [self.heads, self.max_states] + self.marginal_dims
        )
        marginal_matrices = []
        for i in range(len(self.marginal_dims)):
            marginal_matrices.append(marginal_matrix(matrix, i))
        return tuple(marginal_matrices)

    def prior_scores(self) -> T_TFTensor:
        """Calculates the prior scores for the modules' parameters in log-scale.

        Returns:
            Tensor: The prior scores of shape ``(H)``, where ``H``
                is the number of heads.
        """
        log_prior_scores = tf.zeros(shape=[self.heads], dtype=self.dtype)
        matrix = self.matrix()

        # Apply priors to the marginal distributions if they exist
        marginal_matrices = self.marginal_matrices(matrix)
        for i, prior in self._marginal_priors.items():
            log_prior_scores += prior(marginal_matrices[i])

        # Apply a prior to the joint distribution if it exists
        if hasattr(self, "_prior"):
            log_prior_scores += self._prior(matrix)

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

def low_rank_marginal_values(
    marginal_values: Sequence[Sequence[PHMMValueSet]],
    low_rank: int = 2,
) -> Sequence[PHMMValueSet]:
    """Computes low rank initial values.

    Args:
        marginal_values (Sequence[Sequence[PHMMValueSet]]): Value sets for
            the marginal distribution.
        low_rank (int): The rank of the low-rank approximation.

    Returns:
        Sequence[PHMMValueSet]: The joint value sets with low rank kernel
            values.
    """
    _assert_value_sets(marginal_values)
    assert len(marginal_values) == 2,\
        "Low-rank initialization is only supported for exactly two marginals."

    kernel_values: list[PHMMValueSet] = []

    for h in range(len(marginal_values[0])):
        k = 2  # Default low rank
        A, B = AB_from_marginals(
            marginal_values[0][h].match_emissions,
            marginal_values[1][h].match_emissions,
            low_rank,
        )
        joint_match_emissions = flatten_AB(A, B)
        joint_insert_emissions = flatten_AB(
            *AB_from_marginals(
                marginal_values[0][h].insert_emissions,
                marginal_values[1][h].insert_emissions,
                low_rank,
            )
        )
        kernel_values.append(
            PHMMValueSet(
                L=marginal_values[0][h].L,
                match_emissions=joint_match_emissions,
                insert_emissions=joint_insert_emissions,
                transitions=np.empty(()),
                start=np.empty(()),
            )
        )
    return kernel_values

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

def AB_from_marginals(
    p1: np.ndarray,
    p2: np.ndarray,
    low_rank: int,
    noise_std: float = 1e-2,
    epsilon: float = 1e-16,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build A (..., n1, r), B (..., n2, r) so that
    softmax(flatten(A @ B.T)) starts out equal to the independent
    joint p1_i * p2_j (per batch element), with any extra (k - 2)
    columns randomly initialized so gradients can break symmetry and
    learn interaction/dependence.

    Args:
        p1 (np.ndarray): The first marginal distribution of shape (..., n1).
        p2 (np.ndarray): The second marginal distribution of shape (..., n2).
        low_rank (int): The rank of the low-rank approximation.
        noise_std (float): The standard deviation of the noise added to the
            extra columns. Defaults to 1e-2.
        epsilon (float): A small value to avoid log(0). Defaults to 1e-16.
        seed (Optional[int]): The random seed for reproducibility.
            Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the A and B matrices
            of shapes (..., n1, low_rank) and (..., n2, low_rank), respectively.
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    n1, n2 = p1.shape[-1], p2.shape[-1]
    batch_shape = np.broadcast_shapes(p1.shape[:-1], p2.shape[:-1])

    assert np.all(np.isclose(p1.sum(axis=-1), 1.0, atol=1e-6)), (
        "each row of p1 must sum to 1"
    )
    assert np.all(np.isclose(p2.sum(axis=-1), 1.0, atol=1e-6)), (
        "each row of p2 must sum to 1"
    )
    assert low_rank >= 2, "low_rank must be at least 2"

    rng = np.random.default_rng(seed)
    A = np.zeros(batch_shape + (n1, low_rank), dtype=np.float64)
    B = np.zeros(batch_shape + (n2, low_rank), dtype=np.float64)

    A[..., 0] = np.log(p1 + epsilon)
    B[..., 0] = 1.0
    A[..., 1] = 1.0
    B[..., 1] = np.log(p2 + epsilon)

    # Extra columns: zero on one side, small noise on the other. This
    # keeps the initial joint == p1 (x) p2 exactly (the extra columns
    # contribute 0 to the logits) while still giving nonzero gradients
    # on the noisy side from the very first training step.
    if low_rank > 2:
        A[..., 2:] = rng.normal(
            scale=noise_std, size=batch_shape + (n1, low_rank - 2)
        )
        B[..., 2:] = 0.0

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