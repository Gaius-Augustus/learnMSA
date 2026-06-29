import sys
from collections.abc import Sequence

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import numpy as np
import tensorflow as tf
from hidten.hmm import HMMConfig
from hidten.tf.emitter.categorical import T_shapelike, T_TFTensor, TFCategoricalEmitter
from hidten.tf.prior import TFPrior

from learnMSA.hmm.util.value_set import PHMMValueSet

from .profile_emitter import ProfileEmitter


class JointProfileEmitter(ProfileEmitter):
    """A profile emitter for the joint distribution of multiple categorical
    variables. Allows to apply individual priors to the marginal distributions.
    Also allows to initialize the joint distribution from the product of
    marginal distributions.
    """

    _marginal_priors: dict[int, TFPrior] = {}

    def __init__(
        self,
        values: Sequence[PHMMValueSet] | None = None,
        marginal_values: Sequence[Sequence[PHMMValueSet]] | None = None,
        trainable_insertions: bool = True,
        use_full_matmul: bool = True,
        temperature: float = 1.0,
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
        """
        _values : Sequence[PHMMValueSet]
        if values is not None:
            _values = values
        else:
            assert marginal_values is not None,\
                "Either `values` or `marginal_values` must be provided."
            _values = _product_marginal_values(marginal_values)

        if len(_values) == 0:
            raise ValueError("At least one value set must be provided.")

        super().__init__(
            values=_values,
            trainable_insertions=trainable_insertions,
            use_full_matmul=use_full_matmul,
            temperature=temperature,
            **kwargs
        )

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
        s = self.alphabet_size
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
            input_dim = np.prod(self.marginal_dims) # type: ignore
            input_shape = (None, None, input_dim)

        self.share = self._build_share()

        TFCategoricalEmitter.build(self, input_shape)

    @override
    def call(
        self, *emissions: T_TFTensor, use_padding: bool = True,
    ) -> T_TFTensor:
        observation_product = _outer_product_flat(*emissions)
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
            marginal_matrices.append(_marginal_matrix(matrix, i))
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


def _product_marginal_values(
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
    assert len(marginal_values) > 0,\
        "At least one marginal value set is required."
    assert all(len(marginal_values[0]) == len(mv) for mv in marginal_values),\
        "All marginal value sets must have the same number of heads."
    for h in range(len(marginal_values[0])):
        assert all(marginal_values[0][h].L == mv[h].L for mv in marginal_values),\
            "All marginal value sets must have the same length for each head."

    joint_values: list[PHMMValueSet] = []
    for h in range(len(marginal_values[0])):
        match_emission = _outer_product_flat(
            *[tf.constant(mv[h].match_emissions) for mv in marginal_values]
        ).numpy()
        insert_emission = _outer_product_flat(
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

def _outer_product_flat(*emissions: T_TFTensor) -> T_TFTensor:
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
    x = _outer_product_flat_pw(emissions[0], emissions[1])
    for obs in emissions[2:]:
        x = _outer_product_flat_pw(x, obs)
    return x

def _outer_product_flat_pw(x: T_TFTensor, y: T_TFTensor) -> T_TFTensor:
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

def _marginal_matrix(matrix: T_TFTensor, i: int) -> T_TFTensor:
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
