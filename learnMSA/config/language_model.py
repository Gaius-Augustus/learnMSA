from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

from .util import NPArray


class LanguageModelConfig(BaseModel):
    """Protein language model integration parameters."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    use_language_model: bool = False
    """Uses a large protein lanague model to generate per-token
    embeddings that guide the MSA step."""

    plm_cache_dir: str | None = None
    """Directory where the protein language model is stored."""

    language_model: str = "protT5"
    """Name of the language model to use."""

    scoring_model_dim: int = 16
    """Reduced embedding dimension of the scoring model."""

    scoring_model_activation: str = "sigmoid"
    """Activation function of the scoring model."""

    scoring_model_suffix: str = ""
    """Suffix to identify a specific scoring model."""

    temperature: float = 3.0
    """Temperature of the softmax function."""

    temperature_mode: str = "trainable"
    """Temperature mode."""

    use_L2: bool = False
    """Use L2 regularization."""

    L2_match: float = 0.0
    """L2 regularization for match states."""

    L2_insert: float = 1000.0
    """L2 regularization for insert states."""

    embedding_prior_components: int = 32
    """Number of embedding prior components."""

    conditionally_independent: bool = True
    """Whether to use conditionally independent emissions."""

    variance_init_stdev: float = 0.02
    """Initial standard deviation for the normal distribution."""

    inverse_gamma_alpha: float = 3.0
    """Alpha parameter for the inverse gamma prior on variances."""

    inverse_gamma_beta: float = 0.5
    """Beta parameter for the inverse gamma prior on variances."""

    match_expectations: (Sequence[float] | Sequence[Sequence[float]] |
                        Sequence[Sequence[Sequence[float]]] |
                        NPArray | None) = None
    """Optional initialization for match state expectations.
    Can be:
    - None: Initialize with zeros (default).
    - Sequence[float] of length scoring_model_dim: Same for all match states
        in all heads.
    - Sequence[Sequence[float]] of shape (num_heads, scoring_model_dim):
      Head-specific expectations, same for all match states within a head.
    - Sequence[Sequence[Sequence[float]]] of shape
        `(num_heads, length[h], scoring_model_dim)`:
        Fully specified match state expectations for each position in each head.
    """

    match_stddev: (Sequence[float] | Sequence[Sequence[float]] |
                   Sequence[Sequence[Sequence[float]]] |
                   NPArray | None) = None
    """Optional initialization for match state standard deviations.
    Can be:
    - None: Initialize with random normal values (default).
    - Sequence[float] of length scoring_model_dim: Same for all match states
        in all heads.
    - Sequence[Sequence[float]] of shape `(num_heads, scoring_model_dim)`:
      Head-specific standard deviations, same for all match states within a head.
    - Sequence[Sequence[Sequence[float]]] of shape
        `(num_heads, length[h], scoring_model_dim)`: Fully specified match
        state standard deviations for each position in each head.
    """

    insert_expectation: (Sequence[float] | Sequence[Sequence[float]] |
                         NPArray | None) = None
    """Optional initialization for insert state expectations.
    Can be:
    - None: Initialize with zeros (default).
    - Sequence[float] of length scoring_model_dim: Same for all heads.
    - Sequence[Sequence[float]] of shape `(num_heads, scoring_model_dim)`:
        Head-specific insert expectations.
    """

    insert_stddev: (Sequence[float] | Sequence[Sequence[float]] |
                    NPArray | None) = None
    """Optional initialization for insert state standard deviations.
    Can be:
    - None: Initialize with random normal values (default).
    - Sequence[float] of length scoring_model_dim: Same for all heads.
    - Sequence[Sequence[float]] of shape `(num_heads, scoring_model_dim)`:
        Head-specific insert standard deviations.
    """


    def id_string(self) -> str:
        """Generate an identifier string for the language model configuration.

        Returns:
            A string that uniquely identifies the language model configuration.
        """
        return (
            f"{self.language_model}_{self.scoring_model_dim}_reduced_"
            f"mix{self.embedding_prior_components}_"
            f"{self.scoring_model_activation}"
        )

    @field_validator("language_model")
    def validate_language_model(cls, v: str) -> str:
        if not v in {"protT5", "esm2", "proteinBERT", "zeros"}:
            raise ValueError(
                "language_model must be one of 'protT5', 'esm2', "
                "'proteinBERT', or 'zeros'."
            )
        return v

    @field_validator("scoring_model_dim", "embedding_prior_components")
    def validate_positive_ints(cls, v: int, info) -> int:
        if v <= 0:
            raise ValueError(f"{info.field_name} must be greater than 0.")
        return v

    @field_validator("temperature", "inverse_gamma_alpha", "inverse_gamma_beta")
    def validate_positive_floats(cls, v: float, info) -> float:
        if v <= 0:
            raise ValueError(f"{info.field_name} must be greater than 0.")
        return v

    @field_validator("L2_insert", "L2_match")
    def validate_nonnegative_floats(cls, v: float, info) -> float:
        if v < 0:
            raise ValueError(f"{info.field_name} must be non-negative.")
        return v