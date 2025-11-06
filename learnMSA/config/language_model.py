from pydantic import BaseModel, field_validator


class LanguageModelConfig(BaseModel):
    """Protein language model integration parameters."""

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


    @field_validator("language_model")
    def validate_language_model(cls, v: str) -> str:
        if not v in {"protT5", "esm2", "proteinBERT"}:
            raise ValueError(
                "language_model must be one of 'protT5' or 'esm2' or "\
                "'proteinBERT'."
            )
        return v

    @field_validator("scoring_model_dim", "embedding_prior_components")
    def validate_positive_ints(cls, v: int, info) -> int:
        if v <= 0:
            raise ValueError(f"{info.field_name} must be greater than 0.")
        return v

    @field_validator("temperature")
    def validate_positive_floats(cls, v: float, info) -> float:
        if v <= 0:
            raise ValueError(f"{info.field_name} must be greater than 0.")
        return v

    @field_validator("L2_insert", "L2_match")
    def validate_nonnegative_floats(cls, v: float, info) -> float:
        if v < 0:
            raise ValueError(f"{info.field_name} must be non-negative.")
        return v