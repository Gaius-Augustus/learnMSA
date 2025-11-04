"""Language model configuration parameters."""

from pydantic import BaseModel


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
