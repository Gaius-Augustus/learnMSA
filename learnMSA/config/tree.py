from pydantic import BaseModel, ConfigDict


class TreeConfig(BaseModel):
    """Parameters related to the ancestral probabilities (tree) layer."""

    model_config = ConfigDict(extra="forbid")

    use_anc_probs: bool = True
    """Whether to use ancestral state probabilities."""

    trainable_rates: bool = True
    """Whether the per-sequence evolutionary rates are trainable."""

    trainable_exchangeabilities: bool = False
    """Whether exchangeability matrices are trainable."""

    trainable_equilibrium: bool = False
    """Whether equilibrium distributions are trainable."""

    shared_equilibrium: bool = True
    """If True, all mixture components share a single equilibrium distribution."""

    shared_exchangeabilities: bool = False
    """If True, all mixture components share a single exchangeability matrix."""

    exchangeability_noise_std: float = 0.02
    """Noise std for exchangeability matrix perturbation in LG initialization."""

    exchangeability_l2: float = 0.0
    """L2 regularization strength for exchangeability parameters."""

    num_anc_probs_components: int = 1
    """Number of mixture components in the ancestral probabilities layer."""

    low_rank: int | None = None
    """If not None, the rank of the low-rank parameterization of the
    exchangeability matrices. If None, full kernels are used."""
