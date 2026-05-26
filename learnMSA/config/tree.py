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

    shared_exchangeabilities: bool = True
    """If True, all mixture components share a single exchangeability matrix."""

    exchangeability_noise_std: float = 0.05
    """Noise std for exchangeability matrix perturbation in LG initialization."""

    equilibrium_noise_std: float = 0.01
    """Noise std for equilibrium distribution perturbation in LG initialization."""

    num_anc_probs_components: int = 1
    """Number of mixture components in the ancestral probabilities layer."""
