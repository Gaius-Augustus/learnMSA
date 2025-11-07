from pydantic import BaseModel, field_validator


class AdvancedConfig(BaseModel):
    """Advanced/Development parameters."""

    dist_out: str = ""
    """Distribution output file."""

    alpha_flank: float = 7000
    """Alpha parameter for flank."""

    alpha_single: float = 1e9
    """Alpha parameter for single."""

    alpha_global: float = 1e4
    """Alpha parameter for global."""

    alpha_flank_compl: float = 1
    """Alpha parameter for flank complement."""

    alpha_single_compl: float = 1
    """Alpha parameter for single complement."""

    alpha_global_compl: float = 1
    """Alpha parameter for global complement."""

    inverse_gamma_alpha: float = 3.0
    """Inverse gamma alpha parameter."""

    inverse_gamma_beta: float = 0.5
    """Inverse gamma beta parameter."""

    frozen_distances: bool = False
    """Freeze distances during training."""

    initial_distance: float = 0.05
    """Initial distance value."""

    grow_mem: bool = False
    """Enable memory growth for GPUs."""

    insertion_aligner: str = "famsa"
    """Insertion aligner to use."""

    aligner_threads: int = 0
    """Number of threads to use for the aligner."""

    @field_validator(
        "alpha_flank",
        "alpha_single",
        "alpha_global",
        "alpha_flank_compl",
        "alpha_single_compl",
        "alpha_global_compl",
        "inverse_gamma_alpha",
        "inverse_gamma_beta",
        "initial_distance",
    )
    def validate_quantiles(cls, v: float, info) -> float:
        if not v > 0:
            raise ValueError(f"{info.field_name} must be greater than 0.")
        return v