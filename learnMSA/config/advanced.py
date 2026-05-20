from pydantic import BaseModel, field_validator


class AdvancedConfig(BaseModel):
    """Advanced/Development parameters."""

    dist_out: str = ""
    """Distribution output file."""

    initial_distance: float = 0.05
    """Initial distance value."""

    insertion_aligner: str = "famsa"
    """Insertion aligner to use."""

    aligner_threads: int = 0
    """Number of threads to use for the aligner."""

    jit_compile: bool = True
    """Enable XLA JIT compilation in TensorFlow."""

    reset_branch_lengths: bool = True
    """Whether to reset the branch lengths (tau) before training."""

    reset_evo_model: bool = False
    """Whether to reset the evolutionary model parameters (exchangeabilities, equilibrium, and mixture) before training."""

    @field_validator(
        "initial_distance",
    )
    def validate_quantiles(cls, v: float, info) -> float:
        if not v > 0:
            raise ValueError(f"{info.field_name} must be greater than 0.")
        return v