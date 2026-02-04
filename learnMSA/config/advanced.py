from pydantic import BaseModel, field_validator


class AdvancedConfig(BaseModel):
    """Advanced/Development parameters."""

    dist_out: str = ""
    """Distribution output file."""

    initial_distance: float = 0.05
    """Initial distance value."""

    grow_mem: bool = False
    """Enable memory growth for GPUs."""

    insertion_aligner: str = "famsa"
    """Insertion aligner to use."""

    aligner_threads: int = 0
    """Number of threads to use for the aligner."""

    jit_compile: bool = True
    """Enable XLA JIT compilation in TensorFlow."""

    @field_validator(
        "initial_distance",
    )
    def validate_quantiles(cls, v: float, info) -> float:
        if not v > 0:
            raise ValueError(f"{info.field_name} must be greater than 0.")
        return v