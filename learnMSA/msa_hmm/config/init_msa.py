"""MSA initialization configuration parameters."""

from pathlib import Path
from pydantic import BaseModel, field_validator


class InitMSAConfig(BaseModel):
    """Parameters for initializing with existing MSA."""

    from_msa: Path | None = None
    """If set, the initial HMM parameters will inferred from the
    provided MSA in FASTA format."""

    match_threshold: float = 0.5
    """When inferring HMM parameters from an MSA, a column is
    considered a match state if its occupancy (fraction of non-gap
    characters) is at least this value."""

    global_factor: float = 0.1
    """A value in [0, 1] that describes the degree to which the MSA
    provided with --from_msa is considered a global alignment. This
    value is used as a mixing factor and affects how states are counted
    when the data contains fragmentary sequences. A global alignment
    counts flanks as deletions, while a local alignment counts them as
    jumps into the profile using only a single edge."""

    random_scale: float = 1e-3
    """When initializing from an MSA, the initial parameters are
    slightly perturbed by random noise. This parameter controls the
    scale of the noise."""

    pseudocounts: bool = False
    """If set, pseudocounts inferred from Dirichlet priors will be
    added on state transition and emissions counted in the MSA
    input via --from_msa."""

    @field_validator("match_threshold", "global_factor")
    def validate_quantiles(cls, v: float, info) -> float:
        if not 0 <= v <= 1:
            raise ValueError(f"{info.field_name} must be in the range [0, 1].")
        return v

    @field_validator("random_scale")
    def validate_positive_floats(cls, v: float, info) -> float:
        if v <= 0:
            raise ValueError(f"{info.field_name} must be greater than 0.")
        return v
