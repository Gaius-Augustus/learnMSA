from collections.abc import Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from .util import NPArray


class StructureConfig(BaseModel):
    """Configuration for settings related to protein structure."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    use_structure: bool = False
    """Whether to use structural information."""

    structural_alphabet: str = "ACDEFGHIKLMNPQRSTVWY"
    """The structural alphabet. Default: 3Di."""

    background_distribution: Sequence[float] | NPArray = np.array([
        0.034426975947981144, 0.033208414219209156, 0.18404163279880928,
        0.018845173581160408, 0.023106037152191602, 0.024921394845384612,
        0.028174588152144145, 0.016488184182630747, 0.014660738266896399,
        0.08201603241748119, 0.006207818999515547, 0.02770227060714815,
        0.0890314292376826, 0.04996411034912452, 0.031582146390535824,
        0.07714301207074263, 0.015701061346461074, 0.20506960741456495,
        0.017787020895291238, 0.019922351125044764
    ])
    """Default, background distribution over the structural alphabet based on
    (AF2 SwissProt). Source: hmmer3di repository."""

    prior_name: str = "scop_3Di_3_20"
    """Specifies the path to weights for a Dirichlet prior over the
    structural alphabet."""

    prior_components: int = 1
    """The number of mixture components for the Dirichlet prior."""

    prior_temperature: float = 1.0
    """Temperature applied as a factor (1/temperature) to the log prior scores
    """

    use_prior_for_emission_init: bool = True
    """Whether to use the prior distribution for initializing the structural
    emission parameters."""

    emitter_temperature: float = 1.0
    """Temperature applied as an exponent (1/temperature) to the structural
    emission scores."""

    reset_after_surgery: bool = False
    """Whether to reset the structural information emission parameters after
    model surgery. default: False."""

    mask_ends: bool = True
    """Whether to replace the first and last input with unknown values. Only
    applies to structural FASTA input, not logits. This should be enabled for
    structural alphabets that have no meaningful representation of the sequence
    ends, such as Foldseek 3Di. Disable for predicted ProstT5 3Di."""

    joint_emissions: bool = False
    """Whether to model structural tokens conditionally on the amino acid. If
    True, the structural emitter is replaced by a joint emitter for
    ``P(structural token | amino acid, state)`` next to the amino acid emitter
    ``P(amino acid | state)``. Every conditional row is initialized with the
    structural emission initialization and regularized according to
    ``joint_row_prior``. (EXPERIMENTAL)"""

    joint_row_prior: Literal["hierarchical", "per_conditional"] = \
        "hierarchical"
    """The prior on the conditional rows of the joint emitter.

    - ``"hierarchical"``: the structural Dirichlet prior is applied once per
      state to the implied marginal ``sum_a P(a | s) P(. | a, s)`` and each row
      is shrunk toward that marginal with ``Dir(c * marginal + 1)``, where
      ``c`` is ``joint_row_concentration``. Rows without data fall back to the
      marginal.
    - ``"per_conditional"``: the structural Dirichlet prior is applied to every
      conditional row.
    """

    joint_row_concentration: float = 20.0
    """Pseudo-count mass per conditional row with which the hierarchical row
    prior pulls each row toward the state's structural marginal. 0 disables
    the row term, so only the marginal is regularized."""

    joint_emission_low_rank: int = 0
    """If joint_emissions is True, this specifies the rank of the low-rank
    approximation of the joint emission matrix."""

    input_format: Literal["tokens", "logits"] = "tokens"
    """Set when the structural data are loaded: ``"tokens"`` for a 3Di FASTA
    file, ``"logits"`` for per-residue 3Di logits predicted by ProstT5 (an
    ``.npz`` written by ``learnMSA-3di``)."""

    input_temperature: float = 1.24
    """Temperature T of the ProstT5 3Di logits z: the posterior over the
    true 3Di letters is ``softmax(z / T)``. Calibrated on the 3Di letters of
    SCOP2 domains (``util/calibrate_prostt5.py``). Unused for a 3Di FASTA
    file."""

    match_emissions: (Sequence[float] | Sequence[Sequence[float]] |
                      Sequence[Sequence[Sequence[float]]] |
                      NPArray | None) = None
    """Defines the emission distribution ``P(structural token | Match i; h)``.
    Can be:
    - None: Use background_distribution for all match states (default).
    - Sequence[float] of length alphabet_size: Same distribution for all
      match states in all heads.
    - Sequence[Sequence[float]] of shape (num_heads, alphabet_size):
      Head-specific distributions, same for all match states within a head.
    - Sequence[Sequence[Sequence[float]]] of shape (num_heads, length[h],
      alphabet_size): Fully specified match state emissions for each position
      in each head.
    """

    insert_emissions: (Sequence[float] | Sequence[Sequence[float]] |
                       NPArray | None) = None
    """Defines the emission distribution ``P(structural token | Insert; h)``.
    Can be:
    - None: Use background_distribution for all heads (default).
    - Sequence[float] of length alphabet_size: Same distribution for all heads.
    - Sequence[Sequence[float]] of shape (num_heads, alphabet_size):
      Head-specific insertion distributions.
    """

    @field_validator("joint_row_concentration")
    def validate_joint_row_concentration(cls, v: float) -> float:
        if v < 0:
            raise ValueError("joint_row_concentration must be non-negative.")
        return v

    @field_validator("input_temperature")
    def validate_input_temperature(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("input_temperature must be positive.")
        return v

    @model_validator(mode="before")
    @classmethod
    def _legacy_fields(cls, data):
        """Saved models from earlier versions of the logit input."""
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if data.get("input_format") == "prostt5_logits":
            data["input_format"] = "logits"
        if data.get("input_temperature") == -1:
            # The former "auto" value.
            del data["input_temperature"]
        if "soft_input_temperature" in data:
            data.setdefault("input_temperature",
                            data.pop("soft_input_temperature"))
        return data

    @property
    def alphabet_size(self) -> int:
        """The size of the alphabet."""
        return len(self.structural_alphabet)
