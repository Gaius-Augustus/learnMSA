from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

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

    prior_name: str = "pfam_35_3Di"
    """Specifies the path to weights for a Dirichlet prior over the
    structural alphabet."""

    prior_components: int = 9
    """The number of mixture components for the Dirichlet prior."""

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

    @property
    def alphabet_size(self) -> int:
        """The size of the alphabet."""
        return len(self.structural_alphabet)
