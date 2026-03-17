from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from .util import NPArray


class StructureConfig(BaseModel):
    """Configuration for settings related to protein structure."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    use_structure: bool = False
    """Whether to use structural information."""

    structural_alphabet: str = "ARNDCQEGHILKMFPSTWYV"
    """The structural alphabet. Default: 3Di."""

    background_distribution: Sequence[float] | NPArray = np.ones(20) / 20
    """Default, background distribution over the structural alphabet."""

    @property
    def alphabet_size(self) -> int:
        """The size of the alphabet."""
        return len(self.structural_alphabet)
