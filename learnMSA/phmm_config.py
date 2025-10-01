from collections.abc import Sequence
from typing import ClassVar
from pydantic import field_validator

from hidten.config import ModelConfig


class ProfileHMMConfig(ModelConfig):

    lengths: Sequence[int]
    """The number of match states in each head of the pHMM.
    """

    @property
    def states(self) -> Sequence[int]:
        """The total number of states in each head."""
        return [2*L+3 for L in self.lengths]

    @property
    def states_explicit(self) -> Sequence[int]:
        """The total number of states in each head including silent states
        such as deletes.
        """
        return [3*L+5 for L in self.lengths]

    @property
    def heads(self) -> int:
        """The number of pHMM heads."""
        return len(self.lengths)
