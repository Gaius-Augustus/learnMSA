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

    p_begin_match: float | Sequence[float] | Sequence[Sequence[float]]\
        = 0.5
    """If provided a scalar value, is interpreted as P(Match 1 | Begin).
    In that case, P(Match i | Begin) for i > 1 will be chosen uniformly
    depending on head length.
    P(Match i | Begin; h) for all i and h can also be provided explicitly.
    """

    p_match_match: float | Sequence[float] | Sequence[Sequence[float]]\
        = 0.7
    """Defines P(Match i+1 | Match i; h). Can optionally depend on i and h"""

    p_match_insert: float | Sequence[float] | Sequence[Sequence[float]]\
        = 0.1
    """Defines P(Insert i | Match i; h). Can optionally depend on i and h"""

    p_match_end: float | Sequence[float] | Sequence[Sequence[float]]\
        = 0.1
    """Defines P(End | Match i; h). Can optionally depend on i and h"""

    p_insert_insert: float | Sequence[float] | Sequence[Sequence[float]]\
        = 0.38
    """Defines P(Insert i | Insert i; h). Can optionally depend on i and h"""

    p_delete_delete: float | Sequence[float] | Sequence[Sequence[float]]\
        = 0.38
    """Defines P(Delete i+1 | Delete i; h). Can optionally depend on i and h.
    """

    p_left_left: float | Sequence[float] = 0.7
    """Defines P(Left Flank | Left Flank; h). Can optionally depend on h."""

    p_right_right: float | Sequence[float] = 0.7
    """Defines P(Right Flank | Right Flank; h). Can optionally depend on h."""

    p_unanno_unanno: float | Sequence[float] = 0.7
    """Defines P(Unannotated | Unannotated; h). Can optionally depend on h."""

    p_end_unanno: float | Sequence[float] = 1e-5
    """Defines P(Unannotated | End; h). Can optionally depend on h."""

    p_end_right: float | Sequence[float] = 0.5
    """Defines P(Right Flank | End; h). Can optionally depend on h."""

    _length_offsets: ClassVar[dict[str, int]] = {
        "p_begin_match": 0,
        "p_match_match": 0,
        "p_match_insert": -1,
        "p_match_end": -1,
        "p_insert_insert": -1,
        "p_delete_delete": 0,
    }

    @field_validator(
        "p_begin_match",
        "p_match_match",
        "p_match_insert",
        "p_match_end",
        "p_insert_insert",
        "p_delete_delete",
        "p_left_left",
        "p_right_right",
        "p_unanno_unanno",
        "p_end_unanno",
        "p_end_right",
    )
    def check_length(cls, v, info):
        field = info.field_name
        lengths = info.data.get('lengths')
        offset = cls._length_offsets.get(field, 0)
        if lengths is None:
            return v

        # Case 1: float
        if isinstance(v, float):
            return v

        # Case 2: Sequence[float]
        if (isinstance(v, Sequence) and not isinstance(v, str)
            and all(isinstance(x, float) for x in v)):
            if len(v) != len(lengths):
                raise ValueError(
                    f"{field} must have length {len(lengths)} or be a float."
                )
            return v

        # Case 3: Sequence[Sequence[float]]
        if (isinstance(v, Sequence) and not isinstance(v, str)
            and all(
                isinstance(x, Sequence) and not isinstance(x, str) for x in v)
            ):
            if len(v) != len(lengths):
                raise ValueError(
                    f"{field} must have outer length {len(lengths)}."
                )
            for i, inner in enumerate(v):
                expected_inner_len = lengths[i] + offset
                if len(inner) != expected_inner_len:
                    raise ValueError(
                        f"{field}[{i}] must have length {expected_inner_len}."
                    )
                if not all(isinstance(x, float) for x in inner):
                    raise ValueError(
                        f"All elements of {field}[{i}] must be floats."
                    )
            return v

        raise ValueError(
            f"{field} must be a float, a sequence of floats, or a sequence of "
            "sequences of floats."
        )
