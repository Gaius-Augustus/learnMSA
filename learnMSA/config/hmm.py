from collections.abc import Sequence
from typing import ClassVar

from pydantic import BaseModel, field_validator


class HMMConfig(BaseModel):
    """HMM parameters."""

    alphabet: str = "ARNDCQEGHILKMFPSTWYVXUO"
    """The alphabet used in the HMM emissions."""

    amino_acid_background_distribution: Sequence[float] = [
        8.34333437e-02, 5.19266823e-02, 4.93510863e-02, 4.65871696e-02,
        2.24936164e-02, 5.06824822e-02, 6.29644485e-02, 4.72142352e-02,
        3.34919201e-02, 5.26777168e-02, 7.33173001e-02, 6.35075307e-02,
        3.52617111e-02, 3.60992714e-02, 3.46065678e-02, 7.21237089e-02,
        6.52571875e-02, 1.77631364e-02, 3.39407154e-02, 6.65086610e-02,
        7.91449952e-04, 5.83794314e-08, 9.99208434e-33
    ]
    """Default, background distribution over the amino acid alphabet."""

    p_begin_match: float | Sequence[float] | Sequence[Sequence[float]]\
        = 0.5
    """If provided a scalar value, is interpreted as ``P(Match 1 | Begin)``.
    In that case, ``P(Match i | Begin)`` for i > 1 will be chosen uniformly
    depending on head length.
    ``P(Match i | Begin; h)`` for all i and h can also be provided explicitly.
    """

    p_match_match: float | Sequence[float] | Sequence[Sequence[float]]\
        = 0.7
    """Defines ``P(Match i+1 | Match i; h)``.
    Can optionally depend on i and h.
    """

    p_match_insert: float | Sequence[float] | Sequence[Sequence[float]]\
        = 0.1
    """Defines ``P(Insert i | Match i; h)``.
    Can optionally depend on i and h.
    """

    p_match_end: float | Sequence[float] | Sequence[Sequence[float]]\
        = 0.1
    """Defines ``P(End | Match i; h)``.
    Can optionally depend on i and h.
    """

    p_insert_insert: float | Sequence[float] | Sequence[Sequence[float]]\
        = 0.38
    """Defines ``P(Insert i | Insert i; h)``.
    Can optionally depend on i and h.
    """

    p_delete_delete: float | Sequence[float] | Sequence[Sequence[float]]\
        = 0.38
    """Defines ``P(Delete i+1 | Delete i; h)``.
    Can optionally depend on i and h.
    """

    p_begin_delete: float | Sequence[float] = 0.1
    """Defines ``P(Delete 1 | Begin; h)``.
    Can optionally depend on h. This value is not used, if ``p_begin_match``
    is provided as a nested list.
    """

    p_left_left: float | Sequence[float] = 0.7
    """Defines ``P(Left Flank | Left Flank; h)``.
    Can optionally depend on h.
    """

    p_right_right: float | Sequence[float] = 0.7
    """Defines ``P(Right Flank | Right Flank; h)``.
    Can optionally depend on h.
    """

    p_unannot_unannot: float | Sequence[float] = 0.7
    """Defines ``P(Unannotated | Unannotated; h)``.
    Can optionally depend on h.
    """

    p_end_unannot: float | Sequence[float] = 1e-5
    """Defines ``P(Unannotated | End; h)``.
    Can optionally depend on h.
    """

    p_end_right: float | Sequence[float] = 0.5
    """Defines ``P(Right Flank | End; h)``.
    Can optionally depend on h.
    """

    p_start_left_flank: float | Sequence[float] = 0.5
    """Defines the starting probability ``P(Left Flank; h)``."""

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
        "p_unannot_unannot",
        "p_end_unannot",
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


def get_value(param, head: int, index: int | None = None) -> float:
    """Get the value of a parameter for a specific head and index.

    Args:
        param: The parameter which can be a float, a sequence of floats,
               or a sequence of sequences of floats.
        head: The head index.
        index: The index within the head, if applicable.
    """
    if isinstance(param, float):
        return param
    elif isinstance(param, Sequence) and not isinstance(param, str):
        if all(isinstance(x, float) for x in param):
            return param[head]
        elif (
            all(
                isinstance(x, Sequence) and not isinstance(x, str)
                for x in param
            )
            and all(all(isinstance(y, float) for y in x) for x in param)
        ):
            if index is None:
                raise ValueError("Index must be provided for nested sequences.")
            return param[head][index]
    raise ValueError("Invalid parameter type.")
