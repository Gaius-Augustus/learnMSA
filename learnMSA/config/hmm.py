from collections.abc import Sequence
from typing import ClassVar

import numpy as np
from pydantic import (BaseModel, ConfigDict,
                      field_validator, model_validator)

from .util import NPArray


class PHMMPriorConfig(BaseModel):
    """HMM prior parameters for transition priors."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    use_amino_acid_prior: bool = True
    """Whether to use a Dirichlet prior for emissions."""

    alpha_flank: float = 7000.0
    """Alpha parameter for flank prior. Favors high probability of staying in
    flanking states."""

    alpha_single: float = 1e9
    """Alpha parameter for single-hit prior. Favors high probability for a
    single main model hit."""

    alpha_global: float = 1e4
    """Alpha parameter for global prior. Favors models with high prob. to
    enter at the first match and exit after the last match."""

    alpha_flank_compl: float = 1.0
    """Complement parameter for alpha_flank."""

    alpha_single_compl: float = 1e-3
    """Complement parameter for alpha_single."""

    alpha_global_compl: float = 1.0
    """Complement parameter for alpha_global."""

    epsilon: float = 1e-16
    """A small constant for numerical stability in prior computations."""

    @field_validator(
        "alpha_flank",
        "alpha_single",
        "alpha_global",
        "alpha_flank_compl",
        "alpha_single_compl",
        "alpha_global_compl",
        "epsilon",
    )
    def validate_alpha_params(cls, v: float, info) -> float:
        if not v > 0:
            raise ValueError(f"{info.field_name} must be greater than 0.")
        return v


class PHMMConfig(BaseModel):
    """HMM parameters."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    alphabet: str = "ARNDCQEGHILKMFPSTWYVXUO"
    """The alphabet used in the HMM emissions."""

    @property
    def alphabet_size(self) -> int:
        """The size of the alphabet."""
        return len(self.alphabet)

    background_distribution: Sequence[float] | NPArray = [
        8.34333437e-02, 5.19266823e-02, 4.93510863e-02, 4.65871696e-02,
        2.24936164e-02, 5.06824822e-02, 6.29644485e-02, 4.72142352e-02,
        3.34919201e-02, 5.26777168e-02, 7.33173001e-02, 6.35075307e-02,
        3.52617111e-02, 3.60992714e-02, 3.46065678e-02, 7.21237089e-02,
        6.52571875e-02, 1.77631364e-02, 3.39407154e-02, 6.65086610e-02,
        7.91449952e-04, 5.83794314e-08, 9.99208434e-33
    ]
    """Default, background distribution over the amino acid alphabet."""

    use_prior_for_emission_init: bool = True
    """Whether to use the amino acid prior distribution for initializing
    the emissions in the profile emitter. If False, the background_distribution
    is used for initialization. No effect when match_emissions or\
    insert_emissions are provided, respectively, for the relevant states.
    """

    match_emissions: (Sequence[float] | Sequence[Sequence[float]] |
                      Sequence[Sequence[Sequence[float]]] |
                      NPArray | None) = None
    """Defines the emission distribution ``P(amino acid | Match i; h)``.
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
    """Defines the emission distribution ``P(amino acid | Insert; h)``.
    Can be:
    - None: Use background_distribution for all heads (default).
    - Sequence[float] of length alphabet_size: Same distribution for all heads.
    - Sequence[Sequence[float]] of shape (num_heads, alphabet_size):
      Head-specific insertion distributions.
    """

    p_begin_match: float | Sequence[float] | Sequence[Sequence[float]] | \
        NPArray = 0.5
    """If provided a scalar value, is interpreted as ``P(Match 1 | Begin)``.
    In that case, ``P(Match i | Begin)`` for i > 1 will be chosen uniformly
    depending on head length.
    ``P(Match i | Begin; h)`` for all i and h can also be provided explicitly.
    """

    p_match_match: float | Sequence[float] | Sequence[Sequence[float]] | \
        NPArray = 0.78
    """Defines ``P(Match i+1 | Match i; h)``.
    Can optionally depend on i and h.
    """

    p_match_insert: float | Sequence[float] | Sequence[Sequence[float]] | \
        NPArray = 0.1
    """Defines ``P(Insert i | Match i; h)``.
    Can optionally depend on i and h.
    """

    p_match_delete: float | Sequence[float] | Sequence[Sequence[float]] | \
        NPArray = 0.12
    """Defines ``P(Delete i+1 | Match i; h)``.
    Can optionally depend on i and h.
    """

    p_match_end: float | Sequence[float] | Sequence[Sequence[float]] | \
        NPArray | None = None
    """Defines ``P(End | Match i; h)``.
    Can optionally depend on i and h.
    Per default it is determined automatically such that
    ``exit[i] = (1-p)/(L-1)`` for all match states i, where
    p = begin_match_1 + begin_delete_1.
    The other outgoing transitions from match i are rescaled accordingly.
    """

    p_insert_insert: float | Sequence[float] | Sequence[Sequence[float]] | \
        NPArray = 0.38
    """Defines ``P(Insert i | Insert i; h)``.
    Can optionally depend on i and h.
    """

    p_delete_delete: float | Sequence[float] | Sequence[Sequence[float]] | \
        NPArray = 0.38
    """Defines ``P(Delete i+1 | Delete i; h)``.
    Can optionally depend on i and h.
    """

    p_begin_delete: float | Sequence[float] | NPArray = 0.38
    """Defines ``P(Delete 1 | Begin; h)``.
    Can optionally depend on h. This value is not used, if ``p_begin_match``
    is provided as a nested list.
    """

    p_left_left: float | Sequence[float] | NPArray = 0.75
    """Defines ``P(Left Flank | Left Flank; h)``.
    Can optionally depend on h.
    """

    p_right_right: float | Sequence[float] | NPArray = 0.75
    """Defines ``P(Right Flank | Right Flank; h)``.
    Can optionally depend on h.
    """

    p_unannot_unannot: float | Sequence[float] | NPArray = 0.75
    """Defines ``P(Unannotated | Unannotated; h)``.
    Can optionally depend on h.
    """

    p_end_unannot: float | Sequence[float] | NPArray = 1e-6
    """Defines ``P(Unannotated | End; h)``.
    Can optionally depend on h.
    """

    p_end_right: float | Sequence[float] | NPArray = 0.5
    """Defines ``P(Right Flank | End; h)``.
    Can optionally depend on h.
    """

    p_start_left_flank: float | Sequence[float] | NPArray = 0.5
    """Defines the starting probability ``P(Left Flank; h)``."""

    use_noise: bool = False
    """Whether to add Dirichlet noise during HMM initialization."""

    noise_concentration: float = 100.0
    """Concentration of Dirichlet noise added during HMM initialization."""

    shared_flank_transitions: bool = True
    """Whether to share transition parameters of flank states within each
    head."""

    _length_offsets: ClassVar[dict[str, int]] = {
        "p_begin_match": 0,
        "p_match_match": 0,
        "p_match_insert": -1,
        "p_match_delete": -1,
        "p_match_end": -1,
        "p_insert_insert": -1,
        "p_delete_delete": 0,
        "match_emissions": 0,
    }

    @field_validator(
        "p_begin_match",
        "p_match_match",
        "p_match_insert",
        "p_match_delete",
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

        # Case 0: None (only for p_match_end)
        if v is None:
            if field == "p_match_end":
                return v
            else:
                raise ValueError(f"{field} cannot be None.")

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

    @field_validator("match_emissions")
    def check_match_emissions(cls, v, info):
        if v is None:
            return v

        lengths = info.data.get('lengths')
        alphabet = info.data.get('alphabet', 'ARNDCQEGHILKMFPSTWYVXUO')
        alphabet_size = len(alphabet)

        # Handle numpy arrays
        if isinstance(v, np.ndarray):
            if v.ndim == 1:
                # Case 1: 1D array - single distribution for all
                if len(v) != alphabet_size:
                    raise ValueError(
                        f"match_emissions must have length {alphabet_size} "
                        f"(alphabet size), got {len(v)}."
                    )
                return v
            elif v.ndim == 2:
                # Case 2: 2D array - head-specific, position-independent
                if lengths is not None and v.shape[0] != len(lengths):
                    raise ValueError(
                        f"match_emissions outer length must match number of heads "
                        f"({len(lengths)}), got {v.shape[0]}."
                    )
                if v.shape[1] != alphabet_size:
                    raise ValueError(
                        f"match_emissions inner length must be {alphabet_size} "
                        f"(alphabet size), got {v.shape[1]}."
                    )
                return v
            elif v.ndim == 3:
                # Case 3: 3D array - fully specified
                if lengths is not None and v.shape[0] != len(lengths):
                    raise ValueError(
                        f"match_emissions outer length must match number of heads "
                        f"({len(lengths)}), got {v.shape[0]}."
                    )
                for h in range(v.shape[0]):
                    if lengths is not None and v.shape[1] != lengths[h]:
                        raise ValueError(
                            f"match_emissions[{h}] must have length {lengths[h]} "
                            f"(number of match states), got {v.shape[1]}."
                        )
                    if v.shape[2] != alphabet_size:
                        raise ValueError(
                            f"match_emissions[{h}] inner length must be {alphabet_size} "
                            f"(alphabet size), got {v.shape[2]}."
                        )
                return v
            else:
                raise ValueError(
                    f"match_emissions numpy array must be 1D, 2D, or 3D, got {v.ndim}D"
                )

        # Case 1: Sequence[float] - single distribution for all
        if (isinstance(v, Sequence) and not isinstance(v, str)
            and all(isinstance(x, (float, int)) for x in v)):
            if len(v) != alphabet_size:
                raise ValueError(
                    f"match_emissions must have length {alphabet_size} "
                    f"(alphabet size), got {len(v)}."
                )
            return v

        # Case 2 or 3: nested sequences
        if (isinstance(v, Sequence) and not isinstance(v, str)
            and all(isinstance(x, Sequence) and not isinstance(x, str)
                    for x in v)):

            # Check if we can determine num_heads
            if lengths is not None and len(v) != len(lengths):
                raise ValueError(
                    f"match_emissions outer length must match number of heads "
                    f"({len(lengths)}), got {len(v)}."
                )

            for h, inner in enumerate(v):
                # Case 2: Sequence[Sequence[float]] - head-specific,
                # position-independent
                if all(isinstance(x, (float, int)) for x in inner):
                    if len(inner) != alphabet_size:
                        raise ValueError(
                            f"match_emissions[{h}] must have length "
                            f"{alphabet_size} (alphabet size), got {len(inner)}."
                        )
                # Case 3: Sequence[Sequence[Sequence[float]]] - fully specified
                elif all(isinstance(x, Sequence) and not isinstance(x, str)
                        for x in inner):
                    if lengths is not None and len(inner) != lengths[h]:
                        raise ValueError(
                            f"match_emissions[{h}] must have length {lengths[h]} "
                            f"(number of match states), got {len(inner)}."
                        )
                    for i, dist in enumerate(inner):
                        if not all(isinstance(x, (float, int)) for x in dist):
                            raise ValueError(
                                f"match_emissions[{h}][{i}] must contain only "
                                f"floats/ints."
                            )
                        if len(dist) != alphabet_size:
                            raise ValueError(
                                f"match_emissions[{h}][{i}] must have length "
                                f"{alphabet_size} (alphabet size), got {len(dist)}."
                            )
                else:
                    raise ValueError(
                        f"match_emissions[{h}] has invalid structure."
                    )
            return v

        raise ValueError(
            "match_emissions must be None, a sequence of floats, a sequence of "
            "sequences of floats, or a sequence of sequences of sequences of floats."
        )

    @field_validator("insert_emissions")
    def check_insert_emissions(cls, v, info):
        if v is None:
            return v

        lengths = info.data.get('lengths')
        alphabet = info.data.get('alphabet', 'ARNDCQEGHILKMFPSTWYVXUO')
        alphabet_size = len(alphabet)

        # Handle numpy arrays
        if isinstance(v, np.ndarray):
            if v.ndim == 1:
                # Case 1: 1D array - single distribution for all heads
                if len(v) != alphabet_size:
                    raise ValueError(
                        f"insert_emissions must have length {alphabet_size} "
                        f"(alphabet size), got {len(v)}."
                    )
                return v
            elif v.ndim == 2:
                # Case 2: 2D array - head-specific distributions
                if lengths is not None and v.shape[0] != len(lengths):
                    raise ValueError(
                        f"insert_emissions outer length must match number of heads "
                        f"({len(lengths)}), got {v.shape[0]}."
                    )
                if v.shape[1] != alphabet_size:
                    raise ValueError(
                        f"insert_emissions inner length must be {alphabet_size} "
                        f"(alphabet size), got {v.shape[1]}."
                    )
                return v
            else:
                raise ValueError(
                    f"insert_emissions numpy array must be 1D or 2D, got {v.ndim}D"
                )

        # Case 1: Sequence[float] - single distribution for all heads
        if (isinstance(v, Sequence) and not isinstance(v, str)
            and all(isinstance(x, (float, int)) for x in v)):
            if len(v) != alphabet_size:
                raise ValueError(
                    f"insert_emissions must have length {alphabet_size} "
                    f"(alphabet size), got {len(v)}."
                )
            return v

        # Case 2: Sequence[Sequence[float]] - head-specific distributions
        if (isinstance(v, Sequence) and not isinstance(v, str)
            and all(isinstance(x, Sequence) and not isinstance(x, str)
                    for x in v)):

            if lengths is not None and len(v) != len(lengths):
                raise ValueError(
                    f"insert_emissions outer length must match number of heads "
                    f"({len(lengths)}), got {len(v)}."
                )

            for h, inner in enumerate(v):
                if not all(isinstance(x, (float, int)) for x in inner):
                    raise ValueError(
                        f"insert_emissions[{h}] must contain only floats/ints."
                    )
                if len(inner) != alphabet_size:
                    raise ValueError(
                        f"insert_emissions[{h}] must have length {alphabet_size} "
                        f"(alphabet size), got {len(inner)}."
                    )
            return v

        raise ValueError(
            "insert_emissions must be None, a sequence of floats, or a sequence "
            "of sequences of floats."
        )

    @model_validator(mode='after')
    def check_shared_flank_transitions(self):
        if self.shared_flank_transitions:
            LL = self.p_left_left
            RR = self.p_right_right
            if not np.all(np.isclose(LL, RR)):
                raise ValueError(
                    "When shared_flank_transitions is True, p_left_left and "
                    "p_right_right must be equal. "
                    "Small numerical differences are allowed. "
                    "Only p_left_left will be used. "
                    f"Got p_left_left={LL}, p_right_right={RR}."
                )
        return self
