from collections.abc import Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

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

    emitter_temperature: float = 8.0
    """Temperature applied as an exponent (1/temperature) to the structural
    emission scores."""

    reset_after_surgery: bool = False
    """Whether to reset the structural information emission parameters after
    model surgery. default: False."""

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

    observation_noise: Literal["none", "background", "confusion"] = "none"
    """Observation model for misclassified structural tokens, e.g. by a
    predictor such as ProstT5. The observed token o becomes the likelihood
    vector ``v_y = M[y, o] = P(o | true token y)`` before the evolutionary
    model and the emitters see it. (EXPERIMENTAL)

    - ``"none"``: tokens are taken as error-free (``M = I``).
    - ``"background"``: ``M = (1 - s) I + s 1 pi^T`` with ``pi`` the
      ``background_distribution``.
    - ``"confusion"``: ``M = (1 - s) I + s C`` with ``C`` loaded from
      ``observation_confusion_name``.

    ``s`` is ``observation_noise_strength``."""

    observation_noise_strength: float = 0.1
    """Mixing weight ``s`` in [0, 1] of the observation noise model."""

    observation_confusion_name: str = "prostt5_3Di_confusion_homstrad"
    """Name of the shipped ``.npz`` with the row-stochastic confusion matrix
    ``confusion[y, o] = P(observed o | true y)``."""

    trainable_observation_noise: bool = False
    """Whether ``observation_noise_strength`` is only the initial value of a
    trainable mixing weight (one scalar shared by all heads)."""

    input_format: Literal["tokens", "logits"] = "tokens"
    """Set when the structural data are loaded: ``"tokens"`` for a 3Di FASTA
    file, ``"logits"`` for per-residue 3Di logits (an ``.npz`` written by
    ``learnMSA-3di``)."""

    soft_input: Literal["argmax", "posterior", "likelihood"] = "likelihood"
    """How per-residue 3Di logits z become the observation vector v over
    true 3Di letters (only with ``input_format == "logits"``).

    - ``"argmax"``: one-hot of the most likely letter (like a 3Di FASTA).
    - ``"posterior"``: ``v = softmax(z / T)``.
    - ``"likelihood"``: ``v ~ (softmax(z / T) / pi) ** k``, the predictor's
      posterior divided by the letter prior ``pi``
      (``background_distribution``), i.e. a scaled likelihood.

    ``T`` is ``soft_input_temperature`` and ``k`` is
    ``soft_input_sharpness``. (EXPERIMENTAL)"""

    soft_input_temperature: float = 1.25
    """Calibration temperature ``T`` applied to the 3Di logits. The default
    minimises the NLL of true 3Di letters (Homstrad PDB chains, 155k
    residues) under ProstT5 via ``util/calibrate_prostt5.py``: 1.25, with
    1.22 and 1.28 on either half of the families."""

    soft_input_sharpness: float = 1.0
    """Exponent ``k`` of the scaled likelihood; below 1 flattens it."""

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

    @field_validator("soft_input_temperature", "soft_input_sharpness")
    def validate_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("must be positive.")
        return v

    @field_validator("observation_noise_strength")
    def validate_observation_noise_strength(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(
                "observation_noise_strength must be in [0, 1]."
            )
        return v

    @property
    def alphabet_size(self) -> int:
        """The size of the alphabet."""
        return len(self.structural_alphabet)
