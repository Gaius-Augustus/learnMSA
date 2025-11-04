from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar
from pydantic import BaseModel, field_validator


class TrainingConfig(BaseModel):
    """Training parameters."""

    num_model: int = 4
    """Number of models to train."""

    batch_size: int = -1
    """Batch size for training. Default: adaptive."""

    learning_rate: float = 0.1
    """Learning rate for gradient descent."""

    epochs: Sequence[int] = [10, 2, 10]
    """Number of training epochs."""

    max_iterations: int = 2
    """Maximum number of training iterations. If greater than 2, model
    surgery will be applied."""

    length_init: Sequence[int] | None = None
    """Initial lengths for the models. Can be a single integer or a list of integers.
    If a list is provided, the number of models will be set to match the list length."""

    length_init_quantile: float = 0.5
    """Quantile for initial length determination."""

    surgery_quantile: float = 0.5
    """Quantile for model surgery."""

    min_surgery_seqs: int = 100000
    """Minimum number of sequences for model surgery."""

    len_mul: float = 0.8
    """Length multiplier."""

    surgery_del: float = 0.5
    """Discard match states expected less often than this fraction."""

    surgery_ins: float = 0.5
    """Expand insertions expected more often than this fraction."""

    model_criterion: str = "AIC"
    """Criterion for model selection."""

    indexed_data: bool = False
    """Stream training data at the cost of training time."""

    unaligned_insertions: bool = False
    """Insertions will be left unaligned."""

    crop: int | str = "auto"
    """Crop sequences longer than the given value during training."""

    auto_crop_scale: float = 2.0
    """Automatically crop sequences longer than this factor times the
    average length during training."""

    frozen_insertions: bool = False
    """Insertions will be frozen during training."""

    no_sequence_weights: bool = False
    """Do not use sequence weights and strip mmseqs2 from requirements.
    In general not recommended."""

    skip_training: bool = False
    """Only decode an alignment from the provided model."""

    @field_validator("num_model")
    def validate_num_model(cls, v: int) -> int:
        if v < 1:
            raise ValueError("num_model must be greater than or equal to 1.")
        return v

    @field_validator("learning_rate")
    def validate_learning_rate(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("learning_rate must be positive.")
        return v

    @field_validator("epochs", mode="before")
    def validate_epochs(cls, v: int | Sequence[int]) -> Sequence[int]:
        # If it's a single integer, expand it to a 3-element list
        if isinstance(v, int):
            return [v, v, v]

        # If it's a sequence, validate it has exactly 3 elements
        if isinstance(v, Sequence) and not isinstance(v, str):
            v_list = list(v)
            if len(v_list) != 3:
                raise ValueError("epochs must have exactly 3 elements.")
            if not all(isinstance(x, int) for x in v_list):
                raise ValueError("All elements of epochs must be integers.")
            return v_list

        raise ValueError("epochs must be an integer or a sequence of 3 integers.")

    @field_validator("max_iterations")
    def validate_max_iterations(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_iterations must be at least 1.")
        return v

    @field_validator("length_init")
    def validate_length_init(cls, v: Sequence[int] | None) -> Sequence[int] | None:
        if v is None:
            return v
        if not all(x >= 3 for x in v):
            raise ValueError("All elements of length_init must be at least 3.")
        return v

    @field_validator(
        "length_init_quantile",
        "surgery_quantile",
        "surgery_del",
        "surgery_ins",
    )
    def validate_quantiles(cls, v: float, info) -> float:
        if not 0 <= v <= 1:
            raise ValueError(f"{info.field_name} must be in the range [0, 1].")
        return v

    @field_validator(
        "len_mul",
        "auto_crop_scale",
        "min_surgery_seqs",
    )
    def validate_positive_floats(cls, v: float | int, info) -> float | int:
        if v <= 0:
            raise ValueError(f"{info.field_name} must be greater than 0.")
        return v

    @field_validator("crop")
    def validate_crop(cls, v: int | str) -> int | str:
        if isinstance(v, int):
            if v < 1:
                raise ValueError(
                    "crop must be an integer > 0 when provided as a number."
                )
            return v
        elif v in {"disable", "auto"}:
            return v
        raise ValueError(
            "crop must be \"disable\", \"auto\", or an integer > 0."
        )


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


class LanguageModelConfig(BaseModel):
    """Protein language model integration parameters."""
    
    use_language_model: bool = False
    """Uses a large protein lanague model to generate per-token
    embeddings that guide the MSA step."""

    plm_cache_dir: str | None = None
    """Directory where the protein language model is stored."""

    language_model: str = "protT5"
    """Name of the language model to use."""

    scoring_model_dim: int = 16
    """Reduced embedding dimension of the scoring model."""

    scoring_model_activation: str = "sigmoid"
    """Activation function of the scoring model."""

    scoring_model_suffix: str = ""
    """Suffix to identify a specific scoring model."""

    temperature: float = 3.0
    """Temperature of the softmax function."""

    temperature_mode: str = "trainable"
    """Temperature mode."""

    use_L2: bool = False
    """Use L2 regularization."""

    L2_match: float = 0.0
    """L2 regularization for match states."""

    L2_insert: float = 1000.0
    """L2 regularization for insert states."""

    embedding_prior_components: int = 32
    """Number of embedding prior components."""


class VisualizationConfig(BaseModel):
    """Visualization parameters."""
    
    logo: str = ""
    """Produces a pdf of the learned sequence logo."""

    logo_gif: str = ""
    """Produces a gif that animates the learned sequence logo over
    training time. Slows down training significantly."""


class AdvancedConfig(BaseModel):
    """Advanced/Development parameters."""
    
    dist_out: str = ""
    """Distribution output file."""

    alpha_flank: float = 7000
    """Alpha parameter for flank."""

    alpha_single: float = 1e9
    """Alpha parameter for single."""

    alpha_global: float = 1e4
    """Alpha parameter for global."""

    alpha_flank_compl: float = 1
    """Alpha parameter for flank complement."""

    alpha_single_compl: float = 1
    """Alpha parameter for single complement."""

    alpha_global_compl: float = 1
    """Alpha parameter for global complement."""

    inverse_gamma_alpha: float = 3.0
    """Inverse gamma alpha parameter."""

    inverse_gamma_beta: float = 0.5
    """Inverse gamma beta parameter."""

    frozen_distances: bool = False
    """Freeze distances during training."""

    initial_distance: float = 0.05
    """Initial distance value."""

    trainable_rate_matrices: bool = False
    """Make rate matrices trainable."""


class Configuration(BaseModel):
    """Main configuration combining all parameter groups."""

    # Nested configuration groups
    training: TrainingConfig = TrainingConfig()
    """Training parameters."""

    init_msa: InitMSAConfig = InitMSAConfig()
    """Initialize with existing MSA parameters."""

    language_model: LanguageModelConfig = LanguageModelConfig()
    """Protein language model integration parameters."""

    visualization: VisualizationConfig = VisualizationConfig()
    """Visualization parameters."""

    advanced: AdvancedConfig = AdvancedConfig()
    """Advanced/Development parameters."""

    # ==================== HMM ====================

    lengths: Sequence[int]
    """The number of match states in each head of the pHMM.
    """

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