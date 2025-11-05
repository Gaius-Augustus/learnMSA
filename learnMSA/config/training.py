from collections.abc import Sequence
from pydantic import BaseModel, field_validator


class TrainingConfig(BaseModel):
    """Training parameters."""

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
