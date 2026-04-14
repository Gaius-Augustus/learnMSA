from collections.abc import Sequence
from typing import Any
from pydantic import BaseModel

class TrainingConfig(BaseModel):
    """Type stub for TrainingConfig to include num_model parameter."""

    _num_model: int
    batch_size: int
    tokens_per_batch: int
    learning_rate: float
    epochs: Sequence[int]
    max_iterations: int
    length_init: Sequence[int] | None
    length_init_quantile: float
    surgery_quantile: float
    min_surgery_seqs: int
    len_mul: float
    surgery_del: float
    surgery_ins: float
    model_criterion: str
    indexed_data: bool
    unaligned_insertions: bool
    crop: int | str
    auto_crop: bool
    auto_crop_scale: float
    trainable_insertions: bool
    no_sequence_weights: bool
    skip_training: bool
    cluster_seq_id: float
    use_prior: bool
    dirichlet_mix_comp_count: int
    use_anc_probs: bool
    trainable_distances: bool
    only_matches: bool
    surgery_checkpoints: bool
    pre_training_checkpoint: bool
    use_noise: bool
    no_aa: bool
    reset_emissions_after_surgery: bool
    reset_transitions_after_surgery: bool

    def __init__(
        self,
        *,
        num_model: int = 4,
        batch_size: int = -1,
        tokens_per_batch: int = -1,
        learning_rate: float = 0.1,
        epochs: int | Sequence[int] = ...,
        max_iterations: int = 2,
        length_init: Sequence[int] | None = None,
        length_init_quantile: float = 0.5,
        surgery_quantile: float = 0.5,
        min_surgery_seqs: int = 100000,
        len_mul: float = 0.8,
        surgery_del: float = 0.5,
        surgery_ins: float = 0.5,
        model_criterion: str = "AIC",
        indexed_data: bool = False,
        unaligned_insertions: bool = False,
        crop: int | str = "auto",
        auto_crop: bool = True,
        auto_crop_scale: float = 2.0,
        trainable_insertions: bool = False,
        no_sequence_weights: bool = False,
        skip_training: bool = False,
        cluster_seq_id: float = 0.9,
        use_prior: bool = True,
        dirichlet_mix_comp_count: int = 1,
        use_anc_probs: bool = False,
        trainable_distances: bool = True,
        only_matches: bool = False,
        surgery_checkpoints: bool = True,
        pre_training_checkpoint: bool = False,
        use_noise: bool = True,
        no_aa: bool = False,
        reset_emissions_after_surgery: bool = False,
        reset_transitions_after_surgery: bool = False,
        **kwargs: Any,
    ) -> None: ...

    @property
    def num_model(self) -> int: ...

    @num_model.setter
    def num_model(self, value: int) -> None: ...

    def model_dump(self, **kwargs: Any) -> dict[str, Any]: ...
