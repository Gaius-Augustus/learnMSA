"""Type stubs for Configuration class to support num_model in constructor."""

from typing import Any

from pydantic import BaseModel

from .advanced import AdvancedConfig
from .init_msa import InitMSAConfig
from .language_model import LanguageModelConfig
from .training import TrainingConfig
from .visualization import VisualizationConfig

class Configuration(BaseModel):
    """Type stub for Configuration to include num_model parameter."""

    _num_model: int
    training: TrainingConfig
    init_msa: InitMSAConfig
    language_model: LanguageModelConfig
    visualization: VisualizationConfig
    advanced: AdvancedConfig

    def __init__(
        self,
        *,
        num_model: int = 4,
        training: TrainingConfig = ...,
        init_msa: InitMSAConfig = ...,
        language_model: LanguageModelConfig = ...,
        visualization: VisualizationConfig = ...,
        advanced: AdvancedConfig = ...,
        **kwargs: Any,
    ) -> None: ...

    @property
    def num_model(self) -> int: ...

    @num_model.setter
    def num_model(self, value: int) -> None: ...
