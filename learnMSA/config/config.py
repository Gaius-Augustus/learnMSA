from typing import Any

from pydantic import BaseModel, ConfigDict

from .advanced import AdvancedConfig
from .init_msa import InitMSAConfig
from .input_output import InputOutputConfig
from .language_model import LanguageModelConfig
from .training import TrainingConfig
from .visualization import VisualizationConfig


class Configuration(BaseModel):
    """A configuration for learnMSA controlling all aspects of training and
    evaluation.
    """

    model_config = ConfigDict(extra="allow")

    # Nested configuration groups
    input_output: InputOutputConfig = InputOutputConfig()
    """Input/output and general control parameters."""

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
