from typing import Any

from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator

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

    # Private attribute for num_model storage
    _num_model: int = PrivateAttr(default=4)

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

    @property
    def num_model(self) -> int:
        """Number of models to train.

        If training.length_init is provided, returns its length.
        Otherwise, returns the stored num_model value.
        """
        if self.training.length_init is not None:
            return len(self.training.length_init)
        return self._num_model

    @num_model.setter
    def num_model(self, value: int) -> None:
        """Set the number of models.

        This sets the internal _num_model field, but note that if
        training.length_init is set, this value will be overridden.
        """
        if value < 1:
            raise ValueError("num_model must be greater than or equal to 1.")
        self._num_model = value

    @model_validator(mode="before")
    @classmethod
    def extract_num_model(cls, data: Any) -> Any:
        """Extract num_model from input data and store for later initialization."""
        if isinstance(data, dict) and "num_model" in data:
            # Store it with a special key for model_post_init
            data["__num_model_value"] = data.pop("num_model")
        return data

    def model_post_init(self, __context: Any) -> None:
        """Store the num_model value after initialization.

        Args:
            __context: Context passed during validation (typically None).
        """
        # Get the value from the extra field if it was provided
        if hasattr(self, "__num_model_value"):
            num_model_value = getattr(self, "__num_model_value")
            delattr(self, "__num_model_value")
            self._num_model = num_model_value

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Serialize the configuration to a dictionary.

        Overrides the default model_dump to include the num_model property.

        Returns:
            Dictionary representation of the configuration.
        """
        data = super().model_dump(**kwargs)
        # Add num_model to the serialized data
        data["num_model"] = self.num_model
        return data
