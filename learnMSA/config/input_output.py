from pathlib import Path
from typing import Annotated, Union
from pydantic import BaseModel, field_validator, field_serializer, BeforeValidator


def path_validator(v: Union[Path, str]) -> Path:
    """Convert string to Path."""
    if isinstance(v, str):
        return Path(v)
    return v


PathField = Annotated[Union[Path, str], BeforeValidator(path_validator)]


class InputOutputConfig(BaseModel):
    """Input/output and general control parameters."""

    input_file: PathField = Path()
    """Input fasta file containing the protein sequences to align."""

    output_file: PathField = Path()
    """Output file path for the resulting multiple sequence alignment."""

    format: str = "a2m"
    """Format of the output alignment file."""

    input_format: str = "fasta"
    """Format of the input alignment file."""

    save_model: str = ""
    """If set, the trained model parameters will be saved to the specified file."""

    load_model: str = ""
    """If set, learnMSA will load the model parameters from the specified file."""

    silent: bool = False
    """Suppresses all standard output messages."""

    cuda_visible_devices: str = "default"
    """GPU device(s) visible to learnMSA. Use -1 for CPU."""

    work_dir: str = "tmp"
    """Directory where any secondary files are stored."""

    convert: bool = False
    """If True, only convert the input MSA to the format specified with format."""

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        """Validate output format."""
        # Common formats - this can be extended
        valid_formats = {"a2m", "fasta", "stockholm", "clustal", "phylip"}
        if v and v not in valid_formats:
            raise ValueError(
                f"format must be one of {valid_formats} or a valid Biopython "\
                "SeqIO format"
            )
        return v

    @field_validator("input_format")
    @classmethod
    def validate_input_format(cls, v: str) -> str:
        """Validate input format."""
        # Common formats
        valid_formats = {"fasta", "a2m", "stockholm", "clustal"}
        if v and v not in valid_formats:
            raise ValueError(
                f"input_format must be one of {valid_formats} or a valid "\
                "Biopython SeqIO format"
            )
        return v

    @field_validator("cuda_visible_devices")
    @classmethod
    def validate_cuda_devices(cls, v: str) -> str:
        """Validate CUDA visible devices."""
        if v not in ["default", "-1"] and v:
            # Should be comma-separated integers
            try:
                devices = [int(d.strip()) for d in v.split(",")]
                if any(d < 0 for d in devices):
                    raise ValueError("Device IDs must be non-negative integers")
            except ValueError as e:
                if "invalid literal" in str(e):
                    raise ValueError(
                        "cuda_visible_devices must be 'default', '-1', or "\
                        "comma-separated device IDs"
                    )
                raise
        return v

