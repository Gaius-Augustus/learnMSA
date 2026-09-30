"""Backend-neutral entry points for persisting a learnMSA model.

Each backend stores its parameters in its own format -- TensorFlow writes a
``.keras`` archive -- so the actual reading and writing lives in
``learnMSA/model/<backend>/checkpoint.py``. Callers go through the two functions
here and never name a format.

The format is recorded alongside the model (see ``AlignmentModel.save``) so that
a file written by one backend can be recognised, and refused with a clear
message, by another.
"""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from learnMSA.config import Configuration
    from learnMSA.model.model import LearnMSAModel

#: Identifier stored in an alignment's ``meta.json`` for each backend's format.
FORMATS = {"tensorflow": "keras", "pytorch": "pt"}

#: Advanced settings that describe how a run executes rather than what the
#: model is. A checkpoint carries the whole configuration it was trained with,
#: but these are taken from the current run instead, so that a model trained
#: with, say, ``--triton`` is not stuck with the kernels when it is loaded.
RUNTIME_ADVANCED_FIELDS = ("compile", "use_triton", "seed")

#: Structure settings that only change how an alignment is decoded. Training,
#: surgery and model selection score the structural track at temperature 1,
#: so a loaded model is decoded with the current run's value.
RUNTIME_STRUCTURE_FIELDS = ("emitter_temperature",)


def checkpoint_format(backend_name: str | None = None) -> str:
    """The checkpoint format identifier of a backend."""
    if backend_name is None:
        from learnMSA.backend import get_backend
        backend_name = get_backend()
    return FORMATS[backend_name]


def save_model(model: "LearnMSAModel", filepath: str | Path) -> None:
    """Write a model to ``filepath`` in the selected backend's format."""
    from learnMSA.backend import resolve
    resolve("model.checkpoint", "save_model")(model, filepath)


def apply_runtime_config(
    restored: "Configuration",
    config: "Configuration | None",
) -> None:
    """Let the current run's execution and decoding settings win over a
    checkpoint's.

    Args:
        restored: The configuration deserialized from the checkpoint. Modified
            in place.
        config: The configuration of the current run, or ``None`` to keep the
            checkpoint's settings.
    """
    if config is None:
        return
    for field in RUNTIME_ADVANCED_FIELDS:
        setattr(restored.advanced, field, getattr(config.advanced, field))
    for field in RUNTIME_STRUCTURE_FIELDS:
        setattr(restored.structure, field, getattr(config.structure, field))


def load_model(
    filepath: str | Path,
    config: "Configuration | None" = None,
) -> "LearnMSAModel":
    """Read a model from ``filepath`` using the selected backend.

    Args:
        filepath: Path of the checkpoint, without the backend's suffix.
        config: Configuration of the current run. Its
            :data:`RUNTIME_ADVANCED_FIELDS` and
            :data:`RUNTIME_STRUCTURE_FIELDS` replace the ones stored in the
            checkpoint; the rest of the checkpoint's configuration is kept.
    """
    from learnMSA.backend import resolve
    return resolve("model.checkpoint", "load_model")(filepath, config)
