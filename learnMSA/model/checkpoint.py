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
    from learnMSA.model.model import LearnMSAModel

#: Identifier stored in an alignment's ``meta.json`` for each backend's format.
FORMATS = {"tensorflow": "keras", "pytorch": "pt"}


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


def load_model(filepath: str | Path) -> "LearnMSAModel":
    """Read a model from ``filepath`` using the selected backend."""
    from learnMSA.backend import resolve
    return resolve("model.checkpoint", "load_model")(filepath)
