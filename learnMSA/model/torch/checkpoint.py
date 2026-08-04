"""Reading and writing learnMSA models as torch checkpoints.

The TensorFlow backend can lean on the ``.keras`` archive to store both the
model structure and its weights. Torch's ``state_dict`` only carries tensors,
so a checkpoint here is the state dict plus the serialized
:class:`~learnMSA.model.context.LearnMSAContext` that says how to rebuild the
module the tensors belong to.
"""

from pathlib import Path

import torch

from learnMSA.model.context import LearnMSAContext
from learnMSA.model.torch.model import TorchLearnMSAModel

#: Suffix of a saved PyTorch model.
SUFFIX = ".pt"

#: Bumped whenever the layout of the checkpoint dict changes.
FORMAT_VERSION = 1


def save_model(model: TorchLearnMSAModel, filepath: str | Path) -> None:
    """Write the model's parameters and its context to a single file."""
    torch.save(
        {
            "format_version": FORMAT_VERSION,
            "context": model.context.get_config(),
            "state_dict": model.state_dict(),
        },
        str(filepath) + SUFFIX,
    )


def load_model(filepath: str | Path) -> TorchLearnMSAModel:
    """Rebuild a model from a checkpoint and load its parameters."""
    # The checkpoint holds learnMSA's own config objects rather than plain
    # tensors, so it has to be unpickled fully; these files are written by
    # learnMSA itself.
    checkpoint = torch.load(
        str(filepath) + SUFFIX, map_location="cpu", weights_only=False
    )
    version = checkpoint.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"{filepath}{SUFFIX} was written in checkpoint format "
            f"{version!r}, but this version of learnMSA reads format "
            f"{FORMAT_VERSION}."
        )

    context = LearnMSAContext.from_config(checkpoint["context"])
    model = TorchLearnMSAModel(context)
    # The parameters do not exist until the layers are built, so build before
    # loading rather than after.
    model.build()
    model.load_state_dict(checkpoint["state_dict"])
    model.compile()
    return model
