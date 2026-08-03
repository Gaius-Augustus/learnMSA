"""Model construction, context and selection.

``LearnMSAModel`` is resolved lazily against the selected backend so that
importing the neutral parts of this package -- the context, the batch generator,
the model selection -- does not pull in a tensor framework.
"""

from .context import LearnMSAContext
from .select import select_model

__all__ = ["LearnMSAContext", "LearnMSAModel", "select_model"]


def __getattr__(name: str):
    if name == "LearnMSAModel":
        from learnMSA.backend import resolve
        cls = resolve("model.model", "LearnMSAModel")
        globals()["LearnMSAModel"] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
