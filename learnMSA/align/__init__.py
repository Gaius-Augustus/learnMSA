"""Alignment construction and output.

``AlignmentModel`` is resolved lazily so that importing the neutral parts of this
package -- hit alignment, insertion alignment, alignment metadata -- does not
pull in a tensor framework.

The ``align`` function is deliberately *not* re-exported here: it shares its name
with the ``learnMSA.align.align`` submodule, and importing that submodule binds
the module onto this package, which would shadow the function. Import it from its
module instead::

    from learnMSA.align.align import align
"""

__all__ = ["AlignmentModel"]


def __getattr__(name: str):
    if name == "AlignmentModel":
        from .alignment_model import AlignmentModel
        globals()["AlignmentModel"] = AlignmentModel
        return AlignmentModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
