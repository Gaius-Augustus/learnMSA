"""Guards the lazily exported package attributes.

Several packages resolve their public names through a module-level
``__getattr__`` so that importing them does not pull in a tensor framework.

The trap: a package attribute cannot reliably share a name with a submodule.
Importing ``learnMSA.align.align`` makes the import system bind that *module*
onto ``learnMSA.align``, after which ``__getattr__`` is never consulted and
``from learnMSA.align import align`` yields the module. The CLI then dies with
``TypeError: 'module' object is not callable``.

So colliding names are not re-exported at all; they are imported from their own
module. These tests pin both halves of that contract for the neutral packages;
``learnMSA.protein_language_models`` has the same shape but only imports under
TensorFlow, so it is checked in
``tests/protein_language_models/tf/test_lazy_exports.py``.
"""

import importlib

import pytest

from tests.backend import lazy_exports

#: (package, attribute) pairs that must resolve to something callable.
LAZY_CALLABLES = [
    ("learnMSA.align", "AlignmentModel"),
    ("learnMSA.model", "LearnMSAModel"),
]

#: (package, name) pairs that must NOT be re-exported, because a submodule of
#: the same name exists and would shadow them.
COLLIDING_NAMES = [
    ("learnMSA.align", "align"),
]


@pytest.mark.parametrize("package,attribute", LAZY_CALLABLES)
def test_lazy_export_resolves_to_callable(package: str, attribute: str) -> None:
    module = importlib.import_module(package)
    resolved = getattr(module, attribute)
    assert callable(resolved), (
        f"{package}.{attribute} resolved to {type(resolved).__name__}, "
        "expected a callable."
    )
    # Resolving twice must be stable.
    assert getattr(module, attribute) is resolved


@pytest.mark.parametrize("package,name", COLLIDING_NAMES)
def test_colliding_name_is_importable_from_its_own_module(
    package: str, name: str
) -> None:
    """The submodule always provides the callable, whatever import order ran."""
    lazy_exports.check_importable_from_own_module(package, name)


@pytest.mark.parametrize("package,name", COLLIDING_NAMES)
def test_colliding_name_is_not_reexported(package: str, name: str) -> None:
    """Re-exporting it would be a latent 'module is not callable' bug."""
    lazy_exports.check_not_reexported(package, name)


def test_unknown_attribute_still_raises() -> None:
    import learnMSA.align

    with pytest.raises(AttributeError):
        learnMSA.align.does_not_exist
