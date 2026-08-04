"""The colliding-name checks, shared by the packages that use lazy exports.

A package attribute cannot reliably share a name with a submodule of the same
name, so learnMSA never re-exports one. Both halves of that contract are
checked the same way wherever it applies; only the list of packages differs,
and ``learnMSA.protein_language_models`` can only be imported under
TensorFlow -- which is why the checks live here rather than in one test file.
"""

import importlib
import types


def check_importable_from_own_module(package: str, name: str) -> None:
    """The submodule always provides the callable, whatever import order ran."""
    submodule = importlib.import_module(f"{package}.{name}")
    resolved = getattr(submodule, name)
    assert callable(resolved), (
        f"{package}.{name}.{name} should be the callable to import."
    )


def check_not_reexported(package: str, name: str) -> None:
    """Re-exporting it would be a latent 'module is not callable' bug.

    Importing the submodule first is what makes the shadowing permanent, so do
    that here before checking.
    """
    importlib.import_module(f"{package}.{name}")
    module = importlib.import_module(package)
    exported = getattr(module, name, None)
    assert exported is None or isinstance(exported, types.ModuleType), (
        f"{package} re-exports {name!r}, which collides with its submodule. "
        "Import it from its own module instead."
    )
    assert name not in getattr(module, "__all__", ())
