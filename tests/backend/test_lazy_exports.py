"""Guards the lazily exported package attributes.

Several packages resolve their public names through a module-level
``__getattr__`` so that importing them does not pull in a tensor framework.

The trap: a package attribute cannot reliably share a name with a submodule.
Importing ``learnMSA.align.align`` makes the import system bind that *module*
onto ``learnMSA.align``, after which ``__getattr__`` is never consulted and
``from learnMSA.align import align`` yields the module. The CLI then dies with
``TypeError: 'module' object is not callable``.

So colliding names are not re-exported at all; they are imported from their own
module. These tests pin both halves of that contract.
"""

import importlib
import importlib.util
import types

import pytest

#: (package, attribute) pairs that must resolve to something callable.
LAZY_CALLABLES = [
    ("learnMSA.align", "AlignmentModel"),
    ("learnMSA.model", "LearnMSAModel"),
]

#: (package, name) pairs that must NOT be re-exported, because a submodule of
#: the same name exists and would shadow them.
COLLIDING_NAMES = [
    ("learnMSA.align", "align"),
    ("learnMSA.protein_language_models", "compute_embeddings"),
]

#: Packages whose submodules can only be imported under the TensorFlow
#: backend. The protein language models run on TensorFlow only; under the
#: PyTorch backend embeddings are supplied through ``--emb-file`` instead.
TENSORFLOW_ONLY_PACKAGES = {"learnMSA.protein_language_models"}

_HAS_TENSORFLOW = importlib.util.find_spec("tensorflow") is not None


def _skip_if_tensorflow_only(package: str) -> None:
    if package in TENSORFLOW_ONLY_PACKAGES and not _HAS_TENSORFLOW:
        pytest.skip(f"{package} requires TensorFlow")


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
    _skip_if_tensorflow_only(package)
    submodule = importlib.import_module(f"{package}.{name}")
    resolved = getattr(submodule, name)
    assert callable(resolved), (
        f"{package}.{name}.{name} should be the callable to import."
    )


@pytest.mark.parametrize("package,name", COLLIDING_NAMES)
def test_colliding_name_is_not_reexported(package: str, name: str) -> None:
    """Re-exporting it would be a latent 'module is not callable' bug.

    Importing the submodule first is what makes the shadowing permanent, so do
    that here before checking.
    """
    _skip_if_tensorflow_only(package)
    importlib.import_module(f"{package}.{name}")
    module = importlib.import_module(package)
    exported = getattr(module, name, None)
    assert exported is None or isinstance(exported, types.ModuleType), (
        f"{package} re-exports {name!r}, which collides with its submodule. "
        "Import it from its own module instead."
    )
    assert name not in getattr(module, "__all__", ())


@pytest.mark.tf
@pytest.mark.skipif(
    not _HAS_TENSORFLOW, reason="protein language models require TensorFlow"
)
def test_cli_imports_compute_embeddings_callably() -> None:
    """The exact import the CLI performs must yield the function."""
    from learnMSA.protein_language_models.compute_embeddings import \
        compute_embeddings

    assert callable(compute_embeddings)


def test_unknown_attribute_still_raises() -> None:
    import learnMSA.align

    with pytest.raises(AttributeError):
        learnMSA.align.does_not_exist
