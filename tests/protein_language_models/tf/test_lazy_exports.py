"""The lazy-export contract for ``learnMSA.protein_language_models``.

Same shape as ``tests/backend/test_lazy_exports.py``. The package itself is
backend-neutral, so the same checks run under PyTorch in
``tests/protein_language_models/torch/test_lazy_exports.py``.
"""

from tests.backend import lazy_exports

PACKAGE = "learnMSA.protein_language_models"
NAME = "compute_embeddings"


def test_colliding_name_is_importable_from_its_own_module() -> None:
    lazy_exports.check_importable_from_own_module(PACKAGE, NAME)


def test_colliding_name_is_not_reexported() -> None:
    lazy_exports.check_not_reexported(PACKAGE, NAME)


def test_cli_imports_compute_embeddings_callably() -> None:
    """The exact import the CLI performs must yield the function."""
    from learnMSA.protein_language_models.compute_embeddings import \
        compute_embeddings

    assert callable(compute_embeddings)
