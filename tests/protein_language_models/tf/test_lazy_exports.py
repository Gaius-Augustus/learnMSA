"""The lazy-export contract for ``learnMSA.protein_language_models``.

Same shape as ``tests/backend/test_lazy_exports.py``, but this package's
submodules import TensorFlow -- the language models use ``TFEsmModel`` /
``TFT5EncoderModel`` and keras scoring weights -- so the checks can only run
under the TensorFlow backend. Under PyTorch, embeddings are supplied through
``--load_emb`` instead and this module is never imported.
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
