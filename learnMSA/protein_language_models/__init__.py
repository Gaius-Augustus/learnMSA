"""Protein language model embeddings.

This subpackage is TensorFlow-only: the language models are the TensorFlow
``transformers`` classes (``TFEsmModel``, ``TFT5EncoderModel``) and the scoring
models ship as keras artifacts. Other backends consume embeddings that were
precomputed with the TensorFlow backend and passed in with ``--emb-file``.

Nothing is imported here, so ``import learnMSA.protein_language_models`` does not
pull in TensorFlow. Submodules are imported on demand in the usual way::

    from learnMSA.protein_language_models.compute_embeddings import (
        compute_embeddings,
    )

Note that ``compute_embeddings`` names both a submodule and the function inside
it, so it must not be re-exported from this package: importing the submodule
binds the *module* onto the package and would shadow the function.
"""
