"""Protein language model embeddings.

The language model wrappers and the reduction layer exist once per backend,
in the ``tf`` and ``torch`` subpackages; the input encoders and the scoring
model parameters (``scoring_models/*.npz``) are shared by both.

Nothing is imported here, so ``import learnMSA.protein_language_models`` pulls
in neither framework. Note that ``compute_embeddings`` names both a submodule
and the function inside it, so it must not be re-exported from this package:
importing the submodule binds the *module* onto the package and would shadow
the function.
"""
