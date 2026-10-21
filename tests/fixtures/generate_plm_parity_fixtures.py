"""Records reference embedding-pipeline outputs from the TensorFlow backend.

No conda environment for this project has both TensorFlow and PyTorch
installed, so the backends cannot be compared inside one process. Instead this
script -- run under the TensorFlow environment -- computes the two pieces of
the embedding pipeline that had to be written twice, and commits the result.
``tests/protein_language_models/torch/test_parity.py`` then recomputes them
under PyTorch and checks that it arrives at the same numbers.

The language models themselves are deliberately not covered: they are
multi-gigabyte downloads, and their weights are identical across the two
frameworks by construction. What is covered is the code learnMSA itself had to
transliterate -- ``eliminate_start_stop_tokens`` and the bilinear reduction.

Usage::

    conda activate learnMSAdev2
    python tests/fixtures/generate_plm_parity_fixtures.py
"""

from pathlib import Path

import numpy as np

FIXTURE = Path(__file__).resolve().parent / "plm_parity.npz"

#: Shape of the reference embedding batch.
BATCH, TIMESTEPS = 5, 11

#: Width of the reference embeddings. Matches ProtT5, so that the recorded
#: reduction uses a real shipped scoring model.
EMBEDDING_DIM = 1024

#: Unpadded length of each reference sequence.
SEQUENCE_LENGTHS = [11, 9, 8, 6, 4]

#: Crop flags per sequence. All four combinations of (cropped at the start,
#: cropped at the end) appear, because each takes its own branch through
#: ``eliminate_start_stop_tokens``.
CROP = [[0, 0], [1, 0], [0, 1], [1, 1], [0, 0]]

#: The scoring model the reduction is recorded for.
SCORING_MODEL = ("protT5", 16, "sigmoid")

SEED = 20260827


def make_inputs() -> dict[str, np.ndarray]:
    """Build the reference embeddings, crop flags and padding mask."""
    rng = np.random.default_rng(SEED)
    emb = rng.normal(size=(BATCH, TIMESTEPS, EMBEDDING_DIM)).astype(np.float32)
    crop = np.asarray(CROP, dtype=np.float32)
    lens = np.asarray(SEQUENCE_LENGTHS)
    mask = (np.arange(TIMESTEPS)[None] < lens[:, None]).astype(np.float32)
    return {"emb": emb, "crop": crop, "mask": mask}


def scoring_model_config():
    """The :class:`ScoringModelConfig` the reduction is recorded for."""
    from learnMSA.protein_language_models.common import ScoringModelConfig

    lm_name, dim, activation = SCORING_MODEL
    return ScoringModelConfig(
        lm_name=lm_name, dim=dim, activation=activation, scaled=False
    )


def main() -> None:
    import tensorflow as tf

    from learnMSA.protein_language_models.tf.bilinear_symmetric import \
        make_reduction_layer
    from learnMSA.protein_language_models.tf.language_model import \
        TFLanguageModel

    inputs = make_inputs()

    class _LanguageModel(TFLanguageModel):
        """Only the inherited token elimination is under test."""

        def call(self, inputs):
            raise NotImplementedError

    eliminated = _LanguageModel().eliminate_start_stop_tokens(
        tf.constant(inputs["emb"]),
        tf.constant(inputs["crop"]),
        tf.constant(inputs["mask"]),
    ).numpy()

    layer = make_reduction_layer(scoring_model_config())
    reduced = layer._reduce(tf.constant(inputs["emb"]), training=False).numpy()

    np.savez_compressed(
        FIXTURE, **inputs, eliminated=eliminated, reduced=reduced
    )
    print(
        f"Wrote {FIXTURE.name}: eliminated{eliminated.shape}, "
        f"reduced{reduced.shape}."
    )


if __name__ == "__main__":
    main()
