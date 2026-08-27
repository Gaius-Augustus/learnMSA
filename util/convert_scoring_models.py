"""Convert the shipped scoring model weights from Keras ``.h5`` to ``.npz``.

The bilinear scoring models used to be stored as Keras 2 whole-model weight
files, which only TensorFlow can read. A scoring model is really just a
projection ``R`` and a scalar bias ``b``, so ``.npz`` is enough and lets the
PyTorch backend load them too.

This script needs neither TensorFlow nor PyTorch -- only h5py and numpy. Run it
after refitting a scoring model with the legacy pretraining tooling, which
still writes ``.h5``::

    python util/convert_scoring_models.py

The ``.h5`` files are kept in the repository as a fallback; see
:func:`learnMSA.protein_language_models.scoring_weights.load_scoring_weights`.
"""

import argparse
from pathlib import Path

import numpy as np

from learnMSA.protein_language_models.scoring_weights import (
    KERNEL_KEYS, read_h5_scoring_weights, save_scoring_weights)

#: Directory holding the shipped scoring model weights.
WEIGHTS_DIR = (
    Path(__file__).resolve().parents[1]
    / "learnMSA" / "protein_language_models" / "scoring_models"
)


def convert(h5_path: Path, force: bool = False) -> bool:
    """Convert one ``.h5`` scoring model to an ``.npz`` sibling.

    Args:
        h5_path: The legacy Keras file.
        force: Rewrite the ``.npz`` even if it is already there.

    Returns:
        Whether a file was written.
    """
    npz_path = h5_path.with_suffix(".npz")
    if npz_path.exists() and not force:
        print(f"skip   {npz_path.name} (exists)")
        return False

    weights = read_h5_scoring_weights(h5_path)
    save_scoring_weights(weights, npz_path)

    # Read it straight back: a silently truncated conversion would be very hard
    # to notice downstream, where it only shows up as bad alignments.
    with np.load(npz_path) as data:
        for key in KERNEL_KEYS:
            np.testing.assert_array_equal(data[key], weights[key])

    shapes = ", ".join(f"{k}{weights[k].shape}" for k in KERNEL_KEYS)
    print(f"wrote  {npz_path.name}  ({shapes})")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--force", action="store_true", help="overwrite existing .npz files"
    )
    parser.add_argument(
        "--dir", type=Path, default=WEIGHTS_DIR,
        help=f"directory to convert (default: {WEIGHTS_DIR})",
    )
    args = parser.parse_args()

    h5_files = sorted(args.dir.glob("*.h5"))
    if not h5_files:
        raise SystemExit(f"No .h5 scoring models found in {args.dir}.")

    written = sum(convert(path, force=args.force) for path in h5_files)
    print(f"\n{written}/{len(h5_files)} scoring models converted.")


if __name__ == "__main__":
    main()
