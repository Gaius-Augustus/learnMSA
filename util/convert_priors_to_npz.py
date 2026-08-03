"""Convert the shipped prior parameters from Keras ``.weights.h5`` to ``.npz``.

The ``.h5`` files can only be read by building a throwaway ``tf.keras.Model``,
which ties prior loading to TensorFlow. The payload is a single flat kernel
array, so it is stored as a plain ``.npz`` instead and any backend can read it.

Usage::

    python util/convert_priors_to_npz.py            # convert + verify
    python util/convert_priors_to_npz.py --check    # verify only, write nothing

Verification reads the array back and compares it bit-for-bit with the array in
the ``.h5`` file. The legacy files are left in place; see
:func:`learnMSA.hmm.priors.load_prior_kernel` for the fallback that keeps them
working.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from learnMSA.hmm.priors import (KERNEL_KEY, read_h5_kernel,  # noqa: E402
                                 save_prior_kernel)

WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "learnMSA" / "hmm" / "weights"


def convert(check_only: bool = False) -> int:
    h5_files = sorted(WEIGHTS_DIR.glob("*.weights.h5"))
    if not h5_files:
        print(f"No .weights.h5 files found in {WEIGHTS_DIR}")
        return 1

    converted = 0
    failures: list[str] = []
    for h5_path in h5_files:
        # "foo.weights.h5" -> "foo.npz"
        npz_path = h5_path.with_suffix("").with_suffix(".npz")
        try:
            kernel = read_h5_kernel(h5_path)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{h5_path.name}: unreadable ({exc})")
            continue

        if not check_only:
            save_prior_kernel(kernel, npz_path)

        if not npz_path.exists():
            failures.append(f"{npz_path.name}: missing")
            continue

        with np.load(npz_path) as data:
            reloaded = data[KERNEL_KEY]

        if reloaded.shape != kernel.shape or reloaded.dtype != kernel.dtype:
            failures.append(
                f"{npz_path.name}: shape/dtype mismatch "
                f"({reloaded.shape}/{reloaded.dtype} vs "
                f"{kernel.shape}/{kernel.dtype})"
            )
        elif not np.array_equal(reloaded, kernel):
            failures.append(f"{npz_path.name}: values differ")
        else:
            converted += 1

    verb = "verified" if check_only else "converted"
    print(f"{verb} {converted}/{len(h5_files)} prior files")
    for failure in failures:
        print(f"  FAILED {failure}")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify existing .npz files without writing anything.",
    )
    args = parser.parse_args()
    return convert(check_only=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
