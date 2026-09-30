"""Convert the ProstT5 3Di prediction head to the npz shipped with learnMSA.

Downloads the CNN checkpoint that ``scripts/predict_3Di_encoderOnly.py`` of
https://github.com/mheinzinger/ProstT5 (MIT license, Heinzinger et al.) uses,
at a pinned commit, verifies its sha256 and stores only the four classifier
tensors (float32). The optimizer state in the checkpoint is dropped.

Usage:
    python util/convert_prostt5_cnn.py \\
        [--out learnMSA/structure/weights/prostt5_3di_cnn.npz]
"""

import argparse
import hashlib
import io
import urllib.request
from pathlib import Path

import numpy as np
import torch

COMMIT = "839d839aea349b0e471eb5e6c9f3458ed864019e"
URL = (f"https://github.com/mheinzinger/ProstT5/raw/{COMMIT}/"
       "cnn_chkpnt/model.pt")
SHA256 = "d2cb4150884de095dac3de81b8fe7842ab1265de41dd5e8c9970879a71e62f3e"
KEYS = ("classifier.0.weight", "classifier.0.bias",
        "classifier.3.weight", "classifier.3.bias")
SHAPES = ((32, 1024, 7, 1), (32,), (20, 32, 7, 1), (20,))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--out", type=Path,
        default=Path("learnMSA/structure/weights/prostt5_3di_cnn.npz"),
    )
    args = parser.parse_args()

    with urllib.request.urlopen(URL) as response:
        raw = response.read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"sha256 mismatch: expected {SHA256}, got {digest}")
    checkpoint = torch.load(io.BytesIO(raw), map_location="cpu",
                            weights_only=True)
    state = checkpoint["state_dict"]
    arrays = {}
    for key, shape in zip(KEYS, SHAPES):
        value = state[key].to(torch.float32).numpy()
        if value.shape != shape:
            raise SystemExit(f"{key}: shape {value.shape}, expected {shape}")
        arrays[key] = value
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, source=np.array(URL), sha256=np.array(SHA256),
             **arrays)
    print(f"saved {args.out} "
          f"({sum(a.size for a in arrays.values())} parameters)")


if __name__ == "__main__":
    main()
