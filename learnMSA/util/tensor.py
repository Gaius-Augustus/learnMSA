"""Backend-neutral tensor helpers.

The boundary between learnMSA's framework-specific layers and its numpy logic.
Anything that reads a value out of a layer goes through :func:`to_numpy`, so the
consuming code never has to know which framework produced it.

This mirrors ``hidten.visualize._to_numpy``, which cannot be imported here
because ``hidten.visualize`` pulls in the plotting dependencies.
"""

import numpy as np


def to_numpy(x) -> np.ndarray:
    """Convert a backend tensor to a numpy array.

    TensorFlow tensors convert directly. PyTorch tensors raise instead of
    converting when they require gradients or live on a device, in which case
    they are detached and moved to the host first. Plain arrays and sequences
    are passed through :func:`numpy.asarray`.
    """
    numpy_fn = getattr(x, "numpy", None)
    if not callable(numpy_fn):
        return np.asarray(x)
    try:
        return np.asarray(numpy_fn())
    except (RuntimeError, TypeError):
        # torch tensors that require grad or that do not live on the host
        for method in ("detach", "cpu"):
            fn = getattr(x, method, None)
            if callable(fn):
                x = fn()
        return np.asarray(x.numpy())
