"""Backend-neutral tensor helpers.

The boundary between learnMSA's framework-specific layers and its numpy logic.
Anything that reads a value out of a layer goes through :func:`to_numpy`, so the
consuming code never has to know which framework produced it.

This mirrors ``hidten.visualize._to_numpy``, which cannot be imported here
because ``hidten.visualize`` pulls in the plotting dependencies.
"""

import numpy as np


def assign(variable, value) -> None:
    """Write ``value`` into a framework variable, in place.

    The frameworks spell this differently -- ``tf.Variable.assign`` versus
    writing through a torch parameter's ``.data`` -- and neither name exists on
    the other. Dispatching on the attribute rather than the type keeps this
    module free of framework imports, the same trick :func:`to_numpy` uses.

    Args:
        variable: A ``tf.Variable`` or a ``torch.nn.Parameter``/``Tensor``.
        value: The new value, of the same shape.
    """
    assign_fn = getattr(variable, "assign", None)
    if callable(assign_fn):
        assign_fn(value)
        return
    data = getattr(variable, "data", None)
    if data is not None and hasattr(data, "copy_"):
        # Going through .data writes without recording the assignment on the
        # autograd tape, which is what tf.Variable.assign does too.
        data.copy_(_as_same_kind(value, data))
        return
    raise TypeError(
        f"Cannot assign to a {type(variable).__name__}; expected a "
        "TensorFlow variable or a PyTorch parameter."
    )


def _as_same_kind(value, reference):
    """Coerce ``value`` into something ``reference.copy_`` accepts."""
    if hasattr(value, "data") and hasattr(value, "detach"):
        return value.detach()
    if hasattr(value, "copy_"):  # already a torch tensor
        return value
    return np.asarray(to_numpy(value))


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
