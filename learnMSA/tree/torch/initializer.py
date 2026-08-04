"""PyTorch initializers and the bridge from backend-neutral init specs.

The neutral :class:`~learnMSA.tree.initializer.InitSpec` family describes what
a parameter starts at; :func:`to_torch` turns such a description into the plain
callable the torch layers apply to a freshly allocated parameter.

Unlike Keras, torch has no initializer objects -- an initializer is just a
function that fills a tensor in place -- so there is no counterpart of
``ConstantInitializer`` here.
"""

from typing import Callable

import numpy as np
import torch

from learnMSA.tree import initializer as spec
# Re-exported for callers that build the numeric initial values directly.
from learnMSA.tree.initializer import make_substitution_model_init  # noqa: F401

#: A torch initializer fills a tensor in place and returns it.
T_TorchInitializer = Callable[[torch.Tensor], torch.Tensor]


def constant_initializer(
    value: np.ndarray | float,
) -> T_TorchInitializer:
    """An initializer that writes fixed values into a parameter.

    The value is *broadcast* to the parameter's shape, matching
    ``keras.initializers.Constant``, which multiplies its value by a ones
    tensor of the requested shape. learnMSA relies on this: a substitution
    model initializer of shape ``(H, 1, 1, D, D)`` has to fill a mixture kernel
    of shape ``(H, 1, K, D, D)``.
    """

    def initialize(parameter: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if np.isscalar(value):
                parameter.fill_(float(value))  # type: ignore[arg-type]
                return parameter
            array = torch.as_tensor(
                np.asarray(value), dtype=parameter.dtype
            )
            try:
                parameter.copy_(array.expand_as(parameter))
            except RuntimeError as error:
                raise ValueError(
                    f"constant initializer of shape {tuple(array.shape)} does "
                    f"not broadcast to a parameter of shape "
                    f"{tuple(parameter.shape)}."
                ) from error
        return parameter

    return initialize


def random_normal_initializer(
    stddev: float, mean: float = 0.0
) -> T_TorchInitializer:
    """An initializer that draws from a normal distribution."""

    def initialize(parameter: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            parameter.normal_(mean=mean, std=stddev)
        return parameter

    return initialize


def zeros_initializer() -> T_TorchInitializer:
    """An initializer that writes zeros."""

    def initialize(parameter: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            parameter.zero_()
        return parameter

    return initialize


def to_torch(
    initializer: "spec.InitSpec | T_TorchInitializer",
) -> T_TorchInitializer:
    """Materialize a neutral init spec as a torch initializer.

    Callables are passed through unchanged, so layers can accept either form.
    """
    if isinstance(initializer, spec.Constant):
        return constant_initializer(initializer.value)
    if isinstance(initializer, spec.RandomNormal):
        return random_normal_initializer(
            stddev=initializer.stddev, mean=initializer.mean
        )
    if isinstance(initializer, spec.Zeros):
        return zeros_initializer()
    if isinstance(initializer, spec.InitSpec):
        raise TypeError(
            "No PyTorch initializer is defined for "
            f"{type(initializer).__name__}."
        )
    if not callable(initializer):
        raise TypeError(
            f"Expected an InitSpec or a callable, got {type(initializer)}."
        )
    return initializer
