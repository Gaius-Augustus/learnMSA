"""Compute backend selection for learnMSA.

learnMSA supports more than one tensor framework. The framework-specific code
lives in ``tf``/``torch`` subpackages of :mod:`learnMSA.hmm`,
:mod:`learnMSA.model`, :mod:`learnMSA.tree` and :mod:`learnMSA.align`, mirroring
the layout of the ``hidten`` dependency. This module decides which of those
subpackages is used and resolves symbols out of them lazily.

The hard invariant: importing this module never imports a tensor framework, and
selecting one backend never imports the other. Everything framework-specific is
reached through :func:`resolve` or one of the small dispatch helpers below,
which import inside the function body.
"""

import importlib
import importlib.util
import os
import sys

#: The backends learnMSA knows about.
SUPPORTED_BACKENDS: tuple[str, ...] = ("tensorflow", "pytorch")

#: Maps a backend name to the name of its on-disk subpackage.
_SUBPACKAGE: dict[str, str] = {"tensorflow": "tf", "pytorch": "torch"}

#: Maps a backend name to the top-level module of the *opposing* framework.
_OPPOSING_MODULE: dict[str, str] = {"tensorflow": "torch", "pytorch": "tensorflow"}

#: Maps a backend name to the top-level module of its own framework.
_OWN_MODULE: dict[str, str] = {"tensorflow": "tensorflow", "pytorch": "torch"}

#: Environment variable that overrides automatic backend detection.
BACKEND_ENV_VAR = "LEARNMSA_BACKEND"

_selected: str | None = None


def set_backend(name: str) -> None:
    """Select the compute backend.

    Must be called before any model, layer or context is constructed. Calling it
    again with the same backend is a no-op; switching backends within a process
    is not supported because the frameworks cannot be unloaded.

    Args:
        name: One of :data:`SUPPORTED_BACKENDS`.

    Raises:
        ValueError: If ``name`` is not a supported backend.
        RuntimeError: If a different backend was already selected, or if the
            opposing framework has already been imported.
    """
    global _selected
    _validate_backend(name)
    if _selected == name:
        return
    if _selected is not None:
        raise RuntimeError(
            f"The backend is already set to '{_selected}'. learnMSA cannot "
            f"switch to '{name}' within the same process."
        )
    opposing = _OPPOSING_MODULE[name]
    if opposing in sys.modules:
        raise RuntimeError(
            f"Cannot select the '{name}' backend because '{opposing}' has "
            "already been imported in this process. Select the backend before "
            "importing any framework-specific module."
        )
    _selected = name
    # evoten carries its own backend facade for the substitution model math and
    # happens to use the same backend names. Importing evoten is safe; it pulls
    # in neither framework.
    import evoten
    evoten.set_backend(name)


def get_backend() -> str:
    """The selected backend, auto-detecting one on first use.

    Returns:
        One of :data:`SUPPORTED_BACKENDS`.
    """
    if _selected is None:
        set_backend(_default_backend())
    assert _selected is not None
    return _selected


def is_backend_set() -> bool:
    """Whether a backend has been selected explicitly or by auto-detection."""
    return _selected is not None


def resolve(package: str, name: str):
    """Look up a symbol in the backend-specific variant of a module.

    ``resolve("hmm.layer", "PHMMLayer")`` returns
    ``learnMSA.hmm.tf.layer.TFPHMMLayer`` under the TensorFlow backend and
    ``learnMSA.hmm.torch.layer.TorchPHMMLayer`` under the PyTorch backend, i.e.
    the neutral class name is prefixed automatically.

    Args:
        package: Dotted path of the neutral module relative to ``learnMSA``,
            e.g. ``"hmm.layer"`` or ``"model.checkpoint"``.
        name: The neutral symbol name. It is looked up with the backend's class
            prefix first (``TF``/``Torch``) and without a prefix as a fallback,
            so plain functions such as ``make_dataset`` resolve too.
    """
    module = backend_module(package)
    prefixed = class_prefix() + name
    if hasattr(module, prefixed):
        return getattr(module, prefixed)
    if hasattr(module, name):
        return getattr(module, name)
    raise AttributeError(
        f"'{module.__name__}' defines neither '{prefixed}' nor '{name}'."
    )


def backend_module(package: str):
    """Import the backend-specific variant of a neutral module.

    ``backend_module("hmm.layer")`` imports ``learnMSA.hmm.tf.layer`` under the
    TensorFlow backend.
    """
    parent, _, module = package.rpartition(".")
    if not parent:
        raise ValueError(
            f"'{package}' must be a dotted path of at least two components, "
            "e.g. 'hmm.layer'."
        )
    return importlib.import_module(
        f"learnMSA.{parent}.{_SUBPACKAGE[get_backend()]}.{module}"
    )


def class_prefix() -> str:
    """The class-name prefix of the selected backend (``TF`` or ``Torch``)."""
    return {"tensorflow": "TF", "pytorch": "Torch"}[get_backend()]


def clear_session() -> None:
    """Release framework-held memory between training runs."""
    if get_backend() == "tensorflow":
        import tensorflow as tf
        tf.keras.backend.clear_session()
    else:
        import gc

        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def oom_errors() -> tuple[type[BaseException], ...]:
    """Exception types the backend raises when it runs out of device memory."""
    if get_backend() == "tensorflow":
        import tensorflow as tf
        return (tf.errors.ResourceExhaustedError,)
    import torch
    return (torch.cuda.OutOfMemoryError,)


def num_gpus() -> int:
    """The number of GPUs visible to the backend."""
    if get_backend() == "tensorflow":
        import tensorflow as tf
        return len([
            d for d in tf.config.list_logical_devices() if d.device_type == "GPU"
        ])
    import torch
    return torch.cuda.device_count()


def framework_version() -> str:
    """The version string of the selected framework."""
    if get_backend() == "tensorflow":
        import tensorflow as tf
        return tf.__version__
    import torch
    return torch.__version__


def _validate_backend(name: str) -> None:
    if name not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unknown backend '{name}'. Supported backends are "
            f"{list(SUPPORTED_BACKENDS)}."
        )


def _default_backend() -> str:
    """Pick a backend when the user did not select one.

    Precedence: the ``LEARNMSA_BACKEND`` environment variable, then a framework
    that is already imported, then an installed framework (TensorFlow first,
    since it is learnMSA's reference backend).
    """
    env = os.environ.get(BACKEND_ENV_VAR)
    if env:
        env = env.strip().lower()
        # Accept evoten's spelling of the torch extra as an alias.
        if env == "torch":
            env = "pytorch"
        _validate_backend(env)
        return env
    for backend in SUPPORTED_BACKENDS:
        if _OWN_MODULE[backend] in sys.modules:
            return backend
    for backend in SUPPORTED_BACKENDS:
        if importlib.util.find_spec(_OWN_MODULE[backend]) is not None:
            return backend
    raise ImportError(
        "learnMSA needs a tensor framework but neither tensorflow nor torch is "
        "installed. Install one with 'pip install learnMSA[tensorflow]' or "
        "'pip install learnMSA[pytorch]'."
    )


def _reset_for_testing() -> None:
    """Clear the selected backend. Only for use by the test suite."""
    global _selected
    _selected = None
