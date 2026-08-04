"""Test session setup shared by both compute backends.

learnMSA is developed against two conda environments that deliberately do not
overlap: one has TensorFlow, the other has PyTorch. A bare ``pytest`` must
therefore do the right thing in either of them, which means never *collecting*
a module that imports the absent framework -- an ImportError at collection time
aborts the whole run, so `skipif` markers inside those modules would be too
late.

The rule is the directory a test lives in, and nothing else: a test under a
``tf/`` package needs TensorFlow, a test under a ``torch/`` package needs
PyTorch, and everything else must run under whichever backend is present. Tests
of backend-neutral code therefore belong outside both, and reach the framework
through the neutral factories (``make_learnmsa_model``, ``make_phmm_layer``)
rather than importing a backend class directly.
"""

import functools
import importlib.util
import os
import subprocess
import sys

#: Test paths that require the TensorFlow backend. Globs are relative to this
#: directory and mirror the ``learnMSA.*.tf`` subpackages they exercise.
TF_ONLY = [
    "*/tf/*.py",
]

#: Test paths that require the PyTorch backend.
TORCH_ONLY = [
    "*/torch/*.py",
]

#: learnMSA backend name -> (framework module, hidten subpackage).
BACKENDS = {
    "tensorflow": ("tensorflow", "tf"),
    "pytorch": ("torch", "torch"),
}


@functools.lru_cache(maxsize=None)
def backend_is_usable(backend_name: str) -> bool:
    """Whether the backend can actually run here.

    Having the framework importable is necessary but not sufficient: learnMSA
    reaches it through ``hidten``, whose torch subpackage additionally requires
    triton. The TensorFlow environment has a CPU-only torch without triton, so
    ``find_spec("torch")`` succeeds there while the backend is unusable.

    The probe runs in a subprocess so that checking one backend never drags its
    framework into the pytest session, and it is skipped entirely when the
    framework is not installed -- so the common case costs nothing.
    """
    module, subpackage = BACKENDS[backend_name]
    if importlib.util.find_spec(module) is None:
        return False
    probe = subprocess.run(
        [sys.executable, "-c", f"import hidten.{subpackage}"],
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0


HAS_TENSORFLOW = backend_is_usable("tensorflow")
HAS_TORCH = backend_is_usable("pytorch")

collect_ignore_glob: list[str] = []
if not HAS_TENSORFLOW:
    collect_ignore_glob += TF_ONLY
if not HAS_TORCH:
    collect_ignore_glob += TORCH_ONLY


def pytest_configure(config) -> None:
    """Set up framework-specific test environment for the selected backend."""
    backend = os.environ.get("LEARNMSA_BACKEND")
    if backend is None:
        backend = "tensorflow" if HAS_TENSORFLOW else "pytorch"
    if backend == "tensorflow":
        # Without this TensorFlow grabs the whole GPU and later tests in the
        # session fail to allocate.
        os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

    config.addinivalue_line(
        "markers", "tf: test exercises the TensorFlow backend"
    )
    config.addinivalue_line(
        "markers", "torch: test exercises the PyTorch backend"
    )


def pytest_collection_modifyitems(items) -> None:
    """Mark every item by the backend package it lives in.

    Applying the marker here rather than writing ``pytestmark`` into each file
    keeps it from drifting away from the directory, which is what actually
    decides whether the test is collected.
    """
    import pytest

    for item in items:
        parts = item.path.parts
        if "tf" in parts:
            item.add_marker(pytest.mark.tf)
        elif "torch" in parts:
            item.add_marker(pytest.mark.torch)
