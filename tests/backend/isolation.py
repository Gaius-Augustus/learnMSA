"""The backend-isolation checks, shared by the two backend packages.

Running one backend must never import the other, and selecting a backend must
be a one-way door. Both statements can only be checked where the backend under
test is actually installed, so ``tests/backend/tf`` and ``tests/backend/torch``
each call these with their own name -- which is also what keeps the two files
from having to know about each other.

Everything runs in a subprocess: the pytest session has long since imported a
framework by the time these execute, and ``set_backend`` cannot be undone
within a process.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: learnMSA backend name -> (own framework module, backend model module).
_BACKEND_MODULES = {
    "tensorflow": ("tensorflow", "learnMSA.model.tf.model"),
    "pytorch": ("torch", "learnMSA.model.torch.model"),
}

#: The framework each backend must not drag in.
_OPPOSING = {"tensorflow": "torch", "pytorch": "tensorflow"}


def _run(code: str, env: dict[str, str] | None = None):
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
    )


def check_does_not_import_opposing_framework(backend: str) -> None:
    """Selecting ``backend`` and importing its model must not load the other."""
    _, model_module = _BACKEND_MODULES[backend]
    opposing = _OPPOSING[backend]
    probe = f"""
import sys
import learnMSA.backend as backend
backend.set_backend({backend!r})
import {model_module}  # noqa: F401
leaked = sorted(
    m for m in sys.modules
    if m == {opposing!r} or m.startswith({opposing!r} + ".")
)
if leaked:
    raise SystemExit(f"{backend} backend imported {opposing}: {{leaked[:10]}}")
"""
    result = _run(probe)
    assert result.returncode == 0, f"{result.stdout}{result.stderr}"


def check_refuses_to_switch(backend: str) -> None:
    """Switching away from a selected backend must raise, not silently work."""
    other = next(name for name in _BACKEND_MODULES if name != backend)
    probe = f"""
import learnMSA.backend as backend
backend.set_backend({backend!r})
try:
    backend.set_backend({other!r})
except RuntimeError:
    pass
else:
    raise SystemExit("switching backends should have raised RuntimeError")
"""
    result = _run(probe)
    assert result.returncode == 0, f"{result.stdout}{result.stderr}"


def check_env_var_is_honored(backend: str) -> None:
    """``LEARNMSA_BACKEND`` must win over automatic detection."""
    probe = f"""
import learnMSA.backend as backend
assert backend.get_backend() == {backend!r}, backend.get_backend()
"""
    result = _run(probe, env={**os.environ, "LEARNMSA_BACKEND": backend})
    assert result.returncode == 0, f"{result.stdout}{result.stderr}"
