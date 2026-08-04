"""Backend isolation, checked from the PyTorch side.

The mirror of ``tests/backend/tf/test_isolation.py``; see
``tests/backend/isolation.py`` for what each check does and why it runs in a
subprocess.
"""

from tests.backend import isolation

BACKEND = "pytorch"


def test_does_not_import_tensorflow() -> None:
    isolation.check_does_not_import_opposing_framework(BACKEND)


def test_refuses_to_switch() -> None:
    isolation.check_refuses_to_switch(BACKEND)


def test_env_var_is_honored() -> None:
    isolation.check_env_var_is_honored(BACKEND)
