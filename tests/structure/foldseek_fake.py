"""A stand-in ``foldseek`` executable for tests."""

import os
import stat
from pathlib import Path

import pytest

# Logs every call to ``calls.log`` next to the script. ``convert2fasta``
# writes ``FAKE_FOLDSEEK_3DI`` for the 3Di database (``*_ss``) and
# ``FAKE_FOLDSEEK_AA`` otherwise (defaults: two matching sequences); the
# command named by ``FAKE_FOLDSEEK_FAIL`` exits with 3.
_SCRIPT = """#!/bin/sh
echo "$@" >> "$(dirname "$0")/calls.log"
if [ "$1" = "$FAKE_FOLDSEEK_FAIL" ]; then exit 3; fi
if [ "$1" = "convert2fasta" ]; then
    case "$2" in
    *_ss) printf '%b' \\
        "${FAKE_FOLDSEEK_3DI:->seqA desc\\nDVLPD\\n>seqB\\nDPD\\n}" > "$3" ;;
    *) printf '%b' \\
        "${FAKE_FOLDSEEK_AA:->seqA desc\\nACDEF\\n>seqB\\nGHI\\n}" > "$3" ;;
    esac
fi
"""


def install_fake_foldseek(
    directory: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Puts a fake ``foldseek`` first on PATH and returns its call log."""
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / "foldseek"
    script.write_text(_SCRIPT)
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{directory}{os.pathsep}{os.environ['PATH']}")
    return directory / "calls.log"


def uninstall_foldseek(monkeypatch: pytest.MonkeyPatch, tmp: Path) -> None:
    """Makes sure no ``foldseek`` is found."""
    tmp.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("PATH", str(tmp))
