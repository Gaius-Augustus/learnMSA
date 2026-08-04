"""Guards the backend-neutrality invariant.

learnMSA supports more than one tensor framework, and the neutral parts of the
package must import without any framework at all.

These checks run in subprocesses on purpose: the pytest session itself imports
a framework through other test modules, so an in-process
``assert "tensorflow" not in sys.modules`` would be meaningless.

The ``NEUTRAL_MODULES`` list is a ratchet. Every refactoring stage that frees a
module from its framework dependency adds it here, and it can never be removed.

The other half of the invariant -- that running one backend never imports the
other -- can only be checked where that backend is installed, so it lives in
``tests/backend/tf`` and ``tests/backend/torch``.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Modules that must import without pulling in any tensor framework.
NEUTRAL_MODULES = [
    "learnMSA",
    "learnMSA.backend",
    "learnMSA.config",
    "learnMSA.util",
    "learnMSA.util.tensor",
    "learnMSA.util.dataset",
    "learnMSA.util.sequence_dataset",
    "learnMSA.util.aligned_dataset",
    "learnMSA.util.embedding_dataset",
    "learnMSA.util.embedding_cache",
    "learnMSA.util.clustering",
    "learnMSA.hmm.util.value_set",
    "learnMSA.hmm.util.value_set_emb",
    "learnMSA.hmm.util.transition_index_set",
    "learnMSA.hmm.util.util",
    "learnMSA.hmm.priors",
    "learnMSA.hmm.joint_util",
    "learnMSA.hmm.layer",
    "learnMSA.util.visualize",
    "learnMSA.model",
    "learnMSA.model.model",
    "learnMSA.model.checkpoint",
    "learnMSA.model.phmm_mixin",
    "learnMSA.model.surgery",
    "learnMSA.model.context",
    "learnMSA.model.select",
    "learnMSA.model.training_util",
    "learnMSA.model.batch_generator",
    "learnMSA.model.bucketing",
    "learnMSA.align",
    "learnMSA.align.align",
    "learnMSA.align.alignment_model",
    "learnMSA.align.align_hits",
    "learnMSA.align.align_inserts",
    "learnMSA.align.alignment_metadata",
    "learnMSA.align.decode_util",
    "learnMSA.tree.initializer",
    "learnMSA.protein_language_models",
    "learnMSA.run.args",
    "learnMSA.run.args_to_config",
    "learnMSA.run.help",
    "learnMSA.run.util",
    "learnMSA.run.console",
]


#: Frameworks that must not appear in ``sys.modules`` after a neutral import.
FRAMEWORKS = ("tensorflow", "torch", "jax")

_IMPORT_PROBE = """
import importlib, sys
module = sys.argv[1]
importlib.import_module(module)
leaked = [f for f in {frameworks!r} if f in sys.modules]
if leaked:
    raise SystemExit(f"{{module}} imported {{leaked}}")
""".format(frameworks=FRAMEWORKS)


def _run(code: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


@pytest.mark.parametrize("module", NEUTRAL_MODULES)
def test_neutral_module_imports_no_framework(module: str) -> None:
    result = _run(_IMPORT_PROBE, module)
    assert result.returncode == 0, (
        f"{module} is not backend-neutral:\n{result.stdout}{result.stderr}"
    )


# --- static guards -------------------------------------------------------
# An import probe cannot see a framework import that hides inside a function
# body and is never executed. These greps catch that case.

_FRAMEWORK_IMPORT = re.compile(
    r"^\s*(?:import|from)\s+(?:tensorflow|torch)\b", re.MULTILINE
)

#: Files that may import a framework inside a function body, because dispatching
#: to a framework is precisely their job. Everything else must be clean.
LAZY_DISPATCH_ALLOWED = {
    "learnMSA/backend.py",
}


def _neutral_source_files() -> list[Path]:
    """The source file backing each module in ``NEUTRAL_MODULES``.

    Packages resolve to their ``__init__.py`` only; sibling modules that are not
    listed are deliberately not swept in.
    """
    files = []
    for module in NEUTRAL_MODULES:
        path = REPO_ROOT / Path(*module.split("."))
        if (path / "__init__.py").exists():
            files.append(path / "__init__.py")
        elif path.with_suffix(".py").exists():
            files.append(path.with_suffix(".py"))
        else:
            raise AssertionError(f"no source file found for {module}")
    return sorted(set(files))


#: Only the backend packages may reach into the framework HMM object; everyone
#: else goes through the neutral accessors on PHMMLayer.
_HMM_ATTRIBUTE_ACCESS = re.compile(r"\bphmm_layer\.hmm\b|\blayer\.hmm\.")

def _is_backend_package(relative: str) -> bool:
    """Whether a path lives inside a backend subpackage (``.../tf/...``)."""
    return "/tf/" in relative or "/torch/" in relative


def test_only_backend_packages_touch_the_framework_hmm() -> None:
    offenders = []
    for path in sorted((REPO_ROOT / "learnMSA").rglob("*.py")):
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative.startswith("learnMSA/legacy/"):
            continue
        if _is_backend_package(relative):
            continue
        text = path.read_text()
        for match in _HMM_ATTRIBUTE_ACCESS.finditer(text):
            line_no = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{relative}:{line_no}")
    assert not offenders, (
        "reach into PHMMLayer.hmm only from a backend package; use the neutral "
        "accessors (transition_matrix, emission_matrix, ...) instead: "
        + ", ".join(offenders)
    )


def test_neutral_sources_have_no_framework_imports() -> None:
    offenders = []
    for path in _neutral_source_files():
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative in LAZY_DISPATCH_ALLOWED:
            continue
        text = path.read_text()
        for match in _FRAMEWORK_IMPORT.finditer(text):
            line_no = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{relative}:{line_no}")
    assert not offenders, (
        "backend-neutral sources must not import a framework: "
        + ", ".join(offenders)
    )
