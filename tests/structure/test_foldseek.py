import subprocess
from pathlib import Path

import pytest

from learnMSA.structure.foldseek import structures_to_3di
from tests.structure.foldseek_fake import (install_fake_foldseek,
                                           uninstall_foldseek)


@pytest.mark.parametrize("verbose", [False, True])
def test_structures_to_3di_runs_foldseek(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], verbose: bool,
) -> None:
    log = install_fake_foldseek(tmp_path / "bin", monkeypatch)
    output = tmp_path / "work" / "out.fasta"
    structures_to_3di(
        tmp_path / "pdbs", output, tmp_path / "work", verbose=verbose
    )

    calls = [line.split() for line in log.read_text().splitlines()]
    db = calls[0][2]
    # Foldseek only prints errors (-v 1).
    assert calls == [
        ["createdb", str(tmp_path / "pdbs"), db, "-v", "1"],
        ["lndb", f"{db}_h", f"{db}_ss_h", "-v", "1"],
        ["convert2fasta", f"{db}_ss", str(output), "-v", "1"],
    ]
    assert output.read_text() == ">seqA desc\nDVLPD\n>seqB\nDPD\n"
    # The intermediate database is removed, only the output remains.
    assert [p.name for p in (tmp_path / "work").iterdir()] == ["out.fasta"]
    expected = "Structures converted to 3Di with foldseek\n" if verbose else ""
    assert capsys.readouterr().out == expected


def test_structures_to_3di_with_amino_acids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The amino acid sequences come from the same database as the 3Di
    letters; their headers keep only the ID."""
    log = install_fake_foldseek(tmp_path / "bin", monkeypatch)
    work = tmp_path / "work"
    structures_to_3di(
        tmp_path / "pdbs", work / "out.3di.fasta", work,
        aa_output=work / "out.fasta",
    )

    calls = [line.split() for line in log.read_text().splitlines()]
    assert [c[0] for c in calls] == [
        "createdb", "lndb", "convert2fasta", "convert2fasta"
    ]
    assert calls[3][1] == calls[0][2]
    assert (work / "out.fasta").read_text() == ">seqA\nACDEF\n>seqB\nGHI\n"
    assert sorted(p.name for p in work.iterdir()) == [
        "out.3di.fasta", "out.fasta"
    ]


def test_structures_to_3di_failure_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_fake_foldseek(tmp_path / "bin", monkeypatch)
    monkeypatch.setenv("FAKE_FOLDSEEK_FAIL", "lndb")
    with pytest.raises(subprocess.CalledProcessError):
        structures_to_3di(
            tmp_path / "pdbs", tmp_path / "work" / "out.fasta",
            tmp_path / "work",
        )
    assert list((tmp_path / "work").iterdir()) == []


def test_structures_to_3di_without_foldseek(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    uninstall_foldseek(monkeypatch, tmp_path / "empty")
    with pytest.raises(SystemExit) as exc:
        structures_to_3di(
            tmp_path / "pdbs", tmp_path / "out.fasta", tmp_path / "work"
        )
    assert exc.value.code == 1
    assert "foldseek is not installed" in capsys.readouterr().out
