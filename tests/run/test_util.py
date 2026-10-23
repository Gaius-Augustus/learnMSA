from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pytest

from learnMSA import Configuration
from learnMSA.util.sequence_dataset import SequenceDataset
from learnMSA.run.util import load_struct_data


def _write_fasta(path: Path, seqs: list[tuple[str, str]]) -> None:
    with open(path, "w") as f:
        for seq_id, seq in seqs:
            f.write(f">{seq_id}\n{seq}\n")


def test_load_struct_data_reordering(tmp_path: Path) -> None:
    """load_struct_data must reorder the structural dataset to match the
    sequence order of the amino acid dataset."""
    aa_seqs = [
        ("seqA", "ACDEF"),
        ("seqB", "GHIKL"),
        ("seqC", "MNPQR"),
    ]
    # Structural sequences in a *different* order than aa_seqs
    struct_seqs_shuffled = [
        ("seqC", "CDEFG"),
        ("seqA", "ACDEF"),
        ("seqB", "GHIKL"),
    ]

    aa_file = tmp_path / "aa.fasta"
    struct_file = tmp_path / "struct.fasta"
    _write_fasta(aa_file, aa_seqs)
    _write_fasta(struct_file, struct_seqs_shuffled)

    config = Configuration()
    config.input_output.struct_file = struct_file
    # Use the default structural alphabet (20 amino acids as 3Di tokens)
    struct_alphabet = config.structure.structural_alphabet

    with ExitStack() as stack:
        data = stack.enter_context(
            SequenceDataset(aa_file, "fasta")
        )
        struct_data = load_struct_data(config, data, stack)

    assert struct_data is not None
    assert struct_data.seq_ids == data.seq_ids
    # After reordering, seqA should be first, seqB second, seqC third
    assert struct_data.seq_ids == ["seqA", "seqB", "seqC"]


@pytest.mark.parametrize("mask_ends", [False, True])
def test_3di_fasta_termini_are_unknown_if_masked(
    tmp_path: Path, mask_ends: bool
) -> None:
    """Foldseek writes D for the termini, where it cannot compute a 3Di
    state; with mask_ends (the default), learnMSA reads them as unknown."""
    assert Configuration().structure.mask_ends is True
    aa_file = tmp_path / "aa.fasta"
    struct_file = tmp_path / "struct.fasta"
    _write_fasta(aa_file, [("seqA", "ACDEF"), ("seqB", "GHI")])
    _write_fasta(struct_file, [("seqA", "DVLPD"), ("seqB", "DPD")])
    config = Configuration()
    config.input_output.struct_file = struct_file
    config.structure.mask_ends = mask_ends
    alphabet = config.structure.structural_alphabet

    with ExitStack() as stack:
        data = stack.enter_context(SequenceDataset(aa_file, "fasta"))
        struct_data = load_struct_data(config, data, stack)
        encoded = struct_data.get_encoded_seq(0)

    assert config.structure.input_format == "tokens"
    one_hot = np.eye(len(alphabet))[[alphabet.index(c) for c in "DVLPD"]]
    if mask_ends:
        one_hot[[0, -1]] = 1.0
    np.testing.assert_equal(encoded, one_hot)


def test_load_struct_data_from_structure_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory is converted to a 3Di FASTA in the work dir with Foldseek
    and then read like any 3Di FASTA, including the masked termini."""
    from tests.structure.foldseek_fake import install_fake_foldseek

    install_fake_foldseek(tmp_path / "bin", monkeypatch)
    (tmp_path / "pdbs").mkdir()
    aa_file = tmp_path / "aa.fasta"
    _write_fasta(aa_file, [("seqB", "GHI"), ("seqA", "ACDEF")])
    config = Configuration()
    config.input_output.input_file = aa_file
    config.input_output.struct_file = tmp_path / "pdbs"
    config.input_output.work_dir = str(tmp_path / "work")
    alphabet = config.structure.structural_alphabet

    with ExitStack() as stack:
        data = stack.enter_context(SequenceDataset(aa_file, "fasta"))
        struct_data = load_struct_data(config, data, stack)
        encoded = struct_data.get_encoded_seq(1)

    assert (tmp_path / "work" / "aa.3di.fasta").is_file()
    assert config.structure.input_format == "tokens"
    assert config.input_output.struct_file == tmp_path / "pdbs"
    assert struct_data.seq_ids == ["seqB", "seqA"]
    expected = np.eye(len(alphabet))[[alphabet.index(c) for c in "DVLPD"]]
    expected[[0, -1]] = 1.0
    np.testing.assert_equal(encoded, expected)


def test_load_struct_data_structure_directory_id_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.structure.foldseek_fake import install_fake_foldseek

    install_fake_foldseek(tmp_path / "bin", monkeypatch)
    (tmp_path / "pdbs").mkdir()
    aa_file = tmp_path / "aa.fasta"
    _write_fasta(aa_file, [("seqA", "ACDEF"), ("other", "GHI")])
    config = Configuration()
    config.input_output.input_file = aa_file
    config.input_output.struct_file = tmp_path / "pdbs"
    config.input_output.work_dir = str(tmp_path / "work")

    with ExitStack() as stack:
        data = stack.enter_context(SequenceDataset(aa_file, "fasta"))
        with pytest.raises(ValueError, match="_<chain>"):
            load_struct_data(config, data, stack)


def _structure_dir_config(tmp_path: Path) -> Configuration:
    (tmp_path / "pdbs").mkdir()
    config = Configuration()
    config.input_output.struct_file = tmp_path / "pdbs"
    config.input_output.work_dir = str(tmp_path / "work")
    return config


def test_resolve_input_file_infers_sequences_from_structures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without -i, the amino acid sequences and the 3Di letters of a
    structure directory are written to the work dir and used as inputs."""
    import argparse

    from learnMSA.run.util import resolve_input_file
    from tests.structure.foldseek_fake import install_fake_foldseek

    install_fake_foldseek(tmp_path / "bin", monkeypatch)
    config = _structure_dir_config(tmp_path)
    resolve_input_file(config, argparse.ArgumentParser())

    work = tmp_path / "work"
    assert config.input_output.input_file == work / "pdbs.fasta"
    assert config.input_output.struct_file == work / "pdbs.3di.fasta"
    with ExitStack() as stack:
        data = stack.enter_context(
            SequenceDataset(config.input_output.input_file, "fasta")
        )
        struct_data = load_struct_data(config, data, stack)
        assert data.seq_ids == ["seqA", "seqB"]
        assert list(data.seq_lens) == list(struct_data.seq_lens)


def test_resolve_input_file_warns_about_unknown_residues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Unknown residues are kept as X with a warning, not rejected."""
    import argparse

    from learnMSA.run.util import resolve_input_file
    from tests.structure.foldseek_fake import install_fake_foldseek

    install_fake_foldseek(tmp_path / "bin", monkeypatch)
    monkeypatch.setenv("FAKE_FOLDSEEK_AA", ">seqA\nACXEF\n>seqB\nGHI\n")
    config = _structure_dir_config(tmp_path)
    config.input_output.verbose = True
    resolve_input_file(config, argparse.ArgumentParser())

    assert config.input_output.input_file == tmp_path / "work" / "pdbs.fasta"
    out = capsys.readouterr().out
    assert "Warning: 1 structure(s) have unknown residues (e.g. seqA)" in out


@pytest.mark.parametrize("aa, reason", [
    (">seqA\nACDEF\n>seqB\nGHIK\n", "seqB (missing or a different length"),
    (">seqA\nACDEF\n", "seqB (missing or a different length"),
])
def test_resolve_input_file_asks_for_sequences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], aa: str, reason: str,
) -> None:
    import argparse

    from learnMSA.run.util import resolve_input_file
    from tests.structure.foldseek_fake import install_fake_foldseek

    install_fake_foldseek(tmp_path / "bin", monkeypatch)
    monkeypatch.setenv("FAKE_FOLDSEEK_AA", aa)
    config = _structure_dir_config(tmp_path)
    with pytest.raises(SystemExit):
        resolve_input_file(config, argparse.ArgumentParser())
    err = capsys.readouterr().err
    assert reason in err
    assert "-i/--in_file" in err


def test_resolve_input_file_requires_input_without_structures() -> None:
    import argparse

    from learnMSA.run.util import resolve_input_file

    with pytest.raises(SystemExit):
        resolve_input_file(Configuration(), argparse.ArgumentParser())
