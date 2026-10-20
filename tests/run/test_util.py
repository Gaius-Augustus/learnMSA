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
    state; with mask_ends, learnMSA reads them as unknown."""
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
