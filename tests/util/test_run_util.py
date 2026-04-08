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
