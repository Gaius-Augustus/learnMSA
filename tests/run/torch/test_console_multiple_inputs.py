"""The learnMSA command line with several input files (pytorch only)."""

import os
import subprocess
import sys
from pathlib import Path

from learnMSA.util.aligned_dataset import AlignedDataset
from learnMSA.util.sequence_dataset import SequenceDataset

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")
ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")


def test_several_input_files_write_one_alignment_each(tmp_path) -> None:
    inputs = [
        os.path.join(DATA, "felix.fa"),
        os.path.join(DATA, "felix_insert_delete.fa"),
    ]
    out_dir = tmp_path / "alignments"
    result = subprocess.run(
        [
            sys.executable, os.path.join(ROOT, "learnMSA.py"),
            "-i", *inputs, "-o", str(out_dir), "-f", "fasta",
            "--backend", "pytorch", "--silent", "--no_sequence_weights",
            "--epochs", "2", "1", "2", "--work_dir", str(tmp_path / "wd"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    for path in inputs:
        with SequenceDataset(path) as data, AlignedDataset(
            out_dir / (Path(path).stem + ".fasta")
        ) as msa:
            assert msa.seq_ids == data.seq_ids
