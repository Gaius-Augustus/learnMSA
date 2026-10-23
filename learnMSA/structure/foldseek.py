"""Foldseek 3Di letters from a directory of structures (stdlib only)."""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from shutil import which


def structures_to_3di(
    pdb_dir: str | Path,
    output: str | Path,
    work_dir: str | Path,
    aa_output: str | Path | None = None,
    verbose: bool = False,
) -> None:
    """Converts a directory of PDB/mmCIF structures to a 3Di FASTA file.

    Equivalent to ``foldseek createdb``, ``lndb`` and ``convert2fasta``. The
    sequence IDs of the FASTA are the Foldseek record IDs: the file name
    without extension, plus ``_<chain>`` for files with several chains.

    Args:
        pdb_dir: Directory containing PDB/mmCIF structure files.
        output: Output path for the 3Di FASTA file.
        work_dir: Directory for the intermediate Foldseek database, which is
            removed afterwards.
        aa_output: If set, also writes the amino acid sequences of the
            structures (one letter per 3Di letter, X for unknown residues)
            to this FASTA file, with the IDs as the only header.
        verbose: Whether to report the conversion. Foldseek itself only
            prints errors.
    """
    if which("foldseek") is None:
        print(
            "foldseek is not installed or not in path. Please install it "\
            "(e.g. conda install -c conda-forge -c bioconda foldseek "\
            "--no-update-deps) or pass a 3Di FASTA file with --struct."
        )
        sys.exit(1)
    os.makedirs(work_dir, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix="foldseek_3di_", dir=work_dir)
    tmp_db = os.path.join(tmp_dir, "tmp_db")
    try:
        _run_foldseek("createdb", str(pdb_dir), tmp_db)
        _run_foldseek("lndb", f"{tmp_db}_h", f"{tmp_db}_ss_h")
        _run_foldseek("convert2fasta", f"{tmp_db}_ss", str(output))
        if aa_output is not None:
            tmp_aa = os.path.join(tmp_dir, "aa.fasta")
            _run_foldseek("convert2fasta", tmp_db, tmp_aa)
            _strip_descriptions(tmp_aa, aa_output)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    if verbose:
        print("Structures converted to 3Di with foldseek")


def _run_foldseek(*args: str) -> None:
    """Runs a foldseek module; with -v 1 it only prints errors."""
    subprocess.run(["foldseek", *args, "-v", "1"], check=True)


def _strip_descriptions(fasta: str | Path, output: str | Path) -> None:
    """Copies a FASTA file, keeping only the ID of every header. Foldseek
    copies e.g. the PDB title into the description."""
    with open(fasta) as src, open(output, "w") as dst:
        for line in src:
            if line.startswith(">"):
                line = line.split(maxsplit=1)[0] + "\n"
            dst.write(line)
