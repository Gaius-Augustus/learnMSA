import re
from pathlib import Path

import Bio.SeqIO


# Letters outside pyfamsa's FAMSA_ALPHABET ("ARNDCQEGHILKMFPSTWYVBZX*").
# Hard-coded so that pyfamsa is not imported at module load.
_NON_FAMSA_PATTERN = re.compile(r"[^ARNDCQEGHILKMFPSTWYVBZX*]")


def to_famsa_alphabet(seq: str) -> str:
    """Replace residues FAMSA cannot encode (e.g. U, O, J) with X.

    FAMSA rejects such sequences outright. The length of *seq* is preserved.
    """
    return _NON_FAMSA_PATTERN.sub("X", seq)


def align_with_famsa(fasta: str | Path, output: str | Path, threads: int = 0) -> None:
    # keep conditional import, famsa is an optional dependency
    from pyfamsa import Aligner as FamsaAligner, Sequence as FamsaSequence

    # Parse fasta
    sequences = [
        FamsaSequence(
            r.id.encode(), to_famsa_alphabet(str(r.seq).upper()).encode()
        )
        for r in Bio.SeqIO.parse(fasta, "fasta")
    ]

    # Align
    aligner = FamsaAligner(threads = threads)
    msa = aligner.align(sequences)
    msa = [
        (sequence.id.decode(), sequence.sequence.decode()) for sequence in msa
    ]

    # Write output
    with open(output, "w") as file:
        for seq_id, seq in msa:
            file.write(f">{seq_id}\n{seq}\n")
