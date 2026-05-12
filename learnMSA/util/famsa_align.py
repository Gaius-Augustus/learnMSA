import Bio.SeqIO
from pathlib import Path


def align_with_famsa(fasta: str | Path, output: str | Path, threads: int = 0) -> None:
    # keep conditional import, famsa is an optional dependency
    from pyfamsa import Aligner as FamsaAligner, Sequence as FamsaSequence

    # Parse fasta
    sequences = [
        FamsaSequence(r.id.encode(), str(r.seq).encode())
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
