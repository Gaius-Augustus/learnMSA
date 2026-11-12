"""Groups of identical or near-identical sequences of a dataset."""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from learnMSA.util.sequence_dataset import SequenceDataset


@dataclass
class RedundancyMap:
    """Maps every sequence of a dataset to a representative sequence.

    Representatives keep the order of the input. Every representative
    represents itself.
    """

    rep_index: np.ndarray
    """Shape (num_full,). For each original sequence, the position of its
    representative in ``rep_full_index``."""

    rep_full_index: np.ndarray
    """Shape (num_rep,). Ascending original indices of the
    representatives."""

    positions: dict[int, np.ndarray] = field(default_factory=dict)
    """Contains sequences by original index whose residues do not map 1:1
    to those of its representative. For each residue reports the index in the
    representative it is aligned to or -1 for unaligned residues."""

    @property
    def num_full(self) -> int:
        return int(self.rep_index.size)

    @property
    def num_rep(self) -> int:
        return int(self.rep_full_index.size)

    @property
    def is_trivial(self) -> bool:
        """True if every sequence represents itself."""
        return self.num_rep == self.num_full

    @classmethod
    def from_rep_of(
        cls,
        rep_of: np.ndarray,
        positions: dict[int, np.ndarray] | None = None,
    ) -> RedundancyMap:
        """Builds the map from the original index of the representative of
        every sequence (``rep_of[r] == r`` for representatives) and the
        optional residue ``positions``."""
        rep_of = np.asarray(rep_of, dtype=np.int64)
        reps = np.flatnonzero(rep_of == np.arange(rep_of.size))
        slot = np.full(rep_of.size, -1, dtype=np.int64)
        slot[reps] = np.arange(reps.size)
        rep_index = slot[rep_of]
        assert np.all(rep_index >= 0), \
            "Every representative must represent itself."
        return cls(rep_index=rep_index, rep_full_index=reps,
                   positions=positions or {})


def find_duplicates(data: SequenceDataset) -> RedundancyMap:
    """Groups identical sequences. The first occurrence represents a group.

    Sequences are compared after standardization (uppercase, gaps removed).
    Only a 16 byte digest per unique sequence is kept in memory.
    """
    rep_of = np.arange(data.num_seq, dtype=np.int64)
    first: dict[bytes, int] = {}
    for i in range(data.num_seq):
        seq = data.get_standardized_seq(i)
        key = hashlib.blake2b(seq.encode(), digest_size=16).digest()
        j = first.setdefault(key, i)
        # A digest collision leaves the sequence as its own representative
        if j != i and data.get_standardized_seq(j) == seq:
            rep_of[i] = j
    return RedundancyMap.from_rep_of(rep_of)


def find_near_identical(
    data: SequenceDataset,
    identity: float,
    work_dir: Path | str,
    verbose: bool = False,
) -> RedundancyMap:
    """Groups identical or closely related (``identity < 1``) sequences and
    returns a ``RedundancyMap`` including member-to-representative alignments.
    A member joins the group of the representative if the alignment has at
    least ``identity`` identical positions and spans 70% of the member
    (span: `(end - start + 1) / len`). The representative may be longer.
    The coverage criterion follows the template selection of Expresso
    (Armougom et al., 2006), which uses an identity of 0.6.

    Args:
        data: The sequences.
        identity: Minimum fraction of identical positions.
        work_dir: Directory for the temporary clustering files.
        verbose: Print a warning if mmseqs2 is missing.
    """
    duplicates = find_duplicates(data)
    if identity >= 1 or duplicates.num_rep < 2:
        return duplicates
    if shutil.which("mmseqs") is None:
        if verbose:
            print(
                "Warning: mmseqs2 is not installed. Only identical sequences "
                "share embeddings."
            )
        return duplicates

    unique = duplicates.rep_full_index
    lens = np.asarray(data.seq_lens)[unique]
    os.makedirs(work_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        fasta = os.path.join(tmp, "seqs.fasta")
        with open(fasta, "w") as f:
            for k, i in enumerate(unique):
                f.write(f">{k}\n{data.get_standardized_seq(int(i))}\n")
        alignments = _align_to_representatives(fasta, tmp, identity)

    # Representative of each unique sequence, as a position in unique
    unique_rep = np.arange(unique.size)
    unique_pos: dict[int, np.ndarray] = {}
    for r, m, qstart, tstart, qaln, taln in alignments.itertuples(
        index=False
    ):
        if r == m:
            continue
        unique_rep[m] = r
        pos = _aligned_positions(qaln, taln, qstart, tstart, lens[m])
        if lens[m] != lens[r] or np.any(pos != np.arange(lens[m])):
            unique_pos[m] = pos
    remapped = np.flatnonzero(
        np.isin(duplicates.rep_index, list(unique_pos))
    )
    return RedundancyMap.from_rep_of(
        unique[unique_rep[duplicates.rep_index]],
        {int(i): unique_pos[duplicates.rep_index[i]] for i in remapped},
    )


def _align_to_representatives(
    fasta: str, tmp: str, identity: float
) -> pd.DataFrame:
    """Clusters the sequences in ``fasta`` (named by their index) and aligns
    each member to its representative with mmseqs2.

    ``mmseqs cluster`` rather than ``linclust``: the ungapped prefilter of
    linclust misses members with internal indels.

    Returns:
        One row per member (representatives included) whose alignment passes
        the filters, with the columns ``rep``, ``member`` (indices),
        ``qstart``, ``tstart`` (1-based) and the aligned strings ``qaln``
        (representative) and ``taln`` (member).
    """
    db, clu, aln, out = (
        os.path.join(tmp, name) for name in ("db", "clu", "aln", "aln.tsv")
    )
    # --cov-mode 1: coverage of the target (the member); the representative
    # (query) may be longer
    filters = ["--min-seq-id", str(identity), "-c", "0.7", "--cov-mode", "1"]
    for args in (
        ["createdb", fasta, db],
        ["cluster", db, clu, os.path.join(tmp, "tmp"), *filters,
         "--cluster-mode", "2", "--alignment-mode", "3"],
        ["align", db, db, clu, aln, "-a", "-e", "inf", *filters],
        ["convertalis", db, db, aln, out, "--format-output",
         "query,target,qstart,tstart,qaln,taln"],
    ):
        subprocess.run(["mmseqs", *args, "-v", "1"], check=True)
    return pd.read_csv(
        out, sep="\t",
        names=["rep", "member", "qstart", "tstart", "qaln", "taln"],
    )


def _aligned_positions(
    qaln: str, taln: str, qstart: int, tstart: int, length: int
) -> np.ndarray:
    """Position in the representative of each residue of a member of the
    given length, -1 where the alignment does not match it."""
    gap = ord("-")
    q = np.frombuffer(qaln.encode(), np.uint8) != gap
    t = np.frombuffer(taln.encode(), np.uint8) != gap
    pos = np.full(length, -1, dtype=np.int32)
    pos[tstart - 1:tstart - 1 + t.sum()] = np.where(
        q, qstart - 2 + np.cumsum(q), -1
    )[t]
    return pos
