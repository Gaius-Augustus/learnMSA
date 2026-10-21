"""Groups of identical or near-identical sequences of a dataset."""
from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

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
    def from_rep_of(cls, rep_of: np.ndarray) -> RedundancyMap:
        """Builds the map from the original index of the representative of
        every sequence (``rep_of[r] == r`` for representatives)."""
        rep_of = np.asarray(rep_of, dtype=np.int64)
        reps = np.flatnonzero(rep_of == np.arange(rep_of.size))
        slot = np.full(rep_of.size, -1, dtype=np.int64)
        slot[reps] = np.arange(reps.size)
        rep_index = slot[rep_of]
        assert np.all(rep_index >= 0), \
            "Every representative must represent itself."
        return cls(rep_index=rep_index, rep_full_index=reps)


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
    """Groups identical sequences and, if ``identity < 1``, sequences of equal
    length that differ only by substitutions.

    A sequence joins the group of an mmseqs2 cluster representative if both
    have the same length and at least ``identity`` of their positions match.
    Other members of the cluster represent themselves. Without mmseqs2, only
    identical sequences are grouped.

    Args:
        data: The sequences.
        identity: Minimum fraction of identical positions.
        work_dir: Directory for the temporary clustering files.
        verbose: Print a warning if mmseqs2 is missing.
    """
    from learnMSA.util.clustering import run_linclust

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
    os.makedirs(work_dir, exist_ok=True)
    prefix = os.path.join(work_dir, "plm_reuse")
    fasta = prefix + ".fasta"
    with open(fasta, "w") as f:
        for k, i in enumerate(unique):
            f.write(f">s{k}\n{data.get_standardized_seq(int(i))}\n")
    try:
        clustering = run_linclust(
            fasta, prefix, identity,
            ["-c", "0.95", "--cov-mode", "0", "--cluster-mode", "2",
             "--alignment-mode", "3", "--kmer-per-seq", "100"],
        )
    finally:
        for suffix in (".fasta", "_cluster.tsv", "_rep_seq.fasta",
                       "_all_seqs.fasta"):
            if os.path.exists(prefix + suffix):
                os.remove(prefix + suffix)
        shutil.rmtree(prefix + "_tmp", ignore_errors=True)

    # Representative of each unique sequence, as a position in unique
    unique_rep = np.arange(unique.size)
    for rep, member in zip(clustering.representative, clustering.sequence):
        r, m = int(rep[1:]), int(member[1:])
        if r == m:
            continue
        a = data.get_standardized_seq(int(unique[r]))
        b = data.get_standardized_seq(int(unique[m]))
        if len(a) != len(b):
            continue
        mismatches = np.count_nonzero(
            np.frombuffer(a.encode(), np.uint8)
            != np.frombuffer(b.encode(), np.uint8)
        )
        if mismatches <= (1 - identity) * len(a):
            unique_rep[m] = r
    return RedundancyMap.from_rep_of(
        unique[unique_rep[duplicates.rep_index]]
    )
