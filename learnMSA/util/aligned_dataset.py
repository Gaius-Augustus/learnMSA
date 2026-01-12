from pathlib import Path

import numpy as np

from learnMSA.util.sequence_dataset import SequenceDataset


class AlignedDataset(SequenceDataset):
    """
    Manages a multiple sequence alignment.
    """
    def __init__(
            self,
            filepath: Path | str | None = None,
            fmt: str = "fasta",
            aligned_sequences: list[tuple[str, str]] | None = None,
            indexed: bool = False,
            single_seq_ok: bool = False,
    ) -> None:
        """
        Args:
            filepath (Path): Path to a sequence file in any supported format.
            fmt (str): Format of the file. Can be any format supported by
                Biopython's SeqIO.
            aligned_sequences (list): A list of id/sequence pairs as strings.
                If given, filepath and fmt arguments are ignored.
            indexed (bool): If True, avoid loading the whole file into memory
                at once.  Otherwise regular parsing is used.
            single_seq_ok (bool): If True, allow datasets with a single
                sequence.
        """
        super().__init__(filepath, fmt, aligned_sequences, indexed)
        self._single_seq_ok = single_seq_ok
        self.validate_dataset()

        # Create MSA matrix
        self._msa_matrix = np.zeros((self.num_seq, len(self.get_record(0))), dtype=np.int16)
        for i in range(self.num_seq):
            self._msa_matrix[i,:] = self.get_encoded_seq(
                i, remove_gaps=False, dtype=np.int16
            )

        # Compute a mapping from sequence positions to MSA-column index
        # A-B--C -> 112223
        cumsum = np.cumsum(self._msa_matrix != type(self).alphabet.index('-'), axis=1)
        # 112223 -> 0112223 -> [[(i+1) - i]] -> 101001
        diff = np.diff(np.insert(cumsum, 0, 0.0, axis=1), axis=1)
        diff_where = [
            np.argwhere(diff[i,:]).flatten() for i in range(diff.shape[0])
        ]
        self._column_map = np.concatenate(diff_where).flatten()
        self._starting_pos = np.cumsum(self.seq_lens)
        self._starting_pos[1:] = self._starting_pos[:-1]
        self._starting_pos[0] = 0
        self._alignment_len = self._msa_matrix.shape[1]

    def validate_dataset(self) -> None:
        """Raise an error if the MSA is not valid for processing.
        """
        super().validate_dataset(
            single_seq_ok=self._single_seq_ok,
            empty_seq_id_ok=False,
            dublicate_seq_id_ok=False
        )
        record_lens = np.array([
            len(self.get_record(i)) for i in range(self.num_seq)
        ])
        if np.any(record_lens != record_lens[0]):
            raise ValueError(
                f"File {self._filepath} contains sequences of different "\
                "lengths."
            )

    @property
    def column_map(self) ->  np.ndarray:
        """Mapping from sequence positions to MSA-column index."""
        return self._column_map

    @property
    def msa_matrix(self) -> np.ndarray:
        """MSA matrix as a 2D numpy array of shape (num_seq, alignment_len)."""
        return self._msa_matrix

    @property
    def alignment_len(self) -> int:
        """Length of the alignment (number of columns)."""
        return self._alignment_len

    def get_column_map(self, i : int) -> np.ndarray:
        """
        Get the mapping from sequence positions to MSA-column index for a
        specific sequence.
        """
        s = self._starting_pos[i]
        e = s + self.seq_lens[i]
        return self._column_map[s:e]


    def SP_score(
            self, ref_data : "AlignedDataset", batch=512
    ) -> float:
        """
        Compute the SP-score of this alignment with respect to a reference
        alignment.

        Args:
            ref_data (AlignedDataset): Reference alignment.
            batch (int): Number of sequences to process in each batch. Lower
                values reduce memory consumption but increase computation time.
        """
        total_len = sum(self.seq_lens)
        n = 0
        true_positives = 0
        self_positives = 0
        ref_positives = 0
        while n < self.column_map.shape[0]:
            self_homologs = np.expand_dims(self.column_map,0)==\
                np.expand_dims(self.column_map[n:n+batch],1)
            ref_homologs = np.expand_dims(ref_data.column_map,0)==\
                np.expand_dims(ref_data.column_map[n:n+batch],1)
            true_positives += np.sum(np.logical_and(self_homologs, ref_homologs))
            self_positives += np.sum(self_homologs)
            ref_positives += np.sum(ref_homologs)
            n+=batch
        true_positives -= total_len
        sp = true_positives / max(1, ref_positives - total_len)
        return sp
