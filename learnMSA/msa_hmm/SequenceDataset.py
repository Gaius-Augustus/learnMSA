import math
import re
from pathlib import Path
from types import TracebackType
from typing import Self

import numpy as np
from Bio import Seq, SeqIO, SeqRecord
from Bio.File import _IndexedSeqFileDict


class SequenceDataset:
    """
    Manages a set of protein sequences.
    """
    alphabet: str = "ARNDCQEGHILKMFPSTWYVXUO-"

    def __init__(
        self,
        filepath: Path | str | None = None,
        fmt: str = "fasta",
        sequences: list[tuple[str, str]] | None = None,
        indexed: bool = False,
    ) -> None:
        """
        Args:
            filepath (Path): Path to a sequence file in any supported format.
            fmt (str): Format of the file. Can be any format supported by
                Biopython's SeqIO.
            sequences (list): A list of id/sequence pairs as strings. If given,
                filepath and fmt arguments are ignored.
            indexed (bool): If True, avoid loading the whole file into memory
                at once.  Otherwise regular parsing is used.
        """
        if sequences is None:
            # Attempt to parse the file when no sequences are given
            assert filepath is not None, \
                "filepath must be provided when sequences are None"
            if isinstance(filepath, str):
                filepath = Path(filepath)
            self._filepath = filepath
            self._fmt = fmt
            self._indexed = indexed
            try:
                if indexed:
                    self._record_dict = SeqIO.index(filepath, fmt)
                else:
                    with open(filepath, "rt", encoding="utf-8") as handle:
                        self._record_dict = SeqIO.to_dict(
                            SeqIO.parse(handle, fmt)
                        )
                self._parsing_ok = True
            except ValueError as err:
                self._parsing_ok = False
                # hold the error and raise it when calling validate_dataset
                self._err = err
            if not self._parsing_ok:
                return
        else:
            self._parsing_ok = True
            self._filepath = Path()
            self._fmt = ""
            self._indexed = False
            self._record_dict = {
                s[0] : SeqRecord.SeqRecord(Seq.Seq(s[1])
                if isinstance(s[1], str) else s[1], id=s[0])
                for s in sequences
            }
        # Since Python 3.7 key order is preserved in dictionaries so this list
        # is correctly ordered
        self._seq_ids = list(self._record_dict)
        self._num_seq = len(self._seq_ids)
        self._seq_lens = np.array([
            sum([1 for x in str(self.get_record(i).seq) if x.isalpha()])
            for i in range(self._num_seq)
        ])
        self._max_len = np.amax(self._seq_lens) if self._seq_lens.size > 0 else 0

    def __len__(self) -> int:
        return self.num_seq

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None
    ) -> None:
        self.close()

    @property
    def filepath(self) -> Path:
        """Path to the sequence file."""
        return self._filepath

    @property
    def fmt(self) -> str:
        """Format of the sequence file."""
        return self._fmt

    @property
    def seq_ids(self) -> list[str]:
        """List of sequence IDs."""
        if not hasattr(self, "_seq_ids"):
            return []
        return self._seq_ids

    @property
    def num_seq(self) -> int:
        """Total number of sequences in the dataset."""
        if not hasattr(self, "_num_seq"):
            return 0
        return self._num_seq

    @property
    def seq_lens(self) -> np.ndarray:
        """Lengths of the sequences in the dataset."""
        if not hasattr(self, "_seq_lens"):
            return np.array([])
        return self._seq_lens

    @property
    def max_len(self) -> int:
        """Maximum sequence length in the dataset."""
        if not hasattr(self, "_max_len"):
            return 0
        return self._max_len

    @property
    def indexed(self) -> bool:
        """Whether the dataset is indexed."""
        return self._indexed

    @property
    def parsing_ok(self) -> bool:
        """Whether the dataset was parsed successfully."""
        return self._parsing_ok

    @property
    def record_dict(self) -> dict[str, SeqRecord.SeqRecord] | _IndexedSeqFileDict:
        """Dictionary(-like) object that takes sequence IDs as keys and maps
        them to SeqRecord objects."""
        return self._record_dict

    def close(self) -> None:
        if self.indexed and isinstance(self.record_dict, _IndexedSeqFileDict):
            self.record_dict.close()

    def get_record(self, i: int) -> SeqRecord.SeqRecord:
        """ Get the SeqRecord object for sequence i. """
        return self.record_dict[self.seq_ids[i]]  # type: ignore

    def get_header(self, i: int) -> str:
        """ Get the header/description for sequence i. """
        return self.get_record(i).description

    def get_alphabet_no_gap(self) -> str:
        """ Get the alphabet without gap characters. """
        return type(self).alphabet[:-1]

    def get_standardized_seq(
        self,
        i: int,
        remove_gaps: bool = True,
        gap_symbols: str = "-.",
        ignore_symbols: str = "",
        replace_with_x: str = "",
    ):
        """
        Returns a standardized sequence string for sequence i containing only
        uppercase letters from the standard amino acid alphabet and either
        standard gap character '-' or no gap characters at all.

        Args:
            i (int): Index of the sequence to process.
            remove_gaps (bool): If True, all gap characters provided in
                gap_symbols are removed from the sequence. If False, all gap
                characters are replaced with the first character in gap_symbols.
            gap_symbols (str): String containing all characters to be treated
                as gap characters.
            ignore_symbols (str): String containing all characters to be
                ignored/removed from the sequence.
            replace_with_x (str): String containing all characters to be
                replaced with 'X' in the sequence.
        """
        seq_str = str(self.get_record(i).upper().seq)
        # replace non-standard aminoacids with X
        for aa in replace_with_x:
            seq_str = seq_str.replace(aa, 'X')
        if remove_gaps:
            for s in gap_symbols:
                seq_str = seq_str.replace(s, '')
        else:
            # unify gap symbols
            for s in gap_symbols:
                seq_str = seq_str.replace(s, gap_symbols[0])
        # strip other symbols
        for s in ignore_symbols:
            seq_str = seq_str.replace(s, '')
        return seq_str

    def get_encoded_seq(
        self,
        i: int,
        remove_gaps: bool = True,
        gap_symbols: str = "-.",
        ignore_symbols: str = "",
        replace_with_x: str = "BZJ",
        crop_to_length: float = math.inf,
        validate_alphabet: bool = True,
        dtype: type[np.integer] = np.int16,
        return_crop_boundaries: bool = False
    ) -> np.ndarray | tuple[np.ndarray, int, int]:
        """
        Returns sequence i encoded as a numpy array of integers.

        Args:
            i (int): Index of the sequence to process.
            remove_gaps (bool): Passed to get_standardized_seq.
            gap_symbols (str): Passed to get_standardized_seq.
            ignore_symbols (str): Passed to get_standardized_seq.
            replace_with_x (str): Passed to get_standardized_seq.
            crop_to_length (float): If the sequence is longer than this length,
                a random crop of this length is returned. If the sequence is
                shorter than this length, the whole sequence is returned.
            validate_alphabet (bool): If True, check that the sequence contains
                only characters from the defined alphabet.
            dtype (type): Numpy integer type to use for the encoded sequence.
            return_crop_boundaries (bool): If True, also return the start and
                end indices of the crop within the original sequence.
        """
        seq_str = self.get_standardized_seq(
            i, remove_gaps, gap_symbols, ignore_symbols, replace_with_x
        )
        # Make sure the sequences do not contain any other symbols
        if validate_alphabet:
            if bool(re.compile(rf"[^{type(self).alphabet}]").search(seq_str)):
                raise ValueError(
                    "Found unknown character(s) in sequence "\
                    f"{self.seq_ids[i]}. Allowed alphabet: {type(self).alphabet}."
                )
        seq = np.array(
            [type(self).alphabet.index(aa) for aa in seq_str], dtype=dtype
        )
        if seq.shape[0] > crop_to_length:
            # Crop randomly
            start = np.random.randint(0, seq.shape[0] - crop_to_length + 1)
            end = start + crop_to_length
        else:
            start = 0
            end = seq.shape[0]
        seq = seq[start:end]
        if return_crop_boundaries:
            return seq, start, end
        else:
            return seq


    def validate_dataset(
            self,
            single_seq_ok: bool = False,
            empty_seq_id_ok: bool = False,
            dublicate_seq_id_ok: bool = False,
    ) -> None:
        """Raise an error if the dataset is not valid for processing.
        """
        if not self.parsing_ok:
            raise self._err

        if len(self.seq_ids) == 1 and not single_seq_ok:
            raise ValueError(
                f"File {self._filepath} contains only a single sequence."
            )

        if len(self.seq_ids) == 0:
            raise ValueError(
                f"Could not parse any sequences from {self._filepath}."
            )

        if np.amin(self.seq_lens) == 0:
            raise ValueError(f"{self._filepath} contains empty sequences.")

        if not empty_seq_id_ok:
            for sid in self.seq_ids:
                if sid == '':
                    raise ValueError(
                        f"File {self._filepath} contains an empty sequence ID, "\
                        "which is not allowed."
                    )
        if len(self.seq_ids) > len(set(self.seq_ids)) and not dublicate_seq_id_ok:
            raise ValueError(
                f"File {self._filepath} contains duplicated sequence IDs. "
                "learnMSA requires unique sequence IDs."
            )

    def write(
            self,
            filepath: Path | str,
            fmt="fasta",
            standardize_sequences: bool = False,
    ) -> None:
        """
        Write the dataset to a file.

        Args:
            filepath (Path): Path to the output file.
            fmt (str): Format of the output file. Can be any format supported
                by Biopython's SeqIO.
            standardize_sequences (bool): If True, sequences are converted to
                uppercase and non-standard amino acids are replaced with 'X'.
                Dots are replaced with dashes.
        """
        sequences = list(self.record_dict.values())
        for s in sequences:
            if standardize_sequences:
                seq = self.get_standardized_seq(
                    self.seq_ids.index(s.id),
                    remove_gaps=False,
                )
                s.seq = Seq.Seq(seq)
            else:
                s.seq = Seq.Seq(s.seq)
            s.description = ""
        SeqIO.write(sequences, filepath, fmt)


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
