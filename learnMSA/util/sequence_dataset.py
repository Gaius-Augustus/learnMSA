import math
import re
from pathlib import Path
from types import TracebackType
from typing import Self

import numpy as np
from Bio import Seq, SeqIO, SeqRecord
from Bio.File import _IndexedSeqFileDict

from .dataset import Dataset


class SequenceDataset(Dataset):
    """
    Manages a set of unaligned protein sequences.

    Attributes:
        alphabet (str): The amino acid alphabet used for encoding sequences.
        filepath (Path): The path to the sequence file.
        fmt (str): The format of the sequence file.
        num_seq (int): The number of sequences in the dataset.
        seq_lens (np.ndarray): An array containing the lengths of each sequence.
        max_len (int): The length of the longest sequence in the dataset.
        seq_ids (list[str]): The list of sequence IDs.
        indexed (bool): Whether the dataset is indexed.
        parsing_ok (bool): Whether the dataset was parsed successfully.
        record_dict (dict|_IndexedSeqFileDict): A dictionary(-like) object that
            maps sequence IDs to SeqRecord objects.
    """
    _default_alphabet: str = "ARNDCQEGHILKMFPSTWYV"

    def __init__(
        self,
        filepath: Path | str | None = None,
        fmt: str = "fasta",
        sequences: list[tuple[str, str]] | None = None,
        indexed: bool = False,
        alphabet: str | None = None,
        remove_gaps: bool = True,
        gap_symbols: str = "-.",
        ignore_symbols: str = "",
        validate_alphabet: bool = True,
        encode_as_one_hot: bool = False,
        remap: bool = True,
        model_uo: bool = False,
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
            alphabet (str): The alphabet to use for encoding sequences. If None,
                the amino acid alphabet is used and ambiguity codes are resolved
                by remapping (see remap). Providing an explicit alphabet (e.g.
                for 3Di states) disables the amino acid remapping.
            remove_gaps (bool): If True, all gap characters provided in
                gap_symbols are removed from the sequences. If False, all gap
                characters are replaced with the first character in gap_symbols.
            gap_symbols (str): String containing all characters to be treated as
                gap characters.
            ignore_symbols (str): String containing all characters to be
                ignored/removed from the sequence.
            validate_alphabet (bool): If True, raise an error if any sequence
                contains characters not in the specified alphabet after
                standardization.
            encode_as_one_hot (bool): If True, sequences are encoded as one-hot
                vectors instead of integer indices (only used when remap is off,
                e.g. for the 3Di alphabet).
            remap (bool): Amino acid mode only. If True (default),
                get_encoded_seq returns per-residue distributions over the 20
                (or 22 with model_uo) standard amino acids, resolving ambiguity
                codes (B->{D,N}, Z->{E,Q}, J->{I,L}, X->all 20). If False, the
                sequence is encoded as integer indices over the full render
                alphabet (standard + non-standard letters + gap), preserving the
                original residues. Ignored for non-amino-acid alphabets.
            model_uo (bool): Amino acid mode only. If True, U (selenocysteine)
                and O (pyrrolysine) are modeled explicitly (22-letter alphabet);
                otherwise they are resolved like X.
        """
        self.model_uo = model_uo
        self.remap = remap
        # Amino acid mode is used whenever no explicit alphabet is provided.
        self._is_aa = alphabet is None
        if alphabet is not None:
            self.alphabet = alphabet
            self.render_alphabet = alphabet
            self._aa_dist: dict[str, np.ndarray] | None = None
        else:
            base = SequenceDataset._default_alphabet  # 20 standard AAs
            self.alphabet = base + ("UO" if model_uo else "")
            # Characters recognized for faithful (remap=False) rendering,
            # appended in a fixed order with the gap symbol last.
            nonstd = "".join(c for c in "XBZJUO" if c not in self.alphabet)
            self.render_alphabet = self.alphabet + nonstd + "-"
            self._aa_dist = self._build_aa_dist()
        self.remove_gaps = remove_gaps
        self.gap_symbols = gap_symbols
        self.ignore_symbols = ignore_symbols
        self.validate_alphabet = validate_alphabet
        self.encode_as_one_hot = encode_as_one_hot
        self._invalid_char_pattern = re.compile(
            rf"[^{re.escape(self.render_alphabet)}]"
        )

        self.filepath = Path()
        self.fmt = ""
        self.indexed = False
        self.parsing_ok = False
        self.record_dict: dict[str, SeqRecord.SeqRecord] | _IndexedSeqFileDict = {}
        self.seq_ids: list[str] = []
        self.num_seq = 0
        self.seq_lens = np.array([])
        self.max_len = 0

        if sequences is None:
            # Attempt to parse the file when no sequences are given
            assert filepath is not None, \
                "filepath must be provided when sequences are None"
            if isinstance(filepath, str):
                filepath = Path(filepath)
            self.filepath = filepath
            self.fmt = fmt
            self.indexed = indexed
            try:
                if indexed:
                    self.record_dict = SeqIO.index(filepath, fmt)
                else:
                    with open(filepath, "rt", encoding="utf-8") as handle:
                        self.record_dict = SeqIO.to_dict(
                            SeqIO.parse(handle, fmt)
                        )
                self.parsing_ok = True
            except ValueError as err:
                self.parsing_ok = False
                # hold the error and raise it when calling validate_dataset
                self._err = err
            if not self.parsing_ok:
                return
        else:
            self.parsing_ok = True
            self.record_dict = {
                s[0] : SeqRecord.SeqRecord(Seq.Seq(s[1])
                if isinstance(s[1], str) else s[1], id=s[0])
                for s in sequences
            }
        # Since Python 3.7 key order is preserved in dictionaries so this list
        # is correctly ordered
        self.seq_ids = list(self.record_dict)
        self.num_seq = len(self.seq_ids)
        self.seq_lens = np.array([
            sum([1 for x in str(self.get_record(i).seq) if x.isalpha()])
            for i in range(self.num_seq)
        ])
        self.max_len = np.amax(self.seq_lens) if self.seq_lens.size > 0 else 0

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

    def close(self) -> None:
        if self.indexed and isinstance(self.record_dict, _IndexedSeqFileDict):
            self.record_dict.close()

    def get_record(self, i: int) -> SeqRecord.SeqRecord:
        """ Get the SeqRecord object for sequence i. """
        return self.record_dict[self.seq_ids[i]]  # type: ignore

    def get_header(self, i: int) -> str:
        """ Get the header/description for sequence i. """
        return self.get_record(i).description

    def index(self, seq_id: str) -> int:
        """ Get the index of the sequence with the given sequence ID. """
        return self.seq_ids.index(seq_id)

    def get_alphabet_no_gap(self) -> str:
        """ Get the alphabet without a trailing gap symbol. For amino acids the
        model alphabet already has no gap; custom alphabets (e.g. AB-) are
        stripped of their trailing gap. """
        if self.alphabet.endswith('-'):
            return self.alphabet[:-1]
        return self.alphabet

    def _build_aa_dist(self) -> dict[str, np.ndarray]:
        """Build a lookup from residue character to a probability vector over
        the model alphabet (20 or 22 amino acids). Ambiguity codes are resolved
        uniformly over their target set (B->{D,N}, Z->{E,Q}, J->{I,L},
        X->all 20). U/O map to their own column when modeled, else to X.
        """
        aa20 = SequenceDataset._default_alphabet  # 20 standard amino acids
        dim = len(self.alphabet)
        idx = {c: i for i, c in enumerate(self.alphabet)}
        dist: dict[str, np.ndarray] = {}
        # Standard amino acids (and U/O when modeled explicitly)
        for c in self.alphabet:
            v = np.zeros(dim, dtype=np.float32)
            v[idx[c]] = 1.0
            dist[c] = v
        # Ambiguity codes as uniform distributions over their target sets
        for c, targets in (("B", "DN"), ("Z", "EQ"), ("J", "IL"), ("X", aa20)):
            v = np.zeros(dim, dtype=np.float32)
            for t in targets:
                v[idx[t]] = 1.0 / len(targets)
            dist[c] = v
        # U/O resolve like X (uniform over the 20 standard) when not modeled
        if not self.model_uo:
            dist["U"] = dist["X"].copy()
            dist["O"] = dist["X"].copy()
        return dist

    def render_token_distributions(self) -> np.ndarray:
        """Matrix of shape (len(render_alphabet), len(alphabet)) mapping each
        integer render token to its distribution over the model alphabet.
        The gap token maps to an all-zero row. Amino acid mode only.
        """
        mat = np.zeros(
            (len(self.render_alphabet), len(self.alphabet)), dtype=np.float32
        )
        if self._aa_dist is not None:
            for t, ch in enumerate(self.render_alphabet):
                if ch in self._aa_dist:
                    mat[t] = self._aa_dist[ch]
        return mat

    def get_standardized_seq(self, i: int):
        """
        Returns a standardized sequence string for sequence i containing only
        uppercase letters and either the standard gap character '-' or no gap
        characters at all. Ambiguity codes are preserved (resolved later during
        encoding).
        """
        seq_str = str(self.get_record(i).upper().seq)
        if self.remove_gaps:
            for s in self.gap_symbols:
                seq_str = seq_str.replace(s, '')
        else:
            # unify gap symbols
            for s in self.gap_symbols:
                seq_str = seq_str.replace(s, self.gap_symbols[0])
        # strip other symbols
        for s in self.ignore_symbols:
            seq_str = seq_str.replace(s, '')
        return seq_str

    def get_encoded_seq(
        self,
        i: int,
        crop_start: int | None = None,
        crop_end: int | None = None,
        dtype: type[np.integer | np.floating] = np.int16,
        remap: bool | None = None,
    ) -> np.ndarray:
        """
        Returns sequence i encoded as a numpy array.

        In amino acid mode with remapping (the default), the result is a
        float32 array of shape (L, alphabet_size) with one probability
        distribution over the 20 (or 22) amino acids per residue. With
        remapping disabled the result is an integer array of indices over the
        render alphabet (standard + non-standard letters + gap), preserving the
        original residues for output.

        Args:
            i (int): Index of the sequence to process.
            crop_start (int | None): Optional inclusive crop start index.
                If None, defaults to 0.
            crop_end (int | None): Optional exclusive crop end index.
                If None, defaults to sequence length.
            dtype (type): Numpy integer type used for the integer encoding.
                Ignored when a distribution encoding is returned.
            remap (bool | None): Override the encoding mode. If None, the
                instance default (self.remap) is used.
        """
        if remap is None:
            remap = self.remap
        seq_str = self.get_standardized_seq(i)
        # Make sure the sequences do not contain any other symbols
        if self.validate_alphabet:
            if bool(self._invalid_char_pattern.search(seq_str)):
                raise ValueError(
                    "Found unknown character(s) in sequence "\
                    f"{self.seq_ids[i]}. Allowed alphabet: "\
                    f"{self.render_alphabet}."
                )

        if remap:
            # Per-residue probability vectors over the model alphabet. Padding
            # is represented downstream by all-zero vectors.
            if self._is_aa:
                assert self._aa_dist is not None
                if len(seq_str) == 0:
                    seq = np.zeros((0, len(self.alphabet)), dtype=np.float32)
                else:
                    seq = np.stack(
                        [self._aa_dist[aa] for aa in seq_str], axis=0
                    )
            else:
                # One-hot over the (custom) alphabet.
                seq = np.zeros(
                    (len(seq_str), len(self.alphabet)), dtype=np.float32
                )
                for k, ch in enumerate(seq_str):
                    seq[k, self.alphabet.index(ch)] = 1.0
        else:
            seq = np.array(
                [self.render_alphabet.index(aa) for aa in seq_str], dtype=dtype
            )

        start = 0 if crop_start is None else crop_start
        end = seq.shape[0] if crop_end is None else crop_end

        if start < 0 or end < 0:
            raise ValueError("crop_start and crop_end must be non-negative.")
        if end < start:
            raise ValueError(
                f"crop_end must be >= crop_start, got start={start}, end={end}."
            )
        if end > seq.shape[0]:
            raise ValueError(
                f"crop_end {end} exceeds sequence length {seq.shape[0]}."
            )

        seq = seq[start:end]

        if not remap and self.encode_as_one_hot:
            one_hot = np.zeros(
                (seq.shape[0], len(self.render_alphabet)), dtype=np.float32
            )
            one_hot[np.arange(seq.shape[0]), seq] = 1 # type: ignore
            return one_hot

        return seq

    def get_profile(self) -> np.ndarray:
        """
        Get the profile of the dataset, i.e. a 1D array of shape
        (alphabet_size,) with the relative frequencies of each symbol in the
        model alphabet across the whole dataset.
        """
        profile = np.zeros((len(self.alphabet),), dtype=np.float32)
        for i in range(self.num_seq):
            if self._is_aa:
                # Sum the per-residue distributions over the model alphabet
                seq = self.get_encoded_seq(i, remap=True)
                profile += np.sum(seq, axis=0)
            else:
                seq = self.get_encoded_seq(i, remap=False)
                profile += np.bincount(
                    seq, minlength=profile.shape[0]
                )[:profile.shape[0]]
        total = np.sum(profile)
        if total > 0:
            profile /= total
        return profile

    def empty(
        self,
        shape: tuple[int, ...],
        dtype: type[np.integer | np.floating] = np.int16,
    ) -> np.ndarray:
        if self.remap:
            # Vector mode: padding positions are all-zero vectors; the terminal
            # channel is constructed downstream as 1 - sum.
            return np.zeros(shape + (len(self.alphabet),), dtype=np.float32)
        if self.encode_as_one_hot:
            alphabet_size = len(self.render_alphabet)
            one_hot_shape = shape + (alphabet_size,)
            empty = np.zeros(one_hot_shape, dtype=np.float32)
            empty[..., alphabet_size - 1] = 1  # Initialize with terminal symbol
            return empty
        empty = np.zeros(shape, dtype=dtype)
        empty += len(self.render_alphabet)-1 # Initialize with terminal symbols
        return empty


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
                f"File {self.filepath} contains only a single sequence."
            )

        if len(self.seq_ids) == 0:
            raise ValueError(
                f"Could not parse any sequences from {self.filepath}."
            )

        if np.amin(self.seq_lens) == 0:
            raise ValueError(f"{self.filepath} contains empty sequences.")

        if not empty_seq_id_ok:
            for sid in self.seq_ids:
                if sid == '':
                    raise ValueError(
                        f"File {self.filepath} contains an empty sequence ID, "\
                        "which is not allowed."
                    )
        if len(self.seq_ids) > len(set(self.seq_ids)) and not dublicate_seq_id_ok:
            raise ValueError(
                f"File {self.filepath} contains duplicated sequence IDs. "
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
                seq = self.get_standardized_seq(self.seq_ids.index(s.id or ""))
                s.seq = Seq.Seq(seq)
            else:
                s.seq = Seq.Seq(s.seq)
            s.description = ""
        SeqIO.write(sequences, filepath, fmt)

    def reorder(self, permutation: list[int] | np.ndarray) -> None:
        """
        Reorder sequences in-place using a permutation of sequence indices.

        Args:
            permutation: A 1D permutation containing each index in
                [0, num_seq - 1] exactly once.
        """
        perm = np.asarray(permutation)
        perm = perm.astype(np.int64, copy=False)
        self.seq_ids = [self.seq_ids[i] for i in perm]
        self.seq_lens = self.seq_lens[perm]

    def get_dtype(self) -> type[np.integer | np.floating]:
        """Return the dtype of the encoded sequences."""
        if self.remap or self.encode_as_one_hot:
            return np.float32
        return np.int16
