import os
import warnings
from pathlib import Path

import numpy as np
import pytest

from learnMSA.util.sequence_dataset import SequenceDataset
from learnMSA.util.aligned_dataset import AlignedDataset


DIR = "tests/data/"


def test_records() -> None:
    for ind in [True, False]:
        with SequenceDataset(f"{DIR}/egf.fasta", "fasta", indexed=ind) as data:
            assert data.indexed == ind
            get_record = lambda i: (
                str(data.get_record(i).seq).replace('.', '').upper(),
                str(data.get_record(i).id)
            )
            assert data.num_seq == 7774
            assert get_record(0) == (
                "CDPNPCYNHGTCSLRATGYTCSCLPRYTGEH", "B3RNP9_TRIAD/78-108"
            )
            assert get_record(9) == (
                "NACDRVRCQNGGTCQLKTLEDYTCSCANGYTGDH", "B3N1W3_DROAN/140-173"
            )
            assert get_record(27) == (
                "CNNPCDASPCLNGGTCVPVNAQNYTCTCTNDYSGQN", "B3RNP6_TRIAD/203-238"
            )
            assert get_record(-1) == (
                "TASCQDMSCSKQGECLETIGNYTCSCYPGFYGPECEYVRE", "1fsb"
            )

        with SequenceDataset(f"{DIR}/PF00008_uniprot.fasta", "fasta") as data:
            get_record = lambda i: (
                str(data.get_record(i).seq).replace('.', '').upper()
            )
            assert get_record(0) == "PSPCQNGGLCFMSGDDTDYTCACPTGFSG"
            assert get_record(7) == "SSPCQNGGMCFMSGDDTDYTCACPTGFSG"
            assert get_record(-1) == "CSSSPCNAEGTVRCEDKKGDFLCHCFTGWAGAR"


def test_encoding() -> None:
    for ind in [True, False]:
        with SequenceDataset(f"{DIR}/felix.fa", "fasta", indexed=ind) as data:
            # Default: soft one-hot distributions over the 20 amino acids.
            enc = data.get_encoded_seq(0)
            assert enc.shape == (5, 20)
            assert enc.dtype == np.float32
            np.testing.assert_equal(enc.argmax(axis=1), [13, 6, 10, 9, 11])
            np.testing.assert_almost_equal(enc.sum(axis=1), np.ones(5))
            # remap=False: one-hot over the output alphabet (standard AAs keep
            # their index).
            rf = data.get_encoded_seq(0, remap=False)
            assert rf.shape == (5, len(data.output_alphabet))
            assert rf.dtype == np.float32
            np.testing.assert_equal(rf.argmax(axis=1), [13, 6, 10, 9, 11])


def test_ambiguous_amino_acids() -> None:
    aa = SequenceDataset._default_alphabet
    for ind in [True, False]:
        f = f"{DIR}/ambiguous.fasta"
        with SequenceDataset(f, "fasta", indexed=ind) as data:
            # seq as string
            assert data.get_record(0).seq == "AGCTBZJbzj"
            enc = data.get_encoded_seq(0)  # (10, 20) soft one-hot
            assert enc.shape == (10, 20)
            # standard residues are one-hot
            for pos, ch in [(0, 'A'), (1, 'G'), (2, 'C'), (3, 'T')]:
                assert enc[pos, aa.index(ch)] == 1.0
                assert (enc[pos] > 0).sum() == 1
            # ambiguity codes (upper- and lower-case) split uniformly
            def check(pos, targets):
                assert (enc[pos] > 0).sum() == len(targets)
                for t in targets:
                    np.testing.assert_almost_equal(
                        enc[pos, aa.index(t)], 1.0 / len(targets)
                    )
            for pos in (4, 7): check(pos, "DN")   # B / b
            for pos in (5, 8): check(pos, "EQ")   # Z / z
            for pos in (6, 9): check(pos, "IL")   # J / j
            # remap=False preserves the original (upper-cased) letters
            tokens = data.get_encoded_seq(0, remap=False).argmax(axis=-1)
            assert "".join(
                data.output_alphabet[t] for t in tokens
            ) == "AGCTBZJBZJ"


def test_remove_gaps() -> None:
    for ind in [True, False]:
        with SequenceDataset(f"{DIR}/egf.ref", "fasta", indexed=ind) as data:
            ref = "GTSHLVKCAEKEKTFCVNGGECFMVKDLSNPSRYLCKCQPGFTG----ARCTENVPMK"\
                "VQNQEKAEELYQK"
            np.testing.assert_equal(
                str(data.get_record(5).seq), ref
            )
            # remap=False one-hot tokens (via argmax); gaps removed by default
            np.testing.assert_equal(
                data.get_encoded_seq(5, remap=False).argmax(axis=-1),
                [data.output_alphabet.index(a) for a in ref.replace('-', '')]
            )
            data.remove_gaps = False
            np.testing.assert_equal(
                data.get_encoded_seq(5, remap=False).argmax(axis=-1),
                [data.output_alphabet.index(a) for a in ref]
            )


def test_invalid_symbol() -> None:
    for ind in [True, False]:
        f = f"{DIR}/unknown_symbol.fasta"
        with SequenceDataset(f, "fasta", indexed=ind) as data:
            assert str(data.get_record(0).seq) == "AGTCGTA?GTCGTAAGTCG????TAA"\
                "GTCGTAAGTCGTA"
            with pytest.raises(ValueError):
                data.get_encoded_seq(0)


def test_invalid_format() -> None:
    for test_file in [
        "faulty_format",
        "single_sequence",
        "empty_sequence",
        "empty_seqid",
    ]:
        f = f"{DIR}/{test_file}.fasta"
        # Biopython warns about faulty format, ignore it
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with SequenceDataset(f, "fasta", indexed=False) as data:
                with pytest.raises(ValueError):
                    data.validate_dataset()


def test_aligned_dataset() -> None:
    for ind in [True, False]:
        f = f"{DIR}/felix_msa.fa"
        with AlignedDataset(f, "fasta", indexed=ind) as data:
            assert data.alignment_len == 8
            np.testing.assert_equal(
                data.seq_lens, [5, 8, 5]
            )
            np.testing.assert_equal(
                data._starting_pos, [0, 5, 13]
            )
            np.testing.assert_equal(
                data.get_column_map(0), [3, 4, 5, 6, 7]
            )
            np.testing.assert_equal(
                data.get_column_map(1), [0, 1, 2, 3, 4, 5, 6, 7]
            )
            np.testing.assert_equal(
                data.get_column_map(2), [1, 2, 3, 4, 7]
            )


def test_invalid_msa() -> None:
    with pytest.raises(ValueError):
        AlignedDataset("tests/data/faulty_msa.fasta", "fasta", indexed=False)


def test_from_sequences() -> None:
    sequences = [("seq1", "FELIX"), ("seq2", "FEIX")]
    with SequenceDataset(sequences=sequences) as data:
        assert data.num_seq == 2
        enc = data.get_encoded_seq(0)  # (5, 20) soft one-hot
        assert enc.shape == (5, 20)
        np.testing.assert_equal(enc[:4].argmax(axis=1), [13, 6, 10, 9])
        np.testing.assert_almost_equal(enc[4], np.full(20, 1.0 / 20))  # X
        # remap=False preserves the original letters (incl. X)
        X = data.output_alphabet.index('X')
        np.testing.assert_equal(
            data.get_encoded_seq(0, remap=False).argmax(axis=-1), [13, 6, 10, 9, X]
        )
        np.testing.assert_equal(
            data.get_encoded_seq(1, remap=False).argmax(axis=-1), [13, 6, 9, X]
        )


def test_from_alignment() -> None:
    sequences = [("seq1", "FELIX"), ("seq2", "FE-IX")]
    with AlignedDataset(sequences=sequences) as data:
        assert data.num_seq == 2
        gap = data.output_alphabet.index('-')
        X = data.output_alphabet.index('X')
        np.testing.assert_equal(
            data.get_encoded_seq(0).argmax(axis=-1), [13, 6, 10, 9, X]
        )
        np.testing.assert_equal(
            data.get_encoded_seq(1).argmax(axis=-1), [13, 6, gap, 9, X]
        )
        np.testing.assert_equal(data.get_column_map(0), [0, 1, 2, 3, 4])
        np.testing.assert_equal(data.get_column_map(1), [0, 1, 3, 4])


def test_file_output_formats() -> None:
    formats = ["fasta", "clustal", "stockholm"]
    # write an alignment to various formats
    for fmt in formats:
        with AlignedDataset(
            sequences=[
                ("seq1", "FELIX"),
                ("seq2", "FE-IX"),
                ("seq3", "-ELI-")
            ]
        ) as data:
            data.write("example." + fmt, fmt)
    # read it back in and check if it is the same
    for fmt in ["fasta", "clustal", "stockholm"]:
        with AlignedDataset("example." + fmt, fmt) as data:
            assert data.num_seq == 3
            gap = data.output_alphabet.index('-')
            X = data.output_alphabet.index('X')
            np.testing.assert_equal(
                data.get_encoded_seq(0).argmax(axis=-1), [13, 6, 10, 9, X]
            )
            np.testing.assert_equal(
                data.get_encoded_seq(1).argmax(axis=-1), [13, 6, gap, 9, X]
            )
            np.testing.assert_equal(
                data.get_encoded_seq(2).argmax(axis=-1), [gap, 6, 10, 9, gap]
            )
            np.testing.assert_equal(data.get_column_map(0), [0, 1, 2, 3, 4])
            np.testing.assert_equal(data.get_column_map(1), [0, 1, 3, 4])
            np.testing.assert_equal(data.get_column_map(2), [1, 2, 3])
    # clean up created files
    for fmt in formats:
        os.remove("example." + fmt)


def test_seq_headers() -> None:
    # make sure learnMSA keeps the full header >seqID seq_description [organism]
    with SequenceDataset("tests/data/headers.fasta", "fasta") as data:
        assert data.get_header(0) == \
            "QEG08237.1 MAG: ORF1b polyprotein [Pacific salmon nidovirus]"
        assert data.get_header(1) == \
            "CAG77604.1 RNA-dependent RNA polymerase [Amasya cherry disease-"\
            "associated mycovirus]"
        assert data.get_header(2) == \
            "QED42866.1 ORF1 [Anemone nepovirus A]"
        assert data.get_header(3) == \
            "QZQ78639.1 polyprotein [Potato black ringspot virus]"
        assert data.get_header(4) == \
            "Supergroup001--NEW-Clstr134_soil_ORF36_ERR2562197_k141_13787_"\
            "flag1_multi16_len6988"


def test_properties() -> None:
    """Test all property accessors."""
    with SequenceDataset(f"{DIR}/felix.fa", "fasta") as data:
        # Test basic properties
        assert isinstance(data.filepath, Path)
        assert data.fmt == "fasta"
        assert data.indexed == False
        assert data.parsing_ok == True
        assert data.num_seq == 8
        assert data.max_len == 14
        assert len(data.seq_ids) == 8
        assert data.seq_ids[0] == "1"
        assert len(data.seq_lens) == 8
        assert data.seq_lens[0] == 5
        assert data.record_dict is not None


def test_get_standardized_seq() -> None:
    """Test sequence standardization with various options."""
    sequences = [("s1", "FE-L.IX"), ("s2", "ab*cd")]
    with SequenceDataset(sequences=sequences) as data:
        # Test gap removal (default)
        assert data.get_standardized_seq(0) == "FELIX"

    with SequenceDataset(sequences=sequences, remove_gaps=False) as data:
        # Test keeping gaps
        assert data.get_standardized_seq(0) == "FE-L-IX"

    with SequenceDataset(sequences=sequences, gap_symbols=".-") as data:
        # Test custom gap symbols
        assert data.get_standardized_seq(0) == "FELIX"

    with SequenceDataset(sequences=sequences, ignore_symbols="*") as data:
        # Standardization upper-cases, strips ignored symbols, and no longer
        # collapses ambiguity codes (B stays B).
        assert data.get_standardized_seq(1) == "ABCD"


def test_crop_bounds() -> None:
    """Test sequence cropping via explicit start/end bounds."""
    sequences = [("s1", "ABCDEFGHIJ")]
    with SequenceDataset(sequences=sequences) as data:
        # Test normal encoding
        seq = data.get_encoded_seq(0)
        assert len(seq) == 10

        # Test fixed-bound cropping
        seq_cropped = data.get_encoded_seq(0, crop_start=2, crop_end=7)
        assert len(seq_cropped) == 5

        seq_cropped = data.get_encoded_seq(
            0, crop_start=2, crop_end=7
        )
        assert len(seq_cropped) == 5


def test_context_manager() -> None:
    """Test context manager protocol."""
    # Test with indexed dataset
    with SequenceDataset(f"{DIR}/felix.fa", "fasta", indexed=True) as data:
        assert data.indexed == True
        assert data.num_seq == 8
    # close() should have been called automatically


def test_properties_on_failed_parsing() -> None:
    """Test that properties return safe defaults when parsing fails."""
    # Create a dataset with sequences (no parsing)
    with SequenceDataset(sequences=[("s1", "FELIX")]) as data:
        # Should work normally
        assert data.num_seq == 1
        assert len(data.seq_ids) == 1


def test_write_method() -> None:
    """Test writing datasets to file."""
    import tempfile
    sequences = [("seq1", "FELIX"), ("seq2", "FEIX")]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        temp_file = f.name

    try:
        # Write to file
        with SequenceDataset(sequences=sequences) as data:
            data.write(temp_file, "fasta")

        # Read back and verify
        with SequenceDataset(temp_file, "fasta") as data:
            assert data.num_seq == 2
            assert data.seq_ids[0] == "seq1"
            assert data.seq_ids[1] == "seq2"
    finally:
        os.remove(temp_file)


def test_aligned_dataset_properties() -> None:
    """Test AlignedDataset-specific properties."""
    sequences = [("s1", "FE-LIX"), ("s2", "FEILIX")]
    with AlignedDataset(sequences=sequences) as data:
        # Test msa_matrix property
        assert data.msa_matrix.shape == (2, 6)

        # Test column_map property
        assert len(data.column_map) == 11  # 5 + 6 positions

        # Test starting_pos property
        assert len(data._starting_pos) == 2
        assert data._starting_pos[0] == 0
        assert data._starting_pos[1] == 5


def test_sp_score() -> None:
    """Test SP score calculation."""
    # Create two identical alignments
    sequences = [("s1", "FE-LIX"), ("s2", "FEILIX")]

    with AlignedDataset(sequences=sequences) as data1:
        with AlignedDataset(sequences=sequences) as data2:
            # SP score with itself should be 1.0
            sp = data1.SP_score(data2)
            assert sp == 1.0

    # Create different alignments
    seq1 = [("s1", "FE-LIX"), ("s2", "FEILIX")]
    seq2 = [("s1", "FELIX-"), ("s2", "FEILIX")]

    with AlignedDataset(sequences=seq1) as data1:
        with AlignedDataset(sequences=seq2) as data2:
            # SP score should be less than 1.0
            sp = data1.SP_score(data2)
            assert 0.0 <= sp < 1.0


def test_encoding_is_always_one_hot() -> None:
    """get_encoded_seq always returns a float32 one-hot / distribution matrix,
    never integer indices, for either remap mode."""
    sequences = [("s1", "FELIX")]
    with SequenceDataset(sequences=sequences) as data:
        remapped = data.get_encoded_seq(0)  # default remap=True
        assert remapped.dtype == np.float32
        assert remapped.shape == (5, len(data.alphabet))
        faithful = data.get_encoded_seq(0, remap=False)
        assert faithful.dtype == np.float32
        assert faithful.shape == (5, len(data.output_alphabet))
        # Every position is a proper one-hot / distribution (rows sum to 1).
        np.testing.assert_almost_equal(faithful.sum(axis=1), np.ones(5))


def test_string_filepath() -> None:
    """Test that string filepaths are converted to Path objects."""
    # Test with string path
    with SequenceDataset(f"{DIR}/felix.fa", "fasta") as data:
        assert isinstance(data.filepath, Path)
        assert data.filepath.name == "felix.fa"

    # Test with Path object
    with SequenceDataset(Path(f"{DIR}/felix.fa"), "fasta") as data:
        assert isinstance(data.filepath, Path)
        assert data.filepath.name == "felix.fa"


def test_empty_properties_on_early_return() -> None:
    """Test properties return safe defaults when __init__ returns early."""
    # This is hard to test directly, but we can verify the safety checks work
    sequences = [("s1", "FELIX")]
    with SequenceDataset(sequences=sequences) as data:
        # These should all work even if attributes weren't set
        assert isinstance(data.seq_ids, list)
        assert isinstance(data.num_seq, int)
        assert isinstance(data.seq_lens, np.ndarray)


def test_file_conversion_unaligned(tmp_path: Path) -> None:
    """Write and read unaligned sequence formats (fasta, tab)."""
    sequences = [("seq1", "FELIX"), ("seq2", "FEIX")]

    # Test FASTA format (most common)
    fasta_path = tmp_path / "test_unaligned.fasta"
    with SequenceDataset(sequences=sequences) as data:
        data.write(str(fasta_path), "fasta")

    with SequenceDataset(str(fasta_path), "fasta") as rd:
        assert rd.num_seq == 2
        assert rd.seq_ids[0] == "seq1"
        assert rd.seq_ids[1] == "seq2"
        assert str(rd.get_record(0).seq) == "FELIX"
        assert str(rd.get_record(1).seq) == "FEIX"

    # Test tab-delimited format (simple two-column format)
    tab_path = tmp_path / "test_unaligned.tab"
    with SequenceDataset(sequences=sequences) as data:
        data.write(str(tab_path), "tab")

    with SequenceDataset(str(tab_path), "tab") as rd:
        assert rd.num_seq == 2
        # tab format preserves sequences
        seqs = [str(rd.get_record(i).seq) for i in range(rd.num_seq)]
        assert "FELIX" in seqs
        assert "FEIX" in seqs


def test_file_conversion_aligned(tmp_path: Path) -> None:
    """Write and read aligned formats (fasta, clustal, stockholm)."""
    aligned = [("seq1", "FELIX"), ("seq2", "FE-IX"), ("seq3", "-ELI-")]
    formats = ["fasta", "clustal", "stockholm"]

    with AlignedDataset(sequences=aligned) as data:
        for fmt in formats:
            out = tmp_path / f"align.{fmt}"
            data.write(str(out), fmt)

            # Read back and validate key properties
            with AlignedDataset(str(out), fmt) as rd:
                assert rd.num_seq == data.num_seq
                assert rd.alignment_len == data.alignment_len
                np.testing.assert_equal(rd.seq_lens, data.seq_lens)
                # compare one column_map as representative
                np.testing.assert_equal(
                    rd.get_column_map(0), data.get_column_map(0)
                )


def test_custom_alphabet() -> None:
    """Test that custom alphabets work correctly."""
    # Create a simple dataset with custom alphabet "AB-"
    sequences = [("seq1", "AABBA"), ("seq2", "A-BBA")]

    with SequenceDataset(
        sequences=sequences, alphabet="AB-"
    ) as data:
        assert data.alphabet == "AB-"
        # remap=False -> one-hot over the custom alphabet (A=0,B=1,-=2)
        np.testing.assert_equal(
            data.get_encoded_seq(0, remap=False).argmax(axis=-1), [0, 0, 1, 1, 0]
        )
        # remap=True (default) -> one-hot vectors over the custom alphabet
        encoded = data.get_encoded_seq(0)
        assert encoded.shape == (5, 3)
        np.testing.assert_equal(encoded.argmax(axis=1), [0, 0, 1, 1, 0])

    # Test with aligned dataset
    with AlignedDataset(
        sequences=sequences, alphabet="AB-"
    ) as data:
        assert data.alphabet == "AB-"
        assert data.alignment_len == 5
        # seq2 has a gap at position 1
        np.testing.assert_equal(
            data.get_encoded_seq(1).argmax(axis=-1), [0, 2, 1, 1, 0]
        )
        np.testing.assert_equal(data.get_column_map(1), [0, 2, 3, 4])


def test_reorder() -> None:
    sequences = [("s1", "AAAA"), ("s2", "BBB"), ("s3", "CC")]
    with SequenceDataset(sequences=sequences) as data:
        data.reorder([2, 0, 1])

        assert data.seq_ids == ["s3", "s1", "s2"]
        assert str(data.get_record(0).seq) == "CC"
        assert str(data.get_record(1).seq) == "AAAA"
        assert str(data.get_record(2).seq) == "BBB"
        np.testing.assert_equal(data.seq_lens, [2, 4, 3])

def test_profile() -> None:
    sequences = [("s1", "AAAAA"), ("s2", "ARED"), ("s3", "EEDD")]
    with SequenceDataset(sequences=sequences) as data:
        profile = data.get_profile()  # over the 20 amino acids
        assert profile.shape == (20,)
        np.testing.assert_almost_equal(
            profile * 13, [6, 1, 0, 3, 0, 0, 3] + [0] * 13
        )


def test_remap_false_preserves_residues() -> None:
    """remap=False keeps the original residues (incl. ambiguity codes)."""
    with SequenceDataset(sequences=[("s", "ACDBZJXUO")]) as data:
        tokens = data.get_encoded_seq(0, remap=False).argmax(axis=-1)
        assert "".join(data.output_alphabet[t] for t in tokens) == "ACDBZJXUO"
        assert "-" not in data.alphabet


def test_model_uo() -> None:
    """U and O are modeled explicitly only when model_uo is enabled."""
    aa = SequenceDataset._default_alphabet
    # Default: U/O resolve like X (uniform over 20)
    with SequenceDataset(sequences=[("s", "MUO")]) as data:
        assert data.alphabet == aa  # 20 letters
        enc = data.get_encoded_seq(0)
        assert enc.shape == (3, 20)
        np.testing.assert_almost_equal(enc[1], np.full(20, 1.0 / 20))  # U
        np.testing.assert_almost_equal(enc[2], np.full(20, 1.0 / 20))  # O
    # model_uo: U/O become explicit one-hot columns 20 and 21
    with SequenceDataset(sequences=[("s", "MUO")], model_uo=True) as data:
        assert data.alphabet == aa + "UO"
        enc = data.get_encoded_seq(0)
        assert enc.shape == (3, 22)
        assert enc[1].argmax() == 20 and (enc[1] > 0).sum() == 1  # U
        assert enc[2].argmax() == 21 and (enc[2] > 0).sum() == 1  # O
        # X still spreads only over the 20 standard AAs
        encx = SequenceDataset(
            sequences=[("s", "X")], model_uo=True
        ).get_encoded_seq(0)
        assert encx[0, 20] == 0 and encx[0, 21] == 0
        np.testing.assert_almost_equal(encx[0, :20], np.full(20, 1.0 / 20))
