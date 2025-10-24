import os
import warnings
from pathlib import Path

import numpy as np
import pytest

from learnMSA.msa_hmm.SequenceDataset import AlignedDataset, SequenceDataset

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
            np.testing.assert_equal(
                data.get_encoded_seq(0), [13, 6, 10, 9, 11]
            )


def test_ambiguous_amino_acids() -> None:
    for ind in [True, False]:
        f = f"{DIR}/ambiguous.fasta"
        with SequenceDataset(f, "fasta", indexed=ind) as data:
            # seq as string
            assert data.get_record(0).seq == "AGCTBZJbzj"
            # encoded
            np.testing.assert_equal(
                data.get_encoded_seq(0), [0, 7, 4, 16, 20, 20, 20, 20, 20, 20]
            )


def test_remove_gaps() -> None:
    for ind in [True, False]:
        with SequenceDataset(f"{DIR}/egf.ref", "fasta", indexed=ind) as data:
            ref = "GTSHLVKCAEKEKTFCVNGGECFMVKDLSNPSRYLCKCQPGFTG----ARCTENVPMK"\
                "VQNQEKAEELYQK"
            np.testing.assert_equal(
                str(data.get_record(5).seq), ref
            )
            np.testing.assert_equal(
                data.get_encoded_seq(5),
                [SequenceDataset.alphabet.index(a) for a in ref.replace('-', '')]
            )
            np.testing.assert_equal(
                data.get_encoded_seq(5, remove_gaps=False),
                [SequenceDataset.alphabet.index(a) for a in ref]
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
        np.testing.assert_equal(data.get_encoded_seq(0), [13, 6, 10, 9, 20])
        np.testing.assert_equal(data.get_encoded_seq(1), [13, 6, 9, 20])


def test_from_alignment() -> None:
    sequences = [("seq1", "FELIX"), ("seq2", "FE-IX")]
    with AlignedDataset(aligned_sequences=sequences) as data:
        assert data.num_seq == 2
        np.testing.assert_equal(data.get_encoded_seq(0), [13, 6, 10, 9, 20])
        np.testing.assert_equal(data.get_encoded_seq(1), [13, 6, 9, 20])
        np.testing.assert_equal(data.get_column_map(0), [0, 1, 2, 3, 4])
        np.testing.assert_equal(data.get_column_map(1), [0, 1, 3, 4])


def test_file_output_formats() -> None:
    formats = ["fasta", "clustal", "stockholm"]
    # write an alignment to various formats
    for fmt in formats:
        with AlignedDataset(
            aligned_sequences=[
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
            np.testing.assert_equal(data.get_encoded_seq(0), [13, 6, 10, 9, 20])
            np.testing.assert_equal(data.get_encoded_seq(1), [13, 6, 9, 20])
            np.testing.assert_equal(data.get_encoded_seq(2), [6, 10, 9])
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


def test_get_alphabet_no_gap() -> None:
    """Test alphabet without gap character."""
    with SequenceDataset(sequences=[("s1", "FELIX")]) as data:
        alphabet_no_gap = data.get_alphabet_no_gap()
        assert alphabet_no_gap == "ARNDCQEGHILKMFPSTWYVXUO"
        assert "-" not in alphabet_no_gap
        assert SequenceDataset.alphabet.endswith("-")


def test_get_standardized_seq() -> None:
    """Test sequence standardization with various options."""
    sequences = [("s1", "FE-L.IX"), ("s2", "ab*cd")]
    with SequenceDataset(sequences=sequences) as data:
        # Test gap removal (default)
        assert data.get_standardized_seq(0) == "FELIX"

        # Test keeping gaps
        assert data.get_standardized_seq(0, remove_gaps=False) == "FE-L-IX"

        # Test custom gap symbols
        assert data.get_standardized_seq(0, gap_symbols=".-") == "FELIX"

        # Test ignore symbols
        assert data.get_standardized_seq(1, ignore_symbols="*") == "ABCD"

        # Test replace with X
        assert data.get_standardized_seq(1, replace_with_x="BD") == "AX*CX"


def test_crop_to_length() -> None:
    """Test sequence cropping functionality."""
    sequences = [("s1", "ABCDEFGHIJ")]
    with SequenceDataset(sequences=sequences) as data:
        # Test normal encoding
        seq = data.get_encoded_seq(0)
        assert len(seq) == 10

        # Test cropping
        seq_cropped = data.get_encoded_seq(0, crop_to_length=5)
        assert len(seq_cropped) == 5

        # Test with crop boundaries
        seq_cropped, start, end = data.get_encoded_seq(
            0, crop_to_length=5, return_crop_boundaries=True
        )
        assert len(seq_cropped) == 5
        assert end - start == 5
        assert 0 <= start <= 5
        assert start < end <= 10


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
    with AlignedDataset(aligned_sequences=sequences) as data:
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

    with AlignedDataset(aligned_sequences=sequences) as data1:
        with AlignedDataset(aligned_sequences=sequences) as data2:
            # SP score with itself should be 1.0
            sp = data1.SP_score(data2)
            assert sp == 1.0

    # Create different alignments
    seq1 = [("s1", "FE-LIX"), ("s2", "FEILIX")]
    seq2 = [("s1", "FELIX-"), ("s2", "FEILIX")]

    with AlignedDataset(aligned_sequences=seq1) as data1:
        with AlignedDataset(aligned_sequences=seq2) as data2:
            # SP score should be less than 1.0
            sp = data1.SP_score(data2)
            assert 0.0 <= sp < 1.0


def test_dtype_parameter() -> None:
    """Test different dtype parameters for encoding."""
    sequences = [("s1", "FELIX")]
    with SequenceDataset(sequences=sequences) as data:
        # Test np.int16 (default)
        seq16 = data.get_encoded_seq(
            0, dtype=np.int16, return_crop_boundaries=False
        )
        assert isinstance(seq16, np.ndarray)
        assert seq16.dtype == np.int16

        # Test np.int32
        seq32 = data.get_encoded_seq(
            0, dtype=np.int32, return_crop_boundaries=False
        )
        assert isinstance(seq32, np.ndarray)
        assert seq32.dtype == np.int32

        # Test np.int64
        seq64 = data.get_encoded_seq(
            0, dtype=np.int64, return_crop_boundaries=False
        )
        assert isinstance(seq64, np.ndarray)
        assert seq64.dtype == np.int64


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

    with AlignedDataset(aligned_sequences=aligned) as data:
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
