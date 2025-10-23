import os
import warnings

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
                data.starting_pos, [0, 5, 13]
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

