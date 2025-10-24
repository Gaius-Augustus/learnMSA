import subprocess

from learnMSA.msa_hmm.SequenceDataset import SequenceDataset


def test_error_handling() -> None:
    single_seq = "tests/data/single_sequence.fasta"
    faulty_format = "tests/data/faulty_format.fasta"
    empty_seq = "tests/data/empty_sequence.fasta"
    unknown_symbol = "tests/data/unknown_symbol.fasta"

    single_seq_expected_err = f"File {single_seq} contains only a single sequence."
    faulty_format_expected_err = f"Could not parse any sequences from {faulty_format}."
    empty_seq_expected_err = f"{empty_seq} contains empty sequences."
    unknown_symbol_expected_err = f"Found unknown character(s) in sequence ersteSequenz. Allowed alphabet: {SequenceDataset.alphabet}."

    test = subprocess.Popen(
        [
            "python", "learnMSA.py", "--no_sequence_weights", "--silent",
            "-o", "test.out", "-i", single_seq
        ],
        stderr=subprocess.PIPE
    )
    output = test.communicate()[1].strip().decode('ascii')
    assert single_seq_expected_err == output[-len(single_seq_expected_err):]

    test = subprocess.Popen(
        [
            "python", "learnMSA.py", "--no_sequence_weights", "--silent",
            "-o", "test.out", "-i", faulty_format
        ],
        stderr=subprocess.PIPE
    )
    output = test.communicate()[1].strip().decode('ascii')
    assert faulty_format_expected_err == output[-len(faulty_format_expected_err):]

    test = subprocess.Popen(
        [
            "python", "learnMSA.py", "--no_sequence_weights", "--silent",
            "-o", "test.out", "-i", empty_seq
        ],
        stderr=subprocess.PIPE
    )
    output = test.communicate()[1].strip().decode('ascii')
    assert empty_seq_expected_err == output[-len(empty_seq_expected_err):]

    test = subprocess.Popen(
        [
            "python", "learnMSA.py", "--no_sequence_weights", "--silent",
            "-o", "test.out", "-i", unknown_symbol
        ],
        stderr=subprocess.PIPE
    )
    output = test.communicate()[1].strip().decode('ascii')
    assert unknown_symbol_expected_err in output

def test_file_conversion_and_input_format() -> None:
    input_fasta = "tests/data/egf.ref"
    output_clustal = "tests/data/egf.out.clustal"

    # fasta -> clustal
    test = subprocess.Popen(
        [
            "python", "learnMSA.py", "--convert", "--silent",
            "-i", input_fasta,
            "-o", output_clustal,
            "--format", "clustal"
        ],
        stderr=subprocess.PIPE
    )
    test.communicate()
    with SequenceDataset(input_fasta, "fasta") as data:
        original_seq_count = len(data)
    with SequenceDataset(output_clustal, "clustal") as data:
        assert len(data) == original_seq_count

    # clustal -> fasta
    output_fasta = "tests/data/egf.out.fasta"
    test = subprocess.Popen(
        [
            "python", "learnMSA.py", "--convert", "--silent",
            "-i", output_clustal,
            "-o", output_fasta,
            "--input_format", "clustal",
            "--format", "fasta"
        ],
        stderr=subprocess.PIPE
    )
    test.communicate()
    with SequenceDataset(output_fasta, "fasta") as data:
        assert len(data) == original_seq_count

    # cleanup
    subprocess.Popen(
        ["rm", output_clustal, output_fasta],stderr=subprocess.PIPE
    ).communicate()
