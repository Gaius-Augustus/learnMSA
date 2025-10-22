import subprocess

from learnMSA.msa_hmm.SequenceDataset import SequenceDataset


def test_error_handling() -> None:
    single_seq = "test/data/single_sequence.fasta"
    faulty_format = "test/data/faulty_format.fasta"
    empty_seq = "test/data/empty_sequence.fasta"
    unknown_symbol = "test/data/unknown_symbol.fasta"

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
