from pathlib import Path

import numpy as np
import pytest

from learnMSA.util.fasta_stats import read_fasta_stats
from learnMSA.util.sequence_dataset import SequenceDataset


DIR = "tests/data/"


def _naive_lengths(text: str) -> list[int]:
    lengths = []
    for line in text.splitlines():
        if line.startswith(">"):
            lengths.append(0)
        elif lengths:
            lengths[-1] += sum(
                1 for c in line if not c.isspace() and c not in "-."
            )
    return lengths


FASTA_TEXTS = {
    "multiline": ">a desc\nACDE\nFGH\n>b\nKL\n>c\nMNPQRSTVWY\nAC\nD\n",
    "crlf": ">a\r\nACDE\r\nFG\r\n>b\r\nK\r\n",
    "gaps": ">a\n--AC.DE--\n>b\n..kl-m\n",
    "empty_seq": ">a\n>b\nACD\n>c\n",
    "no_final_newline": ">a\nACD\n>b\nACDEFG",
    "leading_blank_lines": "\n\n>a\nAC\n\n>b\nACD\n\n",
    "even_count_median": ">a\nA\n>b\nACDEFGH\n>c\nACD\n>d\nACDEFGHIKLMN\n",
}


@pytest.mark.parametrize("name", FASTA_TEXTS)
@pytest.mark.parametrize("chunk_size", [1, 3, 7, 1024])
def test_stats_match_naive_count(
    tmp_path: Path, name: str, chunk_size: int
) -> None:
    text = FASTA_TEXTS[name]
    path = tmp_path / f"{name}.fasta"
    path.write_bytes(text.encode())
    lengths = _naive_lengths(text)

    stats = read_fasta_stats(path, chunk_size=chunk_size)

    assert stats.num_seqs == len(lengths)
    assert stats.max_len == max(lengths)
    assert stats.total_residues == sum(lengths)
    assert stats.avg_len == pytest.approx(sum(lengths) / len(lengths))
    assert stats.median_len == np.median(lengths)


@pytest.mark.parametrize("chunk_size", [100, 16 * 1024 * 1024])
def test_stats_match_sequence_dataset(chunk_size: int) -> None:
    stats = read_fasta_stats(f"{DIR}/egf.fasta", chunk_size=chunk_size)
    with SequenceDataset(f"{DIR}/egf.fasta", "fasta") as data:
        assert stats.num_seqs == data.num_seq
        assert stats.max_len == int(data.seq_lens.max())
        assert stats.total_residues == int(data.seq_lens.sum())
        assert stats.median_len == np.median(data.seq_lens)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "\n\n",
        "# STOCKHOLM 1.0\n\nseq1 AC-D\nseq2 ACED\n//\n",
        "CLUSTAL W (1.83)\n\nseq1 AC-D\nseq2 ACED\n",
        "ACDE\n>a\nACDE\n",
    ],
)
def test_non_fasta_raises(tmp_path: Path, text: str) -> None:
    path = tmp_path / "input.txt"
    path.write_bytes(text.encode())
    with pytest.raises(ValueError):
        read_fasta_stats(path, chunk_size=4)
