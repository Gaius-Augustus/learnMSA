"""Several input files on the command line (-i a.fasta b.fasta ...)."""

from pathlib import Path

import pytest

from learnMSA.run.args import parse_args
from learnMSA.run.args_to_config import args_to_config
from learnMSA.run.util import resolve_multiple_inputs


def _config(argv: list[str]):
    parser = parse_args("test_version")
    return args_to_config(parser.parse_args(argv)), parser


def test_single_input_file_is_unchanged() -> None:
    config, parser = _config(["-i", "a.fasta", "-o", "a.a2m"])
    resolve_multiple_inputs(config, parser)

    assert config.input_output.input_file == Path("a.fasta")
    assert config.input_output.output_file == Path("a.a2m")


def test_one_output_file_per_input_file() -> None:
    config, parser = _config(
        ["-i", "a.fasta", "b.fasta", "-o", "x.a2m", "y.a2m"]
    )
    resolve_multiple_inputs(config, parser)

    io = config.input_output
    assert io.input_file == [Path("a.fasta"), Path("b.fasta")]
    assert io.output_file == [Path("x.a2m"), Path("y.a2m")]


def test_output_directory(tmp_path: Path) -> None:
    config, parser = _config([
        "-i", "fam/a.fasta", "fam/b.fa", "-o", str(tmp_path / "out"),
        "-f", "fasta",
    ])
    resolve_multiple_inputs(config, parser)

    assert config.input_output.output_file == [
        tmp_path / "out" / "a.fasta", tmp_path / "out" / "b.fasta"
    ]


@pytest.mark.parametrize("argv", [
    # a single input file with several outputs
    ["-i", "a.fasta", "-o", "x.a2m", "y.a2m"],
    # neither one output per input nor a directory
    ["-i", "a.fasta", "b.fasta", "c.fasta", "-o", "x.a2m", "y.a2m"],
    # identical names can not share an output directory
    ["-i", "1/a.fasta", "2/a.fasta", "-o", "out"],
    # options that refer to a single dataset
    ["-i", "a.fasta", "b.fasta", "-o", "out", "--scores", "s.tsv"],
    ["-i", "a.fasta", "b.fasta", "-o", "out", "--struct", "3di.fasta"],
    ["-i", "a.fasta", "b.fasta", "-o", "out", "--save_model"],
])
def test_invalid_combinations(argv: list[str]) -> None:
    config, parser = _config(argv)
    with pytest.raises(SystemExit):
        resolve_multiple_inputs(config, parser)


def test_lists_of_one_file_are_single_files() -> None:
    """A config file can give a single input as a list."""
    config, parser = _config(["-i", "a.fasta", "-o", "a.a2m"])
    config.input_output.input_file = [Path("a.fasta")]
    config.input_output.output_file = [Path("a.a2m")]
    resolve_multiple_inputs(config, parser)

    assert config.input_output.input_file == Path("a.fasta")
    assert config.input_output.output_file == Path("a.a2m")


def test_config_round_trip_keeps_file_lists() -> None:
    from learnMSA import Configuration

    config, _ = _config(["-i", "a.fasta", "b.fasta", "-o", "x.a2m", "y.a2m"])
    restored = Configuration.model_validate(config.model_dump(mode="json"))
    assert restored.input_output.input_file == [
        Path("a.fasta"), Path("b.fasta")
    ]


def test_output_directory_must_not_be_a_file(tmp_path: Path) -> None:
    existing = tmp_path / "out.a2m"
    existing.write_text("")
    config, parser = _config(
        ["-i", "a.fasta", "b.fasta", "-o", str(existing)]
    )
    with pytest.raises(SystemExit):
        resolve_multiple_inputs(config, parser)
