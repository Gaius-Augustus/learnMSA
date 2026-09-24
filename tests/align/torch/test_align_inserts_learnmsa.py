"""learnMSA's own insertion aligner (``insertion_aligner="learnmsa"``).

The slices of long insertions are aligned with one pHMM head each, trained and
decoded jointly by ``align_batch``. The output invariants are the same as for
FAMSA: the MSA is rectangular and every row ungaps to its input sequence.
"""

import numpy as np
import pytest

import learnMSA.align.align as align_module
import learnMSA.align.align_inserts as align_inserts
from learnMSA import Configuration
from learnMSA.align.align_inserts import (_resolve_aligner,
                                          make_aligned_insertions,
                                          make_aligned_insertions_multi)
from learnMSA.align.alignment_model import AlignmentModel
from tests.align.test_output_roundtrip import (_model_with_metadata, _read,
                                               _synthetic)

MODE = AlignmentModel.DecodingMode.VITERBI


@pytest.fixture
def config(tmp_path) -> Configuration:
    config = Configuration()
    config.input_output.work_dir = str(tmp_path / "work")
    return config


@pytest.fixture
def align_batch_calls(monkeypatch) -> list:
    """Shortens the inner training and records every inner run."""
    calls = []
    make_config = align_inserts._insertion_config

    def short_config(config, lengths):
        inner = make_config(config, lengths)
        inner.training.epochs = [1, 1, 1]
        return inner

    run = align_module.align_batch

    def recording_align_batch(data, config):
        calls.append(data.num_datasets)
        return run(data, config)

    monkeypatch.setattr(align_inserts, "_insertion_config", short_config)
    monkeypatch.setattr(align_module, "align_batch", recording_align_batch)
    return calls


def _assert_round_trip(tmp_path, am, aligned_insertions, seqs, fmt) -> None:
    out = am.to_file(
        tmp_path / f"msa_{id(am)}.{fmt}", 0,
        aligned_insertions=aligned_insertions, format=fmt, decoding_mode=MODE,
    )
    rows = _read(out)
    assert len(rows) == len(seqs)
    assert len({len(r) for _, r in rows}) == 1, "ragged alignment"
    for (header, row), original in zip(rows, seqs):
        ungapped = row.replace("-", "").replace(".", "").upper()
        assert ungapped == original, f"{header} does not round-trip"


@pytest.mark.parametrize("fmt", ["fasta", "a2m"])
def test_learnmsa_insertions_round_trip(
    tmp_path, config, align_batch_calls, fmt
) -> None:
    data, meta, seqs = _synthetic(n=40, num_match=5)
    am = _model_with_metadata(data, meta)

    ai = make_aligned_insertions(
        am, 0, decoding_mode=MODE, method="learnmsa", verbose=False,
        config=config,
    )

    assert np.sum(ai.ext_insertions) > 0
    assert align_batch_calls and sum(align_batch_calls) > 1
    _assert_round_trip(tmp_path, am, ai, seqs, fmt)


def test_learnmsa_insertions_in_several_chunks(
    tmp_path, config, align_batch_calls
) -> None:
    data, meta, seqs = _synthetic(n=40, num_match=5)
    am = _model_with_metadata(data, meta)
    _, _, slices = align_inserts._collect_slices(am, 0, MODE)
    max_heads = (len(slices) + 1) // 2
    config.advanced.insertion_max_heads = max_heads

    ai = make_aligned_insertions(
        am, 0, decoding_mode=MODE, method="learnmsa", verbose=False,
        config=config,
    )

    assert align_batch_calls == [max_heads, len(slices) - max_heads]
    _assert_round_trip(tmp_path, am, ai, seqs, "a2m")


def test_insertions_of_several_alignments_share_a_run(
    tmp_path, config, align_batch_calls
) -> None:
    synthetic = [_synthetic(n=40, num_match=5, seed=s) for s in (3, 4)]
    ams = [_model_with_metadata(data, meta) for data, meta, _ in synthetic]

    ais = make_aligned_insertions_multi(
        [(am, 0) for am in ams], decoding_mode=MODE, method="learnmsa",
        verbose=False, config=config,
    )

    # Both alignments' slices fit into one run of up to 64 heads
    assert len(align_batch_calls) == 1
    for am, ai, (_, _, seqs) in zip(ams, ais, synthetic):
        _assert_round_trip(tmp_path, am, ai, seqs, "a2m")


def test_auto_selects_learnmsa_under_pytorch() -> None:
    assert _resolve_aligner("auto") == "learnmsa"
