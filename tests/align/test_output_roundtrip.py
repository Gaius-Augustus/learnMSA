"""End-to-end invariants of the alignment output stage.

Builds a small, fully self-consistent :class:`AlignmentMetaData` over a
synthetic dataset -- multiple repeats, deletions, unannotated segments, short
and long insertions at many positions -- then aligns the insertions and writes
the MSA. Ungapping every row must give back the input sequence, which is the
property the memory rewrite of the output stage has to preserve.
"""

import numpy as np
import pytest

from learnMSA.align.align_inserts import make_aligned_insertions
from learnMSA.align.alignment_metadata import AlignmentMetaData
from learnMSA.align.alignment_model import AlignmentModel
from learnMSA.util.sequence_dataset import SequenceDataset

pytest.importorskip("pyfamsa")

AA = "ARNDCQEGHILKMFPSTWYV"


def _synthetic(n=120, num_match=8, seed=3):
    """Return (dataset, metadata) with matching residue positions."""
    rng = np.random.default_rng(seed)
    seqs = []
    nrpr, hit, loc, ilen, istart = [], [], [], [], []
    lfl, lfs, rfl, rfs, unl, uns = [], [], [], [], [], []

    for i in range(n):
        buf = []

        def emit(k):
            s = len(buf)
            buf.extend(rng.choice(list(AA), size=k))
            return s

        lf = int(rng.choice([0, 2, 24]))
        lfs.append(emit(lf))
        lfl.append(lf)
        n_rep = 1 if rng.random() < 0.6 else 2
        nrpr.append(n_rep)
        for r in range(n_rep):
            start = len(buf)
            hits, il, ist = [], [], []
            for j in range(num_match):
                hits.append(-1 if rng.random() < 0.15 else emit(1))
                if j < num_match - 1:
                    u = rng.random()
                    k = 0 if u < 0.5 else (
                        int(rng.integers(1, 20)) if u < 0.7
                        else int(rng.integers(20, 45))
                    )
                    ist.append(emit(k) if k else -1)
                    il.append(k)
            hit.append(hits)
            ilen.append(il)
            istart.append(ist)
            loc.append([start, len(buf)])
            if r < n_rep - 1:
                k = int(rng.choice([0, 3, 28]))
                uns.append(emit(k) if k else -1)
                unl.append(k)
        rf = int(rng.choice([0, 2, 26]))
        rfs.append(emit(rf))
        rfl.append(rf)
        seqs.append("".join(buf))

    i16 = lambda x: np.asarray(x, dtype=np.int16)
    meta = AlignmentMetaData(
        num_rows=n, num_match=num_match,
        num_repeats_per_row=np.asarray(nrpr, np.int32),
        domain_hit=i16(hit), domain_loc=i16(loc),
        insertion_lens=i16(ilen), insertion_start=i16(istart),
        left_flank_len=i16(lfl), left_flank_start=i16(lfs),
        right_flank_len=i16(rfl), right_flank_start=i16(rfs),
        unannotated_segments_len=i16(unl),
        unannotated_segments_start=i16(uns),
    )
    data = SequenceDataset(
        sequences=[(f"seq_{i}", s) for i, s in enumerate(seqs)]
    )
    return data, meta, seqs


def _model_with_metadata(data, meta):
    am = AlignmentModel(data, None, best_head=0)  # model unused
    am.metadata[0] = meta
    return am


@pytest.mark.parametrize("fmt", ["fasta", "a2m"])
def test_aligned_insertions_roundtrip(tmp_path, fmt) -> None:
    data, meta, seqs = _synthetic()
    am = _model_with_metadata(data, meta)
    mode = AlignmentModel.DecodingMode.VITERBI
    ai = make_aligned_insertions(am, 0, decoding_mode=mode, verbose=False,
                                 threads=1)
    # The synthetic data has long insertions at many positions.
    assert np.sum(ai.ext_insertions) > 0

    out = tmp_path / f"msa.{fmt}"
    am.to_file(out, 0, aligned_insertions=ai, format=fmt, decoding_mode=mode)

    rows = _read(out)
    assert len(rows) == len(seqs)
    widths = {len(r) for _, r in rows}
    assert len(widths) == 1, f"ragged alignment: {sorted(widths)}"
    for (header, row), original in zip(rows, seqs):
        ungapped = row.replace("-", "").replace(".", "").upper()
        assert ungapped == original, f"{header} does not round-trip"


def test_batch_size_does_not_change_the_output(tmp_path) -> None:
    """Streaming in smaller batches must produce the identical file."""
    data, meta, _ = _synthetic()
    am = _model_with_metadata(data, meta)
    mode = AlignmentModel.DecodingMode.VITERBI
    ai = make_aligned_insertions(am, 0, decoding_mode=mode, verbose=False,
                                 threads=1)
    a, b = tmp_path / "a.a2m", tmp_path / "b.a2m"
    am.to_file(a, 0, aligned_insertions=ai, format="a2m", decoding_mode=mode)
    am.to_file(b, 0, aligned_insertions=ai, format="a2m", decoding_mode=mode,
               batch_size=7)
    assert a.read_bytes() == b.read_bytes()


def _read(path):
    rows, header, parts = [], None, []
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith(">"):
            if header is not None:
                rows.append((header, "".join(parts)))
            header, parts = line[1:], []
        else:
            parts.append(line)
    if header is not None:
        rows.append((header, "".join(parts)))
    return rows
