# SPDX-License-Identifier: GPL-3.0-or-later
"""learnMSA-3di-translate: structures to 3Di logits (numpy and gemmi)."""

import struct
import urllib.error
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gemmi")

import learnMSA.structure.translate_3di as t3di
from learnMSA.config.structure import StructureConfig
from learnMSA.structure.io import KIND, SOURCE_KEY, is_logits_file
from learnMSA.structure.io import PDB as PDB_SOURCE
from learnMSA.util import EmbeddingDataset

PDB = Path(__file__).parents[1] / "data" / "8002621_1y0uA.pdb"
#: foldseek structureto3didescriptor output for PDB.
PDB_AA = "NPVRRKILRMLDKGRSEEEIMQTLSLSKKQLDYHLKVLEAGFCIERVGERWVVTDAGKIV"
PDB_3DI = "DVLLLVVLVCLQVVHDPVVSCVVSVDDPVRVVVSVVVCVVVVQWDDDDNGIHGDPVNDDD"


def _random_encoder(seed: int = 0) -> t3di.Encoder3Di:
    rng = np.random.default_rng(seed)
    layers = [
        (rng.normal(size=(10, 10)).astype(np.float32),
         rng.normal(size=10).astype(np.float32), True),
        (rng.normal(size=(10, 2)).astype(np.float32),
         rng.normal(size=2).astype(np.float32), False),
    ]
    return t3di.Encoder3Di(layers, rng.normal(size=(20, 2)))


def _random_chain(length: int, seed: int) -> t3di.Chain:
    """A self-avoiding-ish random backbone with CA spacing 3.8."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(size=(length, 3))
    steps = 3.8 * steps / np.linalg.norm(steps, axis=1, keepdims=True)
    ca = np.cumsum(steps, axis=0)
    n = ca + rng.normal(scale=0.8, size=ca.shape)
    c = ca + rng.normal(scale=0.8, size=ca.shape)
    cb = ca + rng.normal(scale=0.8, size=ca.shape)
    cb[rng.random(length) < 0.3] = np.nan
    return t3di.Chain("x", "A" * length, n, ca, c, cb)


def _reference_mask(chain: t3di.Chain) -> tuple[np.ndarray, np.ndarray]:
    """Literal loops of foldseek's structure2states (before encoding)."""
    length = len(chain)
    cb = chain.cb.copy()
    with np.errstate(invalid="ignore", divide="ignore"):
        for i in range(length):
            if np.isnan(cb[i, 0]):
                cb[i] = t3di.approx_cb(chain.ca[i], chain.n[i], chain.c[i])
            cb[i] = t3di.virtual_center(chain.ca[i], cb[i], chain.n[i])
    valid = [not (np.isnan(chain.ca[i, 0]) or np.isnan(chain.c[i, 0])
                  or np.isnan(chain.n[i, 0])) for i in range(length)]
    partner = [-1] * length
    for i in range(1, length - 1):
        best = np.inf
        for j in range(1, length - 1):
            if i != j and valid[j]:
                d = np.sqrt(((cb[i] - cb[j]) ** 2).sum())
                if d < best:
                    best, partner[i] = d, j
        if partner[i] == -1:
            valid[i] = False
    copy = list(valid)
    mask = list(valid)
    for i in range(1, length - 1):
        j = partner[i]
        if not (copy[i - 1] and copy[i] and copy[i + 1]
                and copy[j - 1] and copy[j] and copy[j + 1]):
            mask[i] = False
    mask[0] = mask[-1] = False
    return np.array(partner), np.array(mask)


def test_class_order_matches_structural_alphabet() -> None:
    assert t3di.CLASS_ORDER == StructureConfig().structural_alphabet


def test_parse_kerasify() -> None:
    w1 = np.arange(6, dtype=np.float32).reshape(2, 3)
    b1 = np.array([1, 2, 3], dtype=np.float32)
    blob = struct.pack("<II", 1, 1) + struct.pack("<III", 2, 3, 3) \
        + w1.tobytes() + b1.tobytes() + struct.pack("<I", 2)
    (weights, bias, relu), = t3di.parse_kerasify(blob)
    np.testing.assert_array_equal(weights, w1)
    np.testing.assert_array_equal(bias, b1)
    assert relu
    with pytest.raises(ValueError):
        t3di.parse_kerasify(blob + b"\0")


def test_parse_centroids() -> None:
    header = """const double centroids[CENTROID_CNT][EMBEDDING_DIM] = {
        { -1.0729,  -0.3600},
        {  0.4948,  2.0},
    };"""
    np.testing.assert_array_equal(t3di.parse_centroids(header),
                                  [[-1.0729, -0.36], [0.4948, 2.0]])


def test_embed_matches_matrix_product() -> None:
    encoder = _random_encoder()
    x = np.random.default_rng(1).normal(size=(7, 10))
    h = np.maximum(x @ encoder.layers[0][0] + encoder.layers[0][1], 0)
    expected = h @ encoder.layers[1][0] + encoder.layers[1][1]
    np.testing.assert_allclose(encoder.embed(x), expected, rtol=1e-4,
                               atol=1e-4)


def test_approx_cb_is_tetrahedral() -> None:
    ca = np.zeros(3)
    n = np.array([1.46, 0.0, 0.0])
    angle = np.arccos(-1 / 3)
    c = 1.52 * np.array([np.cos(angle), np.sin(angle), 0.0])
    cb = t3di.approx_cb(ca, n, c)
    assert np.linalg.norm(cb) == pytest.approx(t3di.DISTANCE_ALPHA_BETA)
    for other in (n, c):
        cos = cb @ other / np.linalg.norm(cb) / np.linalg.norm(other)
        assert cos == pytest.approx(-1 / 3)


@pytest.mark.parametrize("seed", range(5))
def test_partners_and_mask_match_foldseek_loops(seed: int) -> None:
    chain = _random_chain(40, seed)
    rng = np.random.default_rng(seed)
    for atoms in (chain.n, chain.c):
        atoms[rng.choice(40, 3, replace=False)] = np.nan
    _, ref_mask = _reference_mask(chain)
    d2, mask = t3di.translate(chain, _random_encoder())
    np.testing.assert_array_equal(mask, ref_mask)
    assert not mask[0] and not mask[-1]
    assert (d2[~mask] == 0).all()


def test_logits_argmax_and_uninformative_rows() -> None:
    d2 = np.random.default_rng(0).random((6, 20))
    mask = np.array([False, True, True, False, True, True])
    z = t3di.to_logits(d2, mask)
    np.testing.assert_allclose(z[mask], -d2[mask])
    # No 3Di: d = 0 for every letter, which is uninformative.
    np.testing.assert_array_equal(z[~mask], 0)
    assert t3di.states(d2, mask)[0] == "D"


def test_read_chains_names_and_filters(tmp_path) -> None:
    lines = [ln for ln in PDB.read_text().splitlines()
             if ln.startswith("ATOM")]
    chain_b = [ln[:21] + "B" + ln[22:] for ln in lines]
    # Three residues of chain C: too short, skipped by foldseek.
    short = [ln[:21] + "C" + ln[22:] for ln in lines
             if int(ln[22:26]) < int(lines[0][22:26]) + 3]
    water = ["HETATM 9999  O   HOH D   1       0.000   0.000   0.000"
             "  1.00  0.00           O"]
    path = tmp_path / "two.pdb"
    path.write_text("\n".join(lines + ["TER"] + chain_b + ["TER"] + short
                              + ["TER"] + water + ["END"]) + "\n")
    chains = t3di.read_chains(path)
    assert [c.name for c in chains] == ["two_A", "two_B"]
    assert all(c.aa == PDB_AA for c in chains)
    single = t3di.read_chains(PDB)
    assert [c.name for c in single] == ["8002621_1y0uA"]


def test_cli_writes_logits(tmp_path, monkeypatch) -> None:
    encoder = _random_encoder()
    monkeypatch.setattr(t3di.Encoder3Di, "from_foldseek",
                        classmethod(lambda cls, cache_dir=None: encoder))
    lines = [ln for ln in PDB.read_text().splitlines()
             if ln.startswith("ATOM")]
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "two.pdb").write_text("\n".join(
        lines + ["TER"] + [ln[:21] + "B" + ln[22:] for ln in lines]) + "\n")
    out = tmp_path / "x.npz"
    t3di.main(["-i", str(tmp_path / "in"), str(PDB), "-o", str(out),
               "--fasta", str(tmp_path / "x.3di.fa"),
               "--aa_fasta", str(tmp_path / "x.fa"), "--silent"])
    assert is_logits_file(out)
    data = EmbeddingDataset(out)
    assert list(data.seq_ids) == ["two_A", "two_B", "8002621_1y0uA"]
    assert str(data.metadata["kind"]) == KIND
    assert str(data.metadata["alphabet"]) == t3di.CLASS_ORDER
    assert str(data.metadata[SOURCE_KEY]) == PDB_SOURCE
    assert "temperature" not in data.metadata
    aa = (tmp_path / "x.fa").read_text().split()[1::2]
    letters = (tmp_path / "x.3di.fa").read_text().split()[1::2]
    assert aa == [PDB_AA] * 3
    for i, seq in enumerate(letters):
        z = data.get_encoded_seq(i)
        assert len(z) == len(seq) == len(PDB_AA)
        masked = (z == 0).all(1)
        assert masked[0] and masked[-1] and not masked.all()
        # The letter of the closest centroid, i.e. argmax of -d^2.
        argmax = np.array(list(t3di.CLASS_ORDER))[z.argmax(1)]
        np.testing.assert_array_equal(argmax[~masked],
                                      np.array(list(seq))[~masked])


def test_parity_with_foldseek(tmp_path) -> None:
    try:
        encoder = t3di.Encoder3Di.from_foldseek(tmp_path / "cache")
    except (urllib.error.URLError, OSError) as e:
        pytest.skip(f"foldseek weights not available: {e}")
    chain, = t3di.read_chains(PDB)
    assert chain.aa == PDB_AA
    d2, mask = t3di.translate(chain, encoder)
    assert t3di.states(d2, mask) == PDB_3DI
