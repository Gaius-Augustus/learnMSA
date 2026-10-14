# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is a Python port of parts of Foldseek
# (https://github.com/steineggerlab/foldseek, GPL-3.0): the structure to 3Di
# encoder of lib/3di/structureto3di.cpp and the residue parsing and entry
# naming of src/strucclustutils/GemmiWrapper.cpp and structcreatedb.cpp.
# Unlike the rest of learnMSA (MIT), this file is licensed under the GNU
# General Public License v3.0 or later; see LICENSES/GPL-3.0-or-later.txt.
# No MIT-licensed module of learnMSA imports it.
#
# van Kempen M, Kim SS, Tumescheit C, Mirdita M, Lee J, Gilchrist CLM,
# Söding J, Steinegger M. Fast and accurate protein structure search with
# Foldseek. Nature Biotechnology 42, 243-246 (2024).
"""``learnMSA-3di-translate``: per-residue 3Di logits from structures.

Reads PDB/mmCIF files and computes 3Di like ``foldseek createdb``: per
residue, 10 descriptors of the backbone and its nearest partner residue
are mapped by the VQ-VAE encoder to a point ``e`` in 2D. Foldseek assigns
the letter of the nearest of 20 centroids ``c_k``. Here the encoder output
is kept and written as logits ``z_k = -|e - c_k|^2`` instead, to the same
``.npz`` format that ``learnMSA-3di`` writes for ProstT5, marked with
``logit_source = "pdb"``. learnMSA reads them as the posterior
``softmax(-d^2 / T + log(pi))`` over the 3Di letters (``pi``: the 3Di
background, ``T``: ``StructureConfig.input_temperature``); the argmax is
foldseek's letter.

Residues without a 3Di (chain ends, missing backbone atoms, invalid
neighbours or partner) get ``d = 0`` for every letter, which makes them
uninformative; ``--fasta`` writes ``D`` there, as foldseek.

The encoder weights and centroids are downloaded from foldseek at a pinned
commit. Requires ``gemmi``.
"""

import argparse
import hashlib
import os
import re
import struct
import sys
import time
import urllib.request
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Foldseek commit the encoder files and the ported code are taken from.
FOLDSEEK_COMMIT = "463739e0014a1549a527de589102cde98f802f37"
FOLDSEEK_RAW = ("https://raw.githubusercontent.com/steineggerlab/foldseek/"
                + FOLDSEEK_COMMIT)
#: Encoder weights (kerasify) and the header that defines the centroids.
FOLDSEEK_FILES = {
    "encoder_weights_3di.kerasify": (
        "data/encoder_weights_3di.kerasify",
        "9b8245e2855605b8aeeabf0333d38d43906671bab0af977129fa984e13a61e99",
    ),
    "structureto3di.h": (
        "lib/3di/structureto3di.h",
        "0c2b80e1ca8ef176e4f1c11654429dce3d53e8bc7210450abd8a2154f06bb353",
    ),
}

#: 3Di letter of state i (``mat3di.out``); equals the structural alphabet.
CLASS_ORDER = "ACDEFGHIKLMNPQRSTVWY"
#: Foldseek assigns invalid residues to the coil state ``D``.
INVALID_STATE = 2
DISTANCE_ALPHA_BETA = 1.5336
#: Foldseek's value of pi, used for the virtual center angles.
FOLDSEEK_PI = 3.14159265359
#: (alpha, beta, d) of the virtual center.
VIRTUAL_CENTER = (270.0, 0.0, 2.0)

#: Foldseek's ``threeToOneAA`` (adapted from gemmi); other names are X.
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ABA": "A", "ASP": "D", "ASX": "B",
    "CYS": "C", "CSH": "S", "GLN": "Q", "GLU": "E", "GLX": "Z", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "MSE": "M",
    "ORN": "A", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRY": "W",
    "TRP": "W", "TYR": "Y", "UNK": "X", "VAL": "V", "SEC": "C", "PYL": "O",
    "SEP": "S", "TPO": "T", "PCA": "E", "CSO": "C", "PTR": "Y", "KCX": "K",
    "CSD": "C", "LLP": "K", "CME": "C", "MLY": "K", "DAL": "A", "TYS": "Y",
    "OCS": "C", "M3L": "K", "FME": "M", "ALY": "K", "HYP": "P", "CAS": "C",
    "CRO": "T", "CSX": "C", "DPR": "P", "DGL": "E", "DVA": "V", "CSS": "C",
    "DPN": "F", "DSN": "S", "DLE": "L", "HIC": "H", "NLE": "L", "MVA": "V",
    "MLZ": "K", "CR2": "G", "SAR": "G", "DAR": "R", "DLY": "K", "YCM": "C",
    "NRQ": "M", "CGU": "E", "0TD": "D", "MLE": "L", "DAS": "D", "DTR": "W",
    "CXM": "M", "TPQ": "Y", "DCY": "C", "DSG": "N", "DTY": "Y", "DHI": "H",
    "MEN": "N", "DTH": "T", "SAC": "S", "DGN": "Q", "AIB": "A", "SMC": "C",
    "IAS": "D", "CIR": "R", "BMT": "T", "DIL": "I", "FGA": "E", "PHI": "F",
    "CRQ": "Q", "SME": "M", "GHP": "G", "MHO": "M", "NEP": "H", "TRQ": "W",
    "TOX": "W", "ALC": "A", "SCH": "C", "MDO": "A", "MAA": "A", "GYS": "S",
    "MK8": "L", "CR8": "H", "KPI": "K", "SCY": "C", "DHA": "S", "OMY": "Y",
    "CAF": "C", "0AF": "W", "SNN": "N", "MHS": "H", "SNC": "C", "PHD": "D",
    "B3E": "E", "MEA": "F", "MED": "M", "OAS": "S", "GL3": "G", "FVA": "V",
    "PHL": "F", "CRF": "T", "BFD": "D", "MEQ": "Q", "DAB": "A", "AGM": "R",
    "4BF": "Y", "B3A": "A", "B3D": "D", "B3K": "K", "B3Y": "Y", "BAL": "A",
    "DBZ": "A", "GPL": "K", "HSK": "H", "HY3": "P", "HZP": "P", "KYN": "W",
    "MGN": "Q",
}


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

def default_cache_dir() -> Path:
    """Where the foldseek files are cached (``~/.cache/learnmsa/...``)."""
    return Path.home() / ".cache" / "learnmsa" / "foldseek_3di"


def fetch_foldseek_file(name: str, cache_dir: str | Path | None = None
                        ) -> Path:
    """Path of a pinned foldseek file; downloaded and checked if needed."""
    path_in_repo, sha256 = FOLDSEEK_FILES[name]
    target = Path(cache_dir or default_cache_dir()) / FOLDSEEK_COMMIT / name
    if target.is_file() and _sha256(target.read_bytes()) == sha256:
        return target
    url = f"{FOLDSEEK_RAW}/{path_in_repo}"
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    if _sha256(data) != sha256:
        raise RuntimeError(f"Checksum mismatch for {url}.")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    tmp.replace(target)
    return target


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_kerasify(data: bytes) -> list[tuple[np.ndarray, np.ndarray, bool]]:
    """Dense layers ``(W (in, out), b, relu)`` of a kerasify model."""
    pos = 0

    def uint() -> int:
        nonlocal pos
        (value,) = struct.unpack_from("<I", data, pos)
        pos += 4
        return value

    def floats(n: int) -> np.ndarray:
        nonlocal pos
        values = np.frombuffer(data, "<f4", n, pos).astype(np.float32)
        pos += 4 * n
        return values

    layers = []
    for _ in range(uint()):
        if uint() != 1:
            raise ValueError("Only dense layers are supported.")
        rows, cols, n_bias = uint(), uint(), uint()
        weights = floats(rows * cols).reshape(rows, cols)
        bias = floats(n_bias)
        activation = uint()
        if activation not in (1, 2):
            raise ValueError(f"Unsupported activation {activation}.")
        layers.append((weights, bias, activation == 2))
    if pos != len(data):
        raise ValueError("Trailing bytes in kerasify model.")
    return layers


def parse_centroids(header: str) -> np.ndarray:
    """The (20, 2) ``centroids`` array of ``structureto3di.h``."""
    match = re.search(r"centroids\s*\[[^]]*\]\s*\[[^]]*\]\s*=\s*\{(.*?)\};",
                      header, re.S)
    if match is None:
        raise ValueError("No centroids found.")
    values = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", match.group(1))
    return np.array(values, dtype=np.float64).reshape(-1, 2)


@dataclass
class Encoder3Di:
    """The 3Di VQ-VAE encoder and its centroids."""

    layers: list[tuple[np.ndarray, np.ndarray, bool]]
    centroids: np.ndarray

    @classmethod
    def from_foldseek(cls, cache_dir: str | Path | None = None
                      ) -> "Encoder3Di":
        weights = fetch_foldseek_file("encoder_weights_3di.kerasify",
                                      cache_dir)
        header = fetch_foldseek_file("structureto3di.h", cache_dir)
        return cls(parse_kerasify(weights.read_bytes()),
                   parse_centroids(header.read_text()))

    def embed(self, features: np.ndarray) -> np.ndarray:
        """(N, 10) descriptors -> (N, 2) embeddings, in float32 like
        kerasify (inputs accumulated in order, then the bias)."""
        x = np.asarray(features, dtype=np.float32)
        for weights, bias, relu in self.layers:
            out = np.zeros((x.shape[0], weights.shape[1]), dtype=np.float32)
            for i in range(weights.shape[0]):
                out += x[:, i:i + 1] * weights[i]
            out += bias
            x = np.maximum(out, 0) if relu else out
        return x

    def sq_distances(self, embeddings: np.ndarray) -> np.ndarray:
        """(N, 20) squared distances to the centroids, in float64."""
        diff = embeddings.astype(np.float64)[:, None, :] - self.centroids
        return diff[..., 0] ** 2 + diff[..., 1] ** 2


# ---------------------------------------------------------------------------
# Structure parsing
# ---------------------------------------------------------------------------

@dataclass
class Chain:
    """Backbone of one chain; missing atoms are NaN."""

    name: str
    aa: str
    n: np.ndarray
    ca: np.ndarray
    c: np.ndarray
    cb: np.ndarray

    def __len__(self) -> int:
        return len(self.aa)

    @property
    def ca_only(self) -> bool:
        """Foldseek's test for CA-only chains (which it rebuilds)."""
        return bool(np.isnan(self.n[:4, 0]).all()
                    and np.isnan(self.c[:4, 0]).all())


def _entry_stem(path: Path) -> str:
    """File name without extension; compressed files lose two."""
    name = path.name
    drop = 2 if name.endswith((".gz", ".zstd", ".zst")) else 1
    for _ in range(drop):
        if "." in name:
            name = name.rsplit(".", 1)[0]
    return name


def _coor_format(path: Path):
    import gemmi
    name = path.name.lower()
    if name.endswith(".gz"):
        name = name[:-3]
    if name.endswith((".cif", ".mmcif")):
        return gemmi.CoorFormat.Mmcif
    if name.endswith(".json"):
        return gemmi.CoorFormat.Mmjson
    # Foldseek reads everything else as PDB.
    return gemmi.CoorFormat.Pdb


def read_chains(path: str | Path) -> list[Chain]:
    """All protein chains of a structure file, as ``foldseek createdb``.

    Every model and chain is read (first conformer). A residue is kept if
    it is polymer (or of unknown entity), ATOM/HETATM, and has a CA atom.
    Chains with at most 3 residues or only unknown residues are skipped.
    Names are the file stem, plus ``_MODEL_<n>`` if there are several
    models, plus ``_<chain>`` if the file has several chains.
    """
    import gemmi

    path = Path(path)
    # Foldseek's gemmi readers do not merge chain parts; the parts count
    # towards the chain suffix rule.
    st = gemmi.read_structure(str(path), merge_chain_parts=False,
                              format=_coor_format(path))
    keep = (gemmi.EntityType.Polymer, gemmi.EntityType.Unknown)
    stem = _entry_stem(path)
    parsed = []
    for model_count, model in enumerate(st, start=1):
        model_num = getattr(model, "num", None)
        if not isinstance(model_num, int):
            model_num = model_count
        for chain in model:
            aa, coords = [], []
            for res in chain.first_conformer():
                # het_flag is "\0" for residues that are neither ATOM nor
                # HETATM.
                if (res.entity_type not in keep
                        or res.het_flag not in ("A", "H")):
                    continue
                atoms = np.full((4, 3), np.nan)
                has_ca = False
                for atom in res:
                    slot = _ATOM_SLOTS.get(atom.name)
                    if slot is not None:
                        atoms[slot] = (atom.pos.x, atom.pos.y, atom.pos.z)
                        has_ca |= slot == 1
                if has_ca:
                    aa.append(THREE_TO_ONE.get(res.name, "X"))
                    coords.append(atoms)
            parsed.append((model_num, chain.name, "".join(aa), coords))
    several_models = len(st) > 1
    several_chains = len(parsed) > 1
    chains = []
    for model_num, chain_name, aa, coords in parsed:
        if len(aa) <= 3 or set(aa) == {"X"}:
            continue
        name = stem
        if several_models:
            name += f"_MODEL_{model_num}"
        if several_chains:
            name += f"_{chain_name}"
        xyz = np.array(coords, dtype=np.float64)
        chains.append(Chain(name=name.split()[0], aa=aa, n=xyz[:, 0],
                            ca=xyz[:, 1], c=xyz[:, 2], cb=xyz[:, 3]))
    return chains


_ATOM_SLOTS = {"N": 0, "CA": 1, "C": 2, "CB": 3}


# ---------------------------------------------------------------------------
# Geometry (structureto3di.cpp)
# ---------------------------------------------------------------------------

def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def _norm(a: np.ndarray) -> np.ndarray:
    return a / np.sqrt(_dot(a, a))[..., None]


def _distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a - b
    return np.sqrt(_dot(d, d))


def approx_cb(ca: np.ndarray, n: np.ndarray, c: np.ndarray) -> np.ndarray:
    """CB position assuming a tetrahedral CA."""
    v1 = _norm(c - ca)
    v2 = _norm(n - ca)
    b1 = v2 + v1 * (1 / 3.0)
    b2 = np.cross(v1, b1)
    u1 = _norm(b1)
    u2 = _norm(b2)
    v4 = v1 * (-1 / 3.0) + (u1 * (-1 / 2.0) - u2 * (np.sqrt(3) / 2.0)) \
        * (np.sqrt(8) / 3.0)
    return ca + v4 * DISTANCE_ALPHA_BETA


def _rotate(v: np.ndarray, k: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation of v around the unit axis k."""
    return (v * np.cos(angle) + np.cross(k, v) * np.sin(angle)
            + k * _dot(k, v)[..., None] * (1 - np.cos(angle)))


def virtual_center(ca: np.ndarray, cb: np.ndarray, n: np.ndarray,
                   alpha: float = VIRTUAL_CENTER[0],
                   beta: float = VIRTUAL_CENTER[1],
                   d: float = VIRTUAL_CENTER[2]) -> np.ndarray:
    """Rotate CB by alpha in the CA-N-CB plane and by the dihedral beta
    around CA-N; scale the CA distance by d."""
    alpha = alpha / 180 * FOLDSEEK_PI
    beta = beta / 180 * FOLDSEEK_PI
    v = cb - ca
    v = _rotate(v, _norm(np.cross(cb - ca, n - ca)), alpha)
    v = _rotate(v, _norm(n - ca), beta)
    return ca + v * d


def find_partners(centers: np.ndarray, valid: np.ndarray,
                  chunk: int = 2**22) -> np.ndarray:
    """Index of each residue's closest residue by virtual center, or -1.

    The first and last residue, invalid residues and the residue itself
    are never partners; ties go to the lowest index.
    """
    length = len(centers)
    partners = np.full(length, -1, dtype=np.int64)
    candidates = valid.copy()
    candidates[[0, length - 1]] = False
    rows = max(1, chunk // length)
    with np.errstate(invalid="ignore"):
        for start in range(1, length - 1, rows):
            stop = min(start + rows, length - 1)
            dist = _distance(centers[start:stop, None], centers[None])
            dist[np.isnan(dist) | ~candidates[None]] = np.inf
            dist[np.arange(stop - start), np.arange(start, stop)] = np.inf
            best = np.argmin(dist, axis=1)
            found = np.isfinite(dist[np.arange(stop - start), best])
            partners[start:stop] = np.where(found, best, -1)
    return partners


def descriptors(ca: np.ndarray, partners: np.ndarray, valid: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
    """The 10 conformation descriptors and the final residue mask.

    A residue keeps its 3Di only if it, its partner and both their
    sequence neighbours are valid.
    """
    length = len(ca)
    features = np.zeros((length, 10), dtype=np.float64)
    mask = np.zeros(length, dtype=bool)
    i = np.arange(1, length - 1)
    j = partners[i]
    js = np.where(j < 0, 1, j)
    ok = (valid[i - 1] & valid[i] & valid[i + 1] & (j >= 0)
          & valid[js - 1] & valid[js] & valid[js + 1])
    i, j = i[ok], j[ok]
    mask[i] = True
    with np.errstate(invalid="ignore", divide="ignore"):
        u1 = _norm(ca[i] - ca[i - 1])
        u2 = _norm(ca[i + 1] - ca[i])
        u3 = _norm(ca[j] - ca[j - 1])
        u4 = _norm(ca[j + 1] - ca[j])
        u5 = _norm(ca[j] - ca[i])
    offset = (j - i).astype(np.float64)
    features[i] = np.stack([
        _dot(u1, u2), _dot(u3, u4), _dot(u1, u5), _dot(u3, u5),
        _dot(u1, u4), _dot(u2, u3), _dot(u1, u3),
        _distance(ca[i], ca[j]),
        np.copysign(np.minimum(np.abs(offset), 4), offset),
        np.copysign(np.log(np.abs(offset) + 1), offset),
    ], axis=1)
    return features, mask


def translate(chain: Chain, encoder: Encoder3Di
              ) -> tuple[np.ndarray, np.ndarray]:
    """(L, 20) squared centroid distances and the (L,) mask of residues
    with a 3Di; rows of masked residues are undefined (zero)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        cb = np.where(np.isnan(chain.cb[:, :1]),
                      approx_cb(chain.ca, chain.n, chain.c), chain.cb)
        centers = virtual_center(chain.ca, cb, chain.n)
    valid = ~(np.isnan(chain.ca[:, 0]) | np.isnan(chain.c[:, 0])
              | np.isnan(chain.n[:, 0]))
    partners = find_partners(centers, valid)
    # Only inner residues lose validity for lacking a partner.
    valid[1:-1] &= partners[1:-1] >= 0
    features, mask = descriptors(chain.ca, partners, valid)
    d2 = np.zeros((len(chain), len(encoder.centroids)))
    if mask.any():
        d2[mask] = encoder.sq_distances(encoder.embed(features[mask]))
    return d2, mask


def states(d2: np.ndarray, mask: np.ndarray) -> str:
    """Foldseek's 3Di string: the nearest centroid, ``D`` if masked."""
    best = np.where(mask, np.argmin(d2, axis=1), INVALID_STATE)
    return "".join(np.array(list(CLASS_ORDER))[best])


def to_logits(d2: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``-d2``; masked rows 0."""
    return np.where(mask[:, None], -d2, 0.0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def input_files(inputs: Iterable[str | Path]) -> list[Path]:
    """Files given directly, plus all files under given directories."""
    files = []
    for item in map(Path, inputs):
        if item.is_dir():
            files += sorted(p for p in item.rglob("*") if p.is_file())
        else:
            files.append(item)
    return files


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="learnMSA-3di-translate",
        description="Translate protein structures to per-residue 3Di logits "
        "with foldseek's 3Di encoder.",
    )
    parser.add_argument("-i", "--input", required=True, nargs="+",
                        help="PDB/mmCIF files (optionally gzipped) or "
                        "directories of them.")
    parser.add_argument("-o", "--output", required=True,
                        help="Output .npz with per-residue 3Di logits.")
    parser.add_argument("--fasta", default=None,
                        help="Also write the 3Di sequences as FASTA "
                        "(identical to foldseek).")
    parser.add_argument("--aa_fasta", default=None,
                        help="Also write the amino acid sequences of the "
                        "same chains as FASTA (input for learnMSA -i).")
    parser.add_argument("--cache_dir", default=None,
                        help="Cache for the foldseek encoder files "
                        "(default: ~/.cache/learnmsa/foldseek_3di).")
    parser.add_argument("--silent", action="store_true",
                        help="Suppress progress messages.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    from learnMSA.structure.io import KIND, PDB, SOURCE_KEY
    from learnMSA.util.embedding_cache import EmbeddingCache
    from learnMSA.util.embedding_dataset import EmbeddingDataset

    t0 = time.time()
    encoder = Encoder3Di.from_foldseek(args.cache_dir)
    names, aas, logits, letters = [], [], [], []
    failed = []
    files = input_files(args.input)
    for path in files:
        try:
            chains = read_chains(path)
        except (RuntimeError, ValueError) as e:
            failed.append(f"{path}: {e}")
            continue
        for chain in chains:
            if chain.ca_only:
                warnings.warn(f"{chain.name} has only CA atoms; foldseek "
                              "rebuilds its backbone, here it gets no 3Di.")
            d2, mask = translate(chain, encoder)
            names.append(chain.name)
            aas.append(chain.aa)
            logits.append(to_logits(d2, mask))
            letters.append(states(d2, mask))
    if failed:
        print(f"Skipped {len(failed)} unreadable files, e.g. {failed[0]}",
              file=sys.stderr)
    if not names:
        raise SystemExit("No protein chains found.")
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise SystemExit(f"Duplicate entry names: {duplicates[:5]}")
    lengths = np.array([len(s) for s in aas], dtype=np.int64)
    dataset = EmbeddingDataset(
        embedding_cache=EmbeddingCache(
            lengths, len(CLASS_ORDER),
            cache=np.concatenate(logits).astype(np.float16),
        ),
        seq_ids=names,
    )
    path = dataset.write(args.output, metadata={
        "alphabet": CLASS_ORDER,
        "kind": KIND,
        "source": f"foldseek@{FOLDSEEK_COMMIT} 3Di VQ-VAE encoder",
        SOURCE_KEY: PDB,
    })
    for out, seqs in ((args.fasta, letters), (args.aa_fasta, aas)):
        if out:
            _write_fasta(out, names, seqs)
    if not args.silent:
        extra = [f for f in (args.fasta, args.aa_fasta) if f]
        print(f"Translated {len(names)} chains ({lengths.sum()} residues) "
              f"from {len(files) - len(failed)} files. Wrote {path}"
              + "".join(f", {f}" for f in extra)
              + f" ({time.time() - t0:.1f} s in total).", file=sys.stderr)


def _write_fasta(path: str | Path, names: Sequence[str],
                 seqs: Sequence[str]) -> None:
    with open(path, "w") as f:
        for name, seq in zip(names, seqs):
            f.write(f">{name}\n{seq}\n")


if __name__ == "__main__":
    main()
