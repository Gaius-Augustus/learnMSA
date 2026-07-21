"""Fit a Dirichlet prior (single or mixture) to multiple sequence alignments.

Fits a :class:`hidten.tf.prior.dirichlet.TFDirichletPrior` by maximum
likelihood to the column distributions of a directory of alignments and saves
the weights in the format used by the learnMSA HMM
(:mod:`learnMSA.hmm.tf.util`).

``--alphabet aa`` fits the 20-dimensional simplex over the standard amino
acids (the first 20 letters of :class:`learnMSA.config.hmm.PHMMConfig`);
``X``, ``U``, ``O`` and any other symbol are ignored, like a gap.
``--alphabet 3di`` fits the structural alphabet of
:class:`learnMSA.config.structure.StructureConfig`.
``--extended_alphabet`` pads the saved amino-acid components to alphabet size
23 with ``alpha = 1`` for ``X``, ``U``, ``O``, which is the dimension the HMM
loads its amino-acid prior with.

Defaults follow the spirit of HMMER3's ``esl-mixdchlet fit``: a mixture of
Dirichlets fit to conserved alignment columns, with Henikoff & Henikoff (1994)
sequence weighting and an entropy-targeted effective sequence number (Neff).
Several runs from different initializations are compared on a held-out column
set and the best one is saved.

Example
-------
Fit a 9-component amino-acid mixture and overwrite the shipped prior::

    python -m learnMSA.hmm.tf.fit_dirichlet /path/to/msas -c 9 \\
        --extended_alphabet

Fit a 3Di structural-token prior::

    python -m learnMSA.hmm.tf.fit_dirichlet /path/to/3di_msas \\
        --alphabet 3di --name pfam_35_3Di -c 9
"""

import argparse
import importlib.resources as resources
import os
from pathlib import Path

# Must be set before any TensorFlow operations
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import numpy as np
import tensorflow as tf

from hidten.tf.prior.dirichlet import TFDirichletPrior

from learnMSA.config.hmm import PHMMConfig
from learnMSA.config.structure import StructureConfig
from learnMSA.hmm.tf.util import make_dirichlet_model
from learnMSA.util.aligned_dataset import AlignedDataset

AA_ALPHABET: str = PHMMConfig().alphabet
STRUCT_ALPHABET: str = StructureConfig().structural_alphabet

# (parse alphabet, number of fitted tokens) per --alphabet choice. Tokens after
# the fitted ones are ignored. X is always part of the parse alphabet because
# AlignedDataset maps ambiguous symbols onto it.
ALPHABETS: dict[str, tuple[str, int]] = {
    "aa": (AA_ALPHABET, 20),
    "3di": (STRUCT_ALPHABET + "X", len(STRUCT_ALPHABET)),
}

# Directory holding the shipped prior weight files.
WEIGHTS_DIR: Path = Path(str(resources.files("learnMSA.hmm.weights")))


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Fit a single Dirichlet or a mixture of Dirichlets to the column "
            "distributions of a directory of multiple sequence alignments."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing the multiple sequence alignment files.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed information during training."
    )
    parser.add_argument(
        "-c", "--components",
        type=int,
        default=9,
        help="Number of mixture components (1 fits a single Dirichlet).",
    )
    parser.add_argument(
        "--alphabet",
        type=str,
        default="aa",
        choices=sorted(ALPHABETS),
        help="Observation alphabet. 'aa' fits the 20 standard amino acids, "
             "'3di' the structural alphabet. Other symbols are ignored.",
    )
    parser.add_argument(
        "--extended-alphabet", "--extended_alphabet",
        action="store_true",
        help="Amino acids only: pad the saved components to alphabet size "
             f"{len(AA_ALPHABET)} with alpha = 1 for X, U and O.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="amino_acid_dirichlet",
        help="Output base name; the file is <name>_<components>.weights.h5.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=10,
        help="Minimum residue count required to keep a column.",
    )
    parser.add_argument(
        "--min-occupancy",
        type=float,
        default=0.5,
        help="Minimum column occupancy required to keep a column. A column "
             "must satisfy this and --min-count.",
    )
    parser.add_argument(
        "--fmt",
        type=str,
        default="fasta",
        help="File format of the alignments (passed to AlignedDataset).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*",
        help="Glob pattern for selecting alignment files in input_dir.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.05,
        help="Learning rate of the Adam optimizer.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Maximum number of training epochs per run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Number of columns per training batch.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="Number of training runs with different random initialization.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of columns held out for model selection. Set to 0 to "
             "train on all columns without validation (needs --num-runs 1).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Stop a run after this many epochs without loss improvement.",
    )
    parser.add_argument(
        "--no-map-prior",
        action="store_true",
        help="Disable the Dirichlet-Process MAP prior (Nguyen et al. 2013) "
             "and fit by pure maximum likelihood.",
    )
    parser.add_argument(
        "--freeze-hyperparams",
        action="store_true",
        help="Freeze the MAP-prior hyperparameters (gamma, beta, lambda) "
             "instead of estimating them by empirical Bayes.",
    )
    parser.add_argument(
        "--init",
        type=str,
        default="data",
        choices=["data", "random_normal"],
        help="Initialization scheme. 'data' anchors the concentrations to the "
             "background token frequencies, 'random_normal' ignores the data.",
    )
    parser.add_argument(
        "--score",
        type=str,
        default="counts",
        choices=["counts", "probabilities"],
        help="Observation model. 'counts' scores per-column token counts with "
             "the Dirichlet-multinomial marginal (Sjolander et al. 1996), "
             "'probabilities' scores normalized columns with the density.",
    )
    parser.add_argument(
        "--no-neff",
        action="store_true",
        help="Disable HMMER-style sequence weighting and fall back to raw "
             "per-sequence counts.",
    )
    parser.add_argument(
        "--neff-target-bits",
        type=float,
        default=0.59,
        help="Per-column mean-relative-entropy target (in bits) for the Neff "
             "bisection. Ignored with --score probabilities.",
    )
    parser.add_argument(
        "--neff-prior-conc",
        type=float,
        default=10.0,
        help="Total concentration of the reference Dirichlet used inside the "
             "Neff entropy target; a linear gain on every family's Neff. The "
             "self-consistent choice is the concentration of the prior being "
             "fit. Ignored with --score probabilities.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for initialization and subsampling.",
    )
    parser.add_argument(
        "--max-columns",
        type=int,
        default=None,
        help="Optional cap on the number of training columns (for memory). "
             "With --clans this is the number of columns drawn by the "
             "hierarchical sampler.",
    )
    parser.add_argument(
        "--clans",
        type=str,
        default=None,
        help="Optional Pfam-style clans TSV (col0=family accession, col1=clan "
             "accession). When given, training columns are drawn "
             "hierarchically (clan -> family -> column) to remove "
             "large-family bias. The family accession is matched from the "
             "file name up to the first dot (PF01024.25.fasta -> PF01024).",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output weight file. Defaults to the shipped weights directory.",
    )
    args = parser.parse_args()
    if args.extended_alphabet and args.alphabet != "aa":
        parser.error(
            "--extended_alphabet only applies to --alphabet aa; the "
            "structural alphabet has no X, U or O tokens."
        )
    return args


# --------------------------------------------------------------------------
# HMMER-style sequence weighting and effective-sequence-number targeting.
#
# A family with many near-identical sequences would inflate the per-column
# sample size the Dirichlet-multinomial sees. Henikoff & Henikoff (1994)
# position-based weights down-weight redundant sequences, and an entropy-based
# bisection rescales a family's total weight to an effective sequence number.
# --------------------------------------------------------------------------


def emissions_single_dirichlet(
    counts: np.ndarray, alpha: np.ndarray
) -> np.ndarray:
    """Mean-posterior emission probabilities under a single Dirichlet.

    Args:
        counts: Weighted residue counts of shape ``(M, dim)``.
        alpha: Concentration parameters of shape ``(dim,)``.

    Returns:
        Posterior mean probabilities of shape ``(M, dim)``.
    """
    post = counts + alpha
    return post / post.sum(axis=1, keepdims=True)


def henikoff_pb_weights(msa: np.ndarray, dim: int) -> np.ndarray:
    """Henikoff & Henikoff (1994) position-based sequence weights.

    In a column with ``r`` distinct residue types, a sequence carrying a
    residue of type ``a`` seen ``c(a)`` times receives ``1 / (r * c(a))``.

    Args:
        msa: Integer-encoded alignment of shape ``(num_seq, L)``; tokens
            ``>= dim`` (gaps and ignored symbols) contribute nothing.
        dim: Number of fitted tokens.

    Returns:
        Relative weights of shape ``(num_seq,)`` summing to 1. An all-gap
        alignment yields uniform weights.
    """
    num_seq = msa.shape[0]
    w = np.zeros(num_seq, dtype=np.float64)
    for j in range(msa.shape[1]):
        col = msa[:, j]
        obs = col < dim
        if not obs.any():
            continue
        counts = np.bincount(col[obs], minlength=dim)  # c(a) per type
        r = np.count_nonzero(counts)                   # distinct types
        w[obs] += 1.0 / (r * counts[col[obs]])
    total = w.sum()
    return w / total if total > 0 else np.full(num_seq, 1.0 / num_seq)


def weighted_counts(
    msa: np.ndarray, rel_w: np.ndarray, dim: int, columns: np.ndarray
) -> np.ndarray:
    """Accumulate relative-weight residue counts at the given columns.

    Because ``rel_w`` sums to 1, the result is the *base* count vector for
    :func:`target_neff`: the counts at a chosen ``neff`` are ``neff * base``.

    Args:
        msa: Integer-encoded alignment of shape ``(num_seq, L)``.
        rel_w: Per-sequence relative weights of shape ``(num_seq,)``.
        dim: Number of fitted tokens; tokens ``>= dim`` are not counted.
        columns: Column indices to accumulate, shape ``(M,)``.

    Returns:
        Weighted counts of shape ``(len(columns), dim)``.
    """
    counts = np.zeros((len(columns), dim), dtype=np.float64)
    for m, j in enumerate(columns):
        col = msa[:, j]
        obs = col < dim
        np.add.at(counts[m], col[obs], rel_w[obs])
    return counts


def mean_relative_entropy(p: np.ndarray, bg: np.ndarray) -> float:
    """Mean over columns of the relative entropy ``KL(p_col || bg)`` in bits.

    Args:
        p: Per-column probability vectors of shape ``(M, dim)``.
        bg: Background probability vector of shape ``(dim,)``.

    Returns:
        The mean over the ``M`` columns of ``sum_a p_a * log2(p_a / bg_a)``.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = p * (np.log2(p) - np.log2(bg)[None, :])
    terms = np.where(p > 0, terms, 0.0)
    return float(terms.sum(axis=1).mean())


def target_neff(
    base_counts: np.ndarray,
    alpha: np.ndarray,
    bg: np.ndarray,
    target_bits: float = 0.59,
    n_actual: float | None = None,
    lo: float = 1e-3,
    iters: int = 60,
) -> tuple[float, np.ndarray]:
    """Find the effective sequence number matching a relative-entropy target.

    Mirrors HMMER's entropy weighting (``--eent``): the mean relative entropy
    of the posterior-mean emissions grows monotonically with ``neff``, so a
    bisection finds the ``neff`` at which it equals ``target_bits``. If even
    ``neff = n_actual`` falls short, ``n_actual`` is kept. The emissions depend
    on ``alpha`` and ``neff`` only through their ratio, so scaling ``alpha``
    scales the result by the same factor.

    Args:
        base_counts: Relative-weight counts of shape ``(M, dim)``.
        alpha: Reference single-Dirichlet concentrations of shape ``(dim,)``.
        bg: Background probability vector of shape ``(dim,)``.
        target_bits: Target mean relative entropy per column, in bits.
        n_actual: Upper bound on ``neff`` (the actual number of sequences).
        lo: Lower bound of the bisection.
        iters: Number of bisection iterations.

    Returns:
        The effective sequence number and the posterior-mean emissions of
        shape ``(M, dim)`` evaluated at it.
    """
    if n_actual is not None:
        hi = float(n_actual)
    else:
        hi = float(base_counts.shape[0])

    def re_at(neff: float) -> tuple[float, np.ndarray]:
        p = emissions_single_dirichlet(neff * base_counts, alpha)
        return mean_relative_entropy(p, bg), p

    # If the maximum attainable entropy already falls short, keep n_actual.
    re_hi, p_hi = re_at(hi)
    if re_hi <= target_bits:
        return hi, p_hi

    a, b = lo, hi
    mid, p_mid = hi, p_hi
    for _ in range(iters):
        mid = 0.5 * (a + b)
        re_mid, p_mid = re_at(mid)
        if re_mid > target_bits:
            b = mid
        else:
            a = mid
    return mid, p_mid


def collect_columns(
    input_dir: str,
    pattern: str,
    fmt: str,
    alphabet: str,
    min_count: int,
    min_occupancy: float,
    fit_dim: int | None = None,
    neff: bool = False,
    neff_scaling: bool = True,
    neff_target_bits: float = 0.59,
    neff_prior_conc: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Collect per-column token counts from a directory of alignments.

    Each retained column becomes a count vector over the first ``fit_dim``
    tokens of ``alphabet``. Tokens beyond that are ignored: they are neither
    counted nor do they count toward the column filters, exactly like a gap.

    With ``neff`` the counts are HMMER-style effective counts: sequences are
    down-weighted by :func:`henikoff_pb_weights` (which reshapes every column
    distribution) and, if ``neff_scaling`` is set, each family's total weight
    is rescaled to its entropy-targeted effective sequence number
    (:func:`target_neff`, a per-family scalar). The reference background of the
    entropy target is the global marginal over all kept columns.

    Args:
        input_dir: Directory containing the alignment files.
        pattern: Glob pattern selecting files within ``input_dir``.
        fmt: File format passed to :class:`AlignedDataset`.
        alphabet: Non-gap parse alphabet; a gap is appended for parsing.
        min_count: Minimum count over the fitted tokens to keep a column.
        min_occupancy: Minimum column occupancy to keep a column.
        fit_dim: Number of leading ``alphabet`` tokens to fit; the remaining
            ones are ignored. Defaults to the full alphabet.
        neff: Whether to apply Henikoff position-based sequence weighting.
        neff_scaling: Whether to additionally rescale each family's counts to
            its effective sequence number. Pass False when the columns are
            normalized to probabilities anyway, which cancels the scalar.
        neff_target_bits: Entropy target of the Neff bisection.
        neff_prior_conc: Concentration of the reference Dirichlet of the
            entropy target.

    Returns:
        A tuple ``(columns, family_of_column, family_accessions)`` of the
        ``(M, fit_dim)`` count vectors, the family index of each column and
        the accession of each family (the file name up to the first dot).
        Only files that contributed at least one kept column are indexed.
    """
    dim = len(alphabet) if fit_dim is None else fit_dim
    parse_alphabet = alphabet + "-"
    files = sorted(p for p in Path(input_dir).glob(pattern) if p.is_file())
    if not files:
        raise ValueError(
            f"No files matching '{pattern}' found in '{input_dir}'."
        )

    distributions: list[np.ndarray] = []
    family_ids: list[np.ndarray] = []
    family_accessions: list[str] = []
    family_num_seq: list[int] = []  # sequences per family (for Neff)
    bg_accum = np.zeros(dim, dtype=np.float64)  # global marginal counts
    for path in files:
        try:
            data = AlignedDataset(
                filepath=path, fmt=fmt, alphabet=parse_alphabet
            )
        except Exception:  # noqa: BLE001 - skip unreadable files
            continue

        matrix = data.msa_matrix  # (num_seq, L)
        num_seq = matrix.shape[0]
        # Counts over the fitted tokens only; higher indices (ignored symbols
        # and the gap) are not counted anywhere below.
        counts = np.stack(
            [np.sum(matrix == a, axis=0) for a in range(dim)], axis=1
        ).astype(np.float64)  # (L, dim)
        totals = counts.sum(axis=1)
        occupancy = totals / max(num_seq, 1)
        keep = (totals >= max(min_count, 1)) & (occupancy >= min_occupancy)
        if not np.any(keep):
            continue
        # The global background is the marginal over the kept columns.
        bg_accum += counts[keep].sum(axis=0)
        if neff:
            # Scaled to the family's Neff below, once the background needed by
            # the entropy target is known.
            rel_w = henikoff_pb_weights(matrix, dim)
            kept = weighted_counts(matrix, rel_w, dim, np.where(keep)[0])
        else:
            kept = counts[keep]
        family_index = len(family_accessions)
        family_accessions.append(path.name.split(".")[0])
        distributions.append(kept)
        family_ids.append(
            np.full((kept.shape[0],), family_index, dtype=np.int64)
        )
        family_num_seq.append(num_seq)

    if not distributions:
        raise ValueError(
            "No columns passed the filters. Try lowering --min-count or "
            "--min-occupancy."
        )

    if neff and neff_scaling:
        bg = np.clip(bg_accum / max(bg_accum.sum(), 1.0), 1e-8, None)
        bg = bg / bg.sum()
        alpha = bg * neff_prior_conc
        neffs = np.empty(len(distributions), dtype=np.float64)
        for i, (base, n_seq) in enumerate(zip(distributions, family_num_seq)):
            neff_i, _ = target_neff(
                base, alpha, bg, target_bits=neff_target_bits, n_actual=n_seq
            )
            distributions[i] = neff_i * base
            neffs[i] = neff_i
        num_seq_arr = np.maximum(family_num_seq, 1)
        ratios = neffs / num_seq_arr
        # Families that could not reach the entropy target keep Neff=num_seq.
        # A large share means --neff-prior-conc is too high for this data.
        clamped = 100.0 * np.mean(neffs >= num_seq_arr - 1e-6)
        print(
            f"Neff weighting: per-family Neff min/median/mean/max = "
            f"{neffs.min():.2f}/{np.median(neffs):.2f}/{neffs.mean():.2f}/"
            f"{neffs.max():.2f}; mean Neff/num_seq = {ratios.mean():.3f}; "
            f"{clamped:.1f}% of families clamped at num_seq."
        )

    columns = np.concatenate(distributions, axis=0).astype(np.float64)
    family_of_column = np.concatenate(family_ids, axis=0)
    print(
        f"Collected {columns.shape[0]} columns from {len(family_accessions)} "
        f"alignment(s); fitting {dim} of the {len(alphabet)} token(s) in "
        f"'{alphabet}'."
    )
    return columns, family_of_column, family_accessions


def load_clan_of_family(
    clans_path: str, family_accessions: list[str]
) -> np.ndarray:
    """Map each family to a clan index; clanless families are singletons.

    The clans file is a Pfam-style TSV: column 0 is a family accession, column
    1 its clan accession (empty when the family has no clan). A family with an
    empty clan field, or absent from the file, gets its own singleton clan so
    it competes on equal footing in a uniform clan draw.

    Args:
        clans_path: Path to the tab-separated clans file.
        family_accessions: Family accession per family index.

    Returns:
        Array of shape ``(num_families,)`` with a contiguous clan index per
        family.
    """
    fam_to_clan: dict[str, str] = {}
    with open(clans_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if not parts or not parts[0]:
                continue
            clan = parts[1] if len(parts) > 1 and parts[1] else None
            if clan is not None:
                fam_to_clan[parts[0]] = clan

    clan_labels: list[str] = []
    num_singletons = 0
    for acc in family_accessions:
        clan = fam_to_clan.get(acc)
        if clan is None:
            clan = f"__singleton__{acc}"
            num_singletons += 1
        clan_labels.append(clan)

    clan_to_index: dict[str, int] = {}
    clan_of_family = np.empty((len(clan_labels),), dtype=np.int64)
    for i, label in enumerate(clan_labels):
        clan_of_family[i] = clan_to_index.setdefault(label, len(clan_to_index))

    print(
        f"Loaded clans for {len(family_accessions)} famil(y/ies): "
        f"{len(clan_to_index)} distinct clan(s), of which {num_singletons} "
        f"are singleton clans (no clan or absent from the TSV)."
    )
    return clan_of_family


def sample_clan_family_columns(
    family_of_column: np.ndarray,
    clan_of_family: np.ndarray,
    subset_idx: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw column indices by hierarchical clan -> family -> column sampling.

    Draws (with replacement) a clan uniformly, then a family within it, then a
    column within that family. Only the columns in ``subset_idx`` are
    eligible, so training and validation can draw from disjoint pools.

    Args:
        family_of_column: Family index per column, shape ``(M,)``.
        clan_of_family: Clan index per family, shape ``(num_families,)``.
        subset_idx: Indices of the eligible columns.
        n_samples: Number of columns to draw.
        rng: Random generator for the draws.

    Returns:
        Array of ``n_samples`` column indices into ``family_of_column``.
    """
    subset_idx = np.asarray(subset_idx)
    if subset_idx.size == 0:
        raise ValueError("Cannot sample from an empty column subset.")

    # Sorting by family index lays the columns of a family out contiguously,
    # so a family is a flat slice.
    families = family_of_column[subset_idx]
    order = np.argsort(families, kind="stable")
    family_col_indices = subset_idx[order]
    fam_sorted = families[order]
    present_families, first_pos, family_col_count = np.unique(
        fam_sorted, return_index=True, return_counts=True
    )
    family_col_start = first_pos

    # Group the present families by their clan, laid out contiguously likewise.
    clans = clan_of_family[present_families]
    clan_order = np.argsort(clans, kind="stable")
    # Index of each clan slot into the present-family arrays, so drawing a
    # family slot yields a present-family position.
    clan_fam_indices = clan_order
    clan_sorted = clans[clan_order]
    _, clan_first_pos, clan_fam_count = np.unique(
        clan_sorted, return_index=True, return_counts=True
    )
    clan_fam_start = clan_first_pos
    num_clans = clan_fam_start.shape[0]

    c = rng.integers(0, num_clans, size=n_samples)
    fam_slot = clan_fam_start[c] + np.floor(
        rng.random(n_samples) * clan_fam_count[c]
    ).astype(np.int64)
    present_family_pos = clan_fam_indices[fam_slot]
    col_slot = family_col_start[present_family_pos] + np.floor(
        rng.random(n_samples) * family_col_count[present_family_pos]
    ).astype(np.int64)
    return family_col_indices[col_slot]


def random_normal_initializer(
    dim: int, components: int, seed: int
) -> np.ndarray:
    """Build a flat initializer by drawing from a normal distribution.

    The values must be strictly positive, so the draws are exponentiated
    (log-normal concentrations) and passed through a softmax (mixture
    weights). This also breaks the symmetry between mixture components.

    Args:
        dim: Dimension of the categorical distribution.
        components: Number of mixture components.
        seed: Random seed for the draws.

    Returns:
        Flat initializer ``[conc_1, ..., conc_C, mixture_weights]`` (the
        mixture weights are omitted for a single component).
    """
    rng = np.random.default_rng(seed)
    concentrations = np.exp(rng.normal(0.0, 1.0, size=(components, dim)))
    if components == 1:
        return concentrations.reshape(-1).astype(np.float64)
    logits = rng.normal(0.0, 1.0, size=(components,))
    weights = np.exp(logits - logits.max())
    weights = weights / weights.sum()
    return np.concatenate(
        [concentrations.reshape(-1), weights]
    ).astype(np.float64)


def make_initializer(
    columns: np.ndarray, components: int, seed: int, init: str = "data"
) -> np.ndarray:
    """Build a flat initializer for the Dirichlet prior.

    With ``init="data"`` the concentrations are anchored to the background mean
    of the data. For mixtures the symmetry between components must be broken,
    otherwise identical components receive identical gradients and never
    specialize; this is done by drawing a different total concentration per
    component, applying per-token noise and randomizing the mixture weights.
    Different seeds therefore yield genuinely different starting points, which
    is what the multi-run model selection relies on.

    Args:
        columns: Training column distributions of shape ``(M, dim)``.
        components: Number of mixture components.
        seed: Random seed for the noise.
        init: Initialization scheme, ``"data"`` or ``"random_normal"``.

    Returns:
        Flat initializer ``[conc_1, ..., conc_C, mixture_weights]`` (the
        mixture weights are omitted for a single component).
    """
    dim = columns.shape[1]
    if init == "random_normal":
        return random_normal_initializer(dim, components, seed)

    rng = np.random.default_rng(seed)
    # Clipped away from zero so the concentrations stay positive.
    mu = np.clip(columns.mean(axis=0), 1e-4, None)
    mu = mu / mu.sum()

    if components == 1:
        return (mu * 10.0).astype(np.float64)

    total_concentration = rng.uniform(5.0, 20.0, size=(components, 1))
    noise = rng.uniform(0.5, 1.5, size=(components, dim))
    concentrations = mu[np.newaxis, :] * total_concentration * noise
    mixture_weights = rng.dirichlet(np.full((components,), 2.0))
    return np.concatenate(
        [concentrations.reshape(-1), mixture_weights]
    ).astype(np.float64)


# Probability floor keeping the MAP-prior densities and their gradients finite
# when mixture components die: the optimizer drives unused components toward
# zero, where an unguarded log density would evaluate to NaN.
_PROB_FLOOR: float = 1e-8


def _dirichlet_log_pdf(p: tf.Tensor, alpha: tf.Tensor) -> tf.Tensor:
    """Log density ``log Dir(p | alpha)``, summed over the last axis.

    Args:
        p: Probability vectors of shape ``(..., D)``, floored at
            :data:`_PROB_FLOOR`. Broadcasts against ``alpha``.
        alpha: Concentration parameters of shape ``(..., D)``.

    Returns:
        Log densities with the broadcast batch shape.
    """
    p = tf.maximum(p, _PROB_FLOOR)
    log_z = tf.math.lbeta(alpha)
    return tf.reduce_sum(tf.math.xlogy(alpha - 1.0, p), axis=-1) - log_z


class DirichletMAPRegularizer(tf.keras.layers.Layer):
    """Training-only Dirichlet-Process MAP prior over a ``TFDirichletPrior``.

    Reproduces the MAP objective of Nguyen et al. 2013, *"Dirichlet Mixtures,
    the Dirichlet Process, and the Structure of Protein Space"*. The log prior
    sums three normalized densities over the fitted prior's parameters: an
    Exponential(``lambda``) prior on each component's total concentration, a
    symmetric Dirichlet(``gamma / C``) prior on the mixture weights (``C > 1``
    only) and a Dirichlet(``beta * background``) prior on each component mean.

    ``gamma``, ``beta`` and ``lambda`` are trainable by default (empirical
    Bayes); since every term is a proper density, its log-normalizer keeps the
    regularizer from trivially weakening itself. These weights live only in
    the fitting model, so the saved weight layout is untouched.

    Args:
        dim: Dimension of the categorical distribution.
        components: Number of mixture components.
        background_init: Logits initializing the frozen background anchor.
        trainable_hyperparams: Whether to estimate gamma, beta and lambda.
    """

    def __init__(
        self,
        dim: int,
        components: int,
        background_init: np.ndarray,
        trainable_hyperparams: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.dim = dim
        self.components = components
        self.background_init = background_init
        self.trainable_hyperparams = trainable_hyperparams

    def build(self, input_shape=None) -> None:
        # Kernels pass through softplus, so a large init value ~= its softplus.
        self.gamma_kernel = self.add_weight(
            shape=(1,), initializer=tf.constant_initializer(50.0),
            name="gamma_kernel", trainable=self.trainable_hyperparams,
        )
        self.beta_kernel = self.add_weight(
            shape=(1,), initializer=tf.constant_initializer(100.0),
            name="beta_kernel", trainable=self.trainable_hyperparams,
        )
        self.lambda_kernel = self.add_weight(
            shape=(1,), initializer="ones",
            name="lambda_kernel", trainable=self.trainable_hyperparams,
        )
        # The shrinkage anchor is always frozen to the data mean.
        self.background_kernel = self.add_weight(
            shape=(self.dim,),
            initializer=tf.constant_initializer(self.background_init),
            name="background_kernel", trainable=False,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Identity passthrough.

        The layer is inserted into the graph purely so Keras tracks its
        hyperparameter weights. The penalty itself is added in
        :meth:`DirichletTrainingModel.compute_loss`.
        """
        return inputs

    def log_prior(self, prior: TFDirichletPrior) -> tf.Tensor:
        """Total MAP log prior evaluated on ``prior``'s parameters."""
        matrix = prior.matrix()  # (H, Q, P)
        gamma = tf.math.softplus(self.gamma_kernel)
        beta = tf.math.softplus(self.beta_kernel)
        lam = tf.math.softplus(self.lambda_kernel)
        background = tf.nn.softmax(self.background_kernel)

        if self.components == 1:
            conc = matrix  # (H, Q, D)
            sum_alpha = tf.reduce_sum(conc, axis=-1)  # (H, Q)
            # Guard the denominator: a dead component's concentrations
            # underflow to zero and an unguarded 0/0 would give NaN. The
            # floored mean is 0, which _dirichlet_log_pdf handles.
            safe_sum = tf.maximum(sum_alpha, _PROB_FLOOR)
            means = conc / tf.expand_dims(safe_sum, -1)  # (H, Q, D)
            mix_prior = tf.zeros((), dtype=matrix.dtype)
        else:
            conc = prior._slice_concentrations(matrix)  # (H, Q, C, D)
            mix = prior._slice_mixture_coefficients(matrix)  # (H, Q, C)
            sum_alpha = tf.reduce_sum(conc, axis=-1)  # (H, Q, C)
            safe_sum = tf.maximum(sum_alpha, _PROB_FLOOR)
            means = conc / tf.expand_dims(safe_sum, -1)  # (H, Q, C, D)
            mix_conc = tf.ones_like(mix) * gamma / self.components
            mix_prior = tf.reduce_sum(_dirichlet_log_pdf(mix, mix_conc))

        sum_alpha_prior = tf.reduce_sum(tf.math.log(lam) - lam * sum_alpha)
        comp_dist = background * beta  # (D,)
        comp_prior = tf.reduce_sum(_dirichlet_log_pdf(means, comp_dist))

        return sum_alpha_prior + mix_prior + comp_prior


class DirichletTrainingModel(tf.keras.Model):
    """Trainable model scoring observations with a ``TFDirichletPrior``.

    The explicit scoring method is invoked here rather than through the
    prior's ``call`` so the inference-time prior keeps a single, unambiguous
    behavior. A MAP ``regularizer``, if given, is added to the training loss
    only, so the validation loss stays a pure held-out log-likelihood; its
    penalty is scaled by ``1 / num_examples`` to match the per-batch mean
    log-likelihood used as the loss.

    Args:
        prior: The (already built) trainable ``TFDirichletPrior``.
        score_counts: If True, score count vectors with the
            Dirichlet-multinomial marginal; otherwise probability vectors
            with the Dirichlet density.
        regularizer: Optional MAP-prior layer owning the hyperparameters.
        num_examples: Number of training columns; required with
            ``regularizer``.
    """

    def __init__(
        self,
        prior: TFDirichletPrior,
        score_counts: bool,
        regularizer: DirichletMAPRegularizer | None = None,
        num_examples: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.prior = prior
        self.score_counts = score_counts
        self.map_regularizer = regularizer
        self._inv_n = 1.0 / float(num_examples) if num_examples else 0.0

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        if self.score_counts:
            return self.prior.dirichlet_multinomial_scores(inputs)
        return self.prior.prior_scores(inputs)

    def compute_loss(
        self,
        x: tf.Tensor | None = None,
        y: tf.Tensor | None = None,
        y_pred: tf.Tensor | None = None,
        sample_weight: tf.Tensor | None = None,
        training: bool = True,
    ) -> tf.Tensor:
        loss = super().compute_loss(x, y, y_pred, sample_weight, training)
        if training and self.map_regularizer is not None:
            penalty = self.map_regularizer.log_prior(self.prior)
            loss = loss - penalty * self._inv_n
        return loss


def build_trainable_model(
    initializer: np.ndarray,
    dim: int,
    components: int,
    background_init: np.ndarray | None = None,
    num_examples: int | None = None,
    use_map_prior: bool = True,
    freeze_hyperparams: bool = False,
    score_counts: bool = True,
) -> tf.keras.Model:
    """Create a trainable keras model wrapping a ``TFDirichletPrior``.

    Args:
        initializer: Flat initializer for the prior parameters.
        dim: Dimension of the categorical distribution.
        components: Number of mixture components.
        background_init: Logits for the frozen background anchor (required
            when ``use_map_prior`` is True).
        num_examples: Number of training columns for the MAP scaling
            (required when ``use_map_prior`` is True).
        use_map_prior: Whether to attach the Dirichlet-Process MAP prior.
        freeze_hyperparams: If True, freeze gamma/beta/lambda at their inits.
        score_counts: If True, train on count vectors; otherwise on
            probability vectors.

    Returns:
        A built keras model whose prior layer is trainable.
    """
    # make_dirichlet_model builds the prior with the correct sharing layout; we
    # reuse its (single) prior layer as the trainable component of our model.
    base = make_dirichlet_model(
        initializer=initializer, dim=dim, components=components
    )
    prior = base.layers[1]
    # Priors are frozen by default; enable training of the concentrations.
    prior.trainable = True

    regularizer = None
    if use_map_prior:
        assert background_init is not None and num_examples is not None, (
            "background_init and num_examples are required for the MAP prior."
        )
        regularizer = DirichletMAPRegularizer(
            dim, components, background_init,
            trainable_hyperparams=not freeze_hyperparams,
        )
        regularizer.build()

    model = DirichletTrainingModel(
        prior, score_counts=score_counts,
        regularizer=regularizer, num_examples=num_examples,
    )
    # Build the model so a subsequent summary()/fit() has known weights.
    model(tf.zeros((1, 1, dim)))
    return model


def _as_arrays(columns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Turn column distributions into the ``(x, y)`` pair ``fit()`` expects.

    Each column is a single-state observation of shape ``(1, dim)`` paired
    with a dummy target. The arrays are handed to ``model.fit()`` directly
    rather than wrapped in a ``tf.data`` pipeline: a ``Dataset.shuffle`` buffer
    holding millions of tiny per-column tensors crashes the TF runtime
    (segfault in ``Tensor::AllocatedBytes`` on a tf.data worker thread), while
    Keras' array data adapter shuffles by gathering a permuted index vector.
    """
    x = columns[:, np.newaxis, :].astype(np.float32)  # (M, 1, dim)
    y = np.zeros((columns.shape[0],), dtype=np.float32)  # dummy targets
    return x, y


def train(
    model: tf.keras.Model,
    train_columns: np.ndarray,
    val_columns: np.ndarray | None,
    lr: float,
    epochs: int,
    batch_size: int,
    patience: int,
    seed: int,
    verbose: int = 0,
) -> float:
    """Fit the Dirichlet prior by maximizing the column log-likelihood.

    Stops early once the validation loss has not improved for ``patience``
    epochs, restoring the weights of the best epoch. Without validation
    columns the training loss is monitored instead.

    Args:
        model: The trainable Dirichlet model.
        train_columns: Training column distributions of shape ``(M, dim)``.
        val_columns: Validation columns, or ``None`` for no validation.
        lr: Learning rate of the Adam optimizer.
        epochs: Maximum number of training epochs.
        batch_size: Number of columns per batch.
        patience: Epochs without loss improvement before stopping.
        seed: Random seed for the per-epoch shuffling. Keras draws its
            permutation from the global TensorFlow seed, which is set here.
        verbose: Verbosity level passed to ``model.fit()``.

    Returns:
        The best monitored loss, i.e. the negative mean column
        log-likelihood on the validation (or training) columns.
    """
    tf.random.set_seed(seed)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    # The model outputs per-column log densities; minimize their negative mean.
    model.compile(
        optimizer=optimizer,
        loss=lambda y_true, y_pred: -tf.reduce_mean(y_pred),
    )

    x_train, y_train = _as_arrays(train_columns)
    if val_columns is None:
        val_data = None
        monitor = "loss"
    else:
        val_data = _as_arrays(val_columns)
        monitor = "val_loss"

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        mode="min",
        patience=patience,
        restore_best_weights=True,
    )
    history = model.fit(
        x_train,
        y_train,
        validation_data=val_data,
        batch_size=batch_size,
        shuffle=True,
        epochs=epochs,
        callbacks=[early_stopping],
        verbose=verbose,
    )
    best_loss = float(np.min(history.history[monitor]))
    label = "validation" if val_data is not None else "training"
    print(f"Best {label} mean column log-likelihood: {-best_loss:.4f}")
    return best_loss


def extend_concentrations(
    matrix: np.ndarray, dim: int, components: int, extended_dim: int
) -> np.ndarray:
    """Pad each component's concentrations with ``alpha = 1``.

    Lifts a prior fit on the 20 standard amino acids to the full HMM alphabet:
    the ignored tokens (X, U, O) get a flat ``alpha = 1`` instead of a fitted
    value. Mixture weights are carried over unchanged.

    Args:
        matrix: The fitted parameters, i.e. ``prior.matrix()[0, 0]``: the
            concentrations of each component followed by the mixture
            coefficients (the latter only for ``components > 1``).
        dim: Dimension the prior was fit at.
        components: Number of mixture components.
        extended_dim: Dimension to pad to; must be at least ``dim``.

    Returns:
        A flat initializer for :func:`make_dirichlet_model` at
        ``extended_dim``.
    """
    assert extended_dim >= dim, "Cannot extend to a smaller dimension."
    conc = matrix[:components * dim].reshape(components, dim)
    padding = np.ones((components, extended_dim - dim), dtype=conc.dtype)
    extended = np.concatenate([conc, padding], axis=1).reshape(-1)
    if components == 1:
        return extended.astype(np.float64)
    mix = matrix[components * dim:]
    return np.concatenate([extended, mix]).astype(np.float64)


def save(
    model: tf.keras.Model, output_path: Path, dim: int, components: int
) -> None:
    """Save the prior weights and verify they round-trip.

    Args:
        model: The trained Dirichlet model.
        output_path: Destination ``.weights.h5`` file.
        dim: Dimension of the categorical distribution.
        components: Number of mixture components.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(output_path))
    print(f"Saved weights to {output_path}")

    reloaded = make_dirichlet_model(dim=dim, components=components)
    reloaded.load_weights(str(output_path))
    np.testing.assert_allclose(
        reloaded.layers[1].matrix().numpy(),
        model.layers[1].matrix().numpy(),
        atol=1e-6,
    )
    print("Round-trip verification passed.")


def summarize_prior(
    prior: TFDirichletPrior, dim: int, components: int
) -> str:
    """Build a diagnostic summary of a fitted Dirichlet prior.

    Reports the per-component total concentration and, for mixtures, the
    weights and the effective number of active components ``exp(H(weights))``.
    This makes the two degenerate outcomes on sparse data visible: a sub-1
    total concentration and a mixture collapsed onto one component.

    Args:
        prior: The fitted :class:`TFDirichletPrior` layer.
        dim: Dimension of the categorical distribution.
        components: Number of mixture components.

    Returns:
        A multi-line summary string.
    """
    matrix = prior.matrix()[0, 0].numpy()
    lines = ["Prior summary:"]
    if components == 1:
        alpha = matrix[:dim]
        total = float(alpha.sum())
        mean = alpha / total
        dom = int(mean.argmax())
        lines.append(f"  total concentration sum(alpha) = {total:.4f}")
        lines.append(f"  dominant token {dom} (mean prob {mean[dom]:.4f})")
    else:
        conc = matrix[:components * dim].reshape(components, dim)
        mix = matrix[components * dim:]
        totals = conc.sum(axis=1)
        comp_mean = conc / totals[:, np.newaxis]
        dom = comp_mean.argmax(axis=1)
        dom_p = comp_mean.max(axis=1)
        entropy = float(-(mix * np.log(mix + 1e-12)).sum())
        effective = float(np.exp(entropy))
        lines.append(
            f"  effective #components = {effective:.2f} / {components} "
            f"(weight entropy {entropy:.3f})"
        )
        lines.append("  comp   weight   sum(alpha)  dom_token  dom_prob")
        for c in np.argsort(-mix):
            lines.append(
                f"  {c:4d}  {mix[c]:8.4f}  {totals[c]:9.3f}  "
                f"{dom[c]:8d}  {dom_p[c]:8.4f}"
            )
    return "\n".join(lines)


def main() -> None:
    """Entry point: parse arguments, fit the prior, and save the weights."""
    args = parse_args()

    alphabet, dim = ALPHABETS[args.alphabet]
    score_counts = args.score == "counts"
    # Henikoff weighting reshapes each column distribution, so it matters in
    # both scoring modes. The Neff rescaling on top of it is a per-family
    # scalar that cancels once columns are normalized to probabilities.
    use_neff = not args.no_neff
    columns, family_of_column, family_accessions = collect_columns(
        input_dir=args.input_dir,
        pattern=args.pattern,
        fmt=args.fmt,
        alphabet=alphabet,
        min_count=args.min_count,
        min_occupancy=args.min_occupancy,
        fit_dim=dim,
        neff=use_neff,
        neff_scaling=score_counts,
        neff_target_bits=args.neff_target_bits,
        neff_prior_conc=args.neff_prior_conc,
    )
    # The Dirichlet-multinomial marginal scores counts directly; the Dirichlet
    # density needs probability vectors.
    if not score_counts:
        columns = columns / columns.sum(axis=1, keepdims=True)
    if not use_neff:
        weighting = ""
    elif score_counts:
        weighting = " with Henikoff weighting and Neff rescaling"
    else:
        weighting = " with Henikoff weighting"
    print(f"Scoring columns as {args.score}{weighting}.")

    rng = np.random.default_rng(args.seed)
    num_total = columns.shape[0]
    if args.val_fraction == 0 and args.num_runs > 1:
        raise ValueError(
            "--val-fraction 0 disables validation, but --num-runs "
            f"{args.num_runs} needs a held-out set to select the best run; "
            "use --num-runs 1 or a positive --val-fraction."
        )
    # Validation only serves model selection between runs.
    validate = args.num_runs > 1
    if not validate:
        if args.max_columns is not None and num_total > args.max_columns:
            if args.clans is None:
                idx = rng.choice(
                    num_total, size=args.max_columns, replace=False
                )
            else:
                clan_of_family = load_clan_of_family(
                    args.clans, family_accessions
                )
                idx = sample_clan_family_columns(
                    family_of_column,
                    clan_of_family,
                    np.arange(num_total),
                    args.max_columns,
                    rng,
                )
            columns = columns[idx]
        train_columns = columns
        val_columns = None
        print(f"Training on all {train_columns.shape[0]} columns.")
    elif args.clans is None:
        # Flat, un-balanced pool: optionally cap, then split by permutation.
        if args.max_columns is not None and num_total > args.max_columns:
            idx = rng.choice(num_total, size=args.max_columns, replace=False)
            columns = columns[idx]
            print(f"Subsampled to {columns.shape[0]} columns.")
        # The same split is reused across runs so their scores are comparable.
        perm = rng.permutation(columns.shape[0])
        num_val = int(round(args.val_fraction * columns.shape[0]))
        if not 0 < num_val < columns.shape[0]:
            raise ValueError(
                f"--val-fraction {args.val_fraction} yields {num_val} "
                f"validation columns out of {columns.shape[0]}; choose a "
                f"value in (0, 1)."
            )
        val_columns = columns[perm[:num_val]]
        train_columns = columns[perm[num_val:]]
    else:
        # Clan-balanced sampling. Split into disjoint train/val pools first (so
        # the with-replacement draws never share a column), then draw each set
        # hierarchically.
        clan_of_family = load_clan_of_family(args.clans, family_accessions)
        num_val = int(round(args.val_fraction * num_total))
        if not 0 < num_val < num_total:
            raise ValueError(
                f"--val-fraction {args.val_fraction} yields {num_val} "
                f"validation columns out of {num_total}; choose a value in "
                f"(0, 1)."
            )
        perm = rng.permutation(num_total)
        val_pool = perm[:num_val]
        train_pool = perm[num_val:]
        if args.max_columns is not None:
            n_total = args.max_columns
        else:
            n_total = num_total
        n_val = int(round(args.val_fraction * n_total))
        n_train = n_total - n_val
        if not 0 < n_val < n_total:
            raise ValueError(
                f"--val-fraction {args.val_fraction} and --max-columns "
                f"{args.max_columns} yield {n_val} validation columns; choose "
                f"compatible values."
            )
        train_idx = sample_clan_family_columns(
            family_of_column, clan_of_family, train_pool, n_train, rng
        )
        val_idx = sample_clan_family_columns(
            family_of_column, clan_of_family, val_pool, n_val, rng
        )
        train_columns = columns[train_idx]
        val_columns = columns[val_idx]

    if val_columns is not None:
        print(
            f"Split into {train_columns.shape[0]} training and "
            f"{val_columns.shape[0]} validation columns."
        )

    # Frozen background anchor for the MAP prior: the data mean as logits.
    use_map_prior = not args.no_map_prior
    mu = np.clip(train_columns.mean(axis=0), 1e-4, None)
    mu = mu / mu.sum()
    background_init = np.log(mu).astype(np.float64)
    num_examples = train_columns.shape[0]

    # Run several trainings from different initializations and keep the model
    # with the lowest validation loss.
    best_val_loss = np.inf
    best_prior_weights: list[np.ndarray] | None = None
    for run in range(args.num_runs):
        run_seed = args.seed + run
        tf.random.set_seed(run_seed)
        print(
            f"\n=== Training run {run + 1}/{args.num_runs} "
            f"(seed {run_seed}) ==="
        )
        initializer = make_initializer(
            train_columns, args.components, run_seed, args.init
        )
        model = build_trainable_model(
            initializer, dim, args.components,
            background_init=background_init,
            num_examples=num_examples,
            use_map_prior=use_map_prior,
            freeze_hyperparams=args.freeze_hyperparams,
            score_counts=score_counts,
        )
        model.summary()
        val_loss = train(
            model=model,
            train_columns=train_columns,
            val_columns=val_columns,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            seed=run_seed,
            verbose=int(args.verbose),
        )
        print(summarize_prior(model.prior, dim, args.components))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # The MAP-prior hyperparameters are a training-only artifact and
            # are not part of the saved format.
            best_prior_weights = model.prior.get_weights()
            if validate:
                print(f"New best model (validation loss {val_loss:.4f}).")

    assert best_prior_weights is not None  # num_runs >= 1 guarantees a fit
    print(
        f"\nBest {'validation' if validate else 'training'} mean column "
        f"log-likelihood over {args.num_runs} run(s): {-best_val_loss:.4f}"
    )
    # Rebuild a clean, prior-only model and load the best prior kernel, which
    # keeps the saved layout identical to what load_dirichlet expects.
    best_model = make_dirichlet_model(dim=dim, components=args.components)
    best_model.layers[1].set_weights(best_prior_weights)

    save_dim = dim
    if args.extended_alphabet:
        save_dim = len(AA_ALPHABET)
        initializer = extend_concentrations(
            best_model.layers[1].matrix()[0, 0].numpy(),
            dim, args.components, save_dim,
        )
        best_model = make_dirichlet_model(
            initializer=initializer, dim=save_dim, components=args.components
        )
        print(
            f"Extended the fitted {dim} concentrations per component to "
            f"{save_dim} with alpha = 1 for '{AA_ALPHABET[dim:]}'."
        )
    print(summarize_prior(best_model.layers[1], save_dim, args.components))

    if args.output is not None:
        output_path = Path(args.output)
    else:
        output_path = WEIGHTS_DIR / f"{args.name}_{args.components}.weights.h5"
    save(best_model, output_path, save_dim, args.components)


if __name__ == "__main__":
    main()
