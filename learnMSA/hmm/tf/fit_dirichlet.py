"""Fit a Dirichlet prior (single or mixture) to multiple sequence alignments.

This script fits a :class:`hidten.tf.prior.dirichlet.TFDirichletPrior` by
maximum likelihood to the column distributions of a directory of multiple
sequence alignments and saves the resulting weights in the format used by the
learnMSA HMM (see :mod:`learnMSA.hmm.tf.util`).

The observation alphabet is configurable, so the same script can fit an
amino-acid prior (default) as well as a 3Di / structural-token prior by passing
a different ``--alphabet``.

Defaults follow the spirit of HMMER3's ``esl-mixdchlet fit``: a mixture of
Dirichlets (9 components by default) fit to conserved alignment columns.

The columns are split into a training and a held-out validation set (80/20 by
default). Several trainings are run from different random initializations (10
by default), each stopped early once the validation loss stops improving, and
the model with the best validation log-likelihood is saved. With ``--num-runs
1`` there is no model selection, so validation is skipped and all columns are
used for training; ``--val-fraction 0`` requires ``--num-runs 1``.

By default every column contributes equally, so large families dominate. Pass
``--clans`` with a Pfam-style clan TSV to instead draw the training columns
hierarchically (clan uniform -> family uniform -> column uniform), which removes
the bias of large families (and large clans) from the fitted prior.

With the default count-based scoring (``--score counts``) the per-column counts
are additionally reweighted HMMER-style so that redundant sequences within a
family do not inflate the effective sample size: sequences are down-weighted by
Henikoff & Henikoff (1994) position-based weights and each family's total weight
is rescaled to an entropy-targeted effective sequence number (Neff). Pass
``--no-neff`` to fall back to raw per-sequence counts.

Example
-------
Fit a 9-component amino-acid mixture and overwrite the shipped prior::

    python -m learnMSA.hmm.util.fit_dirichlet /path/to/msas -c 9

Fit a 9-component amino-acid mixture with clan-balanced sampling::

    python -m learnMSA.hmm.util.fit_dirichlet /path/to/msas -c 9 \\
        --pattern "*.fasta" --clans Pfam-A.clans.tsv

Fit a 3Di structural-token prior::

    python -m learnMSA.hmm.util.fit_dirichlet /path/to/3di_msas \\
        --alphabet ACDEFGHIKLMNPQRSTVWY --replace-with-x "" \\
        --name pfam_35_3Di -c 9
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

from learnMSA.hmm.tf.util import make_dirichlet_model
from learnMSA.util.aligned_dataset import AlignedDataset
from learnMSA.util.sequence_dataset import SequenceDataset

# Default amino-acid alphabet without the trailing gap symbol.
DEFAULT_ALPHABET: str = SequenceDataset._default_alphabet[:-1]

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
        "-c", "--components",
        type=int,
        default=9,
        help="Number of mixture components (1 fits a single Dirichlet).",
    )
    parser.add_argument(
        "--alphabet",
        type=str,
        default=DEFAULT_ALPHABET,
        help="Non-gap observation tokens. The prior dimension is its length.",
    )
    parser.add_argument(
        "--replace-with-x",
        type=str,
        default="BZJ",
        help="Symbols replaced by X during parsing (pass '' for 3Di).",
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
        help="Minimum total non-gap residue count required to keep a column.",
    )
    parser.add_argument(
        "--min-occupancy",
        type=float,
        default=0.5,
        help="Minimum column occupancy (fraction of non-gap residues) required "
             "to keep a column. Combined with --min-count: a column is kept "
             "only if it satisfies both thresholds.",
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
        help="Fraction of columns held out for validation / model selection. "
             "Set to 0 to train on all columns without validation.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Stop a run after this many epochs without improvement of the "
             "validation loss (or the training loss if no validation is done).",
    )
    parser.add_argument(
        "--no-map-prior",
        action="store_true",
        help="Disable the Dirichlet-Process MAP prior and fit by pure maximum "
             "likelihood. By default a MAP prior (Nguyen et al. 2013) shrinks "
             "each component toward the background token distribution and keeps "
             "the mixture weights from collapsing.",
    )
    parser.add_argument(
        "--freeze-hyperparams",
        action="store_true",
        help="Freeze the MAP-prior hyperparameters (gamma, beta, lambda) at "
             "their initial values instead of estimating them by empirical "
             "Bayes. The background anchor is always frozen to the data mean.",
    )
    parser.add_argument(
        "--init",
        type=str,
        default="data",
        choices=["data", "random_normal"],
        help="Initialization scheme for the prior parameters. 'data' anchors "
             "the concentrations to the background token frequencies; "
             "'random_normal' draws them from a normal distribution instead "
             "(log-normal concentrations, softmax mixture weights).",
    )
    parser.add_argument(
        "--score",
        type=str,
        default="counts",
        choices=["counts", "probabilities"],
        help="Observation model used to score columns. 'counts' (default) "
             "scores raw per-column token counts with the proper "
             "Dirichlet-multinomial marginal (Sjolander et al. 1996), which "
             "accounts for the per-column sample size. 'probabilities' scores "
             "normalized column distributions with the Dirichlet density.",
    )
    parser.add_argument(
        "--no-neff",
        action="store_true",
        help="Disable HMMER-style Neff sequence weighting. By default (with "
             "--score counts) sequences are down-weighted by Henikoff "
             "position-based weights and each family's total weight is rescaled "
             "to an entropy-targeted effective sequence number (Neff), so "
             "redundant families no longer inflate the per-column sample size. "
             "Has no effect with --score probabilities.",
    )
    parser.add_argument(
        "--neff-target-bits",
        type=float,
        default=0.59,
        help="Per-column mean-relative-entropy target (in bits) for the Neff "
             "bisection (HMMER's protein default; 0.45 for nucleotides). Note "
             "the target is calibrated for a 20-letter amino-acid alphabet.",
    )
    parser.add_argument(
        "--neff-prior-conc",
        type=float,
        default=10.0,
        help="Total concentration of the data-anchored reference single "
             "Dirichlet (alpha = background * concentration) used only inside "
             "the Neff entropy target. The target depends only on the ratio "
             "Neff/concentration, so this scales the resulting Neff linearly: "
             "doubling it doubles every family's Neff. The self-consistent "
             "choice is the total concentration of the prior being fit; the "
             "default is that fixed point for Pfam seed columns at "
             "--min-occupancy 0.5 --min-count 30. Set it too low and columns "
             "carry ~1 effective count each, which makes the concentration "
             "unidentifiable and the fit diverge; too high and families clamp "
             "at their actual sequence count (watch the Neff diagnostic).",
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
        help="Optional cap on the number of training columns (for memory). With "
             "--clans this is the total number of columns drawn by the "
             "hierarchical sampler (default: the number of collected columns).",
    )
    parser.add_argument(
        "--clans",
        type=str,
        default=None,
        help="Optional Pfam-style clans TSV (tab-separated; col0=family "
             "accession e.g. PF00001, col1=clan accession e.g. CL0192, empty if "
             "none). When given, training columns are drawn hierarchically to "
             "remove large-family bias: a clan is drawn uniformly, then a family "
             "within it, then a column within the family. Families with no clan "
             "(or absent from the TSV) each form their own singleton clan. The "
             "family accession is matched from the file name up to the first dot "
             "(e.g. PF01024.25.fasta -> PF01024).",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output weight file. Defaults to the shipped weights directory.",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------
# HMMER-style sequence weighting and effective-sequence-number (Neff) targeting.
#
# The count-based fit treats every sequence as an independent observation, so a
# family with many near-identical sequences inflates the per-column sample size
# the Dirichlet-multinomial sees. HMMER counters this in two stages: Henikoff &
# Henikoff (1994) position-based weights down-weight redundant sequences, and an
# entropy-based bisection rescales a family's total weight to an effective
# sequence number (Neff) whose columns carry an information-appropriate sample
# size. Both stages run per family before the counts feed the fit.
# --------------------------------------------------------------------------


def emissions_single_dirichlet(
    counts: np.ndarray, alpha: np.ndarray
) -> np.ndarray:
    """Mean-posterior emission probabilities under a single Dirichlet.

    Args:
        counts: Weighted residue counts of shape ``(M, dim)``.
        alpha: Concentration parameters of shape ``(dim,)``, broadcast over the
            ``M`` columns.

    Returns:
        Posterior mean probabilities of shape ``(M, dim)`` (each row sums to 1).
    """
    post = counts + alpha
    return post / post.sum(axis=1, keepdims=True)


def henikoff_pb_weights(msa: np.ndarray, dim: int, gap: int) -> np.ndarray:
    """Henikoff & Henikoff (1994) position-based sequence weights.

    Purely per-column and free of any pairwise-similarity computation (HMMER's
    default weighting). In a column with ``r`` distinct residue types, a sequence
    carrying a residue of type ``a`` seen ``c(a)`` times receives ``1 / (r *
    c(a))`` from that column; gaps contribute nothing. The per-sequence sums are
    normalized to relative weights summing to 1.

    Args:
        msa: Integer-encoded alignment of shape ``(num_seq, L)``; tokens
            ``0..dim-1`` are residues and ``gap`` marks a gap.
        dim: Alphabet size (number of non-gap tokens).
        gap: Token value marking a gap.

    Returns:
        Relative weights of shape ``(num_seq,)`` summing to 1. An empty or
        all-gap alignment yields uniform weights.
    """
    num_seq = msa.shape[0]
    w = np.zeros(num_seq, dtype=np.float64)
    for j in range(msa.shape[1]):
        col = msa[:, j]
        obs = col != gap
        if not obs.any():
            continue
        counts = np.bincount(col[obs], minlength=dim)  # c(a) per residue type
        r = np.count_nonzero(counts)                   # distinct types in column
        # A residue of type a in this column contributes 1 / (r * c(a)).
        w[obs] += 1.0 / (r * counts[col[obs]])
    total = w.sum()
    return w / total if total > 0 else np.full(num_seq, 1.0 / num_seq)


def weighted_counts(
    msa: np.ndarray, rel_w: np.ndarray, dim: int, columns: np.ndarray, gap: int
) -> np.ndarray:
    """Accumulate relative-weight residue counts at the given columns.

    Because ``rel_w`` sums to 1, the returned vectors are the *base* counts for
    :func:`target_neff`: the effective counts at a chosen ``neff`` are simply
    ``neff * weighted_counts(...)``.

    Args:
        msa: Integer-encoded alignment of shape ``(num_seq, L)``.
        rel_w: Per-sequence relative weights of shape ``(num_seq,)`` (sum to 1).
        dim: Alphabet size (number of non-gap tokens).
        columns: Column indices to accumulate, shape ``(M,)``.
        gap: Token value marking a gap.

    Returns:
        Weighted counts of shape ``(len(columns), dim)``. Each row sums to at
        most 1 (the non-gap weight fraction of that column).
    """
    counts = np.zeros((len(columns), dim), dtype=np.float64)
    for m, j in enumerate(columns):
        col = msa[:, j]
        obs = col != gap
        np.add.at(counts[m], col[obs], rel_w[obs])
    return counts


def mean_relative_entropy(p: np.ndarray, bg: np.ndarray) -> float:
    """Mean over columns of the relative entropy ``KL(p_col || bg)`` in bits.

    Args:
        p: Per-column probability vectors of shape ``(M, dim)`` (each summing
            to 1).
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

    Mirrors HMMER's entropy weighting (``--eent``): the per-column mean relative
    entropy of the posterior-mean emissions increases monotonically with the
    effective sequence number ``neff``, so a bisection finds the ``neff`` at
    which it equals ``target_bits``. ``base_counts`` are relative-weight counts
    (from :func:`weighted_counts`), so the counts at a given ``neff`` are linear,
    ``neff * base_counts``. If even ``neff = n_actual`` cannot reach the target
    (the family is not conserved enough), ``neff = n_actual`` is kept, matching
    HMMER's behavior.

    Note that the emissions depend on ``alpha`` and ``neff`` only through their
    ratio, so scaling ``alpha`` scales the returned ``neff`` by exactly the same
    factor (up to the ``n_actual`` clamp). The total concentration of ``alpha``
    is therefore a linear gain on the effective sample size, not a smoothing
    strength: the extra smoothing it adds is exactly undone by the bisection.

    Args:
        base_counts: Relative-weight counts of shape ``(M, dim)`` at the family's
            match columns (``rel_w`` summed to 1).
        alpha: Reference single-Dirichlet concentrations of shape ``(dim,)``.
        bg: Background probability vector of shape ``(dim,)``.
        target_bits: Target mean relative entropy per column, in bits.
        n_actual: Upper bound on ``neff`` (the actual number of sequences);
            defaults to the number of match columns if not given.
        lo: Lower bound of the bisection.
        iters: Number of bisection iterations.

    Returns:
        A tuple ``(neff, emissions)`` of the effective sequence number and the
        posterior-mean emissions of shape ``(M, dim)`` evaluated at that ``neff``.
    """
    hi = float(n_actual) if n_actual is not None else float(base_counts.shape[0])

    def re_at(neff: float) -> tuple[float, np.ndarray]:
        p = emissions_single_dirichlet(neff * base_counts, alpha)
        return mean_relative_entropy(p, bg), p

    # Monotonicity: mean relative entropy grows with neff. If the maximum
    # attainable value already falls short, keep the full sequence count.
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
    replace_with_x: str,
    min_count: int,
    min_occupancy: float,
    neff: bool = False,
    neff_target_bits: float = 0.59,
    neff_prior_conc: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Collect per-column token counts from a directory of alignments.

    Each retained column is turned into a raw count vector over the ``dim =
    len(alphabet)`` non-gap tokens (callers may normalize it to a probability
    vector if needed). A column is kept only if it satisfies both thresholds:
    its total non-gap residue count is at least ``min_count`` and its occupancy
    (fraction of non-gap residues) is at least ``min_occupancy``.

    With ``neff`` enabled the raw counts are replaced by HMMER-style effective
    counts: each family's sequences are down-weighted by Henikoff position-based
    weights (:func:`henikoff_pb_weights`) and its total weight is rescaled to an
    entropy-targeted effective sequence number (:func:`target_neff`). The column
    selection is unchanged -- the same kept columns are emitted -- but their
    count vectors reflect effective, not raw, sequence numbers. The reference
    background used by the entropy target is the global marginal token
    distribution over all kept columns, so the routine stays alphabet-agnostic.

    The provenance of each column is also returned (which family/file it came
    from), so callers can group columns by family and clan for balanced sampling.

    Args:
        input_dir: Directory containing the alignment files.
        pattern: Glob pattern selecting files within ``input_dir``.
        fmt: File format passed to :class:`AlignedDataset`.
        alphabet: Non-gap observation tokens; a gap is appended for parsing.
        replace_with_x: Symbols replaced by ``X`` during parsing.
        min_count: Minimum total non-gap residue count required to keep a column.
        min_occupancy: Minimum column occupancy (non-gap fraction) required to
            keep a column.
        neff: If True, emit HMMER-style effective counts (Henikoff weighting plus
            entropy-based Neff targeting) instead of raw per-sequence counts.
        neff_target_bits: Per-column mean-relative-entropy target (in bits) for
            the Neff bisection; ignored when ``neff`` is False.
        neff_prior_conc: Total concentration of the data-anchored reference single
            Dirichlet (``alpha = bg * neff_prior_conc``) used by the entropy
            target; ignored when ``neff`` is False. The target depends only on
            the ratio ``neff / neff_prior_conc``, so this is a linear gain on the
            resulting Neff rather than a smoothing strength (see
            :func:`target_neff`). The self-consistent value is the total
            concentration of the prior being fit.

    Returns:
        A tuple ``(columns, family_of_column, family_accessions)`` where
        ``columns`` has shape ``(M, dim)`` with one count vector per
        column, ``family_of_column`` has shape ``(M,)`` and holds the family
        index of each column, and ``family_accessions`` maps each family index
        to its accession (the file name up to the first dot, e.g. ``PF01024``).
        Only files that contributed at least one kept column are indexed.
    """
    dim = len(alphabet)
    parse_alphabet = alphabet + "-"
    gap = dim  # gap token index (parse_alphabet appends the gap symbol)
    files = sorted(p for p in Path(input_dir).glob(pattern) if p.is_file())
    if not files:
        raise ValueError(
            f"No files matching '{pattern}' found in '{input_dir}'."
        )

    distributions: list[np.ndarray] = []
    family_ids: list[np.ndarray] = []
    family_accessions: list[str] = []
    family_num_seq: list[int] = []  # sequences per family (for Neff targeting)
    bg_accum = np.zeros(dim, dtype=np.float64)  # global marginal token counts
    for path in files:
        try:
            data = AlignedDataset(
                filepath=path,
                fmt=fmt,
                alphabet=parse_alphabet,
                replace_with_x=replace_with_x,
            )
        except Exception as err:  # noqa: BLE001 - skip unreadable files
            #print(f"Skipping '{path}': {err}")
            continue

        matrix = data.msa_matrix  # (num_seq, L)
        num_seq = matrix.shape[0]
        # Per-column counts over the dim non-gap token indices (gap == dim).
        counts = np.stack(
            [np.sum(matrix == a, axis=0) for a in range(dim)], axis=1
        ).astype(np.float64)  # (L, dim)
        totals = counts.sum(axis=1)  # (L,) non-gap residues per column
        occupancy = totals / max(num_seq, 1)  # (L,) non-gap fraction
        # A column must satisfy both the count and the occupancy threshold.
        keep = (totals >= max(min_count, 1)) & (occupancy >= min_occupancy)
        if not np.any(keep):
            continue
        # The global background is the marginal over the kept (fitted) columns.
        bg_accum += counts[keep].sum(axis=0)
        if neff:
            # Henikoff-weighted base counts at the kept columns. These are scaled
            # to the family's effective sequence number below, once the global
            # background needed by the entropy target is known.
            rel_w = henikoff_pb_weights(matrix, dim, gap=gap)
            kept = weighted_counts(matrix, rel_w, dim, np.where(keep)[0], gap=gap)
        else:
            kept = counts[keep]
        # Family accession is the file name up to the first dot (strips the
        # version and extension, e.g. PF01024.25.fasta -> PF01024).
        family_index = len(family_accessions)
        family_accessions.append(path.name.split(".")[0])
        distributions.append(kept)
        family_ids.append(np.full((kept.shape[0],), family_index, dtype=np.int64))
        family_num_seq.append(num_seq)

    if not distributions:
        raise ValueError(
            "No columns passed the filters. Try lowering --min-count or "
            "--min-occupancy."
        )

    if neff:
        # Data-anchored reference for the entropy target: the global marginal
        # token distribution and a single Dirichlet alpha = bg * concentration.
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
        # Families that could not reach the entropy target keep Neff = num_seq.
        # A large share means --neff-prior-conc is too high for this data and the
        # entropy target has stopped discounting redundancy.
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
        f"alignment(s) (alphabet size {dim})."
    )
    return columns, family_of_column, family_accessions


def load_clan_of_family(
    clans_path: str, family_accessions: list[str]
) -> np.ndarray:
    """Map each family to a clan index; clanless/absent families are singletons.

    The clans file is a Pfam-style TSV: column 0 is a family accession (e.g.
    ``PF00001``) and column 1 is its clan accession (e.g. ``CL0192``, empty when
    the family has no clan). Only these two columns are used. A family whose clan
    field is empty, or which does not appear in the file at all, is assigned its
    own singleton clan so it competes on equal footing in a uniform clan draw.

    Args:
        clans_path: Path to the tab-separated clans file.
        family_accessions: Family accession per family index (see
            :func:`collect_columns`).

    Returns:
        Array of shape ``(num_families,)`` with a clan index per family; clan
        indices are contiguous starting at 0.
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

    # Map the (string) clan labels to contiguous integer indices.
    clan_to_index: dict[str, int] = {}
    clan_of_family = np.empty((len(clan_labels),), dtype=np.int64)
    for i, label in enumerate(clan_labels):
        clan_of_family[i] = clan_to_index.setdefault(label, len(clan_to_index))

    print(
        f"Loaded clans for {len(family_accessions)} famil(y/ies): "
        f"{len(clan_to_index)} distinct clan(s), of which {num_singletons} are "
        f"singleton clans (family with no clan or absent from the TSV)."
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

    Realizes the generative process (with replacement): draw a clan uniformly,
    then a family uniformly within that clan, then a column uniformly within that
    family. Only the columns in ``subset_idx`` (and hence only the families and
    clans present within that subset) are eligible, so the same routine can be
    used for a disjoint training and validation pool.

    Args:
        family_of_column: Family index per column, shape ``(M,)``.
        clan_of_family: Clan index per family, shape ``(num_families,)``.
        subset_idx: Indices (into ``family_of_column``) of the eligible columns.
        n_samples: Number of columns to draw.
        rng: Random generator for the draws.

    Returns:
        Array of ``n_samples`` column indices (into ``family_of_column``), drawn
        with replacement.
    """
    subset_idx = np.asarray(subset_idx)
    if subset_idx.size == 0:
        raise ValueError("Cannot sample from an empty column subset.")

    # Group the eligible columns by their family. Sorting by family index lays
    # the columns of each family out contiguously so a family is a flat slice.
    families = family_of_column[subset_idx]
    order = np.argsort(families, kind="stable")
    family_col_indices = subset_idx[order]  # original column indices, by family
    fam_sorted = families[order]
    present_families, first_pos, family_col_count = np.unique(
        fam_sorted, return_index=True, return_counts=True
    )
    family_col_start = first_pos  # start of each present family's slice

    # Group the present families by their clan, laid out contiguously likewise.
    clans = clan_of_family[present_families]
    clan_order = np.argsort(clans, kind="stable")
    # For each clan slot (in clan order) we store its index into the present-
    # family arrays, so drawing a family slot yields a present-family position.
    clan_fam_indices = clan_order
    clan_sorted = clans[clan_order]
    _, clan_first_pos, clan_fam_count = np.unique(
        clan_sorted, return_index=True, return_counts=True
    )
    clan_fam_start = clan_first_pos
    num_clans = clan_fam_start.shape[0]

    # Stage 1: draw a clan uniformly.
    c = rng.integers(0, num_clans, size=n_samples)
    # Stage 2: draw a family uniformly within the clan.
    fam_slot = clan_fam_start[c] + np.floor(
        rng.random(n_samples) * clan_fam_count[c]
    ).astype(np.int64)
    present_family_pos = clan_fam_indices[fam_slot]
    # Stage 3: draw a column uniformly within the family.
    col_slot = family_col_start[present_family_pos] + np.floor(
        rng.random(n_samples) * family_col_count[present_family_pos]
    ).astype(np.int64)
    return family_col_indices[col_slot]


def random_normal_initializer(
    dim: int, components: int, seed: int
) -> np.ndarray:
    """Build a flat initializer by drawing from a normal distribution.

    A simple, data-agnostic alternative to :func:`make_initializer`'s
    data-anchored scheme. The initializer values must be strictly positive
    (concentrations are later mapped through ``inverse_softplus`` and mixture
    weights through ``log``), so the normal draws are mapped to valid values:
    concentrations are exponentiated (log-normal) and mixture weights are
    passed through a softmax. This also breaks the symmetry between mixture
    components, which is required for them to specialize.

    Args:
        dim: Dimension of the categorical distribution.
        components: Number of mixture components.
        seed: Random seed for the draws.

    Returns:
        Flat initializer ``[conc_1, ..., conc_C, mixture_weights]`` (the
        mixture weights are omitted for a single component).
    """
    rng = np.random.default_rng(seed)
    # Log-normal concentrations: strictly positive, centered near 1.
    concentrations = np.exp(rng.normal(0.0, 1.0, size=(components, dim)))
    if components == 1:
        return concentrations.reshape(-1).astype(np.float64)
    # Softmax of gaussian logits: positive mixture weights summing to 1.
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

    With ``init="data"`` (default) the concentrations are anchored to the
    background mean of the data so that they are strictly positive and reflect
    the marginal token frequencies (an appropriate starting point for Dirichlet
    parameters). For mixtures the symmetry between components must be broken,
    otherwise identical components would receive identical gradients and never
    specialize. This is achieved by (a) drawing a different total concentration
    per component, (b) applying per-token multiplicative noise, and (c)
    randomizing the mixture weights.

    With ``init="random_normal"`` the parameters are instead drawn from a normal
    distribution (see :func:`random_normal_initializer`), ignoring the data.

    Different ``seed`` values therefore yield genuinely different starting
    points, which is what the multi-run model selection relies on.

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
    # Background mean, clipped away from zero so concentrations stay positive.
    mu = np.clip(columns.mean(axis=0), 1e-4, None)
    mu = mu / mu.sum()

    if components == 1:
        total_concentration = 10.0
        return (mu * total_concentration).astype(np.float64)

    # Per-component total concentration gives components different peakiness.
    total_concentration = rng.uniform(5.0, 20.0, size=(components, 1))
    # Per-token multiplicative noise breaks the symmetry between components.
    noise = rng.uniform(0.5, 1.5, size=(components, dim))
    concentrations = mu[np.newaxis, :] * total_concentration * noise  # (C, dim)
    # Randomize the mixture weights (Dirichlet on the simplex) rather than
    # starting from a uniform mixture, which would also be a symmetric point.
    mixture_weights = rng.dirichlet(np.full((components,), 2.0))
    return np.concatenate(
        [concentrations.reshape(-1), mixture_weights]
    ).astype(np.float64)


# Probability floor keeping the MAP-prior densities and their gradients finite
# when mixture components die: with many components the optimizer drives unused
# components' concentrations and mixture weights toward zero, where an unguarded
# log density (and its gradient) would evaluate to NaN.
_PROB_FLOOR: float = 1e-8


def _dirichlet_log_pdf(p: tf.Tensor, alpha: tf.Tensor) -> tf.Tensor:
    """Log density ``log Dir(p | alpha)``, summed over the last axis.

    ``p`` and ``alpha`` broadcast against each other over any leading batch
    dimensions; the final axis is the categorical dimension ``D``.

    ``p`` is floored at :data:`_PROB_FLOOR` so the density and its gradient stay
    finite when an entry underflows to zero (a dead mixture component); for a
    healthy fit the entries are well above the floor and the value is unchanged.

    Args:
        p: Probability vectors of shape ``(..., D)`` (each summing to 1).
        alpha: Concentration parameters of shape ``(..., D)``.

    Returns:
        Log densities with the batch shape of the broadcast of ``p`` and
        ``alpha`` (the ``D`` axis reduced away).
    """
    p = tf.maximum(p, _PROB_FLOOR)
    log_z = tf.math.lbeta(alpha)
    return tf.reduce_sum(tf.math.xlogy(alpha - 1.0, p), axis=-1) - log_z


class DirichletMAPRegularizer(tf.keras.layers.Layer):
    """Training-only Dirichlet-Process MAP prior over a ``TFDirichletPrior``.

    Reproduces the MAP objective of Nguyen et al. 2013, *"Dirichlet Mixtures,
    the Dirichlet Process, and the Structure of Protein Space"* (also the family
    of methods behind HMMER's ``esl-mixdchlet``). The log prior is the sum of
    three normalized-density terms evaluated on the fitted prior's parameters:

    * ``sum_alpha_prior`` -- Exponential(``lambda``) prior on each component's
      total concentration ``sum(alpha_c)``, guarding against over-concentration.
    * ``mix_prior`` -- symmetric Dirichlet(``gamma / C``) prior on the mixture
      weights, keeping components from collapsing (only for ``C > 1``).
    * ``comp_prior`` -- Dirichlet(``beta * background``) prior on each normalized
      component mean, shrinking components toward the background distribution.

    The hyperparameters ``gamma``, ``beta`` and ``lambda`` are trainable by
    default (empirical Bayes / type-II MLE); each prior term is a proper
    normalized density, so its log-normalizer counter-balances the fit and the
    regularizer cannot trivially weaken itself. The ``background`` anchor is
    always frozen to the data mean so the shrinkage target does not drift.

    These weights live only in the fitting model -- the inference-time
    :class:`TFDirichletPrior` (and its saved weight layout) is untouched.

    Args:
        dim: Dimension of the categorical distribution.
        components: Number of mixture components.
        background_init: Logits initializing the (frozen) background anchor;
            ``softmax(background_init)`` should equal the background mean.
        trainable_hyperparams: If True, estimate ``gamma``, ``beta``, ``lambda``;
            otherwise hold them fixed at their initial values.
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
        # Background is the shrinkage anchor: always frozen to the data mean.
        self.background_kernel = self.add_weight(
            shape=(self.dim,),
            initializer=tf.constant_initializer(self.background_init),
            name="background_kernel", trainable=False,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Identity passthrough.

        The layer holds no data-dependent computation; it is inserted into the
        functional graph purely so Keras tracks its hyperparameter weights as
        part of the model (and hence exposes them to the optimizer). The MAP
        penalty itself is added in the overridden ``compute_loss`` via
        :meth:`log_prior`, which keeps it out of the validation loss.
        """
        return inputs

    def log_prior(self, prior) -> tf.Tensor:
        """Total MAP log prior evaluated on ``prior``'s current parameters."""
        matrix = prior.matrix()  # (H, Q, P)
        gamma = tf.math.softplus(self.gamma_kernel)
        beta = tf.math.softplus(self.beta_kernel)
        lam = tf.math.softplus(self.lambda_kernel)
        background = tf.nn.softmax(self.background_kernel)

        if self.components == 1:
            conc = matrix  # (H, Q, D)
            sum_alpha = tf.reduce_sum(conc, axis=-1)  # (H, Q)
            # Guard the denominator: a dead component's concentrations underflow
            # to zero, and an unguarded 0/0 would make the mean (and the whole
            # loss) NaN. The floored mean is then 0, handled by _dirichlet_log_pdf.
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

        # Exponential prior on the per-component total concentration.
        sum_alpha_prior = tf.reduce_sum(tf.math.log(lam) - lam * sum_alpha)
        # Dirichlet prior pulling each component mean toward the background.
        comp_dist = background * beta  # (D,)
        comp_prior = tf.reduce_sum(_dirichlet_log_pdf(means, comp_dist))

        return sum_alpha_prior + mix_prior + comp_prior


class DirichletTrainingModel(tf.keras.Model):
    """Trainable model scoring observations with a ``TFDirichletPrior``.

    Depending on ``score_counts`` the observations are scored either as raw
    count vectors with the Dirichlet-multinomial marginal
    (:meth:`TFDirichletPrior.dirichlet_multinomial_scores`, Sjolander et al.
    1996) or as probability vectors with the Dirichlet density
    (:meth:`TFDirichletPrior.prior_scores`). The explicit scoring method is
    invoked here rather than through the prior's ``call`` so the inference-time
    prior keeps a single, unambiguous behavior.

    When a MAP ``regularizer`` is given, its Dirichlet-Process penalty is added
    to the training loss only, so the validation loss stays a pure held-out
    log-likelihood (keeping early stopping and cross-run model selection
    comparable). The penalty is scaled by ``1 / num_examples`` so the objective
    is the per-example-averaged MAP density matching the per-batch mean
    log-likelihood used as the loss.

    The prior and the (optional) regularizer are held as attributes, so Keras
    tracks their weights and the optimizer updates them.

    Args:
        prior: The (already built) trainable ``TFDirichletPrior``.
        score_counts: If True, score count vectors with the Dirichlet-multinomial
            marginal; otherwise score probability vectors with the density.
        regularizer: Optional MAP-prior layer owning the hyperparameter weights.
        num_examples: Number of training columns (``N`` in the MAP scaling);
            required when ``regularizer`` is given.
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
        self, x=None, y=None, y_pred=None, sample_weight=None, training=True
    ):
        loss = super().compute_loss(x, y, y_pred, sample_weight, training)
        if training and self.map_regularizer is not None:
            loss = loss - self.map_regularizer.log_prior(self.prior) * self._inv_n
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
        background_init: Logits for the frozen background anchor (required when
            ``use_map_prior`` is True).
        num_examples: Number of training columns for the MAP scaling (required
            when ``use_map_prior`` is True).
        use_map_prior: Whether to attach the Dirichlet-Process MAP prior.
        freeze_hyperparams: If True, freeze gamma/beta/lambda at their inits.
        score_counts: If True, train on count vectors with the
            Dirichlet-multinomial marginal; otherwise on probability vectors
            with the Dirichlet density.

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


def _make_dataset(
    columns: np.ndarray, batch_size: int, shuffle: bool, seed: int
) -> tf.data.Dataset:
    """Turn column distributions into a batched dataset for the prior model.

    Each column is a single-state observation of shape ``(1, dim)`` paired with
    a dummy target. The training dataset is reshuffled every epoch.
    """
    x = columns[:, np.newaxis, :].astype(np.float32)  # (M, 1, dim)
    y = np.zeros((columns.shape[0],), dtype=np.float32)  # dummy targets
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(
            columns.shape[0], seed=seed, reshuffle_each_iteration=True
        )
    return ds.batch(batch_size)


def train(
    model: tf.keras.Model,
    train_columns: np.ndarray,
    val_columns: np.ndarray | None,
    lr: float,
    epochs: int,
    batch_size: int,
    patience: int,
    seed: int,
) -> float:
    """Fit the Dirichlet prior by maximizing the column log-likelihood.

    Training shuffles the columns every epoch and stops early once the
    validation loss has not improved for ``patience`` epochs, restoring the
    weights of the best epoch. Without validation columns the training loss is
    monitored instead.

    Args:
        model: The trainable Dirichlet model.
        train_columns: Training column distributions of shape ``(M, dim)``.
        val_columns: Validation column distributions of shape ``(N, dim)``, or
            ``None`` to train without validation.
        lr: Learning rate of the Adam optimizer.
        epochs: Maximum number of training epochs.
        batch_size: Number of columns per batch.
        patience: Epochs without loss improvement before stopping.
        seed: Random seed for dataset shuffling.

    Returns:
        The best (lowest) monitored loss reached during the run, i.e. the
        negative mean column log-likelihood on the validation columns, or on
        the training columns if no validation is done.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    # The model outputs per-column log densities; minimize their negative mean.
    model.compile(
        optimizer=optimizer,
        loss=lambda y_true, y_pred: -tf.reduce_mean(y_pred),
    )

    train_ds = _make_dataset(train_columns, batch_size, shuffle=True, seed=seed)
    if val_columns is None:
        val_ds = None
        monitor = "loss"
    else:
        val_ds = _make_dataset(
            val_columns, batch_size, shuffle=False, seed=seed
        )
        monitor = "val_loss"

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        mode="min",
        patience=patience,
        restore_best_weights=True,
    )
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stopping],
        verbose=0,
    )
    best_loss = float(np.min(history.history[monitor]))
    label = "validation" if val_ds is not None else "training"
    print(f"Best {label} mean column log-likelihood: {-best_loss:.4f}")
    return best_loss


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

    # Verify the saved weights can be loaded back into a fresh model.
    reloaded = make_dirichlet_model(dim=dim, components=components)
    reloaded.load_weights(str(output_path))
    np.testing.assert_allclose(
        reloaded.layers[1].matrix().numpy(),
        model.layers[1].matrix().numpy(),
        atol=1e-6,
    )
    print("Round-trip verification passed.")


def summarize_prior(prior, dim: int, components: int) -> str:
    """Build a human-readable diagnostic summary of a fitted Dirichlet prior.

    Reports the per-component total concentration ``sum(alpha)`` and, for
    mixtures, the mixture weights and the effective number of active
    components ``exp(H(weights))``. This makes the two degenerate outcomes on
    sparse data directly visible: a sub-1 total concentration, and a mixture
    that has collapsed onto a single component.

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

    dim = len(args.alphabet)
    score_counts = args.score == "counts"
    # Neff weighting only reshapes the count magnitudes, which is meaningful for
    # the count-based marginal but cancels out once columns are normalized to
    # probabilities; it is therefore applied only with --score counts.
    use_neff = score_counts and not args.no_neff
    if args.no_neff and not score_counts:
        print("--no-neff has no effect with --score probabilities.")
    columns, family_of_column, family_accessions = collect_columns(
        input_dir=args.input_dir,
        pattern=args.pattern,
        fmt=args.fmt,
        alphabet=args.alphabet,
        replace_with_x=args.replace_with_x,
        min_count=args.min_count,
        min_occupancy=args.min_occupancy,
        neff=use_neff,
        neff_target_bits=args.neff_target_bits,
        neff_prior_conc=args.neff_prior_conc,
    )
    # collect_columns returns (Neff-weighted or raw) counts. The Dirichlet-
    # multinomial marginal scores counts directly; the Dirichlet density needs
    # probability vectors.
    if not score_counts:
        columns = columns / columns.sum(axis=1, keepdims=True)
    print(
        f"Scoring columns as {args.score}"
        f"{' with Neff weighting' if use_neff else ''}."
    )

    rng = np.random.default_rng(args.seed)
    num_total = columns.shape[0]
    if args.val_fraction == 0 and args.num_runs > 1:
        raise ValueError(
            "--val-fraction 0 disables validation, but --num-runs "
            f"{args.num_runs} needs a held-out set to select the best run; "
            "use --num-runs 1 or a positive --val-fraction."
        )
    # Validation only serves model selection between runs, so a single run does
    # not need a held-out set either.
    validate = args.num_runs > 1
    if not validate:
        if args.max_columns is not None and num_total > args.max_columns:
            if args.clans is None:
                idx = rng.choice(num_total, size=args.max_columns, replace=False)
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
                f"validation columns out of {columns.shape[0]}; choose a value "
                f"in (0, 1)."
            )
        val_columns = columns[perm[:num_val]]
        train_columns = columns[perm[num_val:]]
    else:
        # Clan-balanced sampling. Split the columns into disjoint train/val
        # pools first (so the with-replacement draws never share a column),
        # then draw each set hierarchically (clan -> family -> column).
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
        n_total = args.max_columns if args.max_columns is not None else num_total
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

    # Frozen background anchor for the MAP prior: the data mean expressed as
    # logits, so softmax(background_init) recovers the mean (which sums to 1).
    use_map_prior = not args.no_map_prior
    mu = np.clip(train_columns.mean(axis=0), 1e-4, None)
    mu = mu / mu.sum()
    background_init = np.log(mu).astype(np.float64)
    num_examples = train_columns.shape[0]

    # Run several trainings from different random initializations and keep the
    # model with the lowest validation loss. With a single run there is nothing
    # to select and the reported loss is the training loss.
    best_val_loss = np.inf
    best_prior_weights: list[np.ndarray] | None = None
    for run in range(args.num_runs):
        run_seed = args.seed + run
        tf.random.set_seed(run_seed)
        print(f"\n=== Training run {run + 1}/{args.num_runs} (seed {run_seed}) ===")
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
        )
        print(summarize_prior(model.prior, dim, args.components))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Keep only the prior kernel; the MAP-prior hyperparameters are a
            # training-only artifact and are not part of the saved format.
            best_prior_weights = model.prior.get_weights()
            if validate:
                print(f"New best model (validation loss {val_loss:.4f}).")

    assert best_prior_weights is not None  # num_runs >= 1 guarantees a fit
    print(
        f"\nBest {'validation' if validate else 'training'} mean column "
        f"log-likelihood over {args.num_runs} run(s): {-best_val_loss:.4f}"
    )
    # Rebuild a clean, prior-only model and load the best prior kernel. This
    # keeps the saved weight layout identical to what make_dirichlet_model /
    # load_dirichlet expect at inference time.
    best_model = make_dirichlet_model(dim=dim, components=args.components)
    best_model.layers[1].set_weights(best_prior_weights)
    print(summarize_prior(best_model.layers[1], dim, args.components))

    if args.output is not None:
        output_path = Path(args.output)
    else:
        output_path = WEIGHTS_DIR / f"{args.name}_{args.components}.weights.h5"
    save(best_model, output_path, dim, args.components)


if __name__ == "__main__":
    main()
