import sys
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

from learnMSA.backend import get_backend
from learnMSA.config import Configuration
from learnMSA.util.multi_dataset import MultiSequenceDataset
from learnMSA.util.sequence_dataset import SequenceDataset

if TYPE_CHECKING:
    from learnMSA.align.alignment_model import AlignmentModel

#: Insertions of at least this length make a position worth aligning.
LONG_INSERTION_LENGTH = 20




class SliceColumns:
    """Column mapping of one aligned insertion slice.

    Attributes:
        rows: (n_slice,) int32, indices of the sequences in this slice.
        cols: (n_slice, max_fragment_len) int16/int32. ``cols[i, k]`` is the
            column the k-th residue of ``rows[i]``'s fragment is aligned to.
        width: Number of columns of the slice MSA.
    """

    __slots__ = ("rows", "cols", "width")

    def __init__(self, rows: np.ndarray, cols: np.ndarray, width: int) -> None:
        self.rows = rows
        self.cols = cols
        self.width = width




def _slice_columns(rows: np.ndarray, gapped: list[str]) -> SliceColumns:
    """Build a :class:`SliceColumns` from the gapped strings of a slice MSA.
    """

    _FAMSA_GAP_BYTES = (ord("-"), ord("."))
    # Upper bound on the decoded bytes of a slice MSA held at once.
    _SLICE_CHUNK_BYTES = 8 * 1024 * 1024

    n = len(gapped)
    width = len(gapped[0]) if n > 0 else 0
    gap0, gap1 = chr(_FAMSA_GAP_BYTES[0]), chr(_FAMSA_GAP_BYTES[1])
    frag_lens = np.fromiter(
        (width - g.count(gap0) - g.count(gap1) for g in gapped),
        dtype=np.int64, count=n,
    )
    max_frag = int(frag_lens.max()) if n > 0 else 0
    # Column indices fit in int16 unless the slice MSA is enormous.
    dtype = np.int16 if width <= np.iinfo(np.int16).max else np.int32
    cols = np.zeros((n, max_frag), dtype=dtype)
    step = max(1, _SLICE_CHUNK_BYTES // max(1, width))
    for start in range(0, n, step):
        end = min(start + step, n)
        arr = np.frombuffer(
            "".join(gapped[start:end]).encode("ascii"), dtype=np.uint8
        ).reshape(end - start, width)
        non_gap = (arr != _FAMSA_GAP_BYTES[0]) & (arr != _FAMSA_GAP_BYTES[1])
        del arr
        row_idx, col_idx = np.nonzero(non_gap)
        del non_gap
        chunk_lens = frag_lens[start:end]
        # Position of each residue within its own fragment.
        offsets = np.concatenate([[0], np.cumsum(chunk_lens)[:-1]])
        within = np.arange(row_idx.size) - np.repeat(offsets, chunk_lens)
        cols[start + row_idx, within] = col_idx
    return SliceColumns(np.asarray(rows, dtype=np.int32), cols, width)


class AlignedInsertions():
    def __init__(self,
                 n_total: int = 0,
                 aligned_insertions = None,
                 aligned_left_flank = None,
                 aligned_right_flank = None,
                 aligned_unannotated_segments = None):
        """
        Args:
            n_total: Total number of sequences being aligned. Kept for
                backwards compatibility; no longer used for allocation.
            aligned_insertions: List of lists of :class:`SliceColumns` or None.
                Inner lists have length equal to length of model -1.
                Outer list has length num_repeats.
            aligned_left_flank: A :class:`SliceColumns` or None.
            aligned_right_flank: A :class:`SliceColumns` or None.
            aligned_unannotated_segments: List of :class:`SliceColumns` or None
                of length num_repeats-1.
        """
        self.n_total = n_total
        self.aligned_insertions = aligned_insertions
        self.aligned_left_flank = aligned_left_flank
        self.aligned_right_flank = aligned_right_flank
        self.aligned_unannotated_segments = aligned_unannotated_segments

        if aligned_insertions is None:
            self.ext_insertions = 0
        else:
            self.ext_insertions = np.array([
                [0 if x is None else x.width for x in repeat]
                for repeat in aligned_insertions
            ])

        self.ext_left_flank = (
            0 if aligned_left_flank is None else aligned_left_flank.width
        )
        self.ext_right_flank = (
            0 if aligned_right_flank is None else aligned_right_flank.width
        )

        if aligned_unannotated_segments is None:
            self.ext_unannotated = 0
        else:
            self.ext_unannotated = np.array([
                0 if x is None else x.width
                for x in aligned_unannotated_segments
            ])

    def insertion(self, batch_indices, r, i=None):
        """Custom columns of repeat *r*.

        With *i* given, only the columns of insertion position *i* are
        materialized. Callers should prefer that: materializing every position
        of a repeat at once costs (batch_size, total insertion width) int32,
        which is hundreds of MB for wide alignments.
        """
        if self.aligned_insertions is None:
            return None
        if i is not None:
            return self._get_custom_columns(
                batch_indices, self.aligned_insertions[r][i]
            )
        return [
            self._get_custom_columns(batch_indices, x)
            for x in self.aligned_insertions[r]
        ]

    def left_flank(self, batch_indices):
        return self._get_custom_columns(batch_indices, self.aligned_left_flank)

    def right_flank(self, batch_indices):
        return self._get_custom_columns(batch_indices, self.aligned_right_flank)

    def unannotated_segment(self, batch_indices, r):
        if self.aligned_unannotated_segments is None:
            return None
        return self._get_custom_columns(
            batch_indices, self.aligned_unannotated_segments[r]
        )

    def _get_custom_columns(self, batch_indices, slice_columns):
        """Per-row column map for *batch_indices*.

        Rows without an aligned insertion in this slice keep the identity
        mapping. ``slice_columns.rows`` is ascending, so membership is resolved
        with a binary search instead of a dense lookup table.
        """
        if slice_columns is None:
            return None
        rows, cols = slice_columns.rows, slice_columns.cols
        result = np.tile(
            np.arange(cols.shape[1], dtype=np.int32),
            (batch_indices.shape[0], 1),
        )
        if rows.size == 0:
            return result
        pos = np.searchsorted(rows, batch_indices)
        np.minimum(pos, rows.size - 1, out=pos)
        has_ins = rows[pos] == batch_indices
        if np.any(has_ins):
            result[has_ins] = cols[pos[has_ins]]
        return result


def find_long_insertions_and_get_sequences(data : SequenceDataset, lens, starts, t = LONG_INSERTION_LENGTH, k=2, max_insertions_len=500, max_insertions_len_below_seq_ok = 100, row_to_seq=None, all_fragments=False):
    """
    Finds insertions that have at least length t. If there are at least k of these sequences, returns id + fragment pairs.
    Args: 
        data: Dataset of all sequences.
        slices: Distionary keeping track of the slices.
        lens, starts: Arrays of length n where n is the number of sequences in the dataset. Indicate how long insertions are and where they start respectively.
        name: Identifier for the location of the slice (e.g. left_flank or match_5).
        row_to_seq: Maps an alignment row to its index in *data*. Required when
            the alignment covers a subset of the dataset; the returned row
            indices stay in alignment-row space either way.
        all_fragments: If True, the long insertions only decide whether the
            slice is aligned, and the returned slice holds every non-empty
            insertion at this position.
    """
    at_least_t = lens >= t
    lengths = lens[at_least_t]
    if lengths.size > 1:
        # Very long fragments are dropped when there are many long ones
        drop_longer_than = (
            max_insertions_len
            if lengths.size > max_insertions_len_below_seq_ok else np.inf
        )
        which, id_fragment_pairs = _get_fragments(
            data, np.flatnonzero(at_least_t), lens, starts, row_to_seq,
            drop_longer_than,
        )
        if which.size > k:
            if all_fragments:
                return _get_fragments(
                    data, np.flatnonzero(lens > 0), lens, starts, row_to_seq,
                    drop_longer_than,
                )
            return (which, id_fragment_pairs)
    return None


def _get_fragments(data, which, lens, starts, row_to_seq, drop_longer_than):
    """The fragments of the insertions in the rows *which*, except those
    that consist mostly of non-standard residues or are longer than
    drop_longer_than.

    Returns:
        The rows that were kept and their (id, fragment) pairs.
    """
    # Count residues that are not one of the 20 standard amino acids
    # (ambiguity codes X/B/Z/J and, unless modeled, U/O).
    standard = SequenceDataset._default_alphabet
    keep = []
    id_fragment_pairs = []
    for row in which:
        seq_idx = int(row if row_to_seq is None else row_to_seq[row])
        aa_seq = data.get_standardized_seq(seq_idx)
        segment = aa_seq[starts[row] : starts[row] + lens[row]]
        #sometimes segments look strange (like ones consisting only of X)
        #this can cause problems in the downstream aligner, omit these segments
        non_standard_freq = sum(
            1 for ch in segment if ch not in standard
        ) / max(1, len(segment))
        if non_standard_freq > 0.5 or lens[row] > drop_longer_than:
            continue
        keep.append(row)
        sid = data.seq_ids[seq_idx]+"\n"
        id_fragment_pairs.append((sid, segment))
    return np.asarray(keep, dtype=np.int64), id_fragment_pairs


def make_aligned_insertions(
    am, best_model, decoding_mode, method="famsa", threads=0, verbose=True,
    config: Configuration | None = None,
):
    """
    Aligns insertions with the given method and adds them to the alignment model.

    Args:
        am: Alignment model.
        best_model: The best model to use for extracting insertions.
        decoding_mode: Decoding mode for alignment model.
        method: Alignment method, one of "famsa", "learnmsa" or "auto" (see
            ``AdvancedConfig.insertion_aligner``).
        threads: Number of threads to use (famsa). If 0, uses all available
            threads.
        verbose: Whether to print progress messages.
        config: Configuration of the learnmsa aligner. Defaults to the
            configuration of am's model.
    """
    return make_aligned_insertions_multi(
        [(am, best_model)], decoding_mode, method, threads, verbose, config
    )[0]


def make_aligned_insertions_multi(
    alignments: Sequence[tuple["AlignmentModel", int]],
    decoding_mode,
    method="famsa",
    threads=0,
    verbose=True,
    config: Configuration | None = None,
) -> list[AlignedInsertions]:
    """
    Aligns the insertions of several alignments at once, e.g. of all heads
    of a MultiAlignmentModel, so that the learnmsa aligner trains and decodes
    all of them jointly.

    Args:
        alignments: Pairs of an alignment model and the index of the model
            whose insertions are aligned.
        decoding_mode: Decoding mode for the alignment models.
        method: See :func:`make_aligned_insertions`.
        threads: See :func:`make_aligned_insertions`.
        verbose: Whether to print progress messages.
        config: Configuration of the learnmsa aligner. Defaults to the
            configuration of the model of the first alignment.

    Returns:
        One AlignedInsertions per alignment.
    """
    method = _resolve_aligner(method)
    if config is None and method == "learnmsa":
        config = alignments[0][0].model.context.config

    # Slices of all alignments, keyed by "<alignment>/<slice>"
    meta_datas = []
    rows: dict[str, np.ndarray] = {}
    slices: dict[str, list] = {}
    for a, (am, model_index) in enumerate(alignments):
        # learnMSA trains on every insertion of a selected position
        meta_data, am_rows, am_slices = _collect_slices(
            am, model_index, decoding_mode,
            all_fragments=method == "learnmsa",
        )
        meta_datas.append(meta_data)
        for key in am_slices:
            rows[f"{a}/{key}"] = am_rows[key]
            slices[f"{a}/{key}"] = am_slices[key]

    if verbose:
        print(f"Aligning {len(slices)} insertion slices with {method}.")

    # Align and reduce one slice at a time so that the raw fragments, the
    # gapped fragments and the column maps of all slices are never all alive
    # at the same time.
    columns = make_slice_msas(
        slices, rows, method, threads, config, decoding_mode
    )

    per_alignment: list[dict] = [{} for _ in alignments]
    for key, slice_columns in columns.items():
        a, name = key.split("/", 1)
        per_alignment[int(a)][name] = slice_columns
    return [
        _build_aligned_insertions(meta_data, cols)
        for meta_data, cols in zip(meta_datas, per_alignment)
    ]


def _collect_slices(am, model_index, decoding_mode, all_fragments=False):
    """Collects the long insertions of one model of an alignment.

    Args:
        all_fragments: If True, a slice holds every insertion at its
            position, see :func:`find_long_insertions_and_get_sequences`.

    Returns:
        ``(meta_data, rows, slices)``. ``rows`` and ``slices`` map slice keys
        to the alignment rows and the (id, fragment) pairs of that slice.
    """
    if not model_index in am.metadata:
        am.build_alignment([model_index], decoding_mode)
    meta_data = am.metadata[model_index]
    num_seq = meta_data.left_flank_len.shape[0]
    all_rows = np.arange(num_seq)
    num_ins_pos = meta_data.insertion_lens.shape[1]

    # Collect the raw fragments of every slice. `rows` and `slices` are kept in
    # lockstep and are consumed (and freed) one slice at a time later.
    rows: dict[str, np.ndarray] = {}
    slices: dict[str, list] = {}

    def collect(key, found):
        if found is not None:
            rows[key], slices[key] = found

    # Alignment rows and dataset indices differ when a subset is aligned.
    row_to_seq = am.indices

    collect("left_flank", find_long_insertions_and_get_sequences(
        am.data[0],
        meta_data.left_flank_len_for(all_rows),
        meta_data.left_flank_start_for(all_rows),
        row_to_seq=row_to_seq, all_fragments=all_fragments,
    ))
    collect("right_flank", find_long_insertions_and_get_sequences(
        am.data[0],
        meta_data.right_flank_len_for(all_rows),
        meta_data.right_flank_start_for(all_rows),
        row_to_seq=row_to_seq, all_fragments=all_fragments,
    ))
    for r in range(meta_data.num_repeats):
        for i in range(num_ins_pos):
            # Only the single insertion column i is needed here; pulling the
            # full per-repeat arrays would copy (num_seqs, num_match) int16
            # several times over.
            il_i, is_i = meta_data.get_repeat_insertions(r, all_rows, i)
            collect(f"ins_{r}_{i}", find_long_insertions_and_get_sequences(
                am.data[0], il_i, is_i, row_to_seq=row_to_seq,
                all_fragments=all_fragments,
            ))
    for r in range(meta_data.num_repeats-1):
        uns_l, uns_s = meta_data.get_unannotated_data(r, all_rows)
        collect(f"unannotated_{r}", find_long_insertions_and_get_sequences(
            am.data[0], uns_l, uns_s, row_to_seq=row_to_seq,
            all_fragments=all_fragments,
        ))
    return meta_data, rows, slices


def _build_aligned_insertions(meta_data, columns) -> AlignedInsertions:
    """Assembles the column maps of the slices of one alignment."""
    num_seq = meta_data.left_flank_len.shape[0]
    num_ins_pos = meta_data.insertion_lens.shape[1]
    insertions_long = [
        [columns.get(f"ins_{r}_{i}") for i in range(num_ins_pos)]
        for r in range(meta_data.num_repeats)
    ]
    unannotated_long = [
        columns.get(f"unannotated_{r}")
        for r in range(meta_data.num_repeats-1)
    ]
    return AlignedInsertions(
        num_seq,
        insertions_long,
        columns.get("left_flank"),
        columns.get("right_flank"),
        unannotated_long,
    )


def _resolve_aligner(method: str) -> str:
    """Resolves "auto" to the insertion aligner of the selected backend."""
    if method == "auto":
        return "learnmsa" if get_backend() == "pytorch" else "famsa"
    if method == "learnmsa" and get_backend() != "pytorch":
        raise ValueError(
            "The learnmsa insertion aligner requires the pytorch backend, "
            f"but '{get_backend()}' is selected. Use the famsa aligner "
            "instead."
        )
    return method


def make_slice_msas(
    slices, rows, method="famsa", threads=0,
    config: Configuration | None = None, decoding_mode=None,
):
    """Align every slice and reduce it to a :class:`SliceColumns`.

    *slices* and *rows* are emptied as they are processed, so the raw fragments
    of a slice are freed as soon as its column map exists.
    """
    method = _resolve_aligner(method)
    if method == "famsa":
        return align_with_famsa(slices, rows, threads)
    if method == "learnmsa":
        if config is None:
            raise ValueError("The learnmsa aligner needs a configuration.")
        return align_with_learnmsa(slices, rows, config, decoding_mode)
    print(f"Unknown aligner {method}")
    sys.exit(1)


def align_with_learnmsa(
    slices, rows, config: Configuration, decoding_mode=None
):
    """Aligns every slice with its own pHMM, trained and decoded by learnMSA.

    The slices are sorted by the median length of their long fragments and
    aligned in joint runs of at most ``config.advanced.insertion_max_heads``
    slices, one pHMM head per slice. Each head is trained on all fragments of
    its slice, starting at the median length of the long fragments (see
    :data:`LONG_INSERTION_LENGTH`) capped at
    ``config.advanced.insertion_max_length``, without model surgery.
    Insertions that are much longer than the cap are underfitted.
    """
    # align imports alignment_model, which imports this module
    from learnMSA.align.align import align_batch
    from learnMSA.align.alignment_model import AlignmentModel

    if decoding_mode is None:
        decoding_mode = AlignmentModel.DecodingMode.VITERBI
    adv = config.advanced
    columns = {}
    for keys in _chunk_slice_keys(slices, adv.insertion_max_heads):
        lengths = _insertion_lengths(slices, keys, adv.insertion_max_length)
        # The column maps only need the row order, so ids are positions
        data = MultiSequenceDataset(
            sequences=[
                [(str(j), seq) for j, (_, seq) in enumerate(slices.pop(key))]
                for key in keys
            ],
            model_uo=config.hmm.model_uo,
        )
        am = align_batch(data, _insertion_config(config, lengths))
        am.build_alignment(decoding_mode=decoding_mode)
        for k, key in enumerate(keys):
            gapped = am.to_string(k, add_block_sep=False)
            columns[key] = _slice_columns(rows.pop(key), gapped)
        del am, data
    return columns


def _median_length(fragments: list) -> float:
    """The median length of the long fragments (all if there are none)."""
    lens = np.array([len(seq) for _, seq in fragments])
    long_lens = lens[lens >= LONG_INSERTION_LENGTH]
    return float(np.median(long_lens if long_lens.size else lens))


def _chunk_slice_keys(slices, max_heads: int) -> list[list[str]]:
    """Sorts the slice keys by the median length of their long fragments and
    splits them into chunks of at most max_heads, so that heads of similar
    length share a run."""
    keys = sorted(slices, key=lambda key: _median_length(slices[key]))
    return [keys[i:i + max_heads] for i in range(0, len(keys), max_heads)]


def _insertion_lengths(slices, keys, max_length: int) -> list[int]:
    """The initial model length of each slice: the median length of its long
    fragments, capped at max_length and at least 3."""
    return [
        max(3, min(max_length, int(_median_length(slices[key]))))
        for key in keys
    ]


def _insertion_config(
    config: Configuration, lengths: list[int]
) -> Configuration:
    """The configuration of a learnMSA run that aligns insertions.

    Uses default pHMM heads and training with one head per slice and no
    model surgery. Only general settings (backend, compilation, batch size,
    hit alignment, prior scale) are taken from config.
    """
    inner = Configuration()
    inner.advanced = config.advanced.model_copy(deep=True)
    inner.hmm.model_uo = config.hmm.model_uo
    training = inner.training
    training.length_init = list(lengths)
    training.max_iterations = 1
    # Avoid one mmseqs2 run per slice
    training.no_sequence_weights = True
    # Short fragments dominate most slices, so cropping relative to the mean
    # length would cut the long ones below the model length
    training.auto_crop = False
    training.crop = 2 * config.advanced.insertion_max_length
    training.batch_size = config.training.batch_size
    training.tokens_per_batch = config.training.tokens_per_batch
    training.hit_alignment_mode = config.training.hit_alignment_mode
    training.prior_scale = config.training.prior_scale
    inner.input_output.verbose = False
    inner.input_output.work_dir = str(
        Path(config.input_output.work_dir) / "insertion_alignment"
    )
    return inner


def align_with_famsa(slices, rows, threads):
    #keep conditional import, famsa could be optional in the future
    from pyfamsa import Aligner as FamsaAligner, Sequence as FamsaSequence
    aligner = FamsaAligner(threads = threads)
    columns = {}
    for key in list(slices):
        seqs = slices.pop(key)
        row_idx = rows.pop(key)
        enc_seqs = [
            FamsaSequence(sid.encode(), seq.encode()) for sid, seq in seqs
        ]
        msa = aligner.align(enc_seqs)
        del enc_seqs
        gapped = [sequence.sequence.decode() for sequence in msa]
        # The reduction below assumes row i of the MSA is fragment i of the
        # input. famsa preserves the input order, but verify rather than trust.
        out_ids = [sequence.id.decode() for sequence in msa]
        del msa
        in_ids = [sid for sid, _ in seqs]
        del seqs
        if out_ids != in_ids:
            order = {sid: j for j, sid in enumerate(out_ids)}
            gapped = [gapped[order[sid]] for sid in in_ids]
        del out_ids, in_ids
        columns[key] = _slice_columns(row_idx, gapped)
        del gapped
    return columns
