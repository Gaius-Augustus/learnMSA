import sys

import numpy as np

from learnMSA.util.sequence_dataset import SequenceDataset

#: Characters famsa may emit for a gap. Everything else is a residue.
_GAP_BYTES = (ord("-"), ord("."))


class SliceColumns:
    """Column mapping of one aligned insertion slice.

    Memory note: this deliberately stores only the rows that actually have an
    aligned insertion, not one entry per sequence in the dataset. At millions
    of sequences and hundreds of slices a dense per-slice map costs gigabytes
    (num_slices * num_seqs * 4 bytes), which is what used to kill large runs.

    Attributes:
        rows: (n_slice,) int32, ascending indices of the sequences that
            contribute to this slice.
        cols: (n_slice, max_fragment_len) int16/int32. ``cols[i, k]`` is the
            column the k-th residue of ``rows[i]``'s fragment is aligned to.
            Entries beyond a row's fragment length are never read.
        width: Number of columns of the slice MSA, i.e. by how much this block
            of the output alignment has to be widened.
    """

    __slots__ = ("rows", "cols", "width")

    def __init__(self, rows: np.ndarray, cols: np.ndarray, width: int) -> None:
        self.rows = rows
        self.cols = cols
        self.width = width


#: Upper bound on the decoded bytes of a slice MSA held at once.
_SLICE_CHUNK_BYTES = 8 * 1024 * 1024


def _slice_columns(rows: np.ndarray, gapped: list[str]) -> SliceColumns:
    """Build a :class:`SliceColumns` from the gapped strings of a slice MSA.

    Reads the column map straight out of the aligner's output instead of
    routing it through an :class:`~learnMSA.util.aligned_dataset.AlignedDataset`,
    which would additionally allocate a SeqRecord per fragment, a dense
    (n_slice, width) int16 MSA matrix and a Python list of n_slice arrays.
    A slice can cover a large share of the dataset, so the decoded MSA is
    walked in row chunks rather than materialized in one block.
    """
    n = len(gapped)
    width = len(gapped[0]) if n > 0 else 0
    gap0, gap1 = chr(_GAP_BYTES[0]), chr(_GAP_BYTES[1])
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
        non_gap = (arr != _GAP_BYTES[0]) & (arr != _GAP_BYTES[1])
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


def find_long_insertions_and_get_sequences(data : SequenceDataset, lens, starts, t = 20, k=2, max_insertions_len=500, max_insertions_len_below_seq_ok = 100, row_to_seq=None):
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
    """
    at_least_t = lens >= t
    lengths = lens[at_least_t]
    if lengths.size > 1:
        which = np.squeeze(np.argwhere(at_least_t))
        start = starts[at_least_t]
        id_fragment_pairs = []
        to_delete = [] #keeps track of fragments that are too long
        for j in range(lengths.size):
            row = which[j]
            seq_idx = int(row if row_to_seq is None else row_to_seq[row])
            aa_seq = data.get_standardized_seq(seq_idx)
            segment = aa_seq[start[j] : start[j] + lengths[j]]
            #sometimes segments look strange (like ones consisting only of X)
            #this can cause problems in the downstream aligner, omit these segments
            # Count residues that are not one of the 20 standard amino acids
            # (ambiguity codes X/B/Z/J and, unless modeled, U/O).
            standard = SequenceDataset._default_alphabet
            non_standard_freq = sum(
                1 for ch in segment if ch not in standard
            ) / max(1, len(segment))
            mostly_non_standard_aa = non_standard_freq > 0.5
            if (mostly_non_standard_aa or 
                (lengths[j] > max_insertions_len and 
                    which.size > max_insertions_len_below_seq_ok)):
                to_delete.append(j)
            else:
                sid = data.seq_ids[seq_idx]+"\n"
                id_fragment_pairs.append((sid, segment))
        which = np.delete(which, to_delete)
        if which.size > k:
            return (which, id_fragment_pairs)
    return None


def make_aligned_insertions(
    am, best_model, decoding_mode, method="famsa", threads=0, verbose=True
):
    """
    Aligns insertions with the given method and adds them to the alignment model.

    Args:
        am: Alignment model.
        best_model: The best model to use for extracting insertions.
        method: Alignment method. Currently, only famsa is supported.
        threads: Number of threads to use. If 0, uses all available threads.
        decoding_mode: Decoding mode for alignment model.
    """
    if not best_model in am.metadata:
        am._build_alignment([best_model], decoding_mode)
    meta_data = am.metadata[best_model]
    num_seq = meta_data.left_flank_len.shape[0]
    all_rows = np.arange(num_seq)
    num_ins_pos = meta_data.insertion_lens.shape[1]

    # Collect the raw fragments of every slice. `rows` and `slices` are kept in
    # lockstep and are consumed (and freed) one slice at a time below.
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
        row_to_seq=row_to_seq,
    ))
    collect("right_flank", find_long_insertions_and_get_sequences(
        am.data[0],
        meta_data.right_flank_len_for(all_rows),
        meta_data.right_flank_start_for(all_rows),
        row_to_seq=row_to_seq,
    ))
    for r in range(meta_data.num_repeats):
        for i in range(num_ins_pos):
            # Only the single insertion column i is needed here; pulling the
            # full per-repeat arrays would copy (num_seqs, num_match) int16
            # several times over.
            il_i, is_i = meta_data.get_repeat_insertions(r, all_rows, i)
            collect(f"ins_{r}_{i}", find_long_insertions_and_get_sequences(
                am.data[0], il_i, is_i, row_to_seq=row_to_seq
            ))
    for r in range(meta_data.num_repeats-1):
        uns_l, uns_s = meta_data.get_unannotated_data(r, all_rows)
        collect(f"unannotated_{r}", find_long_insertions_and_get_sequences(
            am.data[0], uns_l, uns_s, row_to_seq=row_to_seq
        ))

    if verbose:
        print(f"Aligning {len(slices)} insertion slices with {method}.")

    # Align and reduce one slice at a time so that the raw fragments, the
    # gapped fragments and the column maps of all slices are never all alive
    # at the same time.
    columns = make_slice_msas(slices, rows, method, threads)

    insertions_long = [
        [columns.get(f"ins_{r}_{i}") for i in range(num_ins_pos)]
        for r in range(meta_data.num_repeats)
    ]
    unannotated_long = [
        columns.get(f"unannotated_{r}")
        for r in range(meta_data.num_repeats-1)
    ]

    aligned_insertions = AlignedInsertions(
        num_seq,
        insertions_long,
        columns.get("left_flank"),
        columns.get("right_flank"),
        unannotated_long,
    )
    return aligned_insertions


def make_slice_msas(slices, rows, method="famsa", threads=0):
    """Align every slice and reduce it to a :class:`SliceColumns`.

    *slices* and *rows* are emptied as they are processed, so the raw fragments
    of a slice are freed as soon as its column map exists.
    """
    if method == "famsa":
        return align_with_famsa(slices, rows, threads)
    print(f"Unknown aligner {method}")
    sys.exit(1)


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
