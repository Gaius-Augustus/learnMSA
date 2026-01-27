import numpy as np
from learnMSA.util.sequence_dataset import SequenceDataset
from learnMSA.util.aligned_dataset import AlignedDataset
import subprocess
from shutil import which
import sys
import Bio.SeqIO



class AlignedInsertions():
    def __init__(self,
                 aligned_insertions = None,
                 aligned_left_flank = None,
                 aligned_right_flank = None,
                 aligned_unannotated_segments = None):
        """
        Args:
            aligned_insertions: List of lists of pairs
                (indices, AlignedDataset with aligned slices) or None.
                Inner lists have length equal to length of model -1.
                Outer list has length num_repeats.
            aligned_left_flank: A pair (indices, AlignedDataset with aligned
                slices) or None.
            aligned_right_flank: A pair (indices, AlignedDataset with aligned
                slices) or None.
            unannotated_data: List of pairs (indices, AlignedDataset with
                aligned slices) or None of length num_repeats-1.
        """
        self.aligned_insertions = aligned_insertions
        self.aligned_left_flank = aligned_left_flank
        self.aligned_right_flank = aligned_right_flank
        self.aligned_unannotated_segments = aligned_unannotated_segments

        def _process(msa_data : AlignedDataset):
            custom_columns = np.zeros(
                (msa_data.num_seq, np.amax(msa_data.alignment_len)),
                dtype=np.int32
            )
            for i in range(msa_data.num_seq):
                cols = msa_data.get_column_map(i)
                custom_columns[i, :cols.size] = cols
            return custom_columns


        if aligned_insertions is None:
            self.ext_insertions = 0
        else:
            self.custom_columns_insertions = []
            for repeat in aligned_insertions:
                self.custom_columns_insertions.append([])
                for x in repeat:
                    if x is None:
                        self.custom_columns_insertions[-1].append(None)
                    else:
                        self.custom_columns_insertions[-1].append(
                            _process(x[1])
                        )
            self.ext_insertions = np.array([
                [np.amax(x)+1 if x is not None else 0 for x in repeats]
                for repeats in self.custom_columns_insertions
            ])

        if aligned_left_flank is None:
            self.ext_left_flank = 0
        else:
            self.custom_columns_left_flank = _process(aligned_left_flank[1])
            self.ext_left_flank = np.amax(self.custom_columns_left_flank)+1

        if aligned_right_flank is None:
            self.ext_right_flank = 0
        else:
            self.custom_columns_right_flank = _process(aligned_right_flank[1])
            self.ext_right_flank = np.amax(self.custom_columns_right_flank)+1

        if aligned_unannotated_segments is None:
            self.ext_unannotated = 0
        else:
            self.custom_columns_unannotated_segments = [
                _process(x[1]) if x is not None else None
                for x in aligned_unannotated_segments
            ]
            self.ext_unannotated = np.array([
                np.amax(x)+1 if x is not None else 0
                for x in self.custom_columns_unannotated_segments
            ])

    def insertion(self, batch_indices, r):
        if self.aligned_insertions is None:
            return None
        else:
            return [
                self._get_custom_columns(
                    batch_indices,
                    self.aligned_insertions[r][i][0],
                    self.custom_columns_insertions[r][i],
                    self.ext_insertions[r,i]
                )
                if self.aligned_insertions[r][i] is not None else None
                for i in range(len(self.aligned_insertions[r]))
            ]

    def left_flank(self, batch_indices):
        if self.aligned_left_flank is None:
            return None
        else:
            return self._get_custom_columns(
                batch_indices,
                self.aligned_left_flank[0],
                self.custom_columns_left_flank,
                self.ext_left_flank
            )

    def right_flank(self, batch_indices):
        if self.aligned_right_flank is None:
            return None
        else:
            return self._get_custom_columns(
                batch_indices,
                self.aligned_right_flank[0],
                self.custom_columns_right_flank,
                self.ext_right_flank
            )

    def unannotated_segment(self, batch_indices, r):
        if self.aligned_unannotated_segments is None:
            return None
        else:
            if self.aligned_unannotated_segments[r] is None:
                return None
            else:
                return self._get_custom_columns(
                    batch_indices,
                    self.aligned_unannotated_segments[r][0],
                    self.custom_columns_unannotated_segments[r],
                    self.ext_unannotated[r]
                )

    def _get_custom_columns(
        self,
        batch_indices,
        custom_indices,
        custom_columns,
        max_len
    ):
        columns = np.stack([np.arange(max_len)]*batch_indices.shape[0])
        for i, c in zip(custom_indices, custom_columns):
            columns[batch_indices == i, :c.size] = c
        return columns


def find_long_insertions_and_get_sequences(data : SequenceDataset, lens, starts, t = 20, k=2, max_insertions_len=500, max_insertions_len_below_seq_ok = 100):
    """
    Finds insertions that have at least length t. If there are at least k of these sequences, returns id + fragment pairs.
    Args: 
        data: Dataset of all sequences.
        slices: Distionary keeping track of the slices.
        lens, starts: Arrays of length n where n is the number of sequences in the dataset. Indicate how long insertions are and where they start respectively.
        name: Identifier for the location of the slice (e.g. left_flank or match_5).
    """
    at_least_t = lens >= t
    lengths = lens[at_least_t]
    if lengths.size > 1:
        which = np.squeeze(np.argwhere(at_least_t))
        start = starts[at_least_t]
        id_fragment_pairs = []
        to_delete = [] #keeps track of fragments that are too long
        for j in range(lengths.size):
            aa_seq = data.get_standardized_seq(which[j])
            segment = aa_seq[start[j] : start[j] + lengths[j]]
            #sometimes segments look strange (like ones consisting only of X)
            #this can cause problems in the downstream aligner, omit these segments
            non_standard_freq = 0
            for aa in data.alphabet[20:]:
                non_standard_freq += segment.count(aa)
            non_standard_freq /= len(segment)
            mostly_non_standard_aa = non_standard_freq > 0.5
            if (mostly_non_standard_aa or 
                (lengths[j] > max_insertions_len and 
                    which.size > max_insertions_len_below_seq_ok)):
                to_delete.append(j)
            else:
                sid = data.seq_ids[which[j]]+"\n"
                id_fragment_pairs.append((sid, segment))
        which = np.delete(which, to_delete)
        if which.size > k:
            return (which, id_fragment_pairs)
    return None


def make_aligned_insertions(am, best_model, method="famsa", threads=0, verbose=True):
    """
    Aligns insertions with the given method and adds them to the alignment model.
    Args:
        am: Alignment model.
        best_model: The best model to use for extracting insertions.
        method: Alignment method. Currently, only famsa is supported.
        threads: Number of threads to use. If 0, uses all available threads.
    """
    if not best_model in am.metadata:
        am._build_alignment([best_model])
    data = am.metadata[best_model]
    num_seq = data.left_flank_len.shape[0]

    insertions_long = []
    for r in range(data.num_repeats):
        insertions_long.append([])
        for i in range(data.insertion_lens.shape[2]):
            ins_long = find_long_insertions_and_get_sequences(am.data, data.insertion_lens[r, :, i], data.insertion_start[r, :, i])
            insertions_long[-1].append(ins_long)
    left_flank_long = find_long_insertions_and_get_sequences(am.data, data.left_flank_len, data.left_flank_start)
    right_flank_long = find_long_insertions_and_get_sequences(am.data, data.right_flank_len, data.right_flank_start)
    unannotated_long = []
    for r in range(data.num_repeats-1):
        unannotated_long.append(find_long_insertions_and_get_sequences(am.data, data.unannotated_segments_len[r], data.unannotated_segments_start[r]))

    slices = {}
    if left_flank_long is not None:
        slices["left_flank"] = left_flank_long[1]
    if right_flank_long is not None:
        slices["right_flank"] = right_flank_long[1]
    for r in range(data.num_repeats):
        for i in range(data.insertion_lens.shape[2]):
            if insertions_long[r][i] is not None:
                slices[f"ins_{r}_{i}"] = insertions_long[r][i][1]
    for r in range(data.num_repeats-1):
        if unannotated_long[r] is not None:
            slices[f"unannotated_{r}"] = unannotated_long[r][1]

    if verbose:
        print(f"Aligning {len(slices)} insertion slices with {method}.")
    alignments = make_slice_msas(slices, method, threads)

    #merge msa
    insertions_long = [[(x[0], AlignedDataset(aligned_sequences = alignments[f"ins_{r}_{i}"] )) if x is not None else None for i,x in enumerate(repeats)] for r,repeats in enumerate(insertions_long)]
    left_flank_long = (left_flank_long[0],  AlignedDataset(aligned_sequences = alignments["left_flank"])) if left_flank_long is not None else None
    right_flank_long = (right_flank_long[0],  AlignedDataset(aligned_sequences = alignments["right_flank"])) if right_flank_long is not None else None
    unannotated_long = [(x[0], AlignedDataset(aligned_sequences = alignments[f"unannotated_{r}"])) if x is not None else None for r,x in enumerate(unannotated_long)]

    aligned_insertions = AlignedInsertions(insertions_long, left_flank_long, right_flank_long, unannotated_long)
    return aligned_insertions


def make_slice_msas(slices, method="famsa", threads=0):
    if method == "famsa":
        alignments = align_with_famsa(slices, threads)
    else:
        print(f"Unknown aligner {method}")
        sys.exit(1)
    return alignments


def align_with_famsa(slices, threads):
    #keep conditional import, famsa could be optional in the future
    from pyfamsa import Aligner as FamsaAligner, Sequence as FamsaSequence
    aligner = FamsaAligner(threads = threads)
    alignments = {}
    for key, seqs in slices.items():
        enc_seqs =  [FamsaSequence(sid.encode(), seq.encode()) for sid,seq in seqs]
        msa = aligner.align(enc_seqs)
        alignments[key] = [(sequence.id.decode(), sequence.sequence.decode()) for sequence in msa]
    return alignments