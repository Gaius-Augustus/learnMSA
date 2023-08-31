import numpy as np
from learnMSA.msa_hmm.SequenceDataset import SequenceDataset, AlignedDataset
import learnMSA.msa_hmm.AlignmentModel as alignment_model
import subprocess
from shutil import which
import sys
import Bio.SeqIO


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
            aa_seq = str(data.get_record(which[j]).seq).upper()
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

        
def make_aligned_insertions(am, method="famsa", threads=0, verbose=True):
    """
    Aligns insertions with the given method and adds them to the alignment model.
    Args:
        am: Alignment model.
        method: Alignment method. Currently, only famsa is supported.
        threads: Number of threads to use. If 0, uses all available threads.
    """
    if not am.best_model in am.metadata:
        am._build_alignment([am.best_model])
    data = am.metadata[am.best_model]
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
    
    aligned_insertions = alignment_model.AlignedInsertions(insertions_long, left_flank_long, right_flank_long, unannotated_long)
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