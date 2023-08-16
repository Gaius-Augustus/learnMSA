import numpy as np
from learnMSA.msa_hmm.SequenceDataset import SequenceDataset, AlignedDataset
import learnMSA.msa_hmm.AlignmentModel as alignment_model
import subprocess
from shutil import which
import sys


def find_long_insertions_and_write_slice(data : SequenceDataset, lens, starts, name, directory, t = 20, k=2, max_insertions_len=500, max_insertions_len_below_seq_ok = 100, verbose=True):
        """
        Finds insertions that have at least length t. If there are at least k of these sequences, writes them to file.
        Args: 
            data: Dataset of all sequences.
            lens, starts: Arrays of length n where n is the number of sequences in the dataset. Indicate how long insertions are and where they start respectively.
            name: Identifier for the location of the slice (e.g. left_flank or match_5).
            directory: Directory where slice files are written.
        """
        at_least_t = lens >= t
        lengths = lens[at_least_t]
        if lengths.size > 1:
            which = np.squeeze(np.argwhere(at_least_t))
            start = starts[at_least_t]
            filename = f"{directory}/slice_{name}"
            to_delete = []
            with open(filename, "w") as slice_file:
                for j in range(lengths.size):
                    aa_seq = str(data.get_record(which[j]).seq)
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
                        slice_file.write(">"+data.seq_ids[which[j]]+"\n")
                        slice_file.write(segment+"\n")
            which = np.delete(which, to_delete)
            if which.size > k:
                if verbose:
                    print(f"Long insertions found at {name}: {which.size}.")
                return (which, filename)
        
        
def make_aligned_insertions(am, directory, method="famsa", threads=0, verbose=True):
    if not am.best_model in am.metadata:
        am._build_alignment([am.best_model])
    data = am.metadata[am.best_model]
    num_seq = data.left_flank_len.shape[0]

    insertions_long = []
    for r in range(data.num_repeats):
        insertions_long.append([])
        for i in range(data.insertion_lens.shape[2]):
            ins_long = find_long_insertions_and_write_slice(am.data, data.insertion_lens[r, :, i], data.insertion_start[r, :, i], f"ins_{r}_{i}", directory, verbose=verbose)
            insertions_long[-1].append(ins_long)
    left_flank_long = find_long_insertions_and_write_slice(am.data, data.left_flank_len, data.left_flank_start, "left_flank", directory, verbose=verbose)
    right_flank_long = find_long_insertions_and_write_slice(am.data, data.right_flank_len, data.right_flank_start, "right_flank", directory, verbose=verbose)
    unannotated_long = []
    for r in range(data.num_repeats-1):
        unannotated_long.append(find_long_insertions_and_write_slice(am.data, data.unannotated_segments_len[r], data.unannotated_segments_start[r], f"unannotated_{r}", directory, verbose=verbose))
        
    slice_files = []
    if left_flank_long is not None:
        slice_files.append(f"{directory}/slice_left_flank")
    if right_flank_long is not None:
        slice_files.append(f"{directory}/slice_right_flank")
    for r in range(data.num_repeats):
        for i in range(data.insertion_lens.shape[2]):
            if insertions_long[r][i] is not None:
                slice_files.append(f"{directory}/slice_ins_{r}_{i}")
    for r in range(data.num_repeats-1):
        if unannotated_long[r] is not None:
            slice_files.append(f"{directory}/slice_unannotated_{r}")
            
    make_slice_msas(slice_files, method, threads)
        
    #merge msa
    insertions_long = [[(x[0], AlignedDataset(x[1]+".aln", "fasta")) if x is not None else None for x in repeats] for repeats in insertions_long]
    left_flank_long = (left_flank_long[0],  AlignedDataset(left_flank_long[1]+".aln", "fasta")) if left_flank_long is not None else None
    right_flank_long = (right_flank_long[0],  AlignedDataset(right_flank_long[1]+".aln", "fasta")) if right_flank_long is not None else None
    unannotated_long = [(x[0], AlignedDataset(x[1]+".aln", "fasta")) if x is not None else None for x in unannotated_long]
    
    aligned_insertions = alignment_model.AlignedInsertions(insertions_long, left_flank_long, right_flank_long, unannotated_long)
    return aligned_insertions


def make_slice_msas(slice_files, method="famsa", threads=0):
    if method == "famsa":
        result_code = align_with_famsa(slice_files, threads)
    elif method == "clustalo":
        result_code = align_with_clustalo(slice_files, threads)
    elif method == "t_coffee":
       result_code = align_with_t_coffee(slice_files, threads)
    else:
        print(f"Unknown aligner {method}")
        sys.exit(1)
    if result_code != 0:
        print(f"Failed to align insertions with {method}. Aligner returned: {result.returncode}.")
        sys.exit(1)


def align_with_famsa(slice_files, threads):
    #from pyfamsa import Aligner, Sequence
    #fasta_file = fasta.Fasta(filename)
    #sequences = [Sequence(fasta_file.seq_ids[i], fasta_file.aminoacid_seq_str(i)) for i in range(fasta_file.num_seq)]
    if which("famsa") is None:
        print("Aligner famsa is not installed or not in PATH. Consider installing it with conda install -c bioconda famsa.")
        sys.exit(1)
    for slice_file in slice_files:
        result = subprocess.run(["famsa", "-t", str(threads), slice_file, slice_file+".aln"])
    return result.returncode


def align_with_clustalo(slice_files, threads):
    if which("clustalo") is None:
        print("Aligner clustalo is not installed or not in PATH. Consider installing it with conda install -c bioconda clustalo.")
        sys.exit(1)
    for slice_file in slice_files:
        if threads:
            result = subprocess.run(["clustalo", "--threads", str(threads), "-i", slice_file, "-o", slice_file+".aln", "--force"])
        else:
            result = subprocess.run(["clustalo", "-i", slice_file, "-o", slice_file+".aln", "--force"])
    return result.returncode


def align_with_t_coffee(slice_files, threads):
    if which("t_coffee") is None:
        print("Aligner t_coffee is not installed or not in PATH. Consider installing it with conda install -c bioconda t-coffee."+
                "Note: As of writing this, there is also an outdated conda package called t_coffee (underscore). Please make sure to install the correct one.")
        sys.exit(1)
    for slice_file in slice_files:
        result = subprocess.run(["t_coffee", "-thread", str(threads), "-reg", "-seq", slice_file, 
                                        "-nseq", "100", "-tree", "mbed", "-method", "mafftginsi_msa", 
                                        "-outfile", slice_file+".aln", "-quiet"])
    return result.returncode