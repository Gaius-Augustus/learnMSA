import numpy as np
import learnMSA.msa_hmm.Fasta as fasta
import learnMSA.msa_hmm.AlignmentModel as alignment_model
import subprocess
from shutil import which
import sys


def find_long_insertions_and_write_slice(fasta_file, lens, starts, name, directory, t = 20, k=2, max_insertions_len=500, verbose=True):
        """
        Finds insertions that have at least length t. If there are at least k of these sequences, writes them to file.
        Args: 
            fasta_file: Fasta file containing the complete sequences.
            lens, starts: Arrays of length n where n is the number of sequences in fasta_file. Indicate how long insertions are and where they start respectively.
            name: Identifier for the location of the slice (e.g. left_flank or match_5).
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
                    aa_seq = fasta_file.aminoacid_seq_str(which[j])
                    segment = aa_seq[start[j] : start[j] + lengths[j]]
                    only_non_standard_aa = True
                    for aa in fasta.alphabet[:20]:
                        if aa in segment:
                            only_non_standard_aa = False
                            break
                    if only_non_standard_aa or lengths[j] > max_insertions_len:
                        to_delete.append(j)
                    else:
                        slice_file.write(">"+fasta_file.seq_ids[which[j]]+"\n")
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
            ins_long = find_long_insertions_and_write_slice(am.fasta_file, data.insertion_lens[r, :, i], data.insertion_start[r, :, i], f"ins_{r}_{i}", directory, verbose=verbose)
            insertions_long[-1].append(ins_long)
    left_flank_long = find_long_insertions_and_write_slice(am.fasta_file, data.left_flank_len, data.left_flank_start, "left_flank", directory, verbose=verbose)
    right_flank_long = find_long_insertions_and_write_slice(am.fasta_file, data.right_flank_len, data.right_flank_start, "right_flank", directory, verbose=verbose)
    unannotated_long = []
    for r in range(data.num_repeats-1):
        unannotated_long.append(find_long_insertions_and_write_slice(am.fasta_file, data.unannotated_segments_len[r], data.unannotated_segments_start[r], f"unannotated_{r}", directory, verbose=verbose))
        
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
            
    #align insertions
    for slice_file in slice_files:
        make_slice_msa(slice_file, method, threads)
        
    #merge msa
    insertions_long = [[(x[0], fasta.Fasta(x[1]+".aln", aligned=True)) if x is not None else None for x in repeats] for repeats in insertions_long]
    left_flank_long = (left_flank_long[0],  fasta.Fasta(left_flank_long[1]+".aln", aligned=True)) if left_flank_long is not None else None
    right_flank_long = (right_flank_long[0],  fasta.Fasta(right_flank_long[1]+".aln", aligned=True)) if right_flank_long is not None else None
    unannotated_long = [(x[0], fasta.Fasta(x[1]+".aln", aligned=True)) if x is not None else None for x in unannotated_long]
    
    aligned_insertions = alignment_model.AlignedInsertions(insertions_long, left_flank_long, right_flank_long, unannotated_long)
    return aligned_insertions


def make_slice_msa(slice_file, method="famsa", threads=0):
    
    if method == "famsa":
        if which(method) is None:
            print("Aligner famsa is not installed or not in PATH. Consider installing it with conda install -c bioconda famsa.")
            sys.exit(1)
        else:
            result = subprocess.run(["famsa", "-t", str(threads), slice_file, slice_file+".aln"])
            
    elif method == "clustalo":
        if which(method) is None:
            print("Aligner clustalo is not installed or not in PATH. Consider installing it with conda install -c bioconda clustalo.")
            sys.exit(1)
        else:
            if threads:
                result = subprocess.run(["famsa", "--threads", str(threads), "-i", slice_file, "-o", slice_file+".aln", "--force"])
            else:
                result = subprocess.run(["clustalo", "-i", slice_file, "-o", slice_file+".aln", "--force"])
                
    elif method == "t_coffee":
        if which(method) is None:
            print("Aligner t_coffee is not installed or not in PATH. Consider installing it with conda install -c bioconda t-coffee."+
                  "Note: As of writing this, there is also an outdated conda package called t_coffee (underscore). Please make sure to install the correct one.")
            sys.exit(1)
        else:
            result = subprocess.run(["t_coffee", "-thread", str(threads), "-reg", "-seq", slice_file, 
                                     "-nseq", "100", "-tree", "mbed", "-method", "mafftginsi_msa", 
                                     "-outfile", slice_file+".aln", "-quiet"])
    else:
        print(f"Unknown aligner {method}")
        sys.exit(1)

    if result.returncode != 0:
        print(f"Failed to align insertions with {method}. Aligner returned: {subprocess.returncode}.")
        sys.exit(1)