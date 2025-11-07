import os
import subprocess
import sys
from shutil import which

import numpy as np
import pandas as pd

from learnMSA.msa_hmm.SequenceDataset import SequenceDataset


#computes clustering based sequence weights if mmseqs2 is installed
def compute_sequence_weights(fasta_filename, directory, cluster_seq_id=0.5, return_clusters=False):
    if which("mmseqs") is None:
        print("mmseqs2 is not installed or not in path. Please install it (e.g. with conda install -c bioconda mmseqs2) or disable sequence weighting with --no_sequence_weights (not recommended).")
        sys.exit(1)
    else:
        cluster_files = directory+"/"+os.path.splitext(os.path.basename(fasta_filename))[0]
        command = ["mmseqs",
                    "easy-linclust",
                    fasta_filename,
                    cluster_files,
                    cluster_files+"_tmp",
                    "--cov-mode", "1",
                    "--cluster-mode", "2",
                    "--alignment-mode", "3",
                    "--kmer-per-seq", "100",
                    "--min-seq-id", str(cluster_seq_id),
                    "--remove-tmp-files", "true",
                    "-v", "1"]
        result = subprocess.run(command, check=True)
        clustering = pd.read_csv(cluster_files + "_cluster.tsv", sep="\t", names=["representative", "sequence"])
        cluster_counts = clustering.groupby("representative").size().to_frame("cluster_size")
        clustering = clustering.merge(cluster_counts, how="left", on="representative")
        clustering["weight"] = 1/clustering["cluster_size"]
        clustering["cluster_index"] = clustering.groupby("representative").ngroup()
        clustering = clustering.set_index("sequence")

        with SequenceDataset(fasta_filename, "fasta") as data:
            data.validate_dataset()
            # mmseqs2 omits database names and database specific accession numbers,
            # we have to omit them too
            # i.e. from ">database|accession|name" mmseqs only keeps ">name"
            # unless there is a whitespace in the id, then it strips everything
            # after that whitespace
            ids = [data.get_header(i) for i in range(data.num_seq)]
            for i in range(len(ids)):
                if " " in ids[i]:
                    pos = ids[i].find(" ")
                    ids[i] = ids[i][:pos]
                elif "|" in ids[i]:
                    pos = ids[i].rfind("|")
                    if pos != -1:
                        ids[i] = ids[i][pos+1:]
            sequence_weights = np.array(
                clustering.loc[ids].weight,
                dtype=np.float32
            )
        if return_clusters:
            clusters = np.array(
                clustering.loc[ids].cluster_index,
                dtype=np.int32
            )
            return sequence_weights, clusters
        else:
            return sequence_weights