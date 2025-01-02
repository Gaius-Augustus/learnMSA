import subprocess
import sys
import os
from shutil import which
import pandas as pd
import numpy as np
from learnMSA.msa_hmm.SequenceDataset import SequenceDataset

# tmp include
sys.path.insert(0, "../TensorTree")
from tensortree import TreeHandler


""" Computes a clustering of the sequences in a fasta file based on sequence identity using mmseqs2.
Args:
    fasta_filename: Path to the fasta file containing the sequences to cluster.
    directory: Working directory to store the clustering results.
    cluster_seq_id: Minimum sequence identity to consider two sequences as similar.
Returns:
    A pandas DataFrame with the sequences as index and the columns: representative, cluster_size, cluster_index.
"""
def compute_clustering(fasta_filename, directory="tmp", cluster_seq_id=0.5, linear=True):
    assert which("mmseqs") is not None, "Unable to find mmseqs2."
    cluster_files = directory+"/"+os.path.splitext(os.path.basename(fasta_filename))[0]
    # mmseqs2 settings are those recommended to cluster protein fragments (cov-mode 1),
    # but prevent fragments from being cluster representatives (cluster-mode 2, alignment-mode 3).
    # kmer-per-seq 100 is used to increase sensitivity compared to the default value of 20
    command = ["mmseqs", 
                "easy-linclust" if linear else "easy-cluster",
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
    clustering["cluster_index"] = clustering.groupby("representative").ngroup()
    clustering = clustering.set_index("sequence")
    return clustering




""" From a clustering, computes a rooted cluster tree that has the sequences as leaves and the clusters as internal nodes.
Args:
    clustering: A pandas DataFrame with the sequences as index and the columns: representative, cluster_size, cluster_index.
Returns:    
    A tree handler object representing the cluster tree. Use tree.to_newick() to get the newick string. 
    The leaves are labeled with the sequence ids and the internal nodes are labeled with prefixed representative ids.
"""
def cluster_tree(clustering, branch_length=1.):
    tree = TreeHandler()
    clusters = clustering.representative.unique()
    cluster_names = ["rep_"+str(cluster) for cluster in clusters]
    tree.split("ROOT", len(clusters), branch_length, cluster_names)
    tree.update() 
    for cluster in clusters:
        members = clustering.loc[clustering.representative == cluster].index
        tree.split("rep_"+str(cluster), len(members), branch_length, members)
    tree.update()
    return tree


""" Creates a star-like tree with the sequences as leaves and a single internal node as root.
"""
def star_tree(sequence_ids, branch_length=1.):
    tree = TreeHandler()
    tree.split("ROOT", len(sequence_ids), branch_length, sequence_ids)
    return tree


""" Creates cluster sets from a tree and a sequence dataset. 
Returns:
    clusters: A vector with the cluster for each sequence.
    leaves: A vector with the leaf for each sequence.
    cluster_nodes: A vector with the cluster node for each cluster.
"""
def create_cluster_sets(data : SequenceDataset, tree_handler : TreeHandler):
    clusters = np.zeros(data.num_seq, dtype=np.int32)
    # tree_handler must have leaves matching with data.seq_ids
    leaf_names = tree_handler.node_names[:tree_handler.num_leaves]
    assert len(leaf_names) == data.num_seq, \
        "The tree handler must have the same number of leaves as the dataset."
    # map each sequence to its leaf index
    try:
        leaves = np.array([leaf_names.index(seq_id) for seq_id in data.seq_ids], dtype=np.int32)
    except ValueError as e:
        raise ValueError("The tree handler must have the same sequence ids as the dataset.") from e
    
    # get a vector that contains the index of the parent node for each leaf/sequence
    parent_indices = tree_handler.get_parent_indices_by_height(0)
    # the set of parent nodes of all leaves becomes the set of cluster nodes
    # map the cluster indices to indices of cluster nodes 
    cluster_nodes = np.unique(parent_indices)
    for i,j in enumerate(cluster_nodes):
        clusters[parent_indices == j] = i
    clusters = clusters[leaves]
    return clusters, leaves, cluster_nodes 


#computes clustering based sequence weights if mmseqs2 is installed
def compute_sequence_weights(fasta_filename, directory, cluster_seq_id=0.5, return_clusters=False):
    clustering = compute_clustering(fasta_filename, directory, cluster_seq_id)
    clustering["weight"] = 1/clustering["cluster_size"]
    with SequenceDataset(fasta_filename, "fasta") as data:
        data.validate_dataset()
        ids = data.seq_ids
        #mmseqs2 omits database names and database specific accession numbers
        #i.e. from ">database|accession|name" mmseqs only keeps ">name"
        for i in range(len(ids)):
            if "|" in ids[i]:
                pos = ids[i].rfind("|")
                if pos != -1:
                    ids[i] = ids[i][pos+1:]
        sequence_weights = np.array(clustering.loc[ids].weight, dtype=np.float32)
    if return_clusters:
        clusters = np.array(clustering.loc[ids].cluster_index, dtype=np.int32)
        return sequence_weights, clusters
    else:
        return sequence_weights
    
