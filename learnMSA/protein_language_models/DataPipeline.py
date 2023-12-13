import os
import numpy as np
import sys
sys.path.insert(1, "../..")
import learnMSA.msa_hmm.SequenceDataset as SequenceDataset
import learnMSA.protein_language_models.Common as Common
import pandas as pd
import tensorflow as tf



PFAM_PATH = "../../../../PFAM"



#preprocessing
def get_family(filepath):
    return ".".join(os.path.basename(filepath).split(".")[:-1])


def get_clan_data(drop_overlap = True #remove families that have at least one sequence with at least 50% id to a homfam sequence
                 ):
    clans_df = pd.read_csv(PFAM_PATH+"/Pfam-A.clans.tsv", header=None, sep="\t")
    clans_df[1] = clans_df[1].fillna(clans_df[0]) #families with no clan become their own clans
    clans_df.set_index(0, inplace=True)
    clans_df.drop([2,3,4], axis=1, inplace=True)
    clans_df.rename(columns={1 : "clan"}, inplace=True)

    print(f"{len(clans_df)} families in total")
    print(f"{clans_df.clan.unique().size} clans in total")

    np.random.seed(77)

    #load all fasta files (takes a while)
    datasets = PFAM_PATH+"/alignments/"
    ext = ".fasta"

    clans_df["fasta_index"] = np.nan

    # load all ref alignments
    fasta_files = []
    if drop_overlap:
        with open("pfam_homfam_overlap.txt", "r") as file:
            to_drop = file.read().splitlines()
    else:
        to_drop = []
    for file in os.listdir(datasets):
        if file.endswith(ext):
            family = ".".join(file.split(".")[:-1])
            if family in to_drop: #skip if family has sequences similar to those in the test dataset
                continue
            fasta = SequenceDataset.AlignedDataset(datasets+file, single_seq_ok=True)
            #omit families with only one sequence
            if fasta.num_seq == 1:
                to_drop.append(family)
            else:
                fasta_files.append(fasta)
                clans_df.loc[family, "fasta_index"] = len(fasta_files)-1

    #drop families
    clans_df = clans_df.drop(to_drop, axis=0)

    assert not clans_df.isna().values.any()
    clans_df = clans_df.astype({"fasta_index": "int32"})

    #unique_clans defines the unique order of clans for batch sampling
    unique_clans = clans_df.clan.unique()
    print(f"{len(clans_df)} families after dropping")
    print(f"{unique_clans.size} clans after dropping")
    print(f"{sum(f.num_seq for f in fasta_files)} sequences")

    fasta_dict = {get_family(f.filename) : f for f in fasta_files}

    clan_sizes = np.array([np.sum(clans_df.clan == clan) for clan in unique_clans])
    #clan families defines the unique order of families within each clan for batch sampling
    clan_families = [clans_df[clans_df.clan == clan].index for clan in unique_clans]
    
    return clans_df, unique_clans, fasta_dict, clan_sizes, clan_families


def _get_features_labels(fasta, rand, max_len, force_segment_start=None):
    """ 
    Picks one sequence and it's corresponding mapping to alignment columns from the fasta file.
    """
    i = int(np.floor(rand * fasta.num_seq))
    seq = fasta.get_standardized_seq(i)
    s = fasta.starting_pos[i]
    if fasta.seq_lens[i] <= max_len:
        t = s + fasta.seq_lens[i]
    elif force_segment_start is not None:
        s += force_segment_start
        t = s + max_len
    else: #random segment
        s += np.random.randint(fasta.seq_lens[i] - max_len) 
        t = s + max_len
    crop_start = s > fasta.starting_pos[i]
    crop_end = t < fasta.starting_pos[i] + fasta.seq_lens[i]
    seq = seq[s - fasta.starting_pos[i] : t - fasta.starting_pos[i]]
    pos_to_col = fasta.column_map[s:t]
    return seq, pos_to_col, (crop_start, crop_end)


def _get_column_occupancies(fasta : SequenceDataset.AlignedDataset):
    """ 
    Returns a boolean mask indicating which columns of the alignment are match columns.
    """
    cols, counts = np.unique(fasta.column_map, return_counts=True)
    column_occupancies = counts[np.argsort(cols)] / fasta.num_seq
    return column_occupancies


def _make_batch(batch_clans, rand_family, rand_seq, max_len, fasta_dict, clan_sizes, clan_families):
    batch_families = np.floor(rand_family * clan_sizes[batch_clans]).astype(batch_clans.dtype)
    seq1, seq2, labels = [], [], []
    crop_1, crop_2 = [], []
    for c,f,r in zip(batch_clans, batch_families, rand_seq):
        f_name = clan_families[c][f]
        s1, c1, cr1 = _get_features_labels(fasta_dict[f_name], r[0], max_len)
        s2, c2, cr2 = _get_features_labels(fasta_dict[f_name], r[1], max_len)
        seq1.append(s1)
        seq2.append(s2)
        labels.append((c1[:, np.newaxis] == c2[np.newaxis, :]).astype(np.float32))
        crop_1.append(cr1)
        crop_2.append(cr2)
    return seq1, seq2, labels, np.array(crop_1), np.array(crop_2)


def _sample_batch(clans, batch_size, max_len, fasta_dict, clan_sizes, clan_families):
    """ Generates a batch of training examples where one example is generated as:
        1. Sample a random clan
        2. Sample a random family from this clan
        3. Sample 2 random sequences from this family
        4. Compute the label matrix that has the 2 sequence lengths as rows/columns 
            and holds a 1 where residues share the same alignment column.
    Returns:
        (query_seqs, target_seqs), labels
        Where query_seqs and target_seqs are batches of sequences (b, L1), (b, L2) 
        and labels are of shape (b, L1, L2). A -1 in the labels indicates an invalid position.
    """
    batch_clans = np.random.choice(clans, size=batch_size)
    rand_family = np.random.rand(batch_size) #sample from [0,1] first as the family sizes differ
    rand_seq = np.random.rand(batch_size, 2) #sample from [0,1] again as the sequence numbers per family differ
    return _make_batch(batch_clans, rand_family, rand_seq, max_len, fasta_dict, clan_sizes, clan_families)


def _make_unsupervised_batch(batch_clans, rand_family, rand_seq, max_len, fasta_dict, clan_sizes, clan_families):
    batch_families = np.floor(rand_family * clan_sizes[batch_clans]).astype(batch_clans.dtype)
    seq, crop = [], []
    for c,f,r in zip(batch_clans, batch_families, rand_seq):
        f_name = clan_families[c][f]
        s, _, cr = _get_features_labels(fasta_dict[f_name], r, max_len)
        seq.append(s)
        crop.append(cr)
    return seq, crop


def _sample_unsupervised_batch(clans, batch_size, max_len, fasta_dict, clan_sizes, clan_families):
    """ Generates a batch of training examples where one example is generated as:
        1. Sample a random clan
        2. Sample a random family from this clan
        3. Sample a random sequence from this family
    Returns:
        sequences (b, L)
    """
    batch_clans = np.random.choice(clans, size=batch_size)
    rand_family = np.random.rand(batch_size) #sample from [0,1] first as the family sizes differ
    rand_seq = np.random.rand(batch_size) #sample from [0,1] again as the sequence numbers per family differ
    return _make_unsupervised_batch(batch_clans, rand_family, rand_seq, max_len, fasta_dict, clan_sizes, clan_families)


def _make_column_prior_batch(batch_clans, rand_family, rand_seq, max_len, fasta_dict, clan_sizes, clan_families):
    batch_families = np.floor(rand_family * clan_sizes[batch_clans]).astype(batch_clans.dtype)
    seq, crop, match_masks = [], [], []
    for c,f,r in zip(batch_clans, batch_families, rand_seq):
        f_name = clan_families[c][f]
        fasta = fasta_dict[f_name]
        s, cols, cr = _get_features_labels(fasta, r, max_len)
        column_occupancies = _get_column_occupancies(fasta)
        match_mask = (column_occupancies > 0.5)[cols].astype(np.float32)
        seq.append(s)
        crop.append(cr)
        match_masks.append(match_mask)
    return seq, crop, match_masks


def _sample_column_prior_batch(clans, batch_size, max_len, fasta_dict, clan_sizes, clan_families):
    """ Generates a batch of training examples where one example is generated as:
        1. Sample a random clan
        2. Sample a random family from this clan
        3. Sample a random sequence from this family
        4. Compute a mask indicating which sequence positions relate to match state columns.
    Returns:
        sequences (b, L), match_positions (b, L)
    """
    batch_clans = np.random.choice(clans, size=batch_size)
    rand_family = np.random.rand(batch_size) #sample from [0,1] first as the family sizes differ
    rand_seq = np.random.rand(batch_size) #sample from [0,1] again as the sequence numbers per family differ
    return _make_column_prior_batch(batch_clans, rand_family, rand_seq, max_len, fasta_dict, clan_sizes, clan_families)


def _get_maxlen(seqs):
    return max(len(s) for s in seqs)


def _tokenize(encoder : Common.InputEncoder, seq1, seq2, labels, crop_1, crop_2):
    input1 = encoder(seq1, crop_1)
    input2 = encoder(seq2, crop_2)
    #merge everything in padded tensors
    batched_labels = np.zeros((len(seq1), _get_maxlen(seq1), _get_maxlen(seq2)), dtype=np.float32)
    for i, (label, s1, s2) in enumerate(zip(labels, seq1, seq2)):
        batched_labels[i, :len(s1), :len(s2)] = label
    return (input1, input2), batched_labels


def _tokenize_unsupervised(encoder : Common.InputEncoder, seq, crop):
    return encoder(seq, crop)


def _tokenize_column_prior(encoder : Common.InputEncoder, seq, crop, match_mask_list):
    input = encoder(seq, crop)
    # match mask is a list of sequences with different length
    # combine them in a padded tensor
    match_mask = np.zeros((len(match_mask_list), max(mm.size for mm in match_mask_list)), dtype=np.float32)
    for i,mm in enumerate(match_mask_list):
        match_mask[i, :mm.size] = mm
    return input, match_mask


def prepare_unshuffled_pairs(clans, fasta_dict, clan_families):
    num_pairs_per_family = [fasta_dict[f[0]].num_seq ** 2 for f in clan_families]
    batch_clans = np.repeat(clans, num_pairs_per_family)
    batch_families = np.zeros_like(batch_clans)
    batch_seqs = []
    for f in clan_families:
        n = fasta_dict[f[0]].num_seq
        s = np.zeros((n**2, 2))
        s[:,0] = np.repeat(np.arange(n, dtype=np.float32) / n, n)
        s[:,1] = np.tile(np.arange(n, dtype=np.float32) / n, n)
        batch_seqs.append(s)
    batch_seqs = np.concatenate(batch_seqs, axis=0)
    return batch_clans, batch_families, batch_seqs, num_pairs_per_family


def prepare_unshuffled_single(clans, fasta_dict, clan_families):
    num_seqs_per_family = [fasta_dict[f[0]].num_seq for f in clan_families]
    batch_clans = np.repeat(clans, num_seqs_per_family)
    batch_families = np.zeros_like(batch_clans)
    batch_seqs = []
    for f in clan_families:
        n = fasta_dict[f[0]].num_seq
        s = np.arange(n, dtype=np.float32) / n
        batch_seqs.append(s)
    batch_seqs = np.concatenate(batch_seqs, axis=0)
    return batch_clans, batch_families, batch_seqs, num_seqs_per_family


def make_dataset(encoder : Common.InputEncoder, clans, batch_size, max_len, fasta_dict, clan_sizes, clan_families, shuffled=True):
    if shuffled:
        def _gen_inputs():
            while True:
                yield _tokenize(encoder, *_sample_batch(clans, batch_size, max_len, fasta_dict, clan_sizes, clan_families))
    else: #currently only if clan size == 1 for all clans
        batch_clans, batch_families, batch_seqs, num_pairs_per_family = prepare_unshuffled_pairs(clans, fasta_dict, clan_families)
        def _gen_inputs():
            for i in range(sum(num_pairs_per_family)):
                yield _tokenize(encoder, *_make_batch(batch_clans[i*batch_size : (i+1)*batch_size], 
                                                       batch_families[i*batch_size : (i+1)*batch_size], 
                                                       batch_seqs[i*batch_size : (i+1)*batch_size], 
                                                       max_len, 
                                                       fasta_dict, 
                                                       clan_sizes, 
                                                       clan_families))
    output_signature = ((encoder.get_signature(), encoder.get_signature()), tf.TensorSpec(shape=(None, None, None), dtype=tf.float32))
    ds = tf.data.Dataset.from_generator(_gen_inputs, output_signature = output_signature)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    if shuffled:
        return ds
    else:
        return ds, sum(num_pairs_per_family)//batch_size+1


def make_unsupervised_dataset(encoder : Common.InputEncoder, clans, batch_size, max_len, fasta_dict, clan_sizes, clan_families):
    def _gen_inputs():
        while True:
            yield _tokenize_unsupervised(encoder, *_sample_unsupervised_batch(clans, batch_size, max_len, fasta_dict, clan_sizes, clan_families))
    ds = tf.data.Dataset.from_generator(_gen_inputs, output_signature = encoder.get_signature())
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def make_column_prior_dataset(encoder : Common.InputEncoder, clans, batch_size, max_len, fasta_dict, clan_sizes, clan_families, shuffled=True):
    if shuffled:
        def _gen_inputs():
            while True:
                yield _tokenize_column_prior(encoder, *_sample_column_prior_batch(clans, batch_size, max_len, fasta_dict, clan_sizes, clan_families))
    else: #currently only if clan size == 1 for all clans
        batch_clans, batch_families, batch_seqs, num_seqs_per_family = prepare_unshuffled_single(clans, fasta_dict, clan_families)
        def _gen_inputs():
            for i in range(sum(num_seqs_per_family)):
                yield _tokenize_column_prior(encoder, *_make_column_prior_batch(batch_clans[i*batch_size : (i+1)*batch_size], 
                                                                                   batch_families[i*batch_size : (i+1)*batch_size], 
                                                                                   batch_seqs[i*batch_size : (i+1)*batch_size], 
                                                                                   max_len, 
                                                                                   fasta_dict, 
                                                                                   clan_sizes, 
                                                                                   clan_families))
    ds = tf.data.Dataset.from_generator(_gen_inputs, output_signature = ((encoder.get_signature()), tf.TensorSpec(shape=(None, None), dtype=tf.float32)))
    ds = ds.prefetch(tf.data.AUTOTUNE)
    if shuffled:
        return ds
    else:
        return ds, sum(num_seqs_per_family)//batch_size+1


def make_homfam_dataset(encoder : Common.InputEncoder, 
                        batch_size, 
                        homfam_path, 
                        ext=".ref",
                        for_prior=False):
    fasta_dict = {}
    clan_families = []
    for file in os.listdir(homfam_path):
        if file.endswith(ext):
            fasta = SequenceDataset.AlignedDataset(homfam_path + file)
            family = get_family(file)
            fasta_dict[family] = fasta
            clan_families.append([family])
    assert len(fasta_dict) == 94
    clans = np.arange(94)
    clan_sizes = np.ones(94)
    if for_prior:
        return make_column_prior_dataset(encoder, clans, batch_size, max([f.num_seq for f in fasta_dict.values()]), fasta_dict, clan_sizes, clan_families, shuffled=False)
    else:
        return make_dataset(encoder, clans, batch_size, max([f.num_seq for f in fasta_dict.values()]), fasta_dict, clan_sizes, clan_families, shuffled=False)


def make_random_data(emb_dim, batch_size, steps=100, loc=0., scale=0.6):
    def _gen_random_inputs():
        for i in range(steps):
            random_emb = np.random.normal(loc=loc, scale=scale, size=(batch_size, 100, emb_dim)).astype(np.float32)
            match_mask = np.ones_like(random_emb[...,0])
            yield random_emb, match_mask
    ds = tf.data.Dataset.from_generator(_gen_random_inputs, output_signature = (tf.TensorSpec(shape=(None, None, emb_dim), dtype=tf.float32), tf.TensorSpec(shape=(None, None), dtype=tf.float32)))
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds