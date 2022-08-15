import tensorflow as tf
import numpy as np
import time
import os
import learnMSA.msa_hmm.Utility as ut
import learnMSA.msa_hmm.Fasta as fasta
import learnMSA.msa_hmm.Training as train
import learnMSA.msa_hmm.MsaHmmLayer as msa_hmm
from learnMSA.msa_hmm.Configuration import as_str


@tf.function
def viterbi_recursion_step(gamma_prev, sequences_i, log_A_val, B, indices_0, indices_1, epsilon):
    n = sequences_i.shape[0]
    m = B.shape[0]
    gamma_subset = tf.gather(gamma_prev, indices_0, axis=1)
    a = tf.math.unsorted_segment_max(log_A_val + gamma_subset, indices_1, n*m) 
    a = tf.reshape(a, (n, m))
    b = tf.linalg.matvec(B, sequences_i)
    b = tf.math.log(b+epsilon)
    return a + b 


# implements the log-version to prevent underflow
# utilizes sparse matrix format for a speedup
def viterbi_dyn_prog(sequences, hmm_cell, epsilon=np.finfo(np.float64).tiny):
    A = hmm_cell.make_A_sparse()
    B = hmm_cell.make_B()
    init = hmm_cell.make_initial_distribution()
    n = sequences.shape[0]
    m = A.shape[0]
    len_seq = sequences.shape[1]
    log_A_val = tf.expand_dims(tf.math.log(A.values), 0)
    indices_0 = A.indices[:,0]
    grid = tf.meshgrid(tf.range(n, dtype=tf.int64) * m,
                       A.indices[:,1], indexing='ij')
    indices_1 = tf.add_n(grid)
    b0 = tf.linalg.matvec(B, sequences[:,0])
    gamma = [tf.math.log(init+epsilon) + tf.math.log(b0+epsilon)]  
    for i in range(1,len_seq):
        gamma_next = viterbi_recursion_step(gamma[-1], 
                                            sequences[:,i], 
                                            log_A_val, B, indices_0, indices_1, epsilon)
        gamma.append(gamma_next) 
    gamma = tf.stack(gamma, axis=1)
    return gamma


@tf.function
def viterbi_backtracking_step(q, gamma_state, log_A_dense):
    A_q = tf.transpose(tf.gather(log_A_dense, q, axis=-1))
    return tf.math.argmax(A_q + gamma_state, axis=-1)

    
def viterbi_backtracking(hmm_cell, gamma, epsilon=np.finfo(np.float64).tiny):
    A = hmm_cell.make_A_sparse()
    n = gamma.shape[0]
    l = gamma.shape[1]
    log_A_dense = tf.math.log(tf.sparse.to_dense(A) + epsilon)
    state_seqs_max_lik = []
    q = tf.math.argmax(gamma[:,-1], axis=-1)
    for i in range(l):
        q = viterbi_backtracking_step(q, gamma[:,l-i-1], log_A_dense)
        state_seqs_max_lik.insert(0, q)
    state_seqs_max_lik = tf.stack(state_seqs_max_lik, axis=1)
    return tf.cast(state_seqs_max_lik, dtype=fasta.ind_dtype)


# returns the most likely sequence of hidden states given unaligned sequences and a model
# pos. i of the returned sequences is the active state of the HMM after observing pos. i of the input
# runs in batches to avoid memory overflow for large data
def viterbi(sequences, hmm_cell, batch_size=64):
    k = 0
    gamma = np.zeros((sequences.shape[0], sequences.shape[1], hmm_cell.num_states), dtype=np.float64)
    while k < sequences.shape[0]:
        if len(sequences.shape) == 2:
            gamma[k:k+batch_size] = viterbi_dyn_prog(tf.one_hot(sequences[k:k+batch_size], fasta.s, dtype=ut.dtype), hmm_cell)
        else:
            gamma[k:k+batch_size] = viterbi_dyn_prog(sequences[k:k+batch_size], hmm_cell)
        k+=batch_size
       # tf.keras.backend.clear_session()
    return viterbi_backtracking(hmm_cell, gamma).numpy()


# runs batch-wise viterbi on all sequences in fasta_file specified by indices
# returns a dense representation of the most likely state sequences
def get_state_seqs_max_lik(fasta_file, 
                           indices, 
                           batch_size,
                           msa_hmm_cell, 
                           anc_probs_layer = None):
    ds = train.make_dataset(fasta_file, batch_size, False, indices)
    seq_len = np.amax(fasta_file.seq_lens[indices])+1
    state_seqs_max_lik = np.zeros((indices.size, seq_len), 
                                      dtype=np.uint16) + (2*msa_hmm_cell.length+2)
    i = 0                              
    for (seq, mask, ind), _ in ds:
        if anc_probs_layer is not None:
            seq = anc_probs_layer(seq, mask, ind)  
        state_seqs_max_lik_batch = viterbi(seq, msa_hmm_cell, batch_size) 
        b,l = state_seqs_max_lik_batch.shape
        state_seqs_max_lik[i:i+b, :l] = state_seqs_max_lik_batch
        i += b
    return state_seqs_max_lik


#given sequences, state sequences, model length and the current positions,
#returns decoded consensus columns as a matrix as well as insertions, lengths and starting positions
#in the sequences
def decode_core(model_length,
                state_seqs_max_lik,
                indices):
    n = state_seqs_max_lik.shape[0]
    c = model_length #alias for code readability
    #initialize the consensus with gaps
    consensus_columns = -np.ones((n, c), dtype=np.int16) 
    #insertion lengths and starting positions per sequence
    insertion_lens = np.zeros((n, c-1), dtype=np.int16)
    insertion_start = -np.ones((n, c-1), dtype=np.int16)
    #is true if and only if the previous hidden state was an insertion state (not counting flanks)
    last_insert = np.zeros(n, dtype=bool)
    A = np.arange(n)
    while True:
        q = state_seqs_max_lik[A, indices]
        is_match = ((q > 0) & (q < c+1))
        is_insert = ((q >= c+1) & (q < 2*c))
        is_insert_start = is_insert & ~last_insert
        is_unannotated = (q == 2*c)
        is_at_end = ((q == 2*c+1) | (q == 2*c+2))
        if np.all(is_unannotated | is_at_end):
            finished = ~is_unannotated
            break
        # track matches
        consensus_columns[A[is_match], q[is_match]-1] = indices[is_match]
        # track insertions
        is_insert_subset = A[is_insert]
        is_insert_start_subset = A[is_insert_start]
        insertion_lens[is_insert_subset, q[is_insert]-c-1] += 1
        insertion_start[is_insert_start_subset, q[is_insert_start]-c-1] = indices[is_insert_start]
        indices[is_match | is_insert] += 1
        last_insert = is_insert
    return consensus_columns, insertion_lens, insertion_start, finished


# an insertion state, active as long as at least one sequence remains in a flank/unannotated state 
# returns insertion length and starting position in each sequence
def decode_flank(state_seqs_max_lik, 
                 flank_state_id, 
                 indices):
    n = state_seqs_max_lik.shape[0]
    insertion_start = np.copy(indices)
    while True:
        q = state_seqs_max_lik[np.arange(n), indices]
        is_flank = (q == flank_state_id)
        if ~np.any(is_flank):
            break
        indices[is_flank] += 1
    insertion_lens = indices - insertion_start
    return insertion_lens, insertion_start


# decodes an implicit alignment from most likely state sequences
# implicit here means that insertions are stored as pairs (starting index, length)
# returns a representation of the consensus, insertion start positions and lengths, as well
# as an indicator for finished sequences
def decode(model_length, state_seqs_max_lik):
    n = state_seqs_max_lik.shape[0]
    c = model_length #alias for code readability
    indices = np.zeros(n, np.int16) # active positions in the sequence
    left_flank = decode_flank(state_seqs_max_lik, 0, indices) 
    core_blocks = []
    unannotated_segments = []
    while True:    
        C, IL, IS, finished = decode_core(model_length, state_seqs_max_lik, indices)
        core_blocks.append((C, IL, IS, finished))
        if np.all(finished):
            break
        unannotated_segments.append( decode_flank(state_seqs_max_lik, 2*c, indices) )
    right_flank = decode_flank(state_seqs_max_lik, 2*c+1, indices) 
    return core_blocks, left_flank, right_flank, unannotated_segments


def get_insertion_block(sequences, 
                        lens, 
                        maxlen,
                        starts,
                        align_to_right=False):
    A = np.arange(sequences.shape[0])
    block = np.zeros((sequences.shape[0], maxlen), dtype=np.uint8) + (fasta.s-1)
    lens = np.copy(lens)
    active = lens > 0
    i = 0
    while np.any(active):
        aa = sequences[A[active], starts[active] + i]
        block[active, i] = aa
        lens -= 1
        active = lens > 0
        i += 1
    if align_to_right:
        block_right_aligned = np.zeros_like(block) + (fasta.s-1)
        for i in range(maxlen):
            block_right_aligned[A, (maxlen-lens+i)%maxlen] = block[:, i]
        block = block_right_aligned
    block += fasta.s #lower case
    return block
    

def get_alignment_block(sequences, 
                        consensus, 
                        ins_len, 
                        ins_len_total,
                        ins_start):
    A = np.arange(sequences.shape[0])
    length = consensus.shape[1] + np.sum(ins_len_total)
    block = np.zeros((sequences.shape[0], length), dtype=np.uint8) + (fasta.s-1)
    i = 0
    for c in range(consensus.shape[1]-1):
        column = consensus[:,c]
        ins_l = ins_len[:,c]
        ins_l_total = ins_len_total[c]
        ins_s = ins_start[:,c]
        #one column
        no_gap = column != -1
        block[no_gap,i] = sequences[A[no_gap],column[no_gap]]
        i += 1
        #insertion
        block[:,i:i+ins_l_total] = get_insertion_block(sequences,
                                                       ins_l,
                                                       ins_l_total, 
                                                       ins_s)
        i += ins_l_total
    #final column
    no_gap = consensus[:,-1] != -1
    block[no_gap,i] = sequences[A[no_gap],consensus[:,-1][no_gap]]
    return block



#constructs an (implicit) alignment given a fasta file of unaligned sequences and a trained model
class Alignment():
    def __init__(self, 
                 fasta_file,
                 indices, #(subset of) the sequences from the fasta to align
                 batch_size, #controls memory consumption of viterbi
                 model,
                 use_anc_probs=True,
                 gap_symbol="-",
                 gap_symbol_insertions=".",
                 build="eager"):
        self.fasta_file = fasta_file
        self.indices = indices
        self.batch_size = batch_size
        self.model = model
        if use_anc_probs:
            self.anc_probs_layer = model.layers[3]
            self.msa_hmm_layer = model.layers[4]
        else:
            self.anc_probs_layer = None
            self.msa_hmm_layer = model.layers[1]
        self.output_alphabet = np.array((fasta.alphabet[:-1] + 
                                        [gap_symbol] + 
                                        [aa.lower() for aa in fasta.alphabet[:-1]] + 
                                        [gap_symbol_insertions, "$"]))
        if build=="eager":
            self.build_alignment()
        else:
            self.built = False
            
            
    #computes an implicit alignment (without storing gaps)
    #eventually, an alignment with explicit gaps can be written 
    #in a memory friendly manner to file
    def build_alignment(self):
        state_seqs_max_lik = get_state_seqs_max_lik(self.fasta_file,
                                                    self.indices, 
                                                    self.batch_size,
                                                    self.msa_hmm_layer.C,
                                                    self.anc_probs_layer)
        (core_blocks, 
         left_flank, 
         right_flank, 
         unannotated_segments) = decode(self.msa_hmm_layer.length,
                                             state_seqs_max_lik)
        self.consensus = np.stack([C for C,_,_,_ in core_blocks])
        self.insertion_lens = np.stack([IL for _,IL,_,_ in core_blocks])
        self.insertion_start = np.stack([IS for _,_,IS,_ in core_blocks])
        self.finished = np.stack([f for _,_,_,f in core_blocks])
        self.left_flank_len = np.stack(left_flank[0])
        self.left_flank_start = np.stack(left_flank[1])
        self.right_flank_len = np.stack(right_flank[0])
        self.right_flank_start = np.stack(right_flank[1])
        if len(unannotated_segments) > 0:
            self.unannotated_segments_len = np.stack([l for l,_ in unannotated_segments])
            self.unannotated_segments_start = np.stack([s for _,s in unannotated_segments])
            self.unannotated_segment_lens_total = np.amax(self.unannotated_segments_len, axis=1)
        else:
            self.unannotated_segment_lens_total = 0
        self.num_repeats = self.consensus.shape[0]
        self.consensus_len = self.consensus.shape[1]
        self.left_flank_len_total = np.amax(self.left_flank_len)
        self.right_flank_len_total = np.amax(self.right_flank_len)
        self.insertion_lens_total = np.amax(self.insertion_lens, axis=1)
        self.alignment_len = (self.left_flank_len_total + 
                              self.consensus_len*self.num_repeats + 
                              np.sum(self.insertion_lens_total) + 
                              np.sum(self.unannotated_segment_lens_total) +
                              self.right_flank_len_total)
        self.built = True
                              
    
    #use only for low sequence numbers
    def to_string(self, batch_size=100000, add_block_sep=True):
        if not self.built:
            self.build_alignment()
        alignment_strings_all = []
        n = self.indices.size
        i = 0
        while i < n:
            batch_indices = np.arange(i, min(n, i+batch_size))
            batch_alignment = self.get_batch_alignment(batch_indices, add_block_sep)
            alignment_strings = self.batch_to_string(batch_alignment)
            alignment_strings_all.extend(alignment_strings)
            i += batch_size
        return alignment_strings_all
    
    
    def to_file(self, filepath, batch_size=100000, add_block_sep=False):
        with open(filepath, "w") as output_file:
            n = self.indices.size
            i = 0
            while i < n:
                batch_indices = np.arange(i, min(n, i+batch_size))
                batch_alignment = self.get_batch_alignment(batch_indices, add_block_sep)
                alignment_strings = self.batch_to_string(batch_alignment)
                for s, seq_ind in zip(alignment_strings, batch_indices):
                    seq_id = self.fasta_file.seq_ids[self.indices[seq_ind]]
                    output_file.write(">"+seq_id+"\n")
                    output_file.write(s+"\n")
                i += batch_size
    
    
    #returns a dense matrix representing a subset of sequences
    #as specified by batch_indices but with respect to the total alignment length
    #(i.e. the sub alignment can contain gap-only columns)
    def get_batch_alignment(self, batch_indices, add_block_sep):
        if not self.built:
            self.build_alignment()
        b = batch_indices.size
        sequences = np.zeros((b, self.fasta_file.max_len), dtype=np.uint16) + (fasta.s-1)
        for i,j in enumerate(batch_indices):
            l = self.fasta_file.seq_lens[self.indices[j]]
            sequences[i, :l] = self.fasta_file.get_raw_seq(self.indices[j])
        blocks = []  
        if add_block_sep:
            sep = np.zeros((b,1), dtype=np.uint16) + 2*fasta.s
        left_flank_block = get_insertion_block(sequences, 
                                               self.left_flank_len[batch_indices],
                                               self.left_flank_len_total,
                                               self.left_flank_start[batch_indices],
                                               align_to_right=True)
        blocks.append(left_flank_block)
        if add_block_sep:
            blocks.append(sep)
        for i in range(self.num_repeats):
            consensus = self.consensus[i]
            ins_len = self.insertion_lens[i]
            ins_start = self.insertion_start[i]
            ins_len_total = self.insertion_lens_total[i]
            alignment_block = get_alignment_block(sequences, 
                                                  consensus[batch_indices], 
                                                  ins_len[batch_indices], 
                                                  ins_len_total,
                                                  ins_start[batch_indices])
            blocks.append(alignment_block)
            if add_block_sep:
                blocks.append(sep)
            if i < self.num_repeats-1:
                unannotated_segment_l = self.unannotated_segments_len[i]
                unannotated_segment_s = self.unannotated_segments_start[i]
                unannotated_block = get_insertion_block(sequences, 
                                                        unannotated_segment_l[batch_indices],
                                                        self.unannotated_segment_lens_total[i],
                                                        unannotated_segment_s[batch_indices])
                blocks.append(unannotated_block)
                if add_block_sep:
                    blocks.append(sep)
        right_flank_block = get_insertion_block(sequences, 
                                               self.right_flank_len[batch_indices],
                                               self.right_flank_len_total,
                                               self.right_flank_start[batch_indices])
        blocks.append(right_flank_block)
        batch_alignment = np.concatenate(blocks, axis=1)
        return batch_alignment
    
    
    def batch_to_string(self, batch_alignment):
        alignment_arr = self.output_alphabet[batch_alignment]
        alignment_strings = [''.join(s) for s in alignment_arr]
        return alignment_strings
        
    
  
        
# Given an alignment, computes positions for match expansions and discards depending
# on the following criteria:
# A position is expanded, if an insertion occurs in at least ins_t % of cases or 
# at least k sequences have an insertion of length > ins_long 
# (i.e. filter for very frequent or long insertions)
# In the first case the position is expanded by the average insertion length (counting zeros).
# In the second case the position is expanded by ins_long.
#
# A position is discarded, if it is deleted in at least del_t % of the sequences of all domain blocks and 
# if it has a prior density value in the lower match_prior_threshold % of the range of prior density values.
# (note that as a consequence we keep very conserved positions even if they are deleted frequently in the sequences)
def get_discard_or_expand_positions(alignment, 
                                    #fraction of gaps beyond which a column can be discarded
                                    del_t=0.5, 
                                    #fraction of insertion openings beyond which additional matches are added
                                    ins_t=0.5, 
                                    #percentage of the mid of range prior value below which a match state can be discarded
                                    match_prior_threshold=0.5, 
                                    #insertion length threshold to detect suspiciously long insertions
                                    ins_long=32,
                                    k=2,
                                    verbose=False):
    n = alignment.indices.size
    r = alignment.num_repeats
    finished_early = np.sum(alignment.finished[:-1], axis=1, keepdims=True)
    num_repeats = r*n-np.sum(finished_early)
    
    #insertions
    #find the fraction of insertion openings in all domain blocks handling repeats as multiple independent hits
    block_ins = np.sum(alignment.insertion_lens > 0, axis=(0,1))
    block_ins_frac = block_ins / num_repeats
    left_ins_frac = np.mean(alignment.left_flank_len > 0)
    right_ins_frac = np.mean(alignment.right_flank_len  > 0)
    ins_frac = np.concatenate([[left_ins_frac], block_ins_frac, [right_ins_frac]], axis=0)
    #compute the average insertion lengths over all sequences/domain blocks 
    #include zeros but do not count "empty" domain hits of finished sequences
    block_ins_lens = np.sum(alignment.insertion_lens, axis=(0,1))
    block_ins_avg_lens = block_ins_lens/num_repeats
    left_ins_lens = np.mean(alignment.left_flank_len)
    right_ins_lens = np.mean(alignment.right_flank_len)
    ins_lens = np.concatenate([[left_ins_lens], block_ins_avg_lens, [right_ins_lens]], axis=0)
    ins_lens = np.ceil(ins_lens).astype(np.int32)
    expand1 = ins_frac > ins_t
    pos_expand1 = np.arange(ins_frac.size, dtype=np.int32)[expand1]
    expansion_lens1 = ins_lens[expand1]
    block_very_long = alignment.insertion_lens > ins_long
    block_very_long = np.minimum(np.sum(block_very_long, axis=0), 1) #clip multi domain cases in the same sequence
    block_very_long = np.sum(block_very_long, axis=0)
    left_very_long = np.sum(alignment.left_flank_len > ins_long)
    right_very_long = np.sum(alignment.right_flank_len  > ins_long)
    very_long = np.concatenate([[left_very_long], block_very_long, [right_very_long]], axis=0)
    expand2 = very_long >= k
    pos_expand2 = np.arange(very_long.size, dtype=np.int32)[expand2]
    expansion_lens2 = np.array([ins_long]*pos_expand2.size, dtype=np.int32)
    #resolve the potential overlap between the two position vectors
    pos_expand, unique_indices = np.unique(np.concatenate([pos_expand2, pos_expand1]), return_index=True) 
    #unique_indices points to the first occurence, therefore we take the entry in pos_expand2
    #over the one in pos_expand1 in case of a dublication
    expansion_lens = np.concatenate([expansion_lens2, expansion_lens1], axis=0)[unique_indices]
    
    #deletions
    #find fraction of gaps in all domain blocks handling repeats as multiple independent hits
    #and without counting "empty" hits of finished sequences
    del_no_finish = np.sum(alignment.consensus == -1, axis=1)
    del_no_finish[1:] -= finished_early
    del_frac = np.sum(del_no_finish, axis=0) / num_repeats
    pos_discard1 = np.arange(del_frac.size, dtype=np.int32)[del_frac > del_t]
    #find match states with low prior
    B = alignment.msa_hmm_layer.C.make_B()[1:alignment.msa_hmm_layer.length+1,:20]
    B /= tf.reduce_sum(B, axis=-1, keepdims=True)
    match_prior = alignment.msa_hmm_layer.emission_dirichlet_1.log_pdf(B).numpy()
    min_prior, max_prior = np.min(match_prior), np.max(match_prior)
    prior_threshold = min_prior + match_prior_threshold * (max_prior - min_prior)
    pos_discard2 = np.arange(match_prior.size, dtype=np.int32)[match_prior <= prior_threshold]
    pos_discard = np.intersect1d(pos_discard1, pos_discard2) 
    
    return pos_expand, expansion_lens, pos_discard


#applies discards and expansions simultaneously to a vector x
#all positions are with respect to the original vector without any modification
#replicates insert_value for the expansions
#assumes that del_marker is a value that does no occur in x
#returns a new vector with all modifications applied
def apply_mods(x, pos_expand, expansion_lens, pos_discard, insert_value, del_marker=-9999):
    #mark discard positions with del_marker, expand thereafter 
    #and eventually remove the marked positions
    x = np.copy(x)
    x[pos_discard] = del_marker
    rep_expand_pos = np.repeat(pos_expand, expansion_lens)
    x = np.insert(x, rep_expand_pos, insert_value, axis=0)
    if len(x.shape) == 2:
        x = x[np.any(x != del_marker, -1)]
    else:
        x = x[x != del_marker]
    return x


# makes updated pos_expand, expansion_lens, pos_discard vectors that fulfill:
#
# - each consecutive segment of discards from i to j is replaced with discards
#   from i+k-1 to j+k and an expansion of length 1 at i+k-1
#   edge cases that do not require an expansion:
#        replaced with discards from i+k to j+k if i+k == 0 and j+k < L-1
#        replaced with discards from i+k-1 to j+k-1 if i+k > 0 and j+k == L-1
#        replaced with discards from i+k to j+k-1 i+k == 0 and j+k == L-1
#
# - an expansion at position i by l is replaced by a discard at i+k-1 and an expansion by l+1 at i+k-1  
#   edge cases that do not require a discard:
#        replaced by an expansion by l at i+k if i+k == 0
#        replaced by an expansion by l at i+k-1 if i+k==L or i+k-1 is already in the discarded positions
#        if all positions are discarded (and the first expansion would add l match states to a model of length 0)
#        the length of the expansion is reduced by 1
#
# k can be any integer 
# L is the length of the array to which the indices of pos_expand and pos_discard belong
def extend_mods(pos_expand, expansion_lens, pos_discard, L, k=0):
    if pos_discard.size == L and pos_expand.size > 0:
        expansion_lens = np.copy(expansion_lens)
        expansion_lens[0] -= 1
    if pos_discard.size > 0:
        #find starting points of all consecutive segments of discards 
        pos_discard_shift = pos_discard + k
        diff = np.diff(pos_discard_shift, prepend=-1)
        diff_where = np.squeeze(np.argwhere(diff > 1))
        segment_starts = np.atleast_1d(pos_discard_shift[diff_where])
        new_pos_discard = np.insert(pos_discard_shift, diff_where, segment_starts-1)
        new_pos_discard = np.unique(new_pos_discard)
        if pos_discard_shift[-1] == L-1:
            new_pos_discard = new_pos_discard[:-1]
            segment_starts = segment_starts[:-1]
        new_pos_expand = segment_starts-1
        new_expansion_lens = np.ones(segment_starts.size, dtype=expansion_lens.dtype)
    else:
        new_pos_discard = pos_discard
        new_pos_expand = np.array([], dtype=pos_expand.dtype)
        new_expansion_lens = np.array([], dtype=expansion_lens.dtype)
    #handle expansions
    if pos_expand.size > 0:
        pos_expand_shift = pos_expand+k
        extend1 = pos_expand_shift > 0
        extend2 = pos_expand_shift < L
        _,indices,_ = np.intersect1d(pos_expand_shift-1, 
                                     np.setdiff1d(np.arange(L), new_pos_discard),
                                     return_indices=True)
        extend3 = np.zeros(pos_expand_shift.size)
        extend3[indices] = 1
        extend = (extend1*extend2*extend3).astype(bool)
        pos_expand_shift[extend1] -= 1
        adj_expansion_lens = np.copy(expansion_lens)
        adj_expansion_lens[extend] += 1
        if new_pos_expand.size == 0:
            new_pos_expand = pos_expand_shift
            new_expansion_lens = adj_expansion_lens
        else:
            if pos_expand_shift.size > 1 and pos_expand_shift[0] == 0 and pos_expand_shift[1] == 0:
                adj_expansion_lens[0] += adj_expansion_lens[1] 
            for i in new_pos_expand:
                a = np.argwhere(pos_expand_shift == i)
                if a.size > 0:
                    adj_expansion_lens[a[0]] += 1
            new_pos_expand = np.concatenate([pos_expand_shift, new_pos_expand])
            new_expansion_lens = np.concatenate([adj_expansion_lens, new_expansion_lens])
            new_pos_expand, indices = np.unique(new_pos_expand, return_index=True)
            new_expansion_lens = new_expansion_lens[indices]
        if new_pos_discard.size > 0:
            new_pos_discard = np.concatenate([new_pos_discard, 
                                              pos_expand_shift[extend]])
            new_pos_discard = np.unique(new_pos_discard)
        else:
            new_pos_discard = pos_expand_shift[extend]
    return new_pos_expand, new_expansion_lens, new_pos_discard


#applies expansions and discards to emission and transition kernels
def update_kernels(alignment,
                    pos_expand, 
                    expansion_lens, 
                    pos_discard, 
                    emission_dummy, 
                    transition_dummy,
                    init_flank_dummy):
    L = alignment.msa_hmm_layer.length
    emissions = alignment.msa_hmm_layer.C.emission_kernel.numpy()
    transitions = { key : kernel.numpy() 
                         for key, kernel in alignment.msa_hmm_layer.C.transition_kernel.items()}
    emissions_new = apply_mods(emissions, 
                                  pos_expand, 
                                  expansion_lens, 
                                  pos_discard, 
                                  emission_dummy[:1])
    transitions_new = {}
    args1 = extend_mods(pos_expand,expansion_lens,pos_discard,L)
    transitions_new["match_to_match"] = apply_mods(transitions["match_to_match"], 
                                                      *args1,
                                                      transition_dummy["match_to_match"][0])
    transitions_new["match_to_insert"] = apply_mods(transitions["match_to_insert"], 
                                                      *args1,
                                                      transition_dummy["match_to_insert"][0])
    transitions_new["insert_to_match"] = apply_mods(transitions["insert_to_match"], 
                                                      *args1,
                                                      transition_dummy["insert_to_match"][0])
    transitions_new["insert_to_insert"] = apply_mods(transitions["insert_to_insert"], 
                                                      *args1,
                                                      transition_dummy["insert_to_insert"][0])
    args2 = extend_mods(pos_expand,expansion_lens,pos_discard,L+1,k=1)
    transitions_new["match_to_delete"] = apply_mods(transitions["match_to_delete"],
                                                     *args2,
                                                      transition_dummy["match_to_delete"][0])
    args3 = extend_mods(pos_expand,expansion_lens,pos_discard,L+1)
    transitions_new["delete_to_match"] = apply_mods(transitions["delete_to_match"],
                                                     *args3,
                                                      transition_dummy["delete_to_match"][0])
    transitions_new["delete_to_delete"] = apply_mods(transitions["delete_to_delete"],
                                                     *args1,
                                                      transition_dummy["delete_to_delete"][0])
    
    #always reset the multi-hit transitions:
    transitions_new["left_flank_loop"] = transition_dummy["left_flank_loop"] 
    transitions_new["left_flank_exit"] = transition_dummy["left_flank_exit"] 
    init_flank_new = init_flank_dummy
    transitions_new["right_flank_loop"] = transition_dummy["right_flank_loop"] 
    transitions_new["right_flank_exit"] = transition_dummy["right_flank_exit"] 
    transitions_new["end_to_unannotated_segment"] = transition_dummy["end_to_unannotated_segment"] 
    transitions_new["end_to_right_flank"] = transition_dummy["end_to_right_flank"] 
    transitions_new["end_to_terminal"] = transition_dummy["end_to_terminal"] 
    transitions_new["unannotated_segment_loop"] = transition_dummy["unannotated_segment_loop"] 
    transitions_new["unannotated_segment_exit"] = transition_dummy["unannotated_segment_exit"] 
    
    # Problem: Discarding or extending positions has the side effect of changing all probabilities
    # in begin-state transition distribution. E.g. 
    # Depending on discarded positions, adjust weights such that the residual distribution after 
    # discarding some match states is unaffected.
    # If an insert position is expanded, the transitions from begin to the new match states should have 
    # probabilities according to the initial dummy distribution and the weights of the old transitions 
    # should also be corrected accordingly.
    #probs = alignment.msa_hmm_layer.C.make_probs()
    #dummy_begin = np.exp(np.concatenate([transition_dummy["match_to_delete"][:1], transition_dummy["begin_to_match"]]))
    #dummy_begin /= np.sum(dummy_begin)
    #if 0 in pos_expand:
    #    cond_expand = np.log(1 - dummy_begin[1] - (np.sum(expansion_lens)-1)*dummy_begin[2])
    #else:
    #    cond_expand = np.log(1 - np.sum(expansion_lens)*dummy_begin[2])
    #cond_discard = np.log(1 - np.sum(transitions["begin_to_match"][pos_discard]))
    transitions_new["begin_to_match"] = apply_mods(transitions["begin_to_match"], 
                                                      pos_expand, 
                                                      expansion_lens, 
                                                      pos_discard, 
                                                      transition_dummy["begin_to_match"][1])
    if 0 in pos_expand:
        transitions_new["begin_to_match"][0] = transition_dummy["begin_to_match"][0]
        
    if L in pos_expand:
        transitions["match_to_end"][-1] = transition_dummy["match_to_end"][0]
    transitions_new["match_to_end"] = apply_mods(transitions["match_to_end"], 
                                                  pos_expand, 
                                                  expansion_lens, 
                                                  pos_discard, 
                                                  transition_dummy["match_to_end"][0])
    return transitions_new, emissions_new, init_flank_new


#simple adpative batch size depending on sequence length
#longer models and sequences require much more memory
def get_adaptive_batch_size(model_length, max_len):
    if max_len * model_length < 320 * 350:
        return 512
    if max_len * model_length < 700 * 650:
        return 256
    else:
        return 128

    

    
SEQ_COUNT_WARNING_THRESHOLD = 100


# Constructs an alignment for the given fasta file
def fit_and_align(fasta_file, 
                  config,
                  subset=None,
                  verbose=True):
    
    if fasta_file.gaps:
        print(f"Warning: The file {fasta_file.filename} already contains gaps. Realining the raw sequences.")
    
    if fasta_file.num_seq < SEQ_COUNT_WARNING_THRESHOLD:
        print(f"Warning: You are aligning {fasta_file.num_seq} sequences, although learnMSA is designed for large scale alignments. We recommend to have a sufficiently deep training dataset of at least {SEQ_COUNT_WARNING_THRESHOLD} sequences for accurate results.")
        
    total_time_start = time.time()
    n = fasta_file.num_seq
    if subset is None:
        subset = np.arange(n)
    #config valus that may be reset during model surgery
    emission_init = config["emission_init"]
    transition_init = config["transition_init"]
    flank_init = config["flank_init"]
    tau_init = config["tau_init"]
    #ignore short sequences for all surgery iterations except the last
    k = int(min(n*config["surgery_quantile"], 
                max(0, n-config["min_surgery_seqs"])))
    #a rough estimate of a set of only full-length sequences
    full_length_estimate = [i for l,i in sorted(zip(fasta_file.seq_lens, list(range(n))))[k:]]   
    full_length_estimate = np.array(full_length_estimate)  
    #initial model length
    model_length = np.quantile(fasta_file.seq_lens, q=config["length_init_quantile"])
    model_length *= config["len_mul"]
    model_length = max(3, int(np.floor(model_length)))
    #model surgery
    finished=config["max_surgery_runs"]==1
    for i in range(config["max_surgery_runs"]):
        if config["batch_size"] == "adaptive":
            _batch_size = get_adaptive_batch_size(model_length, fasta_file.max_len)
        else:
            _batch_size = config["batch_size"]
        _batch_size = min(_batch_size, n)
        if finished:    
            train_indices = np.arange(n)
        else:
            train_indices = full_length_estimate
        model, history = train.fit_model(fasta_file=fasta_file,
                                         indices=train_indices,
                                         model_length=model_length, 
                                         emission_init=emission_init,
                                         transition_init=transition_init,
                                         flank_init=flank_init,
                                         alpha_flank=config["alpha_flank"], 
                                         alpha_single=config["alpha_single"], 
                                         alpha_frag=config["alpha_frag"],
                                         use_prior=config["use_prior"],
                                         dirichlet_mix_comp_count=config["dirichlet_mix_comp_count"],
                                         use_anc_probs=config["use_anc_probs"],
                                         tau_init=tau_init,
                                         trainable_kernels={},
                                         batch_size=_batch_size, 
                                         learning_rate=0.1,
                                         epochs=1+1*(i==0)+2*finished,
                                         verbose=verbose)
        if not finished:
            alignment = Alignment(fasta_file,
                                  train_indices,
                                  batch_size=2*_batch_size, 
                                  model=model,
                                  use_anc_probs=config["use_anc_probs"])
        else:
            alignment = Alignment(fasta_file,
                                   subset,
                                   batch_size=2*_batch_size, 
                                   model=model,
                                   use_anc_probs=config["use_anc_probs"],
                                   build="lazy")
        if finished:
            break
        if i == 0:
            emission_init_0 = alignment.msa_hmm_layer.emission_init
            transition_init_0 = alignment.msa_hmm_layer.transition_init
            flank_init_0 = alignment.msa_hmm_layer.flank_init
        pos_expand, expansion_lens, pos_discard = get_discard_or_expand_positions(alignment, 
                                                                                        del_t=0.5, 
                                                                                        ins_t=0.5,
                                                                                        ins_long=100000, 
                                                                                        k=32, 
                                                                                        match_prior_threshold=1)
        finished = (pos_expand.size == 0 and pos_discard.size == 0) or (i == config["max_surgery_runs"]-2)
        if verbose:
            print("expansions:", list(zip(pos_expand, expansion_lens)))
            print("discards:", pos_discard)
        transition_init, emission_init, flank_init = update_kernels(alignment, 
                                                              pos_expand, expansion_lens, pos_discard,
                                                              emission_init_0, 
                                                              transition_init_0, 
                                                              flank_init_0)
        model_length = emission_init.shape[0]
        if model_length < 3: 
            raise SystemExit("A problem occured during model surgery: The model is too short (length <= 2). This might indicate that there is a problem with your sequences.") 
        if config["use_anc_probs"] and config["keep_tau"]:
            if finished:
                tau_init = np.zeros(n)
                tau_init[full_length_estimate] = alignment.anc_probs_layer.tau.numpy()
            else:
                tau_init = alignment.anc_probs_layer.tau.numpy()
    alignment.total_time = time.time() - total_time_start
    return alignment



def fit_and_align_n(fasta_file, 
                    num_runs, 
                    config,
                    subset=None,
                    verbose=True):
    if verbose:
        print("Training of", num_runs, "independent models on file", os.path.basename(fasta_file.filename))
        print("Configuration:")
        print(as_str(config))
    results = []
    for i in range(num_runs):
        t_s = time.time()
        alignment = fit_and_align(fasta_file, config, subset, verbose)
        t_a = time.time()
         #compute loglik
        #estimate the ll only on a subset of maximum size 200.000
        #otherwise for millions of sequences this step takes rather long for
        #very little benefit
        max_ll_estimate = 200000
        ll_subset = np.arange(fasta_file.num_seq)
        np.random.shuffle(ll_subset)
        ll_subset = ll_subset[:max_ll_estimate]
        ds = train.make_dataset(fasta_file, 
                                        #larger batch size than during training for additional speedup
                                        2*alignment.batch_size, 
                                        shuffle=False,
                                        indices=ll_subset)
        loglik = 0
        for x, _ in ds:
            loglik += np.sum(alignment.model(x))
        loglik /= ll_subset.size
        prior = alignment.msa_hmm_layer.get_prior_log_density().numpy()[0]/fasta_file.num_seq
        if verbose:
            print("Time for alignment:", "%.4f" % (t_a-t_s))
            print("Time for estimating loglik:", "%.4f" % (time.time()-t_a))
            print("Fitted a model has MAP estimate =", "%.4f" % (loglik + prior))
        results.append((loglik + prior, alignment))
    return results
