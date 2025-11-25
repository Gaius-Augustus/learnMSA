from json import encoder
import tensorflow as tf
import numpy as np
import learnMSA.msa_hmm.AncProbsLayer as anc_probs
import learnMSA.msa_hmm.MsaHmmLayer as msa_hmm_layer
import learnMSA.msa_hmm.MsaHmmCell as msa_hmm_cell
from learnMSA.msa_hmm.SequenceDataset import SequenceDataset, AlignedDataset
import learnMSA.msa_hmm.Configuration as config
import learnMSA.msa_hmm.Training as train
import learnMSA.msa_hmm.Viterbi as viterbi
import learnMSA.msa_hmm.Priors as priors
import learnMSA.msa_hmm.Transitioner as trans
import learnMSA.msa_hmm.Emitter as emit
import json
import shutil
from packaging import version
from pathlib import Path

        
# utility class used in AlignmentModel storing useful information on a 
# specific alignment
class AlignmentMetaData():
    def __init__(
        self, 
        core_blocks, 
        left_flank, 
        right_flank, 
        unannotated_segments
    ):
        self.consensus = np.stack([C for C,_,_,_ in core_blocks])
        self.insertion_lens = np.stack([IL for _,IL,_,_ in core_blocks])
        self.insertion_start = np.stack([IS for _,_,IS,_ in core_blocks])
        self.finished = np.stack([f for _,_,_,f in core_blocks])
        self.left_flank_len = np.stack(left_flank[0])
        self.left_flank_start = np.stack(left_flank[1])
        self.right_flank_len = np.stack(right_flank[0])
        self.right_flank_start = np.stack(right_flank[1])
        if len(unannotated_segments) > 0:
            self.unannotated_segments_len = np.stack([
                l for l,_ in unannotated_segments
            ])
            self.unannotated_segments_start = np.stack([
                s for _,s in unannotated_segments
            ])
            self.unannotated_segment_lens_total = np.amax(
                self.unannotated_segments_len, axis=1
            )
        else:
            self.unannotated_segment_lens_total = 0
        self.num_repeats = self.consensus.shape[0]
        self.consensus_len = self.consensus.shape[-1]
        self.left_flank_len_total = np.amax(self.left_flank_len)
        self.right_flank_len_total = np.amax(self.right_flank_len)
        self.insertion_lens_total = np.amax(self.insertion_lens, axis=1)
        # convert at least 1 term to int32 in case of an alignment longer 
        # than 32,767
        self.alignment_len = (
            self.left_flank_len_total.astype(np.int32) + 
            self.consensus_len*self.num_repeats + 
            np.sum(self.insertion_lens_total) + 
            np.sum(self.unannotated_segment_lens_total) +
            self.right_flank_len_total
        )
        
        
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
        



class AlignmentModel():
    """ Decodes alignments from a number of models, stores them in a memory 
        friendly representation and generates table-form (memory unfriendly) 
        alignments on demand (batch-wise mode possible).
    Args:
        data: The dataset of sequences.
        batch_generator: An already configured batch generator.
        indices: (A subset of) The sequence indices from the dataset to align 
            (1D).
        batch_size: Controls memory consumption of viterbi.
        model: A learnMSA model which internally might represent multiple 
            pHMM models.
        gap_symbol: Character used to denote missing match positions.
        gap_symbol_insertions: Character used to denote insertions in other 
            sequences.
        A2M: DEPRECATED. Use format="a2m" or format="fasta" in to_file() method
    """
    def __init__(self, 
                 data : SequenceDataset, 
                 batch_generator,
                 indices, 
                 batch_size, 
                 model,
                 gap_symbol="-",
                 gap_symbol_insertions=".",
                 A2M=None):
        self.data = data
        self.batch_generator = batch_generator
        self.indices = indices
        self.batch_size = batch_size
        self.model = model
        # encoder model is the same as model but with the MsaHmmLayer removed
        # the output of the encoder model will be the input to viterbi
        # in the default learnMSA, the encoder model is only the Ancestral 
        # Probability layer.
        self.encoder_model = None
        for i, layer in enumerate(model.layers[1:]):
            if layer.name.startswith("anc_probs_layer"):
                self.anc_probs_layer = layer
            if layer.name.startswith("msa_hmm_layer"):
                encoder_out = model.layers[i].output
                self.msa_hmm_layer = layer
                self.encoder_model = tf.keras.Model(
                    inputs=self.model.inputs, outputs=[encoder_out]
                )
        assert self.encoder_model is not None, \
            "Can not find a MsaHmmLayer in the specified model."
        self.gap_symbol = gap_symbol
        self.gap_symbol_insertions = gap_symbol_insertions

        self.metadata = {}
        self.num_models = self.msa_hmm_layer.cell.num_models
        self.length = self.msa_hmm_layer.cell.length
        if A2M is not None:
            raise DeprecationWarning(
                "The A2M argument is deprecated. Use format='a2m' or " \
                "format='fasta' in to_file() method."
            )

    #computes an implicit alignment (without storing gaps)
    #eventually, an alignment with explicit gaps can be written
    #in a memory friendly manner to file
    def _build_alignment(self, models):

        assert len(models) == 1, "Not implemented for multiple models."

        if tf.distribute.has_strategy():
            with tf.distribute.get_strategy().scope():
                cell_copy = self.msa_hmm_layer.cell.duplicate(models)
        else:
            cell_copy = self.msa_hmm_layer.cell.duplicate(models)

        cell_copy.build(
            (self.num_models, None, None, self.msa_hmm_layer.cell.dim)
        )
        state_seqs_max_lik = viterbi.get_state_seqs_max_lik(
            self.data,
            self.batch_generator,
            self.indices,
            self.batch_size,
            cell_copy,
            models,
            self.encoder_model
        )
        state_seqs_max_lik = self._clean_up_viterbi_seqs(
            state_seqs_max_lik, models, cell_copy
        )
        for i,l,max_lik_seqs in zip(models, cell_copy.length, state_seqs_max_lik):
            decoded_data = AlignmentModel.decode(l,max_lik_seqs)
            self.metadata[i] = AlignmentMetaData(*decoded_data)

    def _clean_up_viterbi_seqs(self, state_seqs_max_lik, models, cell_copy):
        # state_seqs_max_lik has shape (num_model, num_seq, L)
        faulty_sequences = find_faulty_sequences(
            state_seqs_max_lik, 
            cell_copy.length[0], 
            self.data.seq_lens[self.indices]
        )
        self.fixed_viterbi_seqs = faulty_sequences
        if faulty_sequences.size > 0:
            # repeat Viterbi with a masking that prevents certain transitions
            # that can cause problems
            fixed_state_seqs = viterbi.get_state_seqs_max_lik(
                self.data,
                self.batch_generator,
                faulty_sequences,
                self.batch_size,
                cell_copy,
                models,
                self.encoder_model,
                non_homogeneous_mask_func
            )
            if state_seqs_max_lik.shape[-1] < fixed_state_seqs.shape[-1]:
                state_seqs_max_like = np.pad(
                    state_seqs_max_lik, 
                    ((0,0),(0,0),(0,fixed_state_seqs.shape[-1]-state_seqs_max_lik.shape[-1])), 
                    constant_values=2*cell_copy.length[0]+2
                )
            state_seqs_max_lik[0,faulty_sequences,:fixed_state_seqs.shape[-1]] = fixed_state_seqs[0]
        return state_seqs_max_lik

    def get_output_alphabet(self, a2m : bool = True):
        """ Returns the output alphabet used for string representation of
            alignments.

        Args:
            a2m (bool): Whether to use the a2m format for strings
                (with lowercase letters for inserted amino acids and dots for
                gaps in insertions).
        """
        if a2m:
            output_alphabet = np.array((
                list(self.data.get_alphabet_no_gap()) +
                [self.gap_symbol] +
                list(self.data.get_alphabet_no_gap().lower()) +
                [self.gap_symbol_insertions, "$"]
            ))
        else:
            output_alphabet = np.array((
                list(self.data.get_alphabet_no_gap()) +
                [self.gap_symbol] +
                list(self.data.get_alphabet_no_gap()) +
                [self.gap_symbol, "$"]
            ))
        return output_alphabet


    def to_string(
        self,
        model_index,
        batch_size=100000,
        add_block_sep=True,
        aligned_insertions : AlignedInsertions = AlignedInsertions(),
        a2m=True,
    ):
        """ Uses one model to decode an alignment and returns the sequences
            with gaps in a list.
            Note that this method is not suitable im memory is limited and
            alignment depths and width are large.

        Args:
            model_index: Specifies the model for decoding. Use a suitable
                criterion like loglik to decide for a model.
            batch_size: Defines how many sequences are decoded at a time with
                no effect on the output MSA. It can be useful to lower this if
                memory is sufficient to store the table-form alignment but GPU
                memory used for decoding a batch is limited.
            add_block_sep: If true, columns containing a special character are
                added to the alignment indicating domain boundaries.
            aligned_insertions: Can be used to override insertion metadata if
                insertions are aligned after the main procedure.
            a2m: Whether to use the a2m format for strings
                (with lowercase letters for inserted amino acids and dots for
                gaps in insertions).
        """
        output_alphabet = self.get_output_alphabet(a2m)
        alignment_strings_all = []
        n = self.indices.size
        i = 0
        while i < n:
            batch_indices = np.arange(i, min(n, i+batch_size))
            batch_alignment = self.get_batch_alignment(
                model_index, batch_indices, add_block_sep, aligned_insertions
            )
            alignment_strings = self.batch_to_string(
                batch_alignment, output_alphabet=output_alphabet
            )
            alignment_strings_all.extend(alignment_strings)
            i += batch_size
        return alignment_strings_all
    
    def to_file(
        self, 
        filepath, 
        model_index, 
        batch_size=100000, 
        add_block_sep=False, 
        aligned_insertions : AlignedInsertions = AlignedInsertions(), 
        format="fasta", 
        fasta_line_limit=80,
        only_matches=False,
    ):
        """ Uses one model to decode an alignment and stores it in fasta file 
            format. Currently no other output format is supported.
            The file is written batch wise. The memory required for this 
            operation must be large enough to hold decode and store a single 
            batch of aligned sequences but not the whole alignment.
        Args:
            model_index: Specifies the model for decoding. Use a suitable 
                criterion like loglik to decide for a model.
            batch_size: Defines how many sequences are decoded at a time with 
                no effect on the output MSA. It can be useful to lower this if 
                memory is sufficient to store the table-form alignment but GPU 
                memory used for decoding a batch is limited.
            add_block_sep: If true, columns containing a special character are 
                added to the alignment indicating domain boundaries.
            aligned_insertions: Can be used to override insertion metadata if 
                insertions are aligned after the main procedure.
            format: Output format. Important for large data: learnMSA is only 
                able to stream fasta files. 
                Other formats require a conversion, i.e. the whole alignment is 
                stored in memory.
            fasta_line_limit: Maximum number of characters per line in the 
                fasta file (only applies to sequences).
            only_matches: If true, omit all insertions and write only those
                amino acids that are assigned to match states.
        """
        if format == "fasta" or format == "a2m": #streaming batches to file
            output_alphabet = self.get_output_alphabet(format == "a2m")
            with open(filepath, "w") as output_file:
                n = self.indices.size
                i = 0
                while i < n:
                    batch_indices = np.arange(i, min(n, i+batch_size))
                    batch_alignment = self.get_batch_alignment(
                        model_index,
                        batch_indices,
                        add_block_sep,
                        aligned_insertions,
                        only_matches,
                    )
                    alignment_strings = self.batch_to_string(
                        batch_alignment, output_alphabet=output_alphabet
                    )
                    for s, seq_ind in zip(alignment_strings, batch_indices):
                        seq_header = self.data.get_header(
                            self.indices[seq_ind]
                        )
                        output_file.write(">"+seq_header+"\n")
                        for j in range(0, len(s), fasta_line_limit):
                            output_file.write(s[j:j+fasta_line_limit]+"\n")
                    i += batch_size
        else:
            msa = self.to_string(
                model_index, batch_size, add_block_sep, aligned_insertions
            )
            msa = [
                (self.data.seq_ids[self.indices[i]], msa[i]) 
                for i in range(len(msa))
            ]
            data = AlignedDataset(aligned_sequences=msa)
            data.write(filepath, format)
    
    def get_batch_alignment(
        self, 
        model_index, 
        batch_indices, 
        add_block_sep, 
        aligned_insertions : AlignedInsertions = AlignedInsertions(),
        only_matches=False,
    ):
        """ Returns a dense matrix representing a subset of sequences
            as specified by batch_indices with respect to the alignment of all 
            sequences (i.e. the sub alignment can contain gap-only columns and 
            stacking all batches yields a complete alignment).
        Args:
            model_index: Specifies the model for decoding. Use a suitable 
                criterion like loglik to decide for a model.
            batch_indices: Sequence indices / indices of alignment rows.
            add_block_sep: If true, columns containing a special character are 
                added to the alignment indicating domain boundaries.
            aligned_insertions: Can be used to override insertion metadata if 
                insertions are aligned after the main procedure.
            only_matches: If true, omit all insertions and write only those
                amino acids that are assigned to match states.
        """
        if not model_index in self.metadata:
            self._build_alignment([model_index])
        data = self.metadata[model_index]
        b = batch_indices.size
        sequences = np.zeros((b, self.data.max_len), dtype=np.uint16) 
        sequences += (len(self.data.alphabet)-1)
        for i,j in enumerate(batch_indices):
            l = self.data.seq_lens[self.indices[j]]
            sequences[i, :l] = self.data.get_encoded_seq(self.indices[j])
        blocks = []  
        if add_block_sep:
            sep = np.zeros((b,1), dtype=np.uint16) + 2*len(self.data.alphabet)
        if not only_matches:
            left_flank_block = self.get_insertion_block(
                sequences,
                data.left_flank_len[batch_indices],
                max(data.left_flank_len_total, aligned_insertions.ext_left_flank),
                data.left_flank_start[batch_indices],
                adjust_to_right=True,
                custom_columns=aligned_insertions.left_flank(batch_indices)
            )
            blocks.append(left_flank_block)
            if add_block_sep:
                blocks.append(sep)
        for i in range(data.num_repeats):
            consensus = data.consensus[i]
            #remove columns consisting only of gaps
            is_non_empty = np.any(consensus != -1, axis=0)
            ins_len = data.insertion_lens[i]
            ins_start = data.insertion_start[i]
            alignment_block = self.get_alignment_block(
                sequences, 
                consensus[batch_indices], 
                ins_len[batch_indices], 
                np.maximum(
                    data.insertion_lens_total, 
                    aligned_insertions.ext_insertions
                )[i],
                ins_start[batch_indices],
                is_non_empty=is_non_empty,
                custom_columns=aligned_insertions.insertion(batch_indices, i),
                only_matches=only_matches
            )
            blocks.append(alignment_block)
            if add_block_sep:
                blocks.append(sep)
            if i < data.num_repeats-1 and not only_matches:
                unannotated_segment_l = data.unannotated_segments_len[i]
                unannotated_segment_s = data.unannotated_segments_start[i]
                unannotated_block = self.get_insertion_block(
                    sequences, 
                    unannotated_segment_l[batch_indices],
                    np.maximum(
                        data.unannotated_segment_lens_total, 
                        aligned_insertions.ext_unannotated
                    )[i],
                    unannotated_segment_s[batch_indices],
                    custom_columns=aligned_insertions.unannotated_segment(
                        batch_indices, i
                    )
                )
                blocks.append(unannotated_block)
                if add_block_sep:
                    blocks.append(sep)
        if not only_matches:
            right_flank_block = self.get_insertion_block(
                sequences, 
                data.right_flank_len[batch_indices],
                max(data.right_flank_len_total, aligned_insertions.ext_right_flank),
                data.right_flank_start[batch_indices],
                custom_columns=aligned_insertions.right_flank(batch_indices)
            )
            blocks.append(right_flank_block)
        batch_alignment = np.concatenate(blocks, axis=1)
        return batch_alignment

    def batch_to_string(self, batch_alignment, output_alphabet):
        """ Converts a dense matrix into string format.
        """
        alignment_arr = output_alphabet[batch_alignment]
        alignment_strings = [''.join(s) for s in alignment_arr]
        return alignment_strings

    def compute_loglik(
        self,
        max_seq: int =200000,
        reduce: bool = True,
        no_anc_probs: bool = False,
    ):
        """ Computes the logarithmic likelihood for each underlying model.
        Args:
            max_seq: Threshold for the number of sequences used to compute the
                loglik. If the dataset has more sequences, a random subset is
                drawn.
            reduce: If true, the loglik will be averaged over the number of
                sequences.
            no_anc_probs: If true, the ancestral probabilities will not be
                computed.
        """
        n = self.data.num_seq
        if n > max_seq and reduce:
            #estimate the ll only on a subset, otherwise for millions of 
            # sequences this step takes rather long for little benefit
            ll_subset = np.arange(n)
            np.random.shuffle(ll_subset)
            ll_subset = ll_subset[:max_seq]
            ll_subset = np.sort(ll_subset)
            bucket_by_seq_length = False
        else:
            ll_subset = np.arange(n)
            # Batch sequences sorted by length for efficiency
            bucket_by_seq_length = True
        ds = train.make_dataset(
            ll_subset,
            self.batch_generator,
            self.batch_size,
            shuffle=False,
            bucket_by_seq_length=bucket_by_seq_length,
            model_lengths=self.msa_hmm_layer.cell.length
        )
        if no_anc_probs:
            def _fn(x):
                # split x into sequences and idx (idx are not used here)
                x, _ = x
                x = tf.one_hot(x, depth=self.msa_hmm_layer.cell.dim)
                # learnMSA currently expects model dim first
                x = tf.keras.ops.swapaxes(x, 0, 1)
                y = self.msa_hmm_layer(x)[0]
                y = tf.keras.ops.swapaxes(y, 0, 1)
                return y
            model_fn = tf.function(
                _fn,
                jit_compile=False, # jit compiling is currently slower here
                input_signature=[[
                    tf.TensorSpec((None, None, None), dtype=tf.uint8),
                    tf.TensorSpec((None, None), dtype=tf.int64),
                ]]
            )
        else:
            model_fn = lambda x: self.model(x)[0]
        if reduce:
            loglik = np.zeros((self.msa_hmm_layer.cell.num_models))
            for (*x,_), _ in ds:
                loglik += np.sum(model_fn(x), axis=0)
            loglik /= ll_subset.size
        else:
            loglik = np.zeros((n, self.msa_hmm_layer.cell.num_models))
            for (*x, batch_indices), _ in ds:
                batch_loglik = model_fn(x).numpy()
                loglik[batch_indices,:] = batch_loglik
        return loglik

    def compute_null_model_log_probs(self):
        """ Computes the logarithmic likelihood of each sequence under the
            null model

             S ---> T
            |_^    |_^
             p

            where the emission probabilities of S are amino acid background
            frequencies and T is the terminal state.

            Args:
                p: The probability of staying in the emitting state S.
        """
        # Prepare the data
        n = self.data.num_seq
        ds = train.make_dataset(
            np.arange(n),
            self.batch_generator,
            self.batch_size,
            shuffle=False,
            bucket_by_seq_length=True,
            model_lengths=self.msa_hmm_layer.cell.length
        )

        # Prepare the background frequencies
        aa_dist = self.msa_hmm_layer.cell.emitter[0].prior.emission_dirichlet_mix.make_background().numpy()
        # Append ad-hoc probability for non-standard amino acids
        aa_dist = np.append(aa_dist, [1e-4]*3)
        aa_dist /= np.sum(aa_dist) #normalize
        # Add the terminal symbol emissions
        aa_dist = np.append(aa_dist, [1.0])
        # Log transition probabilities
        aa_dist = np.log(aa_dist + 1e-10)  # shape (24,)

        # Compute emission log probs
        log_probs = np.zeros((n,))
        for (x,_,batch_idx), _ in ds:
            em = tf.reduce_sum(tf.gather(aa_dist, tf.cast(x[:,0,:], tf.int32)), axis=1)
            log_probs[batch_idx] = em.numpy()

        # Add transition log probs based on the target sequence lenghts
        # We'll assume a geometric distribution with the expected length
        # equal to the length of each target sequence
        L = self.data.seq_lens
        M = np.mean(L)
        trans_scores = L * (np.log(M) - np.log(M+1))
        log_probs += trans_scores

        return log_probs

    def compute_log_prior(self):
        """ Computes the logarithmic prior value of each underlying model.
        """
        n = self.data.num_seq
        if n > 0:
            return self.msa_hmm_layer.cell.get_prior_log_density().numpy()/n
        else:
            return 0

    def compute_AIC(self, max_seq=200000, loglik=None):
        """ Computes the Akaike information criterion for each underlying model. 
        Args:
            max_seq: Threshold for the number of sequences used to compute the 
                loglik. If the dataset has mroe sequences, a random subset is 
                drawn.
            loglik: This argument can be set if the loglik was computed before 
                via compute_loglik to avoid overhead. If None, the loglik will 
                be computed internally.
        """
        if loglik is None:
            loglik = self.compute_loglik(max_seq)
        num_param = 34 * np.array(self.length) + 25
        aic = -2 * loglik * self.data.num_seq + 2*num_param
        return aic 

    def compute_consensus_score(self):
        """ Computes a consensus score that rates how plausible each model is 
            with respect to all other models.
            (Relevant for users not using the default emitter: Uses the 
            make_B_amino method of the first emitter.)
        """
        # compute the match sequence of all models padded with terminal symbols 
        match_seqs = np.zeros(
            (self.num_models, max(self.length)+1, self.msa_hmm_layer.cell.dim)
        )
        match_seqs[:,:,-1] = 1 #initialize with terminal symbols
        emitter = self.msa_hmm_layer.cell.emitter[0]
        for i,L in enumerate(self.length):
            match_seqs[i, :L] = emitter.make_B_amino()[i,1:L+1]
        # we need to tile the match sequences over the batch dimension because 
        # each model should see all other models
        match_seqs = tf.stack([match_seqs]*self.num_models, axis=1)
        # rate the match seqs with respect to the models and cancel out 
        # self-rating
        
        # TODO this does not work with self.model which expects indices rather 
        # than distributions
        # this is a workaround, but will break with a user defined encoder model
        # we skip the anc probs for simplicity as this would require fitting 
        # evolutionary times
        consensus_logliks = self.msa_hmm_layer(match_seqs)[1]
        
        consensus_logliks *= 1-tf.eye(self.num_models)
        # Axis 1 means we reduce over the batch dimension rather than the model 
        # dimension,
        # so output i will be the mean loglik if we input all other match 
        # sequenes to model i.
        # Using axis 0 here is not the same!
        # Consider the case that all models have the same match sequence but 
        # one model allows many deletions or insertions.
        # This model is the outlier and should clearly have the lowest score.
        # With axis=0, the likelihood of the outlier model under all other 
        # models is high and the scores of the other models will have a penalty
        # since their match sequences are fed into the outlier model.
        # With axis=1, the scores of all models will be high except for the 
        # outlier model, which has a strong penalty as it rates all other 
        # match sequences involving the deletion/insertion probabilities.
        consensus_score = tf.reduce_mean(consensus_logliks, axis=1)
        return consensus_score


    def write_models_to_file(self, filepath, pack=True):
        """ Writes the underlying models to file.
        Args: 
            filepath: Path of the written file.
            pack: If true, the output will be a zip file, otherwise a directory.
        """
        Path(filepath).mkdir(parents=True, exist_ok=True)
        #serialize metadata
        d = {
            "num_models" : self.num_models,
            "batch_size" : self.batch_size,
            "gap_symbol" : self.gap_symbol,
            "gap_symbol_insertions" : self.gap_symbol_insertions,
        }
        if hasattr(self, "best_model"):
            d["best_model"] = int(self.best_model)
        with open(filepath+"/meta.json", "w") as metafile:
            metafile.write(json.dumps(d, indent=4))
        #serialize indices
        np.savetxt(filepath+"/indices", self.indices, fmt='%i')
        #save the model
        if version.parse(tf.__version__) < version.parse("2.12.0"):
            self.model.save(filepath+".keras", save_traces=False)
        else:
            self.model.save(filepath+".keras")
        if pack:
            shutil.make_archive(filepath, "zip", filepath)
            try:
                shutil.rmtree(filepath)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))


    def write_scores(self, filepath: str, model: int|None = None) -> None:
        """ Writes per-sequence scores (loglik, bitscore) to a
            tsv file sorted by bitscore. The bitscore is computed as+
            ``loglik(S) - log P(S; nullmodel)``.
        Args:
            filepath: Path of the output file.
            model: The model for which scores are written. By default, the best
                model based on the model selection standard criterion is used.
        """
        # Find the model index to use
        if model is None:
            model = self.best_model if hasattr(self, "best_model") else 0 # type: ignore

        # Compute the likelihood and bitscores for all sequences
        loglik = self.compute_loglik(
            self.data.num_seq, reduce=False, no_anc_probs=True
        )[:,model]
        # Compute the bitscore
        log_null = self.compute_null_model_log_probs()
        bitscore = loglik - log_null

        # Sort by bitscore in descending order
        sorted_indices = np.argsort(-bitscore)

        # Write to file
        with open(filepath, "w") as scorefile:
            scorefile.write(
                "\t".join(["seq_id", "loglik", "bit_score"]) + "\n"
            )
            for idx in sorted_indices:
                scorefile.write("\t".join([
                    f"{self.data.seq_ids[idx]}",
                    f"{loglik[idx]}",
                    f"{bitscore[idx]}"
                ]) + "\n")

    @classmethod
    def load_models_from_file(cls, 
                              filepath, 
                              data:SequenceDataset, 
                              from_packed=True, 
                              custom_batch_gen=None,
                              custom_config=None):
        """ Recreates an AlignmentModel instance with underlying models from 
            file.

        Args:
            filepath: Path of the file to load.
            from_packed: Pass true or false depending on the pack argument used 
                with write_models_to_file.
            custom_batch_gen: A custom batch generator to use instead of the 
                default one.
            custom_config: A custom configuration to use instead of the 
                default one.

        Returns:
            An AlignmentModel instance with equivalent behavior as the 
            AlignmentModel instance used while saving the model.
        """
        if from_packed:
            shutil.unpack_archive(filepath+".zip", filepath)
        #deserialize metadata    
        with open(filepath+"/meta.json") as metafile:
            d = json.load(metafile)
        #deserialize indices
        indices = np.loadtxt(filepath+"/indices", dtype=int)
        #load the model
        model = tf.keras.models.load_model(
            filepath+".keras", 
            custom_objects={
                "AncProbsLayer": anc_probs.AncProbsLayer, 
                "MsaHmmLayer": msa_hmm_layer.MsaHmmLayer,
                "MsaHmmCell": msa_hmm_cell.MsaHmmCell, 
                "ProfileHMMTransitioner": trans.ProfileHMMTransitioner, 
                "ProfileHMMEmitter": emit.ProfileHMMEmitter,
                "AminoAcidPrior": priors.AminoAcidPrior,
                "NullPrior": priors.NullPrior,
                "ProfileHMMTransitionPrior": priors.ProfileHMMTransitionPrior
            }
        )
        if from_packed:
            #after loading remove unpacked files and keep only the archive
            try:
                shutil.rmtree(filepath)
            except OSError as e:
                print("Error: %s - %s." % (e.filepath, e.strerror))
        # todo: this is currently a bit limited because it creates a default 
        # batch gen from a default config
        if custom_batch_gen is None:
            batch_gen = train.DefaultBatchGenerator() 
        else:
            batch_gen = custom_batch_gen
        if custom_config is None:
            configuration = config.make_default(d["num_models"]) 
        else:
            configuration = custom_config
        batch_gen.configure(data, configuration) 
        am = cls(
            data, 
            batch_gen, 
            indices,
            d["batch_size"], 
            model,
            d["gap_symbol"], 
            d["gap_symbol_insertions"]
        )
        if "best_model" in d:
            am.best_model = d["best_model"]
        return am
    
    
    @classmethod
    def decode_core(cls, model_length, state_seqs_max_lik, indices):
        """ Decodes consensus columns as a matrix as well as insertion lengths 
            and starting positions as auxiliary vectors.

        Args: 
            model_length: Number of match states 
                (length of the consensus sequence).
            state_seqs_max_lik: A tensor with the most likeli state sequences. 
                Shape: (num_seq, L)
            indices: Indices in the sequences where decoding should start. 
                Shape: (num_seq)
        Returns:
            consensus_columns: Decoded consensus columns. 
                Shape: (num_seq, model_length)
            insertion_lens: Number of amino acids emitted per insertion state. 
                Shape: (num_seq, model_length-1)
            insertion_start: Starting position of each insertion in the 
                sequences. Shape: (num_seq, model_length-1)
            finished: Boolean vector indicating sequences that are fully
                decoded. Shape: (num_seq) 
        """
        n = state_seqs_max_lik.shape[0]
        c = model_length 
        # initialize the consensus with gaps
        consensus_columns = -np.ones((n, c), dtype=np.int16) 
        # insertion lengths and starting positions per sequence
        insertion_lens = np.zeros((n, c-1), dtype=np.int16)
        insertion_start = -np.ones((n, c-1), dtype=np.int16)
        # is true if and only if the previous hidden state was an insertion 
        # state (not counting flanks)
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


    @classmethod
    def decode_flank(cls, state_seqs_max_lik, flank_state_id, indices):
        """ Decodes flanking insertion states. The deconding is active as long 
            as at least one sequence remains in a flank/unannotated state.
        Args: 
            state_seqs_max_lik: A tensor with the most likeli state sequences. 
                Shape: (num_seq, L)
            flank_state_id: Index of the flanking state.
            indices: Indices in the sequences where decoding should start. 
                Shape: (num_seq)
        Returns:
            insertion_lens: Number of amino acids emitted per insertion state. 
                Shape: (num_seq, model_length-1)
            insertion_start: Starting position of each insertion in the 
                sequences. Shape: (num_seq, model_length-1)
        """
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


    @classmethod
    def decode(cls, model_length, state_seqs_max_lik):
        """ Decodes an implicit alignment (insertion start/length are 
            represented as 2 integers) from most likely state sequences.
        Args: 
            model_length: Number of match states (length of the consensus 
                sequence).
            state_seqs_max_lik: A tensor with the most likeli state sequences. 
                Shape: (num_seq, L)
        Returns:
            core_blocks: Representation of the consensus. 
            left_flank:
            right_flank:
            unannotated_segments:
        """
        n = state_seqs_max_lik.shape[0]
        c = model_length #alias for code readability
        indices = np.zeros(n, np.int16) # active positions in the sequence
        left_flank = cls.decode_flank(state_seqs_max_lik, 0, indices) 
        core_blocks = []
        unannotated_segments = []
        while True:    
            C, IL, IS, finished = cls.decode_core(
                model_length, state_seqs_max_lik, indices
            )
            core_blocks.append((C, IL, IS, finished))
            if np.all(finished):
                break
            unannotated_segments.append(
                cls.decode_flank(state_seqs_max_lik, 2*c, indices)
            )
        right_flank = cls.decode_flank(state_seqs_max_lik, 2*c+1, indices) 
        return core_blocks, left_flank, right_flank, unannotated_segments


    @classmethod
    def get_insertion_block(
        cls, 
        sequences, 
        lens, 
        maxlen, 
        starts, 
        adjust_to_right=False, 
        custom_columns=None
    ):
        """ Constructs one insertion block from an implicitly represented 
        alignment.

        Args: 
        Returns:
        """
        n = sequences.shape[0]
        A = np.arange(n)
        s = len(SequenceDataset.alphabet)
        block = np.zeros((n, maxlen), dtype=np.uint8) + s - 1
        count_down_lens = np.copy(lens)
        active = count_down_lens > 0
        i = 0
        columns = np.stack([np.arange(maxlen)]*n) 
        if custom_columns is not None:
            columns[:, :custom_columns.shape[1]] = custom_columns
        while np.any(active):
            aa = sequences[A[active], starts[active] + i]
            block[active, columns[active,i]] = aa
            count_down_lens -= 1
            active = count_down_lens > 0
            i += 1
        if adjust_to_right and custom_columns is None:
            block_right_aligned = np.zeros_like(block) + s - 1
            for i in range(maxlen):
                
                block_right_aligned[A, (maxlen-lens+i)%maxlen] = block[:, i]
            block = block_right_aligned
        block += s #lower case
        return block


    @classmethod
    def get_alignment_block(
        cls, 
        sequences, 
        consensus, 
        ins_len, 
        ins_len_total, 
        ins_start, 
        is_non_empty=None, 
        custom_columns=None,
        only_matches=False
    ):
        """ Constructs one core model hit block from an implicitly represented 
            alignment.

        Args: 

        Returns:
        """
        A = np.arange(sequences.shape[0])
        if only_matches:
            length = consensus.shape[1]
        else:
            length = consensus.shape[1] + np.sum(ins_len_total)
        block = np.zeros((sequences.shape[0], length), dtype=np.uint8) 
        block += len(SequenceDataset.alphabet) - 1
        i = 0
        columns_to_remove = [] #track empty columns to be removed later
        for c in range(consensus.shape[1]-1):
            column = consensus[:,c]
            ins_l = ins_len[:,c]
            ins_l_total = ins_len_total[c]
            ins_s = ins_start[:,c]
            #one column
            no_gap = column != -1
            block[no_gap,i] = sequences[A[no_gap],column[no_gap]]
            #is this column empty in ALL batches? if yes, mark for removal
            if is_non_empty is not None and not is_non_empty[c]:
                columns_to_remove.append(i)
            i += 1
            #insertion
            if not only_matches:
                if custom_columns is None:
                    custom_column = None
                else:
                    custom_column = custom_columns[c]
                block[:,i:i+ins_l_total] = cls.get_insertion_block(
                    sequences,
                    ins_l,
                    ins_l_total, 
                    ins_s, 
                    custom_columns=custom_column
                )
                i += ins_l_total
        #final column
        no_gap = consensus[:,-1] != -1
        block[no_gap,i] = sequences[A[no_gap],consensus[:,-1][no_gap]]
        if is_non_empty is not None and not is_non_empty[-1]:
            columns_to_remove.append(i)
        #remove columns that are empty in ALL batches
        block = np.delete(block, columns_to_remove, axis=1)
        return block

    
@tf.function
def non_homogeneous_mask_func(i, seq_lens, hmm_cell): 
    """ Let S = S_1 … S_L be the sequence and M_1 … M_Z the match states.
    In a Viterbi path pi = pi_1 … pi_L prevent transitions such that either 
    a) (pi_{i-1}, pi_i) = (M_j, E) and L-i <= Z-j or
    b) (pi_{i-1}, pi_i) = (S, M_j) and i <= j.

    Returns:
        A mask of shape (num_models, batch_size, num_states, num_states) 
        indicating allowed transitions.
    """
    k = hmm_cell.num_models
    q = tf.cast(hmm_cell.max_num_states, tf.int32)
    template = tf.ones((1,q,q), dtype=hmm_cell.dtype)
    model_masks = []
    for k,length in enumerate(hmm_cell.length):
        length = tf.cast(length, tf.int32)
        C = 2 * length
        states_left = one_hot_set([C], q, hmm_cell.dtype)
        states_right = one_hot_set([C], q, hmm_cell.dtype)
        allowed_CL_transitions = 1 - one_hot_set(tf.range(i+1, tf.maximum(i+1, length + 1)), q, hmm_cell.dtype)
        number_of_forbidden_match_states = tf.maximum(0, length - seq_lens[k] + i)
        #always allow transitions out of the last match state
        number_of_forbidden_match_states = tf.minimum(length-1, number_of_forbidden_match_states)
        length_mask = 1-tf.cast(tf.sequence_mask(number_of_forbidden_match_states, maxlen=q-1), hmm_cell.dtype)  
        allowed_CR_transitions = tf.concat([tf.ones_like(length_mask[:,:1]), length_mask], axis=1)
        mask_left = states_left[...,tf.newaxis] * allowed_CL_transitions[tf.newaxis]
        mask_left += template * (1 - states_left[:,tf.newaxis])
        mask_right = states_right[tf.newaxis] * allowed_CR_transitions[...,tf.newaxis]
        mask_right += template * (1 - states_right[tf.newaxis])
        mask = mask_left * mask_right 
        model_masks.append(mask)
    return tf.stack(model_masks, axis=0)


@tf.function
def one_hot_set(indices, d, dtype):
    # Returns a vector in {0,1}}^d with a 1 at positions i in indices and 0 
    # elsewhere
    return tf.reduce_sum(tf.one_hot(indices, d, dtype=dtype), axis=0)


def find_faulty_sequences(
    state_seqs_max_lik, model_length, seq_lens, limit=32000
):
    if state_seqs_max_lik.shape[1] > limit:
        return np.array([], dtype=np.int32)
    else:
        # Returns an array of sequences indices for that Viterbi should be 
        # rerun with restrictions
        C = 2 * model_length
        C_state = state_seqs_max_lik == C
        prev_C_state = np.roll(C_state, 1, axis=2)
        prev_C_state[:,:,0] = False
        C_state_starts = C_state & ~prev_C_state
        previous_state = np.roll(state_seqs_max_lik, 1, axis=2).astype(np.int32)
        previous_state[:,:,0] = -1
        previous_is_match = (previous_state > 0) & (previous_state < model_length+1)
        #there are enough match states to align without repeat
        remaining_matches = model_length - previous_state
        remaining_residues = seq_lens[np.newaxis,:,np.newaxis] - np.arange(state_seqs_max_lik.shape[-1])[np.newaxis,np.newaxis,:]
        enough_matches = remaining_matches >= remaining_residues
        faulty_sequences = np.any(C_state_starts & previous_is_match & enough_matches, axis=-1)
        faulty_sequences = np.argwhere(faulty_sequences[0])[:,0]
        return faulty_sequences