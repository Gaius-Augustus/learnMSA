import tensorflow as tf
import numpy as np
import learnMSA.msa_hmm.AncProbsLayer as anc_probs
import learnMSA.msa_hmm.MsaHmmLayer as msa_hmm_layer
import learnMSA.msa_hmm.MsaHmmCell as msa_hmm_cell
import learnMSA.msa_hmm.Fasta as fasta
import learnMSA.msa_hmm.Configuration as config
import learnMSA.msa_hmm.Training as train
import learnMSA.msa_hmm.Viterbi as viterbi
import learnMSA.msa_hmm.Priors as priors
import learnMSA.msa_hmm.Transitioner as trans
import learnMSA.msa_hmm.Emitter as emit
import json
import shutil
from pathlib import Path


class AlignmentModel():
    """ Decodes alignments from a number of models, stores them in a memory friendly representation and
        generates table-form (memory unfriendly) alignments on demand (batch-wise mode possible).
    Args:
        fasta_file: A fasta file with the sequences to decode.
        batch_generator: An already configured batch generator.
        indices: (A subset of) The sequence indices from the fasta to align (1D).
        batch_size: Controls memory consumption of viterbi.
        model: A learnMSA model which internally might represent multiple pHMM models.
        gap_symbol: Character used to denote missing match positions.
        gap_symbol_insertions: Character used to denote insertions in other sequences.
    """
    def __init__(self, 
                 fasta_file, 
                 batch_generator,
                 indices, 
                 batch_size, 
                 model,
                 gap_symbol="-",
                 gap_symbol_insertions="."):
        self.fasta_file = fasta_file
        self.batch_generator = batch_generator
        self.indices = indices
        self.batch_size = batch_size
        self.model = model
        #encoder model is the same as model but with the MsaHmmLayer removed
        #the output of the encoder model will be the input to viterbi
        #in the default learnMSA, the encoder model is only the Ancestral Probability layer.
        self.encoder_model = None
        for i, layer in enumerate(model.layers[1:]):
            if layer.name.startswith("msa_hmm_layer"):
                encoder_out = model.layers[i].output
                self.msa_hmm_layer = layer
                self.encoder_model = tf.keras.Model(inputs=self.model.inputs, outputs=[encoder_out])
        assert self.encoder_model is not None, "Can not find a MsaHmmLayer in the specified model."
        self.gap_symbol = gap_symbol
        self.gap_symbol_insertions = gap_symbol_insertions
        self.output_alphabet = np.array((fasta.alphabet[:-1] + 
                                        [gap_symbol] + 
                                        [aa.lower() for aa in fasta.alphabet[:-1]] + 
                                        [gap_symbol_insertions, "$"]))
        self.metadata = {}
        self.num_models = self.msa_hmm_layer.cell.num_models
        self.length = self.msa_hmm_layer.cell.length
        
    #computes an implicit alignment (without storing gaps)
    #eventually, an alignment with explicit gaps can be written 
    #in a memory friendly manner to file
    def _build_alignment(self, models):
        
        if tf.distribute.has_strategy():
            with tf.distribute.get_strategy().scope():
                cell_copy = self.msa_hmm_layer.cell.duplicate(models)
        else:
            cell_copy = self.msa_hmm_layer.cell.duplicate(models)
            
        cell_copy.build((self.num_models, None, None, self.msa_hmm_layer.cell.dim))
        state_seqs_max_lik = viterbi.get_state_seqs_max_lik(self.fasta_file,
                                                            self.batch_generator,
                                                            self.indices,
                                                            self.batch_size,
                                                            cell_copy,
                                                            models,
                                                            self.encoder_model)
        for i,l,max_lik_seqs in zip(models, cell_copy.length, state_seqs_max_lik):
            decoded_data = AlignmentModel.decode(l,max_lik_seqs)
            self.metadata[i] = AlignmentMetaData(*decoded_data)
        
    def to_string(self, model_index, batch_size=100000, add_block_sep=True):
        """ Uses one model to decode an alignment and returns the sequences with gaps in a list.
            Note that this method is not suitable im memory is limited and alignment depths and width are large.
        Args:
            model_index: Specifies the model for decoding. Use a suitable criterion like loglik to decide for a model.
            batch_size: Defines how many sequences are decoded at a time with no effect on the output MSA. It can be useful to
                        lower this if memory is sufficient to store the table-form alignment but GPU memory used for decoding a batch is limited.
            add_block_sep: If true, columns containing a special character are added to the alignment indicating domain boundaries.
        """
        alignment_strings_all = []
        n = self.indices.size
        i = 0
        while i < n:
            batch_indices = np.arange(i, min(n, i+batch_size))
            batch_alignment = self.get_batch_alignment(model_index, batch_indices, add_block_sep)
            alignment_strings = self.batch_to_string(batch_alignment)
            alignment_strings_all.extend(alignment_strings)
            i += batch_size
        return alignment_strings_all
    
    def to_file(self, filepath, model_index, batch_size=100000, add_block_sep=False):
        """ Uses one model to decode an alignment and stores it in fasta file format. Currently no other output format is supported.
            The file is written batch wise. The memory required for this operation must be large enough to hold decode and store a single batch
            of aligned sequences but not the whole alignment.
        Args:
            model_index: Specifies the model for decoding. Use a suitable criterion like loglik to decide for a model.
            batch_size: Defines how many sequences are decoded at a time with no effect on the output MSA. It can be useful to
                        lower this if memory is sufficient to store the table-form alignment but GPU memory used for decoding a batch is limited.
            add_block_sep: If true, columns containing a special character are added to the alignment indicating domain boundaries.
        """
        with open(filepath, "w") as output_file:
            n = self.indices.size
            i = 0
            while i < n:
                batch_indices = np.arange(i, min(n, i+batch_size))
                batch_alignment = self.get_batch_alignment(model_index, batch_indices, add_block_sep)
                alignment_strings = self.batch_to_string(batch_alignment)
                for s, seq_ind in zip(alignment_strings, batch_indices):
                    seq_id = self.fasta_file.seq_ids[self.indices[seq_ind]]
                    output_file.write(">"+seq_id+"\n")
                    output_file.write(s+"\n")
                i += batch_size
    
    def get_batch_alignment(self, model_index, batch_indices, add_block_sep):
        """ Returns a dense matrix representing a subset of sequences
            as specified by batch_indices with respect to the alignment of all sequences
            (i.e. the sub alignment can contain gap-only columns and stacking all batches 
            yields a complete alignment).
        Args:
            model_index: Specifies the model for decoding. Use a suitable criterion like loglik to decide for a model.
            batch_indices: Sequence indices / indices of alignment rows.
            add_block_sep: If true, columns containing a special character are added to the alignment indicating domain boundaries.
        """
        if not model_index in self.metadata:
            self._build_alignment([model_index])
        b = batch_indices.size
        sequences = np.zeros((b, self.fasta_file.max_len), dtype=np.uint16) + (fasta.s-1)
        for i,j in enumerate(batch_indices):
            l = self.fasta_file.seq_lens[self.indices[j]]
            sequences[i, :l] = self.fasta_file.get_raw_seq(self.indices[j])
        blocks = []  
        if add_block_sep:
            sep = np.zeros((b,1), dtype=np.uint16) + 2*fasta.s
        left_flank_block = AlignmentModel.get_insertion_block(sequences, 
                                               self.metadata[model_index].left_flank_len[batch_indices],
                                               self.metadata[model_index].left_flank_len_total,
                                               self.metadata[model_index].left_flank_start[batch_indices],
                                               align_to_right=True)
        blocks.append(left_flank_block)
        if add_block_sep:
            blocks.append(sep)
        for i in range(self.metadata[model_index].num_repeats):
            consensus = self.metadata[model_index].consensus[i]
            ins_len = self.metadata[model_index].insertion_lens[i]
            ins_start = self.metadata[model_index].insertion_start[i]
            ins_len_total = self.metadata[model_index].insertion_lens_total[i]
            alignment_block = AlignmentModel.get_alignment_block(sequences, 
                                                  consensus[batch_indices], 
                                                  ins_len[batch_indices], 
                                                  ins_len_total,
                                                  ins_start[batch_indices])
            blocks.append(alignment_block)
            if add_block_sep:
                blocks.append(sep)
            if i < self.metadata[model_index].num_repeats-1:
                unannotated_segment_l = self.metadata[model_index].unannotated_segments_len[i]
                unannotated_segment_s = self.metadata[model_index].unannotated_segments_start[i]
                unannotated_block = AlignmentModel.get_insertion_block(sequences, 
                                                        unannotated_segment_l[batch_indices],
                                                        self.metadata[model_index].unannotated_segment_lens_total[i],
                                                        unannotated_segment_s[batch_indices])
                blocks.append(unannotated_block)
                if add_block_sep:
                    blocks.append(sep)
        right_flank_block = AlignmentModel.get_insertion_block(sequences, 
                                               self.metadata[model_index].right_flank_len[batch_indices],
                                               self.metadata[model_index].right_flank_len_total,
                                               self.metadata[model_index].right_flank_start[batch_indices])
        blocks.append(right_flank_block)
        batch_alignment = np.concatenate(blocks, axis=1)
        return batch_alignment
    
    def batch_to_string(self, batch_alignment):
        """ Converts a dense matrix into string format.
        """
        alignment_arr = self.output_alphabet[batch_alignment]
        alignment_strings = [''.join(s) for s in alignment_arr]
        return alignment_strings
    
    def compute_loglik(self, max_seq=200000):
        """ Computes the logarithmic likelihood for each underlying model.
        Args:
            max_seq: Threshold for the number of sequences used to compute the loglik. If
                    the underlying fasta file has mroe sequences, a random subset is drawn.
        """
        if self.fasta_file.num_seq > max_seq:
            #estimate the ll only on a subset, otherwise for millions of 
            # sequences this step takes rather long for little benefit
            ll_subset = np.arange(self.fasta_file.num_seq)
            np.random.shuffle(ll_subset)
            ll_subset = ll_subset[:max_seq]
            ll_subset = np.sort(ll_subset)
        else:
            #use the sorted indices for optimal length distributions in batches
            ll_subset = self.fasta_file.sorted_indices
        ds = train.make_dataset(ll_subset, 
                                self.batch_generator,
                                self.batch_size, 
                                shuffle=False)
        loglik = np.zeros((self.msa_hmm_layer.cell.num_models))
        for x, _ in ds:
            loglik += np.sum(self.model(x), axis=0)
        loglik /= ll_subset.size
        return loglik
    
    def compute_log_prior(self):
        """ Computes the logarithmic prior value of each underlying model.
        """
        return self.msa_hmm_layer.cell.get_prior_log_density().numpy()/self.fasta_file.num_seq
    
    def compute_AIC(self, max_seq=200000, loglik=None):
        """ Computes the Akaike information criterion for each underlying model. 
        Args:
            max_seq: Threshold for the number of sequences used to compute the loglik. If
                    the underlying fasta file has mroe sequences, a random subset is drawn.
            loglik: This argument can be set if the loglik was computed before via compute_loglik to avoid overhead. 
                    If None, the loglik will be computed internally.
        """
        if loglik is None:
            loglik = self.compute_loglik(max_seq)
        num_param = 34 * np.array(self.length) + 25
        aic = -2 * loglik * self.fasta_file.num_seq + 2*num_param
        return aic 
    
    def compute_consensus_score(self):
        """ Computes a consensus score that rates how plausible each model is with respect to all other models.
            (Relevant for users not using the default emitter: Uses the make_B_amino method of the first emitter.)
        """
        #compute the match sequence of all models padded with terminal symbols 
        match_seqs = np.zeros((self.num_models, max(self.length)+1, self.msa_hmm_layer.cell.dim))
        match_seqs[:,:,-1] = 1 #initialize with terminal symbols
        for i,L in enumerate(self.length):
            match_seqs[i, :L] = self.msa_hmm_layer.cell.emitter[0].make_B_amino()[i,1:L+1]
        #we need to tile the match sequences over the batch dimension because each model should see all other models
        match_seqs = tf.stack([match_seqs]*self.num_models, axis=1)
        #rate the match seqs with respect to the models and cancel out self-rating
        
        #TODO this does not work with self.model which expects indices rather than distributions
        #this is a workaround, but will break with a user defined encoder model
        #we skip the anc probs for simplicity as this would require fitting evolutionary times
        consensus_logliks = self.msa_hmm_layer(match_seqs)
        
        consensus_logliks *= 1-tf.eye(self.num_models)
        # Axis 1 means we reduce over the batch dimension rather than the model dimension,
        # so output i will be the mean loglik if we input all other match sequenes to model i.
        # Using axis 0 here is not the same!
        # Consider the case that all models have the same match sequence but one model allows many deletions or insertions.
        # This model is the outlier and should clearly have the lowest score.
        # With axis=0, the likelihood of the outlier model under all other models is high and the scores of the other models will have a penalty
        # since their match sequences are fed into the outlier model.
        # With axis=1, the scores of all models will be high except for the outlier model, which has a strong penalty as it rates all other match sequences
        # involving the deletion/insertion probabilities.
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
            "fasta_file" : self.fasta_file.filename,
            "batch_size" : self.batch_size,
            "gap_symbol" : self.gap_symbol,
            "gap_symbol_insertions" : self.gap_symbol_insertions,
        }
        with open(filepath+"/meta.json", "w") as metafile:
            metafile.write(json.dumps(d, indent=4))
        #serialize indices
        np.savetxt(filepath+"/indices", self.indices, fmt='%i')
        #save the model
        self.model.save(filepath+"/model", save_traces=False) 
        if pack:
            shutil.make_archive(filepath, "zip", filepath)
            try:
                shutil.rmtree(filepath)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))

    
    @classmethod
    def load_models_from_file(cls, filepath, from_packed=True, custom_batch_gen=None):
        """ Recreates an AlignmentModel instance with underlying models from file.
        Args:
            filepath: Path of the file to load.
            from_packed: Pass true or false depending on the pack argument used with write_models_to_file.
        Returns:
            An AlignmentModel instance with equivalent behavior as the AlignmentModel instance used while saving the model.
        """
        if from_packed:
            shutil.unpack_archive(filepath+".zip", filepath)
        #deserialize metadata    
        with open(filepath+"/meta.json") as metafile:
            d = json.load(metafile)
        fasta_file = fasta.Fasta(d["fasta_file"])
        #deserialize indices
        indices = np.loadtxt(filepath+"/indices", dtype=int)
        #load the model
        model = tf.keras.models.load_model(filepath+"/model", 
                                           custom_objects={"AncProbsLayer": anc_probs.AncProbsLayer, 
                                                           "MsaHmmLayer": msa_hmm_layer.MsaHmmLayer,
                                                           "MsaHmmCell": msa_hmm_cell.MsaHmmCell, 
                                                           "ProfileHMMTransitioner": trans.ProfileHMMTransitioner, 
                                                           "ProfileHMMEmitter": emit.ProfileHMMEmitter,
                                                           "AminoAcidPrior": priors.AminoAcidPrior,
                                                           "NullPrior": priors.NullPrior,
                                                           "ProfileHMMTransitionPrior": priors.ProfileHMMTransitionPrior})
        if from_packed:
            #after loading remove unpacked files and keep only the archive
            try:
                shutil.rmtree(filepath)
            except OSError as e:
                print("Error: %s - %s." % (e.filepath, e.strerror))
        #todo: this is currently a bit limited because it creates a default batch gen from a default config
        batch_gen = train.DefaultBatchGenerator() if custom_batch_gen is None else custom_batch_gen
        batch_gen.configure(fasta_file, config.make_default(d["num_models"])) 
        return cls(fasta_file, 
                  batch_gen, 
                  indices,
                  d["batch_size"], 
                  model,
                  d["gap_symbol"], 
                  d["gap_symbol_insertions"])
    
    
    @classmethod
    def decode_core(cls, model_length, state_seqs_max_lik, indices):
        """ Decodes consensus columns as a matrix as well as insertion lengths and starting positions
            as auxiliary vectors.
        Args: 
            model_length: Number of match states (length of the consensus sequence).
            state_seqs_max_lik: A tensor with the most likeli state sequences. Shape: (num_seq, L)
            indices: Indices in the sequences where decoding should start. Shape: (num_seq)
        Returns:
            consensus_columns: Decoded consensus columns. Shape: (num_seq, model_length)
            insertion_lens: Number of amino acids emitted per insertion state. Shape: (num_seq, model_length-1)
            insertion_start: Starting position of each insertion in the sequences. Shape: (num_seq, model_length-1)
            finished: Boolean vector indicating sequences that are fully decoded. Shape: (num_seq) 
        """
        n = state_seqs_max_lik.shape[0]
        c = model_length 
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


    @classmethod
    def decode_flank(cls, state_seqs_max_lik, flank_state_id, indices):
        """ Decodes flanking insertion states. The deconding is active as long as at least one sequence remains 
            in a flank/unannotated state.
        Args: 
            state_seqs_max_lik: A tensor with the most likeli state sequences. Shape: (num_seq, L)
            flank_state_id: Index of the flanking state.
            indices: Indices in the sequences where decoding should start. Shape: (num_seq)
        Returns:
            insertion_lens: Number of amino acids emitted per insertion state. Shape: (num_seq, model_length-1)
            insertion_start: Starting position of each insertion in the sequences. Shape: (num_seq, model_length-1)
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
        """ Decodes an implicit alignment (insertion start/length are represented as 2 integers) 
            from most likely state sequences.
        Args: 
            model_length: Number of match states (length of the consensus sequence).
            state_seqs_max_lik: A tensor with the most likeli state sequences. Shape: (num_seq, L)
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
            C, IL, IS, finished = cls.decode_core(model_length, state_seqs_max_lik, indices)
            core_blocks.append((C, IL, IS, finished))
            if np.all(finished):
                break
            unannotated_segments.append( cls.decode_flank(state_seqs_max_lik, 2*c, indices) )
        right_flank = cls.decode_flank(state_seqs_max_lik, 2*c+1, indices) 
        return core_blocks, left_flank, right_flank, unannotated_segments


    @classmethod
    def get_insertion_block(cls, sequences, lens, maxlen, starts, align_to_right=False):
        """ Constructs one insertion block from an implicitly represented alignment.
        Args: 
        Returns:
        """
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


    @classmethod
    def get_alignment_block(cls, sequences, consensus, ins_len, ins_len_total, ins_start):
        """ Constructs one core model hit block from an implicitly represented alignment.
        Args: 
        Returns:
        """
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
            block[:,i:i+ins_l_total] = cls.get_insertion_block(sequences,
                                                           ins_l,
                                                           ins_l_total, 
                                                           ins_s)
            i += ins_l_total
        #final column
        no_gap = consensus[:,-1] != -1
        block[no_gap,i] = sequences[A[no_gap],consensus[:,-1][no_gap]]
        return block

    
        
        
# utility class used in AlignmentModel storing useful information on a specific alignment
class AlignmentMetaData():
    def __init__(self, 
                 core_blocks, 
                 left_flank, 
                 right_flank, 
                 unannotated_segments):
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