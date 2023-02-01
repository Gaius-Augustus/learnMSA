import tensorflow as tf
import numpy as np
import time
import os
import sys
import learnMSA.msa_hmm.Fasta as fasta
import learnMSA.msa_hmm.Training as train
import learnMSA.msa_hmm.MsaHmmLayer as msa_hmm
import learnMSA.msa_hmm.Viterbi as viterbi
from learnMSA.msa_hmm.Configuration import as_str, assert_config
from pathlib import Path


 
""" Trains k independent models on the sequences in a fasta file and returns k "lazy" alignments, where "lazy" means 
    that decoding will only be carried out when the user wants to print the alignment or write it to a file. 
    Decoding is usually expensive and typically it should only be done after a model selection step.
Args: 
    fasta_file: A Fasta object.
    config: Configuration that can be used to control training and decoding (see msa_hmm.config.make_default).
    model_generator: Optional callback that generates a user defined model (if None, the default model generator will be used). 
    batch_generator: Optional callback that generates sequence batches defined by user (if None, the default batch generator will be used).
    subset: Optional subset of the sequence ids. Only the specified sequences will be aligned but the models will be trained on all sequences 
            (if None, all sequences in the fasta file will be aligned).
    verbose: If False, all output messages will be disabled.
Returns:
    An Alignment object.
"""
def fit_and_align(fasta_file, 
                  config,
                  model_generator=None,
                  batch_generator=None,
                  subset=None,
                  verbose=True):
    assert_config(config)
    model_generator, batch_generator = _make_defaults_if_none(model_generator, batch_generator)
    if verbose:
        _fasta_file_messages(fasta_file)
    n = fasta_file.num_seq
    if subset is None:
        subset = np.arange(fasta_file.num_seq)
    full_length_estimate = get_full_length_estimate(fasta_file, config) 
    model_lengths = get_initial_model_lengths(fasta_file, config)
    #model surgery
    last_iteration=config["max_surgery_runs"]==1
    for i in range(config["max_surgery_runs"]):
        if callable(config["batch_size"]):
            batch_size = config["batch_size"](model_lengths, fasta_file.max_len)
        else:
            batch_size = config["batch_size"]
        #set the batch size to something smaller than the dataset size even though
        #for low sequence numbers it would be feasible to train on all data at once
        batch_size = min(batch_size, get_low_seq_num_batch_size(n))
        if last_iteration:    
            train_indices = np.arange(n)
            decode_indices = subset
        else:
            train_indices = full_length_estimate
            decode_indices = full_length_estimate
        epochs_this_iteration = config["epochs"][0 if i==0 else 1 if not last_iteration else 2]
        model, history = train.fit_model(model_generator,
                                          batch_generator,
                                          fasta_file,
                                          train_indices,
                                          model_lengths, 
                                          config,
                                          batch_size=batch_size, 
                                          epochs=epochs_this_iteration,
                                          verbose=verbose)
        alignment = Alignment(fasta_file,
                               batch_generator,
                               decode_indices,
                               batch_size=batch_size, 
                               model=model)
        if last_iteration:
            loglik, prior = compute_loglik(alignment)
            alignment.loglik = loglik 
            alignment.prior = prior
            expected_state = get_state_expectations(fasta_file,
                                    batch_generator,
                                    np.arange(alignment.fasta_file.num_seq),
                                    batch_size,
                                    alignment.msa_hmm_layer,
                                    alignment.encoder_model)
            alignment.expected_state = expected_state
            if verbose:
                print("Fitted models with MAP estimates = ", 
                      ",".join("%.4f" % (l + p) for l,p in zip(loglik, prior)))
            break
        if i == 0: # remember the initializers used in the first iteration
            emission_init_0, transition_init_0, flank_init_0 = _get_initializers(alignment)
        surgery_converged = True
        #duplicate the previous emitters and transitioner and replace their initializers later
        config["emitter"] = [em.duplicate() for em in alignment.msa_hmm_layer.cell.emitter]
        config["transitioner"] = alignment.msa_hmm_layer.cell.transitioner.duplicate()
        pos_expand, expansion_lens, pos_discard = get_discard_or_expand_positions(alignment, 
                                                                                    del_t=config["surgery_del"], 
                                                                                    ins_t=config["surgery_ins"])
        for k in range(config["num_models"]):
            surgery_converged &= pos_expand[k].size == 0 and pos_discard[k].size == 0
            if verbose:
                print(f"expansions model {k}:", list(zip(pos_expand[k], expansion_lens[k])))
                print(f"discards model {k}:", pos_discard[k])
            transition_init, emission_init, flank_init = update_kernels(alignment, 
                                                                        k,
                                                                        pos_expand[k],
                                                                        expansion_lens[k], 
                                                                        pos_discard[k],
                                                                        [e[k] for e in emission_init_0], 
                                                                        transition_init_0[k], 
                                                                        flank_init_0[k])
            for em, old_em, e_init in zip(config["emitter"], alignment.msa_hmm_layer.cell.emitter, emission_init):
                em.emission_init[k] = tf.constant_initializer(e_init) 
                em.insertion_init[k] = tf.constant_initializer(old_em.insertion_kernel[k].numpy())
            config["transitioner"].transition_init[k] = {key : tf.constant_initializer(t) 
                                         for key,t in transition_init.items()}
            config["transitioner"].flank_init[k] = tf.constant_initializer(flank_init)
            model_lengths[k] = emission_init[0].shape[0]
            if model_lengths[k] < 3: 
                raise SystemExit("A problem occured during model surgery: A pHMM is too short (length <= 2).") 
        if config["encoder_weight_extractor"] is not None:
            if verbose:
                print("Used the encoder_weight_extractor callback to pass the encoder parameters to the next iteration.")
            config["encoder_initializer"] = config["encoder_weight_extractor"](alignment.encoder_model)
        elif verbose:
            print("Re-initialized the encoder parameters.")
        last_iteration = surgery_converged or (i == config["max_surgery_runs"]-2)
    return alignment


def run_learnMSA(train_filename,
                 out_filename,
                 config, 
                 model_generator=None,
                 batch_generator=None,
                 ref_filename="", 
                 verbose=True, 
                 select_best_for_comparison=True):
    """ Wraps fit_and_align and adds file parsing, verbosity, model selection, reference file comparison and an outfile file.
    Args: 
        train_filename: Path of a fasta file with the sequences. 
        out_filename: Filepath of the output fasta file with the aligned sequences.
        config: Configuration that can be used to control training and decoding (see msa_hmm.config.make_default).
        model_generator: Optional callback that generates a user defined model (if None, the default model generator will be used). 
        batch_generator: Optional callback that generates sequence batches defined by user(if None, the default batch generator will be used).
        ref_filename: Optional filepath to a reference alignment. If given, the computed alignment is scored and 
                        the score is returned along with the alignment.
        verbose: If False, all output messages will be disabled.
        select_best_for_comparison: If False, all trained models, not just the one with highest score, will be scored.
    Returns:
        An Alignment object.
    """
    if verbose:
        print("Training of", config["num_models"], "models on file", os.path.basename(train_filename))
        print("Configuration:", as_str(config))
    # load the file
    fasta_file = fasta.Fasta(train_filename)  
    # optionally load the reference and find the corresponding sequences in the train file
    if ref_filename != "":
        ref_fasta = fasta.Fasta(ref_filename, aligned=True)
        subset = np.array([fasta_file.seq_ids.index(sid) for sid in ref_fasta.seq_ids])
    else:
        subset = None
    try:
        t_a = time.time()
        alignment = fit_and_align(fasta_file, 
                                  config=config,
                                  model_generator=model_generator,
                                  batch_generator=batch_generator,
                                  subset=subset, 
                                  verbose=verbose)
        if verbose:
            print("Time for alignment:", "%.4f" % (time.time()-t_a))
    except tf.errors.ResourceExhaustedError as e:
        print("Out of memory. A resource was exhausted.")
        print("Try reducing the batch size (-b). The current batch size was: "+str(config["batch_size"])+".")
        sys.exit(e.error_code)
    if config["model_criterion"] == "loglik":
        alignment.best_model = np.argmax(alignment.loglik + alignment.prior)
    elif config["model_criterion"] == "posterior":
        posterior_sums = [np.sum(alignment.expected_state[i, 1:alignment.length[i]+1]) for i in range(alignment.num_models)]
        alignment.best_model = np.argmax(posterior_sums)
        if verbose:
            print("Per model total expected match states:", posterior_sums)
    else:
        raise SystemExit("Invalid model selection criterion. Valid criteria are loglik and posterior.") 
    if verbose:
        likelihoods = ["%.4f" % ll + " (%.4f)" % p for ll,p in zip(alignment.loglik, alignment.prior)]
        print("Per model likelihoods (priors): ", likelihoods)
        print("Selection criterion:", config["model_criterion"])
        print("Best model: ", alignment.best_model, "(0-based)")
        
    Path(os.path.dirname(out_filename)).mkdir(parents=True, exist_ok=True)
    t = time.time()
    alignment.to_file(out_filename, alignment.best_model)
    
    if verbose:
        print("time for generating output:", "%.4f" % (time.time()-t))
        print("Wrote file", out_filename)

    if ref_filename != "":
        if select_best_for_comparison:
            out_file = fasta.Fasta(out_filename, aligned=True) 
            _,sp = out_file.precision_recall(ref_fasta)
            #tc = out_file.tc_score(ref_fasta)
            if verbose:
                print("SP score =", sp)
        else:
            sp = []
            #without setting the logger level the following code can produce tf retracing warnings 
            #these warnings are expected in this special case and should be disabled to keep the output clear
            tf.get_logger().setLevel('ERROR')
            for i in range(alignment.msa_hmm_layer.cell.num_models):
                tmp_file = "tmp.fasta"
                alignment.to_file(tmp_file, i)
                tmp_fasta = fasta.Fasta(tmp_file, aligned=True) 
                _,sp_i = tmp_fasta.precision_recall(ref_fasta)
                print(f"Model {i} SP score =", sp_i)
                os.remove(tmp_file)
                sp.append(sp_i)
            tf.get_logger().setLevel('WARNING')
        return alignment, sp
    else:
        return alignment
    
    
def get_state_expectations(fasta_file,
                           batch_generator,
                           indices,
                           batch_size,
                           msa_hmm_layer, 
                           encoder,
                           reduce=True):
    """ Computes the expected number of occurences per model and state.
    Args:
        fasta_file: Fasta file object.
        batch_generator: Batch generator.
        indices: Indices that specify which sequences in fasta_file should be decoded. 
        batch_size: Specifies how many sequences will be decoded in parallel. 
        msa_hmm_layer: MsaHmmLayer object. 
        encoder: Encoder model that is applied to the sequences before Viterbi.
        reduce: If true (default), the posterior state probs are summed up over the dataset size. 
            Otherwise the posteriors for each sequence are returned.
    Returns:
        The expected number of occurences per model state. Shape (num_model, max_num_states)
    """
    #does currently not support multi-GPU, scale the batch size to account for that and prevent overflow
    num_gpu = len([x.name for x in tf.config.list_logical_devices() if x.device_type == 'GPU']) 
    num_devices = num_gpu + int(num_gpu==0) #account for the CPU-only case 
    batch_size = int(batch_size / num_devices)
    #compute an optimized order for decoding that sorts sequences of equal length into the same batch
    indices = np.reshape(indices, (-1))
    num_indices = indices.shape[0]
    sorted_indices = np.array([[i,j] for l,i,j in sorted(zip(fasta_file.seq_lens[indices], indices, range(num_indices)))])
    msa_hmm_layer.cell.recurrent_init()
    ds = train.make_dataset(sorted_indices[:,0], 
                            batch_generator, 
                            batch_size,
                            shuffle=False)
    
    cell = msa_hmm_layer.cell
    
    @tf.function(input_signature=[tf.TensorSpec(x.shape, dtype=x.dtype) for x in encoder.inputs])
    def batch_posterior_state_probs(inputs, indices):
        encoded_seq = encoder([inputs, indices]) 
        posterior_probs = msa_hmm_layer.state_posterior_log_probs(encoded_seq)
        posterior_probs = tf.math.exp(posterior_probs)
        #compute expected number of visits per hidden state and sum over batch dim
        posterior_probs = tf.reduce_sum(posterior_probs, -2)
        if reduce:
            posterior_probs = tf.reduce_sum(posterior_probs, 1) / num_indices
        return posterior_probs
    
    if reduce:
        posterior_probs = tf.zeros((cell.num_models, cell.max_num_states), cell.dtype) 
        for inputs, _ in ds:
            posterior_probs += batch_posterior_state_probs(inputs[0], inputs[1])
        return posterior_probs.numpy()
    else:
        posterior_probs = np.zeros((cell.num_models, num_indices, cell.max_num_states), cell.dtype) 
        for i, (inputs, _) in enumerate(ds):
            posterior_probs[:,i*batch_size : (i+1)*batch_size] = batch_posterior_state_probs(inputs[0], inputs[1])
        return posterior_probs
    

def decode_core(model_length,
                state_seqs_max_lik,
                indices):
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


def decode_flank(state_seqs_max_lik, 
                 flank_state_id, 
                 indices):
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



def decode(model_length, state_seqs_max_lik):
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
    

def get_alignment_block(sequences, 
                        consensus, 
                        ins_len, 
                        ins_len_total,
                        ins_start):
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
        block[:,i:i+ins_l_total] = get_insertion_block(sequences,
                                                       ins_l,
                                                       ins_l_total, 
                                                       ins_s)
        i += ins_l_total
    #final column
    no_gap = consensus[:,-1] != -1
    block[no_gap,i] = sequences[A[no_gap],consensus[:,-1][no_gap]]
    return block




class Alignment():
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
            if layer.name.startswith("MsaHmmLayer"):
                encoder_out = model.layers[i].output
                self.msa_hmm_layer = layer
                self.encoder_model = tf.keras.Model(inputs=self.model.inputs, outputs=[encoder_out])
        assert self.encoder_model is not None, "Can not find a MsaHmmLayer in the specified model."
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
            decoded_data = decode(l,max_lik_seqs)
            self.metadata[i] = AlignmentMetaData(*decoded_data)

                              
    
    #use only for low sequence numbers
    def to_string(self, model_index, batch_size=100000, add_block_sep=True):
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
    
    
    #returns a dense matrix representing a subset of sequences
    #as specified by batch_indices with respect to the alignment of all sequences
    #(i.e. the sub alignment can contain gap-only columns and 
    #stacking all batches yields a complete alignment)
    def get_batch_alignment(self, model_index, batch_indices, add_block_sep):
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
        left_flank_block = get_insertion_block(sequences, 
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
            alignment_block = get_alignment_block(sequences, 
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
                unannotated_block = get_insertion_block(sequences, 
                                                        unannotated_segment_l[batch_indices],
                                                        self.metadata[model_index].unannotated_segment_lens_total[i],
                                                        unannotated_segment_s[batch_indices])
                blocks.append(unannotated_block)
                if add_block_sep:
                    blocks.append(sep)
        right_flank_block = get_insertion_block(sequences, 
                                               self.metadata[model_index].right_flank_len[batch_indices],
                                               self.metadata[model_index].right_flank_len_total,
                                               self.metadata[model_index].right_flank_start[batch_indices])
        blocks.append(right_flank_block)
        batch_alignment = np.concatenate(blocks, axis=1)
        return batch_alignment
    
    
    def batch_to_string(self, batch_alignment):
        alignment_arr = self.output_alphabet[batch_alignment]
        alignment_strings = [''.join(s) for s in alignment_arr]
        return alignment_strings
        
    
# utility class used in Alignment
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
  
        
def get_discard_or_expand_positions(alignment, 
                                    del_t=0.5, 
                                    ins_t=0.5):
    """ Given an alignment, computes positions for match expansions and discards based on the posterior state probabilities.
    Args: 
        alignment: An Alignment object.
        del_t: Discards match positions that are expected less often than this number.
        ins_t: Expands insertions that are expected more often than this number. 
                Adds new match states according to the expected insertion length. 
    Returns:
        pos_expand: A list of arrays with match positions to expand per model.
        expansion_lens: A list of arrays with the expansion lengths.
        pos_discard: A list of arrays with match positions to discard.
    """
    expected_state = get_state_expectations(alignment.fasta_file,
                                    alignment.batch_generator,
                                    alignment.indices,
                                    alignment.batch_size,
                                    alignment.msa_hmm_layer,
                                    alignment.encoder_model)
    pos_expand = []
    expansion_lens = []
    pos_discard = []
    for i in range(alignment.num_models):
        model_length = alignment.msa_hmm_layer.cell.length[i]
        #discards
        match_states = expected_state[i, 1:model_length+1]
        discard = np.arange(model_length, dtype=np.int32)[match_states < del_t]
        pos_discard.append(discard)
        #expansions
        insert_states = expected_state[i, model_length+1:2*model_length]
        left_flank_state = expected_state[i, 0]
        right_flank_state = expected_state[i, 2*model_length+1]
        all_inserts = np.concatenate([[left_flank_state], insert_states, [right_flank_state]], axis=0)
        which_to_expand = all_inserts > ins_t
        expand = np.arange(model_length+1, dtype=np.int32)[which_to_expand]
        pos_expand.append(expand)
        #expansion lengths
        expand_len = np.ceil(all_inserts).astype(np.int32)[which_to_expand]
        expansion_lens.append(expand_len)
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
                   model_index, 
                    pos_expand, 
                    expansion_lens, 
                    pos_discard, 
                    emission_dummy, 
                    transition_dummy,
                    init_flank_dummy):
    L = alignment.msa_hmm_layer.cell.length[model_index]
    emissions = [em.emission_kernel[model_index].numpy() for em in alignment.msa_hmm_layer.cell.emitter]
    transitions = { key : kernel.numpy() 
                         for key, kernel in alignment.msa_hmm_layer.cell.transitioner.transition_kernel[model_index].items()}
    dtype = alignment.msa_hmm_layer.cell.dtype
    emission_dummy = [d((1, em.shape[-1]), dtype).numpy() for d,em in zip(emission_dummy, emissions)]
    transition_dummy = { key : transition_dummy[key](t.shape, dtype).numpy() for key, t in transitions.items()}
    init_flank_dummy = init_flank_dummy((1), dtype).numpy()
    emissions_new = [apply_mods(k, 
                                  pos_expand, 
                                  expansion_lens, 
                                  pos_discard, 
                                  d) for k,d in zip(emissions, emission_dummy)]
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
    
    # Maybe TODO?: Discarding or extending positions has the side effect of changing all probabilities
    # in begin-state transition distribution. E.g. 
    # Depending on discarded positions, adjust weights such that the residual distribution after 
    # discarding some match states is unaffected.
    # If an insert position is expanded, the transitions from begin to the new match states should have 
    # probabilities according to the initial dummy distribution and the weights of the old transitions 
    # should also be corrected accordingly.
    
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


    

def compute_loglik(alignment, max_ll_estimate = 200000):
    if alignment.fasta_file.num_seq > max_ll_estimate:
        #estimate the ll only on a subset, otherwise for millions of 
        # sequences this step takes rather long for little benefit
        ll_subset = np.arange(alignment.fasta_file.num_seq)
        np.random.shuffle(ll_subset)
        ll_subset = ll_subset[:max_ll_estimate]
        ll_subset = np.sort(ll_subset)
    else:
        #use the sorted indices for optimal length distributions in batches
        ll_subset = alignment.fasta_file.sorted_indices
    ds = train.make_dataset(ll_subset, 
                            alignment.batch_generator,
                            alignment.batch_size, 
                            shuffle=False)
    loglik = np.zeros((alignment.msa_hmm_layer.cell.num_models))
    for x, _ in ds:
        loglik += np.sum(alignment.model(x), axis=0)
    loglik /= ll_subset.size
    prior = alignment.msa_hmm_layer.cell.get_prior_log_density().numpy()/alignment.fasta_file.num_seq
    return loglik, prior

    
def get_full_length_estimate(fasta_file, config):
    n = fasta_file.num_seq
    #ignore short sequences for all surgery iterations except the last
    k = int(min(n*config["surgery_quantile"], 
                max(0, n-config["min_surgery_seqs"])))
    #a rough estimate of a set of only full-length sequences
    full_length_estimate = fasta_file.sorted_indices[k:]
    return full_length_estimate


def get_initial_model_lengths(fasta_file, config, random=True):
    #initial model length
    model_length = np.quantile(fasta_file.seq_lens, q=config["length_init_quantile"])
    model_length *= config["len_mul"]
    model_length = max(3., model_length)
    if random:
        scale = (3 + model_length/30)
        lens = np.round(np.random.normal(loc=model_length, scale=scale, size=config["num_models"])).astype(np.int32)
        lens = np.maximum(lens, 3)
        return lens
    else:
        return [model_length] * config["num_models"]
    
    
def get_low_seq_num_batch_size(n):
    num_gpu = len([x.name for x in tf.config.list_logical_devices() if x.device_type == 'GPU']) 
    num_devices = num_gpu + int(num_gpu==0) #account for the CPU-only case 
    batch_size = int(np.ceil(n*0.5))
    batch_size -= batch_size % num_devices
    return max(batch_size, num_devices)
    
    
def _make_defaults_if_none(model_generator, batch_generator):
    if model_generator is None:
        model_generator = train.default_model_generator
    if batch_generator is None:
        batch_generator = train.DefaultBatchGenerator()
    return model_generator, batch_generator


def _fasta_file_messages(fasta_file, seq_count_warning_threshold=100):
    if fasta_file.gaps:
        print(f"Warning: The file {fasta_file.filename} already contains gaps. Realining the raw sequences.")
    if fasta_file.num_seq < seq_count_warning_threshold:
        print(f"Warning: You are aligning {fasta_file.num_seq} sequences, although learnMSA is designed for large scale alignments. We recommend to have a sufficiently deep training dataset of at least {seq_count_warning_threshold} sequences for accurate results.")
        
        
def _get_initializers(alignment):
    emission_init = [em.emission_init 
                       for em in alignment.msa_hmm_layer.cell.emitter]
    transition_init = alignment.msa_hmm_layer.cell.transitioner.transition_init
    flank_init = alignment.msa_hmm_layer.cell.transitioner.flank_init
    return emission_init, transition_init, flank_init