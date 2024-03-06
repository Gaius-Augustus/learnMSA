import tensorflow as tf
import numpy as np
import time
import os
import sys
import math
from pathlib import Path
import subprocess
from shutil import which
import pandas as pd
from learnMSA.msa_hmm.SequenceDataset import SequenceDataset
import learnMSA.msa_hmm.Training as train
import learnMSA.msa_hmm.Initializers as initializers
from learnMSA.msa_hmm.AlignmentModel import AlignmentModel
from learnMSA.msa_hmm.Configuration import as_str, assert_config
from learnMSA.msa_hmm.AlignInsertions import make_aligned_insertions



def get_initial_model_lengths(data : SequenceDataset, config, random=True):
    #initial model length
    model_length = np.quantile(data.seq_lens, q=config["length_init_quantile"])
    model_length *= config["len_mul"]
    model_length = max(3., model_length)
    if random:
        #scale = (3 + model_length/30)
        scale = (1 + model_length/50)
        lens = np.round(np.random.normal(loc=model_length, scale=scale, size=config["num_models"])).astype(np.int32)
        lens = np.maximum(lens, 3)
        return lens
    else:
        return [model_length] * config["num_models"]
    
    
""" Trains k independent models on the sequences in a dataset and returns k "lazy" alignments, where "lazy" means 
    that decoding will only be carried out when the user wants to print the alignment or write it to a file. 
    Decoding is usually expensive and typically it should only be done after a model selection step.
Args: 
    data: The sequence dataset to align.
    config: Configuration that can be used to control training and decoding (see msa_hmm.config.make_default).
    model_generator: Optional callback that generates a user defined model (if None, the default model generator will be used). 
    batch_generator: Optional callback that generates sequence batches defined by user (if None, the default batch generator will be used).
    subset: Optional subset of the sequence ids. Only the specified sequences will be aligned but the models will be trained on all sequences 
            (if None, all sequences in the dataset will be aligned).
    verbose: If False, all output messages will be disabled.
Returns:
    An AlignmentModel object.
"""
def fit_and_align(data : SequenceDataset, 
                  config,
                  model_generator=None,
                  batch_generator=None,
                  subset=None,
                  initial_model_length_callback=get_initial_model_lengths,
                  sequence_weights=None,
                  verbose=True):
    assert_config(config)
    model_generator, batch_generator = _make_defaults_if_none(model_generator, batch_generator)
    if verbose:
        _dataset_messages(data)
    if subset is None:
        subset = np.arange(data.num_seq)
    full_length_estimate = get_full_length_estimate(data, config) 
    model_lengths = initial_model_length_callback(data, config)
    if hasattr(config["emitter"], '__iter__'):
        emission_dummy = [em.emission_init[0] for em in config["emitter"]]
    else:
        emission_dummy = [config["emitter"].emission_init[0]]
    transition_dummy = config["transitioner"].transition_init[0]
    flank_init_dummy = config["transitioner"].flank_init[0]
    last_iteration=config["max_surgery_runs"]==1
    # 2 staged main loop: Fits model parameters with GD and optimized model architecture with surgery
    for i in range(config["max_surgery_runs"]):
        if callable(config["batch_size"]):
            batch_size = config["batch_size"](model_lengths, min(data.max_len, config["crop_long_seqs"]))
        else:
            batch_size = config["batch_size"]
        #set the batch size to something smaller than the dataset size even though
        #for low sequence numbers it would be feasible to train on all data at once
        batch_size = min(batch_size, get_low_seq_num_batch_size(data.num_seq))
        if last_iteration:    
            train_indices = np.arange(data.num_seq)
            decode_indices = subset
        else:
            train_indices = full_length_estimate
            decode_indices = full_length_estimate
        epochs_this_iteration = config["epochs"][0 if i==0 else 1 if not last_iteration else 2]
        model, history = train.fit_model(model_generator,
                                          batch_generator,
                                          data,
                                          train_indices,
                                          model_lengths, 
                                          config,
                                          batch_size=batch_size, 
                                          epochs=epochs_this_iteration,
                                          sequence_weights=sequence_weights,
                                          verbose=verbose)
        am = AlignmentModel(data, batch_generator, decode_indices, batch_size=batch_size, model=model)
        if last_iteration:
            break
        config, model_lengths, surgery_converged = do_model_surgery(i,
                                                                    am, 
                                                                    config,
                                                                    emission_dummy, 
                                                                    transition_dummy, 
                                                                    flank_init_dummy, 
                                                                    verbose)
        if config["encoder_weight_extractor"] is not None:
            if config["experimental_evolve_upper_half"]:
                print("Warning: The option experimental_evolve_upper_half is currently not compatible with encoder_weight_extractor. The weight extractor will be ignore.")
            else:
                if verbose:
                    print("Used the encoder_weight_extractor callback to pass the encoder parameters to the next iteration.")
                config["encoder_initializer"] = config["encoder_weight_extractor"](am.encoder_model)
        elif verbose:
            print("Re-initialized the encoder parameters.")
        last_iteration = surgery_converged or (i == config["max_surgery_runs"]-2)
    return am


def fit_and_align_with_logo_gif(data : SequenceDataset, config, initial_model_length_callback, logo_dir):
    from learnMSA.msa_hmm.Visualize import LogoPlotterCallback, make_logo_gif
    import matplotlib.pyplot as plt
    
    assert_config(config)
    model_generator, batch_generator = _make_defaults_if_none(None, None)
    indices = np.arange(data.num_seq)
    model_lengths = initial_model_length_callback(data, config)
    if callable(config["batch_size"]):
        batch_size = config["batch_size"](model_lengths, min(data.max_len, config["crop_long_seqs"]))
    else:
        batch_size = config["batch_size"]
        
    logo_plotter_callback = LogoPlotterCallback(logo_dir, data, batch_generator, indices, batch_size)
    print("Running in logo gif mode. A sequence logo will be generated for each training step.")
    print("This mode is much slower and less accurate (no model surgery and just 1 model) than the default mode")
    print("and should only be used for vizualization and debugging.")
    model, history = train.fit_model(model_generator,
                                      batch_generator,
                                      data,
                                      indices,
                                      model_lengths, 
                                      config,
                                      batch_size=batch_size, 
                                      epochs=config["epochs"][-1],
                                      verbose=True,
                                      train_callbacks=[logo_plotter_callback])
    make_logo_gif(logo_plotter_callback.frame_dir, logo_dir+"/training.gif")
    am = AlignmentModel(data, batch_generator, indices, batch_size=batch_size, model=model)
    return am


def run_learnMSA(data : SequenceDataset,
                 out_filename,
                 config, 
                 model_generator=None,
                 batch_generator=None,
                 subset_ids=[], 
                 align_insertions=False,
                 insertion_aligner="famsa",
                 aligner_threads=0,
                 sequence_weights=None,
                 verbose=True, 
                  initial_model_length_callback=get_initial_model_lengths,
                 select_best_for_comparison=True,
                 logo_gif_mode=False,
                 logo_dir=""):
    """ Wraps fit_and_align and adds file parsing, verbosity, model selection, reference file comparison and an outfile file.
    Args: 
        data: Dataset of sequences. 
        out_filename: Filepath of the output fasta file with the aligned sequences.
        config: Configuration that can be used to control training and decoding (see msa_hmm.config.make_default).
        model_generator: Optional callback that generates a user defined model (if None, the default model generator will be used). 
        batch_generator: Optional callback that generates sequence batches defined by user(if None, the default batch generator will be used).
        subset_ids: A list of sequence ids occuring in the training data. If non-empty, only these sequences will be aligned.
        align_insertions: If true, a third party aligner is used to align long insertions after the main MSA step.
        insertion_aligner: Tool to align insertions; "famsa" is installed by default and "clustalo" or "t_coffee" are supported but must be installed manually.
        aligner_threads: Number of threads to use by the aligner.
        verbose: If False, all output messages will be disabled.
        logo_gif_mode: If true, trains with a special mode that generates a sequence logo per train step.
    Returns:
        An AlignmentModel object.
    """
    if verbose:
        print("Training of", config["num_models"], "models on file", os.path.basename(data.filename))
        print("Configuration:", as_str(config)) 
    # optionally load the reference and find the corresponding sequences in the train file
    subset = np.array([data.seq_ids.index(sid) for sid in subset_ids]) if subset_ids else None
    try:
        t_a = time.time()
        if logo_gif_mode:
            am = fit_and_align_with_logo_gif(data, config, initial_model_length_callback, logo_dir)
        else:
            am = fit_and_align(data, 
                                config=config,
                                model_generator=model_generator,
                                batch_generator=batch_generator,
                                subset=subset, 
                                initial_model_length_callback=initial_model_length_callback,
                                sequence_weights=sequence_weights,
                                verbose=verbose)
        if verbose:
            print("Time for alignment:", "%.4f" % (time.time()-t_a))
    except tf.errors.ResourceExhaustedError as e:
        print("Out of memory. A resource was exhausted.")
        print("Try reducing the batch size (-b). The current batch size was: "+str(config["batch_size"])+".")
        sys.exit(e.error_code)
    am.best_model = select_model(am, config["model_criterion"], verbose)
        
    Path(os.path.dirname(out_filename)).mkdir(parents=True, exist_ok=True)
    t = time.time()
    
    if align_insertions:
        aligned_insertions = make_aligned_insertions(am, insertion_aligner, aligner_threads, verbose=verbose)
        am.to_file(out_filename, am.best_model, aligned_insertions = aligned_insertions)
    else:
        am.to_file(out_filename, am.best_model)
    
    if verbose:
        print("time for generating output:", "%.4f" % (time.time()-t))
        print("Wrote file", out_filename)

    return am
    
    
def get_state_expectations(data : SequenceDataset,
                           batch_generator,
                           indices,
                           batch_size,
                           msa_hmm_layer, 
                           encoder,
                           reduce=True):
    """ Computes the expected number of occurences per model and state.
    Args:
        data: The sequence dataset.
        batch_generator: Batch generator.
        indices: Indices that specify which sequences in the dataset should be decoded. 
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
    sorted_indices = np.array([[i,j] for l,i,j in sorted(zip(data.seq_lens[indices], indices, range(num_indices)))])
    msa_hmm_layer.cell.recurrent_init()
    cell = msa_hmm_layer.cell
    old_crop_long_seqs = batch_generator.crop_long_seqs
    batch_generator.crop_long_seqs = math.inf #do not crop sequences
    ds = train.make_dataset(sorted_indices[:,0], 
                            batch_generator, 
                            batch_size,
                            shuffle=False,
                            bucket_by_seq_length=True,
                            model_lengths=cell.length)
    
    @tf.function(input_signature=[[tf.TensorSpec(x.shape, dtype=x.dtype) for x in encoder.inputs]])
    def batch_posterior_state_probs(inputs):
        encoded_seq = encoder(inputs) 
        posterior_probs = msa_hmm_layer.state_posterior_log_probs(encoded_seq)
        posterior_probs = tf.math.exp(posterior_probs)
        #compute expected number of visits per hidden state and sum over batch dim
        posterior_probs = tf.reduce_sum(posterior_probs, -2)
        if reduce:
            posterior_probs = tf.reduce_sum(posterior_probs, 1) / num_indices
        return posterior_probs
    
    if reduce:
        posterior_probs = tf.zeros((cell.num_models, cell.max_num_states), cell.dtype) 
        for (*inputs, _), _ in ds:
            posterior_probs += batch_posterior_state_probs(inputs)
    else:
        posterior_probs = np.zeros((cell.num_models, num_indices, cell.max_num_states), cell.dtype) 
        for (*inputs, batch_indices), _ in ds:
            posterior_probs[:,batch_indices] = batch_posterior_state_probs(inputs)
    batch_generator.crop_long_seqs = old_crop_long_seqs       
    return posterior_probs
    
        
def get_discard_or_expand_positions(am, 
                                    del_t=0.5, 
                                    ins_t=0.5):
    """ Given an AlignmentModel, computes positions for match expansions and discards based on the posterior state probabilities.
    Args: 
        am: An AlignmentModel object.
        del_t: Discards match positions that are expected less often than this number.
        ins_t: Expands insertions that are expected more often than this number. 
                Adds new match states according to the expected insertion length. 
    Returns:
        pos_expand: A list of arrays with match positions to expand per model.
        expansion_lens: A list of arrays with the expansion lengths.
        pos_discard: A list of arrays with match positions to discard.
    """
    expected_state = get_state_expectations(am.data,
                                    am.batch_generator,
                                    am.indices,
                                    am.batch_size,
                                    am.msa_hmm_layer,
                                    am.encoder_model).numpy()
    pos_expand = []
    expansion_lens = []
    pos_discard = []
    for i in range(am.num_models):
        model_length = am.msa_hmm_layer.cell.length[i]
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
def update_kernels(am,
                   model_index, 
                    pos_expand, 
                    expansion_lens, 
                    pos_discard, 
                    emission_dummy, 
                    transition_dummy,
                    init_flank_dummy,
                    mutate=False):
    L = am.msa_hmm_layer.cell.length[model_index]
    emissions = [em.emission_kernel[model_index].numpy() for em in am.msa_hmm_layer.cell.emitter]
    transitions = { key : kernel.numpy() 
                         for key, kernel in am.msa_hmm_layer.cell.transitioner.transition_kernel[model_index].items()}
    if mutate:
        for i in range(len(emissions)):
            noise = np.random.normal(scale=0.2, size=emissions[i].shape)
            emissions[i] += noise
        for key in transitions:
            noise = np.random.normal(scale=0.2, size=transitions[key].shape)
            transitions[key] += noise
    dtype = am.msa_hmm_layer.cell.dtype
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

    
def get_full_length_estimate(data : SequenceDataset, config):
    n = data.num_seq
    #ignore short sequences for all surgery iterations except the last
    k = int(min(n*config["surgery_quantile"], 
                max(0, n-config["min_surgery_seqs"])))
    #a rough estimate of a set of only full-length sequences
    sorted_indices = np.array([i for l,i in sorted(zip(data.seq_lens, range(data.num_seq)))])
    full_length_estimate = sorted_indices[k:]
    return full_length_estimate
    
    
def get_low_seq_num_batch_size(n):
    num_gpu = len([x.name for x in tf.config.list_logical_devices() if x.device_type == 'GPU']) 
    num_devices = num_gpu + int(num_gpu==0) #account for the CPU-only case 
    batch_size = int(np.ceil(n*0.5))
    batch_size -= batch_size % num_devices
    return max(batch_size, num_devices)


def do_model_surgery(iteration, am : AlignmentModel, config, emission_dummy, transition_dummy, flank_init_dummy, verbose=False):
    config = dict(config)
    surgery_converged = True
    #duplicate the previous emitters and transitioner and replace their initializers later
    config["emitter"] = [em.duplicate() for em in am.msa_hmm_layer.cell.emitter]
    config["transitioner"] = am.msa_hmm_layer.cell.transitioner.duplicate()
    pos_expand, expansion_lens, pos_discard = get_discard_or_expand_positions(am, 
                                                                                del_t=config["surgery_del"], 
                                                                                ins_t=config["surgery_ins"])
    model_lengths = []
    #evolve only after the first iteration
    #otherwise this would rule out models starting with too few or too many
    #matches, that could still turn out good eventually
    #so we wait until all models had one iteration to adapt their length
    if config["experimental_evolve_upper_half"] and iteration > 0:
        scores = get_model_scores(am, config["model_criterion"], verbose)
        p = int(np.floor(config["num_models"]/2))
        best_p_models = tf.argsort(-scores)[:p]
        models_for_next_iteration = tf.tile(best_p_models, [2])
        if verbose:
            print(f"Evolving the upper half of the models.")
    else:
        models_for_next_iteration = range(config["num_models"])
    for i,k in enumerate(models_for_next_iteration):
        surgery_converged &= pos_expand[k].size == 0 and pos_discard[k].size == 0
        if verbose:
            print(f"expansions model {i}:", list(zip(pos_expand[k], expansion_lens[k])))
            print(f"discards model {i}:", pos_discard[k])
        transition_init, emission_init, flank_init = update_kernels(am, 
                                                                    k,
                                                                    pos_expand[k],
                                                                    expansion_lens[k], 
                                                                    pos_discard[k],
                                                                    emission_dummy, 
                                                                    transition_dummy, 
                                                                    flank_init_dummy,
                                                                    mutate=config["experimental_evolve_upper_half"])
        for em, old_em, e_init in zip(config["emitter"], am.msa_hmm_layer.cell.emitter, emission_init):
            em.emission_init[i] = initializers.ConstantInitializer(e_init) 
            em.insertion_init[i] = initializers.ConstantInitializer(old_em.insertion_kernel[k].numpy())
        config["transitioner"].transition_init[i] = {key : initializers.ConstantInitializer(t) 
                                     for key,t in transition_init.items()}
        config["transitioner"].flank_init[i] = initializers.ConstantInitializer(flank_init)
        model_lengths.append(emission_init[0].shape[0])
        if model_lengths[-1] < 3: 
            raise SystemExit("A problem occured during model surgery: A pHMM is too short (length <= 2).") 
    return config, model_lengths, surgery_converged


#computes clustering based sequence weights if mmseqs2 is installed
def compute_sequence_weights(fasta_filename, directory, cluster_seq_id=0.5):
    if which("mmseqs") is None:
        print("mmseqs2 is not installed or not in PATH. Consider installing it with conda install -c bioconda mmseqs2 or disable sequence weighting.")
        sys.exit(1)
    else:
        cluster_files = directory+"/"+os.path.splitext(os.path.basename(fasta_filename))[0]
        result = subprocess.run(["mmseqs", 
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
                                  "-v", "0"
                                 ], check=True)
        clustering = pd.read_csv(cluster_files + "_cluster.tsv", sep="\t", names=["representative", "sequence"])
        cluster_counts = clustering.groupby("representative").size().to_frame("cluster_size")
        clustering = clustering.merge(cluster_counts, how="left", on="representative")
        clustering["weight"] = 1/clustering["cluster_size"]
        clustering = clustering.set_index("sequence")

        with SequenceDataset(fasta_filename, "fasta") as data:
            data.validate_dataset()
            ids = data.seq_ids
            #mmseqs2 omits database names and database specific accession numbers, we have to omit them too
            #i.e. from ">database|accession|name" mmseqs only keeps ">name"
            for i in range(len(ids)):
                if "|" in ids[i]:
                    pos = ids[i].rfind("|")
                    if pos != -1:
                        ids[i] = ids[i][pos+1:]
            sequence_weights = np.array(clustering.loc[ids].weight, dtype=np.float32)
            
        return sequence_weights


def get_model_scores(am, model_criterion, verbose):
    selection_criteria = {
        "posterior": select_model_posterior,
        "loglik": select_model_loglik,
        "AIC": select_model_AIC,
        "consensus": select_model_consensus
    }
    if model_criterion not in selection_criteria:
        raise SystemExit(f"Invalid model selection criterion. Valid criteria are: {list(selection_criteria.keys())}.") 
    return selection_criteria[model_criterion](am, verbose)


def select_model(am, model_criterion, verbose):
    scores = get_model_scores(am, model_criterion, verbose)
    best = np.argmax(scores)
    if verbose:
        print("Selection criterion:", model_criterion)
        print("Best model: ", best, "(0-based)")
    return best
            
    
def select_model_posterior(am, verbose=False):
    expected_state = get_state_expectations(am.data,
                                            am.batch_generator,
                                            np.arange(am.data.num_seq),
                                            am.batch_size,
                                            am.msa_hmm_layer,
                                            am.encoder_model)
    posterior_sums = [np.sum(expected_state[i, 1:am.length[i]+1]) for i in range(am.num_models)]
    if verbose:
        print("Total expected match states:", posterior_sums)
    return posterior_sums


#TODO: the default is to use the prior although not using is seems to be very slightly better
#the default argument should change later to false but keep using prior for now for legacy reasons
def select_model_loglik(am, verbose=False, use_prior=True):
    loglik = am.compute_loglik()
    score = tf.identity(loglik)
    if use_prior:
        prior = am.compute_log_prior()
        score += prior
    if verbose:
        if use_prior:
            likelihoods = ["%.4f" % ll + " (%.4f)" % p for ll,p in zip(loglik, prior)]
            print("Likelihoods (priors): ", likelihoods)
        else:
            likelihoods = ["%.4f" % ll for ll in loglik]
            print("Likelihoods: ", likelihoods)
            print("Mean likelihood: ", np.mean(loglik))
    return score


def select_model_AIC(am, verbose=False):
    loglik = select_model_loglik(am, verbose, use_prior=False)
    aic = am.compute_AIC(loglik=loglik)
    return -aic #negate as we want to take the maximum


def select_model_consensus(am, verbose=False):
    consensus = am.compute_consensus_score()
    if verbose:
        print("Consensus scores: ", ["%.4f" % c for c in consensus])
    return consensus
    
    
def _make_defaults_if_none(model_generator, batch_generator):
    if model_generator is None:
        model_generator = train.default_model_generator
    if batch_generator is None:
        batch_generator = train.DefaultBatchGenerator()
    return model_generator, batch_generator


def _dataset_messages(data : SequenceDataset, seq_count_heuristic_gap_check=100, seq_count_warning_threshold=100):
    # a quick heuristic check of the first sequences to see if they contain gaps
    for i in range(min(data.num_seq, seq_count_heuristic_gap_check)):
        record = data.get_record(i)
        if '-' in record or '.' in record:
            print(f"Warning: The sequences in {data.filename} seem to be already aligned. learnMSA will ignore any gap character.")
    if data.num_seq < seq_count_warning_threshold:
        print(f"Warning: You are aligning {data.num_seq} sequences, although learnMSA is designed for large scale alignments. We recommend to have a sufficiently deep training dataset of at least {seq_count_warning_threshold} sequences for accurate results.")