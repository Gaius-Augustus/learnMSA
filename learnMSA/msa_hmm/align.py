import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

import learnMSA.msa_hmm.Initializers as initializers
import learnMSA.msa_hmm.Training as train
import learnMSA.msa_hmm.training_util as training_util
from learnMSA import Configuration
from learnMSA.msa_hmm.AlignInsertions import make_aligned_insertions
from learnMSA.msa_hmm.AlignmentModel import AlignmentModel
from learnMSA.msa_hmm.learnmsa_context import LearnMSAContext
from learnMSA.msa_hmm.model_surgery import do_model_surgery
from learnMSA.msa_hmm.posterior import get_state_expectations
from learnMSA.msa_hmm.SequenceDataset import SequenceDataset
from learnMSA.protein_language_models.MvnEmitter import \
    AminoAcidPlusMvnEmissionInitializer


np.set_printoptions(legacy='1.25')


def align(data : SequenceDataset, config : Configuration) -> AlignmentModel:
    """ Aligns the sequences in data according to the specified config.
    Args:
        data: Dataset of sequences.
        config: Configuration that can be used to control training and decoding
    Returns:
        An AlignmentModel object.
    """
    # If the input/output config is not set, we use the dataset path
    if config.input_output.input_file == Path():
        config.input_output.input_file = data.filepath

    # Create a context that automatically sets up data-dependent parameters
    context = LearnMSAContext(data, config)

    if config.input_output.verbose:
        print(
            f"Training of {config.training.num_model} models on file "\
            f"{Path(config.input_output.input_file).name}"
        )


    if config.input_output.load_model == Path() and \
            config.training.skip_training:
        # Load a model without any training
        am = AlignmentModel.load_models_from_file(
            config.input_output.load_model,
            data,
            custom_batch_gen=context.batch_gen
        )
    else:
        try:
            # Temporary solution: convert the new config to the legacy config format
            # such that the code runs
            from learnMSA.msa_hmm.legacy import make_legacy_config

            t_a = time.time()
            legacy_config = make_legacy_config(config, context) # type: ignore
            if config.visualization.logo_gif:
                am = _fit_and_align_with_logo_gif(
                    data,
                    legacy_config,
                    context.initial_model_length_cb(data, legacy_config), # type: ignore
                    config.visualization.logo_gif.parent if config.visualization.logo_gif else "", # type: ignore
                )
            else:
                am = _fit_and_align(
                    data,
                    legacy_config,
                    context.initial_model_length_cb(data, legacy_config), # type: ignore
                    model_generator=context.model_gen,
                    batch_generator=context.batch_gen,
                    subset=context.subset,
                    sequence_weights=context.sequence_weights,
                    clusters=context.clusters,
                    verbose=config.input_output.verbose,
                    load_model=config.input_output.load_model
                )
            if config.input_output.verbose:
                print("Time for alignment:", "%.4f" % (time.time()-t_a))
        except tf.errors.ResourceExhaustedError as e:
            print("Out of memory. A resource was exhausted.")
            print(
                "Try reducing the batch size (-b). The current batch size "\
                "was: "+str(config.training.batch_size)+"."
            )
            sys.exit(e.error_code)
    tf.keras.backend.clear_session() #not sure if necessary
    am.best_model = select_model(
        am, config.training.model_criterion, config.input_output.verbose
    )

    if config.input_output.output_file == Path():
        return am

    Path(config.input_output.output_file).parent.mkdir(parents=True, exist_ok=True)
    t = time.time()

    if config.training.unaligned_insertions:
        am.to_file(
            config.input_output.output_file,
            am.best_model,
            format=config.input_output.format
        )
    else:
        aligned_insertions = make_aligned_insertions(
            am,
            config.advanced.insertion_aligner,
            config.advanced.aligner_threads,
            verbose=config.input_output.verbose
        )
        am.to_file(
            config.input_output.output_file,
            am.best_model,
            aligned_insertions=aligned_insertions,
            format=config.input_output.format
        )

    if config.input_output.verbose:
        if am.fixed_viterbi_seqs.size > 0:
            max_show_seqs = 5
            print(f"Fixed {am.fixed_viterbi_seqs.size} Viterbi sequences:")
            print("\n".join([
                am.data.seq_ids[i]
                for i in am.fixed_viterbi_seqs[:max_show_seqs]
            ]))
            if am.fixed_viterbi_seqs.size > max_show_seqs:
                print("...")
        print("time for generating output:", "%.4f" % (time.time()-t))
        print("Wrote file", config.input_output.output_file)

    return am


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
    A2M_output: If True, insertions will be indicated by lower case letters in the output and "." will indicate insertions in other sequences.
                Otherwise all upper case letters and only "-" will be used.
    load_model: Path to a pre-trained model to load (if any).
Returns:
    An AlignmentModel object.
"""
def _fit_and_align(
    data : SequenceDataset,
    config,
    model_lengths,
    model_generator=None,
    batch_generator=None,
    subset=None,
    sequence_weights=None,
    clusters=None,
    verbose=True,
    A2M_output=True,
    load_model=""
) -> AlignmentModel:
    model_generator, batch_generator = _make_defaults_if_none(
        model_generator, batch_generator
    )
    if verbose:
        _dataset_messages(data)
    if subset is None:
        subset = np.arange(data.num_seq)
    full_length_estimate = training_util.get_full_length_estimate(
        data.seq_lens, config["surgery_quantile"], config["min_surgery_seqs"]
    )

    # Make dummy initializers for surgery
    if "scoring_model_config" in config:
        emission_dummy = [
            AminoAcidPlusMvnEmissionInitializer(config["scoring_model_config"])
        ]
    else:
        emission_dummy = [initializers.make_default_emission_init()]
    transition_dummy = initializers.make_default_transition_init()
    flank_init_dummy = initializers.make_default_flank_init()

    if load_model:
        # Load the alignment model from file and use it as initialization
        am = AlignmentModel.load_models_from_file(
            load_model, data, custom_batch_gen=batch_generator
        )
        if verbose:
            print("Loaded model from file", load_model)
        # Prevent model surgery from adding or discarding any states,
        # we'll use it to get initial transition and emission parameters
        original_surgery_del = config["surgery_del"]
        original_surgery_ins = config["surgery_ins"]
        config["surgery_del"] = 0.0
        config["surgery_ins"] = 1.0
        config, model_lengths, _ = do_model_surgery(
            0,
            am,
            config,
            emission_dummy,
            transition_dummy,
            flank_init_dummy,
            verbose
        )

    last_iteration=config["max_surgery_runs"]==1
    # 2 staged main loop: Fits model parameters with GD and optimized model
    # architecture with surgery
    for i in range(config["max_surgery_runs"]):
        if callable(config["batch_size"]):
            batch_size = config["batch_size"](
                model_lengths, min(data.max_len, config["crop_long_seqs"])
            )
        else:
            batch_size = config["batch_size"]
        #set the batch size to something smaller than the dataset size even though
        #for low sequence numbers it would be feasible to train on all data at once
        batch_size = min(
            batch_size, training_util.get_low_seq_num_batch_size(data.num_seq)
        )
        if last_iteration:
            train_indices = np.arange(data.num_seq)
            decode_indices = subset
        else:
            train_indices = full_length_estimate
            decode_indices = full_length_estimate
        epochs_this_iteration = config["epochs"][0 if i==0 else 1 if not last_iteration else 2]
        model, history = train.fit_model(
            model_generator,
            batch_generator,
            data,
            train_indices,
            model_lengths,
            config,
            batch_size=batch_size,
            epochs=epochs_this_iteration,
            sequence_weights=sequence_weights,
            clusters=clusters,
            verbose=verbose
        )
        if verbose:
            print("Creating alignment model...")
        am = AlignmentModel(
            data,
            batch_generator,
            decode_indices,
            batch_size=batch_size,
            model=model
        )
        if verbose:
            print("Successfully created alignment model.")
        if last_iteration:
            break
        config, model_lengths, surgery_converged = do_model_surgery(
            i,
            am,
            config,
            emission_dummy,
            transition_dummy,
            flank_init_dummy,
            verbose
        )
        if config["encoder_weight_extractor"] is not None:
            if config["experimental_evolve_upper_half"]:
                print(
                    "Warning: The option experimental_evolve_upper_half is "\
                    "currently not compatible with encoder_weight_extractor. "\
                    "The weight extractor will be ignore."
                )
            else:
                if verbose:
                    print(
                        "Used the encoder_weight_extractor callback to pass "\
                        "the encoder parameters to the next iteration."
                    )
                config["encoder_initializer"] = config["encoder_weight_extractor"](am.encoder_model)
        elif verbose:
            print("Re-initialized the encoder parameters.")
        if verbose and surgery_converged:
            print("Surgery converged.")
        last_iteration = surgery_converged or (i == config["max_surgery_runs"]-2)
    return am


def _fit_and_align_with_logo_gif(
    data : SequenceDataset, config, model_lengths, logo_dir
) -> AlignmentModel:
    import matplotlib.pyplot as plt

    from learnMSA.msa_hmm.Visualize import LogoPlotterCallback, make_logo_gif

    model_generator, batch_generator = _make_defaults_if_none(None, None)
    indices = np.arange(data.num_seq)
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
    make_logo_gif(logo_plotter_callback.frame_dir, logo_dir / "training.gif")
    am = AlignmentModel(data, batch_generator, indices, batch_size=batch_size, model=model)
    return am


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
    warned = False
    for i in range(min(data.num_seq, seq_count_heuristic_gap_check)):
        record = data.get_record(i)
        if '-' in record or '.' in record:
            if not warned:
                print(f"Warning: The sequences in {data.filepath} seem to be already aligned. learnMSA will ignore any gap character.")
                warned = True
    if data.num_seq < seq_count_warning_threshold:
        print(f"Warning: You are aligning {data.num_seq} sequences, although learnMSA is designed for large scale alignments. We recommend to have a sufficiently deep training dataset of at least {seq_count_warning_threshold} sequences for accurate results.")