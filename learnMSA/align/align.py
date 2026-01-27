import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

import learnMSA.msa_hmm.training_util as training_util
from learnMSA import Configuration
from learnMSA.align.align_inserts import make_aligned_insertions
from learnMSA.align.alignment_model import AlignmentModel
from learnMSA.model.select import SelectionCriterion, select_model
from learnMSA.model.surgery import model_surgery
from learnMSA.model.tf.model import LearnMSAModel
from learnMSA.model.context import LearnMSAContext
from learnMSA.util.sequence_dataset import SequenceDataset

np.set_printoptions(legacy='1.21')


def align(
    data : SequenceDataset, config : Configuration
) -> tuple[AlignmentModel, int]:
    """ Aligns the sequences in data according to the specified config.

    Args:
        data: Dataset of sequences.
        config: Configuration that can be used to cgit ontrol training and
            decoding.

    Returns:
        A tuple containing:
        - An AlignmentModel object
        - The index of the best model selected based on the model criterion
    """
    # If the input/output config is not set, we use the dataset path
    if config.input_output.input_file == Path():
        config.input_output.input_file = data.filepath

    # Create working directory if it does not exist
    work_dir = Path(config.input_output.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Create a context that automatically sets up data-dependent parameters
    context = LearnMSAContext(config, data)

    # Write the config to file in the working directory
    config_path = work_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config.model_dump(mode='json'), f, indent=2)
    if config.input_output.verbose:
        print(f"Configuration saved to {config_path}")

    if config.input_output.verbose:
        print(
            f"Training of {config.training.num_model} models on file "\
            f"{Path(config.input_output.input_file).name}"
        )

    # Either load a model from file or train a new model
    if config.input_output.load_model == Path() and \
            config.training.skip_training:
        # Load a model without any training
        am = AlignmentModel.load(config.input_output.load_model, data)
    else:
        try:
            t_a = time.time()
            if config.visualization.logo_gif:
                am = _fit_and_align_with_logo_gif(data, context)
            else:
                am = _fit_and_align(data, context)
            if config.input_output.verbose:
                print("Time for alignment:", "%.4f" % (time.time()-t_a))
        except tf.errors.ResourceExhaustedError as e:
            print("Out of memory. A resource was exhausted.")
            print(
                "Try reducing the batch size (-b). The current batch size "\
                "was: "+str(config.training.batch_size)+"."
            )
            sys.exit(e.error_code)

    tf.keras.backend.clear_session() # TODO: not sure if necessary

    if data.num_seq > config.training.max_seq_model_select:
        # Sample a random subset of sequences for model selection
        ind = np.random.choice(
            data.num_seq,
            config.training.max_seq_model_select,
            replace=False
        )
    else:
        ind = None
    best_model = select_model(
        am.model,
        data,
        SelectionCriterion(config.training.model_criterion),
        sequence_indices=ind,
        verbose=config.input_output.verbose,
    )

    if config.input_output.output_file == Path():
        return am, best_model

    Path(config.input_output.output_file).parent.mkdir(
        parents=True, exist_ok=True
    )
    t = time.time()

    if config.training.unaligned_insertions or config.training.only_matches:
        # Don't align insertions when requested or when only matches need to
        # be written to the output file
        am.to_file(
            config.input_output.output_file,
            best_model,
            format=config.input_output.format,
            only_matches=config.training.only_matches
        )
    else:
        aligned_insertions = make_aligned_insertions(
            am,
            best_model,
            config.advanced.insertion_aligner,
            config.advanced.aligner_threads,
            verbose=config.input_output.verbose
        )
        am.to_file(
            config.input_output.output_file,
            best_model,
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

    return am, best_model

def _fit_and_align(
    data: SequenceDataset,
    context : LearnMSAContext
) -> AlignmentModel:
    """ Utility method that trains a LearnMSAModel and creates an
    AlignmentModel from it.

    Args:
        data: The sequence dataset to align.
        config: Configuration that can be used to control training and decoding
            (see msa_hmm.config.make_default).
        model_generator: Optional callback that generates a user defined model
            (if None, the default model generator will be used).
        batch_generator: Optional callback that generates sequence batches
            defined by user (if None, the default batch generator will be used).
        subset: Optional subset of the sequence ids. Only the specified
            sequences will be aligned but the models will be trained on all
            sequences (if None, all sequences in the dataset will be aligned).
        verbose: If False, all output messages will be disabled.
        A2M_output: If True, insertions will be indicated by lower case letters
            in the output and "." will indicate insertions in other sequences.
            Otherwise all upper case letters and only "-" will be used.
        load_model: Path to a pre-trained model to load (if any).

    Returns:
        An AlignmentModel object.
    """
    config = context.config
    if config.input_output.verbose:
        _dataset_messages(data)

    # Roughly estimate the full length of a protein
    full_length_estimate = training_util.get_full_length_estimate(
        data.seq_lens,
        config.training.surgery_quantile,
        config.training.min_surgery_seqs
    )

    if config.input_output.load_model:
        # Load the alignment model from file and use it as initialization
        am = AlignmentModel.load(config.input_output.load_model, data)
        if config.input_output.verbose:
            print("Loaded model from file", config.input_output.load_model)

        # TODO: legacy code did override context.lengths here
        # still needed?

    # 2 staged main loop: Fits model parameters with GD and optimized model
    # architecture with surgery
    last_iteration = config.training.max_iterations == 1
    for i in range(config.training.max_iterations):
        if callable(context.batch_size):
            batch_size = context.batch_size(data)
        else:
            batch_size = context.batch_size
        # Set the batch size to something smaller than the dataset size even
        # though or low sequence numbers it would be feasible to train on all
        # data at once
        batch_size = min(
            batch_size, training_util.get_low_seq_num_batch_size(data.num_seq)
        )
        if last_iteration:
            train_indices = np.arange(data.num_seq)
            decode_indices = context.subset
        else:
            train_indices = full_length_estimate
            decode_indices = full_length_estimate

        # Create and compile the model
        context.effective_num_seq = train_indices.shape[0] #todo: workaround
        model = LearnMSAModel(context)
        model.build()
        model.compile()

        # Run training
        model.fit(data, train_indices, i, batch_size)

        am = AlignmentModel(data, model, decode_indices)

        if config.input_output.verbose:
            print("Created alignment model successfully.")

        if last_iteration:
            break

        surgery_result = model_surgery(
            am.model,
            data,
            surgery_del = config.training.surgery_del,
            surgery_ins = config.training.surgery_ins,
            verbose = config.input_output.verbose,
        )

        context.model_lengths = surgery_result.model_lengths
        context.config.hmm = surgery_result.config
        if surgery_result.plm_config is not None:
            context.config.language_model = surgery_result.plm_config
        surgery_converged = surgery_result.surgery_converged

        if config.input_output.verbose:
            print("Re-initialized the encoder parameters.")
            if surgery_converged:
                print("Surgery converged.")

        last_iteration = surgery_converged\
            or (i == config.training.max_iterations-2)

    return am


def _fit_and_align_with_logo_gif(
    data: SequenceDataset,
    context : LearnMSAContext
) -> AlignmentModel:
    from learnMSA.msa_hmm.Visualize import LogoPlotterCallback, make_logo_gif
    config = context.config
    indices = np.arange(data.num_seq)
    if config.visualization.logo_gif:
        logo_dir = config.visualization.logo_gif.parent
    else:
        logo_dir = ""
    if callable(context.batch_size):
        batch_size = context.batch_size(
            context.model_lengths, # type: ignore
            min(data.max_len, config.training.crop) # type: ignore
        )
    else:
        batch_size = context.batch_size

    logo_plotter_callback = LogoPlotterCallback(
        logo_dir, data, context.batch_gen, indices, batch_size
    )

    print(
        "Running in logo gif mode. A sequence logo will be generated for each "\
        "training step. This mode is much slower and less accurate (no model "\
        "surgery and just 1 model) than the default mode and should only be "\
        "used for visualization and debugging."
    )

    # Create and compile the model
    model = LearnMSAModel(context)
    model.build()
    model.compile()

    # Run training
    model.fit(data, indices, 0, batch_size, callbacks=[logo_plotter_callback])

    make_logo_gif(logo_plotter_callback.frame_dir, logo_dir / "training.gif") # type: ignore
    am = AlignmentModel(data, model, indices)
    return am


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