import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from learnMSA.align.align_hits import HitAlignmentMode
import learnMSA.model.training_util as training_util
from learnMSA import Configuration
from learnMSA.align.alignment_model import AlignmentModel
from learnMSA.model.surgery import model_surgery
from learnMSA.model.tf.model import LearnMSAModel
from learnMSA.model.context import LearnMSAContext
from learnMSA.util.sequence_dataset import Dataset, SequenceDataset

np.set_printoptions(legacy='1.21')


def align(
    data : SequenceDataset | tuple[SequenceDataset, *tuple[Dataset, ...]],
    config : Configuration,
) -> AlignmentModel:
    """ Aligns the sequences in data according to the specified config.

    Args:
        data: SequenceDataset or tuple of Dataset(s) with the first dataset
                being a SequenceDataset (of amino acid sequences).
                Whether multiple datasets can be passed and how they should be
                ordered bepends on the configuration.
        config: Configuration that can be used to cgit ontrol training and
            decoding.

    Returns:
        A tuple containing:
        - An AlignmentModel object
        - The index of the best model selected based on the model criterion
    """
    if isinstance(data, SequenceDataset):
        data = (data,)

    # If the input/output config is not set, we use the dataset path
    if config.input_output.input_file == Path():
        config.input_output.input_file = data[0].filepath

    # Create working directory if it does not exist
    work_dir = Path(config.input_output.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Create a context that automatically sets up data-dependent parameters
    context = LearnMSAContext(config, data[0])

    # Write the config to file in the working directory
    config_path = work_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config.model_dump(mode='json'), f, indent=2)
    if config.input_output.verbose:
        print(f"Configuration saved to {config_path}")

    # Either load a model from file or train a new model
    if config.input_output.load_model and \
            config.training.skip_training:
        # Load a model without any training
        am = AlignmentModel.load(config.input_output.load_model, data)
        # Override indices
        am.indices = np.arange(data[0].num_seq)
        am.hit_alignment_mode = HitAlignmentMode.from_str(
            config.training.hit_alignment_mode
        )
    else:
        if config.input_output.verbose:
            print(
                f"Created {config.training.num_model} models from file "\
                f"{Path(config.input_output.input_file).name}"
            )
        # Train a new model
        config.hmm.use_noise = config.training.use_noise
        try:
            t_a = time.time()
            if config.visualization.logo_gif:
                am = _fit_and_align_with_logo_gif(data, context)
            else:
                am = _fit_and_align(data, context)
            if config.input_output.verbose:
                print("Time for alignment:", "%.4f" % (time.time()-t_a))

            # Select the head that best fits the training data
            # according to the criterion specified in the config
            am.select_best()

        except tf.errors.ResourceExhaustedError as e:
            print("Out of memory. A resource was exhausted.")
            runtime_batch_size = context.last_runtime_batch_size
            if runtime_batch_size is None:
                runtime_batch_size = config.training.batch_size
            print(
                "Try reducing the batch size (-b). The current batch size "\
                "was: "+str(runtime_batch_size)+"."
            )
            sys.exit(e.error_code)

    tf.keras.backend.clear_session()

    return am


def _transfer_model_weights(
    dst: LearnMSAModel,
    src: LearnMSAModel,
) -> None:
    """Transfer learned parameters from a loaded model to a freshly built one.

    The pHMM layer weights are copied in full (both models must share the same
    architecture, i.e. the same model lengths).  For ``anc_probs_layer`` the
    rate-matrix parameters (exchangeability and equilibrium kernels) are
    preserved, while the per-sequence evolutionary-distance kernel
    (``tau_kernel``) is only copied when both models have the same shape for
    that variable – otherwise it is left at the new model's initialised values.

    Args:
        dst: Freshly built target model.
        src: Previously loaded source model whose weights should be copied.
    """
    dst.phmm_layer.set_weights(src.phmm_layer.get_weights())
    if hasattr(src, 'anc_probs_layer') and hasattr(dst, 'anc_probs_layer'):
        dst.anc_probs_layer.exchangeability_kernel.assign(
            src.anc_probs_layer.exchangeability_kernel
        )
        dst.anc_probs_layer.equilibrium_kernel.assign(
            src.anc_probs_layer.equilibrium_kernel
        )
        # tau_kernel is per-sequence; only copy when shapes match (same dataset)
        if (src.anc_probs_layer.tau_kernel.shape
                == dst.anc_probs_layer.tau_kernel.shape):
            dst.anc_probs_layer.tau_kernel.assign(
                src.anc_probs_layer.tau_kernel
            )


def _fit_and_align(
    data : SequenceDataset | tuple[SequenceDataset, *tuple[Dataset, ...]],
    context : LearnMSAContext
) -> AlignmentModel:
    """ Utility method that trains a LearnMSAModel and creates an
    AlignmentModel from it.

    Args:
        data: SequenceDataset or tuple of Dataset(s) with the first dataset
                being a SequenceDataset (of amino acid sequences).
        context: LearnMSAContext object containing the configuration and other
            context information for training and decoding.

    Returns:
        An AlignmentModel object.
    """
    if isinstance(data, SequenceDataset):
        data = (data,)

    config = context.config
    if config.input_output.verbose and not config.training.skip_training:
        _dataset_messages(data[0])

    # Roughly estimate the full length of a protein
    full_length_estimate = training_util.get_full_length_estimate(
        data[0].seq_lens,
        config.training.surgery_quantile,
        config.training.min_surgery_seqs
    )

    if config.input_output.load_model:
        # Load the alignment model from file and use it as initialization
        am = AlignmentModel.load(config.input_output.load_model, data[0])
        if config.input_output.verbose:
            print("Loaded model from file", config.input_output.load_model)

        # Make the context use the correct model lengths
        context.model_lengths = am.model.lengths
        # Preserve the use_anc_probs setting from the loaded model so that
        # the new model's architecture matches.
        context.config.training.use_anc_probs = (
            am.model.context.config.training.use_anc_probs
        )
        # Keep a reference to the loaded model for weight transfer later.
        # We do NOT reuse am.model directly: its anc_probs_layer.tau_kernel
        # has shape [old_num_seq, heads, tracks] which is incompatible with
        # the new dataset size, and Keras forbids replacing sub-layers on an
        # already-built model. Instead we build a fresh LearnMSAModel with the
        # new context inside the training loop (where batch_size is known) and
        # copy the relevant weights across.
        loaded_model = am.model
    else:
        loaded_model = None
    model = None

    # 2 staged main loop: Fits model parameters with GD and optimized model
    # architecture with surgery
    last_iteration = config.training.max_iterations == 1
    for i in range(config.training.max_iterations):
        if callable(context.batch_size):
            batch_size = context.batch_size(data[0])
        else:
            batch_size = context.batch_size
        # Set the batch size to something smaller than the dataset size even
        # though or low sequence numbers it would be feasible to train on all
        # data at once
        batch_size = min(
            batch_size, training_util.get_low_seq_num_batch_size(data[0].num_seq)
        )
        if last_iteration:
            train_indices = np.arange(data[0].num_seq)
            decode_indices = context.subset
        else:
            train_indices = full_length_estimate
            decode_indices = full_length_estimate

        # Create and compile the model
        context.effective_num_seq = train_indices.shape[0] #todo: workaround
        if model is None:
            model = LearnMSAModel(context)
            model.build(((batch_size,),))
            if loaded_model is not None:
                _transfer_model_weights(model, loaded_model)
                # Release the loaded model; weights must not be re-applied
                # after surgery (model lengths change).
                loaded_model = None

        _pre_training_checkpoint(config, model, data, train_indices, i)

        # Run training
        model.fit(
            data, indices=train_indices, iteration=i, batch_size=batch_size
        )

        am = AlignmentModel(
            data, model, decode_indices,
            hit_alignment_mode = HitAlignmentMode.from_str(
                config.training.hit_alignment_mode
            ),
        )

        if config.input_output.verbose:
            print("Created alignment model successfully.")

        if last_iteration:
            break

        if config.training.surgery_checkpoints:
            # Save model checkpoint after surgery
            surgery_checkpoint_path = (
                Path(config.input_output.work_dir) /
                f"surgery_checkpoint_iter_{i+1}.model"
            )
            am.save(surgery_checkpoint_path)
            if config.input_output.verbose:
                print(
                    f"Saved surgery checkpoint to {surgery_checkpoint_path}."
                )

        surgery_result = model_surgery(
            am.model,
            data,
            indices=train_indices,
            surgery_del = config.training.surgery_del,
            surgery_ins = config.training.surgery_ins,
            verbose = config.input_output.verbose,
        )

        context.model_lengths = surgery_result.model_lengths
        context.config.hmm = surgery_result.config
        context.init_msa_values = None # don't use init MSA after surgery
        if surgery_result.plm_config is not None:
            context.config.language_model = surgery_result.plm_config
        if surgery_result.structural_config is not None:
            context.config.structure = surgery_result.structural_config
        surgery_converged = surgery_result.surgery_converged

        if config.input_output.verbose:
            print("Re-initialized the encoder parameters.")
            if surgery_converged:
                print("Surgery converged.")

        last_iteration = surgery_converged\
            or (i == config.training.max_iterations-2)

        # Free compiled graphs and cached memory
        del model
        model = None
        tf.keras.backend.clear_session()

    return am


def _fit_and_align_with_logo_gif(
    data : SequenceDataset | tuple[SequenceDataset, *tuple[Dataset, ...]],
    context : LearnMSAContext
) -> AlignmentModel:
    from learnMSA.util.visualize import LogoPlotterCallback, make_logo_gif
    config = context.config
    indices = np.arange(data[0].num_seq)
    if config.visualization.logo_gif:
        logo_dir = config.visualization.logo_gif.parent
    else:
        logo_dir = ""
    if callable(context.batch_size):
        batch_size = context.batch_size(
            context.model_lengths, # type: ignore
            min(data[0].max_len, config.training.crop) # type: ignore
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
    model.build(((batch_size,),))
    model.compile()

    # Run training
    model.fit(
        data,
        indices=indices,
        iteration=0,
        batch_size=batch_size,
        callbacks=[logo_plotter_callback],
    )

    make_logo_gif(logo_plotter_callback.frame_dir, logo_dir / "training.gif") # type: ignore
    am = AlignmentModel(data, model, indices)
    return am


def _dataset_messages(
    data : SequenceDataset,
    seq_count_heuristic_gap_check=100,
    seq_count_warning_threshold=100,
):
    # a quick heuristic check of the first sequences to see if they contain gaps
    warned = False
    for i in range(min(data.num_seq, seq_count_heuristic_gap_check)):
        record = data.get_record(i)
        if '-' in record or '.' in record:
            if not warned:
                print(
                    f"Warning: The sequences in {data.filepath} seem to be " +
                    "already aligned. learnMSA will ignore any gap character."
                )
                warned = True
    if data.num_seq < seq_count_warning_threshold:
        print(
            f"Warning: You are aligning {data.num_seq} sequences, although " +
            "learnMSA is designed for large scale alignments. We recommend " +
            "to have a sufficiently deep training dataset of at least " +
            f"{seq_count_warning_threshold} sequences for accurate results."
        )

def _pre_training_checkpoint(
    config: Configuration,
    model: LearnMSAModel,
    data: SequenceDataset | tuple[SequenceDataset, *tuple[Dataset, ...]],
    indices: np.ndarray,
    iteration: int,
) -> None:
    if not config.training.pre_training_checkpoint:
        return
    alignment_model = AlignmentModel(data, model, indices)
    checkpoint_path = (
        Path(model.context.config.input_output.work_dir) /
        f"pre_training_checkpoint_iter_{iteration}.model"
    )
    alignment_model.save(checkpoint_path)
    if config.input_output.verbose:
        print(
            f"Saved pre-training checkpoint to {checkpoint_path}."
        )
