import os
from contextlib import ExitStack
from pathlib import Path

import learnMSA.run.util as util
from learnMSA import Configuration
from learnMSA.run import args_to_config, handle_help_command, parse_args

# Hide TensorFlow/absl startup logs as early as possible.
# Use defaults so users can still override through environment variables.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_CPP_MIN_VLOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")


def run_main() -> None:
    # Get version and parse arguments
    version = util.get_version()
    if handle_help_command():
        return
    parser = parse_args(version)
    pre_args, _ = parser.parse_known_args()
    if pre_args.config:
        util.apply_baseline_config_defaults(parser, pre_args.config)
    args = parser.parse_args()

    # Convert args to configuration
    config = args_to_config(args)

    # Validate that output_file is provided when required
    util.validate_output_file_requirements(config, parser)

    # Print brief description of the tool
    if config.input_output.verbose and parser.description:
        print(parser.description.split("\n")[0])

    # Create working directory if it does not exist
    work_dir = Path(config.input_output.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # When converting files, do not run the full program, but just convert
    if config.input_output.convert:
        convert_file(config)
        return

    # Set up devices (CPU/GPU)
    util.setup_devices(
        config.input_output.cuda_visible_devices,
        config.input_output.verbose,
    )

    from learnMSA.align.align import align
    from learnMSA.util import SequenceDataset, EmbeddingDataset

    with ExitStack() as stack:
        ## Load the amino acid dataset (mandatory)
        data = stack.enter_context(
            SequenceDataset(
                config.input_output.input_file,
                config.input_output.input_format,
                indexed=config.training.indexed_data,
            )
        )
        # rule out issues with the seq file early on
        data.validate_dataset()
        datasets = (data, )

        ## Load a structural dataset (optional)
        struct_data = util.load_struct_data(config, data, stack)

        ## Load embeddings dataset (optional)
        emb_data = util.load_emb_data(config, data, stack)

        # Compute embeddings if required and not provided as a file
        if (config.language_model.use_language_model
                or config.language_model.only_embeddings):
            if emb_data is None:
                from learnMSA.protein_language_models import compute_embeddings

                # Compute the embeddings if not provided as a file
                cache = compute_embeddings(
                    data,
                    config.language_model,
                    verbose=config.input_output.verbose,
                )
                emb_data = EmbeddingDataset(
                    embedding_cache=cache, seq_ids=data.seq_ids,
                )

                # Save the computed embeddings if a save path is provided
                if (config.input_output.save_emb
                        and config.input_output.save_emb != Path()):
                    emb_data.write(config.input_output.save_emb)
                    if config.input_output.verbose:
                        print(
                            f"Saved computed embeddings to " +
                            f"{config.input_output.save_emb}"
                        )
                    if config.language_model.only_embeddings:
                        if config.input_output.verbose:
                            print(
                                "Found --save-emb argument but not "
                                "--use-language-model. Exiting now without "
                                "running the full alignment."
                            )
                        return

            datasets += (emb_data, )

        # Run a training to align the sequences
        alignment_model, best_model = align(datasets, config)

        if config.input_output.save_model:
            alignment_model.save(config.input_output.save_model)
        if config.input_output.scores != Path():
            alignment_model.write_scores(
                Path(config.input_output.scores), best_model
            )
            if config.input_output.verbose:
                print(f"Wrote scores to {config.input_output.scores}")
        # TODO: fix
        # if args.logo:
        #     plot_and_save_logo(
        #         alignment_model,
        #         best_model,
        #         args.logo,
       #     )
        if args.dist_out:
            raise NotImplementedError(
                "Distribution output is not implemented in this version."
            )
            # i = [l.name for l in alignment_model.encoder_model.layers].index(
            #     "anc_probs_layer")
            # anc_probs_layer = alignment_model.encoder_model.layers[i]
            # tau_all = []
            # for (seq, indices), _ in Training.make_dataset(
            #     alignment_model.indices,
            #     alignment_model.batch_generator,
            #     alignment_model.batch_size,
            #     shuffle=False
            # ):
            #     seq = tf.transpose(seq, [1, 0, 2])
            #     indices = tf.transpose(indices)
            #     # resolves tf 2.12 issues
            #     indices.set_shape([alignment_model.num_models, None])
            #     indices = tf.expand_dims(indices, axis=-1)
            #     tau = anc_probs_layer.make_tau(seq, indices)[
            #         alignment_model.best_model]
            #     tau_all.append(tau.numpy())
            # tau = np.concatenate(tau_all)
            # with open(args.dist_out, "w") as file:
            #     for i, t in zip(alignment_model.data.seq_ids, tau):
            #         file.write(f"{i}\t{t}\n")


def convert_file(config : Configuration) -> None:
    from ..util.sequence_dataset import SequenceDataset

    if config.input_output.format == "a2m":
        raise ValueError("Cannot convert to a2m format.")

    with SequenceDataset(
        config.input_output.input_file, config.input_output.input_format
    ) as data:
        data.write(
            config.input_output.output_file,
            config.input_output.format,
            standardize_sequences=config.input_output.format == "fasta",
        )
    if config.input_output.verbose:
        print(
            f"Converted {config.input_output.input_file} to "\
            f"{config.input_output.output_file} in format "\
            f"{config.input_output.format}."
        )

if __name__ == '__main__':
    run_main()
