import os
import time
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
    default_config = Configuration()
    parser = parse_args(version, default_config)
    args = parser.parse_args()

    # Build configuration: CLI args first, then overlay --config file if given
    config = args_to_config(args, default_config)
    if args.config:
        config = util.merge_config_file(args.config, config, parser)

    # Resolve input file (may use --from_msa when -i is omitted)
    util.resolve_input_file(config, parser)

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
    from learnMSA.align.alignment_model import AlignmentModel
    from learnMSA.align.align_inserts import make_aligned_insertions
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
        # allow single seq fastas when computing scores (no MSA)
        data.validate_dataset(
            single_seq_ok = config.input_output.scores != Path()
        )
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

        if config.structure.use_structure:
            assert struct_data is not None,\
                "Structural data is required but not provided."
            datasets += (struct_data, )

        # Run a training to align the sequences
        am = align(datasets, config) # type:ignore

        if config.input_output.save_model:
            am.save(config.input_output.save_model)

        if config.input_output.output_file != Path():

            Path(config.input_output.output_file).parent.mkdir(
                parents=True, exist_ok=True
            )

            # build the alignment (= predict state sequences and create metadata)
            am.build_alignment(
                [am.best_head],
                decoding_mode=AlignmentModel.DecodingMode.from_str(
                    config.training.decoding_mode
                )
            )

            # measure time of the file generation
            t = time.time()
            if config.input_output.verbose:
                print("Generating output file...")

            assert am.best_head != -1,\
                "Best head was not selected. This should not happen."

            decoding_mode=AlignmentModel.DecodingMode.from_str(
                config.training.decoding_mode
            )
            if config.training.unaligned_insertions\
                    or config.training.only_matches:
                # Don't align insertions when requested or when only matches need to
                # be written to the output file
                am.to_file(
                    config.input_output.output_file,
                    am.best_head,
                    format=config.input_output.format,
                    only_matches=config.training.only_matches,
                    decoding_mode=decoding_mode,
                    add_block_sep=config.input_output.add_block_separator_to_msa,
                )
            else:
                aligned_insertions = make_aligned_insertions(
                    am,
                    am.best_head,
                    decoding_mode=decoding_mode,
                    method=config.advanced.insertion_aligner,
                    threads=config.advanced.aligner_threads,
                    verbose=config.input_output.verbose,
                )
                am.to_file(
                    config.input_output.output_file,
                    am.best_head,
                    aligned_insertions=aligned_insertions,
                    format=config.input_output.format,
                    decoding_mode=decoding_mode,
                    add_block_sep=config.input_output.add_block_separator_to_msa,
                )

        if config.input_output.verbose:
            if am.fixed_viterbi_seqs.size > 0:
                max_show_seqs = 5
                print(f"Fixed {am.fixed_viterbi_seqs.size} Viterbi sequences:")
                print("\n".join([
                    am.data[0].seq_ids[i]
                    for i in am.fixed_viterbi_seqs[:max_show_seqs]
                ]))
                if am.fixed_viterbi_seqs.size > max_show_seqs:
                    print("...")
            print("time for generating output:", "%.4f" % (time.time()-t))
            print("Wrote file", config.input_output.output_file)

        if config.input_output.scores != Path():
            am.write_scores(
                Path(config.input_output.scores), am.best_head
            )
            if config.input_output.verbose:
                print(f"Wrote scores to {config.input_output.scores}")

        if config.visualization.plot:
            from learnMSA.util.visualize import plot_phmm

            if config.visualization.plot_head == -1:
                head = am.best_head # type: ignore
            else:
                head = config.visualization.plot_head
            fig = plot_phmm(am.model.phmm_layer, head)
            fig.savefig(config.visualization.plot, bbox_inches="tight") # type: ignore
            if config.input_output.verbose:
                print(f"Saved HMM plot to {config.visualization.plot}")

        if config.advanced.dist_out:
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
