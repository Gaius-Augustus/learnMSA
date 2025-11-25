import math
import os
import sys
from argparse import Namespace
from functools import partial

import numpy as np

import learnMSA.run.util as util
from learnMSA.run.args import parse_args
from learnMSA.run.help import handle_help_command

#hide tensorflow messages and warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def run_main():


    version = util.get_version()
    if handle_help_command():
        return
    parser = parse_args(version)
    args = parser.parse_args()

    if not args.silent and parser.description:
        print(parser.description.split("\n")[0])

    # Validate that either output_file or scores is provided
    if args.output_file is None and not args.scores:
        parser.error("Either -o/--out_file or --scores must be provided.")

    util.setup_devices(args.cuda_visible_devices, args.silent, args.grow_mem)

    if args.convert:
        convert_file(args)
        return

    from ..msa_hmm import (MSA2HMM, Align, Configuration, Initializers, Priors,
                           Training, Visualize)
    from ..msa_hmm.SequenceDataset import AlignedDataset, SequenceDataset

    if args.logo:
        args.logo = util.validate_filepath(args.logo, ".pdf")
        os.makedirs(args.logo.parent, exist_ok=True)
    if args.logo_gif:
        args.logo_gif = util.validate_filepath(args.logo_gif, ".gif")
        os.makedirs(args.logo_gif.parent, exist_ok=True)
        os.makedirs(args.logo_gif.parent / "frames", exist_ok=True)

    if args.from_msa is not None:

        if args.pseudocounts:
            # Infer meaningful pseudocounts from Dirichlet priors
            aa_prior = Priors.AminoAcidPrior()
            aa_prior.build()
            aa_psc = aa_prior.emission_dirichlet_mix.make_alpha()[0].numpy()
            # Add counts for special amino acids
            aa_psc = np.pad(aa_psc, (0, 3), constant_values=1e-2)
            transition_prior = Priors.ProfileHMMTransitionPrior()
            transition_prior.build()
            match_psc = transition_prior.match_dirichlet.make_alpha()[0].numpy()
            ins_psc = transition_prior.insert_dirichlet.make_alpha()[0].numpy()
            del_psc = transition_prior.delete_dirichlet.make_alpha()[0].numpy()
            del aa_prior
            del transition_prior
        else:
            # Use very small pseudocounts to avoid zero probabilities
            aa_psc = 1e-2
            match_psc = 1e-2
            ins_psc = 1e-2
            del_psc = 1e-2

        # Load the MSA and count
        with AlignedDataset(args.from_msa, "fasta") as input_msa:
            values = MSA2HMM.PHMMValueSet.from_msa(
                input_msa,
                match_threshold=args.match_threshold,
                global_factor=args.global_factor,
            ).add_pseudocounts(
                aa=aa_psc,
                match_transition=match_psc,
                insert_transition=ins_psc,
                delete_transition=del_psc,
                begin_to_match=1e-2,
                begin_to_delete=1e-8,
                match_to_end=1e-2,
                left_flank=ins_psc,
                right_flank=ins_psc,
                unannotated=ins_psc,
                end=1e-2,
                flank_start=ins_psc,
            ).normalize().log()

            if args.from_msa is not None:
                scoring_model_config = get_scoring_model_config(args)
                if args.use_language_model:
                    from learnMSA.protein_language_models.MvnEmitter import \
                        AminoAcidPlusMvnEmissionInitializer
                    dim = len(input_msa.alphabet)-1 + 2 * scoring_model_config.dim
                    emb_kernel = AminoAcidPlusMvnEmissionInitializer(
                        scoring_model_config
                    )((1,1,1,dim)).numpy().squeeze()
                    emb_kernel = emb_kernel[len(input_msa.alphabet)-1:]
                else:
                    emb_kernel = None
            initializers = Initializers.make_initializers_from(
                values,
                num_models=args.num_model,
                # Apply random noise only when using multiple models
                random_scale=args.random_scale if args.num_model > 1 else 0.0,
                emission_kernel_extra=emb_kernel,
            )
            initial_model_length_cb = lambda data, config: \
                                        [values.matches()]*args.num_model
            if not args.silent:
                print(f"Initialized from MSA '{args.from_msa}' with "
                    f"{values.matches()} match states.")
    else:
        initializers = None
        initial_model_length_cb = None

    if args.noA2M:
        raise DeprecationWarning(
            "--noA2M is deprecated. Use --format fasta instead."
        )

    try:
        with SequenceDataset(
            args.input_file, args.input_format, indexed=args.indexed_data
        ) as data:
            # Check if the input data is valid
            data.validate_dataset()

            # Handle length_init: if provided, update num_model and set custom callback
            if args.length_init is not None:
                # Ensure all lengths are at least 3
                args.length_init = [max(3, length) for length in args.length_init]
                # Update num_model to match the number of specified lengths
                args.num_model = len(args.length_init)
                # Create callback to return the specified lengths
                specified_lengths = args.length_init.copy()
                if initial_model_length_cb is None:
                    initial_model_length_cb = lambda data, config: specified_lengths
                if not args.silent:
                    print(
                        "Using user-specified initial model lengths: "\
                        f"{args.length_init}"
                    )

            # Merge parsed arguments into a learnMSA configuration
            config = get_config(args, data, initializers)
            model_gen, batch_gen = get_generators(args, data, config)
            sequence_weights, clusters = get_clustering(args, config)

            # Set up default initial model length when not using warmup
            if args.logo_gif:
                def get_initial_model_lengths_gif(
                        data: SequenceDataset,
                        config
                    ):
                    # initial model length
                    model_length = np.quantile(
                        data.seq_lens, q=config["length_init_quantile"])
                    model_length = max(3, int(model_length))
                    return [model_length]
                if initial_model_length_cb is None:
                    initial_model_length_cb = get_initial_model_lengths_gif
            elif initial_model_length_cb is None:
                initial_model_length_cb = Align.get_initial_model_lengths

            # Run a training to align the sequences
            alignment_model = Align.run_learnMSA(
                data,
                out_filename = args.output_file,
                config = config,
                model_generator=model_gen,
                batch_generator=batch_gen,
                align_insertions=not args.unaligned_insertions if args.output_file else False,
                sequence_weights = sequence_weights,
                clusters = clusters,
                verbose = not args.silent,
                logo_gif_mode = bool(args.logo_gif),
                logo_dir = args.logo_gif.parent if args.logo_gif else "",
                initial_model_length_callback = initial_model_length_cb,
                output_format = args.format,
                load_model = args.load_model,
                only_matches = args.only_matches,
            )
            if args.save_model:
                alignment_model.write_models_to_file(args.save_model)
            if args.scores:
                alignment_model.write_scores(args.scores)
                if not args.silent:
                    print(f"Wrote sequence scores to {args.scores}.")
            if args.logo:
                Visualize.plot_and_save_logo(
                    alignment_model,
                    alignment_model.best_model,
                    args.logo,
                )
            if args.dist_out:
                i = [l.name for l in alignment_model.encoder_model.layers].index(
                    "anc_probs_layer")
                anc_probs_layer = alignment_model.encoder_model.layers[i]
                tau_all = []
                for (seq, indices), _ in Training.make_dataset(
                    alignment_model.indices,
                    alignment_model.batch_generator,
                    alignment_model.batch_size,
                    shuffle=False
                ):
                    seq = tf.transpose(seq, [1, 0, 2])
                    indices = tf.transpose(indices)
                    # resolves tf 2.12 issues
                    indices.set_shape([alignment_model.num_models, None])
                    indices = tf.expand_dims(indices, axis=-1)
                    tau = anc_probs_layer.make_tau(seq, indices)[
                        alignment_model.best_model]
                    tau_all.append(tau.numpy())
                tau = np.concatenate(tau_all)
                with open(args.dist_out, "w") as file:
                    for i, t in zip(alignment_model.data.seq_ids, tau):
                        file.write(f"{i}\t{t}\n")
    except ValueError as e:
        raise SystemExit(e)


def get_scoring_model_config(args : Namespace) -> dict:
    if args.use_language_model:
        import learnMSA.protein_language_models.Common as Common
        scoring_model_config = Common.ScoringModelConfig(
            lm_name=args.language_model,
            dim=args.scoring_model_dim,
            activation=args.scoring_model_activation,
            suffix=args.scoring_model_suffix,
            scaled=False
        )
    else:
        scoring_model_config = None
    return scoring_model_config


def get_config(
    args : Namespace,
    data : "SequenceDataset",
    initializers : "PHMMInitializerSet | None" = None,
) -> dict:

    from ..msa_hmm import Configuration, Initializers
    from ..msa_hmm.AncProbsLayer import inverse_softplus

    config = Configuration.make_default(
        1 if args.logo_gif else args.num_model,
        use_language_model=args.use_language_model,
        scoring_model_config=get_scoring_model_config(args),
        use_l2=args.use_L2,
        L2_match=args.L2_match,
        L2_insert=args.L2_insert,
        num_prior_components=args.embedding_prior_components,
        frozen_insertions=args.frozen_insertions or args.use_language_model,
        temperature_mode=args.temperature_mode,
        V2_emitter=True,
        V2_temperature=args.temperature,
        inv_gamma_alpha=args.inverse_gamma_alpha,
        inv_gamma_beta=args.inverse_gamma_beta,
        plm_cache_dir=args.plm_cache_dir,
        emission_init=initializers.match_emissions if initializers else None,
        insertion_init=initializers.insert_emissions if initializers else None,
        transition_init=initializers.transitions if initializers else None,
        flank_init=initializers.start if initializers else None,
    )

    if args.tokens_per_batch > 0:
        config["batch_size"] = partial(
            Configuration.tokens_per_batch_to_batch_size,
            tokens_per_batch=args.tokens_per_batch
        )
    elif args.batch_size > 0:
        config["batch_size"] = args.batch_size
    config["num_models"] = 1 if args.logo_gif else args.num_model
    config["max_surgery_runs"] = args.max_iterations
    config["length_init_quantile"] = args.length_init_quantile
    config["surgery_quantile"] = args.surgery_quantile
    config["min_surgery_seqs"] = args.min_surgery_seqs
    config["len_mul"] = args.len_mul
    if not args.use_language_model:
        config["learning_rate"] = args.learning_rate
        config["epochs"] = args.epochs
    config["surgery_del"] = args.surgery_del
    config["surgery_ins"] = args.surgery_ins
    config["model_criterion"] = args.model_criterion
    config["trainable_distances"] = not args.frozen_distances
    if args.trainable_rate_matrices:
        config["trainable_rate_matrices"] = True
        config["equilibrium_sample"] = True
        config["transposed"] = True
    assert args.initial_distance >= 0, "The evolutionary distance must be >= 0."
    config["encoder_initializer"][0] = Initializers.ConstantInitializer(
        inverse_softplus(np.array(args.initial_distance) + 1e-8).numpy())
    transitioners = config["transitioner"] if hasattr(
        config["transitioner"], '__iter__') else [config["transitioner"]]
    for trans in transitioners:
        trans.prior.alpha_flank = args.alpha_flank
        trans.prior.alpha_single = args.alpha_single
        trans.prior.alpha_global = args.alpha_global
        trans.prior.alpha_flank_compl = args.alpha_flank_compl
        trans.prior.alpha_single_compl = args.alpha_single_compl
        trans.prior.alpha_global_compl = args.alpha_global_compl

    if args.crop == "disable":
        config["crop_long_seqs"] = math.inf
    elif args.crop == "auto":
        config["crop_long_seqs"] = int(
            np.ceil(args.auto_crop_scale * np.mean(data.seq_lens))
        )
    else:
        config["crop_long_seqs"] = int(args.crop)

    if args.skip_training:
        config["max_surgery_runs"] = 1
        config["epochs"] = [0]*3

    return config

def get_generators(
    args : Namespace,
    data : "SequenceDataset",
    config : dict
):
    if args.use_language_model:

        import learnMSA.protein_language_models.Common as Common
        import learnMSA.protein_language_models.EmbeddingBatchGenerator as EmbeddingBatchGenerator

        # we have to define a special model- and batch generator if using a
        # language model because the emission probabilities are computed
        # differently and the LM requires specific inputs
        model_gen = EmbeddingBatchGenerator.make_generic_embedding_model_generator(
            config["scoring_model_config"].dim
        )
        batch_gen = EmbeddingBatchGenerator.EmbeddingBatchGenerator(
            scoring_model_config = config["scoring_model_config"],
        )
    else:
        model_gen = None
        batch_gen = None
    return model_gen, batch_gen

def get_clustering(
    args : Namespace,
    config : dict
):
    from ..msa_hmm import Align, SequenceDataset
    if not args.no_sequence_weights:
        os.makedirs(args.work_dir, exist_ok=True)
        try:
            if args.input_format == "fasta":
                cluster_file = args.input_file
            else:
                # We need to convert to fasta
                cluster_file = os.path.join(
                    args.work_dir,
                    os.path.basename(args.input_file) + ".temp_for_clustering"
                )
                with SequenceDataset.SequenceDataset(
                    args.input_file, args.input_format
                ) as data:
                    data.write(cluster_file, "fasta")
            sequence_weights, clusters = Align.compute_sequence_weights(
                cluster_file,
                args.work_dir,
                config["cluster_seq_id"],
                return_clusters=True
            )
        except Exception as e:
            print("Error while computing sequence weights.")
            raise SystemExit(e)
    else:
        sequence_weights, clusters = None, None
    return sequence_weights, clusters

def convert_file(args : Namespace) -> None:
    from ..msa_hmm.SequenceDataset import SequenceDataset

    if args.format == "a2m":
        raise ValueError("Cannot convert to a2m format without a model.")

    with SequenceDataset(args.input_file, args.input_format) as data:
        data.write(
            args.output_file,
            args.format,
            standardize_sequences=args.format == "fasta",
        )
    if not args.silent:
        print(
            f"Converted {args.input_file} to {args.output_file} in format "\
            f"{args.format}."
        )


if __name__ == '__main__':
    run_main()
