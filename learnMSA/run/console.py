import argparse
import math
import os

import numpy as np

import learnMSA.run.util as util
from learnMSA.run.args import parse_args

#hide tensorflow messages and warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def run_main():

    version = util.get_version()
    parser = parse_args(version)
    args = parser.parse_args()

    if not args.silent:
        print(parser.description)

    util.setup_devices(args.cuda_visible_devices, args.silent)

    from ..msa_hmm import Align, Training, Visualize, MSA2HMM, Initializers, Priors
    from ..msa_hmm.SequenceDataset import SequenceDataset, AlignedDataset

    if args.logo or args.logo_gif:
        os.makedirs(args.logo_path, exist_ok=True)
        if args.logo_gif:
            os.makedirs(args.logo_path+"/frames/", exist_ok=True)

    if args.from_msa is not None:
        # Load priors to get pseudocounts
        aa_prior = Priors.AminoAcidPrior()
        aa_prior.build()
        aa_psc = aa_prior.emission_dirichlet_mix.make_alpha()[0].numpy()
        # Add counts for special amino acids
        aa_psc = np.pad(aa_psc, (0, 3), constant_values=1)
        transition_prior = Priors.ProfileHMMTransitionPrior()
        transition_prior.build()
        match_psc = transition_prior.match_dirichlet.make_alpha()[0].numpy()
        ins_psc = transition_prior.insert_dirichlet.make_alpha()[0].numpy()
        del_psc = transition_prior.delete_dirichlet.make_alpha()[0].numpy()
        del aa_prior
        del transition_prior

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
                begin_to_match=1,
                match_to_end=1,
                left_flank=ins_psc,
                right_flank=ins_psc,
                unannotated=ins_psc,
                end=1,
                flank_start=ins_psc,
            ).normalize().log()
        initializers = Initializers.make_initializers_from(
            values,
            num_models=args.num_model,
            # Apply random noise only when using multiple models
            random_scale=args.random_scale if args.num_model > 1 else 0.0,
        )
        initial_model_length_cb = lambda data, config: \
                                    [values.matches()]*args.num_model
        if not args.silent:
            print(f"Initialized from MSA '{args.from_msa}' with "
                  f"{values.matches()} match states.")
    else:
        initializers = None
        initial_model_length_cb = None

    try:
        with SequenceDataset(
            args.input_file, "fasta", indexed=args.indexed_data
        ) as data:
            # Check if the input data is valid
            data.validate_dataset()

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

            if args.from_msa is not None:
                config["epochs"] = [3]
                config["learning_rate"] = 1e-2
                config["max_surgery_runs"] = 1

            # Run a training to align the sequences
            alignment_model = Align.run_learnMSA(
                data,
                out_filename = args.output_file,
                config = config,
                model_generator=model_gen,
                batch_generator=batch_gen,
                align_insertions=not args.unaligned_insertions,
                sequence_weights = sequence_weights,
                clusters = clusters,
                verbose = not args.silent,
                logo_gif_mode = args.logo_gif,
                logo_dir = args.logo_path,
                initial_model_length_callback = initial_model_length_cb,
                output_format = args.format,
                load_model = args.load_model,
                A2M_output=not args.noA2M
            )
            if args.save_model:
                alignment_model.write_models_to_file(args.save_model)
            if args.logo:
                Visualize.plot_and_save_logo(
                    alignment_model,
                    alignment_model.best_model,
                    args.logo_path + "/logo.pdf"
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


def get_config(
    args : argparse.Namespace,
    data : "SequenceDataset",
    initializers : "PHMMInitializerSet | None" = None,
) -> dict:

    from ..msa_hmm import Configuration, Initializers
    from ..msa_hmm.AncProbsLayer import inverse_softplus

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

    config = Configuration.make_default(
        1 if args.logo_gif else args.num_model,
        use_language_model=args.use_language_model,
        scoring_model_config=scoring_model_config,
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

    if args.batch_size > 0:
        config["batch_size"] = args.batch_size
    config["num_models"] = 1 if args.logo_gif else args.num_model
    config["max_surgery_runs"] = args.max_surgery_runs
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

    return config

def get_generators(
    args : argparse.Namespace,
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
            config["scoring_model_config"].dim)
        batch_gen = EmbeddingBatchGenerator.EmbeddingBatchGenerator(
            EmbeddingBatchGenerator.generic_compute_embedding_func(
                data, config["scoring_model_config"]
            )
        )
    else:
        model_gen = None
        batch_gen = None
    return model_gen, batch_gen

def get_clustering(
    args : argparse.Namespace,
    config : dict
):
    from ..msa_hmm import Align
    if not args.no_sequence_weights:
        os.makedirs(args.cluster_dir, exist_ok=True)
        try:
            sequence_weights, clusters = Align.compute_sequence_weights(
                args.input_file,
                args.cluster_dir,
                config["cluster_seq_id"],
                return_clusters=True
            )
        except Exception as e:
            print("Error while computing sequence weights.")
            raise SystemExit(e)
    else:
        sequence_weights, clusters = None, None
    return sequence_weights, clusters


if __name__ == '__main__':
    run_main()
