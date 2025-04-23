import os
import numpy as np
import sys
import math
import pandas as pd
from learnMSA.run.args import parse_args

# hide tensorflow messages and warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def run_main():
    
    version = get_version()
    args = parse_args(version)

    if not args.silent:
        print(
            f"learnMSA (version {version}) - " \
            "multiple alignment of protein sequences")

    # import after argparsing to avoid long delay with -h option and to allow the user to change CUDA settings
    # before importing tensorflow
    if not args.cuda_visible_devices == "default":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    from ..msa_hmm import Align, Training, Visualize
    from ..msa_hmm.SequenceDataset import SequenceDataset
    import tensorflow as tf
    from tensorflow.python.client import device_lib

    if not args.silent:
        GPUS = [x.physical_device_desc for x in device_lib.list_local_devices()
                if x.device_type == 'GPU']
        if len(GPUS) == 0:
            if args.cuda_visible_devices == "-1" or args.cuda_visible_devices == "":
                print(
                    "GPUs disabled by user. Running on CPU instead. "\
                    "Expect slower performance especially for longer models.")
            else:
                print("It seems like no GPU is installed. Running on CPU "\
                      "instead. Expect slower performance especially for "\
                      "longer models.")
        else:
            print("Using GPU(s):", GPUS)

        print("Found tensorflow version", tf.__version__)

    if args.logo or args.logo_gif:
        os.makedirs(args.logo_path, exist_ok=True)
        if args.logo_gif:
            os.makedirs(args.logo_path+"/frames/", exist_ok=True)
    try:
        with SequenceDataset(args.input_file, "fasta", indexed=args.indexed_data) as data:
            
            # check if the input data is valid
            data.validate_dataset()

            config = get_config(args, data)
            model_gen, batch_gen = get_generators(args, data, config)
            sequence_weights, clusters = get_clustering(args, config)
            
            # in tree mode, we'll do a warmup run training a joint profile for
            # all sequences without using a tree at first
            if args.tree and args.warmup_epochs > 0:
                warmup_alignment_model = train_warmup_model_and_update_config(
                    args, data, config
                )
                warmup_emitter = warmup_alignment_model.msa_hmm_layer.cell.emitter[0]
                initial_model_length_cb = lambda data, config : warmup_emitter.lengths
            else:
                # set up default initial model length when not using warmup
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
                    initial_model_length_cb = get_initial_model_lengths_gif
                else:
                    initial_model_length_cb = Align.get_initial_model_lengths

            alignment_model = Align.run_learnMSA(
                data,
                out_filename=args.output_file,
                config=config,
                model_generator=model_gen,
                batch_generator=batch_gen,
                align_insertions=not args.unaligned_insertions,
                sequence_weights=sequence_weights,
                clusters=clusters,
                verbose=not args.silent,
                logo_gif_mode=args.logo_gif,
                logo_dir=args.logo_path,
                initial_model_length_callback=initial_model_length_cb,
                output_format=args.format,
                load_model=args.load_model,
                A2M_output=not args.noA2M
            )
            if args.save_model:
                alignment_model.write_models_to_file(args.save_model)
            if args.logo:
                Visualize.plot_and_save_logo(
                    alignment_model, alignment_model.best_model, args.logo_path + "/logo.pdf")
            if args.dist_out:
                i = [l.name for l in alignment_model.encoder_model.layers].index(
                    "anc_probs_layer")
                anc_probs_layer = alignment_model.encoder_model.layers[i]
                tau_all = []
                for (seq, indices), _ in Training.make_dataset(alignment_model.indices, alignment_model.batch_generator, alignment_model.batch_size, shuffle=False):
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


def get_version():
    # get the version without importing learnMSA as a module
    base_dir = str(os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')))
    version_file_path = base_dir + "/_version.py"
    with open(version_file_path, "rt") as version_file:
        version = version_file.readlines()[0].split("=")[1].strip(' "')
    return version


def get_config(args, data):

    from ..msa_hmm import Configuration, Initializers, Clustering
    from ..msa_hmm.AncProbsLayer import inverse_softplus

    if args.use_language_model:
        import learnMSA.protein_language_models.Common as Common
        scoring_model_config = Common.ScoringModelConfig(lm_name=args.language_model,
                                                            dim=args.scoring_model_dim,
                                                            activation=args.scoring_model_activation,
                                                            suffix=args.scoring_model_suffix,
                                                            scaled=False)
    else:
        scoring_model_config = None

    if args.tree:
        tree_handler = get_tree_handler(args)
        cluster_factor = (tree_handler.num_anc-1) / data.num_seq
    else:
        tree_handler = None
        cluster_factor = 1.0

    config = Configuration.make_default(1 if args.logo_gif else args.num_model,
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
                                        tree_handler=tree_handler,
                                        tree_loss_weight=cluster_factor,
                                        use_tree_transitioner=args.use_tree_transitioner)

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

    config["crop_long_seqs"] = get_crop_treshold(data, args)

    return config


def get_generators(args, data, config):
    if args.use_language_model:

        import learnMSA.protein_language_models.Common as Common
        import learnMSA.protein_language_models.EmbeddingBatchGenerator as EmbeddingBatchGenerator

        # we have to define a special model- and batch generator if using a language model
        # because the emission probabilities are computed differently and the LM requires specific inputs
        model_gen = EmbeddingBatchGenerator.make_generic_embedding_model_generator(
            config["scoring_model_config"].dim)
        batch_gen = EmbeddingBatchGenerator.EmbeddingBatchGenerator(
            EmbeddingBatchGenerator.generic_compute_embedding_func(data, config["scoring_model_config"]))
    else:
        model_gen = None
        batch_gen = None
    return model_gen, batch_gen


def get_clustering(args, config):
    from ..msa_hmm import Clustering
    if not args.no_sequence_weights:
        os.makedirs(args.cluster_dir, exist_ok=True)
        try:
            sequence_weights, clusters = Clustering.compute_sequence_weights(
                args.input_file, args.cluster_dir, config["cluster_seq_id"], return_clusters=True)
        except Exception as e:
            print("Error while computing sequence weights.")
            raise SystemExit(e)
    else:
        sequence_weights, clusters = None, None
    return sequence_weights, clusters


def get_tree_handler(args):
    from ..msa_hmm import Clustering
    os.makedirs(args.cluster_dir, exist_ok=True)
    clustering : pd.DataFrame = Clustering.compute_clustering(
        args.input_file, 
        directory=args.cluster_dir,
        cluster_seq_id=0.01, 
        linear=False
    )
    tree_handler = Clustering.cluster_tree(
        clustering, 
        branch_length=0.05
    )
    return tree_handler



def get_warmup_config(args, data):

    from ..msa_hmm import Configuration

    # warmup without tree 
    # this quickly learns an initial profile
    warmup_config = Configuration.make_default(args.num_model)
    # cropping
    warmup_config["crop_long_seqs"] = int(np.ceil(2 * np.mean(data.seq_lens))) 
    # short initial model, focus on important sites
    warmup_config["len_mul"] = args.len_mul
    # no surgery, only a few epochs of warmup
    warmup_config["max_surgery_runs"] = 1
    warmup_config["epochs"][0] = args.warmup_epochs

    return warmup_config


def train_warmup_model_and_update_config(args, data, config):

    from ..msa_hmm import Align, Initializers

    warmup_config = get_warmup_config(args, data)

    warmup_alignment_model = Align.fit_and_align(
        data, warmup_config, verbose=False
    )

    warmup_emitter = warmup_alignment_model.msa_hmm_layer.cell.emitter[0]
    warmup_transitioner = warmup_alignment_model.msa_hmm_layer.cell.transitioner

    # initialize with the warmup parameters
    config["emitter"].emission_init = []
    config["emitter"].insertion_init = []
    config["transitioner"].transition_init = []
    config["transitioner"].flank_init = []
    for i in range(args.num_model):
        config["emitter"].emission_init.append(
            Initializers.ConstantInitializer(
                warmup_emitter.emission_kernel[i].numpy()[np.newaxis,:,:]
            )
        )
        config["emitter"].insertion_init.append(
            Initializers.ConstantInitializer(
                warmup_emitter.insertion_kernel[i].numpy()[np.newaxis,:]
            )
        )
        trans_dict = {}
        config["transitioner"].transition_init.append(trans_dict)
        for key in warmup_transitioner.transition_kernel[0].keys():
            K = warmup_transitioner.transition_kernel[i][key].numpy()
            if args.use_tree_transitioner:
                K = K[np.newaxis,:]
            trans_dict[key] = Initializers.ConstantInitializer(K)
        F = warmup_transitioner.flank_init_kernel[i].numpy()
        if args.use_tree_transitioner:
            F = F[np.newaxis,:]
        config["transitioner"].flank_init.append(
            Initializers.ConstantInitializer(F)
        )

    config["emission_kernel_dummy"] = warmup_emitter.emission_init[0]
    config["transition_kernel_dummy"] = warmup_transitioner.transition_init[0]
    config["flank_init_kernel_dummy"] = warmup_transitioner.flank_init[0]

    return warmup_alignment_model
    


def get_crop_treshold(data, args):
    if args.crop == "disable":
        return math.inf
    elif args.crop == "auto":
        return int(np.ceil(args.auto_crop_scale * np.mean(data.seq_lens)))
    else:
        return int(args.crop)


if __name__ == '__main__':
    run_main()
