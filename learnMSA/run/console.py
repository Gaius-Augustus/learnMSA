import os
import numpy as np
import sys
import argparse
import math

#hide tensorflow messages and warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

def run_main():
    class MsaHmmArgumentParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write('error: %s\n' % message)
            self.print_help()
            sys.exit(2)
            
    #get the version without importing learnMSA as a module
    base_dir = str(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))
    version_file_path = base_dir + "/_version.py"
    with open(version_file_path, "rt") as version_file:
        version = version_file.readlines()[0].split("=")[1].strip(' "')

    parser = MsaHmmArgumentParser(description=f"learnMSA (version {version}) - multiple alignment of protein sequences")
    parser.add_argument("-i", "--in_file", dest="input_file", type=str, required=True,
                        help="Input sequence file in fasta format.")
    parser.add_argument("-o", "--out_file", dest="output_file", type=str, required=True,
                        help="Filepath for the output alignment.")
    parser.add_argument("-n", "--num_model", dest="num_model", type=int, default=4,
                        help="Number of models trained in parallel. (default: %(default)s)")
    parser.add_argument("-s", "--silent", dest="silent", action='store_true', help="Prevents output to stdout.")
    parser.add_argument("-b", "--batch", dest="batch_size", type=int, default=-1,
                        help="Should be lowered if memory issues with the default settings occur. Default: Adaptive, depending on model length (64-512).")
    parser.add_argument("-d", "--cuda_visible_devices", dest="cuda_visible_devices", type=str, default="default",
                        help="Controls the GPU devices visible to learnMSA as a comma-separated list of device IDs. The value -1 forces learnMSA to run on CPU. Per default, learnMSA attempts to use all available GPUs.")
    parser.add_argument("-f", "--format", dest="format", type=str, default="fasta", help="Output file format (see Biopython's SeqIO formats suitable for aligned data). Default: fasta (A2M).")
    parser.add_argument("--save_model", dest="save_model", type=str, default="", help="Optional filepath to store the trained model.")
    parser.add_argument("--load_model", dest="load_model", type=str, default="", help="A pretrained model can be loaded from a file, skipping the training process.")
    parser.add_argument("--noA2M", dest="noA2M", action='store_true', help="If set, the output will use only upper case letters and \"-\" for gaps. Otherwise, lower case letters are used for insertions and \".\" denotes an insertion in another sequence.")

    parser.add_argument("--max_surgery_runs", dest="max_surgery_runs", type=int, default=2, 
                        help="Maximum number of model surgery iterations. (default: %(default)s)")
    parser.add_argument("--length_init_quantile", dest="length_init_quantile", type=float, default=0.5, 
                        help="Quantile of the input sequence lengths that defines the initial model lengths. (default: %(default)s)")
    parser.add_argument("--surgery_quantile", dest="surgery_quantile", type=float, default=0.5, 
                        help="learnMSA will not use sequences shorter than this quantile for training during all iterations except the last. (default: %(default)s)")
    parser.add_argument("--min_surgery_seqs", dest="min_surgery_seqs", type=int, default=100000, 
                        help="Minimum number of sequences used per iteration. Overshadows the effect of --surgery_quantile. (default: %(default)s)")
    parser.add_argument("--len_mul", dest="len_mul", type=float, default=0.8, 
                        help="Multiplicative constant for the quantile used to define the initial model length (see --length_init_quantile). (default: %(default)s)")
    parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=0.1, 
                        help="The learning rate used during gradient descent. (default: %(default)s)")
    parser.add_argument("--epochs", dest="epochs", type=int, nargs=3, default=[10, 2, 10],
                       help="Scheme for the number of training epochs during the first, an intermediate and the last iteration (expects 3 integers in this order). (default: %(default)s)")
    parser.add_argument("--surgery_del", dest="surgery_del", type=float, default=0.5,
                       help="Will discard match states that are expected less often than this fraction. (default: %(default)s)")
    parser.add_argument("--surgery_ins", dest="surgery_ins", type=float, default=0.5,
                       help="Will expand insertions that are expected more often than this fraction. (default: %(default)s)")
    parser.add_argument("--model_criterion", dest="model_criterion", type=str, default="AIC",
                       help="Criterion for model selection. (default: %(default)s)")
    parser.add_argument("--indexed_data", dest="indexed_data", action='store_true', help="Don't load all data into memory at once at the cost of training time.")
    
    parser.add_argument("--unaligned_insertions", dest="unaligned_insertions", action='store_true', help="Insertions will be left unaligned.")
    parser.add_argument("--crop", dest="crop", type=str,  default="auto", help="""During training, sequences longer than the given value will be cropped randomly. 
    Reduces training runtime and memory usage, but might produce inaccurate results if too much of the sequences is cropped. The output alignment will not be cropped. 
    Can be set to auto in which case sequences longer than 3 times the average length are cropped. Can be set to disable. (default: %(default)s)""")
    parser.add_argument("--auto_crop_scale", dest="auto_crop_scale", type=float, default=2., 
                        help="During training sequences longer than this factor times the average length are cropped. (default: %(default)s)")
    parser.add_argument("--frozen_insertions", dest="frozen_insertions", action='store_true', help="Insertions will be frozen during training.")
    
    parser.add_argument("--no_sequence_weights", dest="no_sequence_weights", action='store_true', help="Do not use sequence weights and strip mmseqs2 from requirements. In general not recommended.")
    parser.add_argument("--cluster_dir", dest="cluster_dir", type=str, default="tmp", help="Directory where the sequence clustering is stored. (default: %(default)s)")
    
    parser.add_argument("--alpha_flank", dest="alpha_flank", type=float, default=7000, help=argparse.SUPPRESS)
    parser.add_argument("--alpha_single", dest="alpha_single", type=float, default=1e9, help=argparse.SUPPRESS)
    parser.add_argument("--alpha_global", dest="alpha_global", type=float, default=1e4, help=argparse.SUPPRESS)
    parser.add_argument("--alpha_flank_compl", dest="alpha_flank_compl", type=float, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--alpha_single_compl", dest="alpha_single_compl", type=float, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--alpha_global_compl", dest="alpha_global_compl", type=float, default=1, help=argparse.SUPPRESS)

    parser.add_argument("--inverse_gamma_alpha", dest="inverse_gamma_alpha", type=float, default=3., help=argparse.SUPPRESS)
    parser.add_argument("--inverse_gamma_beta", dest="inverse_gamma_beta", type=float, default=0.5, help=argparse.SUPPRESS)

    parser.add_argument("--frozen_distances", dest="frozen_distances", action='store_true', help="Prevents learning evolutionary distances for all sequences.")
    parser.add_argument("--initial_distance", dest="initial_distance", type=float, default=0.05, help="The initial evolutionary distance for all sequences. (default: %(default)s)")
    parser.add_argument("--trainable_rate_matrices", dest="trainable_rate_matrices", action='store_true', help="Prevents learning the amino acid rate matrices of the evolutionary model.")
    parser.add_argument("--dist_out", dest="dist_out", type=str, default="", help="Optional output file for the learned evolutionary distances.")

    parser.add_argument("--use_language_model", dest="use_language_model", action='store_true', help="Uses a large protein lanague model to generate per-token embeddings that guide the MSA step. (default: %(default)s)")
    parser.add_argument("--plm_cache_dir", dest="plm_cache_dir", type=str, default=None, help="Directory where the protein language model is stored. (default: learnMSA install dir)")
    parser.add_argument("--language_model", dest="language_model", type=str, default="protT5", help="Name of the language model to use. (default: %(default)s)")
    parser.add_argument("--scoring_model_dim", dest="scoring_model_dim", type=int, default=16, 
                        help="Reduced embedding dimension of the scoring model. (default: %(default)s)")
    parser.add_argument("--scoring_model_activation", dest="scoring_model_activation", type=str, default="sigmoid", 
                        help="Activation function of the scoring model. (default: %(default)s)")
    parser.add_argument("--scoring_model_suffix", dest="scoring_model_suffix", type=str, default="", 
                        help="Suffix to identify a specific scoring model. (default: %(default)s)")
    parser.add_argument("--temperature_mode", dest="temperature_mode", type=str, default="trainable", 
                        help="The annealing scheme used. Possible values are: [trainable, length_norm, warm_to_cold, cold_to_warm, constant, none]. (default: %(default)s)")
    parser.add_argument("--use_L2", dest="use_L2", action='store_true', help="Uses L2 regularization on the match and insertion state embeddings.")
    parser.add_argument("--L2_match", dest="L2_match", type=float, default=0.0, help="Strength of the L2 regularization on the match state embedding weights. (default: %(default)s)")
    parser.add_argument("--L2_insert", dest="L2_insert", type=float, default=1000.0, help="Strength of the L2 regularization on the insertion state embedding weights. (default: %(default)s)")
    parser.add_argument("--embedding_prior_components", dest="embedding_prior_components", type=int, default=32, help="Number of components of the multivariate normal prior distribution over the embedding weights. (default: %(default)s)")
    parser.add_argument("--temperature", dest="temperature", type=float, default=3., help="Temperature of the softmax function. (default: %(default)s)")
    
    parser.add_argument("--logo", dest="logo", action='store_true', help="Produces a gif that animates the learned sequence logo over training time.")
    parser.add_argument("--logo_gif", dest="logo_gif", action='store_true', help="Produces a gif that animates the learned sequence logo over training time. Slows down training significantly.")
    parser.add_argument("--logo_path", dest="logo_path", type=str, default="./logo/", help="Filepath used to store created logos and logo gifs. Directories are created. (default: %(default)s)")
    

    args = parser.parse_args()
    
    if not args.silent:
        print(f"learnMSA (version {version}) - multiple alignment of protein sequences")
    
    #import after argparsing to avoid long delay with -h option and to allow the user to change CUDA settings
    #before importing tensorflow
    if not args.cuda_visible_devices == "default":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices  
    from ..msa_hmm import Configuration, Initializers, Align, Emitter, Training, Visualize
    from ..msa_hmm.SequenceDataset import SequenceDataset
    from ..msa_hmm.AncProbsLayer import inverse_softplus
    import tensorflow as tf
    from tensorflow.python.client import device_lib
    
    if not args.silent:
        GPUS = [x.physical_device_desc for x in device_lib.list_local_devices() if x.device_type == 'GPU']
        if len(GPUS) == 0:
            if args.cuda_visible_devices == "-1" or args.cuda_visible_devices == "":
                print("GPUs disabled by user. Running on CPU instead. Expect slower performance especially for longer models.")
            else:
                print("It seems like no GPU is installed. Running on CPU instead. Expect slower performance especially for longer models.")
        else:
            print("Using GPU(s):", GPUS)

        print("Found tensorflow version", tf.__version__)
    
    import learnMSA.protein_language_models.Common as Common
    from learnMSA.protein_language_models.EmbeddingBatchGenerator import EmbeddingBatchGenerator, make_generic_embedding_model_generator
    scoring_model_config = Common.ScoringModelConfig(lm_name=args.language_model,
                                                    dim=args.scoring_model_dim, 
                                                    activation=args.scoring_model_activation,
                                                    suffix=args.scoring_model_suffix,
                                                    scaled=False)
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
                                        plm_cache_dir=args.plm_cache_dir)
    
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
    config["encoder_initializer"][0] = Initializers.ConstantInitializer(inverse_softplus(np.array(args.initial_distance) + 1e-8).numpy())
    transitioners = config["transitioner"] if hasattr(config["transitioner"], '__iter__') else [config["transitioner"]]
    for trans in transitioners:
        trans.prior.alpha_flank = args.alpha_flank
        trans.prior.alpha_single = args.alpha_single
        trans.prior.alpha_global = args.alpha_global
        trans.prior.alpha_flank_compl = args.alpha_flank_compl
        trans.prior.alpha_single_compl = args.alpha_single_compl
        trans.prior.alpha_global_compl = args.alpha_global_compl
    if not args.no_sequence_weights:
        os.makedirs(args.cluster_dir, exist_ok = True) 
        try:
            sequence_weights, clusters = Align.compute_sequence_weights(args.input_file, args.cluster_dir, config["cluster_seq_id"], return_clusters=True)
        except Exception as e:
            print("Error while computing sequence weights.")
            raise SystemExit(e) 
    else:
        sequence_weights, clusters = None, None
    if args.use_language_model:
        # we have to define a special model- and batch generator if using a language model
        # because the emission probabilities are computed differently and the LM requires specific inputs
        model_gen = make_generic_embedding_model_generator(config["scoring_model_config"].dim)
        batch_gen = EmbeddingBatchGenerator(config["scoring_model_config"])
    else:
        model_gen = None
        batch_gen = None
    if args.logo or args.logo_gif:
        os.makedirs(args.logo_path, exist_ok = True)
        if args.logo_gif:
            os.makedirs(args.logo_path+"/frames/", exist_ok = True)
    try:
        with SequenceDataset(args.input_file, "fasta", indexed=args.indexed_data) as data:
            data.validate_dataset()
            if args.crop == "disable":
                config["crop_long_seqs"] = math.inf
            elif args.crop == "auto":
                config["crop_long_seqs"] = int(np.ceil(args.auto_crop_scale * np.mean(data.seq_lens)))
            else:
                config["crop_long_seqs"] = int(args.crop)
            if args.logo_gif:
                def get_initial_model_lengths_logo_gif_mode(data : SequenceDataset, config):
                    #initial model length
                    model_length = np.quantile(data.seq_lens, q=config["length_init_quantile"])
                    model_length = max(3, int(model_length))
                    return [model_length] 
            alignment_model = Align.run_learnMSA(data,
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
                                    initial_model_length_callback = get_initial_model_lengths_logo_gif_mode if args.logo_gif else Align.get_initial_model_lengths,
                                    output_format = args.format,
                                    load_model = args.load_model,
                                    A2M_output=not args.noA2M)
            if args.save_model:
                alignment_model.write_models_to_file(args.save_model)
            if args.logo:
                Visualize.plot_and_save_logo(alignment_model, alignment_model.best_model, args.logo_path + "/logo.pdf")
            if args.dist_out:
                i = [l.name for l in alignment_model.encoder_model.layers].index("anc_probs_layer")
                anc_probs_layer = alignment_model.encoder_model.layers[i]
                tau_all = []
                for (seq, indices),_ in Training.make_dataset(alignment_model.indices, alignment_model.batch_generator, alignment_model.batch_size, shuffle=False):
                    seq = tf.transpose(seq, [1,0,2])
                    indices = tf.transpose(indices)
                    indices.set_shape([alignment_model.num_models,None]) #resolves tf 2.12 issues
                    indices = tf.expand_dims(indices, axis=-1)
                    tau = anc_probs_layer.make_tau(seq, indices)[alignment_model.best_model]
                    tau_all.append(tau.numpy())
                tau = np.concatenate(tau_all)
                with open(args.dist_out, "w") as file:
                    for i,t in zip(alignment_model.data.seq_ids, tau):
                        file.write(f"{i}\t{t}\n")
    except ValueError as e:
        raise SystemExit(e) 
 
            
if __name__ == '__main__':
    run_main()
