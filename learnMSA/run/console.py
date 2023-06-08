import os
import numpy as np
import sys
import argparse

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
    parser.add_argument("-n", "--num_model", dest="num_model", type=int, default=5,
                        help="Number of models trained in parallel. (default: %(default)s)")
    parser.add_argument("-s", "--silent", dest="silent", action='store_true', help="Prevents output to stdout.")
    parser.add_argument("-r", "--ref_file", dest="ref_file", type=str, default="", help=argparse.SUPPRESS) #useful for debudding, do not expose to users
    parser.add_argument("-b", "--batch", dest="batch_size", type=int, default=-1,
                        help="Should be lowered if memory issues with the default settings occur. Default: Adaptive, depending on model length (64-512).")
    parser.add_argument("-d", "--cuda_visible_devices", dest="cuda_visible_devices", type=str, default="default",
                        help="Controls the GPU devices visible to learnMSA as a comma-separated list of device IDs. The value -1 forces learnMSA to run on CPU. Per default, learnMSA attempts to use all available GPUs.")
    
    parser.add_argument("--max_surgery_runs", dest="max_surgery_runs", type=int, default=4, 
                        help="Maximum number of model surgery iterations. (default: %(default)s)")
    parser.add_argument("--length_init_quantile", dest="length_init_quantile", type=float, default=0.5, 
                        help="Quantile of the input sequence lengths that defines the initial model lengths. (default: %(default)s)")
    parser.add_argument("--surgery_quantile", dest="surgery_quantile", type=float, default=0.5, 
                        help="learnMSA will not use sequences shorter than this quantile for training during all iterations except the last. (default: %(default)s)")
    parser.add_argument("--min_surgery_seqs", dest="min_surgery_seqs", type=int, default=10000, 
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
    parser.add_argument("--alpha_flank", dest="alpha_flank", type=float, default=7000, help=argparse.SUPPRESS)
    parser.add_argument("--alpha_single", dest="alpha_single", type=float, default=1e9, help=argparse.SUPPRESS)
    parser.add_argument("--alpha_frag", dest="alpha_frag", type=float, default=1e4, help=argparse.SUPPRESS)
    
    
    parser.add_argument("--align_insertions", dest="align_insertions", action='store_true', help="Aligns long insertions with a third party aligner after the main MSA step. (default: %(default)s)")
    parser.add_argument("--insertion_slice_dir", dest="insertion_slice_dir", type=str, default="tmp", help="Directory where the alignments of the sliced insertions are stored. (default: %(default)s)")
    
    
    args = parser.parse_args()
    
    if not args.silent:
        print(f"learnMSA (version {version}) - multiple alignment of protein sequences")
    
    #import after argparsing to avoid long delay with -h option
    if not args.cuda_visible_devices == "default":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices  
    from .. import msa_hmm
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
    
    config = msa_hmm.config.make_default(args.num_model)
    
    config["batch_size"] = args.batch_size if args.batch_size > 0 else msa_hmm.config.get_adaptive_batch_size
    config["num_models"] = args.num_model
    config["max_surgery_runs"] = args.max_surgery_runs
    config["length_init_quantile"] = args.length_init_quantile
    config["surgery_quantile"] = args.surgery_quantile
    config["min_surgery_seqs"] = args.min_surgery_seqs
    config["len_mul"] = args.len_mul
    config["learning_rate"] = args.learning_rate
    config["epochs"] = args.epochs
    config["surgery_del"] = args.surgery_del
    config["surgery_ins"] = args.surgery_ins
    config["model_criterion"] = args.model_criterion
    transitioners = config["transitioner"] if hasattr(config["transitioner"], '__iter__') else [config["transitioner"]]
    for trans in transitioners:
        trans.prior.alpha_flank = args.alpha_flank
        trans.prior.alpha_single = args.alpha_single
        trans.prior.alpha_frag = args.alpha_frag
    if args.align_insertions:
        os.makedirs(args.insertion_slice_dir, exist_ok = True) 
    _ = msa_hmm.align.run_learnMSA(train_filename = args.input_file,
                                    out_filename = args.output_file,
                                    config = config, 
                                    ref_filename = args.ref_file,
                                    align_insertions=args.align_insertions,
                                    insertion_slice_dir=args.insertion_slice_dir,
                                    verbose = not args.silent)
            
            
if __name__ == '__main__':
    run_main()