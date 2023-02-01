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
    parser.add_argument("-n", "--num_model", dest="num_model", type=int, default=1,
                        help="Number of models trained in parallel (default 1).")
    parser.add_argument("-s", "--silent", dest="silent", action='store_true', help="Prevents output to stdout.")
    parser.add_argument("-r", "--ref_file", dest="ref_file", type=str, default="", help=argparse.SUPPRESS) #useful for debudding, do not expose to users
    parser.add_argument("-b", "--batch", dest="batch_size", type=int, default=-1,
                        help="Should be lowered if memory issues with the default settings occur. Default: Adaptive, depending on model length (128-512).")
    parser.add_argument("-d", "--cuda_visible_devices", dest="cuda_visible_devices", type=str, default="default",
                        help="Controls the GPU devices visible to learnMSA as a comma-separated list of device IDs. The value -1 forces learnMSA to run on CPU. Per default, learnMSA attempts to use all available GPUs.")
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
    
    _ = msa_hmm.align.run_learnMSA(train_filename = args.input_file,
                                    out_filename = args.output_file,
                                    config = config, 
                                    ref_filename = args.ref_file,
                                    verbose = not args.silent)
            
            
if __name__ == '__main__':
    run_main()