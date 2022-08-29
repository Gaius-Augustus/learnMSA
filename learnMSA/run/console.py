import os
import numpy as np
import sys
import time
import argparse
from matplotlib import pyplot as plt
from pathlib import Path

#hide tensorflow messages and warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

version_file_path = "learnMSA/_version.py"
with open(version_file_path, "rt") as version_file:
    version = version_file.readlines()[0].split("=")[1].strip()

def run_main():
    class MsaHmmArgumentParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write('error: %s\n' % message)
            self.print_help()
            sys.exit(2)

    parser = MsaHmmArgumentParser(description=f"learnMSA - multiple alignment of protein sequences (version {version})")
    parser.add_argument("-i", "--in_file", dest="input_file", type=str, required=True,
                        help="Input sequence file in fasta format.")
    parser.add_argument("-o", "--out_file", dest="output_file", type=str, required=True,
                        help="Filepath for the output alignment.")
    parser.add_argument("-n", "--num_runs", dest="num_runs", type=int, default=1,
                        help="Number of trained models (default 1).")
    parser.add_argument("-s", "--silent", dest="silent", action='store_true', help="Prevents output to stdout.")
    parser.add_argument("-r", "--ref_file", dest="ref_file", type=str, default="", help=argparse.SUPPRESS) #useful for debudding, do not expose to users
    parser.add_argument("-b", "--batch", dest="batch_size", type=int, default=-1,
                        help="Should be lowered if memory issues with the default settings occur. Default: Adaptive, depending on model length (128-512).")
    parser.add_argument("-d", "--cuda_visible_devices", dest="cuda_visible_devices", type=str, default="default",
                        help="Controls the GPU devices visible to learnMSA as a comma-separated list of device IDs. The value -1 forces learnMSA to run on CPU. Per default, learnMSA attempts to use all available GPUs.")
    parser.add_argument("-t", "--tau_out", dest="tau_out", type=str, default="",
                        help="Optionally produce a file with the evolutionary times learned in the ancestral probabilities layer.")
    parser.add_argument("-L", "--logo_out", dest="logo_out", type=str, default="",
                        help="Optionally plot a consensus logo of the learned model.")
    parser.add_argument("-p", "--hmm_plot_out", dest="hmm_plot_out", type=str, default="",
                        help="Optionally plot the learned model.")
    args = parser.parse_args()
    
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
    
    config = dict(msa_hmm.config.default)
    
    config["batch_size"] = args.batch_size if args.batch_size > 0 else "adaptive"

    fasta_file = msa_hmm.fasta.Fasta(args.input_file)  

    if args.ref_file != "":
        ref_fasta = msa_hmm.fasta.Fasta(args.ref_file, aligned=True)
        subset = np.array([fasta_file.seq_ids.index(sid) for sid in ref_fasta.seq_ids])
    else:
        subset = None

    try:
        results = msa_hmm.align.fit_and_align_n(fasta_file, args.num_runs, config, subset, not args.silent)
    except tf.errors.ResourceExhaustedError as e:
        print("Out of memory. A resource was exhausted.")
        print("Try reducing the batch size (-b). The current batch size was: "+str(config["batch_size"])+".")
        sys.exit(e.error_code)

    best = np.argmax([ll for ll,_ in results])
    best_ll, best_alignment = results[best]
    
    if not args.silent:
        print("Computed alignments with MAP estimates:", ["%.4f" % ll for ll,_ in results])
        print("Best model has MAP estimate:", "%.4f" % best_ll)

    t = time.time()
    Path(os.path.dirname(args.output_file)).mkdir(parents=True, exist_ok=True)
    best_alignment.to_file(args.output_file)
    
    if not args.silent:
        print("time for generating output:", "%.4f" % (time.time()-t))
        print("Wrote file", args.output_file)

    if args.ref_file != "":
        out_file = msa_hmm.fasta.Fasta(args.output_file, aligned=True) 
        _,r = out_file.precision_recall(ref_fasta)
        #tc = out_file.tc_score(ref_fasta)
        
        if not args.silent:
            print("SP score =", r)#, "TC score =", tc)
            
    if not args.tau_out == "":
        Path(os.path.dirname(args.tau_out)).mkdir(parents=True, exist_ok=True)
        if not best_alignment.anc_probs_layer == None:
            msa_hmm.ut.write_tau_to_file(args.tau_out, best_alignment.anc_probs_layer, fasta_file)
            print("Wrote file", args.tau_out)
        else:
            print("Warning: No ancestral probability output can be produced, because the ancestral"
                  " probability layer is missing from the model. Was the default configuration changed?")
    
    
    if not args.logo_out == "":
        Path(os.path.dirname(args.logo_out)).mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots()
        msa_hmm.vis.make_logo(best_alignment, ax)
        plt.show()
        fig.savefig(args.logo_out, bbox_inches='tight')
        print("Wrote file", args.logo_out)
    
    
    if not args.hmm_plot_out == "":
        Path(os.path.dirname(args.hmm_plot_out)).mkdir(parents=True, exist_ok=True)
        fig = plt.figure(frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        msa_hmm.vis.plot_hmm(best_alignment, ax) 
        plt.show()
        plt.savefig(args.hmm_plot_out, bbox_inches='tight') 
        print("Wrote file", args.hmm_plot_out)
            
            
if __name__ == '__main__':
    run_main()