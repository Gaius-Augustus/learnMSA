import numpy as np
import sys
import time
from argparse import ArgumentParser
import msa_hmm
import tensorflow as tf
from tensorflow.python.client import device_lib

def learnMSA():
    class MsaHmmArgumentParser(ArgumentParser):
        def error(self, message):
            sys.stderr.write('error: %s\n' % message)
            self.print_help()
            sys.exit(2)

    parser = MsaHmmArgumentParser()
    parser.add_argument("-i", "--in_file", dest="input_file", type=str, required=True,
                        help="Unaligned input sequence file in fasta format.")
    parser.add_argument("-o", "--out_file", dest="output_file", type=str, required=True,
                        help="Filepath for the output alignment.")
    parser.add_argument("-n", "--num_runs", dest="num_runs", type=int, default=5,
                        help="Number of trained models.")
    parser.add_argument("-s", "--silent", dest="silent", action='store_true', help="Prevents any output.")
    parser.add_argument("-r", "--ref_file", dest="ref_file", type=str, default="",
                        help="Filepath for a reference alignment; if present, SP and TC scores are computed.")
    parser.add_argument("-b", "--batch", dest="batch_size", type=int, default=-1,
                        help="Default: Adaptive, depending on model length. Can be lowered if memory issues with the default settings occur.")
    args = parser.parse_args()
    
    if not args.silent:
        GPUS = [x.physical_device_desc for x in device_lib.list_local_devices() if x.device_type == 'GPU']
        if len(GPUS) == 0:
            print("It seems like no GPU is installed. Running on CPU instead. Expect slower performance especially for longer models.")
        else:
            print("Using GPU(s):", GPUS)

        print("Found tensorflow version", tf.__version__)
    
    config = dict(msa_hmm.config.default)
    
    config["batch_size"] = args.batch_size if args.batch_size > 0 else "adaptive"

    fasta_file = msa_hmm.fasta.Fasta(args.input_file, 
                             gaps=False, 
                             contains_lower_case=True)  

    if args.ref_file != "":
        ref_fasta = msa_hmm.fasta.Fasta(args.ref_file, gaps=True, contains_lower_case=True)
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
        print("Computed alignments with likelihoods:", [ll for ll,_ in results])
        print("Best model has likelihood:", best_ll)

    t = time.time()
    best_alignment.to_file(args.output_file)
    
    if not args.silent:
        print("time for generating output:", time.time()-t)
        print("Wrote file", args.output_file)

    if args.ref_file != "":
        out_file = msa_hmm.fasta.Fasta(args.output_file, 
                                 gaps=True, 
                                 contains_lower_case=True) 
        _,r = out_file.precision_recall(ref_fasta)
        tc = out_file.tc_score(ref_fasta)
        
        if not args.silent:
            print("SP score =", r, "TC score =", tc)