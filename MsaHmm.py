import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
import sys
import time
from argparse import ArgumentParser



class MsaHmmArgumentParser(ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

parser = MsaHmmArgumentParser()
parser.add_argument("-i", "--in_file", dest="input_file", type=str, required=True,
                    help="unaligned input sequence file in fasta format")
parser.add_argument("-o", "--out_file", dest="output_file", type=str, required=True,
                    help="filepath for the output alignment")
parser.add_argument("-n", "--num_runs", dest="num_runs", type=int, default=5,
                    help="number of independent sugery runs")
parser.add_argument("-v", "--verbose", dest="verbose", type=bool, default=True,
                    help="verbosity options: 0 - no output, 1 - full output")
parser.add_argument("-r", "--ref_file", dest="ref_file", type=str, default="",
                    help="filepath for a reference alignment; if present, SP and TC scores are computed")
args = parser.parse_args()



#import after argparsing to avoid long delay if argument requirements are not met or help is printed
import msa_hmm
import numpy as np

config = msa_hmm.config.default    
    
fasta_file = msa_hmm.fasta.Fasta(args.input_file, 
                         gaps=False, 
                         contains_lower_case=True)  

if args.ref_file != "":
    ref_fasta = msa_hmm.fasta.Fasta(args.ref_file, gaps=True, contains_lower_case=True)
    subset = np.array([fasta_file.seq_ids.index(sid) for sid in ref_fasta.seq_ids])
else:
    subset = None
    
results = msa_hmm.align.fit_and_align_n(fasta_file, args.num_runs, config, subset, args.verbose)


best = np.argmax([ll for ll,_ in results])
best_ll, best_alignment = results[best]
print("Computed alignments with likelihoods:", [ll for ll,_ in results])
print("Best model has likelihood:", best_ll)

t = time.time()
best_alignment.to_file(args.output_file)
print("time for generating output:", time.time()-t)
print("Wrote file", args.output_file)

if args.ref_file != "":
    out_file = msa_hmm.fasta.Fasta(args.output_file, 
                             gaps=True, 
                             contains_lower_case=True) 
    _,r = out_file.precision_recall(ref_fasta)
    tc = out_file.tc_score(ref_fasta)
    print("SP score =", r, "TC score =", tc)