import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import msa_hmm
         
if __name__ == '__main__':
    msa_hmm.learnMSA()