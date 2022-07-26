import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
import msa_hmm
         
if __name__ == '__main__':
    msa_hmm.learnMSA()