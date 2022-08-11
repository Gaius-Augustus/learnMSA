import os
import numpy as np
import copy
import random
import tensorflow as tf
import learnMSA.msa_hmm.DirichletMixture as dm

    
dtype = tf.float64
index_dtype = tf.int16

##################################################################################################
##################################################################################################


PRIOR_PATH = os.path.dirname(__file__)+"/trained_prior/"
DIRICHLET_COMP_COUNT = 1
model, DMP = dm.make_model(DIRICHLET_COMP_COUNT, 20, -1, trainable=False)
model.load_weights(PRIOR_PATH+str(DIRICHLET_COMP_COUNT)+"_components_prior_pdf/ckpt").expect_partial()
emission_dirichlet_mix = dm.DirichletMixturePrior(DIRICHLET_COMP_COUNT, 20, -1,
                                    DMP.alpha_kernel.numpy(),
                                    DMP.q_kernel.numpy(),
                                    trainable=False)
background_distribution = emission_dirichlet_mix.expectation()
#the prior was trained on example distributions over the 20 amino acid alpahbet
#the additional frequencies for 'B', 'Z',  'X', 'U', 'O' were derived from Pfam
background_distribution = np.concatenate([background_distribution, [2.03808244e-05, 1.02731819e-05, 7.92076933e-04, 5.84256792e-08, 1e-32]], axis=0)
background_distribution /= np.sum(background_distribution)

##################################################################################################
##################################################################################################

paml = ['0.425093 \n', '0.276818 0.751878 \n', '0.395144 0.123954 5.076149 \n', '2.489084 0.534551 0.528768 0.062556 \n', '0.969894 2.807908 1.695752 0.523386 0.084808 \n', '1.038545 0.363970 0.541712 5.243870 0.003499 4.128591 \n', '2.066040 0.390192 1.437645 0.844926 0.569265 0.267959 0.348847 \n', '0.358858 2.426601 4.509238 0.927114 0.640543 4.813505 0.423881 0.311484 \n', '0.149830 0.126991 0.191503 0.010690 0.320627 0.072854 0.044265 0.008705 0.108882 \n', '0.395337 0.301848 0.068427 0.015076 0.594007 0.582457 0.069673 0.044261 0.366317 4.145067 \n', '0.536518 6.326067 2.145078 0.282959 0.013266 3.234294 1.807177 0.296636 0.697264 0.159069 0.137500 \n', '1.124035 0.484133 0.371004 0.025548 0.893680 1.672569 0.173735 0.139538 0.442472 4.273607 6.312358 0.656604 \n', '0.253701 0.052722 0.089525 0.017416 1.105251 0.035855 0.018811 0.089586 0.682139 1.112727 2.592692 0.023918 1.798853 \n', '1.177651 0.332533 0.161787 0.394456 0.075382 0.624294 0.419409 0.196961 0.508851 0.078281 0.249060 0.390322 0.099849 0.094464 \n', '4.727182 0.858151 4.008358 1.240275 2.784478 1.223828 0.611973 1.739990 0.990012 0.064105 0.182287 0.748683 0.346960 0.361819 1.338132 \n', '2.139501 0.578987 2.000679 0.425860 1.143480 1.080136 0.604545 0.129836 0.584262 1.033739 0.302936 1.136863 2.020366 0.165001 0.571468 6.472279 \n', '0.180717 0.593607 0.045376 0.029890 0.670128 0.236199 0.077852 0.268491 0.597054 0.111660 0.619632 0.049906 0.696175 2.457121 0.095131 0.248862 0.140825 \n', '0.218959 0.314440 0.612025 0.135107 1.165532 0.257336 0.120037 0.054679 5.306834 0.232523 0.299648 0.131932 0.481306 7.803902 0.089613 0.400547 0.245841 3.151815 \n', '2.547870 0.170887 0.083688 0.037967 1.959291 0.210332 0.245034 0.076701 0.119013 10.649107 1.702745 0.185202 1.898718 0.654683 0.296501 0.098369 2.188158 0.189510 0.249313 \n', '\n', '0.079066 0.055941 0.041977 0.053052 0.012937 0.040767 0.071586 0.057337 0.022355 0.062157 0.099081 0.064600 0.022951 0.042302 0.044040 0.061197 0.053287 0.012066 0.034155 0.069147 \n', 'A R N D C Q E G H I L K M F P S T W Y V']

def read_paml_file():
    Q = np.zeros((20, 20), dtype=np.float32)
    s = [np.fromstring(paml[i], sep=" ") for i in range(19)]
    p = np.fromstring(paml[20], sep=" ")
    for i in range(20):
        for j in range(i):
            Q[i,j] = s[i-1][j] * p[j]
            Q[j,i] = s[i-1][j] * p[i]
    r = np.arange(20)
    Q -= np.sum(Q, axis=-1) * np.eye(20)
    mue = - np.sum(p * np.diagonal(Q))
    Q /= mue
    return Q, s, p 

# sequences.shape = (num_seq, len_seq, len_alphabet)
# Q.shape = (len_alphabet, len_alphabet)
# tau.shape = (num_seq)    #potentially each seq could have its own tau, moreover tau might be learnable
def make_anc_probs(sequences, Q, tau):
    tauQ = tf.reshape(tau, [-1,1,1]) * tf.expand_dims(Q, 0)
    P = tf.linalg.expm(tauQ) # P[r,i,j] = P(X(tau_r) = j | X(0) = i)
    ancprobs = tf.matmul(sequences, P) # Einstein sum, compute all ancestral character probs simultaneously
    return ancprobs
    
##################################################################################################
##################################################################################################
