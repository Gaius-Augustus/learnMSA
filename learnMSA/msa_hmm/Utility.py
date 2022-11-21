import os
import numpy as np
import copy
import random
import tensorflow as tf
import learnMSA.msa_hmm.DirichletMixture as dm

    
dtype = tf.float64
index_dtype = tf.int16


PRIOR_PATH = os.path.dirname(__file__)+"/trained_prior/"
DIRICHLET_COMP_COUNT = 1
model, DMP = dm.make_model(DIRICHLET_COMP_COUNT, 20, -1, trainable=False)
model.load_weights(PRIOR_PATH+str(DIRICHLET_COMP_COUNT)+"_components_prior_pdf/ckpt").expect_partial()
emission_dirichlet_mix = dm.DirichletMixturePrior(DIRICHLET_COMP_COUNT, 20, -1,
                                    DMP.alpha_kernel.numpy(),
                                    DMP.q_kernel.numpy(),
                                    trainable=False)
background_distribution = emission_dirichlet_mix.expectation()
#the prior was trained on example distributions over the 20 amino acid alphabet
#the additional frequencies for 'B', 'Z',  'X', 'U', 'O' were derived from Pfam
background_distribution = np.concatenate([background_distribution, [2.03808244e-05, 1.02731819e-05, 7.92076933e-04, 5.84256792e-08, 1e-32]], axis=0)
background_distribution /= np.sum(background_distribution)
