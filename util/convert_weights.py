import numpy as np
import tensorflow as tf
from hidten.hmm import HMMConfig as HidtenHMMConfig
from hidten.tf.prior.dirichlet import TFDirichletPrior

import learnMSA.msa_hmm.Priors as priors
from learnMSA.hmm.tf_util import make_dirichlet_model

# load the legacy prior
legacy_prior = priors.AminoAcidPrior(dtype=tf.float32)
legacy_prior.build()

legacy_alpha = legacy_prior.emission_dirichlet_mix.make_alpha().numpy().flatten()

# Add alphas for extra amino acids (non-standard + X)
legacy_alpha = np.concatenate([legacy_alpha, [0.01, 0.01, 0.01]])

model1 = make_dirichlet_model(legacy_alpha)
prior = model1.layers[1]

np.testing.assert_allclose(
    prior.matrix().numpy()[0,0], legacy_alpha, atol=1e-7
)

model_path = "learnMSA/hmm/weights/amino_acid_dirichlet.weights.h5"
model1.save_weights(model_path)

model2 = make_dirichlet_model()
model2.load_weights(model_path)
prior2 = model2.layers[1]

np.testing.assert_allclose(
    prior2.matrix().numpy()[0,0], legacy_alpha, atol=1e-7
)