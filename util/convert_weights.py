import numpy as np
import tensorflow as tf
from hidten.hmm import HMMConfig as HidtenHMMConfig
from hidten.tf.prior.dirichlet import TFDirichletPrior
from pyparsing import Callable

import learnMSA.msa_hmm.Priors as priors
from learnMSA.hmm.tf_util import make_dirichlet_model

WEIGHTS_PATH = "learnMSA/hmm/weights/"


def convert_dirichlet(name: str, legacy_alpha: np.ndarray) -> None:
    """Convert the legacy amino acid Dirichlet prior to the new format
    and save it as a weight file.
    """
    model1 = make_dirichlet_model(legacy_alpha)
    prior = model1.layers[1]

    np.testing.assert_allclose(
        prior.matrix().numpy()[0,0], legacy_alpha, atol=1e-7
    )

    model_path = WEIGHTS_PATH + name + ".weights.h5"
    model1.save_weights(model_path)

    model2 = make_dirichlet_model(np.ones_like(legacy_alpha))
    model2.load_weights(model_path)
    prior2 = model2.layers[1]

    np.testing.assert_allclose(
        prior2.matrix().numpy()[0,0], legacy_alpha, atol=1e-7
    )

if __name__ == "__main__":
    # load the legacy amino acid prior
    legacy_prior = priors.AminoAcidPrior(dtype=tf.float32)
    legacy_prior.build()
    legacy_alpha = legacy_prior.emission_dirichlet_mix.make_alpha().numpy().flatten()

    # Add alphas for extra amino acids (non-standard + X)
    legacy_alpha = np.concatenate([legacy_alpha, [0.01, 0.01, 0.01]])

    convert_dirichlet("amino_acid_dirichlet", legacy_alpha)

    # load the legacy transition priors
    legacy_trans_prior = priors.ProfileHMMTransitionPrior(dtype=tf.float32)
    legacy_trans_prior.build()
    legacy_match_alpha = legacy_trans_prior.match_dirichlet.make_alpha()
    legacy_insert_alpha = legacy_trans_prior.insert_dirichlet.make_alpha()
    legacy_delete_alpha = legacy_trans_prior.delete_dirichlet.make_alpha()
    legacy_match_alpha = legacy_match_alpha.numpy().flatten()
    legacy_insert_alpha = legacy_insert_alpha.numpy().flatten()
    legacy_delete_alpha = legacy_delete_alpha.numpy().flatten()

    convert_dirichlet("transition_match_dirichlet", legacy_match_alpha)
    convert_dirichlet("transition_insert_dirichlet", legacy_insert_alpha)
    convert_dirichlet("transition_delete_dirichlet", legacy_delete_alpha)
