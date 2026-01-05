import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

import learnMSA.msa_hmm.Priors as priors
import learnMSA.protein_language_models.Common as common
from learnMSA.hmm.tf_util import make_dirichlet_model
from learnMSA.protein_language_models.MvnPrior import (get_mvn_layer,
                                                       make_pdf_model)

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

def convert_mvn(
        name: str, legacy_expectation: np.ndarray, legacy_variances: np.ndarray
) -> None:
    """Convert the legacy MVN prior to the new format
    and save it as a weight file.
    """
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert legacy priors to new format")
    parser.add_argument(
        "--dirichlet", action="store_true", help="Convert Dirichlet priors"
    )
    parser.add_argument(
        "--mvn", action="store_true", help="Convert MVN priors"
    )
    args = parser.parse_args()

    if args.dirichlet:
        # load the legacy amino acid prior
        legacy_prior = priors.AminoAcidPrior(dtype=tf.float32)
        legacy_prior.build()
        legacy_alpha = legacy_prior.emission_dirichlet_mix.make_alpha().numpy().flatten()

        # Add alphas for extra amino acids (non-standard + X)
        # alpha = 1.0 marks a uniform prior for these amino acids (i.e. irrelevant)
        # This will change the normalization constant compared to the case of only
        # 20 alphas values, but gradient will remain the same.
        legacy_alpha = np.concatenate([legacy_alpha, [1.0]*3])

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

    if args.mvn:
        sm_config = common.ScoringModelConfig()
        num_comp = 1
        prior_path = common.get_prior_path(sm_config, num_comp)
        pdf_model = make_pdf_model(sm_config, num_comp, trainable=False)
        pdf_model.load_weights(
            os.path.dirname(__file__)\
                + f"/../learnMSA/protein_language_models/"\
                + prior_path
        )
        mvn_layer = get_mvn_layer(pdf_model)
        assert mvn_layer is not None
        mix = mvn_layer.get_mixture()
        expectation = mix.component_expectations()[0,0]
        variances = mix.component_covariances()[0,0]
        name = Path(prior_path).stem # e.g. protT5_16_reduced_mix1_sigmoid
        convert_mvn(name, expectation.numpy(), variances.numpy())
