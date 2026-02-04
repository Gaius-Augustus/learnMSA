import argparse
import os
from itertools import product
from pathlib import Path

import numpy as np
import tensorflow as tf

import learnMSA.legacy.Priors as priors
import learnMSA.protein_language_models.Common as common
from learnMSA.hmm.tf.util import make_dirichlet_model, make_mvn_model
from learnMSA.protein_language_models.MvnPrior import (get_mvn_layer,
                                                       make_pdf_model)

WEIGHTS_PATH = "learnMSA/hmm/weights/"


def convert_dirichlet(name: str, legacy_alpha: np.ndarray) -> None:
    """Convert the legacy amino acid Dirichlet prior to the new format
    and save it as a weight file.
    """
    assert not np.any(np.isnan(legacy_alpha)),\
        "legacy_alpha contains NaN values"
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

    print(f"Converted and saved Dirichlet prior '{name}'.")

def convert_mvn(
    name: str,
    legacy_expectation: np.ndarray,
    legacy_variances: np.ndarray,
    legacy_mixture_coefficients: np.ndarray | None,
) -> None:
    """Convert the legacy MVN prior to the new format
    and save it as a weight file.

    Args:
        legacy_expectation: Shape (components, dim)
        legacy_variances: Shape (components, dim)
        legacy_mixture_coefficients: Shape (components,)
    """
    # Extract dimensions
    num_components, dim = legacy_expectation.shape

    # Assert no NaN values in legacy input arrays
    assert not np.any(np.isnan(legacy_expectation)),\
        "legacy_expectation contains NaN values"
    assert not np.any(np.isnan(legacy_variances)),\
        "legacy_variances contains NaN values"
    if legacy_mixture_coefficients is not None:
        assert not np.any(np.isnan(legacy_mixture_coefficients)),\
            "legacy_mixture_coefficients contains NaN values"

    # Concatenate all parameters into initializer array
    if legacy_mixture_coefficients is None:
        initializer = np.concatenate([
            legacy_expectation.flatten(),
            legacy_variances.flatten(),
        ], axis=0)
    else:
        initializer = np.concatenate([
            legacy_expectation.flatten(),
            legacy_variances.flatten(),
            legacy_mixture_coefficients.flatten(),
        ], axis=0)

    # Create model with initializer
    model1 = make_mvn_model(dim, initializer, components=num_components)
    prior = model1.layers[1]

    # Verify the conversion matches the legacy values
    reconstructed_mean = prior.mean().numpy()[0, 0]
    reconstructed_var = prior.variance().numpy()[0, 0]
    reconstructed_coef = prior.mixture_coefficients().numpy()[0, 0]

    np.testing.assert_allclose(
        reconstructed_mean, legacy_expectation, atol=1e-6
    )
    np.testing.assert_allclose(
        reconstructed_var, legacy_variances, atol=1e-6
    )
    if legacy_mixture_coefficients is not None:
        np.testing.assert_allclose(
            reconstructed_coef, legacy_mixture_coefficients, atol=1e-6
        )

    # Save the weights
    model_path = WEIGHTS_PATH + name + ".weights.h5"
    model1.save_weights(model_path)

    # Load back and verify
    model2 = make_mvn_model(dim=dim, components=num_components)
    model2.load_weights(model_path)
    prior2 = model2.layers[1]

    reconstructed_mean2 = prior2.mean().numpy()[0, 0]
    reconstructed_var2 = prior2.variance().numpy()[0, 0]
    reconstructed_coef2 = prior2.mixture_coefficients().numpy()[0, 0]

    np.testing.assert_allclose(
        reconstructed_mean2, legacy_expectation, atol=1e-6
    )
    np.testing.assert_allclose(
        reconstructed_var2, legacy_variances, atol=1e-6
    )
    if legacy_mixture_coefficients is not None:
        np.testing.assert_allclose(
            reconstructed_coef2, legacy_mixture_coefficients, atol=1e-6
        )

    print(f"Converted and saved MVN prior '{name}'.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert legacy priors to new format"
    )
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
        legacy_alpha = legacy_prior.emission_dirichlet_mix
        legacy_alpha = legacy_alpha.make_alpha().numpy().flatten()

        # Add alphas for extra amino acids (non-standard + X)
        # alpha = 1.0 marks a uniform prior for these amino acids
        # (i.e. irrelevant)
        # This will change the normalization constant compared to the case of
        # only 20 alphas values, but gradient will remain the same.
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
        lm_names = ["protT5", "esm2", "proteinBERT"]
        dims = [16, 32, 64, 128]
        activations = ["sigmoid", "softmax"]
        num_comps = [1, 10, 32, 100]
        values = product(lm_names, dims, activations, num_comps)
        for lm_name, dim, activation, num_comp in values:
            sm_config = common.ScoringModelConfig(lm_name, dim, activation)
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
            expectation = np.squeeze(mix.component_expectations(), axis=(0,1))
            variances = np.squeeze(mix.component_covariances(), axis=(0,1))
            if mix.num_components == 1:
                coefficients = None
            else:
                coefficients = np.squeeze(mix.mixture_coefficients(), axis=(0,1))
            name = Path(prior_path).stem # e.g. protT5_16_reduced_mix1_sigmoid
            convert_mvn(name, expectation, variances, coefficients)
