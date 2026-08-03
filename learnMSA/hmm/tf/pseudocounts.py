"""Pseudocounts derived from the shipped Dirichlet priors (TensorFlow).

The pseudocounts are plain numbers, but deriving them means evaluating a prior
layer's ``matrix()``, which is framework code. Keeping that behind this function
lets :class:`~learnMSA.model.context.LearnMSAContext` stay backend-neutral; a
torch backend supplies ``learnMSA/hmm/torch/pseudocounts.py`` with the same
signature.
"""

import numpy as np

from learnMSA.hmm.tf.prior import TFPHMMTransitionPrior
from learnMSA.hmm.tf.util import load_dirichlet
from learnMSA.util.sequence_dataset import SequenceDataset
from learnMSA.util.tensor import to_numpy


def struct_prior_mean(structural_config) -> np.ndarray:
    """The mean of the structural Dirichlet prior, as a numpy array.

    Used by model surgery to pick the emission values of newly inserted
    positions.
    """
    c = structural_config.prior_components
    struct_prior = load_dirichlet(
        f"{structural_config.prior_name}_{c}.weights",
        dim=structural_config.alphabet_size,
        components=structural_config.prior_components,
        states=[1],
    )
    return to_numpy(struct_prior.mean()[0, 0])


def load_pseudocounts(
    prior_config,
    emission_alphabet_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Derive emission and transition pseudocounts from the Dirichlet priors.

    Args:
        prior_config: The ``hmm_prior`` section of the configuration.
        emission_alphabet_size: Size of the emission alphabet the amino acid
            pseudocounts have to broadcast onto.

    Returns:
        ``(aa, match, insert, delete)`` pseudocounts as numpy arrays.
    """
    # Amino acid pseudocounts
    c = prior_config.amino_acid_dirichlet_components
    aa_prior = load_dirichlet(
        f"{prior_config.amino_acid_prior_name}_{c}.weights",
        dim=len(SequenceDataset._default_alphabet),
    )
    alpha = to_numpy(aa_prior.matrix())[0, 0]
    C = aa_prior.config.components
    if C > 1:
        mix_coeff = alpha[C * aa_prior.input_dim:]
        alpha = alpha[:C * aa_prior.input_dim]
        alpha = np.reshape(alpha, (C, aa_prior.input_dim))
        aa_psc = np.sum(mix_coeff[:, np.newaxis] * alpha, axis=0)
    else:
        aa_psc = alpha

    # The prior is 20-dimensional; pad to the emission alphabet size
    # (small mass for U/O) so pseudocounts broadcast onto the emissions.
    if aa_psc.shape[0] < emission_alphabet_size:
        pad = np.full(
            emission_alphabet_size - aa_psc.shape[0], float(aa_psc.min())
        )
        aa_psc = np.concatenate([aa_psc, pad])

    # Transition pseudocounts
    transition_prior = TFPHMMTransitionPrior([5], prior_config)
    match_psc = to_numpy(transition_prior.match_prior.matrix()[0, 0])
    ins_psc = to_numpy(transition_prior.insert_prior.matrix()[0, 0])
    del_psc = to_numpy(transition_prior.delete_prior.matrix()[0, 0])

    return aa_psc, match_psc, ins_psc, del_psc
