"""Pseudocounts derived from the shipped Dirichlet priors (PyTorch).

The counterpart of :mod:`learnMSA.hmm.tf.pseudocounts`. The pseudocounts are
plain numbers, but deriving them means evaluating a prior module's
``matrix()``, which is framework code, so it lives behind this backend
boundary and :class:`~learnMSA.model.context.LearnMSAContext` stays neutral.
"""

import numpy as np
import torch

from learnMSA.hmm.torch.prior import TorchPHMMTransitionPrior
from learnMSA.hmm.torch.util import load_dirichlet
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
    with torch.no_grad():
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
    with torch.no_grad():
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
    transition_prior = TorchPHMMTransitionPrior([5], prior_config)
    with torch.no_grad():
        match_psc = to_numpy(transition_prior.match_prior.matrix()[0, 0])
        ins_psc = to_numpy(transition_prior.insert_prior.matrix()[0, 0])
        del_psc = to_numpy(transition_prior.delete_prior.matrix()[0, 0])

    return aa_psc, match_psc, ins_psc, del_psc
