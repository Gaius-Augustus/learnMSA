from enum import Enum

import numpy as np

from learnMSA.model.tf.model import LearnMSAModel
from learnMSA.util.sequence_dataset import SequenceDataset


class SelectionCriterion(Enum):
    POSTERIOR = "posterior"
    LOGLIK = "loglik"
    AIC = "AIC"
    CONSENSUS = "consensus"

def select_model(
    model: LearnMSAModel,
    data: SequenceDataset,
    model_criterion: SelectionCriterion,
    sequence_indices: np.ndarray|None = None,
    verbose: bool = False,
) -> int:
    """
    Selects the best HMM from the pHMM layer in the LearnMSAModel based on the
    specified selection criterion.

    Args:
        model (LearnMSAModel): The LearnMSAModel containing the pHMM layer.
        data (SequenceDataset): The dataset containing the sequences.
        model_criterion (SelectionCriterion): The criterion to use for model
            selection.
        sequence_indices (np.ndarray|None): The indices of the sequences to use.
            If None, all sequences are used.
        verbose (bool): If True, prints additional information during selection.

    Returns:
        int: The index of the best model according to the selection criterion.
    """
    if sequence_indices is None:
        sequence_indices = np.arange(data.num_seq)
    scores = get_model_scores(
        model, data, model_criterion, sequence_indices, verbose
    )
    best = np.argmax(scores)
    if verbose:
        print("Selection criterion:", model_criterion)
        print("Best model: ", best, "(0-based)")
    return int(best)

def get_model_scores(
    model: LearnMSAModel,
    data: SequenceDataset,
    model_criterion: SelectionCriterion,
    sequence_indices: np.ndarray|None = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Computes selection scores (higher is better) for each pHMM in the
    LearnMSAModel.

    Args:
        model (LearnMSAModel): The LearnMSAModel containing the pHMM layer.
        data (SequenceDataset): The dataset containing the sequences.
        model_criterion (SelectionCriterion): The criterion to use for model
            selection.
        sequence_indices (np.ndarray|None): The indices of the sequences to use.
            If None, all sequences are used.
        verbose (bool): If True, prints additional information during selection.

    Returns:
        np.ndarray: An array of scores for each model of shape `(H,)`.
    """
    if sequence_indices is None:
        ind = np.arange(data.num_seq)
    else:
        ind = sequence_indices
    match model_criterion:
        case SelectionCriterion.POSTERIOR:
            return select_model_posterior(model, data, ind, verbose)
        case SelectionCriterion.LOGLIK:
            return select_model_loglik(model, data, ind, verbose)
        case SelectionCriterion.AIC:
            return select_model_AIC(model, data, ind, verbose)
        case SelectionCriterion.CONSENSUS:
            return select_model_consensus(model, verbose)
        case _:
            raise SystemExit(
                "Invalid model selection criterion. Valid criteria are: "\
                f"{list(SelectionCriterion)}."
            )

def select_model_posterior(
    model: LearnMSAModel,
    data: SequenceDataset,
    sequence_indices: np.ndarray|None = None,
    verbose: bool = False,
) -> np.ndarray:
    model.posterior_mode()
    model.compile()
    expected_state = model.predict(data, sequence_indices, reduce=True)
    posterior_sums = np.array([
        np.sum(expected_state[i, :model.lengths[i]])
        for i in range(model.heads)
    ])
    if verbose:
        print("Total expected match states:", posterior_sums)
    return posterior_sums

#TODO: the default is to use the prior although not using is seems to be very
# slightly better the default argument should change later to false but keep
# using prior for now for legacy reasons
def select_model_loglik(
    model: LearnMSAModel,
    data: SequenceDataset,
    sequence_indices: np.ndarray|None = None,
    use_prior: bool = True,
    verbose: bool = False,
) -> np.ndarray:
    metrics = model.evaluate(data, sequence_indices)
    score = metrics["loglik"]
    if use_prior:
        score += metrics["prior"]
    if verbose:
        if use_prior:
            likelihoods = [
                "%.4f" % ll + " (%.4f)" % p
                for ll,p in zip(metrics["loglik"], metrics["prior"])
            ]
            print("Likelihoods (priors): ", likelihoods)
        else:
            likelihoods = ["%.4f" % ll for ll in metrics["loglik"]]
            print("Likelihoods: ", likelihoods)
            print("Mean likelihood: ", np.mean(metrics["loglik"]))
    return score


def select_model_AIC(
    model: LearnMSAModel,
    data: SequenceDataset,
    sequence_indices: np.ndarray|None = None,
    verbose: bool = False,
) -> np.ndarray:
    loglik = select_model_loglik(
        model, data, sequence_indices, use_prior=False, verbose=verbose
    )
    aic = model.estimate_AIC(data, loglik=loglik)
    return -aic #negate as we want to take the maximum


def select_model_consensus(
    model: LearnMSAModel,
    verbose: bool = False,
) -> np.ndarray:
    consensus = model.compute_consensus_score()
    if verbose:
        print("Consensus scores: ", ["%.4f" % c for c in consensus])
    return np.array(consensus)
