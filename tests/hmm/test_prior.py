import pytest
from hidten.hmm import HMMConfig as HidtenHMMConfig

from learnMSA.config.hmm import HMMConfig
from learnMSA.hmm.prior import TFPHMMTransitionPrior
from learnMSA.hmm.transitioner import PHMMTransitioner
from learnMSA.hmm.value_set import PHMMValueSet


def test_transition_prior() -> None:
    lengths = [4, 3]
    prior = TFPHMMTransitionPrior(lengths)
    prior.hmm_config = HidtenHMMConfig(states = [1])
    prior.build()

    with pytest.raises(NotImplementedError):
        prior.matrix()

    # Test sub priors
    assert prior.match_prior.matrix().shape == (1, 1, 3)
    assert prior.insert_prior.matrix().shape == (1, 1, 2)
    assert prior.delete_prior.matrix().shape == (1, 1, 2)

    # Create a transitioner to get a transition matrix
    values = [
        PHMMValueSet.from_config(L, h, HMMConfig())
        for h, L in enumerate(lengths)
    ]
    transitioner = PHMMTransitioner(values=values)
    transitioner.build()
    A = transitioner.matrix()

    match_scores = prior.compute_transition_prior(
        A, prior.TransitionType.MATCH
    )
    insert_scores = prior.compute_transition_prior(
        A, prior.TransitionType.INSERT
    )
    delete_scores = prior.compute_transition_prior(
        A, prior.TransitionType.DELETE
    )
    scores = prior(A)

    assert match_scores.shape == (2,)
    assert insert_scores.shape == (2,)
    assert delete_scores.shape == (2,)
    assert scores.shape == (2,)
