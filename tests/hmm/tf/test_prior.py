import pytest
from hidten.hmm import HMMConfig as HidtenHMMConfig

from learnMSA.config.hmm import PHMMConfig, PHMMPriorConfig
from learnMSA.hmm.tf.prior import TFPHMMTransitionPrior
from learnMSA.hmm.tf.transitioner import PHMMTransitioner
from learnMSA.hmm.util.value_set import PHMMValueSet


@pytest.fixture
def hmm_config() -> HidtenHMMConfig:
    lengths = [4, 3]
    return HidtenHMMConfig(states=[2*L+2 for L in lengths])

def test_transition_prior(hmm_config: HidtenHMMConfig) -> None:
    lengths = [4, 3]
    prior_config = PHMMPriorConfig()
    prior = TFPHMMTransitionPrior(lengths, prior_config)
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
        PHMMValueSet.from_config(L, h, PHMMConfig())
        for h, L in enumerate(lengths)
    ]
    transitioner = PHMMTransitioner(values=values)
    transitioner.hmm_config = hmm_config
    transitioner.build()
    A = transitioner.explicit_transitioner.matrix()

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
