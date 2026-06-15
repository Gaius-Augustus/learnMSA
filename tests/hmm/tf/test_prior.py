import pytest
from hidten.hmm import HMMConfig as HidtenHMMConfig

from learnMSA.config.hmm import PHMMConfig, PHMMPriorConfig
from learnMSA.hmm.tf.prior import TFPHMMStartPrior, TFPHMMTransitionPrior
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

def test_start_prior(hmm_config: HidtenHMMConfig) -> None:
    lengths = [4, 3]
    prior_config = PHMMPriorConfig()
    start_prior = TFPHMMStartPrior(lengths, prior_config)
    start_prior.hmm_config = HidtenHMMConfig(states=[1])
    start_prior.build()

    with pytest.raises(NotImplementedError):
        start_prior.matrix()

    # Create a transitioner to get a start distribution
    values = [
        PHMMValueSet.from_config(L, h, PHMMConfig())
        for h, L in enumerate(lengths)
    ]
    transitioner = PHMMTransitioner(values=values)
    transitioner.hmm_config = hmm_config
    transitioner.build()
    start_dist = transitioner.explicit_transitioner.start_dist()

    # Test start prior
    scores = start_prior(start_dist)
    assert scores.shape == (2,)

    # Verify that scores are finite (not NaN or inf)
    import tensorflow as tf
    assert tf.reduce_all(tf.math.is_finite(scores))

def test_combined_priors(hmm_config: HidtenHMMConfig) -> None:
    """Test that transition and start priors work together correctly."""
    lengths = [4, 3]
    prior_config = PHMMPriorConfig()

    trans_prior = TFPHMMTransitionPrior(lengths, prior_config)
    trans_prior.hmm_config = HidtenHMMConfig(states=[1])
    trans_prior.build()

    start_prior = TFPHMMStartPrior(lengths, prior_config)
    start_prior.hmm_config = HidtenHMMConfig(states=[1])
    start_prior.build()

    # Create a transitioner to get matrices
    values = [
        PHMMValueSet.from_config(L, h, PHMMConfig())
        for h, L in enumerate(lengths)
    ]
    transitioner = PHMMTransitioner(values=values)
    transitioner.hmm_config = hmm_config
    transitioner.build()

    A = transitioner.explicit_transitioner.matrix()
    start_dist = transitioner.explicit_transitioner.start_dist()

    # Compute prior scores
    trans_scores = trans_prior(A)
    start_scores = start_prior(start_dist)
    combined = trans_scores + start_scores

    # Both should be shape (2,) for 2 heads
    assert trans_scores.shape == (2,)
    assert start_scores.shape == (2,)
    assert combined.shape == (2,)

    # Verify all scores are finite
    import tensorflow as tf
    assert tf.reduce_all(tf.math.is_finite(trans_scores))
    assert tf.reduce_all(tf.math.is_finite(start_scores))
    assert tf.reduce_all(tf.math.is_finite(combined))
