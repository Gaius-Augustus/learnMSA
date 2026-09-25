"""The flank prior stays finite when the flank loops saturate in float32."""

import numpy as np
import pytest
import tensorflow as tf

from learnMSA.config.hmm import PHMMPriorConfig
from learnMSA.hmm.tf.prior import TFPHMMTransitionPrior
from tests.hmm.flank_prior_ref import L, expected_flank_prior, flank_matrix


@pytest.mark.parametrize("exit_prob", [0.1, 1e-3, 1e-9])
def test_flank_prior_matches_the_float64_reference(exit_prob) -> None:
    config = PHMMPriorConfig()
    prior = TFPHMMTransitionPrior(lengths=[L], prior_config=config)
    A = flank_matrix(exit_prob)
    if exit_prob < 1e-7:
        # The loops round to exactly 1, so 1 - loop would be 0
        assert A[0, 3 * L - 1, 3 * L - 1] == 1.0

    score = prior.compute_flank_prior(tf.constant(A)).numpy()

    np.testing.assert_allclose(
        score, [expected_flank_prior(A, config)], rtol=1e-5
    )
