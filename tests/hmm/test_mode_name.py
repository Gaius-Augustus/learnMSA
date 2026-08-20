"""``PHMMLayer.mode_name`` -- the bridge from an HMMMode to a factor key.

``LearnMSAContext._get_impl_factor`` keys the implementation factors by these
names, so a name that disagrees with the layer's own predicates would silently
size batches for the wrong workload.
"""

import pytest

from hidten.hmm import HMMMode

from learnMSA.hmm.layer import PHMMLayer
from learnMSA.model.training_util import MODE_FALLBACK

#: Every mode one of the layer's mode setters can leave behind.
SETTABLE_MODES = {
    HMMMode.VITERBI: "viterbi",
    HMMMode.MEA: "mea",
    HMMMode.POSTERIOR: "posterior",
    HMMMode.LIKELIHOOD_LOG: "loglik",
}


def _layer(mode: HMMMode) -> PHMMLayer:
    """A layer stub carrying only the state mode_name reads."""
    layer = PHMMLayer.__new__(PHMMLayer)
    layer._mode = mode
    return layer


@pytest.mark.parametrize("mode,expected", sorted(
    SETTABLE_MODES.items(), key=lambda kv: kv[1]
))
def test_mode_name_matches_the_predicates(mode, expected):
    layer = _layer(mode)
    assert layer.mode_name() == expected

    predicates = {
        "viterbi": layer.is_viterbi_mode,
        "mea": layer.is_mea_mode,
        "posterior": layer.is_posterior_mode,
        "loglik": layer.is_loglik_mode,
    }
    for name, predicate in predicates.items():
        assert predicate() is (name == expected)


def test_every_name_is_a_workload_or_borrows_one():
    """No name may fall off the end of the factor lookup unnoticed."""
    from learnMSA.model import training_util

    calibrated = {"viterbi", "posterior", "loglik"}
    for name in SETTABLE_MODES.values():
        resolved = MODE_FALLBACK.get(name, name)
        assert resolved in calibrated
        # ...and the aggregate fallback exists for it in any case.
        assert "inference" in training_util.get_impl_factors("pytorch")
