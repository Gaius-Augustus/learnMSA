import logging

import numpy as np
import pytest
from hidten.hmm import HMMConfig as HidtenHMMConfig
from hidten.tf.prior.dirichlet import TFDirichletPrior

from learnMSA.hmm.tf.util import _warn_if_degenerate, load_dirichlet

# Decoded from the 3Di prior of a NaN checkpoint: two concentrations have
# collapsed to ~0 while the rest of the component stays informative.
DEGENERATE_3DI_ALPHA = [
    2.9468, 4.0324, 0.3988, 9.6950, 2.3896, 1.1143, 2.79e-07, 1.1640,
    1.9067, 5.5996, 2.5568, 6.7163, 2.1617, 1.4041, 6.5019, 4.0461,
    2.4106, 4.9721, 2.79e-07, 24.8488,
]


def _make_prior(alpha, components: int = 1) -> TFDirichletPrior:
    dim = len(alpha) // components
    prior = TFDirichletPrior(components=components)
    prior.hmm_config = HidtenHMMConfig(states=[1])
    prior.share = list(range(len(alpha)))
    prior.initializer = alpha
    prior.build((None, None, dim))
    return prior


def test_warns_on_collapsed_dimension(caplog: pytest.LogCaptureFixture) -> None:
    prior = _make_prior(DEGENERATE_3DI_ALPHA)
    with caplog.at_level(logging.WARNING):
        _warn_if_degenerate(prior, "degenerate")
    assert "collapsed dimension" in caplog.text


@pytest.mark.parametrize("name,components", [
    # A sharp prior, a moderate one and a deliberately flat one whose
    # concentrations are all far below 1.
    ("homstrad_3Di_1_20_1", 1),
    ("homstrad_3Di_3_20_1", 1),
    ("homstrad_3Di_no_20_1", 1),
    ("pfam_aa_neff_conc_3_20_1", 1),
    # A mixture whose dead components consist of tiny concentrations only;
    # judged per component, those must not trigger the warning.
    ("pfam_aa_neff_conc_3_20_9", 9),
])
def test_no_warning_for_shipped_priors(
    name: str, components: int, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        load_dirichlet(f"{name}.weights", dim=20, components=components)
    assert caplog.text == ""


def test_flat_component_is_not_degenerate(
    caplog: pytest.LogCaptureFixture
) -> None:
    # Every concentration below 1 is a legitimate anti-concentrated prior.
    prior = _make_prior(list(np.full(20, 0.01)))
    with caplog.at_level(logging.WARNING):
        _warn_if_degenerate(prior, "flat")
    assert caplog.text == ""
