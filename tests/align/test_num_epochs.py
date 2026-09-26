"""The epochs of each training round, including a final round that is final
because surgery converged before ``max_iterations`` was reached."""

import pytest

from learnMSA import Configuration
from learnMSA.align.align import _num_epochs


def _config(max_iterations: int) -> Configuration:
    config = Configuration()
    config.training.epochs = [7, 2, 11]
    config.training.max_iterations = max_iterations
    return config


@pytest.mark.parametrize("max_iterations", [1, 2, 3])
def test_first_round_gets_the_first_epochs(max_iterations) -> None:
    assert _num_epochs(
        _config(max_iterations), 0, last_iteration=max_iterations == 1
    ) == 7


def test_intermediate_round_gets_the_middle_epochs() -> None:
    assert _num_epochs(_config(3), 1, last_iteration=False) == 2


def test_regular_final_round_gets_the_final_epochs() -> None:
    assert _num_epochs(_config(3), 2, last_iteration=True) == 11


def test_early_convergence_gets_the_final_epochs() -> None:
    # max_iterations 3, surgery converged after the first round
    assert _num_epochs(_config(3), 1, last_iteration=True) == 11
