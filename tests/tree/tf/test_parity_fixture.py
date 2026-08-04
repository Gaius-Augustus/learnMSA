"""Keeps the ancestral-probability parity fixture honest.

The counterpart of ``tests/hmm/tf/test_parity_fixture_is_current.py`` for the
tree package: it re-derives ``tests/fixtures/anc_probs_parity.npz`` under
TensorFlow so the committed file cannot silently fall behind the backend the
torch parity tests are supposed to be compared against.
"""

import numpy as np
import pytest

from tests.fixtures.generate_anc_probs_fixtures import FIXTURE, collect

RTOL = 1e-6
ATOL = 1e-6


@pytest.fixture(scope="module")
def regenerated():
    return collect()


def test_fixture_exists() -> None:
    assert FIXTURE.exists(), (
        f"{FIXTURE} is missing. Generate it with "
        "'python tests/fixtures/generate_anc_probs_fixtures.py'."
    )


def test_fixture_has_the_same_keys(regenerated) -> None:
    with np.load(FIXTURE) as stored:
        stored_keys = set(stored.files)
    assert set(regenerated) == stored_keys, (
        "the fixture records a different set of arrays than the generator now "
        "produces; regenerate it."
    )


def test_fixture_values_are_current(regenerated) -> None:
    with np.load(FIXTURE) as archive:
        stored = {key: archive[key] for key in archive.files}
    for key in sorted(set(regenerated) & set(stored)):
        np.testing.assert_allclose(
            regenerated[key], stored[key], rtol=RTOL, atol=ATOL,
            err_msg=(
                f"{key} no longer matches the committed fixture. If the "
                "TensorFlow backend changed on purpose, regenerate the "
                "fixture and re-run the torch parity tests."
            ),
        )
