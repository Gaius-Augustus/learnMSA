"""Checks the PyTorch pHMM layer against recorded TensorFlow outputs.

The reference arrays in ``tests/fixtures/phmm_parity.npz`` were produced by
``tests/fixtures/generate_parity_fixtures.py`` under the TensorFlow
environment; see that module for why the comparison is split in two.

Each test rebuilds the same layer under PyTorch, copies the recorded kernels
into it so the parameters are bit-identical, and then compares what the layer
computes. Tolerances are loose enough for float32 reductions to associate
differently on the two backends, and tight enough that a real translation error
in an emitter, the transitioner or a prior shows up.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tests.fixtures.generate_parity_fixtures import (  # noqa: E402
    FIXTURE, LENGTHS, SCENARIOS, input_shapes, make_configs)

pytestmark = pytest.mark.torch

#: Relative tolerance for probabilities and matrices.
RTOL = 1e-5

#: Absolute floor, so entries that are zero on one backend and denormal on the
#: other do not fail the relative check.
ATOL = 1e-6

#: The prior scores are sums of tens of thousands of log terms, so they carry a
#: correspondingly larger absolute error than the matrices they are derived
#: from.
PRIOR_RTOL = 1e-4


@pytest.fixture(scope="module")
def fixture_data():
    if not FIXTURE.exists():
        pytest.skip(
            f"{FIXTURE.name} is missing; regenerate it with "
            "'python tests/fixtures/generate_parity_fixtures.py' under the "
            "TensorFlow environment."
        )
    with np.load(FIXTURE) as archive:
        yield {key: archive[key] for key in archive.files}


def _build_layer(scenario: str):
    from learnMSA.hmm.torch.layer import TorchPHMMLayer

    config, prior_config, struct_config = make_configs(scenario)
    layer = TorchPHMMLayer(
        lengths=LENGTHS,
        config=config,
        prior_config=prior_config,
        struct_config=struct_config,
    )
    layer.build(input_shapes(struct_config))
    return layer, struct_config


def _kernels(layer) -> dict[str, torch.nn.Parameter]:
    """The same neutral kernel paths the generator records."""
    explicit = layer.hmm.transitioner.explicit_transitioner
    kernels = {
        "transitioner.kernel": explicit.kernel,
        "transitioner.kernel_start": explicit.kernel_start,
    }
    for name in ("profile_emitter", "struct_emitter", "joint_emitter"):
        emitter = getattr(layer, name, None)
        if emitter is not None:
            kernels[f"{name}.kernel"] = emitter.kernel
    return kernels


def _load_scenario(scenario: str, data: dict):
    """Build the layer and overwrite its kernels with the recorded ones."""
    layer, struct_config = _build_layer(scenario)
    kernels = _kernels(layer)

    recorded = {
        key.split("/kernel.", 1)[1]: value
        for key, value in data.items()
        if key.startswith(f"{scenario}/kernel.")
    }
    assert set(recorded) == set(kernels), (
        f"{scenario}: the torch layer's kernels {sorted(kernels)} do not match "
        f"the recorded ones {sorted(recorded)}"
    )
    with torch.no_grad():
        for name, parameter in kernels.items():
            value = recorded[name]
            assert tuple(parameter.shape) == value.shape, (
                f"{scenario}/{name}: shape {tuple(parameter.shape)} does not "
                f"match the recorded {value.shape}"
            )
            parameter.copy_(torch.as_tensor(value, dtype=parameter.dtype))

    inputs = {
        key.split("/input.", 1)[1]: value
        for key, value in data.items()
        if key.startswith(f"{scenario}/input.")
    }
    return layer, struct_config, inputs


def _expected(data: dict, scenario: str, name: str) -> np.ndarray | None:
    return data.get(f"{scenario}/output.{name}")


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_kernel_shapes_match(scenario: str, fixture_data) -> None:
    """A shape mismatch means the two backends parameterise the layer
    differently, which every other comparison would silently paper over."""
    _load_scenario(scenario, fixture_data)


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize(
    "name",
    [
        "transition_matrix",
        "explicit_transition_matrix",
        "emission_matrix.aa",
        "emission_matrix.struct",
        "emission_matrix.joint",
    ],
)
def test_matrices_match(scenario: str, name: str, fixture_data) -> None:
    expected = _expected(fixture_data, scenario, name)
    if expected is None:
        pytest.skip(f"the {scenario} layer has no {name}")

    layer, _, _ = _load_scenario(scenario, fixture_data)
    if name == "transition_matrix":
        got = layer.transition_matrix()
    elif name == "explicit_transition_matrix":
        got = layer.explicit_transition_matrix()
    else:
        got = layer.emission_matrix(name.split(".", 1)[1])

    np.testing.assert_allclose(got, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("name", ["start_dist", "explicit_start_dist"])
def test_start_distributions_match(
    scenario: str, name: str, fixture_data
) -> None:
    layer, _, _ = _load_scenario(scenario, fixture_data)
    transitioner = layer.hmm.transitioner
    if name == "start_dist":
        source = transitioner
    else:
        source = transitioner.explicit_transitioner
    with torch.no_grad():
        got = source.start_dist().numpy()
    expected = _expected(fixture_data, scenario, name)
    np.testing.assert_allclose(got, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_prior_scores_match(scenario: str, fixture_data) -> None:
    layer, _, _ = _load_scenario(scenario, fixture_data)
    expected = _expected(fixture_data, scenario, "prior_scores")
    with torch.no_grad():
        got = layer.prior_scores().numpy()
    np.testing.assert_allclose(got, expected, rtol=PRIOR_RTOL, atol=ATOL)


def _call(layer, inputs: dict[str, np.ndarray]):
    adds = None
    if "struct" in inputs:
        adds = (torch.as_tensor(inputs["struct"]),)
    return layer(
        torch.as_tensor(inputs["x"]),
        padding=torch.as_tensor(inputs["padding"]),
        adds=adds,
    )


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_emission_scores_match(scenario: str, fixture_data) -> None:
    """Everything the scan consumes, so a failure here is an emitter bug."""
    layer, _, inputs = _load_scenario(scenario, fixture_data)
    observations = [torch.as_tensor(inputs["x"])]
    if "struct" in inputs:
        observations.append(torch.as_tensor(inputs["struct"]))
    observations.append(torch.as_tensor(inputs["padding"]))
    with torch.no_grad():
        got = layer.hmm.emission_scores(*observations).numpy()
    expected = _expected(fixture_data, scenario, "emission_scores")
    np.testing.assert_allclose(got, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_loglik_matches(scenario: str, fixture_data) -> None:
    layer, _, inputs = _load_scenario(scenario, fixture_data)
    layer.loglik_mode()
    with torch.no_grad():
        got = _call(layer, inputs).numpy()
    expected = _expected(fixture_data, scenario, "loglik")
    np.testing.assert_allclose(got, expected, rtol=RTOL, atol=1e-4)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_viterbi_paths_match(scenario: str, fixture_data) -> None:
    """State paths are integers, so this one is exact or it is broken."""
    layer, _, inputs = _load_scenario(scenario, fixture_data)
    layer.viterbi_mode()
    with torch.no_grad():
        got = _call(layer, inputs).numpy()
    expected = _expected(fixture_data, scenario, "viterbi")
    np.testing.assert_array_equal(got, expected)
