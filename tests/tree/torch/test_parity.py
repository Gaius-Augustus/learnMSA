"""Checks the PyTorch ancestral probability layer against TensorFlow outputs.

Same construction as ``tests/hmm/torch/test_parity.py``: the reference arrays
were recorded under the TensorFlow environment by
``tests/fixtures/generate_anc_probs_fixtures.py``, because no environment has
both frameworks.
"""

import numpy as np
import pytest
import torch

from tests.fixtures.generate_anc_probs_fixtures import (FIXTURE, SCENARIOS,
                                                        scenario_kwargs)

RTOL = 1e-5
ATOL = 1e-6


@pytest.fixture(scope="module")
def fixture_data():
    assert FIXTURE.exists(), (
        f"{FIXTURE.name} is committed to the repository but is missing here. "
        "Regenerate it with 'python tests/fixtures/"
        "generate_anc_probs_fixtures.py' under the TensorFlow environment."
    )
    with np.load(FIXTURE) as archive:
        yield {key: archive[key] for key in archive.files}


def _kernels(layer) -> dict[str, torch.nn.Parameter]:
    """The same neutral kernel paths the generator records."""
    kernels = {
        "exchangeability_delta_kernel": layer.exchangeability_delta_kernel,
        "equilibrium_kernel": layer.equilibrium_kernel,
        "tau_kernel": layer.tau_kernel,
    }
    if layer.num_components > 1:
        kernels["mixture_kernel"] = layer.mixture_kernel
        kernels["scale_kernel"] = layer.scale_kernel
    if layer.input_tracks > 1:
        kernels["tau_track_kernel"] = layer.tau_track_kernel
    return kernels


def _load_scenario(name: str, data: dict):
    """Build the layer and overwrite its kernels with the recorded ones."""
    from learnMSA.tree.torch.anc_probs_layer import TorchAncProbsLayer

    layer = TorchAncProbsLayer(**scenario_kwargs(name))
    layer.build()

    kernels = _kernels(layer)
    recorded = {
        key.split("/kernel.", 1)[1]: value
        for key, value in data.items()
        if key.startswith(f"{name}/kernel.")
    }
    assert set(recorded) == set(kernels), (
        f"{name}: the torch layer's kernels {sorted(kernels)} do not match the "
        f"recorded ones {sorted(recorded)}"
    )
    with torch.no_grad():
        for key, parameter in kernels.items():
            value = recorded[key]
            assert tuple(parameter.shape) == value.shape, (
                f"{name}/{key}: shape {tuple(parameter.shape)} does not match "
                f"the recorded {value.shape}"
            )
            parameter.copy_(torch.as_tensor(value, dtype=parameter.dtype))

    # The eigendecomposition is cached at build time from the kernels, so it
    # has to be redone now that they hold the recorded values.
    layer.substitution_model._precompute_gtr_decomposition()

    inputs = {
        key.split("/input.", 1)[1]: value
        for key, value in data.items()
        if key.startswith(f"{name}/input.")
    }
    return layer, inputs


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_kernel_shapes_match(scenario: str, fixture_data) -> None:
    """A shape mismatch means the backends parameterise the layer
    differently, which every other comparison would paper over."""
    _load_scenario(scenario, fixture_data)


@pytest.mark.parametrize("scenario", list(SCENARIOS))
@pytest.mark.parametrize("name", ["R", "p", "w", "scale", "Q", "tau", "P"])
def test_intermediates_match(
    scenario: str, name: str, fixture_data
) -> None:
    layer, inputs = _load_scenario(scenario, fixture_data)
    with torch.no_grad():
        tau = layer.make_tau(torch.as_tensor(inputs["rate_indices"]))
        got = {
            "R": layer.make_R,
            "p": layer.make_p,
            "w": layer.make_w,
            "scale": layer.make_scale,
            "Q": layer.make_Q,
            "tau": lambda: tau,
            "P": lambda: layer.make_P(tau),
        }[name]().numpy()
    expected = fixture_data[f"{scenario}/output.{name}"]
    np.testing.assert_allclose(got, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_anc_probs_match(scenario: str, fixture_data) -> None:
    layer, inputs = _load_scenario(scenario, fixture_data)
    sequences = [
        torch.as_tensor(inputs[key])
        for key in sorted(inputs) if key.startswith("sequence")
    ]
    with torch.no_grad():
        result = layer(
            *sequences,
            rate_indices=torch.as_tensor(inputs["rate_indices"]),
        )
    values = result if isinstance(result, tuple) else (result,)
    for i, value in enumerate(values):
        expected = fixture_data[f"{scenario}/output.anc_probs{i}"]
        np.testing.assert_allclose(
            value.numpy(), expected, rtol=RTOL, atol=ATOL
        )


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_transition_matrices_are_stochastic(
    scenario: str, fixture_data
) -> None:
    """An independent sanity check that does not depend on the fixture.

    ``time_reversed`` transposes P on the way out, so the stochastic axis moves
    with it: rows sum to one in the forward parameterisation, columns in the
    time-reversed one.
    """
    layer, inputs = _load_scenario(scenario, fixture_data)
    with torch.no_grad():
        tau = layer.make_tau(torch.as_tensor(inputs["rate_indices"]))
        P = layer.make_P(tau).numpy()
    assert (P >= 0).all()
    axis = -2 if layer.substitution_model.time_reversed else -1
    np.testing.assert_allclose(P.sum(axis=axis), 1.0, atol=1e-5)


def test_regularization_loss_is_reported() -> None:
    """The torch stand-in for the Keras ``regularizer=`` hook."""
    from learnMSA.tree.torch.anc_probs_layer import TorchAncProbsLayer

    kwargs = scenario_kwargs("single")
    kwargs.update(
        exchangeability_l2=0.5, trainable_exchangeabilities=True
    )
    layer = TorchAncProbsLayer(**kwargs)
    layer.build()

    # The delta kernel starts at zero, so the penalty starts at zero too.
    assert float(layer.regularization_loss().detach()) == pytest.approx(0.0)

    with torch.no_grad():
        layer.exchangeability_delta_kernel.fill_(1.0)
    expected = (
        0.5 / layer.heads * layer.exchangeability_delta_kernel.numel()
    )
    assert float(layer.regularization_loss().detach()) == pytest.approx(expected)
