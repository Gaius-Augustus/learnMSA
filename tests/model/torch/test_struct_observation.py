"""Tests for the structural observation (misclassification) model."""

import numpy as np
import pytest
import torch

from learnMSA.config import Configuration, TrainingConfig
from learnMSA.config.structure import StructureConfig
from learnMSA.model.context import LearnMSAContext
from learnMSA.model.torch.model import TorchLearnMSAModel as LearnMSAModel
from learnMSA.model.torch.struct_observation import (
    StructObservationLayer,
    load_confusion,
)

LENGTHS = [6, 4]
NUM_HEADS = len(LENGTHS)
BATCH, SEQ_LEN = 3, 7


def _make_config(
    noise: str = "none",
    strength: float = 0.1,
    use_anc_probs: bool = True,
    joint_emissions: bool = False,
    trainable: bool = False,
) -> Configuration:
    config = Configuration(training=TrainingConfig(length_init=LENGTHS))
    config.structure.use_structure = True
    config.structure.joint_emissions = joint_emissions
    config.structure.observation_noise = noise
    config.structure.observation_noise_strength = strength
    config.structure.trainable_observation_noise = trainable
    config.tree.use_anc_probs = use_anc_probs
    return config


def _build(config: Configuration) -> LearnMSAModel:
    model = LearnMSAModel(LearnMSAContext(config=config, num_seq=10))
    model.build()
    model.loglik_mode()
    # Break the symmetry of a fresh model; identical seeds give identical
    # parameters for models with the same parameter list.
    generator = torch.Generator(device="cpu").manual_seed(0)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim == 0:
                continue
            noise = torch.randn(
                parameter.shape, generator=generator, dtype=torch.float32
            )
            parameter.add_(noise.to(parameter.device, parameter.dtype))
    return model


def _onehot(rng, depth: int) -> np.ndarray:
    out = np.zeros((BATCH, SEQ_LEN, NUM_HEADS, depth), dtype=np.float32)
    tokens = rng.integers(0, depth, size=(BATCH, SEQ_LEN, NUM_HEADS))
    np.put_along_axis(out, tokens[..., None], 1.0, axis=-1)
    return out


def _inputs(config: Configuration, seed: int = 0) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    aa = _onehot(rng, config.hmm.alphabet_size)
    struct = _onehot(rng, config.structure.alphabet_size)
    indices = np.array([[0, 0], [3, 3], [7, 7]], dtype=np.int64)
    return aa, struct, indices


def _run(model: LearnMSAModel, inputs) -> torch.Tensor:
    with torch.no_grad():
        return model(
            tuple(torch.as_tensor(t).to(model.device) for t in inputs)
        )


@pytest.mark.parametrize("noise", ["background", "confusion"])
def test_matrix_is_row_stochastic(noise: str) -> None:
    for strength in (0.0, 0.3, 1.0):
        config = StructureConfig(
            observation_noise=noise, observation_noise_strength=strength
        )
        M = StructObservationLayer(config).matrix().numpy()
        np.testing.assert_allclose(M.sum(-1), 1.0, atol=1e-6)
        assert (M >= 0).all()
        if strength == 0.0:
            np.testing.assert_allclose(M, np.eye(20), atol=1e-7)


def test_shipped_confusion_matches_alphabet() -> None:
    C = load_confusion(StructureConfig().observation_confusion_name)
    assert C.shape == (20, 20)
    np.testing.assert_allclose(C.sum(-1), 1.0, atol=1e-5)
    # Predicted tokens are right more often than any single confusion.
    assert (np.diag(C) > 0.3).all()


def test_background_full_strength_is_uninformative() -> None:
    config = StructureConfig(
        observation_noise="background", observation_noise_strength=1.0
    )
    layer = StructObservationLayer(config)
    x = torch.eye(20)
    v = layer(x)
    # Each observation gives the same likelihood to every true token.
    torch.testing.assert_close(v, v[:, :1].expand(-1, 20))


def test_none_is_rejected() -> None:
    with pytest.raises(ValueError):
        StructObservationLayer(StructureConfig())


@pytest.mark.parametrize("use_anc_probs", [True, False])
@pytest.mark.parametrize("joint_emissions", [False, True])
@pytest.mark.parametrize("noise", ["background", "confusion"])
def test_zero_strength_reproduces_error_free_model(
    noise: str, use_anc_probs: bool, joint_emissions: bool
) -> None:
    base = _make_config(
        use_anc_probs=use_anc_probs, joint_emissions=joint_emissions
    )
    noisy = _make_config(
        noise, 0.0, use_anc_probs=use_anc_probs,
        joint_emissions=joint_emissions,
    )
    inputs = _inputs(base)
    torch.testing.assert_close(
        _run(_build(noisy), inputs), _run(_build(base), inputs),
        rtol=1e-6, atol=1e-6,
    )


@pytest.mark.parametrize("use_anc_probs", [True, False])
@pytest.mark.parametrize("joint_emissions", [False, True])
@pytest.mark.parametrize("noise", ["background", "confusion"])
def test_channel_equals_transformed_input(
    noise: str, use_anc_probs: bool, joint_emissions: bool
) -> None:
    """The layer acts once, on the struct track, before anc-probs."""
    noisy_config = _make_config(
        noise, 0.4, use_anc_probs=use_anc_probs,
        joint_emissions=joint_emissions,
    )
    noisy = _build(noisy_config)
    base = _build(_make_config(
        use_anc_probs=use_anc_probs, joint_emissions=joint_emissions
    ))
    aa, struct, indices = _inputs(noisy_config)
    M = noisy.struct_observation_layer.matrix().cpu().numpy()
    out_noisy = _run(noisy, (aa, struct, indices))
    out_manual = _run(base, (aa, struct @ M.T, indices))
    torch.testing.assert_close(out_noisy, out_manual, rtol=1e-5, atol=1e-5)
    assert torch.isfinite(out_noisy).all()
    # The channel must actually change the likelihood.
    assert not torch.allclose(out_noisy, _run(base, (aa, struct, indices)))


def test_trainable_strength_gets_gradient_and_is_carried() -> None:
    config = _make_config("confusion", 0.2, trainable=True)
    context = LearnMSAContext(config=config, num_seq=10)
    model = LearnMSAModel(context)
    model.build()
    model.loglik_mode()
    layer = model.struct_observation_layer
    assert layer.num_parameters() == 1
    assert abs(float(layer.strength().detach()) - 0.2) < 1e-6
    out = model(tuple(
        torch.as_tensor(t).to(model.device) for t in _inputs(config)
    ))
    out.sum().backward()
    assert layer.strength_logit.grad is not None
    assert torch.isfinite(layer.strength_logit.grad)

    # A later round starts from the carried value (as align.py sets it).
    context.struct_observation_strength = 0.37
    carried = LearnMSAModel(context)
    assert abs(float(carried.struct_observation_layer.strength().detach()) - 0.37) \
        < 1e-6


def test_fixed_strength_adds_no_parameters() -> None:
    config = _make_config("confusion", 0.2)
    model = _build(config)
    base = _build(_make_config())
    assert model.struct_observation_layer.num_parameters() == 0
    assert sum(p.numel() for p in model.parameters()) == \
        sum(p.numel() for p in base.parameters())
