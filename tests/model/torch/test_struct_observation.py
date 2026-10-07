"""Per-residue 3Di logits as structural input (StructLogitObservationLayer)."""

import numpy as np
import pytest
import torch

from learnMSA.config import Configuration, TrainingConfig
from learnMSA.config.structure import StructureConfig
from learnMSA.model.context import LearnMSAContext
from learnMSA.model.torch.model import TorchLearnMSAModel as LearnMSAModel

LENGTHS = [6, 4]
NUM_HEADS = len(LENGTHS)
BATCH, SEQ_LEN = 3, 7


def _make_config(
    use_anc_probs: bool = True,
    joint_emissions: bool = False,
) -> Configuration:
    config = Configuration(training=TrainingConfig(length_init=LENGTHS))
    config.structure.use_structure = True
    config.structure.joint_emissions = joint_emissions
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


def _logit_config(use_anc_probs: bool = True, temperature: float = 1.0,
                  joint_emissions: bool = False) -> Configuration:
    config = _make_config(use_anc_probs, joint_emissions)
    config.structure.input_format = "logits"
    config.structure.soft_input_temperature = temperature
    return config


def _manual_v(z: np.ndarray, temperature: float) -> np.ndarray:
    zt = z / temperature
    post = np.exp(zt - zt.max(-1, keepdims=True))
    post /= post.sum(-1, keepdims=True)
    pi = np.asarray(StructureConfig().background_distribution, float)
    score = np.log(post) - np.log(pi / pi.sum())
    return np.exp(score - score.max(-1, keepdims=True))


@pytest.mark.parametrize("joint_emissions", [False, True])
@pytest.mark.parametrize("use_anc_probs", [True, False])
@pytest.mark.parametrize("temperature", [1.0, 1.5])
def test_logits_equal_the_transformed_token_input(
    temperature, use_anc_probs, joint_emissions
) -> None:
    """The logit layer acts once on the struct track, before anc-probs; the
    rest of the model sees v exactly like a (soft) token input."""
    logit_model = _build(_logit_config(use_anc_probs, temperature,
                                       joint_emissions))
    token_model = _build(_make_config(use_anc_probs, joint_emissions))
    aa, _, indices = _inputs(_make_config())
    z = np.random.default_rng(5).normal(
        scale=3.0, size=(BATCH, SEQ_LEN, NUM_HEADS, 20)).astype(np.float32)
    v = _manual_v(z, temperature).astype(np.float32)
    out_logits = _run(logit_model, (aa, z, indices))
    out_tokens = _run(token_model, (aa, v, indices))
    torch.testing.assert_close(out_logits, out_tokens, rtol=1e-5, atol=1e-4)
    assert torch.isfinite(out_logits).all()
