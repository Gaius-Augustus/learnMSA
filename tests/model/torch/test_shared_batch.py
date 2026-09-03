"""Single-head inputs must give exactly the same result as per-model inputs.

``training.share_batch`` lets the batch generator emit a model
axis of size 1 instead of one column per trained model. Nothing downstream may
change as a result: the anc-probs layer broadcasts the tracks it consumes,
``torch.einsum`` broadcasts a labelled axis of size 1 for the emitters, and the
sequence indices are grown to the full head count so that every head keeps its
own evolutionary time. This test pins that contract for every combination of
input tracks, by comparing against the batch the generator would have built in
the per-model mode -- the same sequences, repeated across the model axis.
"""

import numpy as np
import pytest
import torch

from learnMSA.config import Configuration, TrainingConfig
from learnMSA.model.context import LearnMSAContext
from learnMSA.model.torch.model import TorchLearnMSAModel as LearnMSAModel

#: Two heads of different length, so a collapsed model axis cannot hide behind
#: two identical heads.
LENGTHS = [6, 4]
NUM_HEADS = len(LENGTHS)
BATCH, SEQ_LEN = 3, 7


def _make_config(
    use_structure: bool = False,
    use_language_model: bool = False,
    no_aa: bool = False,
    use_anc_probs: bool = True,
) -> Configuration:
    config = Configuration(
        training=TrainingConfig(length_init=LENGTHS, no_aa=no_aa)
    )
    config.structure.use_structure = use_structure
    config.language_model.use_language_model = use_language_model
    config.tree.use_anc_probs = use_anc_probs
    return config


#: (id, config kwargs). Covers every path through ``encode_batch``: with and
#: without the anc-probs layer, and with each extra track that bypasses it.
CASES = [
    ("aa", {}),
    ("aa+struct", {"use_structure": True}),
    ("aa+plm", {"use_language_model": True}),
    ("aa+struct+plm", {"use_structure": True, "use_language_model": True}),
    ("no_aa+struct", {"no_aa": True, "use_structure": True}),
    ("no_anc_probs", {"use_anc_probs": False}),
    ("no_anc_probs+plm", {"use_anc_probs": False, "use_language_model": True}),
]


def _random_onehot(rng, num_models: int, depth: int) -> np.ndarray:
    """A per-residue distribution track, one-hot as the datasets produce it."""
    out = np.zeros((BATCH, SEQ_LEN, num_models, depth), dtype=np.float32)
    tokens = rng.integers(0, depth, size=(BATCH, SEQ_LEN, num_models))
    np.put_along_axis(out, tokens[..., None], 1.0, axis=-1)
    return out


def _make_inputs(config: Configuration, rng) -> tuple[np.ndarray, ...]:
    """The single-head batch, in the track order the model expects."""
    tracks = [_random_onehot(rng, 1, config.hmm.alphabet_size)]
    if config.structure.use_structure:
        tracks.append(_random_onehot(rng, 1, config.structure.alphabet_size))
    if config.language_model.use_language_model:
        tracks.append(
            rng.normal(
                size=(BATCH, SEQ_LEN, 1, config.language_model.scoring_model_dim)
            ).astype(np.float32)
        )
    # Distinct rate indices, so a head reading the wrong row of tau shows up.
    tracks.append(np.array([[0], [3], [7]], dtype=np.int64))
    return tuple(tracks)


def _randomize_parameters(model: LearnMSAModel) -> None:
    """Break the symmetry a freshly initialized model has across its heads."""
    generator = torch.Generator(device="cpu").manual_seed(0)
    with torch.no_grad():
        for parameter in model.parameters():
            noise = torch.randn(
                parameter.shape, generator=generator, dtype=torch.float32
            )
            parameter.add_(noise.to(parameter.device, parameter.dtype))


@pytest.mark.parametrize(
    "kwargs", [c[1] for c in CASES], ids=[c[0] for c in CASES]
)
def test_single_head_input_matches_per_model_input(kwargs: dict) -> None:
    config = _make_config(**kwargs)
    context = LearnMSAContext(config=config, num_seq=10)
    model = LearnMSAModel(context)
    model.build()
    model.loglik_mode()

    # A freshly built model has identical parameters across heads and across
    # rate classes -- tau_kernel in particular is a single constant -- so an
    # untiled index array would go unnoticed. Perturb everything first, and
    # the comparison below actually depends on each head reading its own row.
    _randomize_parameters(model)

    rng = np.random.default_rng(0)
    shared = _make_inputs(config, rng)
    # What the batch generator emits without share_batch: the
    # very same sequences, one column per model.
    per_model = tuple(
        np.repeat(t, NUM_HEADS, axis=2 if t.ndim == 4 else 1) for t in shared
    )

    def run(inputs):
        with torch.no_grad():
            return model(
                tuple(torch.as_tensor(t).to(model.device) for t in inputs)
            )

    shared_out, per_model_out = run(shared), run(per_model)

    # A head axis that silently collapsed would show up here first.
    assert shared_out.shape == (BATCH, NUM_HEADS)
    torch.testing.assert_close(shared_out, per_model_out, rtol=1e-6, atol=1e-6)

    # The heads must still differ -- they have different lengths, so equal
    # log-likelihoods would mean the head axis carries a single model.
    assert not torch.allclose(shared_out[:, 0], shared_out[:, 1])
