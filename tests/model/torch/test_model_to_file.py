"""Round-trips a PyTorch learnMSA model through a checkpoint file.

The counterpart of ``tests/model/tf/test_model_to_file.py``. A ``.pt``
checkpoint has to carry enough to rebuild the module -- the state dict alone
does not say what shape of model the tensors belong to -- so this checks that
the reloaded model has the same parameters *and* computes the same thing.
"""

import numpy as np
import pytest
import torch

import tests.hmm.ref as ref
from learnMSA.config import Configuration, TrainingConfig, TreeConfig
from learnMSA.config.hmm import PHMMPriorConfig
from learnMSA.model.checkpoint import checkpoint_format
from learnMSA.model.context import LearnMSAContext
from learnMSA.model.torch.checkpoint import (SUFFIX, load_model,
                                             save_model)
from learnMSA.model.torch.model import TorchLearnMSAModel
from learnMSA.util.sequence_dataset import SequenceDataset


@pytest.fixture
def model() -> TorchLearnMSAModel:
    hmm_config = ref.config.model_copy(deep=True)
    hmm_config.use_prior_for_emission_init = False
    config = Configuration(
        training=TrainingConfig(length_init=[4, 3]),
        tree=TreeConfig(use_anc_probs=False),
        hmm=hmm_config,
        hmm_prior=PHMMPriorConfig(use_amino_acid_prior=False),
    )
    context = LearnMSAContext(
        config=config,
        num_seq=10,
        sequence_weights=np.arange(10, dtype=float),
    )
    model = TorchLearnMSAModel(context)
    model.build()
    model.compile()
    return model


@pytest.fixture
def data() -> SequenceDataset:
    return SequenceDataset(
        sequences=[(str(i), "ABA") for i in range(6)], alphabet="AB"
    )


def test_checkpoint_format_is_pt() -> None:
    assert checkpoint_format("pytorch") == "pt"


def test_round_trip_preserves_predictions(
    model: TorchLearnMSAModel, data: SequenceDataset, tmp_path
) -> None:
    model.loglik_mode()
    before = model.predict(data)

    path = tmp_path / "model"
    save_model(model, path)
    assert (tmp_path / ("model" + SUFFIX)).exists()

    loaded = load_model(path)
    loaded.loglik_mode()
    after = loaded.predict(data)

    np.testing.assert_allclose(after, before, rtol=1e-6, atol=1e-6)


def test_round_trip_preserves_weights(
    model: TorchLearnMSAModel, data: SequenceDataset, tmp_path
) -> None:
    # Train first, so the parameters differ from their initial values and a
    # checkpoint that silently reinitialised them would be caught.
    model.fit(data, batch_size=3, epochs=1, steps_per_epoch=2)

    path = tmp_path / "model"
    save_model(model, path)
    loaded = load_model(path)

    original = model.phmm_layer.get_weights()
    restored = loaded.phmm_layer.get_weights()
    assert len(original) == len(restored)
    for a, b in zip(original, restored):
        np.testing.assert_allclose(b, a, rtol=1e-6, atol=1e-6)


def test_round_trip_preserves_context(
    model: TorchLearnMSAModel, tmp_path
) -> None:
    path = tmp_path / "model"
    save_model(model, path)
    loaded = load_model(path)

    assert list(loaded.context.model_lengths) == list(
        model.context.model_lengths
    )
    assert loaded.context.config.hmm.alphabet == \
        model.context.config.hmm.alphabet


def test_refuses_a_foreign_checkpoint_format(
    model: TorchLearnMSAModel, tmp_path
) -> None:
    """A checkpoint written by an incompatible version must be rejected with a
    clear message rather than loaded into a mismatched module."""
    path = tmp_path / "model"
    save_model(model, path)

    checkpoint = torch.load(
        str(path) + SUFFIX, map_location="cpu", weights_only=False
    )
    checkpoint["format_version"] = -1
    torch.save(checkpoint, str(path) + SUFFIX)

    with pytest.raises(ValueError, match="checkpoint format"):
        load_model(path)
