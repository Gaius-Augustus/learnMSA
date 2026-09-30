"""Round-trips a PyTorch learnMSA model through a checkpoint file.

The counterpart of ``tests/model/tf/test_model_to_file.py``. A ``.pt``
checkpoint has to carry enough to rebuild the module -- the state dict alone
does not say what shape of model the tensors belong to -- so this checks that
the reloaded model has the same parameters *and* computes the same thing.
"""

import numpy as np
import pytest
import torch
from hidten.torch.triton import step_launch

import tests.hmm.ref as ref
from learnMSA.align.alignment_model import AlignmentModel
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


@pytest.fixture
def restore_triton_thresholds():
    """A model built with ``use_triton`` sets hidten's process-wide
    step-launch thresholds; put them back for the other tests."""
    saved = (step_launch.AUTO_MIN_Q, step_launch.AUTO_MIN_Q_UNALIGNED)
    yield
    step_launch.set_auto_min_q(*saved)


def test_runtime_settings_come_from_the_current_run(
    model: TorchLearnMSAModel, tmp_path, restore_triton_thresholds
) -> None:
    """``--compile`` and ``--triton`` say how *this* run executes, so the
    values a checkpoint was trained with must not carry over."""
    model.context.config.advanced.use_triton = False
    model.context.config.advanced.compile = "off"
    path = tmp_path / "model"
    save_model(model, path)

    # Without a run config, the checkpoint's own settings stand.
    kept = load_model(path)
    assert kept.context.config.advanced.use_triton is False
    assert kept.context.config.advanced.compile == "off"
    assert kept.phmm_layer.use_triton is False

    run_config = model.context.config.model_copy(deep=True)
    run_config.advanced.use_triton = True
    run_config.advanced.compile = "on"
    loaded = load_model(path, run_config)

    assert loaded.context.config.advanced.use_triton is True
    assert loaded.context.config.advanced.compile == "on"
    assert loaded.phmm_layer.use_triton == "auto"
    # The rest of the checkpoint's configuration is untouched.
    assert list(loaded.context.model_lengths) == list(
        model.context.model_lengths
    )


@pytest.mark.parametrize("noise", ["none", "confusion"])
def test_decoding_temperature_comes_from_the_current_run(
    noise: str, tmp_path
) -> None:
    """The structural emitter temperature only changes decoding, so a loaded
    model is decoded with the current run's ``--struct_emitter_temperature``.
    """
    config = Configuration(training=TrainingConfig(length_init=[4, 3]))
    config.structure.use_structure = True
    config.structure.observation_noise = noise
    config.structure.observation_noise_strength = 0.5
    model = TorchLearnMSAModel(LearnMSAContext(config=config, num_seq=10))
    model.build()
    path = tmp_path / "model"
    save_model(model, path)
    trained_temperature = config.structure.emitter_temperature

    # Without a run config, the checkpoint's own temperature stands.
    kept = load_model(path)
    kept.viterbi_mode()
    assert kept.phmm_layer.struct_emitter.temperature == trained_temperature

    run_config = config.model_copy(deep=True)
    run_config.structure.emitter_temperature = 2.0
    loaded = load_model(path, run_config)
    assert loaded.context.config.structure.emitter_temperature == 2.0
    loaded.viterbi_mode()
    assert loaded.phmm_layer.struct_emitter.temperature == 2.0
    # Likelihoods are still computed at temperature 1.
    loaded.loglik_mode()
    assert loaded.phmm_layer.struct_emitter.temperature == 1

    if noise == "none":
        assert loaded.struct_observation_layer is None
    else:
        torch.testing.assert_close(
            loaded.struct_observation_layer.matrix(),
            model.struct_observation_layer.matrix().to(
                loaded.struct_observation_layer.matrix().device
            ),
        )


def test_loading_writes_only_into_the_current_work_dir(tmp_path) -> None:
    """A saved model may belong to another run, so loading must not write
    next to it; scratch files go to the current run's work dir and are
    removed again."""
    config = Configuration()
    config.training.num_model = 1
    config.training.no_sequence_weights = True
    config.training.length_init = [5]
    data = SequenceDataset("tests/data/simple.fa")
    model = TorchLearnMSAModel(LearnMSAContext(config, data))
    model.build()
    am = AlignmentModel(data, model, np.array([0, 1]))
    am.best_head = 0
    saved = tmp_path / "other_run"
    saved.mkdir()
    am.save(saved / "model")
    before = sorted(p.name for p in saved.iterdir())

    run_config = config.model_copy(deep=True)
    run_config.input_output.work_dir = str(tmp_path / "this_run")
    saved.chmod(0o555)  # any write next to the archive now fails
    try:
        loaded = AlignmentModel.load(saved / "model", data, config=run_config)
    finally:
        saved.chmod(0o755)

    assert sorted(p.name for p in saved.iterdir()) == before
    assert list((tmp_path / "this_run").iterdir()) == []
    assert loaded.best_head == 0
    np.testing.assert_array_equal(loaded.indices, [0, 1])


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
