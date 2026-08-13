"""Framework-free parts of ``util/calibrate_impl_factor.py``.

The script is not importable as a package member, so it is loaded from its path
here. Only the parent-side helpers and the probe workload construction are
covered -- running a probe needs a GPU and is not a unit test.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from learnMSA.model.training_util import IMPL_FACTORS
from learnMSA.util import EmbeddingDataset, SequenceDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "util" / "calibrate_impl_factor.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "calibrate_impl_factor", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


calib = _load_script()


def make_spec(features: str, **kwargs):
    """A ProbeSpec small enough to build a dataset for in milliseconds."""
    defaults = dict(
        phase="train",
        batch_size=2,
        num_model=2,
        model_len=5,
        seq_len=8,
        steps=1,
        backend="pytorch",
        compile_mode="off",
        no_triton=True,
        features=features,
    )
    defaults.update(kwargs)
    return calib.ProbeSpec(**defaults)


@pytest.mark.parametrize("features", sorted(calib.FEATURES))
@pytest.mark.parametrize("phase", ["train", "inference"])
def test_factor_key_names_a_real_impl_factor(features, phase):
    """Every sweep must report under a key that IMPL_FACTORS actually has."""
    key = calib.factor_key(features, phase)
    for factors in IMPL_FACTORS.values():
        assert key in factors


def test_factor_key_rejects_unknown_features():
    with pytest.raises(ValueError):
        calib.factor_key("3di", "train")


def test_every_impl_factor_key_is_reachable():
    """No key may be unmeasurable: each one must be some sweep's output."""
    reachable = {
        calib.factor_key(features, phase)
        for features in calib.FEATURES
        for phase in ("train", "inference")
    }
    assert reachable == set(IMPL_FACTORS["pytorch"])


@pytest.mark.parametrize("features", sorted(calib.FEATURES))
def test_probe_datasets_have_the_tracks_the_config_switches_on(features):
    spec = make_spec(features)
    config = calib.make_probe_config(spec)
    use_structure, use_language_model, _ = calib.FEATURES[features]

    assert config.structure.use_structure is use_structure
    assert config.language_model.use_language_model is use_language_model

    datasets = calib.make_probe_datasets(spec, config)
    assert len(datasets) == 1 + use_structure + use_language_model
    assert isinstance(datasets[0], SequenceDataset)
    if use_structure:
        assert isinstance(datasets[1], SequenceDataset)
    if use_language_model:
        assert isinstance(datasets[-1], EmbeddingDataset)


@pytest.mark.parametrize("features", sorted(calib.FEATURES))
def test_probe_tracks_agree_on_lengths_and_ids(features):
    """The batch generator crops every track by data[0]'s lengths."""
    spec = make_spec(features)
    datasets = calib.make_probe_datasets(spec, calib.make_probe_config(spec))

    for dataset in datasets[1:]:
        assert list(dataset.seq_ids) == list(datasets[0].seq_ids)
        np.testing.assert_array_equal(dataset.seq_lens, datasets[0].seq_lens)


def test_probe_sequences_all_pad_to_seq_len():
    """Uniform lengths are what makes the padded batch shape predictable."""
    spec = make_spec("both", seq_len=13)
    datasets = calib.make_probe_datasets(spec, calib.make_probe_config(spec))
    assert set(datasets[0].seq_lens) == {spec.seq_len - 1}


def test_probe_embeddings_are_the_width_the_model_expects():
    spec = make_spec("language_model")
    config = calib.make_probe_config(spec)
    emb = calib.make_probe_datasets(spec, config)[-1]

    dim = config.language_model.scoring_model_dim
    encoded = emb.get_encoded_seq(0, 0, spec.seq_len - 1)
    assert encoded.shape == (spec.seq_len - 1, dim)
    assert emb.empty((3, spec.seq_len)).shape == (3, spec.seq_len, dim)


def test_probe_struct_track_is_encoded_over_the_3di_alphabet():
    spec = make_spec("structure")
    config = calib.make_probe_config(spec)
    struct = calib.make_probe_datasets(spec, config)[1]

    encoded = struct.get_encoded_seq(0, 0, spec.seq_len - 1)
    assert encoded.shape == (spec.seq_len - 1, config.structure.alphabet_size)


def test_labels_separate_the_feature_sets():
    """Otherwise probes of different track combinations would be aggregated."""
    labels = {
        make_spec(features).label for features in calib.FEATURES
    }
    assert len(labels) == len(calib.FEATURES)


def test_derive_factor_only_sees_its_own_feature_set():
    results = [
        calib.ProbeResult(
            spec=make_spec(features),
            status="ok",
            effective_batch_size=2,
            impl_factor=factor,
            resulting_batch_size=1,
            batch_size_cap=1_000,
            cap_bound=False,
        )
        for features, factor in [("aa", 10.0), ("both", 90.0)]
    ]
    assert calib.derive_factor(results, "train", "aa") == 10.0
    assert calib.derive_factor(results, "train", "both") == 90.0
    assert calib.derive_factor(results, "train", "structure") is None
