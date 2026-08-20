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

from learnMSA.model.training_util import IMPL_FACTORS, MODE_FALLBACK
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
        workload="train",
        batch_size=2,
        num_model=2,
        model_len=5,
        seq_len=8,
        steps=1,
        backend="pytorch",
        compile_mode="off",
        use_triton=False,
        features=features,
    )
    defaults.update(kwargs)
    return calib.ProbeSpec(**defaults)


@pytest.mark.parametrize("features", sorted(calib.FEATURES))
@pytest.mark.parametrize("workload", ["train", "inference"])
def test_factor_key_names_a_real_impl_factor(features, workload):
    """Every sweep must report under a key that IMPL_FACTORS actually has."""
    key = calib.factor_key(features, workload)
    for factors in IMPL_FACTORS.values():
        assert key in factors


def test_factor_key_rejects_unknown_features():
    with pytest.raises(ValueError):
        calib.factor_key("3di", "train")


def test_every_impl_factor_key_is_reachable():
    """No key may be unmeasurable: each one must be some sweep's output.

    ``inference`` is reachable too: it is not a workload, but every sweep
    derives it as the maximum over the inference workloads it ran.
    """
    reachable = {
        calib.factor_key(features, workload)
        for features in calib.FEATURES
        for workload in (*calib.WORKLOADS, "inference")
    }
    assert set(IMPL_FACTORS["pytorch"]) <= reachable


def test_mea_is_not_a_workload():
    """It borrows the posterior factor instead of being measured."""
    assert "mea" not in calib.WORKLOADS
    assert MODE_FALLBACK["mea"] == "posterior"


def test_inference_fallback_is_the_worst_measured_mode():
    """A fallback that underestimates would hand out a batch that will not fit."""
    results = [
        calib.ProbeResult(
            spec=make_spec("aa", workload=workload),
            status="ok",
            effective_batch_size=2,
            impl_factor=factor,
            resulting_batch_size=1,
            batch_size_cap=1_000,
            cap_bound=False,
        )
        for workload, factor in [
            ("train", 30.0), ("viterbi", 11.0), ("posterior", 14.0),
            ("loglik", 6.0),
        ]
    ]
    aggregate = calib.derive_inference_factor(results, calib.WORKLOADS, "aa")
    assert aggregate == 14.0  # the posterior, not the train factor


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


@pytest.mark.parametrize("fits_up_to", range(-1, 7))
def test_binary_search_finds_the_same_winner_as_an_exhaustive_ladder(
    monkeypatch, fits_up_to
):
    """The bisection is only legitimate if it never changes the answer.

    ``fits_up_to`` is the index of the largest rung that fits, -1 meaning the
    workload does not fit at any batch size.
    """
    ladder = [
        make_spec("aa", batch_size=b) for b in calib.DEFAULT_BATCH_SIZES
    ]
    probed: list[int] = []

    def fake_run_probe(spec, timeout, verbose=False):
        index = list(calib.DEFAULT_BATCH_SIZES).index(spec.batch_size)
        probed.append(index)
        ok = index <= fits_up_to
        return calib.ProbeResult(
            spec=spec,
            status="ok" if ok else "oom",
            effective_batch_size=spec.batch_size if ok else 0,
            impl_factor=1.0,
            batch_size_cap=1_000,
            cap_bound=False,
        )

    monkeypatch.setattr(calib, "run_probe", fake_run_probe)
    results = calib.search_group(ladder, timeout=1.0)

    exhaustive = (
        calib.DEFAULT_BATCH_SIZES[fits_up_to] if fits_up_to >= 0 else None
    )
    winners = [r for r in results if r.status == "ok"]
    best = max((r.effective_batch_size for r in winners), default=None)
    assert best == exhaustive

    # ...and it must get there in ceil(log2(n + 1)) probes, not n.
    import math
    assert len(probed) <= math.ceil(math.log2(len(ladder) + 1))


def test_binary_search_selection_matches_select_probes(monkeypatch):
    """select_probes must still name the bisection's winner."""
    ladder = [
        make_spec("aa", batch_size=b) for b in calib.DEFAULT_BATCH_SIZES
    ]

    def fake_run_probe(spec, timeout, verbose=False):
        ok = spec.batch_size <= 64
        return calib.ProbeResult(
            spec=spec,
            status="ok" if ok else "oom",
            effective_batch_size=spec.batch_size if ok else 0,
            impl_factor=1.0,
            batch_size_cap=1_000,
            cap_bound=False,
        )

    monkeypatch.setattr(calib, "run_probe", fake_run_probe)
    results = calib.search_group(ladder, timeout=1.0)
    selected = calib.select_probes(results, "train", "aa")
    assert [r.effective_batch_size for r in selected] == [64]


def test_workloads_split_the_groups():
    """Two workloads at one shape must never be aggregated together."""
    specs = [make_spec("aa", workload=w) for w in calib.WORKLOADS]
    results = [
        calib.ProbeResult(
            spec=spec, status="ok", effective_batch_size=2,
            impl_factor=float(i + 1), resulting_batch_size=1,
            batch_size_cap=1_000, cap_bound=False,
        )
        for i, spec in enumerate(specs)
    ]
    factors = {
        w: calib.derive_factor(results, w, "aa") for w in calib.WORKLOADS
    }
    assert factors == {w: float(i + 1) for i, w in enumerate(calib.WORKLOADS)}


def test_probe_phase_follows_the_workload():
    """Only the clamps care about the phase, and only gradients matter there."""
    assert make_spec("aa", workload="train").phase == "train"
    for workload in calib.INFERENCE_WORKLOADS:
        assert make_spec("aa", workload=workload).phase == "inference"
