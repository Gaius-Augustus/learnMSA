"""Tests for AlignmentModel.posterior_to_file (--posterior_file)."""

import numpy as np
import pytest

from learnMSA.align.alignment_model import (AlignmentModel,
                                            read_posterior_file)
from tests.align.test_alignment_model import (_duplicate_models,
                                              felix_with_duplicates)

MODES = [AlignmentModel.DecodingMode.VITERBI, AlignmentModel.DecodingMode.MEA]


def _split(matrix: np.ndarray, lens: np.ndarray) -> list[np.ndarray]:
    return np.split(matrix, np.cumsum(lens)[:-1])


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("head", [0, 1])
@pytest.mark.parametrize("data_name", ["simple_data", "multi_hit_data"])
def test_posterior_file(request, simple_model, mode, head, data_name, tmp_path):
    data = request.getfixturevalue(data_name)
    n = data.num_seq
    lens = data.seq_lens[:n]
    L = simple_model.context.model_lengths[head]

    am = AlignmentModel(data, simple_model)
    states_ref = am.get_batch_states(head, np.arange(n), mode)
    strings_ref = am.states_to_string(head, mode)

    # Full posteriors
    full_path = tmp_path / "post.npy"
    am.posterior_to_file(full_path, head, mode, full=True)
    full = np.load(full_path)
    assert full.shape == (lens.sum(), 2 * L + 2)
    np.testing.assert_allclose(full.sum(-1), 1.0, atol=1e-5)
    assert am.states_to_string(head, mode) == strings_ref

    # Same as the plain posterior mode (no structure, so no temperature)
    simple_model.posterior_mode()
    post = simple_model.predict(data, models=[head])[:, :, 0, :-1]
    for i, p in enumerate(_split(full, lens)):
        np.testing.assert_allclose(p, post[i, :lens[i]], atol=1e-5)

    # Decoded: posterior of the state the decoder chose
    text_path = tmp_path / "post.txt"
    am.posterior_to_file(text_path, head, mode, batch_size=3)
    assert am.states_to_string(head, mode) == strings_ref
    headers, values = read_posterior_file(text_path)
    assert headers == [data.get_header(i) for i in range(n)]
    for i, p in enumerate(_split(full, lens)):
        expected = p[np.arange(lens[i]), states_ref[i, :lens[i]]]
        np.testing.assert_allclose(values[i], expected, atol=6e-5)


def test_mea_posterior_computed_once(simple_data, simple_model, tmp_path):
    """MEA reuses its posteriors: no extra forward-backward pass."""
    hmm = simple_model.phmm_layer.hmm
    calls = []
    original = hmm.posterior

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    hmm.posterior = counting
    try:
        am = AlignmentModel(simple_data, simple_model)
        am.build_alignment([0], AlignmentModel.DecodingMode.MEA)
        without = len(calls)
        calls.clear()
        am.posterior_to_file(
            tmp_path / "p.txt", 0, AlignmentModel.DecodingMode.MEA
        )
        assert len(calls) == without > 0
    finally:
        del hmm.posterior


def test_posterior_duplicates(simple_config, felix_with_duplicates, tmp_path):
    full, dedup = _duplicate_models(simple_config, felix_with_duplicates)
    for full_out in (False, True):
        paths = [tmp_path / f"{name}_{full_out}" for name in ("a", "b")]
        for am, path in zip((full, dedup), paths):
            am.posterior_to_file(path, 0, full=full_out, batch_size=3)
        assert paths[0].read_bytes() == paths[1].read_bytes()


@pytest.mark.parametrize("mode", MODES)
def test_posterior_npy(multi_hit_data, simple_model, mode, tmp_path):
    """A .npy path writes the decoded-state posteriors as one float32 array,
    records in the order and lengths of the text file."""
    am = AlignmentModel(multi_hit_data, simple_model)
    am.posterior_to_file(tmp_path / "p.txt", 0, mode)
    am.posterior_to_file(tmp_path / "p.npy", 0, mode)
    _, values = read_posterior_file(tmp_path / "p.txt")
    arr = np.load(tmp_path / "p.npy")
    assert arr.dtype == np.float32 and arr.ndim == 1
    assert arr.size == sum(len(v) for v in values)
    np.testing.assert_allclose(arr, np.concatenate(values), atol=6e-5)


def test_posterior_npy_duplicates(
    simple_config, felix_with_duplicates, tmp_path
):
    full, dedup = _duplicate_models(simple_config, felix_with_duplicates)
    paths = [tmp_path / f"{name}.npy" for name in ("a", "b")]
    for am, path in zip((full, dedup), paths):
        am.posterior_to_file(path, 0)
    np.testing.assert_array_equal(np.load(paths[0]), np.load(paths[1]))
