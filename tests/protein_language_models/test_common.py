import pytest

from learnMSA.protein_language_models.common import (SCORING_MODEL_PATH,
                                                     TF_ONLY_LANGUAGE_MODELS,
                                                     InputEncoder,
                                                     ScoringModelConfig,
                                                     dims,
                                                     get_language_model,
                                                     get_scoring_model_path,
                                                     make_cache_dir)


def test_scoring_model_config_defaults() -> None:
    config = ScoringModelConfig()
    assert config.lm_name == "protT5"
    assert config.dim == 16
    assert config.activation == "sigmoid"
    assert config.scaled is False
    assert config.suffix == ""


def test_scoring_model_config_is_hashable() -> None:
    """Frozen, so it can key a cache without surprises."""
    assert ScoringModelConfig() == ScoringModelConfig()
    assert len({ScoringModelConfig(), ScoringModelConfig()}) == 1


def test_get_scoring_model_path() -> None:
    config = ScoringModelConfig(lm_name="esm2", dim=64, activation="softmax")
    assert get_scoring_model_path(config) == (
        f"{SCORING_MODEL_PATH}/esm2_64_softmax.h5"
    )


def test_get_scoring_model_path_with_suffix() -> None:
    config = ScoringModelConfig(suffix="_v2")
    assert get_scoring_model_path(config).endswith("protT5_16_sigmoid_v2.h5")


def test_make_cache_dir_creates_the_root(tmp_path) -> None:
    root = tmp_path / "nested" / "cache"
    path = make_cache_dir(root, "esm2")
    assert root.is_dir()
    assert path == str(root / "esm2")


def test_unknown_language_model_raises() -> None:
    with pytest.raises(ValueError, match="not supported"):
        get_language_model("does-not-exist")


def test_tf_only_models_are_refused_under_torch() -> None:
    """ProteinBERT is Keras-only; asking for it under torch must say so."""
    from learnMSA.backend import get_backend

    if get_backend() == "tensorflow":
        pytest.skip("the TensorFlow backend supports every language model")

    for name in TF_ONLY_LANGUAGE_MODELS:
        with pytest.raises(NotImplementedError, match="tensorflow backend"):
            get_language_model(name)


def test_dims_cover_every_supported_model() -> None:
    assert set(dims) == {"proteinBERT", "esm2", "protT5", "zeros"}


def test_modify_cropped_shifts_start_cropped_sequences() -> None:
    """A sequence cropped at the start loses its start token."""
    import numpy as np

    class _Encoder(InputEncoder):
        def __call__(self, str_seq, crop):
            raise NotImplementedError

    ids = np.array([[0, 5, 6, 7, 2], [0, 5, 6, 2, 1]], dtype=np.int32)
    crop = np.array([[True, False], [False, True]])
    _Encoder().modify_cropped(ids, crop, lens=[3, 2], pad_id=1)

    # first sequence: rolled left, padded at the end
    np.testing.assert_array_equal(ids[0], [5, 6, 7, 2, 1])
    # second sequence: end token at index lens+1 overwritten with padding
    np.testing.assert_array_equal(ids[1], [0, 5, 6, 1, 1])
