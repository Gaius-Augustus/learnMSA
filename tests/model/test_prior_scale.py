"""The log prior of every head is divided by a per-head divisor
(``context.prior_scale``) that normalizes it by the dataset size, applies
``training.prior_scale`` and weakens the prior of heads trained on few
sequences."""

import os

import numpy as np
import pytest

from learnMSA import Configuration
from learnMSA.align.align_inserts import _insertion_config
from learnMSA.model.context import LearnMSAContext
from learnMSA.model.model import make_learnmsa_model
from learnMSA.util.multi_dataset import MultiSequenceDataset
from learnMSA.util.sequence_dataset import SequenceDataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FILES = [os.path.join(DATA_DIR, f)
         for f in ("felix.fa", "felix_insert_delete.fa")]  # 8 and 6 seqs


def _config(prior_scale: float = 1.0) -> Configuration:
    config = Configuration()
    config.training.no_sequence_weights = True
    config.training.prior_scale = prior_scale
    return config


@pytest.mark.parametrize("prior_scale", [1.0, 0.25])
def test_small_datasets_get_a_weaker_prior(prior_scale) -> None:
    context = LearnMSAContext(_config(prior_scale), SequenceDataset(FILES[0]))
    expected = 8 / (prior_scale * LearnMSAContext.PRIOR_DATA_FACTOR * 8)
    np.testing.assert_allclose(context.prior_scale, expected)


def test_large_datasets_keep_the_full_prior() -> None:
    n = int(np.ceil(1 / LearnMSAContext.PRIOR_DATA_FACTOR)) + 1
    config = _config()
    config.training.length_init = [5]
    context = LearnMSAContext(config, num_seq=n)
    np.testing.assert_allclose(context.prior_scale, n)


def test_every_head_is_scaled_by_its_own_dataset() -> None:
    context = LearnMSAContext(
        _config(), MultiSequenceDataset(filepaths=FILES)
    )
    n = np.array([8.0, 6.0])
    np.testing.assert_allclose(
        context.prior_scale, n / (LearnMSAContext.PRIOR_DATA_FACTOR * n)
    )


def test_prior_scale_zero_disables_the_prior() -> None:
    config = _config(0.0)
    config.training.length_init = [5]
    model = make_learnmsa_model(
        LearnMSAContext(config, SequenceDataset(FILES[0]))
    )
    model.build()
    log_prior = model.log_prior()
    log_prior = log_prior.detach().cpu() if hasattr(log_prior, "detach") \
        else log_prior
    np.testing.assert_array_equal(np.asarray(log_prior), 0.0)


def test_insertion_aligner_uses_the_prior_scale() -> None:
    config = Configuration()
    config.training.prior_scale = 0.5
    assert _insertion_config(config, [10]).training.prior_scale == 0.5


def test_prior_scale_must_not_be_negative() -> None:
    with pytest.raises(ValueError):
        Configuration(training={"prior_scale": -1.0})
