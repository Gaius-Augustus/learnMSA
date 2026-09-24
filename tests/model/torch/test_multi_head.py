"""One model per dataset: joint prediction over per-model index tables.

A 2-D index table gives every head its own sequences in one pass. Its
predictions must equal those of separate passes with ``models=[k]``, and the
log prior of each head must be normalized by its own dataset.
"""

import os

import numpy as np
import pytest
import torch

from learnMSA.config import Configuration
from learnMSA.model.batch_generator import (get_index_table,
                                            sort_index_table)
from learnMSA.model.context import LearnMSAContext
from learnMSA.model.torch.model import TorchLearnMSAModel as LearnMSAModel
from learnMSA.util.multi_dataset import MultiSequenceDataset

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")


@pytest.fixture
def multi_model() -> tuple[MultiSequenceDataset, LearnMSAModel]:
    data = MultiSequenceDataset(filepaths=[
        os.path.join(DATA, "felix.fa"),
        os.path.join(DATA, "felix_insert_delete.fa"),
    ])
    config = Configuration()
    config.training.no_sequence_weights = True
    config.training.length_init = [5, 4]
    context = LearnMSAContext(config, data)
    model = LearnMSAModel(context)
    model.build()
    return data, model


def _head_indices(data: MultiSequenceDataset) -> list[np.ndarray]:
    return [data.global_indices(k) for k in range(data.num_datasets)]


def test_log_prior_is_normalized_per_head(multi_model) -> None:
    _, model = multi_model
    np.testing.assert_equal(model.context.prior_scale, [8, 6])
    with torch.no_grad():
        np.testing.assert_allclose(
            model.log_prior().cpu().numpy(),
            model.phmm_layer.prior_scores().cpu().numpy() / [8, 6],
            rtol=1e-6,
        )


def test_table_loglik_equals_per_head_passes(multi_model) -> None:
    data, model = multi_model
    heads = _head_indices(data)
    table, positions = sort_index_table(get_index_table(heads), data.seq_lens)

    model.loglik_mode()
    joint = model.predict(data, indices=table)
    assert joint.shape == (table.shape[0], 2)
    for k, idx in enumerate(heads):
        single = model.predict(data, indices=idx, models=[k])[:, 0]
        filled = positions[:, k] >= 0
        np.testing.assert_allclose(
            joint[filled, k], single[positions[filled, k]], rtol=1e-5
        )


def test_table_posterior_reduce_equals_per_head_passes(multi_model) -> None:
    data, model = multi_model
    heads = _head_indices(data)
    table, _ = sort_index_table(get_index_table(heads), data.seq_lens)

    model.posterior_mode()
    joint = model.predict(data, indices=table, reduce=True)
    for k, idx in enumerate(heads):
        single = model.predict(data, indices=idx, models=[k], reduce=True)
        q = single.shape[1]
        np.testing.assert_allclose(joint[k, :q], single[0], atol=1e-5)
        np.testing.assert_allclose(joint[k, q:], 0, atol=1e-6)


def test_table_viterbi_on_a_head_subset_without_multi_hits(
    multi_model,
) -> None:
    """A subset of several heads (here reordered) in single-hit mode, as the
    greedy_single re-run of MultiAlignmentModel uses it."""
    data, model = multi_model
    heads = _head_indices(data)[::-1]
    models = [1, 0]
    table, positions = sort_index_table(get_index_table(heads), data.seq_lens)

    model.phmm_layer.enable_multi_hits(False)
    model.viterbi_mode()
    joint = model.predict(data, indices=table, models=models)
    for c, (k, idx) in enumerate(zip(models, heads)):
        single = model.predict(data, indices=idx, models=[k])
        filled = positions[:, c] >= 0
        rows = positions[filled, c]
        L = single.shape[1]
        np.testing.assert_equal(joint[filled, :L, c], single[rows, :, 0])
    model.phmm_layer.enable_multi_hits(True)
