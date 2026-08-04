"""End-to-end alignment quality for the PyTorch backend.

The counterpart of ``test_alignment_egf`` in ``tests/test_alignment.py``. The
parity fixtures pin the layer outputs, but only a full run says whether the
backend actually produces a usable alignment: training, model surgery, model
selection, Viterbi decoding and insertion alignment all have to work together.

The gate is the sum-of-pairs score against the curated reference alignment,
with the same threshold the TensorFlow test uses.
"""

import os

import numpy as np
import pytest

pytest.importorskip("torch")

from learnMSA.align.align import align  # noqa: E402
from learnMSA.config import Configuration  # noqa: E402
from learnMSA.util.aligned_dataset import AlignedDataset  # noqa: E402
from learnMSA.util.sequence_dataset import SequenceDataset  # noqa: E402

pytestmark = pytest.mark.torch

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def test_alignment_egf(tmp_path) -> None:
    """Align the EGF family and score it against the reference."""
    egf_fasta_path = os.path.join(DATA, "egf.fasta")
    egf_ref_path = os.path.join(DATA, "egf.ref")
    egf_out_path = str(tmp_path / "egf.out.fasta")

    with SequenceDataset(egf_fasta_path) as data:
        with AlignedDataset(egf_ref_path) as ref_msa:
            seq_ids = ref_msa.seq_ids

        config = Configuration()
        config.training.num_model = 1
        config.training.no_sequence_weights = True
        config.training.epochs = [5, 1, 5]
        config.training.max_iterations = 2
        config.training.length_init = [25]
        config.input_output.subset_ids = seq_ids
        config.training.crop = 999999
        config.training.auto_crop = False

        # Fit the alignment model
        am = align(data, config)
        am.select_best()

        # Evaluate the model
        eval_output = am.model.evaluate(data, models=[am.best_head])

    # Check some friendly thresholds to check if the alignment makes sense
    assert np.amin(eval_output["loglik"].mean()) > -70
    # Surgery should have added match states
    assert am.model.lengths[am.best_head] > 25

    am.to_file(egf_out_path, 0)
    with AlignedDataset(egf_out_path) as pred_msa:
        sp = pred_msa.SP_score(ref_msa)
        # The same threshold the TensorFlow test asserts: any half decent
        # hyperparameter choice should reach it.
        assert sp > 0.7, f"SP score {sp:.3f} is below the 0.7 threshold"
