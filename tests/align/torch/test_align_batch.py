import os

import numpy as np
import pytest

from learnMSA import Configuration
from learnMSA.align.align import align_batch
from learnMSA.align.align_hits import HitAlignmentMode
from learnMSA.align.alignment_model import AlignmentModel
from learnMSA.align.multi_alignment_model import MultiAlignmentModel
from learnMSA.model.context import LearnMSAContext
from learnMSA.model.model import make_learnmsa_model
from learnMSA.util.aligned_dataset import AlignedDataset
from learnMSA.util.multi_dataset import MultiSequenceDataset

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")
SMALL = [os.path.join(DATA, f) for f in ("felix.fa", "felix_insert_delete.fa")]


def _config(work_dir) -> Configuration:
    config = Configuration()
    config.training.no_sequence_weights = True
    config.training.epochs = [5, 1, 5]
    config.training.max_iterations = 2
    config.input_output.work_dir = str(work_dir)
    config.input_output.verbose = False
    return config


def test_align_batch_egf_and_bowman(tmp_path) -> None:
    names = ["egf", "bowman"]
    refs = [AlignedDataset(os.path.join(DATA, f"{n}.ref")) for n in names]
    data = MultiSequenceDataset(
        filepaths=[os.path.join(DATA, f"{n}.fasta") for n in names]
    )
    config = _config(tmp_path)
    config.input_output.subset_ids = [s for r in refs for s in r.seq_ids]

    am = align_batch(data, config)

    assert isinstance(am, MultiAlignmentModel)
    assert am.model.heads == 2
    # Surgery should have added match states to the egf model
    assert am.model.lengths[0] > 25
    for k, ref in enumerate(refs):
        np.testing.assert_equal(
            np.sort(am.head_indices[k]),
            np.sort([data.seq_ids.index(s) for s in ref.seq_ids]),
        )
    am.build_alignment()
    # based on experience, any half decent hyperparameter choice should
    # yield at least these scores (bowman is small and noisy)
    thresholds = [0.7, 0.4]
    for k, (name, ref) in enumerate(zip(names, refs)):
        path = am.to_file(tmp_path / f"{name}.out.fasta", k)
        with AlignedDataset(path) as pred_msa:
            assert set(pred_msa.seq_ids) == set(ref.seq_ids)
            assert pred_msa.SP_score(ref) > thresholds[k]


@pytest.mark.parametrize(
    "mode", [HitAlignmentMode.GREEDY_SINGLE, HitAlignmentMode.GREEDY_SCORES]
)
def test_joint_decoding_equals_per_head_decoding(mode) -> None:
    data = MultiSequenceDataset(filepaths=SMALL)
    config = Configuration()
    config.training.no_sequence_weights = True
    config.training.length_init = [5, 4]
    context = LearnMSAContext(config, data)
    model = make_learnmsa_model(context)
    model.build()
    head_indices = [data.global_indices(k) for k in range(2)]

    am = MultiAlignmentModel(
        data, model, head_indices, hit_alignment_mode=mode
    )
    am.build_alignment()
    for k, indices in enumerate(head_indices):
        single = AlignmentModel(
            data, model, indices, best_head=k, hit_alignment_mode=mode
        )
        assert am.to_string(k) == single.to_string(k)
        assert am.states_to_string(k) == single.states_to_string(k)


def test_align_batch_rejects_unsupported_settings(tmp_path) -> None:
    data = MultiSequenceDataset(filepaths=SMALL)
    with pytest.raises(ValueError):
        align_batch(data.datasets[0], _config(tmp_path))

    config = _config(tmp_path)
    config.training.skip_training = True
    with pytest.raises(NotImplementedError):
        align_batch(data, config)

    config = _config(tmp_path)
    config.init_msa.seeded = True
    with pytest.raises(ValueError):
        align_batch(data, config)

    # Every dataset needs a sequence in subset_ids
    config = _config(tmp_path)
    config.input_output.subset_ids = [data.datasets[0].seq_ids[0]]
    with pytest.raises(ValueError):
        align_batch(data, config)


def test_multi_alignment_model_has_no_best_head() -> None:
    data = MultiSequenceDataset(filepaths=SMALL)
    config = Configuration()
    config.training.no_sequence_weights = True
    config.training.length_init = [5, 4]
    model = make_learnmsa_model(LearnMSAContext(config, data))
    model.build()
    am = MultiAlignmentModel(
        data, model, [data.global_indices(k) for k in range(2)]
    )
    with pytest.raises(ValueError):
        am.select_best()
    with pytest.raises(ValueError):
        MultiAlignmentModel(data, model, [data.global_indices(0)])
