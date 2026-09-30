"""learnMSA reads per-residue 3Di logits as its structural track."""

import numpy as np
import pytest

from learnMSA.config import Configuration
from learnMSA.run.util import load_struct_data
from learnMSA.structure.io import write_logits
from learnMSA.util.sequence_dataset import SequenceDataset

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def _setup(tmp_path, lens=(4, 6), ids=("s1", "s2")):
    data = SequenceDataset(sequences=[("s2", "MKTAYI"), ("s1", "GAVL")])
    logits = np.random.default_rng(0).normal(
        size=(sum(lens), 20)).astype(np.float16)
    path = write_logits(tmp_path / "z.npz", logits, np.array(lens),
                        list(ids), ALPHABET[::-1])
    config = Configuration()
    config.input_output.struct_file = path
    config.structure.use_structure = True
    return config, data, logits


def test_logits_are_loaded_in_input_order(tmp_path) -> None:
    config, data, logits = _setup(tmp_path)
    struct = load_struct_data(config, data)
    assert config.structure.input_format == "logits"
    assert struct.seq_ids == ["s2", "s1"]
    # Columns come back in the structural alphabet order.
    np.testing.assert_array_equal(struct.get_encoded_seq(1),
                                  logits[:4, ::-1].astype(np.float32))


def test_length_mismatch_is_rejected(tmp_path) -> None:
    config, data, _ = _setup(tmp_path, lens=(4, 5))
    with pytest.raises(ValueError, match="differ in length"):
        load_struct_data(config, data)


def test_id_mismatch_is_rejected(tmp_path) -> None:
    config, data, _ = _setup(tmp_path, ids=("s1", "s3"))
    with pytest.raises(ValueError, match="do not match"):
        load_struct_data(config, data)
