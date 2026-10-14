"""learnMSA reads per-residue 3Di logits as its structural track."""

import numpy as np
import pytest

from learnMSA.config import Configuration
from learnMSA.config.structure import DEFAULT_INPUT_TEMPERATURES
from learnMSA.run.util import load_struct_data
from learnMSA.structure.io import KIND, PDB, PROSTT5, SOURCE_KEY
from learnMSA.util import EmbeddingCache, EmbeddingDataset
from learnMSA.util.sequence_dataset import SequenceDataset

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def _setup(tmp_path, lens=(4, 6), ids=("s1", "s2"), stored=ALPHABET[::-1],
           extra=None):
    data = SequenceDataset(sequences=[("s2", "MKTAYI"), ("s1", "GAVL")])
    logits = np.random.default_rng(0).normal(
        size=(sum(lens), 20)).astype(np.float16)
    path = EmbeddingDataset(
        embedding_cache=EmbeddingCache(np.array(lens), 20, cache=logits),
        seq_ids=list(ids),
    ).write(tmp_path / "z.npz",
            metadata={"alphabet": stored, "kind": KIND, **(extra or {})})
    config = Configuration()
    config.input_output.struct_file = path
    config.structure.use_structure = True
    return config, data, logits


def test_logits_are_loaded_in_input_order(tmp_path) -> None:
    config, data, logits = _setup(tmp_path)
    struct = load_struct_data(config, data)
    assert config.structure.input_format == "prostt5_logits"
    assert struct.seq_ids == ["s2", "s1"]
    # Columns come back in the structural alphabet order.
    np.testing.assert_array_equal(struct.get_encoded_seq(1),
                                  logits[:4, ::-1].astype(np.float32))


def test_logit_source_is_read_from_the_file(tmp_path) -> None:
    # Files without a hint are ProstT5 predictions.
    config, data, _ = _setup(tmp_path)
    load_struct_data(config, data)
    assert config.structure.input_format == f"{PROSTT5}_logits"
    assert config.structure.input_temperature == \
        DEFAULT_INPUT_TEMPERATURES["prostt5_logits"]
    config, data, _ = _setup(tmp_path, extra={SOURCE_KEY: PDB})
    load_struct_data(config, data)
    assert config.structure.input_format == f"{PDB}_logits"
    assert config.structure.input_temperature == \
        DEFAULT_INPUT_TEMPERATURES["pdb_logits"]
    # A given temperature is kept.
    config, data, _ = _setup(tmp_path, extra={SOURCE_KEY: PDB})
    config.structure.input_temperature = 0.3
    load_struct_data(config, data)
    assert config.structure.input_temperature == 0.3
    config, data, _ = _setup(tmp_path, extra={SOURCE_KEY: "esm"})
    with pytest.raises(ValueError, match="logit source"):
        load_struct_data(config, data)


def test_length_mismatch_is_rejected(tmp_path) -> None:
    config, data, _ = _setup(tmp_path, lens=(4, 5))
    with pytest.raises(ValueError, match="differ in length"):
        load_struct_data(config, data)


def test_id_mismatch_is_rejected(tmp_path) -> None:
    config, data, _ = _setup(tmp_path, ids=("s1", "s3"))
    with pytest.raises(ValueError, match="do not match"):
        load_struct_data(config, data)


def test_foreign_alphabet_is_rejected(tmp_path) -> None:
    config, data, _ = _setup(tmp_path, stored="ACDEFGHIKLMNPQRSTVWX")
    with pytest.raises(ValueError, match="do not match"):
        load_struct_data(config, data)
