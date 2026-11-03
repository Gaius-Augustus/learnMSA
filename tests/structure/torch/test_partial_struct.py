"""Structures for only some of the sequences: the 3Di of the others is
predicted with ProstT5 (a tiny random encoder here, no download)."""

import numpy as np
import pytest
import torch

import learnMSA.structure.prostt5 as prostt5
from learnMSA.config import Configuration
from learnMSA.model.torch.struct_observation import \
    StructLogitObservationLayer
from learnMSA.run import util
from learnMSA.run.util import load_struct_data
from learnMSA.structure.io import KIND, tokens_as_logits
from learnMSA.util import EmbeddingCache, EmbeddingDataset
from learnMSA.util.sequence_dataset import SequenceDataset
from tests.structure.torch.test_prostt5 import _tiny_encoder

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
SEQS = [("s1", "MKTAYI"), ("s2", "GAVL"), ("s3", "MKTAYI"), ("s4", "PQRS")]


@pytest.fixture(autouse=True)
def tiny_prostt5(monkeypatch):
    monkeypatch.setattr(prostt5, "load_encoder",
                        lambda device, cache_dir=None: _tiny_encoder())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


# float16 rounding of T * log(pi) in unknown rows
ATOL = 5e-3


def _observe(config, z):
    layer = StructLogitObservationLayer(config.structure)
    return layer(torch.as_tensor(np.asarray(z, dtype=np.float32))).numpy()


def _predicted(data, ids):
    """ProstT5 logits of the sequences ``ids`` in ``data`` on their own."""
    subset = data.subset(np.array([data.seq_ids.index(i) for i in ids]))
    return prostt5.predict_dataset(subset, verbose=False)


def test_tokens_as_logits_are_observed_as_tokens() -> None:
    config = Configuration()
    st = config.structure
    x = np.eye(20, dtype=np.float32)[[3, 0, 19, 7]]
    x[[0, -1]] = 1.0
    z = tokens_as_logits(x, st.background_distribution, st.input_temperature)
    assert z.dtype == np.float16
    np.testing.assert_allclose(_observe(config, z), x, atol=ATOL)


def test_partial_3di_fasta_is_completed_with_prostt5(tmp_path) -> None:
    (tmp_path / "s.fasta").write_text(">s2\nDVLP\n>s1\nDPQRSD\n")
    config = Configuration()
    config.input_output.struct_file = tmp_path / "s.fasta"
    config.input_output.work_dir = str(tmp_path / "work")
    config.structure.use_structure = True
    data = SequenceDataset(sequences=SEQS)
    struct = load_struct_data(config, data)

    assert config.structure.input_format == "logits"
    assert struct.seq_ids == [i for i, _ in SEQS]
    np.testing.assert_array_equal(struct.seq_lens, data.seq_lens)
    # Letters from the structure, termini unknown.
    x = np.eye(20)[[ALPHABET.index(c) for c in "DPQRSD"]]
    x[[0, -1]] = 1.0
    np.testing.assert_allclose(
        _observe(config, struct.get_encoded_seq(0)), x, atol=ATOL)
    # Predicted logits for the others, in input order.
    predicted = _predicted(data, ["s3", "s4"])
    for i, k in [(2, 0), (3, 1)]:
        np.testing.assert_allclose(
            struct.get_encoded_seq(i), predicted.get_encoded_seq(k),
            rtol=1e-3, atol=1e-3)


def test_partial_logits_file_is_completed_with_prostt5(tmp_path) -> None:
    logits = np.random.default_rng(0).normal(size=(4, 20)).astype(np.float16)
    path = EmbeddingDataset(
        embedding_cache=EmbeddingCache(np.array([4]), 20, cache=logits),
        seq_ids=["s2"],
    ).write(tmp_path / "z.npz",
            metadata={"alphabet": ALPHABET[::-1], "kind": KIND})
    config = Configuration()
    config.input_output.struct_file = path
    config.input_output.work_dir = str(tmp_path / "work")
    config.structure.use_structure = True
    data = SequenceDataset(sequences=SEQS)
    struct = load_struct_data(config, data)

    np.testing.assert_array_equal(struct.get_encoded_seq(1),
                                  logits[:, ::-1].astype(np.float32))
    predicted = _predicted(data, ["s1", "s3", "s4"])
    np.testing.assert_allclose(
        struct.get_encoded_seq(3), predicted.get_encoded_seq(2),
        rtol=1e-3, atol=1e-3)


def test_partial_structures_reject_plm_reuse(tmp_path) -> None:
    (tmp_path / "s.fasta").write_text(">s2\nDVLP\n")
    config = Configuration()
    config.input_output.struct_file = tmp_path / "s.fasta"
    config.language_model.plm_reuse_identity = 0.9
    with pytest.raises(ValueError, match="plm_reuse_identity"):
        load_struct_data(config, SequenceDataset(sequences=SEQS))


def test_partial_structures_need_pytorch(tmp_path, monkeypatch) -> None:
    (tmp_path / "s.fasta").write_text(">s2\nDVLP\n")
    config = Configuration()
    config.input_output.struct_file = tmp_path / "s.fasta"
    monkeypatch.setattr(util, "get_backend", lambda: "tensorflow")
    with pytest.raises(NotImplementedError, match="only some"):
        load_struct_data(config, SequenceDataset(sequences=SEQS))


def test_3di_letters_of_another_length_are_rejected(tmp_path) -> None:
    (tmp_path / "s.fasta").write_text(">s2\nDVLPD\n")
    config = Configuration()
    config.input_output.struct_file = tmp_path / "s.fasta"
    with pytest.raises(ValueError, match="differ in length"):
        load_struct_data(config, SequenceDataset(sequences=SEQS))
