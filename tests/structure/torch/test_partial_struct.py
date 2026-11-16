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
    # s3 is identical to s1 and shares its structure.
    np.testing.assert_array_equal(struct.get_encoded_seq(2),
                                  struct.get_encoded_seq(0))
    # Predicted logits for the other.
    np.testing.assert_allclose(
        struct.get_encoded_seq(3),
        _predicted(data, ["s4"]).get_encoded_seq(0), rtol=1e-3, atol=1e-3)


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


@pytest.mark.parametrize("cluster_id", [1.0, 0.6])
@pytest.mark.parametrize("with_structure, without", [("s1", 2), ("s3", 0)])
def test_partial_structures_are_shared_within_clusters(
        tmp_path, monkeypatch, cluster_id, with_structure, without) -> None:
    """s1 and s3 are identical. Whichever has a structure (the
    representative s1 or its member s3) lends its 3Di to the other, and
    ProstT5 runs only for the sequences of clusters without a structure."""
    (tmp_path / "s.fasta").write_text(f">{with_structure}\nDPQRSD\n")
    config = Configuration()
    config.input_output.struct_file = tmp_path / "s.fasta"
    config.input_output.work_dir = str(tmp_path / "work")
    config.input_output.verbose = False
    config.structure.use_structure = True
    config.language_model.plm_cluster_id = cluster_id
    predicted = []
    predict_dataset = prostt5.predict_dataset

    def counting(seqs, **kwargs):
        predicted.extend(seqs.seq_ids)
        return predict_dataset(seqs, **kwargs)

    monkeypatch.setattr(prostt5, "predict_dataset", counting)
    data = SequenceDataset(sequences=SEQS)
    struct = load_struct_data(config, data)

    assert predicted == ["s2", "s4"]
    assert config.structure.input_format == "logits"
    np.testing.assert_array_equal(struct.get_encoded_seq(without),
                                  struct.get_encoded_seq(2 - without))
    x = np.eye(20)[[ALPHABET.index(c) for c in "DPQRSD"]]
    x[[0, -1]] = 1.0
    np.testing.assert_allclose(
        _observe(config, struct.get_encoded_seq(without)), x, atol=ATOL)


def test_structures_for_all_representatives_need_no_prediction(
        tmp_path, monkeypatch) -> None:
    (tmp_path / "s.fasta").write_text(
        ">s1\nDPQRSD\n>s2\nDVLP\n>s4\nAAAA\n")
    config = Configuration()
    config.input_output.struct_file = tmp_path / "s.fasta"
    config.input_output.work_dir = str(tmp_path / "work")
    config.input_output.verbose = False
    config.structure.use_structure = True
    monkeypatch.setattr(prostt5, "predict_dataset", None)
    struct = load_struct_data(config, SequenceDataset(sequences=SEQS))

    assert config.structure.input_format == "logits"
    np.testing.assert_array_equal(struct.get_encoded_seq(2),
                                  struct.get_encoded_seq(0))


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
