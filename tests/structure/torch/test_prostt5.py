"""The ProstT5 3Di wrapper, with a tiny random encoder (no download)."""

import numpy as np
import pytest
import torch

import learnMSA.structure.prostt5 as prostt5
from learnMSA.config.structure import StructureConfig
from learnMSA.structure.io import KIND
from learnMSA.structure.predict_3di import main as predict_3di_main
from learnMSA.structure.prostt5 import (CNN_WEIGHTS, PROSTT5_CLASS_ORDER,
                                        ProstT5CNN, ProstT5Predictor,
                                        split_chunks, tokenize)
from learnMSA.util import EmbeddingDataset


def _tiny_encoder() -> torch.nn.Module:
    from transformers import T5Config, T5EncoderModel
    torch.manual_seed(0)
    config = T5Config(vocab_size=150, d_model=1024, d_kv=8, d_ff=16,
                      num_layers=1, num_heads=2)
    return T5EncoderModel(config).eval()


@pytest.fixture(scope="module")
def predictor() -> ProstT5Predictor:
    torch.manual_seed(1)
    cnn = ProstT5CNN()
    return ProstT5Predictor(device="cpu", encoder=_tiny_encoder(), cnn=cnn)


def test_class_order_matches_structural_alphabet() -> None:
    assert PROSTT5_CLASS_ORDER == StructureConfig().structural_alphabet


def test_tokenize_matches_foldseek_rules() -> None:
    np.testing.assert_array_equal(tokenize("ACa"), [3, 22, 3])
    # B/O/U/Z have their own tokens; letters without one become X.
    np.testing.assert_array_equal(tokenize("BOUZXJ*"),
                                  [24, 25, 26, 27, 23, 23, 23])


def test_token_ids_match_the_pinned_tokenizer() -> None:
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.T5Tokenizer.from_pretrained(
            prostt5.PROSTT5_REPO, revision=prostt5.PROSTT5_REVISION,
            cache_dir=str(prostt5.default_cache_dir()),
            local_files_only=True, legacy=True,
        )
    except OSError:
        pytest.skip("ProstT5 tokenizer not cached")
    for letter, token_id in prostt5.TOKEN_IDS.items():
        assert tok.convert_tokens_to_ids("▁" + letter) == token_id
    assert tok.convert_tokens_to_ids("<AA2fold>") == prostt5.AA2FOLD_ID
    assert tok.eos_token_id == prostt5.EOS_ID
    assert tok.pad_token_id == prostt5.PAD_ID


@pytest.mark.parametrize("length, expected", [
    (1000, [(0, 1000)]),
    (1024, [(0, 1024)]),
    (1025, [(0, 1022), (1022, 1025)]),
    (2048, [(0, 1022), (1022, 2044), (2044, 2048)]),
    (3000, [(0, 1024), (1024, 2048), (2048, 3000)]),
])
def test_split_matches_foldseek(length, expected) -> None:
    assert split_chunks(length) == expected
    assert split_chunks(length, 0) == [(0, length)]


def test_shipped_cnn_loads() -> None:
    cnn = ProstT5CNN.from_npz()
    with np.load(CNN_WEIGHTS) as data:
        np.testing.assert_array_equal(
            cnn.classifier[3].bias.detach().numpy(),
            data["classifier.3.bias"])
    out = cnn(torch.zeros((1, 5, 1024)))
    assert out.shape == (1, 5, 20)


def test_logits_do_not_depend_on_the_batch(predictor) -> None:
    short, long = "MKTAYIAKQR", "MSTNPKPQRKTKRNTNRRPQDVKFPGG" * 3
    alone = predictor.predict([short])
    together = predictor.predict([long, short, "GA"])
    np.testing.assert_allclose(together[len(long):len(long) + len(short)],
                               alone, rtol=1e-4, atol=1e-5)
    assert np.isfinite(together).all()


def test_split_pieces_are_predicted_separately(predictor) -> None:
    seq = "MKTAYIAKQRQISFVKSHFSRQ"
    got = predictor.predict([seq], split_length=8)
    pieces = [predictor.predict([seq[s:e]], split_length=0)
              for s, e in split_chunks(len(seq), 8)]
    np.testing.assert_allclose(got, np.concatenate(pieces),
                               rtol=1e-4, atol=1e-5)


def test_cli_writes_logits_and_argmax_fasta(tmp_path, monkeypatch) -> None:
    fasta = tmp_path / "in.fasta"
    fasta.write_text(">s1\nMKTAYIAKQR\n>s2 desc\nGAbu\n>s3\nMSTNPK\n")
    monkeypatch.setattr(prostt5, "load_encoder",
                        lambda device, cache_dir=None: _tiny_encoder())
    predict_3di_main(["-i", str(fasta), "-o", str(tmp_path / "out"),
                      "--fasta", str(tmp_path / "out.fasta"),
                      "--device", "cpu", "--silent"])
    data = EmbeddingDataset(tmp_path / "out.npz")
    assert str(data.metadata["kind"]) == KIND
    assert str(data.metadata["alphabet"]) == PROSTT5_CLASS_ORDER
    assert data.seq_ids == ["s1", "s2", "s3"]
    np.testing.assert_array_equal(data.seq_lens, [10, 4, 6])
    lines = (tmp_path / "out.fasta").read_text().split()
    assert lines[0::2] == [">s1", ">s2", ">s3"]
    for i, letters in enumerate(lines[1::2]):
        best = np.argmax(data.get_encoded_seq(i), axis=1)
        assert letters == "".join(PROSTT5_CLASS_ORDER[j] for j in best)
