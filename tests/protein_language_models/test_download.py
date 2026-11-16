"""``learnMSA-download-weights``, with the downloads replaced."""

import pytest

import learnMSA.protein_language_models.download as download
from learnMSA.protein_language_models import prot_t5


@pytest.fixture
def calls(monkeypatch) -> list[tuple[str, str | None]]:
    calls = []
    for name in download.MODELS:
        monkeypatch.setitem(
            download.MODELS, name,
            lambda cache, name=name: calls.append((name, cache)) or name)
    return calls


def test_downloads_both_models_into_the_cache_dir(calls, capsys) -> None:
    download.main(["prostt5", "prott5", "--plm_cache_dir", "/w"])
    assert calls == [("prostt5", "/w"), ("prott5", "/w")]
    assert capsys.readouterr().out.splitlines() == [
        "prostt5: prostt5", "prott5: prott5"]


@pytest.mark.parametrize("name", ["prostt5", "prott5"])
def test_downloads_only_the_named_model(calls, name) -> None:
    download.main([name])
    assert calls == [(name, None)]


def test_a_repeated_model_is_downloaded_once(calls) -> None:
    download.main(["prott5", "prott5"])
    assert calls == [("prott5", None)]


@pytest.mark.parametrize("argv", [[], ["esm2"]])
def test_needs_known_models(argv) -> None:
    with pytest.raises(SystemExit) as e:
        download.parse_args(argv)
    assert e.value.code == 2


def test_prott5_download_uses_the_model_cache(tmp_path, monkeypatch) -> None:
    huggingface_hub = pytest.importorskip("huggingface_hub")
    seen = []

    def fake_snapshot_download(repo_id, **kwargs):
        seen.append((repo_id, kwargs))
        return "snapshot"

    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        fake_snapshot_download)
    assert prot_t5.download(str(tmp_path)) == "snapshot"
    assert seen == [(prot_t5.MODEL_CHECKPOINT, {
        "cache_dir": str(tmp_path / prot_t5.CACHE_ID),
        "allow_patterns": ["*.json", "spiece.model", "pytorch_model.bin"],
    })]
