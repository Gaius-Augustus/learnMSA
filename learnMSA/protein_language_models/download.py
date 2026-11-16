"""``learnMSA-download-weights``: fetch language model weights ahead of a run.

Downloads the weights that ``learnMSA --struct`` (ProstT5) and
``learnMSA --use_language_model`` (ProtT5) would otherwise download on their
first run, into the same cache. Useful where the run itself must not download,
e.g. a cluster job with a file size limit (``ulimit -f``) below the 5.3 GiB of
the ProstT5 checkpoint. Cached files are not downloaded again.
"""

import argparse


def _download_prostt5(plm_cache_dir: str | None) -> str:
    from learnMSA.structure.prostt5 import download_encoder
    return str(download_encoder(plm_cache_dir))


def _download_prott5(plm_cache_dir: str | None) -> str:
    from learnMSA.protein_language_models.prot_t5 import download
    return download(plm_cache_dir)


#: Model name on the command line -> download function. Each imports its
#: model's module only when asked for (ProstT5 needs torch).
MODELS = {
    "prostt5": _download_prostt5,
    "prott5": _download_prott5,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="learnMSA-download-weights",
        description="Download protein language model weights into the cache "
        "learnMSA reads them from.",
    )
    parser.add_argument("models", nargs="+", choices=sorted(MODELS),
                        help="prostt5: 3Di prediction (learnMSA --struct). "
                        "prott5: embeddings (learnMSA --use_language_model).")
    parser.add_argument("--plm_cache_dir", default=None,
                        help="Directory where the protein language models are "
                        "stored; pass the same to learnMSA. "
                        "(default: ~/.cache/learnmsa)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    # dict.fromkeys drops repeated names, keeping their order
    for name in dict.fromkeys(args.models):
        path = MODELS[name](args.plm_cache_dir)
        print(f"{name}: {path}", flush=True)


if __name__ == "__main__":
    main()
