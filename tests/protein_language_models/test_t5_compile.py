import numpy as np

from learnMSA.protein_language_models.t5_compile import compile_pays_off


def test_short_datasets_run_eagerly() -> None:
    assert not compile_pays_off(np.full(10000, 100))


def test_long_datasets_are_compiled() -> None:
    assert compile_pays_off(np.full(10000, 1000))


def test_many_short_sequences_add_up() -> None:
    assert compile_pays_off(np.full(1_000_000, 100))
