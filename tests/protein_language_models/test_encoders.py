"""The backend-neutral input encoders.

Only the ``zeros`` encoder is exercised: the ESM-2 and ProtT5 encoders need a
tokenizer download, which is far too heavy for the test suite. What is checked
here is the contract every encoder shares -- numpy arrays with explicitly set
dtypes -- because the backends convert what they get verbatim, and the
TensorFlow one is traced against a static signature that has to agree with it.
See ``tests/protein_language_models/tf/test_signatures.py`` for that half.
"""

import numpy as np

from learnMSA.protein_language_models.zeros import ZerosInputEncoder


def _no_crop(n: int) -> np.ndarray:
    return np.repeat([[False, False]], n, axis=0)


def test_zeros_encoder_masks_each_sequence_to_its_length() -> None:
    encoder = ZerosInputEncoder()
    ids, mask = encoder(["ACGT", "ACG", "ACGTA"], _no_crop(3))

    assert ids.shape == (3, 5)
    np.testing.assert_array_equal(mask.sum(-1), [4, 3, 5])
    np.testing.assert_array_equal(ids, np.zeros((3, 5), dtype=np.int32))


def test_zeros_encoder_handles_an_empty_batch() -> None:
    ids, mask = ZerosInputEncoder()([], _no_crop(0))
    assert ids.shape == (0, 0)
    assert mask.shape == (0, 0)


def test_zeros_encoder_emits_int32_numpy_arrays() -> None:
    """The dtypes are the encoder's whole contract with the backends.

    Nothing downstream casts them, so a platform-dependent default here would
    reach the TensorFlow input signature as a mismatch.
    """
    outputs = ZerosInputEncoder()(["ACGT", "ACG"], _no_crop(2))

    assert len(outputs) == 2
    for array in outputs:
        assert isinstance(array, np.ndarray)
        assert array.dtype == np.int32
        assert array.ndim == 2
