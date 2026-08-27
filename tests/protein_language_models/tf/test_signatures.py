"""The input signatures the TensorFlow language models are traced against.

TensorFlow needs a static signature so that ``tf/embed.py`` traces its
``tf.function`` once instead of once per batch shape. The signature used to
live on the encoder, next to the code producing the arrays; it now lives on the
model wrapper, so that nothing about TensorFlow's requirement leaks into the
backend-neutral encoders. These tests are what keeps the two halves agreeing.

The wrapper *classes* can be imported without downloading a checkpoint, so the
structural checks cover all four models. The encoder cross-check needs a real
encoder and therefore only covers ``zeros``, the one that needs no download.
"""

import numpy as np
import pytest
import tensorflow as tf

from learnMSA.protein_language_models.tf.esm2 import TFESM2LanguageModel
from learnMSA.protein_language_models.tf.language_model import TFLanguageModel
from learnMSA.protein_language_models.tf.prot_t5 import TFProtT5LanguageModel
from learnMSA.protein_language_models.tf.protein_bert import \
    TFProteinBERTLanguageModel
from learnMSA.protein_language_models.tf.zeros import TFZerosLanguageModel
from learnMSA.protein_language_models.zeros import ZerosInputEncoder

#: Every concrete TensorFlow language model wrapper.
WRAPPERS = [
    TFESM2LanguageModel,
    TFProtT5LanguageModel,
    TFProteinBERTLanguageModel,
    TFZerosLanguageModel,
]

#: dtypes the neutral encoders emit.
ALLOWED_DTYPES = (tf.int32, tf.float32)


@pytest.mark.parametrize("wrapper", WRAPPERS, ids=lambda w: w.__name__)
def test_signature_is_a_tuple_of_tensor_specs(wrapper) -> None:
    signature = wrapper.INPUT_SIGNATURE
    assert isinstance(signature, tuple)
    assert signature, f"{wrapper.__name__} declares an empty signature"
    for spec in signature:
        assert isinstance(spec, tf.TensorSpec)
        assert spec.dtype in ALLOWED_DTYPES


@pytest.mark.parametrize("wrapper", WRAPPERS, ids=lambda w: w.__name__)
def test_batch_axis_is_dynamic(wrapper) -> None:
    """A static batch size would retrace on the last, short batch."""
    for spec in wrapper.INPUT_SIGNATURE:
        assert spec.shape.rank is not None
        assert spec.shape[0] is None


def test_the_base_class_declares_the_contract_without_a_default() -> None:
    """Annotation only, so a wrapper that forgets it fails loudly."""
    assert "INPUT_SIGNATURE" in TFLanguageModel.__annotations__
    assert not hasattr(TFLanguageModel, "INPUT_SIGNATURE")


def test_zeros_encoder_output_matches_its_signature() -> None:
    """The cross-check that used to be implicit in ``output_specs``.

    ``zeros`` is the only encoder that needs no tokenizer download; for the
    others this same agreement is only exercised by an actual embedding run.
    """
    encoder = ZerosInputEncoder()
    outputs = encoder(["ACGT", "ACG", "ACGTA"], np.zeros((3, 2), dtype=bool))
    signature = TFZerosLanguageModel.INPUT_SIGNATURE

    assert len(outputs) == len(signature)
    for array, spec in zip(outputs, signature):
        assert tf.as_dtype(array.dtype) == spec.dtype
        assert array.ndim == spec.shape.rank
        for size, expected in zip(array.shape, spec.shape):
            if expected is not None:
                assert size == expected

    # The signature has to actually accept the arrays, not merely describe them.
    for array, spec in zip(outputs, signature):
        assert spec.is_compatible_with(tf.constant(array))
