"""Gates the backend-neutral initializer layer.

``LearnMSAContext`` used to build keras initializers directly and to call the
TensorFlow ``inverse_softplus``. Both now have backend-neutral counterparts, so
these check that the neutral versions agree with the tensor implementations and
that the TF bridge materializes every spec.
"""

import numpy as np
import pytest
import tensorflow as tf

from learnMSA.tree import initializer as spec
from learnMSA.tree.tf import initializer as tf_initializer
from learnMSA.tree.tf.util import inverse_softplus as tf_inverse_softplus


@pytest.mark.parametrize(
    "values",
    [
        np.array([1e-8, 1e-3, 0.05, 1.0, 5.0, 100.0], dtype=np.float64),
        np.array([0.5], dtype=np.float32),
        np.linspace(1e-6, 20.0, 97, dtype=np.float64),
    ],
    ids=["spread-f64", "single-f32", "dense-f64"],
)
def test_numpy_inverse_softplus_matches_tensorflow(values: np.ndarray) -> None:
    """The neutral inverse_softplus must reproduce the tensor version.

    Not bit-exact: numpy and TensorFlow round ``log(expm1(x))`` independently and
    can land one ULP apart. The tolerance is a few ULP of float64, which is far
    below anything that could move a substitution model.
    """
    expected = tf_inverse_softplus(values).numpy()
    actual = spec.inverse_softplus(values)
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=1e-15)


def test_substitution_model_init_matches_legacy_values() -> None:
    """The neutral builder reproduces the values the TF module produced.

    ``make_substitution_model_init`` only differed in using the tensor
    ``inverse_softplus``; this pins that the swap changed nothing.
    """
    R, p = spec.make_substitution_model_init(2, num_components=1)
    from evoten.substitution_models import LG

    from learnMSA.util.sequence_dataset import SequenceDataset

    R_ref, p_ref = LG(SequenceDataset._default_alphabet[:20])
    expected_R = np.tile(
        tf_inverse_softplus(R_ref + 1e-32).numpy()[None, None, None],
        [2, 1, 1, 1, 1],
    )
    expected_p = np.tile(np.log(p_ref)[None, None, None], [2, 1, 1, 1])
    np.testing.assert_allclose(R, expected_R, rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(p, expected_p, rtol=0, atol=0)


def test_to_tf_materializes_every_spec() -> None:
    constant = tf_initializer.to_tf(spec.Constant(np.array([1.0, 2.0])))
    assert isinstance(constant, tf_initializer.ConstantInitializer)
    np.testing.assert_allclose(constant(shape=(2,)).numpy(), [1.0, 2.0])

    zeros = tf_initializer.to_tf(spec.Zeros())
    np.testing.assert_allclose(zeros(shape=(3,)).numpy(), np.zeros(3))

    normal = tf_initializer.to_tf(spec.RandomNormal(stddev=0.1))
    assert isinstance(normal, tf.keras.initializers.RandomNormal)
    assert normal.stddev == pytest.approx(0.1)


def test_to_tf_passes_keras_initializers_through() -> None:
    """Layers keep accepting keras initializers directly."""
    original = tf.keras.initializers.Zeros()
    assert tf_initializer.to_tf(original) is original


def test_context_initializers_are_neutral_specs() -> None:
    """The context must describe initialization without keras objects."""
    from learnMSA import Configuration
    from learnMSA.model.context import LearnMSAContext

    config = Configuration()
    config.training.length_init = [10]
    context = LearnMSAContext(config, num_seq=17)

    for field in ("R_init", "p_init", "t_init", "mix_init", "R_delta_init"):
        value = getattr(context, field)
        assert isinstance(value, spec.InitSpec), (
            f"{field} is {type(value).__name__}, expected a neutral InitSpec"
        )
