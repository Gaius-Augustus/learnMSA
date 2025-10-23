import numpy as np
import pytest
import tensorflow as tf

from learnMSA.protein_language_models import TrainingUtil


@pytest.fixture
def test_data() -> tuple[np.ndarray, np.ndarray]:
    """Fixture providing test data for pretraining utilities tests."""
    y_true = np.array([[[1., 0, 0, 0, 0],
                        [0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0]],
                        [[1, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0],
                         [0, 0, 1, 0, 0],
                         [0, 0, 0, 1, 0]]])
    y_pred = np.array([[[0.6, 0.4, 0, 0, 0],
                        [0, 0.6, 0.4, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0]],
                        [[0.6, 0.4, 0, 0, 0],
                         [0, 0.6, 0.4, 0, 0],
                         [0, 0, 0.6, 0.4, 0],
                         [0, 0, 0, 1, 0]]])
    return y_true, y_pred


def test_make_masked_categorical(
        test_data: tuple[np.ndarray, np.ndarray]
) -> None:
    y_true, y_pred = test_data
    y_true_masked, y_pred_masked, norm_masked = TrainingUtil.make_masked_categorical(y_true, y_pred)
    np.testing.assert_almost_equal(
        y_true_masked,
        [[1., 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0]]
    )
    np.testing.assert_almost_equal(
        y_pred_masked,
        [[0.6, 0.4, 0, 0, 0],
        [0, 0.6, 0.4, 0, 0],
        [0, 0, 0, 1, 0],
        [0.6, 0.4, 0, 0, 0],
        [0, 0.6, 0.4, 0, 0],
        [0, 0, 0.6, 0.4, 0],
        [0, 0, 0, 1, 0]]
    )
    np.testing.assert_almost_equal(norm_masked, [3., 3, 3, 4, 4, 4, 4])


def test_make_masked_binary(test_data : tuple[np.ndarray, np.ndarray]) -> None:
    y_true, y_pred = test_data
    y_true_masked, y_pred_masked, norm_masked = TrainingUtil.make_masked_binary(y_true, y_pred)
    np.testing.assert_almost_equal(
        y_true_masked,
        [[1.], [0], [0], [0], [1], [0], [0], [0], [1],
        [1], [0], [0], [0], [0], [1], [0], [0],
        [0], [0], [1], [0], [0], [0], [0], [1]]
    )
    np.testing.assert_almost_equal(
        y_pred_masked,
        [[0.6], [0.4], [0], [0], [0.6], [0], [0], [0], [1],
        [0.6], [0.4], [0], [0], [0], [0.6], [0.4], [0],
        [0], [0], [0.6], [0.4], [0], [0], [0], [1]]
    )
    np.testing.assert_almost_equal(norm_masked, [9.]*9 + [16.]*16)


def test_masked_loss_categorical(
        test_data : tuple[np.ndarray, np.ndarray]
) -> None:
    y_true, y_pred = test_data
    loss = TrainingUtil.make_masked_func(
        tf.keras.losses.categorical_crossentropy, categorical=True, name="cee"
    )
    loss_value = loss(y_true, y_pred)
    np.testing.assert_almost_equal(loss_value, -(2*np.log(0.6)/3 + 3*np.log(0.6) / 4) / 2)


def test_masked_acc_categorical(
        test_data : tuple[np.ndarray, np.ndarray]
) -> None:
    y_true, y_pred = test_data
    acc = TrainingUtil.make_masked_func(
        tf.keras.metrics.categorical_accuracy, categorical=True, name="acc"
    )
    acc_value = acc(y_true, y_pred)
    np.testing.assert_almost_equal(acc_value, 1.)


def test_masked_loss_binary(
        test_data : tuple[np.ndarray, np.ndarray]
) -> None:
    y_true, y_pred = test_data
    loss = TrainingUtil.make_masked_func(
        tf.keras.losses.binary_crossentropy, categorical=False, name="bce"
    )
    loss_value = loss(y_true, y_pred)
    np.testing.assert_almost_equal(loss_value, -(3*np.log(0.6) / 9 + 6*np.log(0.6)/16)/2)
