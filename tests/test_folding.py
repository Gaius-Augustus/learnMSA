import numpy as np
import tensorflow as tf

from hidten import HMMMode
from hidten.tf import TFHMM
from hidten.tf.prior.dirichlet import TFDirichletPrior


# Test an HMM where the transitions are determined by a parameterized
# function rather than based on an internal kernel.
#
# Specifically, we have the following implicit HMM:
#
# s1 -> s2 -> s3 -> s4
#  \       __/  __/
#   \__z1/__z2/
#
# where z1, z2 are silent states (without any emission) that can be used
# to skip s2 or s3 by "pausing" emissions of observations.
#
# Silent emissions can not be implemented directly with hidten. Instead,
# we can fold the implicit HMM to an explicit HMM without z1 and z2:
#
# s1 -> s2 -> s3 -> s4
#  \\____\____/    //
#   \_____\_______//
#          \______/
#
# and set:
# P(s3 | s1) = P(z1 | s1) P(s3 | z1)
# P(s4 | s1) = P(z1 | s1) P(z2 | z1) P(s4 | z2)
# P(s4 | s2) = P(z2 | s2) P(s4 | z2)
#
# The transitions in the explicit HMM are computed from the
# transitions in the implicit HMM and thus being interdependent.
#

class HMMFoldingModel(tf.keras.Model):

    def __init__(self) -> None:
        super().__init__()
        self.hmm = TFHMM(states=4)

        self.hmm.emitter[0].initializer = tf.keras.initializers.GlorotNormal()

        self.hmm.transitioner.allow = [
            (0, 1), (1, 2), (2, 3), (0, 2), (0, 4), (1, 3)
        ]
        self.hmm.transitioner.initializer()

        self.out = self.add_weight(
            shape=(1, 4, 5),
            initializer=tf.keras.initializers.GlorotNormal(),
        )

    def build(self, input_shape: tuple[int | None, ...]) -> None:
        self.hmm.build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = tf.nn.softmax(x)
        hmm_out = self.hmm(x, mode=HMMMode.POSTERIOR, parallel=25)
        return tf.einsum("bthd,hdo->bto", hmm_out, self.out)


def test_eager():
    ds = tf.data.Dataset.from_tensor_slices((
        np.random.normal(size=(64, 1000, 16)),
        np.random.randint(0, 5, size=(64, 1000)),
    ))
    ds = ds.batch(32)

    model = HMMTestModel()
    model.build((None, None, 16))

    model.compile(
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        jit_compile=False,
        run_eagerly=True,
    )

    model.fit(ds, verbose=False, epochs=2)


def test_uncompiled():
    ds = tf.data.Dataset.from_tensor_slices((
        np.random.normal(size=(64, 1000, 16)),
        np.random.randint(0, 5, size=(64, 1000)),
    ))
    ds = ds.batch(32)

    model = HMMTestModel()
    model.build((None, None, 16))

    model.compile(
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        jit_compile=False,
    )

    model.fit(ds, verbose=False, epochs=2)


def test_compiled():

    ds = tf.data.Dataset.from_tensor_slices((
        np.random.normal(size=(64, 1000, 16)),
        np.random.randint(0, 5, size=(64, 1000)),
    ))
    ds = ds.batch(32)

    model = HMMTestModel()
    model.build((None, None, 16))

    model.compile(
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        jit_compile=True,
    )

    model.fit(ds, verbose=False, epochs=2)


def test_with_prior():

    ds = tf.data.Dataset.from_tensor_slices((
        np.random.normal(size=(64, 1000, 16)),
        np.random.randint(0, 5, size=(64, 1000)),
    ))
    ds = ds.batch(32)

    model = HMMTestModel(use_prior=True)
    model.build((None, None, 16))

    model.compile(
        # larger learning rate to see some effect of the prior in a very short
        # training
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
        loss= tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        jit_compile=True,
    )

    before_training = model.hmm.emitter[0].matrix()

    model.fit(ds, verbose=False, epochs=3)

    after_training = model.hmm.emitter[0].matrix()

    # check if the first symbol with a high prior concentration dominates
    # in the learned matrix
    assert np.all(
        after_training[..., 0] > before_training[..., 0]
    )
    assert np.all(
        after_training[..., 0] > np.max(after_training[..., 1:], axis=-1)
    )
