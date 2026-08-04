"""Records reference ancestral-probability outputs from the TensorFlow backend.

The companion of :mod:`tests.fixtures.generate_parity_fixtures` for the tree
package; see that module for why the cross-backend comparison has to go through
a committed file.

Usage::

    conda activate learnMSAdev2
    python tests/fixtures/generate_anc_probs_fixtures.py
"""

import argparse
from pathlib import Path

import numpy as np

FIXTURE = Path(__file__).resolve().parent / "anc_probs_parity.npz"

HEADS = 2
ALPHABET_SIZE = 20
NUM_RATES = 6
BATCH, LENGTH = 4, 9
SEED = 20240804

#: (name, kwargs) of the layer configurations that get recorded. Between them
#: they cover the single-component fast path, the mixture path, the second
#: input track (which activates ``tau_track_kernel``), and clustering.
SCENARIOS = {
    "single": dict(input_tracks=1, num_components=1, time_reversed=True),
    "mixture": dict(input_tracks=1, num_components=3, time_reversed=True),
    "two_tracks": dict(input_tracks=2, num_components=1, time_reversed=True),
    "forward": dict(input_tracks=1, num_components=1, time_reversed=False),
    "clustered": dict(
        input_tracks=1, num_components=1, time_reversed=True, clustered=True
    ),
}


def make_init():
    """The neutral init specs every scenario is built from."""
    from learnMSA.tree.initializer import (Constant,
                                           make_substitution_model_init)

    R_init, p_init = make_substitution_model_init(HEADS)
    return {
        "exchangeability_init": Constant(R_init),
        "equilibrium_init": Constant(p_init),
        "rate_init": Constant(0.0),
    }


def scenario_kwargs(name: str) -> dict:
    """Constructor arguments for one scenario."""
    options = dict(SCENARIOS[name])
    clustered = options.pop("clustered", False)
    kwargs = dict(
        heads=HEADS,
        rates=NUM_RATES,
        alphabet_size=ALPHABET_SIZE,
        **make_init(),
        **options,
    )
    if clustered:
        # Two clusters over NUM_RATES sequences.
        kwargs["clusters"] = np.arange(NUM_RATES) % 2
    return kwargs


def make_inputs(input_tracks: int) -> dict[str, np.ndarray]:
    """Deterministic one-hot sequences and the rate indices they map to."""
    rng = np.random.default_rng(SEED)
    inputs = {}
    for track in range(input_tracks):
        symbols = rng.integers(
            0, ALPHABET_SIZE, size=(BATCH, LENGTH, HEADS)
        )
        array = np.zeros(
            (BATCH, LENGTH, HEADS, ALPHABET_SIZE), dtype=np.float32
        )
        b, ell, h = np.meshgrid(
            np.arange(BATCH),
            np.arange(LENGTH),
            np.arange(HEADS),
            indexing="ij",
        )
        array[b, ell, h, symbols] = 1.0
        inputs[f"sequence{track}"] = array
    inputs["rate_indices"] = rng.integers(
        0, NUM_RATES, size=(BATCH, HEADS)
    ).astype(np.int32)
    return inputs


def kernel_paths(layer) -> dict[str, object]:
    """Every parameter of the layer, keyed by a backend-neutral path."""
    kernels = {
        "exchangeability_delta_kernel": layer.exchangeability_delta_kernel,
        "equilibrium_kernel": layer.equilibrium_kernel,
        "tau_kernel": layer.tau_kernel,
    }
    if layer.num_components > 1:
        kernels["mixture_kernel"] = layer.mixture_kernel
        kernels["scale_kernel"] = layer.scale_kernel
    if layer.input_tracks > 1:
        kernels["tau_track_kernel"] = layer.tau_track_kernel
    return kernels


def reference_outputs(layer, inputs: dict[str, np.ndarray]) -> dict:
    """The tensors the torch backend has to reproduce."""
    from learnMSA.util.tensor import to_numpy

    rate_indices = inputs["rate_indices"]
    tau = layer.make_tau(rate_indices)
    outputs = {
        "R": to_numpy(layer.make_R()),
        "p": to_numpy(layer.make_p()),
        "w": to_numpy(layer.make_w()),
        "scale": to_numpy(layer.make_scale()),
        "Q": to_numpy(layer.make_Q()),
        "tau": to_numpy(tau),
        "P": to_numpy(layer.make_P(tau)),
    }
    sequences = [
        inputs[key] for key in sorted(inputs) if key.startswith("sequence")
    ]
    result = layer(*sequences, rate_indices=rate_indices)
    if isinstance(result, tuple):
        for i, value in enumerate(result):
            outputs[f"anc_probs{i}"] = to_numpy(value)
    else:
        outputs["anc_probs0"] = to_numpy(result)
    return outputs


def build_layer(name: str):
    from learnMSA.tree.tf.anc_probs_layer import AncProbsLayer

    layer = AncProbsLayer(**scenario_kwargs(name))
    layer.build()
    return layer


def collect() -> dict[str, np.ndarray]:
    import tensorflow as tf

    from learnMSA.util.tensor import to_numpy

    # Full float32, not TF32; see generate_parity_fixtures._disable_tf32.
    tf.config.experimental.enable_tensor_float_32_execution(False)

    fixture: dict[str, np.ndarray] = {}
    for name in SCENARIOS:
        tf.keras.utils.set_random_seed(SEED)
        layer = build_layer(name)
        inputs = make_inputs(layer.input_tracks)
        for key, array in inputs.items():
            fixture[f"{name}/input.{key}"] = array
        for key, kernel in kernel_paths(layer).items():
            fixture[f"{name}/kernel.{key}"] = to_numpy(kernel)
        for key, value in reference_outputs(layer, inputs).items():
            fixture[f"{name}/output.{key}"] = np.asarray(value)
    return fixture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare against the committed fixture instead of rewriting it",
    )
    args = parser.parse_args()

    fixture = collect()
    if args.check:
        stored = np.load(FIXTURE)
        differing = set(fixture) ^ set(stored.files)
        if differing:
            raise SystemExit(f"fixture keys differ: {sorted(differing)}")
        for key, value in fixture.items():
            np.testing.assert_allclose(
                value, stored[key], rtol=1e-6, atol=1e-6,
                err_msg=f"{key} drifted from the committed fixture",
            )
        print(f"{FIXTURE.name} is up to date ({len(fixture)} arrays).")
        return

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FIXTURE, **fixture)
    print(f"wrote {FIXTURE} ({len(fixture)} arrays)")


if __name__ == "__main__":
    main()
