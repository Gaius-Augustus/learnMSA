"""Records reference pHMM outputs from the TensorFlow backend.

No conda environment for this project has both TensorFlow and PyTorch
installed, so the backends cannot be compared inside one process. Instead this
script -- run under the TensorFlow environment -- builds a set of pHMM layers,
writes their raw kernels together with the tensors they produce, and commits
the result. ``tests/hmm/torch/test_parity.py`` then rebuilds the same layers
under PyTorch, copies the recorded kernels in, and checks that it arrives at
the same numbers.

Kernels are keyed by the *path* of the owning module inside the layer rather
than by a framework-specific weight name, so the two backends agree on them
without either having to know about the other's naming.

Usage::

    conda activate learnMSAdev2
    python tests/fixtures/generate_parity_fixtures.py
"""

import argparse
from pathlib import Path

import numpy as np

FIXTURE = Path(__file__).resolve().parent / "phmm_parity.npz"

#: Match-state counts of the two heads every scenario is built with. Different
#: lengths on purpose: it exercises the per-head padding and masking paths.
LENGTHS = [5, 7]

#: Amino acid alphabet size the reference inputs are generated for.
ALPHABET_SIZE = 20

#: Shape of the observation batch the reference outputs are computed on.
BATCH, TIMESTEPS = 3, 11

#: Number of unpadded positions in each reference sequence.
SEQUENCE_LENGTH = 9

SEED = 20240804


def make_configs(scenario: str):
    """Build the (config, struct_config) pair for a scenario."""
    from learnMSA.config import PHMMConfig, PHMMPriorConfig
    from learnMSA.config.structure import StructureConfig

    config = PHMMConfig()
    prior_config = PHMMPriorConfig()
    struct_config = None
    if scenario == "struct":
        struct_config = StructureConfig()
        struct_config.use_structure = True
    elif scenario == "joint":
        struct_config = StructureConfig()
        struct_config.use_structure = True
        struct_config.joint_emissions = True
    elif scenario != "aa":
        raise ValueError(f"Unknown scenario {scenario!r}")
    return config, prior_config, struct_config


#: The scenarios recorded. "aa" is the plain profile emitter, "struct" adds a
#: second categorical track, "joint" replaces both with the joint emitter.
SCENARIOS = ["aa", "struct", "joint"]

#: The emission tracks each scenario has, in the spelling
#: ``PHMMLayer.emission_matrix`` takes. Stated rather than discovered, so that
#: an emitter which silently fails to build shows up as a missing fixture key
#: instead of as one fewer comparison, and so the torch test can parametrize
#: over exactly the combinations that exist.
EMISSION_TRACKS = {
    "aa": ("aa",),
    "struct": ("aa", "struct"),
    "joint": ("aa", "joint"),
}

#: (scenario, track) pairs, flattened for ``pytest.mark.parametrize``.
EMISSION_CASES = [
    (scenario, track)
    for scenario, tracks in EMISSION_TRACKS.items()
    for track in tracks
]


def input_shapes(struct_config, batch: int | None = None):
    """The ``build`` input shapes for a scenario, matching the model's."""
    shapes = ((batch, None, len(LENGTHS), ALPHABET_SIZE),)
    if struct_config is not None and struct_config.use_structure:
        shapes += ((batch, None, len(LENGTHS), struct_config.alphabet_size),)
    shapes += ((batch, None, len(LENGTHS), 1),)
    return shapes


def make_inputs(struct_config) -> dict[str, np.ndarray]:
    """Deterministic observation batch: one-hot tracks plus a padding mask."""
    rng = np.random.default_rng(SEED)
    heads = len(LENGTHS)

    def one_hot(dim: int) -> np.ndarray:
        symbols = rng.integers(0, dim, size=(BATCH, SEQUENCE_LENGTH, heads))
        array = np.zeros((BATCH, TIMESTEPS, heads, dim), dtype=np.float32)
        b, t, h = np.meshgrid(
            np.arange(BATCH),
            np.arange(SEQUENCE_LENGTH),
            np.arange(heads),
            indexing="ij",
        )
        array[b, t, h, symbols] = 1.0
        return array

    inputs = {"x": one_hot(ALPHABET_SIZE)}
    if struct_config is not None and struct_config.use_structure:
        inputs["struct"] = one_hot(struct_config.alphabet_size)
    padding = np.zeros((BATCH, TIMESTEPS, heads, 1), dtype=np.float32)
    padding[:, :SEQUENCE_LENGTH] = 1.0
    inputs["padding"] = padding
    return inputs


def kernel_paths(layer) -> dict[str, object]:
    """Every trainable kernel of a pHMM layer, keyed by a neutral path.

    Reaching into ``layer.hmm`` is what the neutral accessors exist to avoid in
    library code, but a parity fixture has to pin the *parameters*, not just
    the outputs they produce.
    """
    explicit = layer.hmm.transitioner.explicit_transitioner
    kernels: dict[str, object] = {
        "transitioner.kernel": explicit.kernel,
        "transitioner.kernel_start": explicit.kernel_start,
    }
    for name in ("profile_emitter", "struct_emitter", "joint_emitter"):
        emitter = getattr(layer, name, None)
        if emitter is not None:
            kernels[f"{name}.kernel"] = emitter.kernel
    return kernels


def reference_outputs(
    layer, scenario: str, inputs: dict[str, np.ndarray]
) -> dict:
    """The tensors the torch backend has to reproduce."""
    from learnMSA.util.tensor import to_numpy

    transitioner = layer.hmm.transitioner
    outputs = {
        "transition_matrix": layer.transition_matrix(),
        "explicit_transition_matrix": layer.explicit_transition_matrix(),
        # The start distributions are not reachable through the neutral
        # accessors, but the folded one alone decides where a Viterbi path
        # begins, so a fixture without it would miss half the transitioner.
        "start_dist": to_numpy(transitioner.start_dist()),
        "explicit_start_dist": to_numpy(
            transitioner.explicit_transitioner.start_dist()
        ),
        "prior_scores": to_numpy(layer.prior_scores()),
    }
    for track in EMISSION_TRACKS[scenario]:
        outputs[f"emission_matrix.{track}"] = layer.emission_matrix(track)

    adds = ("struct",) if "struct" in inputs else ()
    call_adds = tuple(inputs[name] for name in adds) or None

    # The joint emission scores of all emitters, i.e. everything the scan
    # consumes. Recording them separates an emitter bug from a scan bug.
    observations = (inputs["x"],) + tuple(inputs[n] for n in adds)
    observations += (inputs["padding"],)
    outputs["emission_scores"] = to_numpy(
        layer.hmm.emission_scores(*observations)
    )

    layer.loglik_mode()
    outputs["loglik"] = to_numpy(
        layer(inputs["x"], padding=inputs["padding"], adds=call_adds)
    )
    layer.viterbi_mode()
    outputs["viterbi"] = to_numpy(
        layer(inputs["x"], padding=inputs["padding"], adds=call_adds)
    )
    layer.loglik_mode()
    return outputs


def _disable_tf32() -> None:
    """Record full float32, not TF32.

    TensorFlow enables TF32 matmuls by default on Ampere and later, which costs
    about three decimal digits -- enough that a faithful port looks like a
    ~6e-4 relative error in the emission scores. The reference has to be the
    precision both backends can agree on.
    """
    import tensorflow as tf

    tf.config.experimental.enable_tensor_float_32_execution(False)


def build_layer(scenario: str):
    """Construct and build a TensorFlow pHMM layer for a scenario."""
    from learnMSA.hmm.tf.layer import TFPHMMLayer

    config, prior_config, struct_config = make_configs(scenario)
    layer = TFPHMMLayer(
        lengths=LENGTHS,
        config=config,
        prior_config=prior_config,
        struct_config=struct_config,
    )
    layer.build(input_shape=input_shapes(struct_config))
    return layer, struct_config


def collect() -> dict[str, np.ndarray]:
    """Record kernels, inputs and outputs for every scenario."""
    from learnMSA.util.tensor import to_numpy

    _disable_tf32()
    np.random.seed(SEED)
    fixture: dict[str, np.ndarray] = {}
    for scenario in SCENARIOS:
        layer, struct_config = build_layer(scenario)
        inputs = make_inputs(struct_config)
        for name, array in inputs.items():
            fixture[f"{scenario}/input.{name}"] = array
        for name, kernel in kernel_paths(layer).items():
            fixture[f"{scenario}/kernel.{name}"] = to_numpy(kernel)
        for name, value in reference_outputs(
            layer, scenario, inputs
        ).items():
            fixture[f"{scenario}/output.{name}"] = np.asarray(value)
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
        missing = set(fixture) ^ set(stored.files)
        if missing:
            raise SystemExit(f"fixture keys differ: {sorted(missing)}")
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
