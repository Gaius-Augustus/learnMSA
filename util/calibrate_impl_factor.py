#!/usr/bin/env python3
"""Calibrate the backend implementation factors for adaptive batch sizing.

learnMSA picks a batch size with
:func:`learnMSA.model.training_util.get_adaptive_batch_size`::

    B = floor(safety_margin * mem
              / (num_model_factor * L * S * f * dtype_size))

where ``L`` is the maximum model length, ``S`` the padded sequence length and
``f`` the implementation factor of the backend. Writing

    K     = num_model_factor(num_model) * L * S         # units of work
    ratio = peak_device_memory / (K * B)                # bytes per unit

and substituting the predicted ``B`` back into the formula shows that the run
consumes exactly ``safety_margin * mem`` when

    f = ratio / dtype_size

So the implementation factor is the measured memory cost per unit of work --
how many ``L x S`` values the implementation keeps live, per model and per
sequence -- and ``safety_margin`` is what pays for the CUDA context, allocator
fragmentation and the batch-independent part of the footprint.

This script measures ``ratio``. Peak memory depends only on the workload shape
``(num_model, L, S, B)`` and the phase, never on the residues themselves, so
the probes run on synthetic sequences: for every ``(L, S)`` of a predefined
grid the script runs short probes at a ladder of batch sizes, in a fresh child
process each, and reports the factor each probe implies plus a recommended
aggregate. Nothing here is specific to a backend: pass ``--backend tensorflow``
to recalibrate the TensorFlow column with the same code.

Probes are deterministic by construction: the probe dataset is a seeded draw of
uniform-length random sequences of exactly ``S - 1`` residues, so every batch
pads to ``S`` no matter how the batch generator permutes it, and ``L`` is
pinned with ``training.length_init``.

Example::

    python util/calibrate_impl_factor.py \\
        --backend pytorch --compile off --no-triton \\
        -o util/impl_factor_calibration.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

# Make "python util/calibrate_impl_factor.py" work from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

DEFAULT_L: tuple[int, ...] = (20, 50, 100, 200, 300, 500, 700, 1000)
DEFAULT_SEQ_LEN_FACTORS: tuple[float, ...] = (1.42, 3.0)
DEFAULT_BATCH_SIZES: tuple[int, ...] = (4, 8, 32, 64, 128, 256, 512)
DEFAULT_PHASES: tuple[str, ...] = ("train", "inference")
DEFAULT_STEPS: int = 3
# Successful probes finish in seconds; anything near this is an allocator
# thrashing on a workload that does not fit.
DEFAULT_PROBE_TIMEOUT: float = 300.0
PROBE_SEED: int = 0
MIN_PROBE_SEQUENCES: int = 64


def default_shapes() -> list[tuple[int, int]]:
    """Every model length at every sequence length factor.

    1.42 is the ratio a real family produces; 3.0 covers families whose padded
    length is set by long outlier sequences.
    """
    return [
        (model_len, round(model_len * factor))
        for model_len in DEFAULT_L
        for factor in DEFAULT_SEQ_LEN_FACTORS
    ]


@dataclass
class ProbeSpec:
    """One measurement: a workload of known shape at a known batch size."""

    phase: str
    batch_size: int
    num_model: int
    model_len: int
    seq_len: int
    steps: int
    backend: str
    compile_mode: str
    no_triton: bool
    inference_mode: str = "viterbi"

    @property
    def label(self) -> str:
        """The shape this probe belongs to."""
        return f"L{self.model_len}xS{self.seq_len}"

    def key(self) -> str:
        return f"{self.label}/{self.phase}/B{self.batch_size}"


@dataclass
class ProbeResult:
    """The outcome of a :class:`ProbeSpec`."""

    spec: ProbeSpec
    status: str = "ok"
    effective_batch_size: int = 0
    baseline_bytes: int = 0
    peak_bytes: int = 0
    ratio: float = 0.0
    impl_factor: float = 0.0
    resulting_batch_size: int = 0
    batch_size_cap: int = 0
    cap_bound: bool = True
    seconds: float = 0.0
    message: str = ""

    def informative(self) -> bool:
        """True if this probe constrains the implementation factor."""
        return self.status == "ok" and not self.cap_bound


@dataclass
class CalibrationReport:
    """Everything needed to compare two calibration runs."""

    backend: str
    framework_version: str
    learnmsa_version: str
    device_name: str
    device_memory_bytes: float
    command: list[str] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    factors: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Formula bookkeeping. Kept in one place so the parent and the child agree.
# ---------------------------------------------------------------------------


def work_units(num_model: int, model_len: int, seq_len: int) -> float:
    """The ``num_model_factor * L * S`` term of the batch size formula."""
    num_model_factor = num_model if num_model <= 4 else num_model ** 1.1
    return num_model_factor * float(model_len) * float(seq_len)


def impl_factor_from_peak(
    peak_bytes: int,
    batch_size: int,
    num_model: int,
    model_len: int,
    seq_len: int,
    data_type_size: int = 4,
) -> tuple[float, float]:
    """Invert the batch size formula.

    Args:
        peak_bytes: Peak device memory the probe held.
        batch_size: The batch size the probe ran at.
        num_model: Number of pHMM heads.
        model_len: Maximum number of match states.
        seq_len: Padded sequence length.
        data_type_size: Size of the data type in bytes.

    Returns:
        A tuple ``(ratio, impl_factor)`` of bytes per unit of work and the
        implementation factor that reproduces it.
    """
    units = work_units(num_model, model_len, seq_len) * float(batch_size)
    if units <= 0.0:
        return 0.0, 0.0
    ratio = float(peak_bytes) / units
    return ratio, ratio / data_type_size


def batch_size_cap(seq_len: int, phase: str) -> int:
    """The largest batch size that is not memory-bound for this workload.

    Mirrors the clamps in :func:`get_adaptive_batch_size` plus the additional
    ``min(batch_size, 512)`` the training loop applies.
    """
    from learnMSA.model.training_util import MAX_BATCH_SIZE, MAX_TOKENS_PER_BATCH

    cap = min(MAX_BATCH_SIZE, max(1, MAX_TOKENS_PER_BATCH // seq_len))
    if phase == "train":
        cap = min(cap, 512)
    return cap


# ---------------------------------------------------------------------------
# Parent process: probe planning, child dispatch, aggregation, reporting.
# ---------------------------------------------------------------------------


def plan_probes(args: argparse.Namespace) -> list[ProbeSpec]:
    """Build the full probe grid."""
    return [
        ProbeSpec(
            phase=phase,
            batch_size=batch_size,
            num_model=args.num_model,
            model_len=model_len,
            seq_len=seq_len,
            steps=args.steps,
            backend=args.backend,
            compile_mode=args.compile,
            no_triton=args.no_triton,
            inference_mode=args.inference_mode,
        )
        for model_len, seq_len in args.shapes
        for phase in args.phases
        for batch_size in args.batch_sizes
    ]


def run_probe(
    spec: ProbeSpec, timeout: float, verbose: bool = False
) -> ProbeResult:
    """Run one probe in a child process and return its result.

    The child is isolated so that the framework's peak-memory counters start
    clean, a fragmented allocator cannot leak into the next probe, and an OOM
    only kills the probe.

    The timeout is what makes an over-large workload survivable. PyTorch raises
    on an allocation it cannot serve, but TensorFlow retries inside its
    allocator and can sit at 0% utilization holding the whole device
    indefinitely. A probe that outlives the timeout is treated exactly like an
    OOM: it marks the device limit for that shape and stops its ladder.
    """
    import tempfile

    with tempfile.TemporaryDirectory(prefix="learnmsa_probe_") as tmp:
        spec_path = Path(tmp) / "spec.json"
        result_path = Path(tmp) / "result.json"
        spec_path.write_text(json.dumps(asdict(spec)))
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--_probe", str(spec_path), "--_result", str(result_path),
        ]
        env = dict(os.environ)
        env["LEARNMSA_BACKEND"] = spec.backend
        try:
            proc = subprocess.run(
                cmd, env=env, capture_output=True, text=True, check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return ProbeResult(
                spec=spec,
                status="timeout",
                message=f"no result after {timeout:.0f}s",
            )
        if result_path.is_file():
            payload = json.loads(result_path.read_text())
            payload["spec"] = spec
            return ProbeResult(**payload)

    status = "oom" if _looks_like_oom(proc.stdout + proc.stderr) else "error"
    tail = (proc.stderr or proc.stdout).strip().splitlines()
    if verbose and tail:
        print("\n".join(tail[-15:]), file=sys.stderr)
    return ProbeResult(
        spec=spec,
        status=status,
        message=tail[-1] if tail else f"exit code {proc.returncode}",
    )


def _looks_like_oom(output: str) -> bool:
    lowered = output.lower()
    return any(marker in lowered for marker in (
        "out of memory", "outofmemoryerror", "resourceexhausted",
    ))


def select_probes(
    results: Iterable[ProbeResult], phase: str
) -> list[ProbeResult]:
    """The probe that speaks for each shape in one phase.

    Two filters, and the order matters. First pick the largest batch size that
    ran: the measured ratio falls with the batch size, because the
    batch-independent part of the footprint (weights, optimizer state, CUDA
    workspaces) is amortized over more sequences, and learnMSA picks the
    largest batch that fits. Only then ask whether that probe is cap-bound.

    Testing cap-bound-ness per probe instead would invert the selection: at a
    small batch size the inflated factor shrinks the predicted batch size below
    the cap, so a capped shape looks memory-bound exactly at the batch sizes
    whose measurement is worthless, and its trustworthy large-batch probes are
    the ones discarded.
    """
    per_shape: dict[str, ProbeResult] = {}
    for r in results:
        if r.spec.phase != phase or r.status != "ok":
            continue
        best = per_shape.get(r.spec.label)
        if best is None or r.effective_batch_size > best.effective_batch_size:
            per_shape[r.spec.label] = r
    return [
        per_shape[k] for k in sorted(per_shape) if per_shape[k].informative()
    ]


def derive_factor(results: Iterable[ProbeResult], phase: str) -> float | None:
    """The recommended implementation factor for one phase.

    The maximum over the per-shape operating points, i.e. the worst case among
    the workloads that are actually memory-bound.
    """
    selected = select_probes(results, phase)
    return max(r.impl_factor for r in selected) if selected else None


def print_table(results: Sequence[ProbeResult]) -> None:
    """Print the per-probe table."""
    header = (
        f"{'phase':<11}{'L':>6}{'S':>6}{'B':>6}{'B_eff':>7}{'peak MiB':>10}"
        f"{'ratio':>9}{'factor':>9}{'B_fit':>7}{'cap':>6}  status"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        s = r.spec
        note = r.status if r.status != "ok" else (
            "cap-bound" if r.cap_bound else "ok"
        )
        print(
            f"{s.phase:<11}{s.model_len:>6}{s.seq_len:>6}{s.batch_size:>6}"
            f"{r.effective_batch_size:>7}{r.peak_bytes / 1024 ** 2:>10.0f}"
            f"{r.ratio:>9.4f}{r.impl_factor:>9.3f}"
            f"{r.resulting_batch_size:>7}{r.batch_size_cap:>6}  {note}"
        )


def build_report(
    args: argparse.Namespace,
    results: Sequence[ProbeResult],
) -> CalibrationReport:
    """Assemble the JSON report."""
    import learnMSA.backend as backend
    from learnMSA.run.util import get_avail_memory_bytes, get_gpu_memory

    device_name = "unknown"
    try:
        import subprocess as sp
        device_name = sp.check_output(
            "nvidia-smi --query-gpu=name --format=csv,noheader".split()
        ).decode().strip().splitlines()[0]
    except Exception:  # nvidia-smi is optional
        pass

    report = CalibrationReport(
        backend=args.backend,
        framework_version=backend.framework_version(),
        learnmsa_version=_learnmsa_version(),
        device_name=device_name,
        device_memory_bytes=(
            float(get_gpu_memory()[0]) * 1e6 if get_gpu_memory()
            else get_avail_memory_bytes()
        ),
        command=sys.argv,
        results=[
            {**{k: v for k, v in asdict(r).items() if k != "spec"},
             "spec": asdict(r.spec)}
            for r in results
        ],
    )
    for phase in args.phases:
        factor = derive_factor(results, phase)
        if factor is not None:
            report.factors[phase] = factor
    return report


def _learnmsa_version() -> str:
    try:
        from learnMSA.run.util import get_version
        return get_version()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Child process: one probe.
# ---------------------------------------------------------------------------


def make_probe_config(spec: ProbeSpec) -> Any:
    """A Configuration that pins every variable the formula depends on."""
    from learnMSA import Configuration

    return Configuration(**{
        "training": {
            # length_init pins L and implies num_model.
            "length_init": [spec.model_len] * spec.num_model,
            "auto_crop": False,
            "crop": spec.seq_len - 1,
            "max_iterations": 1,
            # mmseqs clustering costs minutes and no device memory. It is also
            # the only consumer of input_output.input_file, which stays unset.
            "no_sequence_weights": True,
        },
        "advanced": {
            "backend": spec.backend,
            "compile": spec.compile_mode,
            "no_triton": spec.no_triton,
        },
    })


def make_probe_dataset(spec: ProbeSpec) -> Any:
    """A dataset of uniform-length random sequences.

    Peak memory does not depend on the residues, only on the shape. Uniform
    lengths make the padded batch shape ``seq_len`` regardless of how the batch
    generator permutes or crops, which is what makes a probe reproducible.
    """
    from learnMSA.util.sequence_dataset import SequenceDataset

    alphabet = np.frombuffer(
        SequenceDataset._default_alphabet.encode(), dtype="S1"
    )
    num_seq = max(2 * spec.steps * spec.batch_size, MIN_PROBE_SEQUENCES)
    rng = np.random.default_rng(PROBE_SEED)
    residues = rng.choice(alphabet, size=(num_seq, spec.seq_len - 1))
    return SequenceDataset(sequences=[
        (f"probe_{i}", row.tobytes().decode())
        for i, row in enumerate(residues)
    ])


def probe_main(spec: ProbeSpec, result_path: Path) -> None:
    """Run one probe and write its result."""
    import time

    import learnMSA.backend as backend
    from learnMSA.backend import set_backend

    set_backend(spec.backend)

    from learnMSA.model.context import LearnMSAContext
    from learnMSA.model.model import make_learnmsa_model
    from learnMSA.run.util import setup_devices

    setup_devices("default", verbose=False, one_dnn_opts=False)
    if backend.num_gpus() == 0:
        raise RuntimeError(
            "No GPU visible. The calibration is only meaningful on a GPU."
        )

    config = make_probe_config(spec)
    data = make_probe_dataset(spec)
    context = LearnMSAContext(config=config, data=data)

    model = make_learnmsa_model(context)
    model.build(((spec.batch_size,),))
    model.compile(total_steps=spec.steps)

    baseline_bytes = backend.peak_memory_bytes()
    backend.reset_peak_memory()
    start = time.perf_counter()

    if spec.phase == "train":
        model.fit(
            data,
            indices=np.arange(data.num_seq),
            batch_size=spec.batch_size,
            epochs=1,
            steps_per_epoch=spec.steps,
        )
        effective_batch_size = context.last_runtime_batch_size \
            or spec.batch_size
    else:
        _set_inference_mode(model, spec.inference_mode)
        indices = np.arange(min(data.num_seq, spec.steps * spec.batch_size))
        model.predict(
            data,
            indices=indices,
            # A single bucket wide enough for every sequence, so the bucketing
            # cannot override the probe's batch size or padded length.
            bucket_boundaries=[spec.seq_len],
            bucket_batch_sizes=[spec.batch_size, spec.batch_size],
        )
        effective_batch_size = spec.batch_size

    seconds = time.perf_counter() - start
    peak_bytes = backend.peak_memory_bytes()

    ratio, impl_factor = impl_factor_from_peak(
        peak_bytes,
        effective_batch_size,
        spec.num_model,
        spec.model_len,
        spec.seq_len,
    )
    cap = batch_size_cap(spec.seq_len, spec.phase)
    resulting = _resulting_batch_size(spec, impl_factor)

    result = {
        "status": "ok",
        "effective_batch_size": int(effective_batch_size),
        "baseline_bytes": int(baseline_bytes),
        "peak_bytes": int(peak_bytes),
        "ratio": ratio,
        "impl_factor": impl_factor,
        "resulting_batch_size": int(resulting),
        "batch_size_cap": int(cap),
        "cap_bound": bool(resulting >= cap),
        "seconds": seconds,
        "message": "",
    }
    result_path.write_text(json.dumps(result))


def _set_inference_mode(model: Any, mode: str) -> None:
    """Select what the prediction pass computes."""
    modes = {
        "viterbi": model.viterbi_mode,
        "mea": model.mea_mode,
        "posterior": model.posterior_mode,
        "loglik": model.loglik_mode,
    }
    if mode not in modes:
        raise ValueError(
            f"Unknown inference mode '{mode}'. "
            f"Choose one of {sorted(modes)}."
        )
    modes[mode]()


def _resulting_batch_size(spec: ProbeSpec, impl_factor: float) -> int:
    """The batch size the formula would pick with the measured factor."""
    from learnMSA.model.training_util import get_adaptive_batch_size

    batch_size = get_adaptive_batch_size(
        model_len=spec.model_len,
        num_model=spec.num_model,
        seq_len=spec.seq_len,
        impl_factor=impl_factor,
    )
    if spec.phase == "train":
        batch_size = min(batch_size, 512)
    return batch_size


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backend", choices=["tensorflow", "pytorch"], default="pytorch",
        help="Backend to calibrate.",
    )
    parser.add_argument(
        "--compile", choices=["auto", "on", "off", "jit"], default="off",
        help="Value of advanced.compile used for the probes.",
    )
    parser.add_argument(
        "--no-triton", action="store_true",
        help="Disable Triton kernels in the probes (PyTorch only).",
    )
    parser.add_argument(
        "--num-model", type=int, default=4,
        help="Number of pHMM heads to probe with.",
    )
    parser.add_argument(
        "--batch-sizes", type=_int_list, default=list(DEFAULT_BATCH_SIZES),
        help="Comma-separated batch size ladder, ascending.",
    )
    parser.add_argument(
        "--phases", type=_str_list, default=list(DEFAULT_PHASES),
        help="Comma-separated phases to probe: train and/or inference.",
    )
    parser.add_argument(
        "--steps", type=int, default=DEFAULT_STEPS,
        help="Number of batches per probe.",
    )
    parser.add_argument(
        "--probe-timeout", type=float, default=DEFAULT_PROBE_TIMEOUT,
        help="Seconds before a probe is killed and counted as a device limit. "
             "Guards against allocators that thrash instead of raising.",
    )
    parser.add_argument(
        "--shapes", type=_shape_list, default=default_shapes(),
        help="Comma-separated L:S pairs that override the default grid, e.g. "
             "'330:465,360:505'.",
    )
    parser.add_argument(
        "--inference-mode", default="viterbi",
        choices=["viterbi", "mea", "posterior", "loglik"],
        help="What the inference probes compute.",
    )
    parser.add_argument(
        "-o", "--output", default="",
        help="Path of the JSON report.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Echo the output of failing probes.",
    )
    # Internal: the child process entry point.
    parser.add_argument("--_probe", default="", help=argparse.SUPPRESS)
    parser.add_argument("--_result", default="", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def _int_list(value: str) -> list[int]:
    return [int(v) for v in value.replace(" ", "").split(",") if v]


def _str_list(value: str) -> list[str]:
    return [v for v in value.replace(" ", "").split(",") if v]


def _shape_list(value: str) -> list[tuple[int, int]]:
    shapes = []
    for item in value.replace(" ", "").split(","):
        if not item:
            continue
        model_len, _, seq_len = item.partition(":")
        if not seq_len:
            raise argparse.ArgumentTypeError(
                f"'{item}' is not a model_len:seq_len pair."
            )
        shapes.append((int(model_len), int(seq_len)))
    return shapes


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args._probe:
        spec = ProbeSpec(**json.loads(Path(args._probe).read_text()))
        probe_main(spec, Path(args._result))
        return 0

    # Keep the parent out of the other framework: it only reports versions,
    # but auto-detection would import whichever framework it finds first.
    os.environ["LEARNMSA_BACKEND"] = args.backend

    specs = plan_probes(args)

    results: list[ProbeResult] = []
    stopped: set[tuple[str, str]] = set()
    for spec in specs:
        group = (spec.label, spec.phase)
        if group in stopped:
            # A smaller batch size already hit the device limit.
            results.append(ProbeResult(spec=spec, status="skipped"))
            continue
        print(f"probing {spec.key()} ...", flush=True)
        result = run_probe(spec, args.probe_timeout, verbose=args.verbose)
        if result.status != "ok":
            stopped.add(group)
        results.append(result)

    print()
    print_table(results)
    print()

    report = build_report(args, results)
    for phase, factor in report.factors.items():
        selected = select_probes(results, phase)
        speakers = ", ".join(
            f"{r.spec.label}@B{r.effective_batch_size}" for r in selected
        )
        print(
            f"recommended {phase} implementation factor: {factor:.3f} "
            f"(from {speakers})"
        )
        for r in selected:
            if r.effective_batch_size * 2 < r.resulting_batch_size:
                print(
                    f"  warning: {r.spec.label} was probed at "
                    f"B={r.effective_batch_size} but would run at "
                    f"B={r.resulting_batch_size}. Extend --batch-sizes for a "
                    f"tighter estimate."
                )
    if not report.factors:
        print(
            "No informative probe: every workload was cap-bound. Probe longer "
            "sequences or raise --num-model."
        )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(asdict(report), indent=2))
        print(f"\nreport written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
