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

``--features`` selects which input tracks a sweep probes and ``--workloads``
what the probes compute; together they name the key of ``IMPL_FACTORS`` being
measured, ``<prefix>_<workload>``. The factors are absolute per configuration,
so each value is measured on its own and no arithmetic relates them:

===================== ==========================================
``--features``        prefix in ``IMPL_FACTORS``
===================== ==========================================
``aa``                none, the bare workload name
``structure``         ``structure_*``
``language_model``    ``language_model_*``
``both``              ``language_model_and_structure_*``
===================== ==========================================

The workloads are ``train`` plus the pHMM call modes inference runs in:
``viterbi``, ``posterior`` and ``loglik``. They differ enough to deserve their
own factors -- Viterbi keeps a backtrace, the posterior runs the backward sweep
as well as the forward one, and the log-likelihood keeps only the final carry.
Each sweep additionally reports ``<prefix>_inference``, the maximum over the
measured modes, which ``IMPL_FACTORS`` uses as the fallback for any mode
without a key of its own (MEA, via ``training_util.MODE_FALLBACK``).

Only the largest batch size that fits speaks for a shape, and peak memory grows
monotonically with the batch size, so each ``(features, shape, workload)`` group
is bisected rather than walked: three probes for the default seven-rung ladder.

The struct and embedding tracks are fabricated the same way the amino acid one
is -- a 3Di ``SequenceDataset`` and an in-memory ``EmbeddingCache`` of width
``language_model.scoring_model_dim``. No protein language model is loaded:
learnMSA consumes only the *reduced* embeddings, which the cache supplies
directly.

Probes are deterministic by construction: the probe dataset is a seeded draw of
uniform-length random sequences of exactly ``S - 1`` residues, so every batch
pads to ``S`` no matter how the batch generator permutes it, and ``L`` is
pinned with ``training.length_init``.

Example::

    python util/calibrate_impl_factor.py \\
        --backend pytorch --compile off \\
        --features aa,structure,language_model,both \\
        --workloads train,viterbi,posterior,loglik \\
        -o util/impl_factor_calibration.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

# Make "python util/calibrate_impl_factor.py" work from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

# Model lengths 20 and 50 are deliberately absent. In the calibration this grid
# replaces, they were cap-bound in all 32 measured (features, phase) groups
# across both backends, so they can never constrain a factor. 100 stays: it won
# both language model inference sweeps.
DEFAULT_L: tuple[int, ...] = (100, 200, 300, 500, 700, 1000)
DEFAULT_SEQ_LEN_FACTORS: tuple[float, ...] = (1.42, 3.0)
DEFAULT_BATCH_SIZES: tuple[int, ...] = (4, 8, 32, 64, 128, 256, 512)
#: What a probe computes. ``train`` holds gradients; the rest are the pHMM
#: call modes that inference runs in. MEA is absent on purpose: it computes
#: posteriors and then decodes them, so it borrows the posterior factor
#: through ``training_util.MODE_FALLBACK`` instead of being measured.
INFERENCE_WORKLOADS: tuple[str, ...] = ("viterbi", "posterior", "loglik")
WORKLOADS: tuple[str, ...] = ("train", *INFERENCE_WORKLOADS)
DEFAULT_WORKLOADS: tuple[str, ...] = WORKLOADS
DEFAULT_FEATURES: tuple[str, ...] = ("aa",)
DEFAULT_STEPS: int = 3
# Successful probes finish in seconds; anything near this is an allocator
# thrashing on a workload that does not fit.
DEFAULT_PROBE_TIMEOUT: float = 300.0
PROBE_SEED: int = 0
MIN_PROBE_SEQUENCES: int = 64

#: The input tracks each ``--features`` value switches on, and the prefix of
#: the ``IMPL_FACTORS`` key it measures. ``aa`` carries no prefix because the
#: amino-acid-only factors are the unqualified ``train``/``inference``.
FEATURES: dict[str, tuple[bool, bool, str]] = {
    #                use_structure, use_language_model, key prefix
    "aa": (False, False, ""),
    "structure": (True, False, "structure"),
    "language_model": (False, True, "language_model"),
    "both": (True, True, "language_model_and_structure"),
}


def factor_key(features: str, workload: str) -> str:
    """The ``IMPL_FACTORS`` key a sweep over ``features`` measures.

    Args:
        features: One of the keys of :data:`FEATURES`.
        workload: One of :data:`WORKLOADS`, or ``"inference"`` for the
            aggregate fallback key.

    Returns:
        The key to paste the measured factor under, so that a report can be
        transcribed into ``IMPL_FACTORS`` without renaming anything.
    """
    if features not in FEATURES:
        raise ValueError(
            f"Unknown features '{features}'. Choose one of {sorted(FEATURES)}."
        )
    prefix = FEATURES[features][2]
    return f"{prefix}_{workload}" if prefix else workload


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

    workload: str
    batch_size: int
    num_model: int
    model_len: int
    seq_len: int
    steps: int
    backend: str
    compile_mode: str
    use_triton: bool
    features: str = "aa"

    @property
    def phase(self) -> str:
        """Whether gradients are held, which is all the clamps care about."""
        return "train" if self.workload == "train" else "inference"

    @property
    def label(self) -> str:
        """The shape this probe belongs to.

        The features are part of it: the same ``L x S`` costs a different
        amount of memory per track combination, so those measurements must
        never be aggregated together.
        """
        return f"{self.features}/L{self.model_len}xS{self.seq_len}"

    def key(self) -> str:
        return f"{self.label}/{self.workload}/B{self.batch_size}"


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


def plan_probes(args: argparse.Namespace) -> list[list[ProbeSpec]]:
    """Build the probe grid, grouped by the ladder each group searches.

    One group is one ``(features, shape, workload)`` combination, holding its
    batch size ladder in ascending order. Only the largest batch size that runs
    speaks for the group, so :func:`search_group` probes a group rather than
    every rung of its ladder.
    """
    return [
        [
            ProbeSpec(
                workload=workload,
                batch_size=batch_size,
                num_model=args.num_model,
                model_len=model_len,
                seq_len=seq_len,
                steps=args.steps,
                backend=args.backend,
                compile_mode=args.compile,
                use_triton=args.use_triton,
                features=features,
            )
            for batch_size in sorted(args.batch_sizes)
        ]
        for features in args.features
        for model_len, seq_len in args.shapes
        for workload in args.workloads
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


def search_group(
    ladder: Sequence[ProbeSpec], timeout: float, verbose: bool = False
) -> list[ProbeResult]:
    """Probe one group for the largest batch size that runs.

    Peak memory grows monotonically with the batch size, so "does this batch
    size fit" is a monotone predicate and the largest fitting rung can be
    bisected for. That costs ``ceil(log2(len(ladder) + 1))`` probes -- three for
    the default seven-rung ladder -- instead of walking the ladder from the
    bottom until it breaks. The winner is identical either way, because
    :func:`select_probes` only ever keeps the largest batch size that ran.

    Args:
        ladder: The group's probes, ascending by batch size.
        timeout: Seconds before a probe counts as a device limit.
        verbose: Echo the output of failing probes.

    Returns:
        The probes that were actually run, in the order they were run.
    """
    results: list[ProbeResult] = []
    lo, hi = 0, len(ladder) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        spec = ladder[mid]
        print(f"probing {spec.key()} ...", flush=True)
        result = run_probe(spec, timeout, verbose=verbose)
        results.append(result)
        if result.status == "ok":
            lo = mid + 1
        else:
            hi = mid - 1
    return results


def _looks_like_oom(output: str) -> bool:
    """Whether a failed probe failed for want of device memory.

    Only the label depends on this -- every non-ok status is treated the same
    way by :func:`search_group` -- but a report that calls an OOM an "error"
    invites someone to go hunting for a bug that is not there.

    The last two markers are TensorFlow's: it dumps its allocator state and can
    die on the spot without ever raising a Python-level exception, in which case
    the traceback markers above never appear.
    """
    lowered = output.lower()
    return any(marker in lowered for marker in (
        "out of memory", "outofmemoryerror", "resourceexhausted",
        "oom when allocating", "largestfreeblock",
    ))


def select_probes(
    results: Iterable[ProbeResult], workload: str, features: str = "aa"
) -> list[ProbeResult]:
    """The probe that speaks for each shape in one workload and feature set.

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
        if r.spec.workload != workload or r.spec.features != features:
            continue
        if r.status != "ok":
            continue
        best = per_shape.get(r.spec.label)
        if best is None or r.effective_batch_size > best.effective_batch_size:
            per_shape[r.spec.label] = r
    return [
        per_shape[k] for k in sorted(per_shape) if per_shape[k].informative()
    ]


def derive_factor(
    results: Iterable[ProbeResult], workload: str, features: str = "aa"
) -> float | None:
    """The recommended implementation factor for one workload and feature set.

    The maximum over the per-shape operating points, i.e. the worst case among
    the shapes that are actually memory-bound.
    """
    selected = select_probes(results, workload, features)
    return max(r.impl_factor for r in selected) if selected else None


def derive_inference_factor(
    results: Iterable[ProbeResult],
    workloads: Iterable[str],
    features: str = "aa",
) -> float | None:
    """The aggregate ``<prefix>_inference`` factor for one feature set.

    ``IMPL_FACTORS`` keeps this key as the fallback for any inference mode that
    has none of its own, so it has to be the most expensive measured mode -- a
    fallback that underestimates would hand out a batch size that does not fit.
    """
    results = list(results)
    measured = []
    for workload in workloads:
        if workload == "train":
            continue
        factor = derive_factor(results, workload, features)
        if factor is not None:
            measured.append(factor)
    return max(measured) if measured else None


def print_table(results: Sequence[ProbeResult]) -> None:
    """Print the per-probe table."""
    header = (
        f"{'features':<16}{'workload':<11}{'L':>6}{'S':>6}{'B':>6}{'B_eff':>7}"
        f"{'peak MiB':>10}{'ratio':>9}{'factor':>9}{'B_fit':>7}{'cap':>6}"
        "  status"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        s = r.spec
        note = r.status if r.status != "ok" else (
            "cap-bound" if r.cap_bound else "ok"
        )
        print(
            f"{s.features:<16}{s.workload:<11}{s.model_len:>6}{s.seq_len:>6}"
            f"{s.batch_size:>6}"
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
    for features in args.features:
        for workload in args.workloads:
            factor = derive_factor(results, workload, features)
            if factor is not None:
                report.factors[factor_key(features, workload)] = factor
        aggregate = derive_inference_factor(results, args.workloads, features)
        if aggregate is not None:
            report.factors[factor_key(features, "inference")] = aggregate
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
    """A Configuration that pins every variable the formula depends on.

    Only the two track switches are set for ``--features``; every other
    track-related setting stays at its default, which is what a real run uses
    and what the probe datasets are built to match. Note that
    ``use_language_model`` makes ``LearnMSAContext`` force
    ``trainable_insertions=False`` -- that override is part of the workload
    being measured, so it is deliberately left in place.
    """
    from learnMSA import Configuration

    use_structure, use_language_model, _ = FEATURES[spec.features]

    return Configuration(**{
        "structure": {"use_structure": use_structure},
        "language_model": {"use_language_model": use_language_model},
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
            "use_triton": spec.use_triton,
        },
    })


def make_probe_datasets(spec: ProbeSpec, config: Any) -> tuple:
    """The dataset tuple of uniform-length random sequences to probe with.

    Peak memory does not depend on the residues, only on the shape. Uniform
    lengths make the padded batch shape ``seq_len`` regardless of how the batch
    generator permutes or crops, which is what makes a probe reproducible.

    The tuple is ordered ``(amino acids, [structure], [embeddings])``, the
    order ``learnMSA.run.console.run_main`` assembles and the only one the
    model understands. All tracks share the sequence lengths, because the batch
    generator derives the crop bounds and the padded length from the first
    dataset alone.
    """
    from learnMSA.util import EmbeddingCache, EmbeddingDataset, SequenceDataset

    use_structure, use_language_model, _ = FEATURES[spec.features]
    num_seq = max(2 * spec.steps * spec.batch_size, MIN_PROBE_SEQUENCES)
    residues_per_seq = spec.seq_len - 1
    rng = np.random.default_rng(PROBE_SEED)
    seq_ids = [f"probe_{i}" for i in range(num_seq)]

    def draw(alphabet: str) -> list[str]:
        letters = np.frombuffer(alphabet.encode(), dtype="S1")
        drawn = rng.choice(letters, size=(num_seq, residues_per_seq))
        return [row.tobytes().decode() for row in drawn]

    aa_data = SequenceDataset(sequences=list(zip(
        seq_ids, draw(SequenceDataset._default_alphabet)
    )))
    datasets: tuple = (aa_data, )

    if use_structure:
        # Same construction as learnMSA.run.util.load_struct_data: an ordinary
        # SequenceDataset over the 3Di alphabet, with the amino acid remapping
        # disabled.
        datasets += (SequenceDataset(
            sequences=list(zip(
                seq_ids, draw(config.structure.structural_alphabet)
            )),
            alphabet=config.structure.structural_alphabet,
            remap=False,
        ), )

    if use_language_model:
        # learnMSA only ever sees the embeddings the scoring model has already
        # reduced to scoring_model_dim, so a filled cache of that width is a
        # complete stand-in and no protein language model has to be loaded.
        dim = config.language_model.scoring_model_dim
        seq_lens = aa_data.seq_lens
        cache = EmbeddingCache(
            seq_lens,
            dim,
            cache=rng.standard_normal(
                (int(seq_lens.sum()), dim)
            ).astype(np.float16),
        )
        datasets += (
            EmbeddingDataset(embedding_cache=cache, seq_ids=seq_ids),
        )

    return datasets


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
    datasets = make_probe_datasets(spec, config)
    context = LearnMSAContext(config=config, data=datasets[0])

    model = make_learnmsa_model(context)
    model.build(((spec.batch_size,),))
    model.compile(total_steps=spec.steps)

    baseline_bytes = backend.peak_memory_bytes()
    backend.reset_peak_memory()
    start = time.perf_counter()

    num_seq = datasets[0].num_seq
    if spec.phase == "train":
        model.fit(
            datasets,
            indices=np.arange(num_seq),
            batch_size=spec.batch_size,
            epochs=1,
            steps_per_epoch=spec.steps,
        )
        effective_batch_size = context.last_runtime_batch_size \
            or spec.batch_size
    else:
        _set_inference_mode(model, spec.workload)
        indices = np.arange(min(num_seq, spec.steps * spec.batch_size))
        predict_kwargs: dict[str, Any] = {}
        if spec.workload == "posterior":
            # Every real posterior caller reduces -- model surgery and model
            # selection both want expected state counts, not per-sequence
            # posteriors -- and the unreduced array would not fit in host
            # memory at a large batch size anyway.
            predict_kwargs["reduce"] = True
        model.predict(
            datasets,
            indices=indices,
            # A single bucket wide enough for every sequence, so the bucketing
            # cannot override the probe's batch size or padded length.
            bucket_boundaries=[spec.seq_len],
            bucket_batch_sizes=[spec.batch_size, spec.batch_size],
            **predict_kwargs,
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


def _set_inference_mode(model: Any, workload: str) -> None:
    """Select what the prediction pass computes.

    MEA is absent by design: it is not calibrated in its own right and takes
    the posterior factor through ``training_util.MODE_FALLBACK``.
    """
    modes = {
        "viterbi": model.viterbi_mode,
        "posterior": model.posterior_mode,
        "loglik": model.loglik_mode,
    }
    if workload not in modes:
        raise ValueError(
            f"'{workload}' is not an inference workload. "
            f"Choose one of {sorted(modes)}."
        )
    modes[workload]()


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
        "--triton", dest="use_triton", action="store_true",
        help="Enable Triton kernels in the probes (PyTorch only).",
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
        "--workloads", type=_workload_list, default=list(DEFAULT_WORKLOADS),
        help="Comma-separated workloads to probe: "
             f"{', '.join(WORKLOADS)}. Each one measures its own key of "
             "IMPL_FACTORS, and the inference ones additionally aggregate "
             "into the '<prefix>_inference' fallback key.",
    )
    parser.add_argument(
        "--features", type=_features_list, default=list(DEFAULT_FEATURES),
        help="Comma-separated input track combinations to probe: "
             f"{', '.join(sorted(FEATURES))}. Each one measures its own key of "
             "IMPL_FACTORS.",
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


def _workload_list(value: str) -> list[str]:
    workloads = _str_list(value)
    unknown = [w for w in workloads if w not in WORKLOADS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown workloads {unknown}. Choose from {list(WORKLOADS)}."
        )
    return workloads


def _features_list(value: str) -> list[str]:
    features = _str_list(value)
    unknown = [f for f in features if f not in FEATURES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown features {unknown}. Choose from {sorted(FEATURES)}."
        )
    return features


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

    groups = plan_probes(args)
    print(
        f"{len(groups)} groups, at most "
        f"{math.ceil(math.log2(len(args.batch_sizes) + 1))} probes each",
        flush=True,
    )

    results: list[ProbeResult] = []
    for ladder in groups:
        results.extend(
            search_group(ladder, args.probe_timeout, verbose=args.verbose)
        )

    print()
    print_table(results)
    print()

    report = build_report(args, results)
    for features in args.features:
        for workload in [*args.workloads, "inference"]:
            key = factor_key(features, workload)
            if key not in report.factors:
                continue
            if workload == "inference":
                print(
                    f"recommended IMPL_FACTORS['{args.backend}']['{key}']: "
                    f"{report.factors[key]:.3f} (max over "
                    f"{', '.join(w for w in args.workloads if w != 'train')}, "
                    "the fallback for uncalibrated modes)"
                )
                continue
            selected = select_probes(results, workload, features)
            speakers = ", ".join(
                f"{r.spec.label}@B{r.effective_batch_size}" for r in selected
            )
            print(
                f"recommended IMPL_FACTORS['{args.backend}']['{key}']: "
                f"{report.factors[key]:.3f} (from {speakers})"
            )
            for r in selected:
                if r.effective_batch_size * 2 < r.resulting_batch_size:
                    print(
                        f"  warning: {r.spec.label} was probed at "
                        f"B={r.effective_batch_size} but would run at "
                        f"B={r.resulting_batch_size}. Extend --batch-sizes for "
                        f"a tighter estimate."
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
