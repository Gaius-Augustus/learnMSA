from typing import Literal

from pydantic import BaseModel, field_validator


class AdvancedConfig(BaseModel):
    """Advanced/Development parameters."""

    backend: Literal["auto", "tensorflow", "pytorch"] = "auto"
    """Tensor framework used for training and decoding. ``auto`` picks whichever
    framework is installed, preferring TensorFlow when both are available."""

    compile: Literal["auto", "on", "off", "jit"] = "off"
    """Graph compilation policy. ``auto`` compiles only when the run is long
    enough to pay for it. ``on`` always builds a graph (``tf.function`` without
    XLA under TensorFlow, ``torch.compile(fullgraph=True)`` under PyTorch),
    ``off`` never does, and ``jit`` forces XLA -- TensorFlow only."""

    use_triton: bool = False
    """Whether to run the HMM's forward and Viterbi recursions with the Triton
    kernels instead of the plain torch scan. PyTorch only; the kernels are
    usually faster, but they are opt-in because they are not available
    everywhere and can misbehave."""

    dist_out: str = ""
    """Distribution output file."""

    initial_distance: float = 0.05
    """Initial distance value. Is only used when not using sequence weights.
    If weights are used, the initial distance is calculated from the cluster
    sequence identity."""

    insertion_aligner: str = "famsa"
    """Insertion aligner to use."""

    aligner_threads: int = 0
    """Number of threads to use for the aligner."""

    one_dnn_opts: bool = False
    """Whether to use oneDNN optimizations in TensorFlow. This can improve
    performance on CPUs. Set the to false per default as for HMMs it caused
    numerical instabilities on many CPU."""

    reset_branch_lengths: bool = True
    """Whether to reset the branch lengths (tau) before training."""

    reset_evo_model: bool = False
    """Whether to reset the evolutionary model parameters (exchangeabilities, equilibrium, and mixture) before training."""

    @field_validator(
        "initial_distance",
    )
    def validate_quantiles(cls, v: float, info) -> float:
        if not v > 0:
            raise ValueError(f"{info.field_name} must be greater than 0.")
        return v
