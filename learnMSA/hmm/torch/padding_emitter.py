from collections.abc import Sequence

from hidten.torch.emitter import TorchPaddingEmitter


class TorchSubsetPaddingEmitter(TorchPaddingEmitter):
    """A padding emitter that supports head subsets."""

    head_subset: Sequence[int] | None = None
    """If set, only these heads are used in computations."""

    @property
    def max_states(self) -> int:
        """The maximum number of states across all heads."""
        if self.head_subset is not None:
            return max(self.hmm_config.states[i] for i in self.head_subset)
        return self.hmm_config.max_states
