import sys
from collections.abc import Sequence

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import numpy as np
import torch
from hidten.hmm import HMMConfig as HidtenHMMConfig
from hidten.prior import Prior
from hidten.torch.transitioner import (T_shapelike, T_TorchTensor,
                                       TorchTransitioner, TransitionerState,
                                       TransitionMode, shared_tensor)
from hidten.torch.util import (log_zero, safe_log, tiny, to_tensor,
                               zero_row_softmax)

from learnMSA.hmm.util.transition_index_set import PHMMTransitionIndexSet
from learnMSA.hmm.util.value_set import PHMMValueSet


def logsumexp(x: T_TorchTensor, y: T_TorchTensor) -> T_TorchTensor:
    """Compute log(exp(x) + exp(y)) in a numerically stable way."""
    return torch.logsumexp(torch.stack([x, y], dim=0), dim=0)


class TorchPHMMExplicitTransitioner(TorchTransitioner):
    """A transitioner for explicit pHMMs with deletion states.
    This transitioner contains silent states and needs to be folded.

    The order of states in each head is:

    M1 ... ML I1 ... IL-1 D1 ... DL L B E C R [... T],

    where L is the number of match states in that head.
    [... T] indicates that the last position is always the terminal state, no
    matter how many states there are in the head, with optional padding states
    to fill up to the maximum number of states across all heads.
    """

    def __init__(
        self,
        values: Sequence[PHMMValueSet],
        shared_flanks: bool = False,
        allow_multi_hits: bool = True,
        **kwargs
    ) -> None:
        """
        Args:
            values: A sequence of value sets, one per head, with
                probabilities.
            shared_flanks: Whether to share transition parameters of flank
                states within each head.
            allow_multi_hits: Whether to allow multiple domain hits.
        """
        super().__init__(**kwargs)
        allow_list, value_list, shared_list = [], [], []
        start, start_values = [], []
        states = []
        # Plain ints, not numpy scalars: the lengths drive Python-level control
        # flow (range, slicing), which a graph compiler models as tensor
        # operations when they are numpy types.
        self.lengths = [int(value_set.L) for value_set in values]
        max_states = PHMMTransitionIndexSet.num_states_unfolded(
            max(self.lengths)
        )
        value_sum = 0
        for h, value_set in enumerate(values):
            index_set = PHMMTransitionIndexSet(
                value_set.L, folded=False, shared_flanks=shared_flanks
            )
            # Transitions
            # get all index pairs (i,j) for this head
            allowed_trans = index_set.as_array()  # (n, 2)

            # get and array of the same length as allowed_trans,
            # containing indices into the values of this head, accounting
            # for shared transitions
            shared_indices = index_set.shared_indices()
            # For each unique parameter index, find the first allowed
            # transition that uses it -- this is the representative transition
            # for that parameter.
            _, first_occ = np.unique(shared_indices, return_index=True)
            shared_trans = allowed_trans[first_occ]

            # Add the values without duplicates for shared transitions
            v = value_set.transitions[shared_trans[:, 0], shared_trans[:, 1]]
            value_list.append(v)

            # Keep track of which values belong to multiple, shared
            # transitions. Same length as the allow_list; indices into the
            # full value_list.
            shared_list.append(shared_indices + value_sum)
            value_sum += shared_trans.shape[0]

            # Handle negative indices (access from the end)
            allowed_trans[allowed_trans < 0] += max_states
            # Add the indices with head index
            allow_list.append(
                np.pad(allowed_trans, ((0, 0), (1, 0)), constant_values=h)
            )

            # Start distribution
            start_indices = index_set.start[:, np.newaxis]
            # Handle negative indices (access from the end)
            start_indices[start_indices < 0] += max_states
            start.append(np.pad(
                start_indices, ((0, 0), (1, 0)), constant_values=h
            ))
            start_values.append(value_set.start)

            states.append(
                PHMMTransitionIndexSet.num_states_unfolded(L=value_set.L)
            )

        # Set a custom HMMConfig for the explicit model
        # because the state count differs from the folded model
        self.hmm_config = HidtenHMMConfig(states=states)

        self.allow = np.vstack(allow_list)
        self.initializer = np.hstack(value_list)

        self.allow_start = np.vstack(start)
        self.initializer_start = np.hstack(start_values)

        self.share = np.hstack(shared_list)

        self.allow_multi_hits = allow_multi_hits

    @override
    def matrix(
        self, transition_delta: T_TorchTensor | None = None
    ) -> T_TorchTensor:
        """Override to add numerical stability to avoid numerical issues
        when folding."""
        if transition_delta is not None:
            kernel = self.kernel + transition_delta
        else:
            kernel = self.kernel
        device = kernel.device
        # Clip by epsilon in probability space to ensure that no allowed
        # transition has vanishing probability
        kernel = kernel.clamp_min(float(np.log(1e-16)))
        dense_tensor = shared_tensor(
            indices=to_tensor(self.allow, dtype=torch.int64, device=device),
            values=kernel,
            shape=(self.heads, self.max_states, self.matrix_dim),
            share=to_tensor(self.share, dtype=torch.int64, device=device),
        )
        if not self.allow_multi_hits:
            # set P(C | E) to zero
            heads, rows, cols = [], [], []
            for h, L in enumerate(self.lengths):
                idx = PHMMTransitionIndexSet(L, folded=False)
                heads.append(h)
                rows.append(idx.E)
                cols.append(idx.C)
            dense_tensor = dense_tensor.clone()
            dense_tensor[heads, rows, cols] = -1e16
        return zero_row_softmax(dense_tensor)


class TorchPHMMTransitioner(TorchTransitioner):
    """A transitioner for folded pHMMs without deletion states. Wraps an
    explicit transitioner which holds all the parameters. Overrides the matrix
    and start distribution methods to provide the folded versions.

    The order of states in each head is:

    M1 ... ML I1 ... IL-1 L C R [... T],

    where L is the number of match states in that head.
    [... T] indicates that the last position is always the terminal state, no
    matter how many states there are in the head, with optional padding states
    to fill up to the maximum number of states across all heads.
    """

    @property
    def max_states(self) -> int:
        """The maximum number of states across all heads. May be restricted
        by head_subset."""
        return self.hmm_config.max_states + 1

    @property
    def states(self) -> list[int]:
        """The number of states for each head. May be restricted
        by head_subset."""
        return [Q + 1 for Q in self.hmm_config.states]

    @property
    def prior(self) -> "Prior[T_TorchTensor] | None":
        return self.explicit_transitioner.prior

    @prior.setter
    def prior(self, prior: "Prior[T_TorchTensor]") -> None:
        self.explicit_transitioner.prior = prior

    @property
    def prior_start(self) -> "Prior[T_TorchTensor] | None":
        return self.explicit_transitioner.prior_start

    @prior_start.setter
    def prior_start(self, prior_start: "Prior[T_TorchTensor]") -> None:
        self.explicit_transitioner.prior_start = prior_start

    head_subset: Sequence[int] | None = None
    """If set, only these heads are used in computations."""

    def __setattr__(self, name: str, value) -> None:
        """Route prior assignment to the transitioner that owns the parameters.

        :meth:`hidten.torch.base.TorchLayer.__setattr__` captures ``prior``
        and ``prior_start`` before Python consults the property, because
        ``nn.Module`` would otherwise swallow a prior into ``_modules``. That
        capture would store the prior on this wrapper, where nothing reads it
        -- the folded transitioner holds no parameters of its own. Intercept
        first and forward to the explicit transitioner, matching what the
        property getters return.
        """
        if (
            name in ("prior", "prior_start")
            and isinstance(value, Prior)
            and hasattr(self, "explicit_transitioner")
        ):
            setattr(self.explicit_transitioner, name, value)
            return
        super().__setattr__(name, value)

    def __init__(
        self,
        values: Sequence[PHMMValueSet],
        shared_flanks: bool = False,
        allow_multi_hits: bool = True,
        **kwargs
    ) -> None:
        """
        Args:
            values: A sequence of value sets, one per head.
            shared_flanks: Whether to share flank parameters across heads.
            allow_multi_hits: Whether to allow multiple domain hits.
        """
        super().__init__(**kwargs)

        self.explicit_transitioner = self._make_explicit_transitioner(
            values,
            shared_flanks=shared_flanks,
            allow_multi_hits=allow_multi_hits,
        )
        # See TorchPHMMExplicitTransitioner.__init__ on why these are ints.
        self.lengths = [int(value_set.L) for value_set in values]

        # Construct allow indices for the folded models
        transitions, start = [], []
        states = []
        max_states = PHMMTransitionIndexSet.num_states_folded(
            max(self.lengths)
        )
        for h, L in enumerate(self.lengths):
            index_set = PHMMTransitionIndexSet(L, folded=True)
            # Transitions
            # get all index pairs (i,j) and add head index (h,i,j)
            indices = index_set.as_array()
            # Handle negative indices (access from the end)
            indices[indices < 0] += max_states
            transitions.append(
                np.pad(indices, ((0, 0), (1, 0)), constant_values=h)
            )

            # Start distribution
            start_indices = index_set.start[:, np.newaxis]
            # Handle negative indices (access from the end)
            start_indices[start_indices < 0] += max_states
            start.append(np.pad(
                start_indices, ((0, 0), (1, 0)), constant_values=h
            ))

            states.append(PHMMTransitionIndexSet.num_states_folded(L=L))

        self.allow = np.vstack(transitions)
        self.allow_start = np.vstack(start)

        self._build_gather_buffers()

    #: The explicit-matrix index groups the folding reads, in the order in
    #: which they are concatenated into the gather buffers.
    _EXPLICIT_GROUPS = (
        "begin_to_match", "match_to_end", "end", "match_to_match",
        "match_to_insert", "insert_to_insert", "insert_to_match",
        "left_flank", "right_flank", "unannotated", "terminal",
        "match_to_delete", "delete_to_delete", "delete_to_match",
        "begin_to_delete", "delete_to_end",
    )

    def _build_gather_buffers(self) -> None:
        """Resolve every explicit-matrix index once, as tensors.

        The indices are pure structure: they depend only on the head lengths.
        All groups of all heads are concatenated so that the folding reads the
        explicit matrix with a *single* gather. Resolving them per call instead
        repeats the same numpy work on every forward pass, and under
        ``torch.compile`` it lands in the graph as hundreds of tensor ops --
        more than half of the model's forward graph.

        The buffers are non-persistent: they are structure derived from the
        head lengths, not state, and must stay out of ``state_dict`` so that
        saved models keep loading.
        """
        max_states = PHMMTransitionIndexSet.num_states_unfolded(
            max(self.lengths)
        )
        self._group_offsets: list[dict[str, tuple[int, int]]] = []
        self._mskip_offsets: list[tuple[int, int]] = []
        heads, rows_cols, mskip_flat = [], [], []
        start, mskip_start = 0, 0

        for h, L in enumerate(self.lengths):
            index_set = PHMMTransitionIndexSet(L, folded=False)
            offsets = {}
            for group in self._EXPLICIT_GROUPS:
                indices = np.asarray(getattr(index_set, group))
                rows_cols.append(indices)
                heads.append(np.full(indices.shape[0], h))
                offsets[group] = (start, start + indices.shape[0])
                start += indices.shape[0]
            self._group_offsets.append(offsets)

            # The folded match-skip probabilities read the ragged upper
            # triangle M_skip[i, i:-1] for i in 1..L-2. One flat gather
            # replaces that Python loop, which unrolls into L-2 slices.
            flat = (
                np.concatenate([
                    np.arange(i, L - 1) + i * L for i in range(1, L - 1)
                ]) if L > 2 else np.zeros(0)
            )
            mskip_flat.append(flat)
            self._mskip_offsets.append((mskip_start, mskip_start + flat.size))
            mskip_start += flat.size

        indices = np.concatenate(rows_cols, axis=0)
        # Negative indices count from the end of the explicit matrix, which is
        # square, so resolving them here matches torch's own semantics.
        indices = np.where(indices < 0, indices + max_states, indices)
        for name, values in (
            ("_gather_heads", np.concatenate(heads)),
            ("_gather_rows", indices[:, 0]),
            ("_gather_cols", indices[:, 1]),
            ("_mskip_flat", np.concatenate(mskip_flat)),
        ):
            self.register_buffer(
                name,
                torch.as_tensor(
                    np.ascontiguousarray(values), dtype=torch.int64
                ),
                persistent=False,
            )

    def _gather_explicit(
        self, log_explicit_matrix: T_TorchTensor
    ) -> T_TorchTensor:
        """Read every index group of every head in one gather.

        Slice the result with :meth:`_group`.
        """
        return log_explicit_matrix[
            self._gather_heads, self._gather_rows, self._gather_cols
        ]

    def _group(
        self, values: T_TorchTensor, h: int, group: str
    ) -> T_TorchTensor:
        """One index group of head ``h`` out of a :meth:`_gather_explicit`
        result."""
        start, stop = self._group_offsets[h][group]
        return values[start:stop]

    def enable_multi_hits(self, enable: bool = True) -> None:
        """Enable or disable multiple hits by setting the corresponding
        transitions in the explicit transitioner."""
        self.explicit_transitioner.allow_multi_hits = enable

    @override
    def build(self, input_shape: T_shapelike | None = None) -> None:
        # don't call super().build(), as this transitioner only folds and has
        # no own parameters
        self.explicit_transitioner.build()

    @override
    def launch(
        self,
        transition_delta: T_TorchTensor | None = None,
        mode: TransitionMode = TransitionMode.SUM,
        use_padding: bool = True,  # not used
    ) -> tuple[T_TorchTensor, TransitionerState]:
        # The HMM may pass use_padding=True, but this transitioner already
        # manages the terminal/padding semantics explicitly.
        # Keep the folded matrix/start dimensions at Q (no extra padding
        # row/col).
        log_explicit_matrix = safe_log(
            self.explicit_transitioner.matrix(transition_delta)
        )
        folded_transition_probs, folded_start_probs = \
            self._compute_folded_prob_vectors(log_explicit_matrix)

        A = self._build_folded_matrix(folded_transition_probs)

        H, Q, _ = A.shape

        # Compute a starting distribution depending on the mode
        if TransitionMode.REVERSE in mode:
            start_dist = torch.ones(
                (H, Q), dtype=A.dtype, device=A.device
            )
        else:
            start_dist = self._build_folded_start_dist(folded_start_probs)

        if TransitionMode.ALLOWED in mode:
            A = torch.where(A > tiny(A), 1., 0.)
            start_dist = torch.where(
                start_dist > tiny(start_dist), 1., 0.
            )

        A_log = A_log_T = None
        if TransitionMode.LOG_SUM_EXP in mode or TransitionMode.MAX in mode:
            # Build A_log directly from the log-space folded_transition_probs
            A_log = self._build_folded_log_matrix(folded_transition_probs)
            A_log_T = A_log.transpose(-2, -1)

        return start_dist, TransitionerState(A, A_log, A_log_T, mode)

    def _build_folded_log_matrix(
        self, folded_transition_log_probs: T_TorchTensor
    ) -> T_TorchTensor:
        """Build the folded transition matrix directly in log-space.

        Identical to _build_folded_matrix but works in log-space throughout,
        avoiding the exp->log roundtrip that loses precision for very small
        probabilities (e.g. long deletion chains).
        """
        device = folded_transition_log_probs.device
        log_matrix = shared_tensor(
            indices=to_tensor(self.allow, dtype=torch.int64, device=device),
            values=folded_transition_log_probs,
            shape=(self.heads, self.max_states, self.matrix_dim),
            share=None,
        )
        if self.head_subset is not None:
            log_matrix = log_matrix.index_select(
                0, self._head_subset_tensor(device)
            )
            max_states_subset = max(
                [self.hmm_config.states[h] for h in self.head_subset]
            )
            terminal_log_in = log_matrix[:, :max_states_subset, -1:]
            # Terminal state emits to itself with probability 1 (log = 0)
            terminal_log_out = safe_log(
                torch.nn.functional.one_hot(
                    torch.tensor([[max_states_subset]], device=device),
                    num_classes=max_states_subset + 1,
                ).to(log_matrix.dtype)
            )
            log_matrix = log_matrix[:, :max_states_subset, :max_states_subset]
            log_matrix = torch.cat([log_matrix, terminal_log_in], dim=2)
            log_matrix = torch.cat([log_matrix, terminal_log_out], dim=1)
        return log_matrix

    def _build_folded_matrix(
        self, folded_transition_probs: T_TorchTensor
    ) -> T_TorchTensor:
        device = folded_transition_probs.device
        matrix = torch.exp(shared_tensor(
            indices=to_tensor(self.allow, dtype=torch.int64, device=device),
            values=folded_transition_probs,
            shape=(self.heads, self.max_states, self.matrix_dim),
            share=None,
        ))
        if self.head_subset is not None:
            matrix = matrix.index_select(0, self._head_subset_tensor(device))
            max_states_subset = max(
                [self.hmm_config.states[h] for h in self.head_subset]
            )
            terminal_state_in = matrix[:, :max_states_subset, -1:]
            terminal_state_out = torch.nn.functional.one_hot(
                torch.tensor([[max_states_subset]], device=device),
                num_classes=max_states_subset + 1,
            ).to(matrix.dtype)
            matrix = matrix[:, :max_states_subset, :max_states_subset]
            matrix = torch.cat([matrix, terminal_state_in], dim=2)
            matrix = torch.cat([matrix, terminal_state_out], dim=1)
        return matrix

    def _build_folded_start_dist(
        self, folded_start_probs: T_TorchTensor
    ) -> T_TorchTensor:
        device = folded_start_probs.device
        start_dist = torch.exp(shared_tensor(
            indices=to_tensor(
                self.allow_start, dtype=torch.int64, device=device
            ),
            values=folded_start_probs,
            shape=(self.heads, self.max_states),
            share=None,
        ))
        if self.head_subset is not None:
            start_dist = start_dist.index_select(
                0, self._head_subset_tensor(device)
            )
            max_states_subset = max(
                [self.hmm_config.states[h] for h in self.head_subset]
            )
            terminal_state = start_dist[:, -1:]
            start_dist = start_dist[:, :max_states_subset]
            start_dist = torch.cat([start_dist, terminal_state], dim=1)
        return start_dist

    @override
    def matrix(
        self, transition_delta: T_TorchTensor | None = None
    ) -> T_TorchTensor:
        log_explicit_matrix = safe_log(
            self.explicit_transitioner.matrix(transition_delta)
        )
        folded_transition_probs, _ = self._compute_folded_prob_vectors(
            log_explicit_matrix
        )
        return self._build_folded_matrix(folded_transition_probs)

    @override
    def start_dist(self) -> T_TorchTensor:
        log_explicit_matrix = safe_log(self.explicit_transitioner.matrix())
        _, folded_start_probs = self._compute_folded_prob_vectors(
            log_explicit_matrix
        )
        return self._build_folded_start_dist(folded_start_probs)

    @override
    def prior_scores(self) -> T_TorchTensor:
        return self.explicit_transitioner.prior_scores()

    def _head_subset_tensor(self, device: torch.device) -> T_TorchTensor:
        return to_tensor(
            np.asarray(self.head_subset), dtype=torch.int64, device=device
        )

    def _compute_folded_prob_vectors(
        self,
        log_explicit_matrix: T_TorchTensor,
    ) -> tuple[T_TorchTensor, T_TorchTensor]:
        """Compute folded transition and start log-probabilities in one pass.

        Returns:
            Tuple of flat tensors (transition_probs, start_probs), each ordered
            according to self.allow and self.allow_start respectively.
        """
        explicit_start = safe_log(self.explicit_transitioner.start_dist())
        gathered = self._gather_explicit(log_explicit_matrix)

        folded_transition_probs = []
        folded_start_probs = []

        for h, L in enumerate(self.lengths):
            log_mat = log_explicit_matrix[h]
            M_skip = self._compute_match_skip_matrix(h, gathered=gathered)

            def get(group, h=h):
                return self._group(gathered, h, group)

            BM = get("begin_to_match")
            ME = get("match_to_end")
            E = get("end")

            log_z = log_zero(log_mat)
            entry_add = logsumexp(
                BM,
                torch.nn.functional.pad(
                    M_skip[0, :-1], (1, 0), value=log_z
                ),
            )
            exit_add = logsumexp(
                ME,
                torch.nn.functional.pad(
                    M_skip[1:, -1], (0, 1), value=log_z
                ),
            )

            # Folded transition probabilities
            MM = get("match_to_match")
            MI = get("match_to_insert")
            II = get("insert_to_insert")
            IM = get("insert_to_match")
            folded_transition_probs.extend([MM, MI, II, IM])

            # M_skip[i, i:-1] for i in 1..L-2, read in one gather.
            mskip_start, mskip_stop = self._mskip_offsets[h]
            if mskip_stop > mskip_start:
                folded_transition_probs.append(
                    M_skip.reshape(-1)[
                        self._mskip_flat[mskip_start:mskip_stop]
                    ]
                )

            MU = exit_add + E[0]
            MR = exit_add + E[1]
            MT = exit_add + E[2]
            folded_transition_probs.extend([MU, MR, MT])

            LF = get("left_flank")
            folded_transition_probs.append(LF[:1])
            folded_transition_probs.append(LF[1:2] + entry_add)

            LFE = LF[1:2] + M_skip[0, -1]
            folded_transition_probs.append(LFE + E[0])
            folded_transition_probs.append(LFE + E[1])
            folded_transition_probs.append(LFE + E[2])

            p_right_flank = get("right_flank")
            folded_transition_probs.extend([
                p_right_flank[0].unsqueeze(0),
                p_right_flank[1].unsqueeze(0),
            ])

            U = get("unannotated")
            UE = U[1] + M_skip[0, -1]
            UU = logsumexp(U[0], UE + E[0])
            folded_transition_probs.append(UU.unsqueeze(0))
            folded_transition_probs.append(U[1] + entry_add)
            folded_transition_probs.append((UE + E[1]).unsqueeze(0))
            folded_transition_probs.append((UE + E[2]).unsqueeze(0))

            folded_transition_probs.append(get("terminal"))

            # Folded start probabilities
            start_left = explicit_start[h, 3 * L - 1]
            start_begin = explicit_start[h, 3 * L]
            BE = start_begin + M_skip[0, -1]

            folded_start_probs.append(start_begin + entry_add)
            folded_start_probs.append(start_left.unsqueeze(0))
            folded_start_probs.append((BE + E[0]).unsqueeze(0))
            folded_start_probs.append((BE + E[1]).unsqueeze(0))
            folded_start_probs.append((BE + E[2]).unsqueeze(0))

        return (
            torch.cat(folded_transition_probs, dim=0),
            torch.cat(folded_start_probs, dim=0),
        )

    def _compute_match_skip_matrix(
        self,
        h: int,
        gathered: T_TorchTensor | None = None,
    ) -> T_TorchTensor:
        """
        Utility method that computes the `L x L` match skip transition matrix
        for head `h` with `match_skip(i,j) = P(Mj+2 | Mi)`.
        With `M0 := Begin` and `ML+1 := End`.

        Args:
            gathered: The result of :meth:`_gather_explicit`, if the caller has
                one already. Otherwise the explicit matrix is read here.
        """
        if gathered is None:
            gathered = self._gather_explicit(
                safe_log(self.explicit_transitioner.matrix())
            )

        def get(group):
            return self._group(gathered, h, group)

        # Get transition log probabilities
        MD = get("match_to_delete")  # Shape: (L-1,)
        DD = get("delete_to_delete")  # Shape: (L-1,)
        DM = get("delete_to_match")  # Shape: (L-1,)

        # Concatenate B -> D1 and DL -> E transitions
        begin_to_delete = get("begin_to_delete")  # Shape: scalar
        delete_to_end = get("delete_to_end")  # Shape: scalar

        MD = torch.cat([begin_to_delete, MD], dim=0)  # Shape: (L,)
        DM = torch.cat([DM, delete_to_end], dim=0)    # Shape: (L,)

        # Compute cumulative sum of delete-to-delete transitions
        # Prepend 0 for the first delete state D_0
        # Shape: (L,)
        DD_cumsum = torch.nn.functional.pad(
            torch.cumsum(DD, dim=0), (1, 0), value=0.0
        )

        # Compute the difference matrix for cumulative sums
        # Shape: (L, L)
        DD_diff = DD_cumsum.unsqueeze(0) - DD_cumsum.unsqueeze(1)

        # Build M_skip matrix
        MD_expanded = MD.unsqueeze(-1)  # Shape: (L, 1)
        DM_expanded = DM.unsqueeze(0)  # Shape: (1, L)

        M_skip = MD_expanded + DD_diff + DM_expanded  # Shape: (L, L)
        return M_skip

    def _make_explicit_transitioner(
        self,
        values: Sequence[PHMMValueSet],
        shared_flanks: bool = False,
        allow_multi_hits: bool = False,
    ) -> TorchPHMMExplicitTransitioner:
        """Helper to create the explicit transitioner with the same
        parameters."""
        return TorchPHMMExplicitTransitioner(
            values=values,
            shared_flanks=shared_flanks,
            allow_multi_hits=allow_multi_hits,
        )
