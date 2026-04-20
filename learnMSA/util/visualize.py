import imageio
import numpy as np
from hidten.visualize import Figure, SubFigure, plot_emissions, plot_transition_graph
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from learnMSA.util import SequenceDataset
from learnMSA.hmm.tf.layer import PHMMLayer


def phmm_layout(
    L: int,
    h_spacing: float = 1.5,
    v_spacing: float = 1.5,
) -> tuple[dict, list[str]]:
    """Compute node positions and labels for an explicit (unfolded) profile HMM
    with ``L`` match states.

    State index convention (matches :class:`PHMMTransitionIndexSet` unfolded):

    * ``0 … L-1``      : M1…ML  (match states)
    * ``L … 2L-2``     : I1…IL-1 (insert states)
    * ``2L-1 … 3L-2``  : D1…DL  (delete states)
    * ``3L-1``         : L  (left flank)
    * ``3L``           : B  (begin)
    * ``3L+1``         : E  (end)
    * ``3L+2``         : C  (unannotated)
    * ``3L+3``         : R  (right flank)
    * ``3L+4``         : T  (terminal)

    Layout rows (y values are multiples of ``v_spacing``):

    * Very top  (y=3*vs) : C, centered over the match row
    * Top       (y=2*vs) : M1…ML
    * Middle    (y=vs)   : I1…IL-1 (between adjacent match states); B, L to
                           the left; E, R, T to the right
    * Bottom    (y=0)    : D1…DL

    Args:
        L: Number of match states.
        h_spacing: Horizontal spacing between adjacent states (default: 1.5).
        v_spacing: Vertical spacing between rows (default: 1.5).

    Returns:
        Tuple ``(pos, labels)`` where ``pos`` is a dict mapping node index to
        ``(x, y)`` and ``labels`` is a list of human-readable state names
        indexed by node index.
    """
    from learnMSA.hmm.util.transition_index_set import PHMMTransitionIndexSet

    max_states = PHMMTransitionIndexSet.num_states_unfolded(L)
    T_idx = max_states - 1

    pos: dict[int, tuple[float, float]] = {}
    # Match states M1..ML: top row
    for i in range(L):
        pos[i] = ((i + 1) * h_spacing, 1.5 * v_spacing)
    # Insert states I1..IL-1: middle row, between adjacent match states
    for i in range(L - 1):
        pos[L + i] = ((i + 1.5) * h_spacing, 0.5*v_spacing)
    # Delete states D1..DL: bottom row, aligned with match states
    for i in range(L):
        pos[2 * L - 1 + i] = ((i + 1) * h_spacing, 0)
    # Left-flank (L)
    pos[3 * L - 1] = (-h_spacing, v_spacing)
    # Begin (B): closest to the match row
    pos[3 * L] = (0, v_spacing)
    # End (E)
    pos[3 * L + 1] = ((L + 1) * h_spacing, v_spacing)
    # Unannotated/C: very top, centered over match row
    pos[3 * L + 2] = ((L + 1) * h_spacing / 2, 2.5 * v_spacing)
    # Right-flank (R): index 3L+3
    pos[3 * L + 3] = ((L + 2) * h_spacing, v_spacing)
    # Terminal (T): last state
    pos[T_idx] = ((L + 3) * h_spacing, v_spacing)

    labels: list[str] = [''] * max_states
    for i in range(L):
        labels[i] = f'M{i + 1}'
    for i in range(L - 1):
        labels[L + i] = f'I{i + 1}'
    for i in range(L):
        labels[2 * L - 1 + i] = f'D{i + 1}'
    labels[3 * L - 1] = 'L'
    labels[3 * L] = 'B'
    labels[3 * L + 1] = 'E'
    labels[3 * L + 2] = 'C'
    labels[3 * L + 3] = 'R'
    labels[T_idx] = 'T'

    return pos, labels


def plot_phmm(
    layer: PHMMLayer,
    head: int = 0,
    h_spacing: float = 1.0,
    v_spacing: float = 1.0,
    ax=None,
    title: str | None = None,
    threshold: float = 1e-12,
    edge_label_fmt: str = "{:.2f}",
    node_size: int = 6000,
    font_size: int = 30,
    edge_font_size: int = 24,
    label_pos: float = 0.5,
    connectionstyle: str = "arc3,rad=0.15",
    self_loop_connectionstyle: str = "arc3,rad=0.4",
    arrows_style: str = "-|>",
    title_font_size: int = 32,
    inset_offset: tuple[float, float] = (0, 0.05),
    fast_mode: bool = False,
    all_state_emissions: bool = False,
    show_information_content: bool = False,
) -> Figure | SubFigure | None:
    """Plots the state transition graph of a profile HMM using the explicit
    (unfolded) transition matrix with a structured layout:

    - Match states (M1...ML) in the top row.
    - Insert states (I1...IL-1) in the middle row, between adjacent match states.
    - Delete states (D1...DL) in the bottom row, aligned below match states.
    - Left-flank (L) and begin (B) states to the left.
    - End (E), right-flank (R), and terminal (T) states to the right.
    - Unannotated/center (C) state at the very top, centered on the x-axis.

    The HMM must have a ``PHMMTransitioner`` as its ``.transitioner``.

    Args:
        hmm: The HMM to visualize.
        head: Index of the HMM head to visualize (default: 0).
        h_spacing: Horizontal spacing between adjacent states (default: 1.0).
        v_spacing: Vertical spacing between rows (default: 1.0).
        ax: A matplotlib Axes to draw on. If None, a new figure is created.
        title: Optional title for the plot.
        threshold: Edges with probability below this value are omitted.
        edge_label_fmt: Format string for edge probability labels.
        node_size: Size of each node circle.
        font_size: Font size for node labels.
        edge_font_size: Font size for edge probability labels.
        label_pos: Position of edge label along edge (0=head, 0.5=center, 1=tail).
        connectionstyle: Matplotlib connectionstyle string for curved edges.
        self_loop_connectionstyle: Matplotlib connectionstyle string for
            self-loop edges.
        arrows_style: Matplotlib arrow style string.
        title_font_size: Font size for the plot title.
        fast_mode: If *True*, use a two-panel vertical layout suited for
            large models (hundreds of states). The **top panel** contains one
            row per active emitter; each row shows the full match-state profile
            as a single sequence logo spanning all *L* positions (one
            ``logomaker.Logo`` call) plus a single-position insert
            representative logo next to it. The **bottom panel** shows the
            full transition graph via :func:`plot_transition_graph` with no
            emission insets. Figure width is capped at 400 px-per-inch so the
            output remains renderable for large *L*. Requires ``ax=None``
            (default: ``False``).
        all_state_emissions: If *True*, show emission insets for all states.
            If *False* (default), only show insets for match states and the
            insert representative.
        show_information_content: If *True*, scale each logo column by the
            KL divergence (information content) of the match-state emission
            relative to the insert-state background distribution, so taller
            columns reflect positions that deviate more from background.
            Only affects ``TFCategoricalEmitter`` rows. Default: *False*.

    Returns:
        The matplotlib Figure.
    """
    from learnMSA.hmm.tf.transitioner import PHMMTransitioner

    transitioner = layer.hmm.transitioner
    if not isinstance(transitioner, PHMMTransitioner):
        raise TypeError(
            "plot_phmm requires an HMM with a PHMMTransitioner, "
            f"got {type(transitioner).__name__}"
        )

    explicit = transitioner.explicit_transitioner
    L = transitioner.lengths[head]

    effective_h_spacing = h_spacing * (0.45 + 0.2 * (len(layer.hmm.emitter)-1))
    effective_v_spacing = 0.35 * v_spacing
    pos, labels = phmm_layout(
        L, h_spacing=effective_h_spacing, v_spacing=effective_v_spacing
    )

    ys = [y for _, y in pos.values()]
    y_pad = (max(ys) - min(ys)) * 0.15 + effective_v_spacing * 0.5

    _plot_title = f"Profile HMM (head {head})" if title is None else title

    def _draw_transition_graph(
        target_ax,
        _node_size: int = node_size,
        _font_size: int = font_size,
        _edge_font_size: int = edge_font_size,
    ) -> None:
        plot_transition_graph(
            explicit,
            head=head,
            pos=pos,
            state_labels=labels,
            self_loop_connectionstyle=self_loop_connectionstyle,
            node_size=_node_size,
            font_size=_font_size,
            edge_font_size=_edge_font_size,
            arrows_style=arrows_style,
            edge_label_fmt=edge_label_fmt,
            threshold=threshold,
            label_pos=label_pos,
            connectionstyle=connectionstyle,
            title=_plot_title,
            ax=target_ax,
        )

    if fast_mode:
        if ax is not None:
            raise ValueError("fast_mode=True requires ax=None.")

        n_active = sum([not layer.no_aa, layer.use_language_model, layer.use_structure])
        logo_row_h = 3   # inches per emitter row in the logo panel

        xs = [x for x, _ in pos.values()]
        x_data_range = max(xs) - min(xs) if len(xs) > 1 else 1.0
        y_display_range = max(ys) - min(ys) + 2 * y_pad
        fig_w = min(L * 0.5 + 4, 400)

        # Graph height matched to data aspect ratio, scaled 3× vertically so
        # row-to-row spacing is comfortable relative to node diameter.
        graph_h = max(4, min(fig_w * y_display_range / x_data_range * 3.0, 30))

        fig = plt.figure(figsize=(fig_w, n_active * logo_row_h + graph_h))
        outer_gs = GridSpec(
            2, 1, figure=fig,
            height_ratios=[n_active * logo_row_h, graph_h],
            hspace=0.02,
        )

        # ── Top panel: one logo row per active emitter ──────────────────────
        top_gs = GridSpecFromSubplotSpec(
            n_active, 2, subplot_spec=outer_gs[0],
            width_ratios=[L, 1], wspace=0.02,
        )

        emitter_row = 0
        if not layer.no_aa:
            ax_m = fig.add_subplot(top_gs[emitter_row, 0])
            ax_i = fig.add_subplot(top_gs[emitter_row, 1])
            _render_emitter_fast(
                layer.hmm.emitter[emitter_row], head, L,
                ax_m, ax_i,
                alphabet=SequenceDataset._default_alphabet,
                color_scheme="skylign_protein",
                label="AA",
                show_information_content=show_information_content,
            )
            emitter_row += 1

        if layer.use_language_model:
            ax_m = fig.add_subplot(top_gs[emitter_row, 0])
            ax_i = fig.add_subplot(top_gs[emitter_row, 1])
            _render_emitter_fast(
                layer.hmm.emitter[emitter_row], head, L,
                ax_m, ax_i,
                label="emb",
                show_information_content=show_information_content,
            )
            emitter_row += 1

        if layer.use_structure:
            assert layer.structural_config is not None, \
                "Structural config must be provided if use_structure is True"
            ax_m = fig.add_subplot(top_gs[emitter_row, 0])
            ax_i = fig.add_subplot(top_gs[emitter_row, 1])
            _render_emitter_fast(
                layer.hmm.emitter[emitter_row], head, L,
                ax_m, ax_i,
                alphabet=layer.structural_config.structural_alphabet,
                color_scheme="NajafabadiEtAl2017",
                label="3Di",
                show_information_content=show_information_content,
            )

        # ── Bottom panel: transition graph, no emission insets ───────────────
        # Node diameter target: 50% of horizontal display spacing.
        # node_size is in pt²; diameter = 2·√(s/π), so s = π/4 · d² = π/4 · (0.5·sp)² ≈ 0.196·sp²
        spacing_pt = fig_w * 72.0 / (x_data_range / effective_h_spacing + 1)
        fast_node_size = max(50, int(0.196 * spacing_pt ** 2))
        fast_font_size = max(4, int((fast_node_size / 3.14159) ** 0.5 * 0.55))
        fast_edge_font_size = max(3, int(fast_font_size * 0.75))

        ax_graph = fig.add_subplot(outer_gs[1])
        _draw_transition_graph(
            ax_graph,
            _node_size=fast_node_size,
            _font_size=fast_font_size,
            _edge_font_size=fast_edge_font_size,
        )
        ax_graph.set_ylim(min(ys) - y_pad, max(ys) + y_pad)

        return fig

    # Default single-panel layout
    if ax is None:
        fig, ax = plt.subplots(figsize=(L * 4, 50))
    _draw_transition_graph(ax)
    ax = fig.gca()
    _plot_phmm_emissions(
        layer, head, pos, ax, inset_offset, title_font_size,
        states=list(range(2*L-1)) if all_state_emissions else list(range(L+1)),
    )
    ax.set_ylim(min(ys) - y_pad, max(ys) + y_pad)

    return fig


def _render_emitter_fast(
    emitter,
    head: int,
    L: int,
    ax_match,
    ax_ins,
    alphabet: str = "ARNDCQEGHILKMFPSTWYVXUO",
    color_scheme: str = "skylign_protein",
    label: str = "",
    show_information_content: bool = False,
) -> None:
    """Draw a full-length match-state profile logo and a single insert
    representative logo side by side using direct draw calls.

    Args:
        emitter: The emitter whose matrix is rendered.
        head: Emitter head index.
        L: Number of match states. Match states are indices 0…L-1; the
            representative insert state is index *L*.
        ax_match: Axes for the full-length match profile.
        ax_ins: Axes for the single insert-representative logo.
        alphabet: Alphabet string used for categorical emitters.
        color_scheme: Logomaker color scheme for categorical emitters.
        label: Y-axis label placed on *ax_match*.
        show_information_content: If *True*, scale match-state logo heights by
            the per-position KL divergence (bits) relative to the insert-state
            background. Has no effect for non-categorical emitters.
    """
    import warnings
    import pandas as pd
    import logomaker

    mro_names = {c.__name__ for c in type(emitter).__mro__}

    M = emitter.matrix()
    try:
        M_np = M.numpy()
    except AttributeError:
        M_np = np.array(M)

    if "TFCategoricalEmitter" in mro_names:
        A = M_np.shape[2]
        chars = list(alphabet[:A])
        match_probs = M_np[head, :L, :]        # (L, A)
        ins_probs = M_np[head, L : L + 1, :]  # (1, A)
        if show_information_content:
            background = M_np[head, L, :]      # insert state as background
            eps = 1e-10
            p = np.clip(match_probs, eps, 1.0)
            q = np.clip(background, eps, 1.0)
            # KL divergence (bits) per position: sum_a p * log2(p/q)
            ic = np.sum(
                p * np.log2(p / q[np.newaxis, :]), axis=-1, keepdims=True
            )  # (L, 1)
            match_data = ic * match_probs
        else:
            match_data = match_probs
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*not in color_dict.*", category=UserWarning
            )
            # Full match profile — one logomaker call for all L positions
            df_match = pd.DataFrame(match_data, columns=chars)
            logomaker.Logo(
                df_match, ax=ax_match,
                color_scheme=color_scheme, vpad=0.1, width=0.8,
            )
            # Insert representative (state L) — always raw probabilities
            df_ins = pd.DataFrame(ins_probs, columns=chars)
            logomaker.Logo(
                df_ins, ax=ax_ins,
                color_scheme=color_scheme, vpad=0.1, width=0.8,
            )
        if show_information_content:
            ax_match.set_ylabel(f"{label}\n(bits)", fontsize=8)
            ax_match.set_xticks([])
            from matplotlib.ticker import FuncFormatter
            ax_match.yaxis.set_major_formatter(
                FuncFormatter(lambda x, _: f"{x:.1f}")
            )
        else:
            ax_match.set_ylabel(label, fontsize=8)
            ax_match.set_xticks([])
            ax_match.set_yticks([])
        ax_ins.set_title("ins.", fontsize=8)
        ax_ins.set_xticks([])
        ax_ins.set_yticks([])

    elif "TFMVNormalEmitter" in mro_names:
        D = M_np.shape[2] // 2
        colors = plt.cm.tab10.colors  # type: ignore
        # Match: heatmap of per-dimension means across L states
        means_match = M_np[head, :L, :D]  # (L, D)
        ax_match.imshow(
            means_match.T, aspect="auto", cmap="RdBu_r",
            interpolation="nearest",
        )
        ax_match.set_ylabel(label, fontsize=8)
        ax_match.set_xticks([])
        ax_match.set_yticks([])
        # Insert representative: means as a bar chart
        ins_means = M_np[head, L, :D]
        ax_ins.bar(
            range(D), ins_means,
            color=[colors[d % len(colors)] for d in range(D)],
        )
        ax_ins.set_title("ins.", fontsize=8)
        ax_ins.set_xticks([])
        ax_ins.set_yticks([])


def _plot_phmm_emissions(
    layer: PHMMLayer,
    head: int,
    pos: dict,
    ax,
    inset_offset: tuple[float, float],
    title_font_size: int,
    states: list[int],
    inset_size: float = 0.25,
) -> None:
    """Call :func:`plot_emissions` for every active emitter in *layer*."""
    emitter_index = 0
    col = 0
    if not layer.no_aa:
        plot_emissions(
            layer.hmm.emitter[emitter_index],
            head=head,
            pos=pos,
            state_labels="AA",
            inset_column=col,
            alphabet=SequenceDataset._default_alphabet,
            ax=ax,
            inset_offset=inset_offset,
            title_font_size=title_font_size,
            states=states,
            inset_size=inset_size,
        )
        emitter_index += 1
        col += 1
    if layer.use_language_model:
        plot_emissions(
            layer.hmm.emitter[emitter_index],
            head=head,
            pos=pos,
            state_labels="emb",
            inset_column=(col, col + 2),
            ax=ax,
            inset_offset=inset_offset,
            title_font_size=title_font_size,
            states=states,
            inset_size=inset_size,
            normal_linewidth=3,
            normal_alpha=0.5,
        )
        emitter_index += 1
        col += 2
    if layer.use_structure:
        assert layer.structural_config is not None, \
            "Structural config must be provided if use_structure is True"
        plot_emissions(
            layer.hmm.emitter[emitter_index],
            head=head,
            pos=pos,
            state_labels="3Di",
            inset_column=col,
            alphabet=layer.structural_config.structural_alphabet,
            ax=ax,
            inset_offset=inset_offset,
            title_font_size=title_font_size,
            states=states,
            inset_size=inset_size,
            color_scheme="NajafabadiEtAl2017",
        )



# class LogoPlotterCallback(tf.keras.callbacks.Callback):
#     def __init__(
#             self, logo_dir, data, batch_generator, decode_indices, batch_size
#         ):
#         self.logo_dir = logo_dir
#         self.data = data
#         self.batch_generator = batch_generator
#         self.decode_indices = decode_indices
#         self.batch_size = batch_size
#         self.i = 0
#         self.frame_dir = self.logo_dir / "frames"

#     def on_train_batch_end(self, batch, logs=None):
#         am = msa_hmm.AlignmentModel.AlignmentModel(
#             self.data,
#             self.batch_generator,
#             self.decode_indices,
#             batch_size=self.batch_size,
#             model=self.model,
#         )
#         fig, ax = plt.subplots(1, 1, figsize=(6, 4))
#         plot_logo(am, 0, ax)
#         plt.savefig(self.frame_dir / f"{self.i}.png", format="png", bbox_inches="tight")
#         self.i += 1
#         plt.close()


# def make_logo_gif(
#     frame_dir: Path,
#     gif_filepath: str,
#     frame_filter: Callable[[list[Path]], list[Path]] | None = None,
# ) -> None:
#     """Creates a gif from png frames."""

#     # get filenames and sort by frame number
#     filenames = [
#         (int(file.split(".")[0]), frame_dir / file)
#         for file in os.listdir(frame_dir) if file.endswith(".png")
#     ]
#     filenames.sort()

#     # simple heuristic to reduce the number of frames, which depends on the
#     # number of training steps
#     # we want roughly 100 frames for a nice short and memory friendly gif
#     # also, we want to focus on frames from the first half of the training
#     # which have the most variability
#     if len(filenames)//2 > 100:
#         filenames = filenames[:len(filenames)//2: len(filenames)//200]
#     #write the gif
#     with imageio.get_writer(gif_filepath, mode='I') as writer:
#         for _,filename in filenames:
#             image = imageio.imread(filename)
#             writer.append_data(image)