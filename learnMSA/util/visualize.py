import imageio
from hidten.visualize import Figure, SubFigure, plot_emissions, plot_transition_graph
from matplotlib import pyplot as plt

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

    Returns:
        Tuple ``(fig, pos)`` where ``pos`` is the node-position dict used for
        drawing (can be passed to :func:`plot_emissions`).
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

    if ax is None:
        fig, ax = plt.subplots(figsize=(L * 4, 50))

    L = transitioner.lengths[head]
    fig, _ = plot_transition_graph(
        explicit,
        head=head,
        pos=pos,
        state_labels=labels,
        self_loop_connectionstyle=self_loop_connectionstyle,
        node_size=node_size,
        font_size=font_size,
        edge_font_size=edge_font_size,
        arrows_style=arrows_style,
        edge_label_fmt=edge_label_fmt,
        threshold=threshold,
        ax=ax,
        label_pos=label_pos,
        connectionstyle=connectionstyle,
        title=f"Profile HMM (head {head})" if title is None else title,
    )
    ax = fig.gca()

    emitter_index = 0

    if not layer.no_aa:
        fig=plot_emissions(
            layer.hmm.emitter[emitter_index],
            head=head,
            pos=pos,
            state_labels="AA",
            inset_column=emitter_index,
            alphabet=SequenceDataset._default_alphabet,
            ax=ax,
            inset_offset=inset_offset,
            title_font_size=title_font_size,
            states=list(range(L+1)),
        )
        emitter_index += 1

    if layer.use_language_model:
        fig=plot_emissions(
            layer.hmm.emitter[emitter_index],
            head=head,
            pos=pos,
            state_labels="emb",
            inset_column=emitter_index,
            ax=ax,
            inset_offset=inset_offset,
            title_font_size=title_font_size,
            states=list(range(L+1)),
            normal_linewidth=3,
            normal_alpha=0.5,
        )
        emitter_index += 1

    if layer.use_structure:
        assert layer.structural_config is not None,\
            "Structural config must be provided if use_structure is True"
        fig=plot_emissions(
            layer.hmm.emitter[emitter_index],
            head=head,
            pos=pos,
            state_labels="3Di",
            inset_column=emitter_index,
            alphabet=layer.structural_config.structural_alphabet,
            ax=ax,
            inset_offset=inset_offset,
            title_font_size=title_font_size,
            states=list(range(L+1)),
            color_scheme="NajafabadiEtAl2017",
        )

    ys = [y for _, y in pos.values()]
    y_pad = (max(ys) - min(ys)) * 0.15 + effective_v_spacing * 0.5
    ax.set_ylim(min(ys) - y_pad, max(ys) + y_pad)

    return fig





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