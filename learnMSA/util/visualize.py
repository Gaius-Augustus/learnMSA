import imageio
import warnings
import logomaker
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from hidten.hmm import HMM
from hidten.transitioner import Transitioner
from hidten.tf.emitter import TFCategoricalEmitter, TFMVNormalEmitter
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import norm


def plot_transition_graph(
    transitioner: Transitioner,
    head: int = 0,
    threshold: float = 1e-16,
    ax=None,
    state_labels: list[str] | None = None,
    edge_label_fmt: str = "{:.2f}",
    node_size: int = 1500,
    font_size: int = 10,
    edge_font_size: int = 8,
    label_pos: float = 0.5,
    connectionstyle: str = "arc3,rad=0.2",
    self_loop_connectionstyle: str = "arc3,rad=0.4",
    arrows_style: str = "-|>",
    pos: dict | None = None,
):
    """Plots the state transition graph of the HMM from its transition matrix.

    Args:
        transitioner: The Transitioner whose transition graph is plotted.
        head: Index of the Transitioner head to visualize (default: 0).
        threshold: Edges with probability below this value are omitted.
        ax: A matplotlib Axes to draw on. If None, a new figure is created.
        state_labels: Optional list of state name strings (length Q).
        edge_label_fmt: Format string for edge probability labels.
        node_size: Size of each node circle.
        font_size: Font size for node labels.
        edge_font_size: Font size for edge probability labels.
        label_pos: Position of edge label along edge
            (0=head, 0.5=center, 1=tail).
        connectionstyle: Matplotlib connectionstyle string for curved edges
            (e.g. 'arc3,rad=0.2').
        self_loop_connectionstyle: Matplotlib connectionstyle string for
            self-loop edges (e.g. 'arc3,rad=0.8').
        arrows_style: Matplotlib arrow style string (e.g. '-|>').
        pos: Optional dict mapping node indices to ``(x, y)`` positions. If
            *None*, positions are computed with
            :func:`networkx.circular_layout`.

    Returns:
        The matplotlib Figure that was drawn on.
    """
    A = transitioner.matrix()
    # A has shape (H, Q, Q); detach from any autograd graph and convert to numpy
    try:
        A_np = A[head].numpy()
    except AttributeError:
        A_np = np.array(A[head])

    Q = transitioner.states[head]
    if state_labels is not None:
        labels = state_labels
    else:
        labels = [str(i) for i in range(Q)]

    G = nx.DiGraph()
    G.add_nodes_from(range(Q))
    for i in range(Q):
        for j in range(Q):
            p = float(A_np[i, j])
            if p >= threshold:
                G.add_edge(i, j, weight=p)

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, Q * 1.4), max(5, Q * 1.2)))
    else:
        fig = ax.get_figure()

    if pos is None:
        pos = nx.circular_layout(G)
    node_label_map = {i: labels[i] for i in range(Q)}

    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=node_size, node_color="steelblue"
    )
    nx.draw_networkx_labels(
        G, pos, labels=node_label_map, ax=ax,
        font_size=font_size, font_color="white",
    )

    regular_edges = [(u, v) for u, v in G.edges() if u != v]
    self_loop_edges = [(u, v) for u, v in G.edges() if u == v]

    nx.draw_networkx_edges(
        G, pos, edgelist=regular_edges, ax=ax,
        arrowstyle=arrows_style,
        connectionstyle=connectionstyle,
        node_size=node_size,
    )
    if self_loop_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=self_loop_edges, ax=ax,
            arrowstyle=arrows_style,
            connectionstyle=self_loop_connectionstyle,
            node_size=node_size/7.,
        )

    all_edge_labels = {
        (u, v): edge_label_fmt.format(d["weight"])
        for u, v, d in G.edges(data=True)
    }
    regular_edge_labels = {k: v for k, v in all_edge_labels.items() if k[0] != k[1]}
    self_loop_labels = {k: v for k, v in all_edge_labels.items() if k[0] == k[1]}

    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=regular_edge_labels, ax=ax,
        font_size=edge_font_size,
        label_pos=label_pos,
        connectionstyle=connectionstyle,
        node_size=node_size,
    )
    if self_loop_labels:
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=self_loop_labels, ax=ax,
            font_size=edge_font_size,
            label_pos=label_pos,
            connectionstyle=self_loop_connectionstyle,
            node_size=node_size/7.,
        )

    ax.set_title(f"State Transition Graph (head {head})")
    ax.axis("off")
    fig.tight_layout()
    return fig, pos


def plot_emissions(
    hmm: HMM,
    pos: dict,
    head: int = 0,
    emitter_index: int = 0,
    ax=None,
    inset_size: float | tuple[float, float] = 0.25,
    inset_offset: tuple[float, float] = (0.1, 0.05),
    inset_column: int = 0,
    alphabet: str = "ARNDCQEGHILKMFPSTWYVXUO",
    color_scheme: str = "skylign_protein",
    state_labels: list[str] | str | None = None,
    title_font_size: int = 32,
    states: list[int] | np.ndarray | None = None,
):
    """Plots small inset bar charts at each node showing the emission
    distribution for a given head and emitter.

    This function is designed to be called *after* ``plot_transition_graph``
    so that it overlays the emission insets on top of an existing graph plot.

    Args:
        hmm: The HMM whose emissions are visualised.
        G: The networkx graph that was drawn (used only for iterating nodes).
        pos: Node-position dict returned by the layout used when drawing ``G``
            (e.g. the value returned by :func:`plot_transition_graph`).
        head: Index of the HMM head to visualise (default: 0).
        emitter_index: Index of the emitter to visualise (default: 0).
        ax: The matplotlib Axes that the transition graph was drawn on.
            If *None* the current active axes is used.
        inset_size: Width and height of each inset axes expressed as a
            fraction of the data-coordinate span of the main axes. Either a
            single float (square) or a ``(width, height)`` tuple. The same
            size is used for all emitter types.
        inset_offset: (dx, dy) offset applied to each node position so the
            inset does not sit directly on top of the node label.
        inset_column: Integer column index used to stack multiple emitter
            insets side by side. Column ``k`` is shifted by
            ``k * (inset_width + margin)`` in the x direction relative to
            ``inset_offset``, where margin is 10 % of the inset width.
        alphabet: The alphabet used to create logo plots for categorical
            emitters. Ignored for non-categorical emitters.
        color_scheme: Logomaker color scheme for categorical emitters
            (default: ``'skylign_protein'``). Any scheme accepted by
            :func:`logomaker.Logo` can be used (e.g.
            ``'chemistry'``, ``'NajafabadiEtAl2017'``).
        state_labels: Optional list of state name strings or a single string.
            When provided, each inset receives a title matching the label of
            the corresponding node. If *None*, the node index is used as the title.
        title_font_size: Font size for the inset titles (default: 5).
        states: Optional list or array of state indices to plot. When *None*
            (default), emission insets are drawn for every state in ``pos``.
            When provided, only states whose index appears in ``states`` are
            plotted.

    Returns:
        The matplotlib Figure.
    """
    if ax is None:
        ax = plt.gca()
    fig = ax.get_figure()

    iw, ih = (inset_size, inset_size) if isinstance(inset_size, (int, float)) else inset_size
    margin = iw * 0.1
    col_shift = inset_column * (iw + margin)

    dx, dy = inset_offset
    dx = dx + col_shift

    active_pos = {
        node: xy for node, xy in pos.items()
        if states is None or node in states
    }

    # Find out the type of emitter to determine the type of plot
    if isinstance(hmm.emitter[emitter_index], TFCategoricalEmitter):
        emitter = hmm.emitter[emitter_index]
        # B has shape (H, Q, A)
        B = emitter.matrix()
        try:
            B_np = B.numpy()
        except AttributeError:
            B_np = np.array(B)

        A = B_np.shape[2]
        chars = list(alphabet[:A])

        for node, (x, y) in active_pos.items():
            probs = B_np[head, node, :]  # shape (A,)
            df = pd.DataFrame([probs], columns=chars)

            axins = inset_axes(
                ax,
                width="100%",
                height="100%",
                bbox_to_anchor=(x + dx, y + dy, iw, ih),
                bbox_transform=ax.transData,
                loc="center",
                borderpad=0,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*not in color_dict.*", category=UserWarning)
                logomaker.Logo(df, ax=axins, color_scheme=color_scheme, vpad=0.1, width=0.8)
            if isinstance(state_labels, list):
                title = state_labels[node]
            elif isinstance(state_labels, str):
                title = state_labels
            else:
                title = str(node)
            axins.set_title(title, fontsize=title_font_size, pad=1)
            axins.set_xticks([])
            axins.set_yticks([])
            axins.patch.set_alpha(0.7)

    elif isinstance(hmm.emitter[emitter_index], TFMVNormalEmitter):
        emitter = hmm.emitter[emitter_index]
        # M has shape (H, Q, 2*D): first D entries are means, last D are variances
        M = emitter.matrix()
        try:
            M_np = M.numpy()
        except AttributeError:
            M_np = np.array(M)

        D = M_np.shape[2] // 2
        colors = plt.cm.tab10.colors

        for node, (x, y) in active_pos.items():
            means = M_np[head, node, :D]      # shape (D,)
            variances = M_np[head, node, D:]  # shape (D,)
            stds = np.sqrt(variances)

            # Shared x-range covering all dimensions
            x_min = min(means[d] - 4 * stds[d] for d in range(D))
            x_max = max(means[d] + 4 * stds[d] for d in range(D))
            xs = np.linspace(x_min, x_max, 200)

            # Compute all curves to set a consistent y-limit
            all_ys = [norm.pdf(xs, loc=means[d], scale=stds[d]) for d in range(D)]
            y_max = max(ys.max() for ys in all_ys)

            axins = inset_axes(
                ax,
                width="100%",
                height="100%",
                bbox_to_anchor=(x + dx, y + dy, iw, ih),
                bbox_transform=ax.transData,
                loc="center",
                borderpad=0,
            )
            for d in range(D):
                axins.plot(xs, all_ys[d], color=colors[d % len(colors)], linewidth=1.0)
            axins.set_xlim(x_min, x_max)
            axins.set_ylim(0, y_max * 1.05)
            axins.set_xticks([float(f"{m:.2g}") for m in means])
            axins.set_yticks([0, round(y_max, 2)])
            axins.tick_params(labelsize=4, pad=1)
            axins.set_title(
                state_labels[node] if state_labels is not None else str(node),
                fontsize=title_font_size,
                pad=1,
            )
            axins.patch.set_alpha(0.7)
    else:
        # Unknown emitter, keep sample plots for testing

        rng = np.random.default_rng(seed=42)
        emission_data = {
            i: rng.dirichlet(np.ones(4)) for i in range(2)
        }

        for node, (x, y) in active_pos.items():
            axins = inset_axes(
                ax,
                width="100%",
                height="100%",
                bbox_to_anchor=(x + dx, y + dy, iw, ih),
                bbox_transform=ax.transData,
                loc="center",
                borderpad=0,
            )
            data = emission_data[node % 2]
            axins.bar(range(len(data)), data, color="steelblue", width=0.8)
            axins.set_xlim(-0.5, len(data) - 0.5)
            axins.set_ylim(0, 1)
            axins.set_title(
                state_labels[node] if state_labels is not None else str(node),
                fontsize=title_font_size,
                pad=1,
            )
            axins.set_xticks([])
            axins.set_yticks([])
            axins.patch.set_alpha(0.7)

    # Restore the main axes as the active axes so subsequent calls to
    # plt.gca() return the graph axes, not the last-created inset.
    fig.sca(ax)
    return fig

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
        pos[i] = ((i + 1) * h_spacing, 2 * v_spacing)
    # Insert states I1..IL-1: middle row, between adjacent match states
    for i in range(L - 1):
        pos[L + i] = ((i + 1.5) * h_spacing, v_spacing)
    # Delete states D1..DL: bottom row, aligned with match states
    for i in range(L):
        pos[2 * L - 1 + i] = ((i + 1) * h_spacing, 0)
    # Left-flank (L)
    pos[3 * L - 1] = (-h_spacing, 1.5 * v_spacing)
    # Begin (B): closest to the match row
    pos[3 * L] = (0, 1.5 * v_spacing)
    # End (E)
    pos[3 * L + 1] = ((L + 1) * h_spacing, 1.5 * v_spacing)
    # Unannotated/C: very top, centered over match row
    pos[3 * L + 2] = ((L + 1) * h_spacing / 2, 3 * v_spacing)
    # Right-flank (R): index 3L+3
    pos[3 * L + 3] = ((L + 2) * h_spacing, 1.5 * v_spacing)
    # Terminal (T): last state
    pos[T_idx] = ((L + 3) * h_spacing, 1.5 * v_spacing)

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
    hmm: HMM,
    head: int = 0,
    h_spacing: float = 1.5,
    v_spacing: float = 1.5,
    ax=None,
    threshold: float = 1e-3,
    edge_label_fmt: str = "{:.2f}",
    node_size: int = 1200,
    font_size: int = 8,
    edge_font_size: int = 7,
    label_pos: float = 0.5,
    connectionstyle: str = "arc3,rad=0.15",
    self_loop_connectionstyle: str = "arc3,rad=0.4",
    arrows_style: str = "-|>",
):
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
        h_spacing: Horizontal spacing between adjacent states (default: 1.5).
        v_spacing: Vertical spacing between rows (default: 1.5).
        ax: A matplotlib Axes to draw on. If None, a new figure is created.
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

    Returns:
        Tuple ``(fig, pos)`` where ``pos`` is the node-position dict used for
        drawing (can be passed to :func:`plot_emissions`).
    """
    from learnMSA.hmm.tf.transitioner import PHMMTransitioner

    transitioner = hmm.transitioner
    if not isinstance(transitioner, PHMMTransitioner):
        raise TypeError(
            "plot_phmm requires an HMM with a PHMMTransitioner, "
            f"got {type(transitioner).__name__}"
        )

    explicit = transitioner.explicit_transitioner
    L = transitioner.lengths[head]

    pos, labels = phmm_layout(L, h_spacing=h_spacing, v_spacing=v_spacing)
    T_idx = max(pos.keys())  # terminal state index

    # ---- Build explicit transition matrix ----
    A = explicit.matrix()
    try:
        A_np = A[head].numpy()
    except AttributeError:
        A_np = np.array(A[head])

    # ---- Build graph restricted to nodes with defined positions ----
    G = nx.DiGraph()
    G.add_nodes_from(pos.keys())
    for i in pos:
        for j in pos:
            p = float(A_np[i, j])
            if p >= threshold:
                G.add_edge(i, j, weight=p)

    # ---- Create axes ----
    if ax is None:
        fig_w = max(8, (L + 5) * h_spacing * 1.1)
        fig_h = max(5, 4 * v_spacing)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    else:
        fig = ax.get_figure()

    node_label_map = {i: labels[i] for i in pos}

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size, node_color="steelblue")
    nx.draw_networkx_labels(
        G, pos, labels=node_label_map, ax=ax,
        font_size=font_size, font_color="white",
    )

    regular_edges = [(u, v) for u, v in G.edges() if u != v]
    self_loop_edges = [(u, v) for u, v in G.edges() if u == v]

    nx.draw_networkx_edges(
        G, pos, edgelist=regular_edges, ax=ax,
        arrowstyle=arrows_style,
        connectionstyle=connectionstyle,
        node_size=node_size,
    )
    if self_loop_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=self_loop_edges, ax=ax,
            arrowstyle=arrows_style,
            connectionstyle=self_loop_connectionstyle,
            node_size=node_size / 7.,
        )

    all_edge_labels = {
        (u, v): edge_label_fmt.format(d["weight"])
        for u, v, d in G.edges(data=True)
    }
    regular_edge_labels = {k: v for k, v in all_edge_labels.items() if k[0] != k[1]}
    self_loop_labels = {k: v for k, v in all_edge_labels.items() if k[0] == k[1]}

    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=regular_edge_labels, ax=ax,
        font_size=edge_font_size,
        label_pos=label_pos,
        connectionstyle=connectionstyle,
        node_size=node_size,
    )
    if self_loop_labels:
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=self_loop_labels, ax=ax,
            font_size=edge_font_size,
            label_pos=label_pos,
            connectionstyle=self_loop_connectionstyle,
            node_size=node_size / 7.,
        )

    ax.set_title(f"Profile HMM – explicit transitions (head {head})")
    ax.axis("off")
    fig.tight_layout()
    return fig, pos


def plot(hmm: HMM, pos: dict | None = None):
    """
    Shows a visualization of the HMM including the state transition graph
    and the emission distributions (if a suitable plotting method is
    implemented for the emitter).

    Args:
        hmm: The HMM to visualize.
        pos: Optional dict mapping node indices to ``(x, y)`` positions
            passed through to :func:`plot_transition_graph`. If *None*,
            positions are computed with :func:`networkx.circular_layout`.
    """
    # Only head 0 for now
    plot_transition_graph(hmm.transitioner, head=0, pos=pos)