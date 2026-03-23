import imageio
import logomaker
import networkx as nx
import numpy as np
import seaborn as sns
from hidten.hmm import HMM
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plot_transition_graph(
    hmm: HMM,
    head: int = 0,
    threshold: float = 1e-3,
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
):
    """Plots the state transition graph of the HMM from its transition matrix.

    Args:
        hmm: The HMM whose transition graph is plotted.
        head: Index of the HMM head to visualize (default: 0).
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

    Returns:
        The matplotlib Figure that was drawn on.
    """
    A = hmm.transitioner.matrix()
    # A has shape (H, Q, Q); detach from any autograd graph and convert to numpy
    try:
        A_np = A[head].numpy()
    except AttributeError:
        A_np = np.array(A[head])

    Q = A_np.shape[0]
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
    inset_size: float = 0.25,
    inset_offset: tuple[float, float] = (0.05, 0.05),
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
            fraction of the data-coordinate span of the main axes.
        inset_offset: (dx, dy) offset applied to each node position so the
            inset does not sit directly on top of the node label.

    Returns:
        The matplotlib Figure.
    """
    if ax is None:
        ax = plt.gca()
    fig = ax.get_figure()

    Q = len(pos)

    # Try to obtain real emission probabilities; fall back to random data so
    # the function is always usable even before parameters are fitted.
    # try:
    #     emitter = hmm.emitter[emitter_index]
    #     # emission_matrix() is expected to return an array of shape (H, Q, A)
    #     B = emitter.emission_matrix()
    #     try:
    #         B_np = B[head].numpy()
    #     except AttributeError:
    #         B_np = np.array(B[head])
    #     emission_data = {node: B_np[node] for node in G.nodes()}
    # except Exception:
    #     # Fall back to example / placeholder data
    rng = np.random.default_rng(seed=42)
    emission_data = {
        i: rng.dirichlet(np.ones(4)) for i in range(2)
    }


    dx, dy = inset_offset
    for node, (x, y) in pos.items():
        axins = inset_axes(
            ax,
            width="100%",
            height="100%",
            bbox_to_anchor=(x + dx, y + dy, inset_size, inset_size),
            bbox_transform=ax.transData,
            loc="center",
            borderpad=0,
        )
        data = emission_data[node]
        axins.bar(range(len(data)), data, color="steelblue", width=0.8)
        axins.set_xlim(-0.5, len(data) - 0.5)
        axins.set_ylim(0, 1)
        axins.set_xticks([])
        axins.set_yticks([])
        axins.patch.set_alpha(0.7)

    return fig


def plot(hmm : HMM):
    """
    Shows a visualization of the HMM including the state transition graph
    and the emission distributions (if a suitable plotting method is
    implemented for the emitter).
    """
    G = nx.DiGraph()

    A = hmm.transitioner.matrix()

    # Only head 1 for now
    plot_transition_graph(hmm, head=0)