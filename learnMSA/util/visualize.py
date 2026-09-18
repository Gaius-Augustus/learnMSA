import math
from dataclasses import dataclass

import imageio
import numpy as np
from hidten.visualize import Figure, SubFigure, plot_transition_graph
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path

from learnMSA.util import SequenceDataset
from learnMSA.hmm.layer import PHMMLayer
from learnMSA.util.tensor import to_numpy


#: Height of one emission logo band in data units (one unit = one match column).
_LOGO_BAND_HEIGHT = 1.8
#: Vertical gap between the lowest logo band and the insert state row.
_BAND_TO_INSERT = 0.7
#: Vertical distance between the match, insert and delete rows.
_STATE_ROW_SPACING = 0.85
#: Space below the delete row before the next row starts.
_ROW_GAP = 0.55
#: Space above the first row for the title and the start distribution.
_HEADER_HEIGHT = 1.3
#: Width left of the first match column (insert logo, L and B states).
_LEFT_MARGIN = 2.9
#: Width right of the last match column (ghost states, E, R and T states).
_RIGHT_MARGIN = 3.0
#: Space below the last row; the unannotated state C sits in it.
_FOOTER_HEIGHT = 0.7
#: Distance of C below the end of the last row.
_UNANNOTATED_OFFSET = 0.2


@dataclass(frozen=True)
class PHMMGeometry:
    """Row geometry of a wrapped profile HMM plot.

    Match states are laid out like lines of text: ``matches_per_row`` match
    columns per row, rows stacked top to bottom. One data unit is the distance
    between two adjacent match columns; match column ``c`` (0-based within its
    row) sits at ``x = c + 1``.

    Attributes:
        L: Number of match states.
        matches_per_row: Match columns per row.
        n_bands: Emission logo bands reserved above each row.
    """
    L: int
    matches_per_row: int
    n_bands: int = 0

    @property
    def n_rows(self) -> int:
        return math.ceil(self.L / self.matches_per_row)

    @property
    def row_height(self) -> float:
        return _row_height(self.n_bands)

    def row_of(self, i: int) -> int:
        """Row of the 0-based match index *i*."""
        return i // self.matches_per_row

    def x_of(self, i: int) -> float:
        """x position of the 0-based match index *i*."""
        return i % self.matches_per_row + 1.0

    def row_top(self, r: int) -> float:
        return -_HEADER_HEIGHT - r * self.row_height

    def band_bottom(self, r: int, b: int) -> float:
        """Lower edge of logo band *b* (0 = top band) in row *r*."""
        return self.row_top(r) - (b + 1) * _LOGO_BAND_HEIGHT

    def y_insert(self, r: int) -> float:
        return (
            self.row_top(r) - self.n_bands * _LOGO_BAND_HEIGHT - _BAND_TO_INSERT
        )

    def y_match(self, r: int) -> float:
        return self.y_insert(r) - _STATE_ROW_SPACING

    def y_delete(self, r: int) -> float:
        return self.y_match(r) - _STATE_ROW_SPACING

    @property
    def y_unannotated(self) -> float:
        return self.row_top(self.n_rows) - _UNANNOTATED_OFFSET

    @property
    def boundaries(self) -> list[int]:
        """0-based indices of the last match of every row but the final one."""
        return [
            (r + 1) * self.matches_per_row - 1 for r in range(self.n_rows - 1)
        ]

    @property
    def xlim(self) -> tuple[float, float]:
        return 0.5 - _LEFT_MARGIN, self.matches_per_row + 0.5 + _RIGHT_MARGIN

    @property
    def ylim(self) -> tuple[float, float]:
        return self.row_top(self.n_rows) - _FOOTER_HEIGHT, 0.0


def _row_height(n_bands: int) -> float:
    return (
        n_bands * _LOGO_BAND_HEIGHT + _BAND_TO_INSERT
        + 2 * _STATE_ROW_SPACING + _ROW_GAP
    )


def choose_matches_per_row(
    L: int,
    n_bands: int = 1,
    target_aspect: float = 1.6,
    min_per_row: int = 12,
) -> int:
    """Pick the number of match columns per row so that the whole plot has a
    width/height ratio close to *target_aspect*.

    Rows are only wrapped as long as they keep at least *min_per_row* matches,
    so small models stay on a single row.
    """
    best, best_score = L, math.inf
    for n_rows in range(1, L + 1):
        per_row = math.ceil(L / n_rows)
        if n_rows > 1 and per_row < min_per_row:
            break
        width = per_row + _LEFT_MARGIN + _RIGHT_MARGIN
        height = _HEADER_HEIGHT + n_rows * _row_height(n_bands) + _FOOTER_HEIGHT
        score = abs(math.log(width / height / target_aspect))
        if score < best_score:
            best, best_score = per_row, score
    return best


def phmm_layout(
    L: int,
    matches_per_row: int | None = None,
    n_bands: int = 0,
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

    The match chain is wrapped into rows of ``matches_per_row`` columns (see
    :class:`PHMMGeometry`). Within a row, insert states are on top (between
    adjacent match states), match states in the middle and delete states at
    the bottom, aligned with the match states. ``n_bands`` logo bands are kept
    free above each row. L and B sit left of the first row, E, R and T right of
    the last match state and C below the last row, under B.

    Args:
        L: Number of match states.
        matches_per_row: Match columns per row. ``None`` puts all match states
            in a single row.
        n_bands: Number of emission logo bands reserved above each row.

    Returns:
        Tuple ``(pos, labels)`` where ``pos`` is a dict mapping node index to
        ``(x, y)`` and ``labels`` is a list of human-readable state names
        indexed by node index.
    """
    from learnMSA.hmm.util.transition_index_set import PHMMTransitionIndexSet

    geo = PHMMGeometry(L, matches_per_row or L, n_bands)
    max_states = PHMMTransitionIndexSet.num_states_unfolded(L)
    T_idx = max_states - 1

    pos: dict[int, tuple[float, float]] = {}
    for i in range(L):
        r, x = geo.row_of(i), geo.x_of(i)
        pos[i] = (x, geo.y_match(r))
        pos[2 * L - 1 + i] = (x, geo.y_delete(r))
        if i < L - 1:
            pos[L + i] = (x + 0.5, geo.y_insert(r))
    x_last = geo.x_of(L - 1)
    y_last = geo.y_match(geo.n_rows - 1)
    pos[3 * L - 1] = (-1.0, geo.y_match(0))        # L
    pos[3 * L] = (0.0, geo.y_match(0))             # B
    pos[3 * L + 1] = (x_last + 1, y_last)          # E
    pos[3 * L + 2] = (0.0, geo.y_unannotated)      # C
    pos[3 * L + 3] = (x_last + 2, y_last)          # R
    pos[T_idx] = (x_last + 3, y_last)              # T

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


def _is_phmm_transitioner(transitioner) -> bool:
    """Whether a transitioner is a pHMM transitioner, backend-independently.

    The concrete classes are backend specific (``TFPHMMTransitioner``,
    ``TorchPHMMTransitioner``, ...), so the family is derived from the class
    names in the MRO rather than from an isinstance check.
    """
    return any(
        cls.__name__.endswith("PHMMTransitioner") for cls in type(transitioner).__mro__
    )


class _DisplayTransitioner:
    """Single-head stand-in for a transitioner, holding a transition matrix
    edited for display. :func:`plot_transition_graph` only reads ``matrix()``,
    ``start_dist()`` and ``states``."""

    def __init__(self, A: np.ndarray, P: np.ndarray):
        self._A = A[np.newaxis]
        self._P = P[np.newaxis]
        self.states = [A.shape[0]]

    def matrix(self) -> np.ndarray:
        return self._A

    def start_dist(self) -> np.ndarray:
        return self._P


@dataclass(frozen=True)
class _GraphStyle:
    node_size: float
    font_size: float
    edge_font_size: float
    threshold: float
    edge_label_fmt: str
    label_pos: float
    connectionstyle: str
    arrows_style: str
    jump_label_threshold: float


def _auto_sizes(unit_in: float) -> tuple[float, float, float]:
    """Node size (pt²) and font sizes for a given size of one data unit."""
    diameter_pt = 0.5 * unit_in * 72.0  # half the match column spacing
    node_size = math.pi / 4 * diameter_pt ** 2
    font_size = max(3.0, 0.28 * diameter_pt)
    edge_font_size = max(2.5, 0.2 * diameter_pt)
    return node_size, font_size, edge_font_size


# Node colors
_MATCH_COLOR = "#3B6FB6"
_INSERT_COLOR = "#E8B730"
_DELETE_COLOR = "#C8453C"
_OTHER_COLOR = "#696969"


def _state_colors(L: int, Q: int) -> tuple[list[str], list[str]]:
    face = (
        [_MATCH_COLOR] * L + [_INSERT_COLOR] * (L - 1)
        + [_DELETE_COLOR] * L + [_OTHER_COLOR] * (Q - 3 * L + 1)
    )
    label = ["black" if c == _INSERT_COLOR else "white" for c in face]
    return face, label


def _edge_color_width(p: float) -> tuple:
    return plt.get_cmap("winter")(p), 1.0 + 5.0 * p


def _draw_phmm_graph(
    ax,
    A: np.ndarray,
    P: np.ndarray,
    geo: PHMMGeometry,
    style: _GraphStyle,
) -> None:
    """Draw the wrapped transition graph of one pHMM head.

    Args:
        ax: Axes whose data coordinates follow *geo*.
        A: Unfolded transition matrix of the head, shape ``(Q, Q)``.
        P: Start distribution of the head, shape ``(Q,)``.
        geo: Row geometry.
        style: Sizes and edge options.
    """
    L = geo.L
    ax.set_xlim(*geo.xlim)
    ax.set_ylim(*geo.ylim)
    pos, labels = phmm_layout(L, geo.matches_per_row, geo.n_bands)
    Q = len(labels)
    B, E, C = 3 * L, 3 * L + 1, 3 * L + 2

    A_disp = A.copy()
    # Entry and exit jumps are annotated per state instead of drawn.
    A_disp[B, 1:L] = 0
    A_disp[0:L - 1, E] = 0
    # E -> C and C -> B are routed along the page border by hand.
    A_disp[E, C] = 0
    A_disp[C, B] = 0
    # networkx sizes self-loops relative to the axes, which makes them huge
    # on a large page; they are drawn by hand with a fixed size.
    loops = np.diag(A_disp).copy()
    np.fill_diagonal(A_disp, 0)

    # Transitions crossing a row boundary end in ghost copies of the first
    # match and delete state of the next row, placed right of the row.
    ghost_rows = []
    ghost_edges = []  # (source, ghost index, probability)
    for k in geo.boundaries:
        r = geo.row_of(k)
        x = geo.matches_per_row + 1.0
        g_m = Q + 2 * len(ghost_rows)
        g_d = g_m + 1
        ghost_rows.append(((x, geo.y_match(r)), (x, geo.y_delete(r)), k + 1))
        m, i, d = k, L + k, 2 * L - 1 + k  # M, I and D state of column k
        for src, dst, ghost in (
            (m, m + 1, g_m), (i, m + 1, g_m), (d, m + 1, g_m),
            (m, d + 1, g_d), (d, d + 1, g_d),
        ):
            ghost_edges.append((src, ghost, A_disp[src, dst]))
            A_disp[src, dst] = 0

    n_ghosts = 2 * len(ghost_rows)
    A_full = np.zeros((Q + n_ghosts, Q + n_ghosts), dtype=A.dtype)
    A_full[:Q, :Q] = A_disp
    for src, ghost, p in ghost_edges:
        A_full[src, ghost] = p
    P_full = np.zeros(Q + n_ghosts, dtype=P.dtype)
    labels_full = list(labels)
    pos_full = dict(pos)
    for (pos_m, pos_d, first) in ghost_rows:
        pos_full[len(labels_full)] = pos_m
        labels_full.append(f"M{first + 1}")
        pos_full[len(labels_full)] = pos_d
        labels_full.append(f"D{first + 1}")

    n_texts = len(ax.texts)
    n_collections = len(ax.collections)
    plot_transition_graph(
        _DisplayTransitioner(A_full, P_full),
        head=0,
        pos=pos_full,
        state_labels=labels_full,
        node_size=style.node_size,
        font_size=style.font_size,
        edge_font_size=style.edge_font_size,
        arrows_style=style.arrows_style,
        edge_label_fmt=style.edge_label_fmt,
        threshold=style.threshold,
        label_pos=style.label_pos,
        connectionstyle=style.connectionstyle,
        show_start_dist=False,
        ax=ax,
    )
    # networkx puts edge labels at the same zorder as the edges; lift them so
    # their white boxes cover crossing edges, but keep them below the nodes.
    for text in ax.texts[n_texts:]:
        text.set_zorder(max(text.get_zorder(), 1.5))

    # Color the states by group. hidten draws all nodes in one collection
    # (in node order) and their labels as the first texts, before the edge
    # labels.
    face, label_color = _state_colors(L, Q)
    face += [_OTHER_COLOR] * n_ghosts  # ghosts are hollow and hidden below
    for coll in ax.collections[n_collections:]:
        if len(coll.get_offsets()) == Q + n_ghosts:
            coll.set_facecolor(face)
            break
    for text, color in zip(ax.texts[n_texts:n_texts + Q], label_color):
        text.set_color(color)

    # Ghost nodes: hollow, dashed and labelled in the color of their group.
    for g in range(Q, Q + n_ghosts):
        x, y = pos_full[g]
        color = _MATCH_COLOR if (g - Q) % 2 == 0 else _DELETE_COLOR
        ax.scatter(
            [x], [y], s=style.node_size, facecolors="white",
            edgecolors=color, linestyles="--", linewidths=1.0, zorder=3,
        )
        ax.text(
            x, y, labels_full[g], ha="center", va="center", zorder=4,
            fontsize=style.font_size, color=color,
        )
    # Continuation arrows at the start of the following row.
    for k in geo.boundaries:
        r = geo.row_of(k) + 1
        for y, color in (
            (geo.y_match(r), _MATCH_COLOR), (geo.y_delete(r), _DELETE_COLOR),
        ):
            ax.annotate(
                "", xy=(0.72, y), xytext=(0.2, y),
                arrowprops=dict(
                    arrowstyle=style.arrows_style, color=color,
                    linestyle=":", shrinkA=0, shrinkB=0,
                ),
            )

    # Per-state entry (B -> Mi) and exit (Mi -> E) probabilities, stacked in
    # the free space above the match state, between the insert states.
    ann_font = style.edge_font_size
    for i in range(L):
        x = pos[i][0]
        y = geo.y_insert(geo.row_of(i))
        if i > 0 and A[B, i] >= style.jump_label_threshold:
            ax.text(
                x, y + 0.1, "in \n" + style.edge_label_fmt.format(A[B, i]),
                ha="center", va="center", fontsize=ann_font, color="dimgray",
            )
        if i < L - 1 and A[i, E] >= style.jump_label_threshold:
            ax.text(
                x, y - 0.15, "out \n" + style.edge_label_fmt.format(A[i, E]),
                ha="center", va="center", fontsize=ann_font, color="dimgray",
            )

    # Self-loops: left of C, above everything else.
    x_px = ax.transData.transform((1, 0))[0] - ax.transData.transform((0, 0))[0]
    unit_pt = x_px * 72.0 / ax.get_figure().dpi
    radius = math.sqrt(style.node_size / math.pi) / unit_pt
    for q in range(Q):
        if loops[q] < style.threshold:
            continue
        direction = 180.0 if q == C else 90.0
        _draw_self_loop(ax, pos[q], radius, direction, float(loops[q]), style)

    # Border-routed E -> C (down, then left along the bottom) and C -> B (up).
    radius_pt = math.sqrt(style.node_size / math.pi)
    for src, dst, cstyle, label_xy in (
        (E, C, "angle,angleA=-90,angleB=0,rad=0",
         ((pos[E][0] + pos[C][0]) / 2, pos[C][1])),
        (C, B, "arc3,rad=0",
         (pos[C][0], (pos[C][1] + pos[B][1]) / 2)),
    ):
        p = float(A[src, dst])
        if p < style.threshold:
            continue
        color, width = _edge_color_width(p)
        ax.add_patch(FancyArrowPatch(
            pos[src], pos[dst], connectionstyle=cstyle,
            arrowstyle=style.arrows_style, mutation_scale=10,
            color=color, linewidth=width,
            shrinkA=radius_pt, shrinkB=radius_pt, zorder=1,
        ))
        ax.text(
            *label_xy, style.edge_label_fmt.format(p),
            ha="center", va="center", fontsize=style.edge_font_size,
            bbox=dict(boxstyle="round", ec="white", fc="white"), zorder=2,
        )



def _draw_self_loop(
    ax,
    xy: tuple[float, float],
    radius: float,
    direction: float,
    p: float,
    style: _GraphStyle,
) -> None:
    """A small self-loop on the node at *xy* pointing towards *direction*
    (degrees), with its probability label at the far end."""
    def at(angle: float, dist: float) -> tuple[float, float]:
        a = math.radians(direction + angle)
        return xy[0] + dist * math.cos(a), xy[1] + dist * math.sin(a)

    reach = 3.2 * radius
    path = Path(
        [at(30, radius), at(40, reach), at(-40, reach), at(-30, radius)],
        [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4],
    )
    # The cubic's apex lies at 1/4 of the rim and 3/4 of the control points.
    apex = 0.25 * radius * math.cos(math.radians(30)) \
        + 0.75 * reach * math.cos(math.radians(40))
    color, width = _edge_color_width(p)
    ax.add_patch(FancyArrowPatch(
        path=path, arrowstyle=style.arrows_style, mutation_scale=8,
        color=color, linewidth=width, zorder=1,
    ))
    ax.text(
        *at(0, apex + 0.1), style.edge_label_fmt.format(p),
        ha="center", va="center", fontsize=style.edge_font_size,
    )


def _draw_header(
    ax,
    geo: PHMMGeometry,
    title: str,
    labels: list[str],
    P: np.ndarray,
    style: _GraphStyle,
    title_font_size: float,
) -> None:
    """Title, start distribution and a legend for the jump annotations."""
    x0, x1 = geo.xlim
    ax.text(
        (x0 + x1) / 2, -0.4, title, ha="center", va="center",
        fontsize=title_font_size, fontweight="bold",
    )
    start = ", ".join(
        f"{labels[i]} {style.edge_label_fmt.format(float(P[i]))}"
        for i in range(len(labels)) if float(P[i]) > style.threshold
    )
    ax.text(
        x0 + 0.2, -0.95,
        f"Start: {start}     in = B→Mi, out = Mi→E "
        f"(shown if ≥ {style.jump_label_threshold:g})",
        ha="left", va="center", fontsize=style.font_size,
    )


def plot_phmm(
    layer: PHMMLayer,
    head: int = 0,
    ax=None,
    title: str | None = None,
    matches_per_row: int | None = None,
    target_aspect: float = 1.6,
    unit_size: float = 0.9,
    threshold: float = 1e-12,
    jump_label_threshold: float = 0.01,
    edge_label_fmt: str = "{:.2f}",
    node_size: float | None = None,
    font_size: float | None = None,
    edge_font_size: float | None = None,
    label_pos: float = 0.62,
    connectionstyle: str = "arc3,rad=0.15",
    arrows_style: str = "-|>",
    title_font_size: float | None = None,
    fast_mode: bool = False,
    show_information_content: bool = False,
) -> Figure | SubFigure | None:
    """Plots the state transition graph of a profile HMM using the explicit
    (unfolded) transition matrix, with the emission logos of the match states
    on top.

    The match chain is wrapped into rows, like lines of text, so that the
    figure keeps a readable aspect ratio even for hundreds of match states:

    - Each row shows its emission logos (one band per emitter), then insert
      states between adjacent match states, the match states and delete
      states aligned below the match states.
    - Transitions into the next row end in dashed ghost copies of that row's
      first match and delete state; a dotted arrow marks where the row
      continues.
    - Left-flank (L) and begin (B) states are left of the first row, end (E),
      right-flank (R) and terminal (T) states right of the last match state
      and the unannotated state (C) below the last row.
    - Entry (B→Mi) and exit (Mi→E) jumps are written next to the match states
      instead of being drawn as edges.

    The HMM must have a ``PHMMTransitioner`` as its ``.transitioner``.

    Args:
        layer: The pHMM layer to visualize.
        head: Index of the HMM head to visualize (default: 0).
        ax: A matplotlib Axes to draw on. If None, a new figure sized to the
            layout is created.
        title: Optional title for the plot.
        matches_per_row: Match states per row. ``None`` (default) chooses it
            so that the figure's width/height ratio is close to
            *target_aspect*.
        target_aspect: Desired width/height ratio when *matches_per_row* is
            chosen automatically.
        unit_size: Size in inches of the spacing between two adjacent match
            states. Node and font sizes scale with it. Ignored when *ax* is
            given; the layout is then fitted into the axes.
        threshold: Edges with probability below this value are omitted.
        jump_label_threshold: Entry and exit probabilities below this value
            are not annotated.
        edge_label_fmt: Format string for edge probability labels.
        node_size: Size of each node circle in pt². ``None`` derives it from
            *unit_size*.
        font_size: Font size for node labels. ``None`` derives it from
            *unit_size*.
        edge_font_size: Font size for edge probability labels. ``None``
            derives it from *unit_size*.
        label_pos: Position of edge label along edge (0=head, 0.5=center, 1=tail).
        connectionstyle: Matplotlib connectionstyle string for curved edges.
        arrows_style: Matplotlib arrow style string.
        title_font_size: Font size for the plot title. ``None`` derives it
            from the node font size.
        fast_mode: If *True*, use a two-panel vertical layout. The **top
            panel** contains one row per active emitter; each row shows the
            full match-state profile as a single sequence logo spanning all *L*
            positions plus a single-position insert representative logo next
            to it. The **bottom panel** shows the transition graph in a single
            row. Figure width is capped at 400 inches. Requires ``ax=None``
            (default: ``False``).
        show_information_content: If *True*, scale each logo column by the
            KL divergence (information content) of the match-state emission
            relative to the insert-state background distribution, so taller
            columns reflect positions that deviate more from background.
            Only affects categorical emission tracks. Default: *False*.

    Returns:
        The matplotlib Figure.
    """
    transitioner = layer.transitioner
    if not _is_phmm_transitioner(transitioner):
        raise TypeError(
            "plot_phmm requires an HMM with a PHMMTransitioner, "
            f"got {type(transitioner).__name__}"
        )

    explicit = transitioner.explicit_transitioner
    L = transitioner.lengths[head]
    _, labels = phmm_layout(L)
    Q = len(labels)
    A = to_numpy(explicit.matrix())[head, :Q, :Q]
    P = to_numpy(explicit.start_dist())[head, :Q]

    _plot_title = f"Profile HMM (head {head})" if title is None else title
    panels = _emission_panels(layer)

    if fast_mode:
        if ax is not None:
            raise ValueError("fast_mode=True requires ax=None.")
        return _plot_phmm_fast(
            A, P, L, head, panels, _plot_title, labels,
            threshold=threshold,
            jump_label_threshold=jump_label_threshold,
            edge_label_fmt=edge_label_fmt,
            label_pos=label_pos,
            connectionstyle=connectionstyle,
            arrows_style=arrows_style,
            show_information_content=show_information_content,
        )

    n_bands = len(panels)
    if matches_per_row is None:
        matches_per_row = choose_matches_per_row(L, n_bands, target_aspect)
    geo = PHMMGeometry(L, min(matches_per_row, L), n_bands)

    (x0, x1), (y0, y1) = geo.xlim, geo.ylim
    if ax is None:
        fig = plt.figure(figsize=((x1 - x0) * unit_size, (y1 - y0) * unit_size))
        ax = fig.add_axes((0, 0, 1, 1))
    else:
        fig = ax.get_figure()
        # Fit the layout into the given axes with one unit equal in x and y.
        ax.set_aspect("equal", adjustable="box")
        box = ax.get_position()
        fig_w, fig_h = fig.get_size_inches()
        unit_size = min(box.width * fig_w / (x1 - x0), box.height * fig_h / (y1 - y0))

    auto_node, auto_font, auto_edge_font = _auto_sizes(unit_size)
    style = _GraphStyle(
        node_size=auto_node if node_size is None else node_size,
        font_size=auto_font if font_size is None else font_size,
        edge_font_size=(
            auto_edge_font if edge_font_size is None else edge_font_size
        ),
        threshold=threshold,
        edge_label_fmt=edge_label_fmt,
        label_pos=label_pos,
        connectionstyle=connectionstyle,
        arrows_style=arrows_style,
        jump_label_threshold=jump_label_threshold,
    )

    _draw_phmm_graph(ax, A, P, geo, style)
    _draw_header(
        ax, geo, _plot_title, labels, P, style,
        title_font_size=(
            1.6 * style.font_size if title_font_size is None
            else title_font_size
        ),
    )
    _draw_logo_bands(
        ax, geo, panels, head, style.font_size, show_information_content,
    )
    return fig


@dataclass(frozen=True)
class _EmissionPanel:
    matrix: np.ndarray
    kind: str
    label: str
    alphabet: str = "ARNDCQEGHILKMFPSTWYVXUO"
    color_scheme: str = "skylign_protein"


def _emission_panels(layer: PHMMLayer) -> list[_EmissionPanel]:
    """The emission tracks of *layer*, in display order, as numpy matrices."""
    panels = []
    aa_kwargs = dict(
        kind="TFCategoricalEmitter", label="AA",
        alphabet=SequenceDataset._default_alphabet,
        color_scheme="skylign_protein",
    )
    if layer.joint_emitter:
        if layer.joint_emitter.conditional:
            assert layer.profile_emitter is not None
            aa_matrix = layer.profile_emitter.matrix()
            struct_matrix = layer.joint_emitter.marginal_matrix_from_conditional(
                prior=layer.profile_emitter.matrix(),
            )
        else:
            aa_matrix, struct_matrix = layer.joint_emitter.marginal_matrices()
        assert layer.structural_config is not None, \
            "Structural config must be provided if use_structure is True"
        panels.append(_EmissionPanel(to_numpy(aa_matrix), **aa_kwargs))
        panels.append(_EmissionPanel(
            to_numpy(struct_matrix), "TFCategoricalEmitter", "3Di",
            alphabet=layer.structural_config.structural_alphabet,
            color_scheme="NajafabadiEtAl2017",
        ))
    else:
        if not layer.no_aa:
            assert layer.profile_emitter is not None
            panels.append(
                _EmissionPanel(layer.emission_matrix("aa"), **aa_kwargs)
            )
        if layer.use_structure:
            assert layer.structural_config is not None, \
                "Structural config must be provided if use_structure is True"
            assert layer.struct_emitter is not None
            panels.append(_EmissionPanel(
                layer.emission_matrix("struct"), "TFCategoricalEmitter", "3Di",
                alphabet=layer.structural_config.structural_alphabet,
                color_scheme="NajafabadiEtAl2017",
            ))
    if layer.use_language_model:
        assert layer.embedding_emitter is not None
        panels.append(_EmissionPanel(
            layer.emission_matrix("emb"), "TFMVNormalEmitter", "emb",
        ))
    return panels


def _draw_logo_bands(
    ax,
    geo: PHMMGeometry,
    panels: list[_EmissionPanel],
    head: int,
    font_size: float,
    show_information_content: bool,
) -> None:
    """One logo strip per row and emission track, with column j of a strip
    directly above the j-th match state of the row, plus one insert
    representative logo per track left of the first row."""
    for r in range(geo.n_rows):
        first = r * geo.matches_per_row
        states = range(first, min(first + geo.matches_per_row, geo.L))
        for b, panel in enumerate(panels):
            height = _LOGO_BAND_HEIGHT - 0.2
            y = geo.band_bottom(r, b) + 0.1
            ax_match = ax.inset_axes(
                (0.5, y, len(states), height), transform=ax.transData,
            )
            ax_ins = None
            if r == 0:
                ax_ins = ax.inset_axes(
                    (-2.15, y, 1.0, height), transform=ax.transData,
                )
            _render_emitter_fast(
                panel.matrix, panel.kind, head, geo.L, ax_match, ax_ins,
                alphabet=panel.alphabet,
                color_scheme=panel.color_scheme,
                label=panel.label,
                show_information_content=show_information_content,
                states=states,
                font_size=font_size,
                insert_title=b == 0,
            )


def _plot_phmm_fast(
    A: np.ndarray,
    P: np.ndarray,
    L: int,
    head: int,
    panels: list[_EmissionPanel],
    title: str,
    labels: list[str],
    show_information_content: bool,
    **style_kwargs,
):
    logo_row_h = 3   # inches per emitter row in the logo panel
    n_active = max(1, len(panels))
    geo = PHMMGeometry(L, L, n_bands=0)

    fig_w = min(L * 0.5 + 4, 400)
    x0, x1 = geo.xlim
    y0, y1 = geo.ylim
    graph_h = max(4, min(fig_w * (y1 - y0) / (x1 - x0), 30))

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
    for row, panel in enumerate(panels):
        _render_emitter_fast(
            panel.matrix, panel.kind, head, L,
            fig.add_subplot(top_gs[row, 0]), fig.add_subplot(top_gs[row, 1]),
            alphabet=panel.alphabet,
            color_scheme=panel.color_scheme,
            label=panel.label,
            show_information_content=show_information_content,
        )

    # ── Bottom panel: transition graph, no emission insets ───────────────
    unit_in = fig_w / (x1 - x0)
    node_size, font_size, edge_font_size = _auto_sizes(unit_in)
    style = _GraphStyle(
        node_size=node_size, font_size=font_size,
        edge_font_size=edge_font_size, **style_kwargs,
    )
    ax_graph = fig.add_subplot(outer_gs[1])
    _draw_phmm_graph(ax_graph, A, P, geo, style)
    _draw_header(
        ax_graph, geo, title, labels, P, style,
        title_font_size=1.6 * font_size,
    )
    return fig


def _render_emitter_fast(
    matrix: np.ndarray,
    type: str,
    head: int,
    L: int,
    ax_match,
    ax_ins=None,
    alphabet: str = "ARNDCQEGHILKMFPSTWYVXUO",
    color_scheme: str = "skylign_protein",
    label: str = "",
    show_information_content: bool = False,
    states: range | None = None,
    font_size: float = 8,
    insert_title: bool = True,
) -> None:
    """Draw a match-state profile logo and a single insert representative logo
    side by side using direct draw calls.

    Args:
        matrix: The matrix whose emitter is rendered.
        type: Type of the emitter.
        head: Emitter head index.
        L: Number of match states. Match states are indices 0…L-1; the
            representative insert state is index *L*.
        ax_match: Axes for the match profile.
        ax_ins: Axes for the single insert-representative logo, or ``None``
            to skip it.
        alphabet: Alphabet string used for categorical emitters.
        color_scheme: Logomaker color scheme for categorical emitters.
        label: Y-axis label placed on *ax_match*.
        show_information_content: If *True*, scale match-state logo heights by
            the per-position KL divergence (bits) relative to the insert-state
            background. Has no effect for non-categorical emitters.
        states: Match states drawn in *ax_match*, one column each (default:
            all *L*). Column ``j`` is centered at ``x = j``.
        font_size: Font size of the axis label and the insert title.
        insert_title: Whether to title *ax_ins* with "ins.".
    """
    import warnings
    import pandas as pd
    import logomaker

    M_np = matrix
    if states is None:
        states = range(L)
    idx = np.arange(states.start, states.stop)

    if type == "TFCategoricalEmitter":
        A = M_np.shape[2]
        chars = list(alphabet[:A])
        match_probs = M_np[head, idx, :]       # (n, A)
        ins_probs = M_np[head, L : L + 1, :]  # (1, A)
        if show_information_content:
            background = M_np[head, L, :]      # insert state as background
            eps = 1e-10
            p = np.clip(match_probs, eps, 1.0)
            q = np.clip(background, eps, 1.0)
            # KL divergence (bits) per position: sum_a p * log2(p/q)
            ic = np.sum(
                p * np.log2(p / q[np.newaxis, :]), axis=-1, keepdims=True
            )  # (n, 1)
            match_data = ic * match_probs
        else:
            match_data = match_probs
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*not in color_dict.*", category=UserWarning
            )
            # Match profile — one logomaker call for all positions
            df_match = pd.DataFrame(match_data, columns=chars)
            logomaker.Logo(
                df_match, ax=ax_match,
                color_scheme=color_scheme, vpad=0.1, width=0.8,
            )
            if ax_ins is not None:
                # Insert representative (state L) — always raw probabilities
                df_ins = pd.DataFrame(ins_probs, columns=chars)
                logomaker.Logo(
                    df_ins, ax=ax_ins,
                    color_scheme=color_scheme, vpad=0.1, width=0.8,
                )
        if show_information_content:
            ax_match.set_ylabel(f"{label}\n(bits)", fontsize=font_size)
            ax_match.set_xticks([])
            ax_match.tick_params(axis="y", labelsize=0.8 * font_size)
            from matplotlib.ticker import FuncFormatter
            ax_match.yaxis.set_major_formatter(
                FuncFormatter(lambda x, _: f"{x:.1f}")
            )
        else:
            ax_match.set_ylabel(label, fontsize=font_size)
            ax_match.set_xticks([])
            ax_match.set_yticks([])

    elif type == "TFMVNormalEmitter":
        D = M_np.shape[2] // 2
        colors = plt.cm.tab10.colors  # type: ignore
        # Match: heatmap of per-dimension means across the match states
        means_match = M_np[head, idx, :D]  # (n, D)
        ax_match.imshow(
            means_match.T, aspect="auto", cmap="RdBu_r",
            interpolation="nearest",
        )
        ax_match.set_ylabel(label, fontsize=font_size)
        ax_match.set_xticks([])
        ax_match.set_yticks([])
        if ax_ins is not None:
            # Insert representative: means as a bar chart
            ins_means = M_np[head, L, :D]
            ax_ins.bar(
                range(D), ins_means,
                color=[colors[d % len(colors)] for d in range(D)],
            )

    ax_match.set_xlim(-0.5, len(idx) - 0.5)
    if ax_ins is not None:
        if insert_title:
            ax_ins.set_title("ins.", fontsize=font_size)
        ax_ins.set_xticks([])
        ax_ins.set_yticks([])




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