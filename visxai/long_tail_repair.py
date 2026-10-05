"""
Graph repair needs long-tailed attraction: a toy example.

A circle's k-nearest-neighbour graph is corrupted by rewiring: some true edges
dropped, a few random long-range "shortcut" edges added. The corrupted graph
is laid out twice from the same spectral start, minimising

    F(Y) = sum_edges phi(d^2) + gamma * sum_all_pairs rho(d^2),
    phi = -log q,  rho = -log(1 - q),

with exact all-pairs repulsion and Adam, for two attraction kernels q:

    spring       q = exp(-d^2)       pull grows with distance
    heavy tail   q = 1 / (1 + d^2)   pull fades with distance

Springs let the shortcuts drag the circle out of shape; the heavy tail lets
the rest of the graph stretch the shortcuts and the circle recovers.

    manim-slides render long_tail_repair.py LongTailRepair

The layouts take a few seconds and are computed at render time.
"""
from manim import *

import sys

sys.path.append("..")  # Add parent directory to path to import config
sys.path.append(".")

from config import (
    apply_defaults,
    COLOR_CYCLE,
    DEFAULT_COLOR,
    ACCENT_COLOR,
    HIGHLIGHT_COLOR,
    BACKGROUND_COLOR,
    add_logo_to_background,
    create_styled_axes,
    TIMCSlide,
    ThreeDTIMCSlide,
    PhaseSlide,
    colormap_color,
    create_logo,
)

import numpy as np

apply_defaults()

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
N, HALF = 120, 5           # points; each joined to HALF neighbours either side
N_SHORTCUTS, N_MISSING = 10, 30
SEED = 1
STEPS, SNAPSHOT_EVERY = 1500, 15
GAMMA = 0.1 * 96 / N * HALF / 3

SPRING_COLOR = COLOR_CYCLE[0]
TAIL_COLOR = COLOR_CYCLE[2]
SHORTCUT_COLOR = HIGHLIGHT_COLOR
EDGE_COLOR = DEFAULT_COLOR
RUN_SECONDS = 8.0
T_CYCLE = [COLOR_CYCLE[i] for i in (0, 4, 3, 1, 2, 5)]   # cyclic point colours


# ---------------------------------------------------------------------------
# The toy graph and the two layouts
# ---------------------------------------------------------------------------
def gap(a, b):
    x = np.abs(a - b) % (2 * np.pi)
    return np.minimum(x, 2 * np.pi - x)


# derivative (in s = d^2) of attraction phi and of repulsion rho
KERNELS = {
    "spring": (lambda s: 1.0 + 0 * s,
               lambda s: -np.exp(-s) / (1 - np.exp(-s) + 1e-12)),
    "tail": (lambda s: 1 / (1 + s),
             lambda s: -1 / (s * (1 + s) + 1e-12)),
}


def corrupted_circle():
    rng = np.random.default_rng(SEED)
    theta = 2 * np.pi * (np.arange(N) + 0.3 * rng.uniform(-1, 1, N)) / N
    true = sorted({(min(i, (i + s) % N), max(i, (i + s) % N))
                   for i in range(N) for s in range(1, HALF + 1)})
    shortcuts = set()
    while len(shortcuts) < N_SHORTCUTS:
        a, b = rng.choice(N, 2, replace=False)
        if gap(theta[a], theta[b]) > np.pi / 2:
            shortcuts.add((min(a, b), max(a, b)))
    drop = set(rng.choice(len(true), N_MISSING, replace=False).tolist())
    kept = [e for m, e in enumerate(true) if m not in drop]
    missing = [e for m, e in enumerate(true) if m in drop]
    return theta, np.array(kept), np.array(missing), np.array(sorted(shortcuts))


def spectral(edges, n):
    W = np.zeros((n, n))
    W[edges[:, 0], edges[:, 1]] = W[edges[:, 1], edges[:, 0]] = 1
    inv = 1 / np.sqrt(W.sum(1))
    _, vecs = np.linalg.eigh(np.eye(n) - inv[:, None] * W * inv[None])
    Y = vecs[:, 1:3] * inv[:, None]
    return 2.0 * Y / np.abs(Y).max()


def layout(Y0, edges, kernel, lr=0.02):
    """Adam on F(Y) with exact all-pairs repulsion; returns snapshots."""
    dphi, drho = KERNELS[kernel]
    Y, n = Y0.copy(), len(Y0)
    m, v = np.zeros_like(Y), np.zeros_like(Y)
    a, b = edges[:, 0], edges[:, 1]
    snaps = [Y.copy()]
    for t in range(1, STEPS + 1):
        diff = Y[:, None] - Y[None]
        s = (diff ** 2).sum(-1)
        R = drho(s + np.eye(n))
        np.fill_diagonal(R, 0)
        g = GAMMA * 2 * (R[:, :, None] * diff).sum(1)
        e = Y[a] - Y[b]
        pull = dphi((e ** 2).sum(1))[:, None] * 2 * e
        np.add.at(g, a, pull)
        np.add.at(g, b, -pull)
        m = 0.9 * m + 0.1 * g
        v = 0.99 * v + 0.01 * g ** 2
        Y -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.99 ** t)) + 1e-8)
        if t % SNAPSHOT_EVERY == 0:
            snaps.append(Y.copy())
    return np.array(snaps)


def normalise(Y, centre, radius):
    """Centre a layout and scale its RMS radius (layouts come out at
    arbitrary scale) for display."""
    Y = Y - Y.mean(0)
    Y = Y * radius / np.sqrt((Y ** 2).sum(1).mean())
    return np.c_[Y, np.zeros(len(Y))] + centre


def cyclic_color(u):
    x = (u % 1.0) * len(T_CYCLE)
    i = int(np.floor(x))
    return interpolate_color(T_CYCLE[i], T_CYCLE[(i + 1) % len(T_CYCLE)], x - i)


# ---------------------------------------------------------------------------
class LongTailRepair(TIMCSlide):

    def heading(self, title, subtitle):
        t = Text(title, color=DEFAULT_COLOR).scale(0.6).to_edge(UP, buff=0.3)
        s = Text(subtitle, color=ACCENT_COLOR).scale(0.32).next_to(
            t, DOWN, buff=0.12)
        return VGroup(t, s)

    def graph_mobjects(self, positions_fn, kept, shortcuts, dot_radius=0.05):
        """Dots, true edges and shortcut edges that follow positions_fn()."""
        P = positions_fn()
        dots = VGroup(*[Dot(P[i], radius=dot_radius,
                            color=cyclic_color(self.theta[i] / TAU))
                        for i in range(N)])
        true_edges = VMobject(stroke_color=EDGE_COLOR, stroke_width=1.4,
                              stroke_opacity=0.5)
        short_edges = VMobject(stroke_color=SHORTCUT_COLOR, stroke_width=3)

        def segs(m, E):
            P = positions_fn()
            A, B = P[E[:, 0]], P[E[:, 1]]
            pts = np.empty((4 * len(A), 3))
            pts[0::4], pts[3::4] = A, B
            pts[1::4] = A + (B - A) / 3
            pts[2::4] = A + 2 * (B - A) / 3
            m.set_points(pts)

        segs(true_edges, kept)
        segs(short_edges, shortcuts)
        true_edges.add_updater(lambda m: segs(m, kept))
        short_edges.add_updater(lambda m: segs(m, shortcuts))

        def move(group):
            P = positions_fn()
            for d, p in zip(group, P):
                d.move_to(p)

        dots.add_updater(move)
        return dots, true_edges, short_edges

    def construct(self):
        self.theta, kept, missing, shortcuts = corrupted_circle()
        observed = np.r_[kept, shortcuts]
        start = spectral(observed, N)
        snaps = {k: layout(start, observed, k) for k in ("spring", "tail")}

        # ---- 1. the circle and its kNN graph ----
        circle = np.c_[2.4 * np.cos(self.theta), 2.4 * np.sin(self.theta),
                       np.zeros(N)] + 0.35 * DOWN
        dots, true_e, _ = self.graph_mobjects(lambda: circle,
                                              np.r_[kept, missing],
                                              np.zeros((0, 2), int))
        head = self.heading("A circle and its nearest-neighbour graph",
                            f"each point joined to its {2 * HALF} nearest "
                            "neighbours along the circle")
        self.play(FadeIn(head))
        self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in dots],
                              lag_ratio=0.01))
        self.play(Create(true_e), run_time=1.5)
        self.marked_next_slide()

        # ---- 2. rewiring: drop some edges, add random shortcuts ----
        new = self.heading("Rewired edges",
                           f"{N_MISSING} true edges dropped, {N_SHORTCUTS} "
                           "random long-range shortcuts added")
        kept_e = VMobject(stroke_color=EDGE_COLOR, stroke_width=1.4,
                          stroke_opacity=0.5)
        A, B = circle[kept[:, 0]], circle[kept[:, 1]]
        pts = np.empty((4 * len(A), 3))
        pts[0::4], pts[3::4] = A, B
        pts[1::4], pts[2::4] = A + (B - A) / 3, A + 2 * (B - A) / 3
        kept_e.set_points(pts)
        chords = VGroup(*[Line(circle[a], circle[b], color=SHORTCUT_COLOR,
                               stroke_width=3) for a, b in shortcuts])
        self.add(kept_e)
        self.play(FadeOut(true_e), FadeTransform(head, new))
        self.play(LaggedStart(*[Create(c) for c in chords], lag_ratio=0.15),
                  run_time=2)
        head = new
        self.marked_next_slide()

        # ---- 3. two attraction profiles ----
        new = self.heading("Two kinds of attraction",
                           "how hard an edge pulls its two ends together, "
                           "by how far apart they are")
        self.play(FadeOut(dots, kept_e, chords), FadeTransform(head, new))
        head = new
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 3, 1], x_length=6.5,
                    y_length=3.6, tips=False,
                    axis_config=dict(color=DEFAULT_COLOR, stroke_width=2)
                    ).move_to(0.5 * DOWN)
        x_lab = Text("distance", font_size=24, color=DEFAULT_COLOR).next_to(
            axes.x_axis, DOWN, buff=0.25)
        y_lab = Text("pull", font_size=24, color=DEFAULT_COLOR).next_to(
            axes.y_axis, LEFT, buff=0.25)
        spring = axes.plot(lambda d: 2 * d, x_range=[0, 1.5],
                           color=SPRING_COLOR, stroke_width=5)
        tail = axes.plot(lambda d: 2 * d / (1 + d * d), x_range=[0, 4],
                         color=TAIL_COLOR, stroke_width=5)
        spring_lab = Text("spring: pulls harder the further apart",
                          font_size=24, color=SPRING_COLOR).next_to(
            axes.c2p(1.5, 3), RIGHT, buff=0.2)
        tail_lab = Text("heavy tail: the pull fades with distance",
                        font_size=24, color=TAIL_COLOR).next_to(
            axes.c2p(2.6, 0.45), DOWN, buff=0.1)
        self.play(Create(axes), FadeIn(x_lab, y_lab))
        self.play(Create(spring), FadeIn(spring_lab))
        self.play(Create(tail), FadeIn(tail_lab))
        self.marked_next_slide()

        # ---- 4. lay out the rewired graph both ways ----
        new = self.heading("Lay out the rewired graph",
                           "same graph, same spectral start; only the "
                           "attraction differs")
        self.play(FadeOut(axes, x_lab, y_lab, spring, tail, spring_lab,
                          tail_lab), FadeTransform(head, new))
        head = new
        frame = ValueTracker(0.0)
        F = len(snaps["spring"])
        centres = {"spring": np.array([-3.5, -0.3, 0]),
                   "tail": np.array([3.5, -0.3, 0])}

        def positions(kind):
            def fn():
                f = frame.get_value()
                i = int(np.clip(np.floor(f), 0, F - 2))
                a = f - i
                Y = (1 - a) * snaps[kind][i] + a * snaps[kind][i + 1]
                return normalise(Y, centres[kind], 1.8)
            return fn

        panels = {}
        for kind, name, col in (("spring", "Spring attraction", SPRING_COLOR),
                                ("tail", "Heavy-tailed attraction",
                                 TAIL_COLOR)):
            d, e, s = self.graph_mobjects(positions(kind), kept, shortcuts,
                                          dot_radius=0.045)
            title = Text(name, font_size=28, color=col).move_to(
                centres[kind] + 2.5 * UP)
            panels[kind] = (VGroup(e, s, d), title)
            self.play(FadeIn(title), FadeIn(VGroup(e, s, d)), run_time=0.8)
        self.marked_next_slide()

        self.play(frame.animate.set_value(F - 1), run_time=RUN_SECONDS,
                  rate_func=smooth)
        self.marked_next_slide()

        # ---- 5. the outcome ----
        notes = VGroup(
            Text("the shortcuts pull hardest:\nthe circle is dragged out "
                 "of shape", font_size=22, color=SPRING_COLOR,
                 line_spacing=0.8).move_to(centres["spring"] + 3.05 * DOWN),
            Text("the shortcuts give way, stretched long:\nthe circle "
                 "recovers", font_size=22, color=TAIL_COLOR,
                 line_spacing=0.8).move_to(centres["tail"] + 3.05 * DOWN))
        new = self.heading("Repair needs long-tailed attraction",
                           "a few bad edges are outvoted by the rest of the "
                           "graph -- if their pull fades with distance")
        self.play(FadeIn(notes), FadeTransform(head, new))
        self.marked_next_slide()
