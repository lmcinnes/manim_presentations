"""
Reading the attraction kernel: what each parameter means, then what it does.

    q(d) = (1 + (1/alpha) (d/sigma0)^m)^(-alpha)

    sigma0   the units: the typical distance to a nearest neighbour
    m        target intrinsic dimension: how space-filling the layout is
    alpha    certainty about the local density
    m*alpha  the tail, pull ~ m alpha / d: the denoising dial

Part 1 explains each parameter with a curved-arrow note on a large formula.
Part 2 moves the formula up, adds edge-probability and pull plots, and sweeps
each parameter (labelled with a short name only); m and alpha are followed by
real layouts precomputed by kernel_layouts_prep.py (MNIST for m, the rewired
circle for alpha).

    python kernel_layouts_prep.py       # once: writes kernel_layouts.npz
    manim-slides render kernel_knobs.py KernelKnobs
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

import os
from pathlib import Path

import numpy as np

apply_defaults()

_HERE = Path(__file__).resolve().parent
DATA_FILE = Path(os.environ.get("KERNEL_LAYOUTS",
                                _HERE / "kernel_layouts.npz"))

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
SIGMA_COLOR = COLOR_CYCLE[0]
M_COLOR = COLOR_CYCLE[3]
ALPHA_COLOR = COLOR_CYCLE[2]
TAIL_COLOR = HIGHLIGHT_COLOR
CURVE_COLOR = DEFAULT_COLOR
GHOST_OPACITY = 0.25

DEFAULTS = dict(sigma0=1.0, m=1.5, alpha=1.0)
SWEEPS = dict(sigma0=(0.55, 1.7), m=(1.0, 2.0), alpha=(0.5, 12.0))
SWEEP_SECONDS = 3.0
EVIDENCE_SECONDS = 5.0
D_MAX, PULL_MAX = 4.0, 2.2
CLASS_COLORS = COLOR_CYCLE[:10]
T_CYCLE = [COLOR_CYCLE[i] for i in (0, 4, 3, 1, 2, 5)]   # circle colours


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------
def q_kernel(d, sigma0, m, alpha):
    u = (np.maximum(d, 1e-9) / sigma0) ** m
    return (1 + u / alpha) ** (-alpha)


def pull(d, sigma0, m, alpha):
    """Attraction -d/dd log q."""
    d = np.maximum(d, 1e-9)
    u = (d / sigma0) ** m
    return m * u / d / (1 + u / alpha)


def curve(axes, f, y_max, color, width=4.5, opacity=1.0, n=240):
    """Plot f over [0, D_MAX], stopping where it leaves the top."""
    xs = np.linspace(1e-3, D_MAX, n)
    ys = np.array([f(x) for x in xs])
    keep = np.cumprod(ys <= y_max).astype(bool)
    vm = VMobject(stroke_color=color, stroke_width=width,
                  stroke_opacity=opacity)
    pts = [axes.c2p(x, y) for x, y in zip(xs[keep], ys[keep])]
    if len(pts) > 1:
        vm.set_points_as_corners(pts)
    return vm


# ---------------------------------------------------------------------------
# Layout panels (evidence)
# ---------------------------------------------------------------------------
def cyclic_color(u):
    x = (u % 1.0) * len(T_CYCLE)
    i = int(np.floor(x))
    return interpolate_color(T_CYCLE[i], T_CYCLE[(i + 1) % len(T_CYCLE)],
                             x - i)


def frame_at(snaps, progress):
    f = progress * (len(snaps) - 1)
    i = int(np.clip(np.floor(f), 0, len(snaps) - 2))
    a = f - i
    return (1 - a) * snaps[i] + a * snaps[i + 1]


def segments(P, E):
    A, B = P[E[:, 0]], P[E[:, 1]]
    pts = np.empty((4 * len(A), 3))
    pts[0::4], pts[3::4] = A, B
    pts[1::4], pts[2::4] = A + (B - A) / 3, A + 2 * (B - A) / 3
    return pts


# ---------------------------------------------------------------------------
class KernelKnobs(TIMCSlide):

    def heading(self, title, subtitle):
        t = Text(title, color=DEFAULT_COLOR).scale(0.6).to_edge(UP, buff=0.3)
        s = Text(subtitle, color=ACCENT_COLOR).scale(0.32).next_to(
            t, DOWN, buff=0.12)
        return VGroup(t, s)

    def arrow(self, start, end, color, angle):
        return CurvedArrow(start, end, angle=angle, color=color,
                           stroke_width=2.5, tip_length=0.16)

    def mnist_panels(self, progress):
        d = self.data
        y = d["mnist_labels"]
        rgba = np.array([[*ManimColor(CLASS_COLORS[l]).to_rgb(), 1.0]
                         for l in y])
        group = Group()
        for j, v in enumerate(d["mnist_m"]):
            snaps = d[f"mnist_m_{v:g}"]
            centre = np.array([(j - 1) * 4.6, -2.15, 0])

            def pos(snaps=snaps, centre=centre):
                Y = frame_at(snaps, progress.get_value())
                Y = Y - np.median(Y, axis=0)
                Y = Y * 1.3 / max(np.quantile(np.abs(Y), 0.99), 1e-9)
                return np.c_[Y, np.zeros(len(Y))] + centre

            pm = PMobject(stroke_width=1.6)
            pm.add_points(pos(), rgbas=rgba)
            pm.add_updater(lambda m_, pos=pos: m_.set(points=pos()))
            cap = (f"m = {v:g}:  measured dim "
                   f"{float(d[f'mnist_m_{v:g}_dimension']):.2f}, recall "
                   f"{float(d[f'mnist_m_{v:g}_recall']):.2f}")
            label = Text(cap, font_size=16, color=DEFAULT_COLOR).move_to(
                centre + 1.6 * UP)
            group.add(Group(pm, label))
        return group

    def circle_panels(self, progress):
        d = self.data
        th, kept, short = d["circle_theta"], d["circle_kept"], d["circle_short"]
        group = VGroup()
        for j, al in enumerate(d["circle_alphas"]):
            snaps = d[f"circle_alpha_{al:g}"]
            centre = np.array([(j - 1.5) * 3.5, -2.2, 0])

            def pos(snaps=snaps, centre=centre):
                Y = frame_at(snaps, progress.get_value())
                Y = Y - Y.mean(0)
                Y = Y * 1.05 / np.sqrt((Y ** 2).sum(1).mean())
                return np.c_[Y, np.zeros(len(Y))] + centre

            e_true = VMobject(stroke_color=DEFAULT_COLOR, stroke_width=1,
                              stroke_opacity=0.35)
            e_short = VMobject(stroke_color=TAIL_COLOR, stroke_width=2.2)
            for e, E in ((e_true, kept), (e_short, short)):
                e.set_points(segments(pos(), E))
                e.add_updater(lambda m_, pos=pos, E=E: m_.set_points(
                    segments(pos(), E)))
            dots = VGroup(*[Dot(radius=0.03, color=cyclic_color(t / TAU))
                            for t in th])

            def move(g, pos=pos):
                for dd, p in zip(g, pos()):
                    dd.move_to(p)

            move(dots)
            dots.add_updater(move)
            name = ("  (spring-like)" if al >= 4 else
                    "  (sceptical)" if al < 1 else "")
            label = Text(f"alpha = {al:g}{name}", font_size=18,
                         color=DEFAULT_COLOR).move_to(centre + 1.55 * UP)
            group.add(VGroup(e_true, e_short, dots, label))
        return group

    # ---- scene -----------------------------------------------------------
    def construct(self):
        with np.load(DATA_FILE) as z:
            self.data = {k: z[k] for k in z.files}

        formula = MathTex(
            r"q(d) \;=\; \Big(1 + ", r"\tfrac{1}{\alpha}", r"\,\Big(d \,/",
            r"\sigma_0", r"\Big)", r"^{m}", r"\Big)", r"^{-\alpha}",
            font_size=80).move_to(0.2 * UP)
        alpha_in, sigma_in, m_in, alpha_ex = (formula[1], formula[3],
                                              formula[5], formula[7])

        # ================= part 1: what each parameter means =============
        head = self.heading("Reading the attraction kernel",
                            "what each parameter means")
        self.play(FadeIn(head))
        self.play(Write(formula))
        self.marked_next_slide()

        def note(text, color, pos):
            return Text(text, font_size=22, color=color, line_spacing=0.8
                        ).move_to(pos)

        explained = [
            (sigma_in, [], SIGMA_COLOR,
             "units: the typical distance to a nearest\n"
             "neighbour -- it rescales the whole layout,\n"
             "and nothing else", [-4.2, -2.3, 0], UR, DOWN, TAU / 8),
            (m_in, [], M_COLOR,
             "target intrinsic dimension: how\n"
             "space-filling the layout should be --\n"
             "1: compact clusters,  2: fills the plane",
             [4.3, 2.35, 0], DL, UP, TAU / 8),
            (alpha_in, [alpha_ex], ALPHA_COLOR,
             "certainty about the local density --\n"
             "large: density known, a spring;\n"
             "small: sceptical, long edges explained away",
             [-4.05, 2.35, 0], DR, UP, -TAU / 8),
            (alpha_ex, [], TAIL_COLOR,
             "together, m and alpha set the tail:\n"
             "a long edge pulls like  m alpha / d --\n"
             "how hard a long, possibly spurious,\n"
             "edge can pull: the denoising dial",
             [4.3, -2.35, 0], UL, DOWN, -TAU / 8),
        ]
        notes = VGroup()
        for part, also, color, text, pos, from_dir, to_dir, ang in explained:
            n = note(text, color, pos)
            a = self.arrow(n.get_corner(from_dir) + 0.1 * from_dir,
                           part.get_critical_point(to_dir) + 0.08 * to_dir,
                           color, ang)
            recolour = [p.animate.set_color(color) for p in [part, *also]
                        if color != TAIL_COLOR]
            self.play(*recolour, FadeIn(n), Create(a))
            notes.add(n, a)
            self.marked_next_slide()

        k_line = Text("(counting k neighbours instead of one gives each point "
                      "personal space, at a cost in neighbour recall -- we "
                      "keep k = 1)", font_size=18, color=ACCENT_COLOR
                      ).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(k_line))
        self.marked_next_slide()

        # ================= part 2: what each parameter does ===============
        new_head = self.heading("What each parameter does",
                                "edge probability and pull against distance; "
                                "faint curve: the default")
        self.play(FadeOut(notes, k_line), FadeTransform(head, new_head),
                  formula.animate.scale(0.6).move_to(1.7 * UP))
        head = new_head

        P = {name: ValueTracker(v) for name, v in DEFAULTS.items()}
        P["alpha"] = ValueTracker(np.log(DEFAULTS["alpha"]))  # log scale

        def params():
            return (P["sigma0"].get_value(), P["m"].get_value(),
                    float(np.exp(P["alpha"].get_value())))

        dflt = (DEFAULTS["sigma0"], DEFAULTS["m"], DEFAULTS["alpha"])
        kw = dict(tips=False, x_length=5.2, y_length=2.5,
                  axis_config=dict(color=DEFAULT_COLOR, stroke_width=2))
        ax_q = Axes(x_range=[0, D_MAX, 1], y_range=[0, 1, 0.5], **kw)
        ax_p = Axes(x_range=[0, D_MAX, 1], y_range=[0, PULL_MAX, 1], **kw)
        VGroup(ax_q, ax_p).arrange(RIGHT, buff=1.5).move_to(1.95 * DOWN)
        ax_labels = VGroup(
            Text("probability of an edge", font_size=20).next_to(ax_q, UP, buff=0.1),
            Text("attraction (pull)", font_size=20).next_to(ax_p, UP, buff=0.1),
            Text("distance", font_size=18).next_to(ax_q.x_axis, DOWN, buff=0.12),
            Text("distance", font_size=18).next_to(ax_p.x_axis, DOWN, buff=0.12),
        ).set_color(DEFAULT_COLOR)
        ghost = VGroup(
            curve(ax_q, lambda d: q_kernel(d, *dflt), 1, CURVE_COLOR, 3,
                  GHOST_OPACITY),
            curve(ax_p, lambda d: pull(d, *dflt), PULL_MAX, CURVE_COLOR, 3,
                  GHOST_OPACITY))
        live = always_redraw(lambda: VGroup(
            curve(ax_q, lambda d: q_kernel(d, *params()), 1, CURVE_COLOR),
            curve(ax_p, lambda d: pull(d, *params()), PULL_MAX, CURVE_COLOR)))
        plots = VGroup(ax_q, ax_p, ax_labels, ghost, live)
        self.play(Create(ax_q), Create(ax_p), FadeIn(ax_labels))
        self.add(ghost)
        self.play(FadeIn(live))
        self.marked_next_slide()

        def tag(part, text, color, side):
            """A short name for a parameter, with a small arrow."""
            label = Text(text, font_size=24, color=color)
            label.next_to(part, side, buff=0.4)
            a = Arrow(label.get_edge_center(-side), part.get_edge_center(side),
                      buff=0.08, color=color, stroke_width=2.5,
                      max_tip_length_to_length_ratio=0.3)
            return VGroup(label, a)

        def readout(i, tex, color, fmt):
            return always_redraw(lambda: MathTex(
                tex + "=" + fmt(params()[i]), font_size=30, color=color
            ).next_to(ax_p.c2p(D_MAX, PULL_MAX), DL, buff=0.05))

        def sweep(name, there, back, log=False):
            f = np.log if log else (lambda v: v)
            for target in (there, back, DEFAULTS[name]):
                self.play(P[name].animate.set_value(f(target)),
                          run_time=SWEEP_SECONDS, rate_func=smooth)

        def evidence(panels, progress):
            """Swap the plots for layouts, play them, then swap back."""
            self.play(FadeOut(plots), FadeIn(panels))
            self.play(progress.animate.set_value(1.0),
                      run_time=EVIDENCE_SECONDS, rate_func=smooth)
            self.marked_next_slide()
            self.remove(panels)          # point clouds don't fade
            self.play(FadeIn(plots))

        tags = VGroup()
        knobs = [
            ("sigma0", sigma_in, "units", SIGMA_COLOR, DOWN, 0, r"\sigma_0"),
            ("m", m_in, "dimension", M_COLOR, UP, 1, "m"),
            ("alpha", alpha_in, "certainty", ALPHA_COLOR, DOWN, 2, r"\alpha"),
        ]
        for name, part, text, color, side, i, tex in knobs:
            t = tag(part, text, color, side)
            r = readout(i, tex, color, lambda v: f"{v:.2f}")
            self.play(FadeIn(t), FadeIn(r))
            sweep(name, *SWEEPS[name], log=(name == "alpha"))
            self.play(FadeOut(r))
            tags.add(t)
            self.marked_next_slide()
            if name == "m":
                evidence(self.mnist_panels(prog := ValueTracker(0.0)), prog)
                self.marked_next_slide()
            if name == "alpha":
                evidence(self.circle_panels(prog := ValueTracker(0.0)), prog)
                self.marked_next_slide()

        # ---- the tail: m * alpha ----
        t = tag(alpha_ex, "denoising", TAIL_COLOR, RIGHT)
        m0, a0 = DEFAULTS["m"], DEFAULTS["alpha"]
        xs = np.linspace(m0 * a0 / PULL_MAX, D_MAX, 120)
        asymptote = DashedVMobject(VMobject(
            stroke_color=TAIL_COLOR, stroke_width=3).set_points_as_corners(
            [ax_p.c2p(x, m0 * a0 / x) for x in xs]), num_dashes=30)
        asym_label = MathTex(r"\frac{m\alpha}{d}", font_size=30,
                             color=TAIL_COLOR).next_to(
            ax_p.c2p(3.6, m0 * a0 / 3.6), UP, buff=0.12)
        self.play(FadeIn(t))
        self.play(Create(asymptote), FadeIn(asym_label))
        self.marked_next_slide()
