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
    cyclic_color,
    make_heading,
    SmallText,
    install_round_points,
    res_px,
    PMFadeIn,
    PMFadeOut,
)

import os
from pathlib import Path

import numpy as np

apply_defaults()
install_round_points()


def split_point_clouds(group):
    """(point clouds, everything else at leaf level) in a panel group."""
    clouds = [m for m in group.get_family() if isinstance(m, PMobject)]
    others = [m for m in group.get_family()
              if not isinstance(m, PMobject) and not m.submobjects]
    return clouds, others


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
MNIST_POINT_SIZE = 2.13            # PMobject point size, pixels at 1440p
D_TRUE, D_SPURIOUS = 0.8, 3.4      # a true neighbour and a spurious long edge
TAIL_SWEEP = (0.33, 4.0)           # alpha values: tail m*alpha ~ 0.5 ... 6
PULL_ARROW_SCALE = 1.25            # pull-meter bar length per unit of pull


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
        return make_heading(title, subtitle)

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

            pm = PMobject(stroke_width=res_px(MNIST_POINT_SIZE))
            pm.add_points(pos(), rgbas=rgba)
            pm.add_updater(lambda m_, pos=pos: m_.set(points=pos()))
            cap = (f"m = {v:g}:  measured dim "
                   f"{float(d[f'mnist_m_{v:g}_dimension']):.2f}, recall "
                   f"{float(d[f'mnist_m_{v:g}_recall']):.2f}")
            label = SmallText(cap, font_size=16, color=DEFAULT_COLOR).move_to(
                centre + 1.6 * UP)
            group.add(Group(pm, label))
        return group

    def ring_panels(self, progress, theta, kept, short, snap_list, captions,
                    spacing):
        """Rewired-circle layouts animating from their shared start as
        `progress` goes 0 -> 1; shortcuts drawn in the tail colour."""
        group = VGroup()
        n = len(snap_list)
        for j, (snaps, cap) in enumerate(zip(snap_list, captions)):
            centre = np.array([(j - (n - 1) / 2) * spacing, -2.2, 0])

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
                            for t in theta])

            def move(g, pos=pos):
                for dd, p in zip(g, pos()):
                    dd.move_to(p)

            move(dots)
            dots.add_updater(move)
            label = SmallText(cap, font_size=18, color=DEFAULT_COLOR
                              ).next_to(centre + 1.3 * UP, UP, buff=0)
            group.add(VGroup(e_true, e_short, dots, label))
        return group

    def circle_panels(self, progress):
        """The certainty dial: the rewired circle at several alpha (m = 2)."""
        d = self.data
        alphas = d["circle_alphas"]
        caps = [f"alpha = {al:g}" + ("  (spring-like)" if al >= 4 else
                                     "  (sceptical)" if al < 1 else "")
                for al in alphas]
        return self.ring_panels(
            progress, d["circle_theta"], d["circle_kept"], d["circle_short"],
            [d[f"circle_alpha_{al:g}"] for al in alphas], caps, 3.5)

    def denoise_panels(self, progress):
        """The denoising dial: a more heavily rewired circle at three tail
        values m * alpha."""
        d = self.data
        tails = d["denoise_tails"]
        caps = {0.5: "tail 0.5\ntoo sceptical: it breaks into clumps",
                1.5: "tail 1.5\nthe circle",
                6.0: "tail 6\ntoo trusting: the shortcuts drag it"}
        return self.ring_panels(
            progress, d["denoise_theta"], d["denoise_kept"],
            d["denoise_short"], [d[f"denoise_tail_{t:g}"] for t in tails],
            [caps.get(float(t), f"tail {t:g}") for t in tails], 4.6)

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
            return SmallText(text, font_size=22, color=color, line_spacing=0.8
                             ).move_to(pos)

        # Each arrow leaves the middle of one edge of its note (from_dir) and
        # arrives at one side of its parameter (to_dir); the bend (angle,
        # positive = anticlockwise) is chosen from where things sit:
        #   sigma0: note bottom-left, sigma0 right of centre -> leave the
        #           note's right edge, sweep up into sigma0 from below
        #   m:      note directly above m -> drop from its bottom edge
        #   alpha:  note top-left, alpha at the centre -> leave the right
        #           edge heading right, turn down onto alpha from above
        #   tail:   note directly below the exponent -> rise from its top
        explained = [
            (sigma_in, [], SIGMA_COLOR,
             "units: the typical distance to a nearest\n"
             "neighbour — it rescales the whole layout,\n"
             "and nothing else", [-4.2, -2.3, 0], RIGHT, DOWN, TAU / 4),
            (m_in, [], M_COLOR,
             "target intrinsic dimension: how\n"
             "space-filling the layout should be —\n"
             "1: compact clusters,  2: fills the plane",
             [4.3, 2.35, 0], DOWN, UP, TAU / 14),
            (alpha_in, [alpha_ex], ALPHA_COLOR,
             "certainty about the local density —\n"
             "large: density known, a spring;\n"
             "small: sceptical, long edges explained away",
             [-4.05, 2.35, 0], RIGHT, UP, -TAU / 4),
            (alpha_ex, [], TAIL_COLOR,
             "together, m and alpha set the tail:\n"
             "a long edge pulls like  m alpha / d —\n"
             "how hard a long, possibly spurious,\n"
             "edge can pull: the denoising dial",
             [4.3, -2.35, 0], UP, DOWN, -TAU / 14),
        ]
        notes = VGroup()
        for part, also, color, text, pos, from_dir, to_dir, ang in explained:
            n = note(text, color, pos)
            a = self.arrow(n.get_critical_point(from_dir) + 0.1 * from_dir,
                           part.get_critical_point(to_dir) + 0.08 * to_dir,
                           color, ang)
            recolour = [p.animate.set_color(color) for p in [part, *also]
                        if color != TAIL_COLOR]
            self.play(*recolour, FadeIn(n), Create(a))
            notes.add(n, a)
            self.marked_next_slide()

        k_line = SmallText("(counting k neighbours instead of one gives each "
                           "point personal space, at a cost in neighbour "
                           "recall — we keep k = 1)", font_size=18,
                           color=ACCENT_COLOR).to_edge(DOWN, buff=0.3)
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
            SmallText("probability of an edge", font_size=20).next_to(ax_q, UP, buff=0.1),
            SmallText("attraction (pull)", font_size=20).next_to(ax_p, UP, buff=0.1),
            SmallText("distance", font_size=18).next_to(ax_q.x_axis, DOWN, buff=0.12),
            SmallText("distance", font_size=18).next_to(ax_p.x_axis, DOWN, buff=0.12),
        ).set_color(DEFAULT_COLOR)
        ghost = VGroup(
            curve(ax_q, lambda d: q_kernel(d, *dflt), 1, CURVE_COLOR, 3,
                  GHOST_OPACITY),
            curve(ax_p, lambda d: pull(d, *dflt), PULL_MAX, CURVE_COLOR, 3,
                  GHOST_OPACITY))
        live_q = always_redraw(lambda: curve(
            ax_q, lambda d: q_kernel(d, *params()), 1, CURVE_COLOR))
        live_p = always_redraw(lambda: curve(
            ax_p, lambda d: pull(d, *params()), PULL_MAX, CURVE_COLOR))
        live = VGroup(live_q, live_p)
        plots = VGroup(ax_q, ax_p, ax_labels, ghost, live)
        self.play(Create(ax_q), Create(ax_p), FadeIn(ax_labels))
        self.add(ghost)
        self.play(FadeIn(live))
        self.marked_next_slide()

        def tag(part, text, color, side):
            """A short name for a parameter, with a small arrow."""
            label = SmallText(text, font_size=24, color=color)
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
            clouds, others = split_point_clouds(panels)
            self.play(FadeOut(plots), run_time=0.6)
            fade_ins = [PMFadeIn(c) for c in clouds]    # (sets them to bg)
            self.add(*clouds)
            self.play(*fade_ins, *[FadeIn(o) for o in others], run_time=0.8)
            # register the panels themselves: some updaters (e.g. the one
            # moving a panel's dots) live on groups, not on their leaves
            self.add(panels)
            self.play(progress.animate.set_value(1.0),
                      run_time=EVIDENCE_SECONDS, rate_func=smooth)
            self.marked_next_slide()
            self.play(*[PMFadeOut(c) for c in clouds],
                      *[FadeOut(o) for o in others], run_time=0.8)
            self.remove(panels)
            self.play(FadeIn(plots), run_time=0.6)

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
        def tail_now():
            sg, mm, al = params()
            return mm * al

        t = tag(alpha_ex, "denoising", TAIL_COLOR, RIGHT)
        asymptote = always_redraw(lambda: DashedVMobject(VMobject(
            stroke_color=TAIL_COLOR, stroke_width=3).set_points_as_corners(
            [ax_p.c2p(x, tail_now() / x) for x in np.linspace(
                min(tail_now() / PULL_MAX, D_MAX - 0.1), D_MAX, 120)]),
            num_dashes=30))
        asym_label = MathTex(r"\frac{m\alpha}{d}", font_size=30,
                             color=TAIL_COLOR).next_to(
            ax_p.c2p(3.6, tail_now() / 3.6), UP, buff=0.12)
        self.play(FadeIn(t))
        self.play(Create(asymptote), FadeIn(asym_label))
        self.marked_next_slide()

        # ---- what the tail means: a true edge against a spurious one ----
        new_head = self.heading(
            "How hard can a long edge pull?",
            "a true neighbour sits close by; corruption adds long, "
            "spurious edges")
        # the picture replaces the edge-probability plot
        live_q.clear_updaters()
        self.play(FadeOut(ax_q, ax_labels[0], ax_labels[2], ghost[0], live_q,
                          asym_label),
                  FadeTransform(head, new_head))
        head = new_head

        left = ax_q.get_center()
        unit = 1.05                       # scene units per unit of distance
        x0 = left[0] - unit * D_SPURIOUS / 2
        tug_key = SmallText("bars: how hard each edge pulls its ends "
                            "together", font_size=18, color=ACCENT_COLOR
                            ).move_to([left[0], left[1] + 1.55, 0])
        rows = []
        for k, (dist, name, col) in enumerate(
                ((D_TRUE, "a true neighbour", DEFAULT_COLOR),
                 (D_SPURIOUS, "a spurious long edge", TAIL_COLOR))):
            y = left[1] + 0.75 - 1.45 * k
            a_pt, b_pt = np.array([x0, y, 0]), np.array([x0 + unit * dist, y, 0])
            edge = Line(a_pt, b_pt, color=col, stroke_width=3)
            ends = VGroup(Dot(a_pt, radius=0.07, color=DEFAULT_COLOR),
                          Dot(b_pt, radius=0.07, color=DEFAULT_COLOR))
            lab = SmallText(name, font_size=20, color=col).next_to(
                a_pt, UP, buff=0.18, aligned_edge=LEFT)
            # a pull meter: all bars start at the same x, so they compare
            meter = always_redraw(lambda y=y, dist=dist, col=col: Rectangle(
                width=max(PULL_ARROW_SCALE * pull(dist, *params()), 0.02),
                height=0.16, stroke_width=0, fill_color=col,
                fill_opacity=0.85).move_to([x0, y - 0.32, 0], aligned_edge=LEFT))
            rows.append(VGroup(edge, ends, lab, meter))

        # matching markers on the pull curve
        def marker(dist, col):
            return always_redraw(lambda: VGroup(
                DashedLine(ax_p.c2p(dist, 0),
                           ax_p.c2p(dist, min(pull(dist, *params()),
                                              PULL_MAX)),
                           color=col, stroke_width=2, dash_length=0.06),
                Dot(ax_p.c2p(dist, min(pull(dist, *params()), PULL_MAX)),
                    radius=0.08, color=col)))

        markers = VGroup(marker(D_TRUE, DEFAULT_COLOR),
                         marker(D_SPURIOUS, TAIL_COLOR))
        self.play(*[FadeIn(r) for r in rows], FadeIn(tug_key),
                  FadeIn(markers))
        self.marked_next_slide()

        # ---- turn the dial ----
        new_head = self.heading(
            "The denoising dial",
            "small: a long edge barely pulls, so the rest of the graph "
            "keeps its ends apart; large: the layout follows it")
        r = readout(2, r"m\alpha", TAIL_COLOR,
                    lambda v: f"{DEFAULTS['m'] * v:.2f}")
        self.play(FadeTransform(head, new_head), FadeIn(r))
        head = new_head
        verdict = always_redraw(lambda: SmallText(
            "rejected: it barely pulls" if
            pull(D_SPURIOUS, *params()) < 0.45 * pull(D_TRUE, *params()) else
            "followed: it pulls nearly as hard" if
            pull(D_SPURIOUS, *params()) > 0.8 * pull(D_TRUE, *params()) else
            " ", font_size=22, color=TAIL_COLOR).move_to(
            [left[0], rows[1][0].get_y() - 0.85, 0]))
        self.add(verdict)
        for target in (*TAIL_SWEEP, DEFAULTS["alpha"]):
            self.play(P["alpha"].animate.set_value(np.log(target)),
                      run_time=SWEEP_SECONDS, rate_func=smooth)
            self.wait(0.6)
        self.marked_next_slide()

        # ---- the dial on a rewired circle: three tail values ----
        new_head = self.heading(
            "The dial on a rewired circle",
            "25 random shortcuts: too sceptical loses the circle's own "
            "structure, too trusting follows the shortcuts")
        self.play(FadeOut(*rows, tug_key, markers, verdict, r, ax_p,
                          ax_labels[1], ax_labels[3], ghost[1], live_p,
                          asymptote),
                  FadeTransform(head, new_head), run_time=0.8)
        head = new_head
        prog = ValueTracker(0.0)
        panels = self.denoise_panels(prog)
        self.play(FadeIn(panels), run_time=0.8)
        self.add(panels)
        self.play(prog.animate.set_value(1.0), run_time=EVIDENCE_SECONDS,
                  rate_func=smooth)
        self.marked_next_slide(auto_next=True)
        self.clear_slide()
