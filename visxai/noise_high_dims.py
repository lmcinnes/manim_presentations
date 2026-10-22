"""
How isotropic Gaussian noise behaves in high dimensions (lead-in to the
circle / cylinder intuition slides).

Model: x_i = mu_i + sigma z_i, z_i ~ N(0, I_D); delta = ||mu_i - mu_j||,
d = ||x_i - x_j||. All facts below were checked by simulation.

  1. each point moves ||sigma z|| ~ sigma sqrt(D)   (spread ~ sigma / sqrt 2)
  2. E[d^2] = delta^2 + 2 sigma^2 D,  Var[d^2] = 8 sigma^2 delta^2 + 8 sigma^4 D
  3. for sigma^2 D >> delta^2:
         d ~ sigma sqrt(2D) + delta^2 / (sigma sqrt(8D)),  SD(d) ~ sigma
     the offset grows, the jitter stays sigma, the structure shrinks ~ 1/sqrt(D)
  4. ranking two candidates j, k for the same i: the noise in
     d_ij^2 - d_ik^2 has SD ~ sigma^2 sqrt(12 D), so true neighbours within
     ~ sigma (12 D)^(1/4) cannot be told apart

    manim-slides render noise_high_dims.py NoiseInHighDimensions
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

OFFSET_COLOR = COLOR_CYCLE[0]
STRUCTURE_COLOR = COLOR_CYCLE[2]
JITTER_COLOR = HIGHLIGHT_COLOR
PLOT_SIGMA, PLOT_DELTA = 0.1, 0.5      # the example pair in the plot


class NoiseInHighDimensions(TIMCSlide):

    def heading(self, title, subtitle):
        t = Text(title, color=DEFAULT_COLOR).scale(0.6).to_edge(UP, buff=0.3)
        # \mbox stops TeX from wrapping the line at its own text width
        s = Tex(r"\mbox{" + subtitle + "}", color=ACCENT_COLOR,
                font_size=19).next_to(t, DOWN, buff=0.12)
        return VGroup(t, s)

    def swap(self, old, title, subtitle, *anims):
        new = self.heading(title, subtitle)
        self.play(FadeTransform(old, new), *anims)
        return new

    def note(self, text, color=ACCENT_COLOR, size=24):
        """A note in the slide font; inline maths in $...$, line breaks as
        newlines."""
        lines = [r"\mbox{" + line + "}" for line in text.split("\n")]
        return Tex(r"\\ ".join(lines), font_size=size * 1.15, color=color)

    def construct(self):
        # ---- 1. the model ----
        head = self.heading("Noise in high dimensions",
                            "a little Gaussian noise in every one of $D$ "
                            "directions")
        model = MathTex(r"x_i \;=\; \mu_i + \sigma z_i, \qquad "
                        r"z_i \sim \mathcal{N}(0,\ I_D)", font_size=52
                        ).move_to(0.6 * UP)
        legend = VGroup(
            MathTex(r"\delta = \lVert \mu_i - \mu_j \rVert", font_size=36),
            self.note("the true distance", size=22),
            MathTex(r"d = \lVert x_i - x_j \rVert", font_size=36),
            self.note("the distance we observe", size=22),
        )
        legend[1].next_to(legend[0], RIGHT, buff=0.3)
        legend[2].next_to(legend[0], DOWN, buff=0.35, aligned_edge=LEFT)
        legend[3].next_to(legend[2], RIGHT, buff=0.3)
        legend.move_to(1.3 * DOWN)
        self.play(FadeIn(head))
        self.play(Write(model))
        self.play(FadeIn(legend))
        self.marked_next_slide()

        # ---- 2. how far each point moves ----
        head = self.swap(head, "How far noise moves each point",
                         "almost exactly the same distance for every point",
                         FadeOut(model, legend))
        move = MathTex(r"\mathbb{E}\,\lVert \sigma z_i \rVert^{2} \;=\; "
                       r"\sigma^{2} D \qquad \Longrightarrow \qquad "
                       r"\lVert \sigma z_i \rVert \;\approx\; \sigma\sqrt{D}",
                       font_size=48).move_to(0.4 * UP)
        move_note = self.note("small per direction, but there are $D$ "
                              "directions: the total grows like $\\sqrt{D}$").next_to(move, DOWN, buff=0.6)
        self.play(Write(move))
        self.play(FadeIn(move_note))
        self.marked_next_slide()

        # ---- 3. the squared distance: offset and fluctuation ----
        head = self.swap(head, "Distances between noisy points",
                         "every pair gets the same offset -- plus a "
                         "fluctuation", FadeOut(move, move_note))
        expand = MathTex(r"d^{2} \;=\; \delta^{2} \;+\; "
                         r"2\sigma\langle \mu_i - \mu_j,\ z_i - z_j \rangle"
                         r" \;+\; \sigma^{2}\lVert z_i - z_j \rVert^{2}",
                         font_size=42).move_to(1.3 * UP)
        mean = MathTex(r"\mathbb{E}[d^{2}] \;=\; \delta^{2} + ",
                       r"2\sigma^{2} D", font_size=44)
        var = MathTex(r"\mathrm{Var}[d^{2}] \;=\; 8\sigma^{2}\delta^{2} + ",
                      r"8\sigma^{4} D", font_size=44)
        mean[1].set_color(OFFSET_COLOR)
        var[1].set_color(JITTER_COLOR)
        VGroup(mean, var).arrange(DOWN, buff=0.45).next_to(expand, DOWN,
                                                           buff=0.6)
        notes = VGroup(
            self.note("the same for every pair: an offset that grows with $D$",
                      OFFSET_COLOR, 20).next_to(mean, RIGHT, buff=0.4),
            self.note("noise that also grows with $D$", JITTER_COLOR, 20
                      ).next_to(var, RIGHT, buff=0.4))
        group = VGroup(mean, var, notes)
        group.set_x(0)
        self.play(Write(expand))
        self.play(Write(mean), FadeIn(notes[0]))
        self.play(Write(var), FadeIn(notes[1]))
        self.marked_next_slide()

        # ---- 4. in distance units: offset, structure, jitter ----
        head = self.swap(head, "What is left of the true structure",
                         "in distance units, once the noise dominates "
                         "($\\sigma^2 D \\gg \\delta^2$)",
                         FadeOut(expand, group))
        approx = MathTex(r"d \;\approx\;", r"\sigma\sqrt{2D}", r"\;+\;",
                         r"\frac{\delta^{2}}{\sigma\sqrt{8D}}", r"\;\pm\;",
                         r"\sigma", font_size=50).move_to([-3.3, 1.2, 0])
        approx[1].set_color(OFFSET_COLOR)
        approx[3].set_color(STRUCTURE_COLOR)
        approx[5].set_color(JITTER_COLOR)
        parts = VGroup(
            self.note("offset: grows like $\\sqrt{D}$", OFFSET_COLOR, 21),
            self.note("true structure: shrinks like $1/\\sqrt{D}$",
                      STRUCTURE_COLOR, 21),
            self.note("jitter: stays at $\\sigma$", JITTER_COLOR, 21),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).next_to(
            approx, DOWN, buff=0.55).align_to(approx, LEFT)

        s, dl = PLOT_SIGMA, PLOT_DELTA
        Y0 = -2                                     # log10 of the bottom edge
        ax = Axes(x_range=[0, 4, 1], y_range=[0, 3, 1], x_length=5.2,
                  y_length=4.0, tips=False,
                  axis_config=dict(color=DEFAULT_COLOR, stroke_width=2)
                  ).move_to([3.6, -0.4, 0])
        x_lab = VGroup(*[MathTex(f"10^{{{k}}}", font_size=22).next_to(
            ax.c2p(k, 0), DOWN, buff=0.12) for k in range(0, 5)])
        y_lab = VGroup(*[MathTex(f"10^{{{k}}}", font_size=22).next_to(
            ax.c2p(0, k - Y0), LEFT, buff=0.12) for k in range(-2, 2)])
        ax_names = VGroup(
            Tex("dimension $D$", font_size=23).next_to(x_lab, DOWN, buff=0.12),
            Tex(f"size  ($\\sigma = {s:g}$, $\\delta = {dl:g}$)", font_size=21,
                color=ACCENT_COLOR).next_to(ax, UP, buff=0.1))

        def logcurve(f, color):
            Ds = np.geomspace(1, 1e4, 200)
            pts = [ax.c2p(np.log10(D), np.log10(f(D)) - Y0) for D in Ds]
            return VMobject(stroke_color=color, stroke_width=4
                            ).set_points_smoothly(pts)

        # full (not asymptotic) forms, so the curves are right at small D too
        offset = lambda D: s * np.sqrt(2 * D)
        structure = lambda D: np.sqrt(dl ** 2 + 2 * s ** 2 * D) - offset(D)
        jitter = lambda D: (np.sqrt(8 * s ** 2 * dl ** 2 + 8 * s ** 4 * D)
                            / (2 * np.sqrt(dl ** 2 + 2 * s ** 2 * D)))
        curves = VGroup(logcurve(offset, OFFSET_COLOR),
                        logcurve(structure, STRUCTURE_COLOR),
                        logcurve(jitter, JITTER_COLOR))
        d_star = dl ** 4 / (8 * s ** 4)
        cross = DashedLine(ax.c2p(np.log10(d_star), 0),
                           ax.c2p(np.log10(d_star), 3), color=ACCENT_COLOR,
                           stroke_width=2)
        cross_lab = Text("structure drops\nbelow the jitter", font_size=17,
                         color=ACCENT_COLOR, line_spacing=0.8).next_to(
            ax.c2p(np.log10(d_star), 0.35), RIGHT, buff=0.12)
        self.play(Write(approx))
        self.play(FadeIn(parts))
        self.play(Create(ax), FadeIn(x_lab, y_lab, ax_names))
        self.play(LaggedStart(*[Create(c) for c in curves], lag_ratio=0.4),
                  run_time=2.5)
        self.play(Create(cross), FadeIn(cross_lab))
        self.marked_next_slide()

        # ---- 5. the consequence, and the lead-in ----
        head = self.swap(head, "Near neighbours become indistinguishable",
                         "ranking two candidates for the same point: the "
                         "noise in the comparison grows with $D$",
                         FadeOut(approx, parts, ax, x_lab, y_lab, ax_names,
                                 curves, cross, cross_lab))
        rank = MathTex(r"\mathrm{SD}\big[d_{ij}^{2} - d_{ik}^{2}\big] \;\approx\;"
                       r"\sigma^{2}\sqrt{12D}", font_size=46).move_to(1.3 * UP)
        window = self.note(r"neighbours within $W \approx \sigma\,(12D)^{1/4}$"
                           " look the same", JITTER_COLOR, 34).next_to(
            rank, DOWN, buff=0.6)
        lead = self.note("the noise has total size $\\sigma\\sqrt{D}$, spread "
                         "thinly over $D$ directions --\n"
                         "next: pack all of it into one direction, and watch "
                         "what it does to a neighbourhood graph",
                         DEFAULT_COLOR, 22).next_to(window, DOWN, buff=0.8)
        self.play(Write(rank))
        self.play(Write(window))
        self.play(FadeIn(lead))
        self.marked_next_slide()
