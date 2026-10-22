"""
Where heavy-tailed attraction comes from (the knobs and their effects on real
layouts are in kernel_knobs.py).

Treat the points around i in the layout as a Poisson point process. A ball of
radius d has volume c d^m, where m is the dimension the neighbourhood fills
(its intrinsic dimension, not necessarily the screen's). With v the volume
each point occupies (at the mean density), the expected count in the ball is
c d^m / v, and j at distance d is i's nearest neighbour when the ball is empty:

    density known:     q(d) = exp(-c d^m / v)                    (a spring)
    density uncertain: lambda ~ Gamma(shape alpha, rate alpha v)  (mean 1/v)
                       q(d) = (1 + (1/alpha) (d / sigma0)^m)^(-alpha),
                       sigma0 = (v / c)^(1/m)

sigma0 is the radius of a ball holding one point's share of space. Any
density estimate comes from the same noisy graph, so treating it as known
makes long edges impossible; averaging over the uncertainty keeps them
possible but unlikely -- a heavy tail.

    manim-slides render attraction_family.py AttractionFamily
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
CURVE_COLOR = COLOR_CYCLE[2]
REFERENCE_COLOR = COLOR_CYCLE[0]       # the known-density (spring) curve
POINT_COLOR = ACCENT_COLOR
INSIDE_COLOR = HIGHLIGHT_COLOR


# ---------------------------------------------------------------------------
class AttractionFamily(TIMCSlide):

    def heading(self, title, subtitle):
        t = Text(title, color=DEFAULT_COLOR).scale(0.6).to_edge(UP, buff=0.3)
        s = Text(subtitle, color=ACCENT_COLOR).scale(0.32).next_to(
            t, DOWN, buff=0.12)
        return VGroup(t, s)

    def swap(self, old, title, subtitle, *anims):
        new = self.heading(title, subtitle)
        self.play(FadeTransform(old, new), *anims)
        return new

    def construct(self):
        # ---- 1. the geometry, and what m means ----
        head = self.heading("When is j the nearest neighbour of i?",
                            "when no other point falls inside the disc around "
                            "i that reaches j")
        rng = np.random.default_rng(3)
        pts = rng.uniform([-6.5, -3.4], [6.5, 2.4], (150, 2))
        pts = pts[np.linalg.norm(pts - [0, -0.5], axis=1) > 0.25]
        centre = np.array([0.0, -0.5, 0.0])
        r = ValueTracker(0.7)
        cloud = VGroup(*[Dot([x, y, 0], radius=0.05, color=POINT_COLOR)
                         for x, y in pts])

        def colour_inside(g):
            for dot in g:
                inside = np.linalg.norm(dot.get_center() - centre) < r.get_value()
                dot.set_color(INSIDE_COLOR if inside else POINT_COLOR)

        cloud.add_updater(colour_inside)
        disc = always_redraw(lambda: Circle(
            radius=r.get_value(), color=INSIDE_COLOR, stroke_width=2,
            fill_opacity=0.08).move_to(centre))
        i_dot = Dot(centre, radius=0.09, color=DEFAULT_COLOR)
        j_dot = always_redraw(lambda: Dot(
            centre + r.get_value() * np.array([np.cos(0.5), np.sin(0.5), 0]),
            radius=0.09, color=CURVE_COLOR))
        labels = VGroup(
            always_redraw(lambda: Text("i", font_size=26).next_to(i_dot, DL, buff=0.05)),
            always_redraw(lambda: Text("j", font_size=26, color=CURVE_COLOR
                                       ).next_to(j_dot, UR, buff=0.05)))
        count = always_redraw(lambda: Text(
            f"points inside: {sum(np.linalg.norm(p - centre[:2]) < r.get_value() for p in pts)}",
            font_size=24, color=INSIDE_COLOR).to_corner(DR, buff=0.6))
        self.play(FadeIn(head), FadeIn(cloud))
        self.play(Create(disc), FadeIn(i_dot, j_dot, labels, count))
        self.play(r.animate.set_value(1.9), run_time=3, rate_func=smooth)
        self.marked_next_slide()

        growth = VGroup(
            MathTex(r"\text{volume of the ball} \;=\; c\, d^{\,m}",
                    font_size=40),
            Text("m: the dimension the neighbourhood fills -- its intrinsic "
                 "dimension, not the screen's", font_size=20,
                 color=ACCENT_COLOR),
            Text("2 if points spread over the plane, 1 if they line up along "
                 "a curve, in between for clumps and filaments",
                 font_size=20, color=ACCENT_COLOR),
        ).arrange(DOWN, buff=0.15)
        box = BackgroundRectangle(growth, color=BACKGROUND_COLOR,
                                  fill_opacity=0.92, buff=0.2)
        VGroup(box, growth).to_edge(DOWN, buff=0.35)
        self.play(FadeOut(count), FadeIn(box), Write(growth[0]))
        self.play(FadeIn(growth[1:]))
        self.marked_next_slide()

        # ---- 2. density known: a spring ----
        head = self.swap(head, "If the local density were known",
                         "each point occupies a volume v; the chance that the "
                         "disc is empty",
                         FadeOut(cloud, disc, i_dot, j_dot, labels, box,
                                 growth))
        count_eq = MathTex(r"\text{expected count} \;=\; "
                           r"\frac{c\, d^{\,m}}{v}"
                           r"\;=\;\frac{\text{volume of the ball}}"
                           r"{\text{volume per point}}",
                           font_size=38).move_to(1.4 * UP)
        known = MathTex(r"q(d) \;=\; e^{-c\, d^{m} / v}",
                        font_size=56, color=REFERENCE_COLOR).move_to(0.1 * DOWN)
        note = Text("a long edge is very unlikely, so it pulls very hard:\n"
                    "a spring -- the pull grows with distance",
                    font_size=24, color=REFERENCE_COLOR, line_spacing=0.8
                    ).next_to(known, DOWN, buff=0.5)
        self.play(Write(count_eq))
        self.play(Write(known))
        self.play(FadeIn(note))
        self.marked_next_slide()

        # ---- 3. density uncertain: a heavy tail ----
        head = self.swap(head, "But the density is uncertain",
                         "any estimate comes from the same noisy graph -- "
                         "treating it as known makes long edges impossible",
                         FadeOut(count_eq, note),
                         known.animate.scale(0.7).move_to([-4.6, 0.9, 0]))
        unknown = MathTex(r"q(d) \;=\; \Big(1 + \tfrac{1}{\alpha}"
                          r"\Big(\frac{d}{\sigma_0}\Big)^{m}\Big)^{-\alpha}",
                          font_size=52, color=CURVE_COLOR).move_to([3.5, 0.9, 0])
        arrow = Arrow([-2.7, 0.9, 0], unknown.get_left() + 0.45 * LEFT,
                      color=DEFAULT_COLOR, buff=0.1)
        prior = MathTex(r"\lambda \sim \mathrm{Gamma}(\alpha,\ \alpha v)",
                        font_size=30).next_to(arrow, UP, buff=0.12)
        prior_note = Text("same mean density, 1/v", font_size=18,
                          color=ACCENT_COLOR).next_to(arrow, DOWN, buff=0.12)
        sigma = MathTex(r"\sigma_0 \;=\; \sqrt[m]{v / c}", font_size=40
                        ).next_to(unknown, DOWN, buff=0.55)
        sigma_note = Text("the radius of a ball holding one point's share of "
                          "space", font_size=20, color=ACCENT_COLOR
                          ).next_to(sigma, DOWN, buff=0.15)
        note = Text("long edges stay possible, just unlikely: a heavy tail --\n"
                    "the pull fades with distance", font_size=24,
                    color=CURVE_COLOR, line_spacing=0.8).move_to([0, -2.6, 0])
        self.play(GrowArrow(arrow), FadeIn(prior, prior_note))
        self.play(Write(unknown))
        self.play(Write(sigma), FadeIn(sigma_note))
        self.play(FadeIn(note))
        self.marked_next_slide()

        # ---- 4. the family ----
        head = self.swap(head, "One family of attraction kernels",
                         "familiar kernels are points in it",
                         FadeOut(known, arrow, prior, prior_note, unknown,
                                 sigma, sigma_note, note))
        rows = [
            (r"\alpha \to \infty", r"e^{-(d/\sigma_0)^{m}}",
             "density known: a spring"),
            (r"\alpha = 1,\ m = 2", r"\dfrac{1}{1 + (d/\sigma_0)^{2}}",
             "Cauchy: the t-SNE kernel"),
            (r"m = 2,\ \alpha = \tfrac{\nu+1}{2}",
             r"\left(1 + \tfrac{d^{2}}{\nu}\right)^{-\frac{\nu+1}{2}}",
             "Student-t with nu degrees of freedom"),
            (r"\alpha = 1,\ m = 2b", r"\dfrac{1}{1 + (d/\sigma_0)^{2b}}",
             "the form of UMAP's curve"),
        ]
        table = VGroup()
        for params, form, name in rows:
            table.add(VGroup(MathTex(params, font_size=32),
                             MathTex(form, font_size=34, color=CURVE_COLOR),
                             Text(name, font_size=22, color=ACCENT_COLOR)))
        table.arrange(DOWN, buff=0.45).move_to(0.4 * DOWN)
        for row in table:
            row[0].set_x(-3.9)
            row[1].set_x(-0.4)
            row[2].align_to(1.9 * RIGHT, LEFT)
        for row in table:
            self.play(FadeIn(row, shift=0.1 * UP), run_time=0.7)
        self.marked_next_slide()
