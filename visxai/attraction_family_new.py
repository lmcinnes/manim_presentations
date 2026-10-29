"""
Where heavy-tailed attraction comes from.

The question: how likely should two points at distance d in a layout be to
share an edge? Derivation (points around i treated as scattered at random
with density lambda, i.e. a Poisson process):

  * j is i's nearest neighbour exactly when the ball of radius d around i
    holds no other point
  * that ball expects lambda * c d^m points (c d^m: its volume; m: the
    dimension the neighbourhood fills)
  * the chance it is empty is exp(-expected count):
        q(d | lambda) = exp(-lambda c d^m)                    (a spring)
  * write the mean density as 1/v (v: volume per point) and let
    sigma0 = (v/c)^(1/m), the radius of a ball of volume v; at exactly the
    mean density, q = exp(-(d/sigma0)^m)
  * the density is unknown, so average over a prior with that mean:
        lambda ~ Gamma(shape alpha, rate alpha v)     (relative spread 1/sqrt(alpha))
        q(d) = integral exp(-lambda c d^m) Gamma(lambda; alpha, alpha v) dlambda
             = (1 + (1/alpha) (d/sigma0)^m)^(-alpha)          (a heavy tail)

Plots use the plane (m = 2) with distance in units of sigma0.

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
    make_heading,
    SmallText,
    SmallTex,
)

import numpy as np

apply_defaults()

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
KNOWN_COLOR = COLOR_CYCLE[0]  # a known density: a spring
DENSE_COLOR = COLOR_CYCLE[3]
SPARSE_COLOR = COLOR_CYCLE[5]
TAIL_COLOR = COLOR_CYCLE[2]  # averaged over the prior: a heavy tail
POINT_COLOR = ACCENT_COLOR
INSIDE_COLOR = HIGHLIGHT_COLOR
LEFT_CENTRE = np.array([-3.5, -0.5, 0])
AXES_CENTRE = np.array([3.55, -0.55, 0])
X_MAX = 4.0  # distance axis, in units of sigma0
DENSE, SPARSE = 3.0, 1 / 3  # densities relative to the mean
N_FAN = 15  # curves drawn from the prior
RESULT_SIZE = 44  # the family formula on the averaging slide
TABLE_HEAD_SIZE = 52  # ... and at the head of the final table


def txt(text, size, color=DEFAULT_COLOR):
    return SmallText(text, font_size=size, color=color)


def tex(text, size, color=DEFAULT_COLOR):
    """size is a Tex font size (see config.TEX_TO_TEXT)."""
    return SmallTex(text, font_size=size, color=color)


# ---------------------------------------------------------------------------
class AttractionFamily(TIMCSlide):

    def heading(self, title, subtitle):
        return make_heading(title, subtitle, title_tex=True, subtitle_tex=True)

    def swap(self, old, title, subtitle, *anims):
        new = self.heading(title, subtitle)
        self.play(FadeTransform(old, new), *anims)
        return new

    def plot(self, f, color, width=4, opacity=1.0):
        return self.axes.plot(
            f,
            x_range=[0, X_MAX, 0.02],
            color=color,
            stroke_width=width,
            stroke_opacity=opacity,
        )

    def scatter(self, centre, half, density_per_unit, seed, exclude=0.2):
        """Random points in a box; returns (positions, dots)."""
        rng = np.random.default_rng(seed)
        n = rng.poisson(density_per_unit * (2 * half[0]) * (2 * half[1]))
        P = centre[:2] + rng.uniform(-1, 1, (n, 2)) * half
        P = P[np.linalg.norm(P - centre[:2], axis=1) > exclude]
        dots = VGroup(*[Dot([x, y, 0], radius=0.045, color=POINT_COLOR) for x, y in P])
        return P, dots

    def construct(self):
        # ================= 1. the question ================================
        head = self.heading(
            "How likely is an edge?",
            "two points a distance $d$ apart in the layout: "
            "how likely should they be to share an edge?",
        )
        ax = Axes(
            x_range=[0, X_MAX, 1],
            y_range=[0, 1, 0.5],
            x_length=5.6,
            y_length=3.8,
            tips=False,
            axis_config=dict(color=DEFAULT_COLOR, stroke_width=2),
        ).move_to(AXES_CENTRE)
        self.axes = ax
        ax_lab = VGroup(
            tex("distance $d$", 22).next_to(ax.x_axis, DOWN, buff=0.15),
            tex("probability of an edge", 22)
            .rotate(PI / 2)
            .next_to(ax.y_axis, LEFT, buff=0.2),
        )
        qmark = txt("?", 120, ACCENT_COLOR).move_to(ax.c2p(2, 0.55))
        question = (
            VGroup(
                tex("t-SNE uses $1/(1+d^2)$", 28),
                tex("UMAP fits a curve to a setting", 28),
                tex(
                    "each simply picks a curve —\ncan we derive one?", 28, ACCENT_COLOR
                ),
            )
            .arrange(DOWN, buff=0.45, aligned_edge=LEFT)
            .move_to(LEFT_CENTRE)
        )
        self.play(FadeIn(head))
        self.play(Create(ax), FadeIn(ax_lab), FadeIn(qmark))
        self.play(
            LaggedStart(*[FadeIn(q, shift=0.1 * UP) for q in question], lag_ratio=0.4)
        )
        self.marked_next_slide()

        # ================= 2. the mechanism ===============================
        head = self.swap(
            head,
            "When is $j$ the nearest neighbour of $i$?",
            "exactly when no other point is closer: when the "
            "disc around $i$ that reaches $j$ is empty",
            FadeOut(question),
        )
        centre = LEFT_CENTRE + np.array([0, 0.1, 0])
        P, cloud = self.scatter(
            centre, np.array([3.0, 2.3]), 4.0, seed=3, exclude=0.5
        )  # j starts with an empty disc
        r = ValueTracker(0.4)

        def colour_inside(g):
            for dot, p in zip(g, P):
                inside = np.linalg.norm(p - centre[:2]) < r.get_value()
                dot.set_color(INSIDE_COLOR if inside else POINT_COLOR)

        cloud.add_updater(colour_inside)
        direction = np.array([np.cos(0.6), np.sin(0.6), 0])
        disc = always_redraw(
            lambda: Circle(
                radius=r.get_value(),
                color=INSIDE_COLOR,
                stroke_width=2.5,
                fill_opacity=0.10,
            ).move_to(centre)
        )
        i_dot = Dot(centre, radius=0.09, color=DEFAULT_COLOR)
        j_dot = always_redraw(
            lambda: Dot(
                centre + r.get_value() * direction, radius=0.09, color=TAIL_COLOR
            )
        )
        labels = VGroup(
            tex("$i$", 28).next_to(i_dot, DL, buff=0.04),
            always_redraw(
                lambda: tex("$j$", 28, TAIL_COLOR).next_to(j_dot, UR, buff=0.04)
            ),
        )

        def empty():
            return not np.any(np.linalg.norm(P - centre[:2], axis=1) < r.get_value())

        verdict = always_redraw(
            lambda: tex(
                "disc empty: an edge" if empty() else "someone closer: no edge",
                24,
                TAIL_COLOR if empty() else INSIDE_COLOR,
            ).move_to(LEFT_CENTRE + 2.6 * DOWN)
        )
        self.play(FadeOut(qmark), FadeIn(cloud))
        self.play(
            Create(disc), FadeIn(i_dot, j_dot, labels[0]), FadeIn(labels[1], verdict)
        )
        self.marked_next_slide()
        self.play(r.animate.set_value(1.6), run_time=3, rate_func=smooth)
        self.marked_next_slide()

        # ================= 3. density makes it a number ===================
        head = self.swap(
            head,
            "Density makes it a number",
            "points scattered at random, $\\lambda$ of them per " "unit of volume",
        )
        count = MathTex(
            r"\text{expected count} \;=\; \lambda \cdot c\,d^{\,m}", font_size=40
        ).move_to(LEFT_CENTRE + 2.0 * DOWN)
        count_bg = BackgroundRectangle(
            count, color=BACKGROUND_COLOR, fill_opacity=0.92, buff=0.15
        )
        count_note = tex(
            "$c\\,d^{\\,m}$: the disc's volume;  $m$: the "
            "dimension the\nneighbourhood fills (2 for a plane, "
            "1 for a curve)",
            20,
            ACCENT_COLOR,
        )
        count_note.next_to(count, DOWN, buff=0.15)
        note_bg = BackgroundRectangle(
            count_note, color=BACKGROUND_COLOR, fill_opacity=0.92, buff=0.1
        )
        self.play(FadeOut(verdict), FadeIn(count_bg), Write(count))
        self.play(FadeIn(note_bg), FadeIn(count_note))
        self.marked_next_slide()

        # ================= 4. the chance the disc is empty ================
        head = self.swap(
            head,
            "The chance the disc is empty",
            "split it into many tiny patches, each almost surely "
            "empty: multiplying them gives an exponential",
            FadeOut(
                cloud, disc, i_dot, j_dot, labels, count_bg, count, note_bg, count_note
            ),
        )
        known = MathTex(
            r"q(d \mid \lambda) \;=\; e^{-\lambda\, c\, d^{\,m}}",
            font_size=52,
            color=KNOWN_COLOR,
        ).move_to(LEFT_CENTRE + 0.9 * UP)
        spring = tex(
            "$-\\log q = \\lambda c\\, d^{\\,m}$ grows without "
            "limit:\na spring — the pull grows with distance",
            24,
            KNOWN_COLOR,
        ).next_to(known, DOWN, buff=0.4)
        scale_eq = MathTex(
            r"\lambda = \tfrac{1}{v}, \quad "
            r"\sigma_0 = \sqrt[m]{v/c}"
            r"\quad\Longrightarrow\quad "
            r"q = e^{-(d/\sigma_0)^{m}}",
            font_size=34,
        ).next_to(spring, DOWN, buff=0.55)
        scale_note = tex(
            "$v$: volume per point at the mean density;\n"
            "$\\sigma_0$: radius of a ball holding one point's "
            "share",
            20,
            ACCENT_COLOR,
        ).next_to(scale_eq, DOWN, buff=0.2)
        c_known = self.plot(lambda x: np.exp(-(x**2)), KNOWN_COLOR)
        c_lab = tex("mean density", 20, KNOWN_COLOR).next_to(
            ax.c2p(1.15, np.exp(-(1.15**2))), UR, buff=0.08
        )
        x_unit = tex(
            "(distance in units of $\\sigma_0$;  plotted for $m = 2$)", 18, ACCENT_COLOR
        ).next_to(ax_lab[0], DOWN, buff=0.08)
        self.play(Write(known))
        self.play(FadeIn(spring))
        self.marked_next_slide()
        self.play(Write(scale_eq), FadeIn(scale_note))
        self.play(Create(c_known), FadeIn(c_lab, x_unit))
        self.marked_next_slide()

        # ================= 5. the curve depends on density ================
        head = self.swap(
            head,
            "But the curve depends on the density",
            "the same distance: dense region — someone is "
            "closer;  sparse region — the disc is empty",
            FadeOut(known, spring, scale_eq, scale_note),
        )
        panels = VGroup()
        for k, (rho, name, col, seed) in enumerate(
            ((DENSE, "dense", DENSE_COLOR, 5), (SPARSE, "sparse", SPARSE_COLOR, 6))
        ):
            c = LEFT_CENTRE + np.array([(k - 0.5) * 3.3, 0.35, 0])
            rad = 0.95
            # the sparse panel shows an empty disc, so keep it clear of points
            Pk, dk = self.scatter(
                c,
                np.array([1.45, 1.45]),
                4.0 * rho,
                seed,
                exclude=0.12 if rho > 1 else rad + 0.08,
            )
            for dot, p in zip(dk, Pk):
                if np.linalg.norm(p - c[:2]) < rad:
                    dot.set_color(INSIDE_COLOR)
            ring = Circle(
                radius=rad, color=col, stroke_width=3, fill_opacity=0.08
            ).move_to(c)
            ii = Dot(c, radius=0.08, color=DEFAULT_COLOR)
            jj = Dot(c + rad * direction, radius=0.08, color=TAIL_COLOR)
            lab = txt(name, 26, col).next_to(c + 1.45 * DOWN, DOWN, buff=0.1)
            panels.add(VGroup(dk, ring, ii, jj, lab))
        c_dense = self.plot(lambda x: np.exp(-DENSE * x**2), DENSE_COLOR)
        c_sparse = self.plot(lambda x: np.exp(-SPARSE * x**2), SPARSE_COLOR)
        lab_d = txt("dense", 20, DENSE_COLOR).next_to(
            ax.c2p(0.62, np.exp(-DENSE * 0.62**2)), LEFT, buff=0.08
        )
        lab_s = txt("sparse", 20, SPARSE_COLOR).next_to(
            ax.c2p(2.6, np.exp(-SPARSE * 2.6**2)), UR, buff=0.06
        )
        unknown = tex(
            "and we don't know the density: with noise and "
            "corruption\nit isn't even one fixed number",
            22,
            ACCENT_COLOR,
        ).move_to(LEFT_CENTRE + 2.55 * DOWN)
        self.play(FadeIn(panels))
        self.play(Create(c_dense), Create(c_sparse), FadeIn(lab_d, lab_s))
        self.play(FadeIn(unknown))
        self.marked_next_slide()

        # ================= 6. average over plausible densities ============
        head = self.swap(
            head,
            "Average over plausible densities",
            "a prior on $\\lambda$ with the same mean, $1/v$; its "
            "relative spread is $1/\\sqrt{\\alpha}$",
            FadeOut(panels, unknown, c_dense, c_sparse, lab_d, lab_s),
        )
        prior = MathTex(
            r"\lambda \;\sim\; \mathrm{Gamma}"
            r"\big(\text{shape } \alpha,\ \text{rate } \alpha v\big)",
            font_size=40,
        ).move_to(LEFT_CENTRE + 1.9 * UP)
        prior_note = tex(
            "$\\alpha = 1$: the density as uncertain as its "
            "mean;  $\\alpha \\to \\infty$: known",
            20,
            ACCENT_COLOR,
        ).next_to(prior, DOWN, buff=0.18)
        # a fan of spring curves for densities spread like the prior
        # (alpha = 1: exponential; evenly spaced quantiles)
        u = (np.arange(N_FAN) + 0.5) / N_FAN
        rhos = -np.log(1 - u)
        fan = VGroup(
            *[
                self.plot(lambda x, rr=rr: np.exp(-rr * x**2), KNOWN_COLOR, 2, 0.35)
                for rr in rhos
            ]
        )
        fan_note = tex(
            "one spring for each plausible density", 22, KNOWN_COLOR
        ).move_to(LEFT_CENTRE + 0.25 * UP)
        self.play(Write(prior), FadeIn(prior_note))
        self.play(
            LaggedStart(*[Create(c) for c in fan], lag_ratio=0.08),
            FadeIn(fan_note),
            run_time=2.5,
        )
        self.marked_next_slide()

        integral = MathTex(
            r"q(d) \;=\; \int_0^\infty e^{-\lambda\, c\, d^{\,m}}\;"
            r"\mathrm{Gamma}(\lambda;\ \alpha,\ \alpha v)\; d\lambda",
            font_size=36,
        ).move_to(LEFT_CENTRE + 0.15 * DOWN)
        result = MathTex(
            r"q(d) \;=\; \Big(1 + \tfrac{1}{\alpha}"
            r"\Big(\frac{d}{\sigma_0}\Big)^{m}\Big)^{-\alpha}",
            font_size=RESULT_SIZE,
            color=TAIL_COLOR,
        ).next_to(integral, DOWN, buff=0.4)
        tail_note = tex(
            "their average: long edges stay possible, just "
            "unlikely —\na heavy tail; the pull fades with "
            "distance",
            22,
            TAIL_COLOR,
        ).next_to(result, DOWN, buff=0.3)
        c_tail = self.plot(lambda x: 1 / (1 + x**2), TAIL_COLOR, 5)
        t_lab = tex("average ($\\alpha = 1$)", 20, TAIL_COLOR).next_to(
            ax.c2p(2.7, 1 / (1 + 2.7**2)), UP, buff=0.12
        )
        self.play(FadeOut(fan_note), Write(integral))
        self.play(fan.animate.set_stroke(opacity=0.12), Create(c_tail), FadeIn(t_lab))
        self.play(Write(result))
        self.play(FadeIn(tail_note))
        self.marked_next_slide()

        # ================= 7. the family ==================================
        head = self.swap(
            head,
            "One family of attraction kernels",
            "familiar kernels are points in it",
            FadeOut(
                prior,
                prior_note,
                fan,
                integral,
                tail_note,
                ax,
                ax_lab,
                x_unit,
                c_known,
                c_lab,
                c_tail,
                t_lab,
            ),
            result.animate.scale(TABLE_HEAD_SIZE / RESULT_SIZE).move_to(2.15 * UP),
        )
        rows = [
            (r"\alpha \to \infty", r"e^{-(d/\sigma_0)^{m}}", "density known: a spring"),
            (
                r"\alpha = 1,\ m = 2",
                r"\dfrac{1}{1 + (d/\sigma_0)^{2}}",
                "Cauchy: the t-SNE kernel",
            ),
            (
                r"m = 2,\ \alpha = \tfrac{\nu+1}{2}",
                r"\left(1 + \tfrac{d^{2}}{\nu}\right)^{-\frac{\nu+1}{2}}",
                r"Student-$t$ with $\nu$ degrees of freedom",
            ),
            (
                r"\alpha = 1,\ m = 2b",
                r"\dfrac{1}{1 + (d/\sigma_0)^{2b}}",
                "the form of UMAP's curve",
            ),
        ]
        table = VGroup()
        for params, form, name in rows:
            table.add(
                VGroup(
                    MathTex(params, font_size=32),
                    MathTex(form, font_size=34, color=TAIL_COLOR),
                    tex(name, 22, ACCENT_COLOR),
                )
            )
        table.arrange(DOWN, buff=0.42).move_to(0.85 * DOWN)
        for row in table:
            row[0].set_x(-3.9)
            row[1].set_x(-0.4)
            row[2].align_to(1.9 * RIGHT, LEFT)
        for row in table:
            self.play(FadeIn(row, shift=0.1 * UP), run_time=0.7)
        self.marked_next_slide(auto_next=True)
        self.new_section("Interpreting Kernel Parameters")
