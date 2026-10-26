"""
Conclusion: the shape of the argument, what it means for reading a layout,
and questions to take away.

  1. The ladder: a story of giving up trust, rung by rung -- everything,
     global distances, exact local distances, their rank order, even the
     neighbours -- with what survives at each step, down to a corrupted kNN
     graph read as topology and repaired using global information.
  2. The thesis: every layout's kernel is a set of assumptions about how
     much to trust the graph (the kernel formula returns, with its tags).
  3. Reading a layout: what it repairs, and what it declines to preserve.
  4. Questions for the audience.

    manim-slides render conclusion.py Conclusion
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
    TIMCSlide,
    make_heading,
    SmallText,
    mixbox_interpolate_color,
)

import numpy as np

apply_defaults()

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
LOST_COLOR = HIGHLIGHT_COLOR  # what we stop trusting / give up
KEPT_COLOR = COLOR_CYCLE[2]  # what survives / is repaired
ICON_COLOR = DEFAULT_COLOR
SHORTCUT_COLOR = HIGHLIGHT_COLOR
STRESS_COLOR = ManimColor("#9e1b32")  # as in the swiss roll stress map
SIGMA_COLOR, M_COLOR, ALPHA_COLOR = COLOR_CYCLE[0], COLOR_CYCLE[3], COLOR_CYCLE[2]
TAIL_COLOR = HIGHLIGHT_COLOR

RUNG_Y0, RUNG_STEP = 2.0, 0.9  # ladder rows

# Contact details, faded into the bottom corners of the last slide.
# By default each gets a simple drawn icon (an envelope, a speech bubble);
# to use logo files instead, give their paths here (e.g. the official SVGs
# from the services' brand / press kits) and they are loaded and sized to fit.
CONTACT_EMAIL = "leland.mcinnes@gmail.com"
CONTACT_SOCIAL = "@leland_mcinnes"
EMAIL_ICON_SVG = "gmail.svg"  # e.g. "assets/email_logo.svg"
SOCIAL_ICON_SVG = "x_logo.svg"  # e.g. "assets/x_logo.svg"
CONTACT_ICON_HEIGHT = 0.3
X_LOST, X_ICON, X_KEPT = -3.9, -0.55, 3.2
TEXT_SIZE = 22


# ---------------------------------------------------------------------------
# Icons (about 1.1 x 0.75 scene units each)
# ---------------------------------------------------------------------------
def dot(p, r=0.045, color=ICON_COLOR):
    return Dot(p, radius=r, color=color)


def icon_globe():
    """Everything: a globe -- no flat map keeps it all."""
    r = 0.36
    g = VGroup(Circle(radius=r, color=ICON_COLOR, stroke_width=2.5))
    for w in (0.45, 0.9):
        g.add(
            Ellipse(width=2 * r * w, height=2 * r, color=ICON_COLOR, stroke_width=1.5)
        )
    g.add(Line(LEFT * r, RIGHT * r, color=ICON_COLOR, stroke_width=1.5))
    for y in (0.5, -0.5):
        hw = r * np.sqrt(1 - y**2)
        g.add(Line([-hw, y * r, 0], [hw, y * r, 0], color=ICON_COLOR, stroke_width=1.2))
    return g


def icon_roll():
    """Global distances: a rolled strip, and a straight chord across the gap
    that unrolling stretches."""
    t = np.linspace(1.5 * np.pi, 4.3 * np.pi, 200)
    a = 0.026
    spiral = VMobject(color=ICON_COLOR, stroke_width=2.5)
    spiral.set_points_smoothly([[a * s * np.cos(s), a * s * np.sin(s), 0] for s in t])
    p = lambda s: np.array([a * s * np.cos(s), a * s * np.sin(s), 0])
    s1, s2 = 2.0 * np.pi, 4.0 * np.pi  # adjacent layers
    chord = Line(p(s1), p(s2), color=STRESS_COLOR, stroke_width=3)
    return VGroup(
        spiral, chord, dot(p(s1), color=STRESS_COLOR), dot(p(s2), color=STRESS_COLOR)
    )


def icon_concentration():
    """Exact local distances: every neighbour at nearly the same distance."""
    c = dot(ORIGIN, 0.06)
    ring = DashedVMobject(
        Circle(radius=0.32, color=ACCENT_COLOR, stroke_width=1.5), num_dashes=24
    )
    rng = np.random.default_rng(1)
    pts = VGroup(
        *[
            dot(
                0.32
                * (1 + 0.04 * rng.standard_normal())
                * np.array([np.cos(a), np.sin(a), 0])
            )
            for a in np.linspace(0, TAU, 7)[:-1] + 0.3
        ]
    )
    spokes = VGroup(
        *[Line(ORIGIN, d.get_center(), color=ACCENT_COLOR, stroke_width=1) for d in pts]
    )
    return VGroup(ring, spokes, pts, c)


def icon_rank():
    """Their rank order: inside the window, near neighbours are shuffled."""
    c = dot(ORIGIN, 0.06)
    window = DashedVMobject(
        Circle(radius=0.36, color=ACCENT_COLOR, stroke_width=1.5), num_dashes=24
    )
    labs = VGroup()
    for k, (ang, rad) in enumerate(((0.4, 0.14), (2.3, 0.22), (4.3, 0.29))):
        p = rad * np.array([np.cos(ang), np.sin(ang), 0])
        labs.add(dot(p))
        labs.add(
            SmallText(str((2, 3, 1)[k]), font_size=17, color=LOST_COLOR).next_to(
                p, UR, buff=0.01
            )
        )
    return VGroup(window, labs, c)


def icon_shortcut():
    """Even the neighbours: a ring of points with a shortcut across it."""
    n, r = 12, 0.34
    P = [
        r * np.array([np.cos(a), np.sin(a), 0]) for a in np.linspace(0, TAU, n + 1)[:-1]
    ]
    edges = VGroup(
        *[
            Line(P[i], P[(i + 1) % n], color=ICON_COLOR, stroke_width=1.5)
            for i in range(n)
        ]
    )
    cut = Line(P[1], P[7], color=SHORTCUT_COLOR, stroke_width=3)
    return VGroup(edges, cut, *[dot(p) for p in P])


def icon_tail():
    """The repair: a pull that rises, peaks and fades (the heavy tail)."""
    w, h = 1.0, 0.55
    ax = VGroup(
        Line([-w / 2, -h / 2, 0], [w / 2, -h / 2, 0]),
        Line([-w / 2, -h / 2, 0], [-w / 2, h / 2, 0]),
    ).set_stroke(ICON_COLOR, 1.5)
    xs = np.linspace(0, 4, 80)
    ys = 2 * xs / (1 + xs**2)
    pts = [[-w / 2 + w * x / 4, -h / 2 + h * y / 1.05, 0] for x, y in zip(xs, ys)]
    curve = VMobject(color=KEPT_COLOR, stroke_width=3).set_points_smoothly(pts)
    return VGroup(ax, curve)


def icon_noise():
    """Noise in your representation: isotropic versus structured."""
    rng = np.random.default_rng(3)
    iso = VGroup(
        *[dot((0.12 * rng.standard_normal(2)).tolist() + [0], 0.025) for _ in range(40)]
    ).shift(0.3 * LEFT)
    A = np.array([[0.2, 0.0], [0.11, 0.035]])
    aniso = VGroup(
        *[
            dot((A @ rng.standard_normal(2)).tolist() + [0], 0.025, ACCENT_COLOR)
            for _ in range(40)
        ]
    ).shift(0.32 * RIGHT)
    return VGroup(iso, aniso)


def icon_dial():
    """Setting the knobs: a dial."""
    arc = Arc(
        radius=0.36,
        start_angle=-PI / 6,
        angle=4 * PI / 3,
        color=ICON_COLOR,
        stroke_width=2.5,
    )
    ticks = VGroup(
        *[
            Line(
                0.29 * np.array([np.cos(a), np.sin(a), 0]),
                0.36 * np.array([np.cos(a), np.sin(a), 0]),
                color=ICON_COLOR,
                stroke_width=1.5,
            )
            for a in np.linspace(-PI / 6, 7 * PI / 6, 7)
        ]
    )
    needle = Line(
        ORIGIN,
        0.3 * np.array([np.cos(2.0), np.sin(2.0), 0]),
        color=ALPHA_COLOR,
        stroke_width=3,
    )
    return VGroup(arc, ticks, needle, dot(ORIGIN, 0.05))


def icon_given_up():
    """Showing what was given up: short edges kept, one long edge stretched."""
    rng = np.random.default_rng(5)
    P = np.c_[np.linspace(-0.45, 0.45, 9), 0.06 * rng.standard_normal(9), np.zeros(9)]
    short = VGroup(
        *[
            Line(
                P[i],
                P[i + 1],
                color=mixbox_interpolate_color(
                    ManimColor("#c4c4c4"), DEFAULT_COLOR, 0.3
                ),
                stroke_width=2,
            )
            for i in range(8)
        ]
    )
    stretched = ArcBetweenPoints(
        P[1], P[7], angle=-PI / 2.2, color=STRESS_COLOR, stroke_width=3
    )
    return VGroup(short, stretched, *[dot(p, 0.035) for p in P])


def icon_envelope():
    body = Rectangle(width=0.42, height=0.28, color=ICON_COLOR, stroke_width=2)
    flap = VMobject(color=ICON_COLOR, stroke_width=2).set_points_as_corners(
        [body.get_corner(UL), body.get_center() + 0.02 * DOWN, body.get_corner(UR)]
    )
    return VGroup(body, flap)


def icon_speech_bubble():
    bubble = RoundedRectangle(
        width=0.42, height=0.28, corner_radius=0.08, color=ICON_COLOR, stroke_width=2
    )
    tail = VMobject(color=ICON_COLOR, stroke_width=2).set_points_as_corners(
        [
            bubble.get_bottom() + 0.12 * LEFT,
            bubble.get_bottom() + 0.16 * LEFT + 0.1 * DOWN,
            bubble.get_bottom() + 0.02 * LEFT,
        ]
    )
    dots = VGroup(
        *[
            Dot(bubble.get_center() + x * RIGHT, radius=0.022, color=ICON_COLOR)
            for x in (-0.09, 0, 0.09)
        ]
    )
    return VGroup(bubble, tail, dots)


def contact_icon(svg_path, fallback):
    """A logo file if one is given (and exists), else a drawn icon."""
    from pathlib import Path

    if svg_path and Path(svg_path).exists():
        return SVGMobject(svg_path).set(height=CONTACT_ICON_HEIGHT)
    return fallback().set(height=CONTACT_ICON_HEIGHT)


# ---------------------------------------------------------------------------
class Conclusion(TIMCSlide):

    def heading(self, title, subtitle):
        return make_heading(title, subtitle)

    def swap(self, old, title, subtitle, *anims):
        new = self.heading(title, subtitle)
        self.play(FadeTransform(old, new), *anims)
        return new

    def rung(self, k, lost, icon, kept):
        y = RUNG_Y0 - k * RUNG_STEP
        lost_t = SmallText(lost, font_size=TEXT_SIZE, color=LOST_COLOR).move_to(
            [X_LOST, y, 0]
        )
        icon.scale(1.2).move_to([X_ICON, y, 0])
        kept_t = SmallText(kept, font_size=TEXT_SIZE, color=KEPT_COLOR).move_to(
            [X_KEPT, y, 0]
        )
        arrow = Arrow(
            [X_ICON + 0.75, y, 0],
            [X_KEPT - kept_t.width / 2 - 0.15, y, 0],
            buff=0,
            color=ACCENT_COLOR,
            stroke_width=2,
            max_tip_length_to_length_ratio=0.12,
        )
        return VGroup(lost_t, icon, arrow, kept_t)

    def construct(self):
        # ================= 1. the ladder =================================
        head = self.heading(
            "What we gave up, and what survived",
            "the talk as a ladder: each step trusts less of " "the data",
        )
        cols = VGroup(
            SmallText(
                "we stop trusting",
                font="Marcellus SC",
                font_size=28,
                color=LOST_COLOR,
            ).move_to([X_LOST, RUNG_Y0 + 0.65, 0]),
            SmallText(
                "what survives",
                font="Marcellus SC",
                font_size=28,
                color=KEPT_COLOR,
            ).move_to([X_KEPT, RUNG_Y0 + 0.65, 0]),
        )
        rungs = [
            ("everything", icon_globe(), "a choice of what to keep"),
            ("global distances", icon_roll(), "local neighbourhoods"),
            ("exact local distances", icon_concentration(), "which points are near"),
            ("their rank order", icon_rank(), "the neighbourhood as a set"),
            ("even the neighbours", icon_shortcut(), "a graph that is mostly right"),
        ]
        trust = Arrow(
            [X_LOST - 2.55, RUNG_Y0 + 0.2, 0],
            [X_LOST - 2.55, RUNG_Y0 - 4 * RUNG_STEP - 0.2, 0],
            buff=0,
            color=ACCENT_COLOR,
            stroke_width=2.5,
            max_tip_length_to_length_ratio=0.06,
        )
        trust_lab = (
            SmallText("trust", font_size=18, color=ACCENT_COLOR)
            .rotate(PI / 2)
            .next_to(trust, LEFT, buff=0.1)
        )
        self.play(FadeIn(head), FadeIn(cols), GrowArrow(trust), FadeIn(trust_lab))
        ladder = VGroup()
        for k, (lost, icon, kept) in enumerate(rungs):
            row = self.rung(k, lost, icon, kept)
            self.play(FadeIn(row[0], shift=0.1 * RIGHT), FadeIn(row[1]))
            self.play(GrowArrow(row[2]), FadeIn(row[3], shift=0.1 * RIGHT))
            ladder.add(row)
            self.marked_next_slide()

        bottom_text = SmallText(
            "a corrupted kNN graph -- read as topology, not geometry --\n"
            "repaired by a layout that uses global information",
            font_size=TEXT_SIZE,
            color=DEFAULT_COLOR,
        )
        bottom_icon = icon_tail()
        bottom = VGroup(bottom_icon, bottom_text).arrange(RIGHT, buff=0.45)
        box = SurroundingRectangle(
            bottom, color=KEPT_COLOR, buff=0.2, corner_radius=0.1, stroke_width=2
        )
        VGroup(box, bottom).move_to([0, RUNG_Y0 - 5 * RUNG_STEP - 0.25, 0])
        self.play(Create(box), FadeIn(bottom))
        self.marked_next_slide()

        # ================= 2. the thesis =================================
        head = self.swap(
            head,
            "Every layout encodes assumptions",
            "whether you chose them or not",
            FadeOut(cols, trust, trust_lab, ladder, box, bottom),
        )
        thesis = SmallText(
            "a layout's kernel is a set of assumptions\n"
            "about how much to trust the graph",
            font_size=40,
            color=DEFAULT_COLOR,
        ).move_to(1.2 * UP)
        formula = MathTex(
            r"q(d) \;=\; \Big(1 + ",
            r"\tfrac{1}{\alpha}",
            r"\,\Big(d \,/",
            r"\sigma_0",
            r"\Big)",
            r"^{m}",
            r"\Big)",
            r"^{-\alpha}",
            font_size=60,
        ).move_to(0.9 * DOWN)
        formula[1].set_color(ALPHA_COLOR)
        formula[3].set_color(SIGMA_COLOR)
        formula[5].set_color(M_COLOR)
        formula[7].set_color(ALPHA_COLOR)

        def tag(part, text, color, side):
            label = SmallText(text, font_size=24, color=color)
            label.next_to(part, side, buff=0.45)
            arrow = Arrow(
                label.get_edge_center(-side),
                part.get_edge_center(side),
                buff=0.08,
                color=color,
                stroke_width=2.5,
                max_tip_length_to_length_ratio=0.3,
            )
            return VGroup(label, arrow)

        def tag_at(part, text, color, label_pos):
            """A label at label_pos with a straight arrow to the part."""
            label = SmallText(text, font_size=24, color=color).move_to(label_pos)
            start = (
                label.get_bottom() if label_pos[1] > part.get_y() else label.get_top()
            )
            end = part.get_top() if label_pos[1] > part.get_y() else part.get_bottom()
            return VGroup(
                label,
                Arrow(
                    start,
                    end,
                    buff=0.08,
                    color=color,
                    stroke_width=2.5,
                    tip_length=0.16,
                    max_tip_length_to_length_ratio=0.5,
                ),
            )

        fb = formula.get_bottom()[1]
        tags = VGroup(
            tag_at(
                formula[5],
                "what dimension neighbourhoods fill",
                M_COLOR,
                [min(formula[5].get_x(), 3.6), formula.get_top()[1] + 0.55, 0],
            ),
            tag_at(
                formula[1],
                "how certain the local density is",
                ALPHA_COLOR,
                [formula[1].get_x() - 0.8, fb - 0.45, 0],
            ),
            tag_at(
                formula[7],
                "how hard a long edge may pull",
                TAIL_COLOR,
                [min(formula[7].get_x(), 4.9), fb - 1.25, 0],
            ),
        )
        self.play(FadeIn(thesis))
        self.marked_next_slide()
        self.play(Write(formula))
        for t in tags:
            self.play(FadeIn(t), run_time=0.8)
        self.marked_next_slide()

        # ================= 3. reading a layout ===========================
        head = self.swap(
            head,
            "Reading a layout",
            "what it works to repair, and what it has declared " "it will not preserve",
            FadeOut(thesis, formula, tags),
        )

        def column(title, items, color, x):
            """A title and items on fixed rows, so the two columns line up."""
            # align by the top: "repaired" has a descender, "declined" none
            t = SmallText(title, font_size=32, color=color).move_to(
                [x, 1.45, 0], aligned_edge=UP
            )
            rows = VGroup(
                *[
                    SmallText(a, font_size=22, color=DEFAULT_COLOR).move_to(
                        [x, 0.15 - 1.15 * k, 0]
                    )
                    for k, a in enumerate(items)
                ]
            )
            return VGroup(t, rows)

        repaired = column(
            "repaired",
            [
                "neighbourhoods\n(short distances survive)",
                "connectivity\n(shortcuts rejected, true edges kept)",
                "grouping",
            ],
            KEPT_COLOR,
            -3.4,
        )
        declined = column(
            "declined",
            [
                "long distances\n(the tail lets long edges go)",
                "gaps between clusters\n(the same)",
                "density and cluster size\n(averaged out by the prior)",
            ],
            LOST_COLOR,
            3.4,
        )
        divider = DashedLine(1.6 * UP, 2.4 * DOWN, color=ACCENT_COLOR, stroke_width=1.5)
        self.play(FadeIn(repaired[0]), FadeIn(declined[0]), Create(divider))
        for a, b in zip(repaired[1], declined[1]):
            self.play(
                FadeIn(a, shift=0.1 * UP), FadeIn(b, shift=0.1 * UP), run_time=0.8
            )
        self.marked_next_slide()

        # ================= 4. questions ==================================
        head = self.swap(
            head,
            "Questions to take away",
            "three places this community is well placed to " "push further",
            FadeOut(repaired, declined, divider),
        )
        questions = [
            (
                icon_noise(),
                "What is noise in your representation?",
                "our model was isotropic Gaussian noise --\n"
                "activations and embeddings almost certainly are not",
            ),
            (
                icon_dial(),
                "Can the knobs be set from the data?",
                "a layout's measured dimension tracks the m you ask for --\n"
                "but any estimate comes from the same corrupted graph",
            ),
            (
                icon_given_up(),
                "How do we show what a layout gave up?",
                "it cannot be judged point by point:\n"
                "good and bad hubs look the same locally",
            ),
        ]
        qs = VGroup()
        for k, (icon, q, why) in enumerate(questions):
            y = 1.55 - 1.75 * k
            icon.scale(1.25).move_to([-5.3, y, 0])
            q_t = SmallText(q, font_size=30, color=DEFAULT_COLOR)
            why_t = SmallText(why, font_size=20, color=ACCENT_COLOR)
            text = VGroup(q_t, why_t).arrange(DOWN, buff=0.15)
            text.move_to([0.9, y, 0])
            qs.add(VGroup(icon, text))
            self.play(FadeIn(icon), FadeIn(q_t, shift=0.1 * RIGHT))
            self.play(FadeIn(why_t))
            self.marked_next_slide()

        # ---- contact details, in the bottom corners ----
        contacts = VGroup()
        for text, svg, fallback, corner in (
            (CONTACT_EMAIL, EMAIL_ICON_SVG, icon_envelope, DL),
            (CONTACT_SOCIAL, SOCIAL_ICON_SVG, icon_speech_bubble, DR),
        ):
            if not text:
                continue
            item = VGroup(
                contact_icon(svg, fallback),
                SmallText(text, font_size=20, color=ACCENT_COLOR),
            ).arrange(RIGHT, buff=0.18)
            contacts.add(item.to_corner(corner, buff=0.35))
        if len(contacts):
            self.play(FadeIn(contacts), run_time=1.5)
            self.marked_next_slide()
