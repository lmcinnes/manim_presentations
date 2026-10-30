"""
Opening: the channels a visualization can encode data with, and why this
talk focuses on position.

Channels (a curated subset of Munzner, Visualization Analysis & Design,
2014, ch. 5, ranked by effectiveness):
  magnitude (ordered data, "how much?"): position on a common scale,
      position on an unaligned scale, length, tilt / angle, area,
      colour luminance, colour saturation
  identity (categories, "what? where?"): spatial region, colour hue, shape

Position tops both lists. The talk studies position -- a layout chooses
positions so that near on screen means similar in the data -- but the same
structure can be carried by other channels (shown: colour), and most of what
follows carries over.

    manim-slides render encodings.py VisualEncodings
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

# the tile / shading helpers from the conclusion scene, so the icon style
# stays identical across the deck (functions only -- no scene classes)
from conclusion import tint, shade, shaded_disc, icon_tile

apply_defaults()

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
GLYPH_COLOR = DEFAULT_COLOR
GUIDE_COLOR = ACCENT_COLOR
FOCUS_COLOR = HIGHLIGHT_COLOR  # the highlight on position
ROW_STEP = 0.66
GLYPH_TILE = (1.3, 0.54)  # the tile each row's marks sit on
MAG_X, ID_X = -2.2, 3.6  # column anchors (glyphs sit 1.05 right of these)
COL_TOP = 1.55
LABEL_SIZE = 21
CITATION = "channels after Munzner, Visualization Analysis & Design (2014), ch. 5"
FOOTNOTE = (
    "channels differ in how many levels they offer and how accurately "
    "they are read — position just shows these effects most clearly"
)
N_CLUSTER_POINTS = 26


# ---------------------------------------------------------------------------
# Channel glyphs: three marks differing only in one channel. Each returns the
# marks only; channel_rows() puts them on a tile (icon_tile).
# ---------------------------------------------------------------------------
MARK_COLOR = COLOR_CYCLE[0]


def ball(p, r=0.06, color=MARK_COLOR):
    # return shaded_disc(r, color, np.array(p, float), n=5, highlight=False)  # r > 0.08)
    return Dot(p, radius=r, color=color)


def guide(a, b):
    return Line(a, b).set_stroke(GUIDE_COLOR, 1.5)


def g_position_common():
    return VGroup(
        guide([-0.6, -0.2, 0], [0.6, -0.2, 0]),
        *[ball([x, -0.2 + h, 0]) for x, h in ((-0.4, 0.12), (0, 0.38), (0.4, 0.24))],
    )


def g_position_unaligned():
    g = VGroup()
    for x, y0, h in ((-0.42, -0.2, 0.2), (0, -0.05, 0.22), (0.42, -0.12, 0.3)):
        g.add(guide([x - 0.14, y0, 0], [x + 0.14, y0, 0]), ball([x, y0 + h, 0]))
    return g


def g_length():
    g = VGroup()
    for x, h in ((-0.4, 0.18), (0, 0.42), (0.4, 0.3)):
        bar = RoundedRectangle(width=0.1, height=h, corner_radius=0.03)
        bar.set_fill([tint(MARK_COLOR, 0.3), shade(MARK_COLOR, 0.2)], 1)
        bar.set_sheen_direction(RIGHT).set_stroke(shade(MARK_COLOR, 0.35), 0.8)
        g.add(bar.move_to([x, -0.2 + h / 2, 0]))
    return g


def g_angle():
    g = VGroup()
    for x, a in ((-0.42, 20), (0, 50), (0.42, 80)):
        p0 = np.array([x - 0.1, -0.18, 0])
        d = 0.38 * np.array([np.cos(a * DEGREES), np.sin(a * DEGREES), 0])
        g.add(
            Line(p0, p0 + d)
            .set_stroke(MARK_COLOR, 6)
            .set_cap_style(CapStyleType.ROUND),
            ball(p0, 0.035),
        )
    return g


def g_area():
    return VGroup(
        *[ball([x, 0, 0], r) for x, r in ((-0.42, 0.07), (0, 0.17), (0.42, 0.12))]
    )


def swatches(colours):
    g = VGroup()
    for x, c in zip((-0.42, 0, 0.42), colours):
        sq = RoundedRectangle(width=0.3, height=0.3, corner_radius=0.06)
        sq.set_fill([tint(c, 0.12), shade(c, 0.08)], 1).set_sheen_direction(DR)
        g.add(sq.set_stroke(shade(c, 0.3), 0.8).move_to([x, 0, 0]))
    return g


def g_luminance():
    light = ManimColor("#f2f2f2")
    return swatches(
        [mixbox_interpolate_color(light, DEFAULT_COLOR, t) for t in (0.2, 0.55, 0.95)]
    )


def g_saturation():
    grey = ManimColor("#9a9a9a")
    return swatches(
        [mixbox_interpolate_color(grey, COLOR_CYCLE[0], t) for t in (0.05, 0.5, 1.0)]
    )


def g_region():
    rng = np.random.default_rng(2)
    g = VGroup()
    for cx in (-0.42, 0, 0.42):
        g.add(
            *[
                ball(
                    [
                        cx + 0.06 * rng.standard_normal(),
                        0.06 * rng.standard_normal(),
                        0,
                    ],
                    0.035,
                )
                for _ in range(5)
            ]
        )
    return g


def g_hue():
    return VGroup(
        *[
            ball([x, 0, 0], 0.11, c)
            for x, c in zip(
                (-0.42, 0, 0.42), (COLOR_CYCLE[0], COLOR_CYCLE[2], COLOR_CYCLE[3])
            )
        ]
    )


def g_shape():
    def fill(m):
        return (
            m.set_fill([tint(MARK_COLOR, 0.3), shade(MARK_COLOR, 0.2)], 1)
            .set_sheen_direction(DR)
            .set_stroke(shade(MARK_COLOR, 0.35), 0.8)
        )

    return VGroup(
        ball([-0.42, 0, 0], 0.11),
        fill(RoundedRectangle(width=0.21, height=0.21, corner_radius=0.03)).move_to(
            [0, 0, 0]
        ),
        fill(Triangle()).set(height=0.24).move_to([0.42, 0, 0]),
    )


MAGNITUDE = [
    ("position on a common scale", g_position_common),
    ("position on an unaligned scale", g_position_unaligned),
    ("length", g_length),
    ("tilt / angle", g_angle),
    ("area", g_area),
    ("colour luminance", g_luminance),
    ("colour saturation", g_saturation),
]
IDENTITY = [
    ("spatial region", g_region),
    ("colour hue", g_hue),
    ("shape", g_shape),
]


# ---------------------------------------------------------------------------
class VisualEncodings(TIMCSlide):

    def heading(self, title, subtitle):
        return make_heading(title, subtitle)

    def swap(self, old, title, subtitle, *anims):
        new = self.heading(title, subtitle)
        self.play(FadeTransform(old, new), *anims)
        return new

    def channel_rows(self, rows, x):
        """Label (right-aligned) and glyph per row, top to bottom."""
        out = VGroup()
        for k, (name, glyph) in enumerate(rows):
            y = COL_TOP - k * ROW_STEP
            tile = icon_tile(glyph(), GUIDE_COLOR, GLYPH_TILE, pad=0.08)
            tile.move_to([x + 1.05, y, 0])
            lab = SmallText(name, font_size=LABEL_SIZE, color=DEFAULT_COLOR)
            lab.next_to(tile, LEFT, buff=0.3)
            out.add(VGroup(lab, tile))
        return out

    def row_intro(self, row):
        lab, tile = row
        return AnimationGroup(FadeIn(lab, shift=0.1 * RIGHT), tile.intro())

    def column_header(self, title, question, rows, x, **kwargs):
        h = SmallText(title, font_size=26, color=DEFAULT_COLOR, **kwargs)
        q = SmallText(question, font_size=20, color=ACCENT_COLOR, **kwargs)
        head = VGroup(h, q).arrange(DOWN, buff=0.08)
        return head.next_to(rows, UP, buff=0.3).set_x(rows.get_x())

    def rank_arrow(self, rows):
        a = Arrow(
            rows.get_corner(UL) + 0.3 * LEFT + 0.05 * DOWN,
            rows.get_corner(DL) + 0.3 * LEFT + 0.05 * UP,
            buff=0,
            color=GUIDE_COLOR,
            stroke_width=2,
            tip_length=0.14,
            max_tip_length_to_length_ratio=0.5,
        )
        lab = (
            SmallText("less effective", font_size=16, color=GUIDE_COLOR)
            .rotate(PI / 2)
            .next_to(a, LEFT, buff=0.08)
        )
        return VGroup(a, lab)

    def construct(self):
        # ================= 1. the channels ===============================
        head = self.heading(
            "Encoding data visually",
            "a mark — here a point — and the channels " "that can carry data",
        )
        mag = self.channel_rows(MAGNITUDE, MAG_X)
        ident = self.channel_rows(IDENTITY, ID_X)
        mag_head = self.column_header(
            "magnitude channels", "how much?", mag, MAG_X, font="Marcellus SC"
        )
        id_head = self.column_header(
            "identity channels", "what?  where?", ident, ID_X, font="Marcellus SC"
        )
        id_head.align_to(mag_head, UP)
        mag_rank, id_rank = self.rank_arrow(mag), self.rank_arrow(ident)
        cite = SmallText(CITATION, font_size=16, color=ACCENT_COLOR).to_edge(
            DOWN, buff=0.3
        )
        self.play(FadeIn(head), FadeIn(cite))
        self.play(FadeIn(mag_head), FadeIn(id_head))
        self.play(
            LaggedStart(*[self.row_intro(r) for r in mag], lag_ratio=0.3),
            FadeIn(mag_rank),
            run_time=3,
        )
        self.play(
            LaggedStart(*[self.row_intro(r) for r in ident], lag_ratio=0.3),
            FadeIn(id_rank),
            run_time=1.8,
        )
        self.marked_next_slide()

        # ================= 2. position tops both lists ===================
        head = self.swap(
            head,
            "Position comes first",
            "the most effective channel for quantities and for " "categories alike",
        )
        boxes = VGroup(
            SurroundingRectangle(
                VGroup(mag[0], mag[1]),
                color=FOCUS_COLOR,
                buff=0.1,
                corner_radius=0.1,
                stroke_width=2.5,
            ),
            SurroundingRectangle(
                ident[0],
                color=FOCUS_COLOR,
                buff=0.1,
                corner_radius=0.1,
                stroke_width=2.5,
            ),
        )
        self.play(Create(boxes))
        self.marked_next_slide()

        # ================= 3. this talk: position ========================
        self.clear_slide()
        self.add_centered_text("This talk will focus on just position")
        self.marked_next_slide()
        # head = self.swap(
        #     head,
        #     "This talk: position",
        #     "a layout places points so that near on screen "
        #     "means similar in the data",
        #     FadeOut(
        #         mag[2:],
        #         ident[1:],
        #         mag_rank,
        #         id_rank,
        #         mag_head,
        #         id_head,
        #         boxes,
        #         mag[:2],
        #         ident[0],
        #         cite,
        #     ),
        # )
        # rng = np.random.default_rng(4)
        # centres = np.array([[-1.4, 0.6], [1.2, 0.9], [0.1, -1.0]])
        # P = np.vstack(
        #     [c + 0.38 * rng.standard_normal((N_CLUSTER_POINTS, 2)) for c in centres]
        # )
        # lo, hi = P.min(0), P.max(0)

        # def layout_colour(p):
        #     """Colour from position: blend of four corner colours."""
        #     u, v = (p - lo) / (hi - lo)
        #     top = mixbox_interpolate_color(COLOR_CYCLE[0], COLOR_CYCLE[3], u)
        #     bot = mixbox_interpolate_color(COLOR_CYCLE[2], COLOR_CYCLE[1], u)
        #     return mixbox_interpolate_color(bot, top, v)

        # pts = VGroup(*[ball([*(p * 0.95), 0], 0.06, MARK_COLOR) for p in P])
        # scatter = icon_tile(
        #     pts,
        #     GUIDE_COLOR,
        #     (4.4, 3.3),
        #     pad=0.3,
        #     intro=lambda c: LaggedStart(
        #         *[FadeIn(d, scale=0.4) for d in c], lag_ratio=0.01
        #     ),
        # )
        # scatter.move_to([0, -0.45, 0])
        # self.play(scatter.intro(), run_time=1.6)
        self.marked_next_slide()
        self.clear_slide()
        self.add_centered_text(
            "Most lessons for position will carry over to other channels"
        )
        self.marked_next_slide()

        # # ================= 4. ...and it carries over =====================
        # head = self.swap(
        #     head,
        #     "...and it carries over",
        #     "the same structure can be carried by another " "channel -- here, colour",
        # )
        # self.play(scatter.animate.scale(0.8).move_to([-3.4, -0.55, 0]))
        # # recolour each point by where it sits: rebuild its shading
        # self.play(
        #     *[
        #         Transform(d, ball(d.get_center(), d.width / 2, layout_colour(p)))
        #         for d, p in zip(pts, P)
        #     ]
        # )
        # # the same items in a fixed grid: only colour carries the structure
        # order = rng.permutation(len(P))
        # cols = 9
        # cells = VGroup()
        # for slot, i in enumerate(order):
        #     r, c = divmod(slot, cols)
        #     cells.add(ball([0.36 * c, -0.36 * r, 0], 0.08, layout_colour(P[i])))
        # grid = icon_tile(
        #     cells,
        #     GUIDE_COLOR,
        #     (scatter.width, scatter.height),
        #     pad=0.3,
        #     intro=lambda c: LaggedStart(
        #         *[FadeIn(d, scale=0.4) for d in c], lag_ratio=0.01
        #     ),
        # )
        # grid.move_to([3.4, -0.55, 0])
        # arrow = Arrow(
        #     scatter.get_right() + 0.1 * RIGHT,
        #     grid.get_left() + 0.1 * LEFT,
        #     buff=0,
        #     color=GUIDE_COLOR,
        #     stroke_width=2.5,
        #     tip_length=0.16,
        #     max_tip_length_to_length_ratio=0.3,
        # )
        # labels = VGroup(
        #     SmallText("position", font_size=22, color=DEFAULT_COLOR).next_to(
        #         scatter, UP, buff=0.15
        #     ),
        #     SmallText(
        #         "colour (items in a fixed grid)", font_size=22, color=DEFAULT_COLOR
        #     ).next_to(grid, UP, buff=0.15),
        # )
        # foot = SmallText(FOOTNOTE, font_size=16, color=ACCENT_COLOR).to_edge(
        #     DOWN, buff=0.3
        # )
        # self.play(GrowArrow(arrow), FadeIn(labels[0]))
        # self.play(grid.intro(), FadeIn(labels[1]), run_time=1.6)
        # self.play(FadeIn(foot))
        # self.marked_next_slide()

        self.new_section("Making Compromises", auto_next=True)
