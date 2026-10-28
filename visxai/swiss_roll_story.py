"""
The swiss roll, in three steps of the argument (built on swiss_roll.py):

  1. SwissRollWhyUnroll   the points come from a rolled-up sheet; unrolling
                          that sheet is the natural flat picture, and its
                          two directions are the sheet's own coordinates
                          (shown with a 2D colour map: hue = position along
                          the roll, lightness = height)

  2. SwissRollShortDistances   nearest-neighbour distances barely change
                               as the roll unrolls
  3. SwissRollLongDistances    long distances, especially across the gaps
                               between layers, are stretched enormously --
                               and giving them up is what lets it lie flat
  4. SwissRollDistanceContrast  both at once: many short distances stay grey
                               while a few long ones turn red

Edges are coloured by stress: how much longer each one is now than its
original straight-line length on the roll (log scale, grey = unchanged).

    manim-slides render swiss_roll_story.py SwissRollWhyUnroll
    manim-slides render swiss_roll_story.py SwissRollShortDistances
    manim-slides render swiss_roll_story.py SwissRollLongDistances
    manim-slides render swiss_roll_story.py SwissRollDistanceContrast
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
    mixbox_interpolate_color,
    SmallText,
)

from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from swiss_roll import SwissRollUnroll

apply_defaults()


# ---------------------------------------------------------------------------
# A 2D colour map: hue for position along the roll, lightness for height
# (OKLCH, constant chroma, so hue and lightness read independently)
# ---------------------------------------------------------------------------
HUE_START, HUE_SPAN = 330.0, 260.0  # skips the blue-violet band (~250-300)
L_RANGE = (0.32, 0.92)  # lightness, bottom to top
CHROMA = 0.08  # kept in sRGB gamut at every L and hue


def oklch_to_rgb(L, C, h_deg):
    h = np.radians(h_deg)
    a, b = C * np.cos(h), C * np.sin(h)
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_**3, m_**3, s_**3
    lin = np.stack(
        [
            4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
            -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
            -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
        ],
        -1,
    )
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)


def sheet_colour(u_along, v_up):
    """u_along, v_up in [0, 1] -> RGB in [0, 1]."""
    L = L_RANGE[0] + (L_RANGE[1] - L_RANGE[0]) * np.asarray(v_up)
    h = (HUE_START + HUE_SPAN * np.asarray(u_along)) % 360
    return oklch_to_rgb(L, CHROMA, h)


# ---------------------------------------------------------------------------
# Stress: log2(current length / original length), coloured dark blue -> accent blue -> orange ->
# deep red. Edges essentially never shorten, so the map is one-sided.
# ---------------------------------------------------------------------------
STRESS_MAX = 2.5  # log2 ratio shown at full colour (8x)
STRESS_BINS = 32
STRESS_RED = ManimColor("#de1b32")
STRESS_STOPS = [
    # DEFAULT_COLOR,
    ACCENT_COLOR,
    HIGHLIGHT_COLOR,
    # ManimColor("#9e1b32"),
    STRESS_RED,
]


def stress_colour(v):
    """v = log2(length ratio), clipped to [0, STRESS_MAX]."""
    x = np.clip(v / STRESS_MAX, 0, 1) * (len(STRESS_STOPS) - 1)
    i = min(int(np.floor(x)), len(STRESS_STOPS) - 2)
    return mixbox_interpolate_color(STRESS_STOPS[i], STRESS_STOPS[i + 1], x - i)


class SwissRollStory(SwissRollUnroll):
    """Shared pieces for the story scenes: headings that stay on screen,
    and a 'squash' (a linear projection that collapses the roll's depth)."""

    SHEET_ALONG, SHEET_UP = 120, 10  # surface faces along / up the sheet

    def setup_scene(self):
        self.setup_data()
        self.unroll = ValueTracker(0.0)
        self.squash = ValueTracker(0.0)
        self.place = ValueTracker(0.0)  # 0: roll centred; 1: at the pinned end
        # recolour the points with the 2D map (sheet coordinates)
        self.u_along = (self.s - self.s.min()) / np.ptp(self.s)
        self.v_up = (self.height - self.height.min()) / np.ptp(self.height)
        rgb = sheet_colour(self.u_along, self.v_up)
        self.rgbas = np.c_[rgb, np.ones(len(rgb))]
        cloud = PMobject(stroke_width=self.POINT_SIZE)
        cloud.add_points(self.current_positions(), rgbas=self.rgbas)
        cloud.add_updater(lambda m: m.set(points=self.current_positions()))
        self.set_camera_orientation(**self.ROLLED_CAMERA)
        return cloud

    def sheet_positions(self, s_vals, h_vals):
        """The base positions, with the optional squash (a projection that
        collapses the roll's depth)."""
        P = super().sheet_positions(s_vals, h_vals)
        P[:, 1] *= 1 - self.squash.get_value()
        return P

    def build_sheet(self):
        """The sheet as a depth-sorted mesh of faces coloured by the 2D map,
        following the unrolling. Lighting is switched off so the colours
        (which encode height) are not altered."""
        self.renderer.camera.should_apply_shading = False
        na, nu = self.SHEET_ALONG, self.SHEET_UP
        s_lo, s_hi = self.s.min(), self.s.max()
        h_lo, h_hi = self.height.min(), self.height.max()
        S, H = np.meshgrid(
            np.linspace(s_lo, s_hi, na + 1),
            np.linspace(h_lo, h_hi, nu + 1),
            indexing="ij",
        )
        self._sheet_s, self._sheet_h = S.ravel(), H.ravel()
        faces = VGroup()
        for i in range(na):
            for j in range(nu):
                c = sheet_colour((i + 0.5) / na, (j + 0.5) / nu)
                col = rgb_to_color(c)
                f = VMobject(
                    fill_color=col,
                    fill_opacity=1.0,
                    stroke_color=col,
                    stroke_width=0.6,
                    shade_in_3d=True,
                )
                faces.add(f)
        idx = lambda i, j: i * (nu + 1) + j

        def update(g):
            P = self.sheet_positions(self._sheet_s, self._sheet_h)
            k = 0
            for i in range(na):
                for j in range(nu):
                    g[k].set_points_as_corners(
                        [
                            P[idx(i, j)],
                            P[idx(i + 1, j)],
                            P[idx(i + 1, j + 1)],
                            P[idx(i, j + 1)],
                            P[idx(i, j)],
                        ]
                    )
                    k += 1

        update(faces)
        faces.add_updater(update)
        return faces

    # ---- stress-coloured edges ------------------------------------------
    def stress_edges(self, pairs, width):
        """Edges between sample pairs, coloured by stress and updated as the
        roll unrolls (one VMobject per colour bin, so it stays fast)."""
        pairs = np.asarray(pairs)
        P0 = self.current_positions()
        self._edge_len0 = getattr(self, "_edge_len0", {})
        L0 = np.linalg.norm(P0[pairs[:, 0]] - P0[pairs[:, 1]], axis=1)
        layers = VGroup(
            *[
                VMobject(
                    stroke_color=stress_colour((b + 0.5) / STRESS_BINS * STRESS_MAX),
                    stroke_width=width,
                )
                for b in range(STRESS_BINS)
            ]
        )

        def bins_now(A, B):
            ratio = np.linalg.norm(A - B, axis=1) / np.maximum(L0, 1e-9)
            v = np.log2(np.maximum(ratio, 1e-9))
            return np.clip(
                (v / STRESS_MAX * STRESS_BINS).astype(int), 0, STRESS_BINS - 1
            )

        def update(g):
            P = self.current_positions()
            A, B = P[pairs[:, 0]], P[pairs[:, 1]]
            # frozen: keep each edge's colour from the moment of freezing
            b = g.frozen if g.frozen is not None else bins_now(A, B)
            for k, layer in enumerate(g):
                sel = b == k
                if not sel.any():
                    layer.set_points(np.zeros((0, 3)))
                    continue
                a, c = A[sel], B[sel]
                pts = np.empty((4 * len(a), 3))
                pts[0::4], pts[3::4] = a, c
                pts[1::4], pts[2::4] = a + (c - a) / 3, a + 2 * (c - a) / 3
                layer.set_points(pts)

        layers.frozen = None
        update(layers)
        layers.add_updater(update)
        layers.lengths0 = L0

        def freeze():
            P = self.current_positions()
            layers.frozen = bins_now(P[pairs[:, 0]], P[pairs[:, 1]])

        layers.freeze = freeze
        return layers

    # ---- which pairs to draw --------------------------------------------
    SHORT_RADIUS = 0.25  # scene units on the roll: ~3 point spacings, well
    # under the ~0.75 gap between layers

    def short_pairs(self, n_pairs, seed=0):
        """A random sample of close pairs (within SHORT_RADIUS on the roll)."""
        from sklearn.neighbors import NearestNeighbors

        P0 = self.current_positions()
        nbrs = NearestNeighbors(radius=self.SHORT_RADIUS).fit(P0)
        idx = nbrs.radius_neighbors(P0, return_distance=False)
        allp = np.array([(i, j) for i, row in enumerate(idx) for j in row if j > i])
        rng = np.random.default_rng(seed)
        take = rng.choice(len(allp), min(n_pairs, len(allp)), replace=False)
        return allp[take]

    def long_pairs(self, n_random, n_ends, seed=2):
        """Pairs far apart along the sheet: random ones (more than a quarter
        of the sheet apart) plus some from one end to the other."""
        rng = np.random.default_rng(seed)
        n = len(self.s)
        cand = rng.integers(0, n, (20 * max(n_random, 1), 2))
        far = np.abs(self.s[cand[:, 0]] - self.s[cand[:, 1]]) > 0.25 * np.ptp(self.s)
        rand = cand[far][:n_random]
        lo = np.flatnonzero(self.u_along < 0.08)
        hi = np.flatnonzero(self.u_along > 0.92)
        ends = np.c_[rng.choice(lo, n_ends), rng.choice(hi, n_ends)]
        return np.r_[rand, ends].astype(int)

    def stress_ratios(self, pairs, L0):
        P = self.current_positions()
        return np.linalg.norm(P[pairs[:, 0]] - P[pairs[:, 1]], axis=1) / L0

    def stress_bar(self):
        """A colour bar for stress, fixed to the screen (bottom right)."""
        n = 60
        cells = VGroup(
            *[
                Rectangle(
                    width=3.0 / n,
                    height=0.18,
                    stroke_width=0,
                    fill_color=stress_colour(i / (n - 1) * STRESS_MAX),
                    fill_opacity=1,
                )
                for i in range(n)
            ]
        ).arrange(RIGHT, buff=0)
        ticks = VGroup()
        for v, lab in ((0, "unchanged"), (1, "2x"), (2, "4x"), (np.log2(6), "6x")):
            t = SmallText(lab, font_size=16, color=DEFAULT_COLOR)
            t.next_to(
                cells.get_left() + (v / STRESS_MAX) * cells.width * RIGHT,
                DOWN,
                buff=0.16,
            )
            ticks.add(t)
        title = SmallText(
            "length now / length on the roll", font_size=18, color=ACCENT_COLOR
        )
        title.next_to(cells, UP, buff=0.16)
        return VGroup(cells, ticks, title).to_corner(DR, buff=0.45).shift(0.25 * UP)

    def points_cloud(self, size=1.6):
        cloud = PMobject(stroke_width=size)
        cloud.add_points(self.current_positions(), rgbas=self.rgbas)
        cloud.add_updater(lambda m: m.set(points=self.current_positions()))
        return cloud

    def unroll_sequence(self):
        """Camera to face the strip while the roll settles, slide it to the
        pinned end, unroll."""
        self.move_camera(
            **self.FLAT_CAMERA,
            added_anims=[self.place.animate.set_value(0.5)],
            run_time=self.PLACE_SECONDS,
            rate_func=smooth,
        )
        self.play(
            self.place.animate.set_value(1.0),
            run_time=self.PLACE_SECONDS,
            rate_func=smooth,
        )
        self.play(
            self.unroll.animate.set_value(1.0),
            run_time=self.UNROLL_SECONDS,
            rate_func=smooth,
        )

    def reroll_sequence(self):
        """The unroll in reverse: roll up, slide back, turn the camera."""
        self.play(
            self.unroll.animate.set_value(0.0),
            run_time=self.UNROLL_SECONDS,
            rate_func=smooth,
        )
        self.play(
            self.place.animate.set_value(0.5),
            run_time=self.PLACE_SECONDS,
            rate_func=smooth,
        )
        self.move_camera(
            **self.ROLLED_CAMERA,
            added_anims=[self.place.animate.set_value(0.0)],
            run_time=self.PLACE_SECONDS,
            rate_func=smooth,
        )

    def heading(self, title, subtitle):
        return make_heading(title, subtitle)

    def swap_heading(self, old, title, subtitle, *anims, **kw):
        new = self.heading(title, subtitle)
        self.add_fixed_in_frame_mobjects(new)
        if old is None:
            self.play(FadeIn(new), *anims, **kw)
        else:
            self.play(FadeOut(old), FadeIn(new), *anims, **kw)
            self.remove_fixed_in_frame_mobjects(old)
        return new

    def spin_once(self, direction=1):
        self.begin_ambient_camera_rotation(rate=direction * self.SPIN_RATE)
        self.wait(TAU / self.SPIN_RATE)
        self.stop_ambient_camera_rotation()


# ---------------------------------------------------------------------------
class SwissRollWhyUnroll(SwissRollStory):
    """The points come from a rolled-up sheet; unrolling the sheet is the
    natural flat picture."""

    POINT_SIZE = 5.0
    SPIN_RATE = 1.0

    def construct(self):

        self.add_centered_text(
            "Without known structure we have to hope we can discover it from the data"
        )
        self.marked_next_slide()
        self.clear_slide()
        self.add_centered_text(
            "We then have to hope it can be unpacked into a few dimensions"
        )
        self.marked_next_slide()
        self.clear_slide()

        cloud = self.setup_scene()
        head = self.swap_heading(
            None,
            "Classic Swiss-Roll data",
            "",
        )
        self.play(FadeIn(cloud), run_time=1.5)
        self.add(cloud)
        self.marked_next_slide()
        self.spin_once()

        # ---- the sheet they were sampled from ----
        self.marked_next_slide()
        sheet = self.build_sheet()
        head = self.swap_heading(
            head,
            "There is an implicit structure",
            "",
            FadeIn(sheet),
            FadeOut(cloud),
            run_time=2,
        )
        self.marked_next_slide()
        self.spin_once(direction=-1)

        # ---- unroll it ----
        self.marked_next_slide()
        head = self.swap_heading(
            head,
            "And a natural 2D representation",
            "",
        )
        self.move_camera(
            **self.FLAT_CAMERA,
            added_anims=[self.place.animate.set_value(0.5)],
            run_time=self.PLACE_SECONDS,
            rate_func=smooth,
        )
        self.play(
            self.place.animate.set_value(1.0),
            run_time=self.PLACE_SECONDS,
            rate_func=smooth,
        )
        self.play(
            self.unroll.animate.set_value(1.0),
            run_time=self.UNROLL_SECONDS,
            rate_func=smooth,
        )
        self.marked_next_slide()

        # ---- the unrolled sheet's coordinates ----
        P = self.sheet_positions(self._sheet_s, self._sheet_h)
        x0, x1 = P[:, 0].min(), P[:, 0].max()
        z0, z1 = P[:, 2].min(), P[:, 2].max()
        along = Arrow3D(
            [x0, 0, z0 - 0.35], [x1, 0, z0 - 0.35], color=DEFAULT_COLOR, thickness=0.015
        )
        up = Arrow3D(
            [x0 - 0.35, 0, z0], [x0 - 0.35, 0, z1], color=DEFAULT_COLOR, thickness=0.015
        )
        lab_along = SmallText(
            "position along the roll  (hue)", font_size=23.76, color=DEFAULT_COLOR
        ).move_to([0, 0, z0 - 0.75])
        lab_up = SmallText(
            "height  (lightness)", font_size=23.76, color=DEFAULT_COLOR
        ).move_to([x0 + 0.9, 0, z1 + 0.4])
        self.add_fixed_orientation_mobjects(lab_along, lab_up)
        head = self.swap_heading(
            head,
            "With implicit coordinates",
            "",
            Create(along),
            Create(up),
            FadeIn(lab_along, lab_up),
        )
        self.marked_next_slide()

        self.clear_slide()
        self.marked_next_slide()


# ---------------------------------------------------------------------------
class SwissRollShortDistances(SwissRollStory):
    """Short distances barely change as the roll unrolls."""

    N_SHORT = 1200

    def construct(self):

        self.setup_scene()
        pairs = self.short_pairs(self.N_SHORT)
        cloud = self.points_cloud()
        edges = self.stress_edges(pairs, width=1.0)
        bar = self.stress_bar()
        self.add_fixed_in_frame_mobjects(bar)

        head = self.swap_heading(
            None,
            "Short distances",
            "a random sample of pairs of points close " "together on the roll",
            FadeIn(cloud),
            FadeIn(bar),
        )
        self.add(cloud)
        self.play(Create(edges), run_time=2)
        self.marked_next_slide()
        self.spin_once()

        self.marked_next_slide()
        head = self.swap_heading(
            head,
            "Short distances",
            "unroll, colouring each line by how much it " "stretches",
        )
        self.unroll_sequence()
        ratio = self.stress_ratios(pairs, edges.lengths0)
        kept = np.mean(np.abs(ratio - 1) < 0.10) * 100
        head = self.swap_heading(
            head,
            "Short distances survive",
            f"{kept:.0f}% of short distances change by " "less than 10%",
        )
        self.marked_next_slide()


class SwissRollLongDistances(SwissRollStory):
    """Long distances, especially across the gaps between layers, are
    stretched enormously; giving them up is what lets the sheet lie flat."""

    N_RANDOM, N_ENDS = 22, 8

    def construct(self):
        self.setup_scene()
        pairs = self.long_pairs(self.N_RANDOM, self.N_ENDS)
        cloud = self.points_cloud()
        edges = self.stress_edges(pairs, width=2.4)
        bar = self.stress_bar()
        self.add_fixed_in_frame_mobjects(bar)

        head = self.swap_heading(
            None,
            "Long distances",
            "pairs of points far apart along the sheet, " "joined by straight lines",
            FadeIn(cloud),
            FadeIn(bar),
        )
        self.add(cloud)
        self.play(Create(edges), run_time=2)
        self.marked_next_slide()
        self.spin_once()

        self.marked_next_slide()
        head = self.swap_heading(
            head,
            "Long distances",
            "many of these lines cut across the gaps "
            "between layers, where there is no sheet",
        )
        self.unroll_sequence()
        ratio = self.stress_ratios(pairs, edges.lengths0)
        big = np.mean(ratio > 2) * 100
        head = self.swap_heading(
            head,
            "Long distances are given up",
            f"{big:.0f}% of these lines end up more than "
            "twice as long -- that is what lets the "
            "sheet lie flat",
        )
        self.marked_next_slide()


class SwissRollDistanceContrast(SwissRollStory):
    """Short and long distances in one unroll: the many short ones stay grey
    while the few long ones turn red."""

    N_SHORT = 900
    N_RANDOM, N_ENDS = 8, 4
    POINT_SIZE = 5.0

    def construct(self):

        self.add_centered_text("If unrolling is natural, what are we giving up?")
        self.marked_next_slide()
        self.clear_slide()

        self.setup_scene()
        head = self.swap_heading(
            None,
            "Classic Swiss-Roll data",
            "",
        )
        cloud = self.points_cloud(size=3.2)
        self.set_camera_orientation(theta=self.ROLLED_CAMERA["theta"] - TAU)
        self.play(FadeIn(cloud), run_time=1.5)
        self.add(cloud)
        self.marked_next_slide()
        short = self.short_pairs(self.N_SHORT)
        long_ = self.long_pairs(self.N_RANDOM, self.N_ENDS)
        e_short = self.stress_edges(short, width=0.5)
        e_long = self.stress_edges(long_, width=1.0)
        bar = self.stress_bar()
        self.add_fixed_in_frame_mobjects(bar)

        head = self.swap_heading(
            head,
            "Track some distances",
            "Many short distances, and a few long ones",
            # FadeIn(cloud),
            FadeIn(bar),
        )
        # self.add(cloud)
        self.play(Create(e_short), run_time=1.5)
        self.play(Create(e_long), run_time=1.5)
        self.marked_next_slide()
        self.spin_once()

        self.marked_next_slide()
        head = self.swap_heading(
            head,
            "Watch the stress",
            "Colour by how much each distance stretches",
        )
        self.unroll_sequence()
        r_s = self.stress_ratios(short, e_short.lengths0)
        r_l = self.stress_ratios(long_, e_long.lengths0)
        head = self.swap_heading(
            head,
            "Keep the short, give up the long",
            f"{np.mean(np.abs(r_s - 1) < 0.10) * 100:.0f}% of short distances "
            f"change by less than 10%;  {np.mean(r_l > 2) * 100:.0f}% of long "
            "ones more than double",
        )
        self.marked_next_slide()

        # ---- roll back up, keeping each line's stretched colour ----
        e_short.freeze()
        e_long.freeze()
        new_title = SmallText(
            "stretch when unrolled", font_size=16, color=ACCENT_COLOR
        ).move_to(bar[2])
        self.add_fixed_in_frame_mobjects(new_title)
        head = self.swap_heading(
            head,
            "Roll it back up",
            "Keep the stress colouring",
            FadeOut(bar[2]),
            FadeIn(new_title),
        )
        self.remove_fixed_in_frame_mobjects(bar[2])
        self.reroll_sequence()
        head = self.swap_heading(
            head,
            "Why was it good to drop the long distances?",
            "They cut across the gaps between layers where there is no sheet",
        )
        self.marked_next_slide()
        self.spin_once()
        self.marked_next_slide()

        self.new_section("High Dimensions")
