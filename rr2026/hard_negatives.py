"""Class 2: hard negatives from random projections.

    python assets_negatives.py            # once (add --synthetic to test offline)
    manim-slides render hard_negatives.py HardNegatives
"""

from manim import *

import json
import sys
from pathlib import Path

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

apply_defaults()

from umap_talk_common import *

apply_umap_defaults()

ASSETS = Path(__file__).parent / "assets"

TOY_BOX = dict(width=8.8, height=5.4, center=(-2.1, -0.15))
COLUMN_TOP_LEFT = (2.95, 2.45)
STRIP_LENGTH = 8.4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def to3(p):
    return np.array([p[0], p[1], 0.0])


def caption(text, font_size=22, color=DEFAULT_COLOR, **kwargs):
    return crisp_text(text, font_size=font_size, color=color, **kwargs)


def caption_stack(lines, font_size=22, buff=0.1, color=DEFAULT_COLOR):
    group = VGroup(*[caption(line, font_size, color=color) for line in lines])
    return group.arrange(DOWN, aligned_edge=LEFT, buff=buff)


class Column:
    """Stacks captions top-down in the right-hand column."""

    def __init__(self, top_left=COLUMN_TOP_LEFT):
        self.top_left = to3(top_left)
        self.items = []

    def place(self, mob, buff=0.32):
        if not self.items:
            mob.move_to(self.top_left, aligned_edge=UL)
        else:
            mob.next_to(self.items[-1], DOWN, buff=buff, aligned_edge=LEFT)
        self.items.append(mob)
        return mob


def fit_box(points, width, height, center):
    """(mid, scale, map) placing data points in a width x height box."""
    lo, hi = points.min(axis=0), points.max(axis=0)
    mid, span = 0.5 * (lo + hi), np.maximum(hi - lo, 1e-9)
    scale = float(min(width / span[0], height / span[1]))
    center = np.asarray(center, dtype=float)
    return mid, scale, (lambda p: center + (np.asarray(p) - mid) * scale)


def direction(angle):
    return np.array([np.cos(angle), np.sin(angle)])


def window_for(points, source, angle, size):
    """Rank-centred negative-selection window, as in the 0.6 kernels."""
    t = (points - points[source]) @ direction(angle)
    order = np.argsort(t, kind="stable")
    rank = np.empty(len(points), dtype=np.int64)
    rank[order] = np.arange(len(points))
    start = int(np.clip(rank[source] - size // 2, 0, len(points) - size))
    window = order[start : start + size]
    return window, t, t[window].min(), t[window].max()


def slab_band(source_pt, angle, t_lo, t_hi, clip, opacity=0.16):
    u = direction(angle)
    v = np.array([-u[1], u[0]])
    far = 30.0
    corners = [
        source_pt + t_lo * u - far * v,
        source_pt + t_hi * u - far * v,
        source_pt + t_hi * u + far * v,
        source_pt + t_lo * u + far * v,
    ]
    band = Polygon(*[to3(c) for c in corners])
    return Intersection(
        band, clip, fill_color=SECONDARY_COLOR, fill_opacity=opacity, stroke_width=0
    )


def line_in_box(point, u, center, width, height):
    """End points of the line through ``point`` along ``u`` inside the box."""
    lo = np.asarray(center) - [width / 2, height / 2]
    hi = np.asarray(center) + [width / 2, height / 2]
    t_min, t_max = -np.inf, np.inf
    for k in range(2):
        if abs(u[k]) > 1e-9:
            a, b = (lo[k] - point[k]) / u[k], (hi[k] - point[k]) / u[k]
            t_min, t_max = max(t_min, min(a, b)), min(t_max, max(a, b))
    return point + t_min * u, point + t_max * u


def source_marker(point, radius=0.11):
    return Dot(
        to3(point),
        radius=radius,
        color=SOURCE_COLOR,
        stroke_color=STRUCTURE_COLOR,
        stroke_width=2.5,
    )


def negative_lines(source_pt, targets, forces):
    """Source-to-negative lines, opacity showing the force each exerts."""
    group = VGroup()
    for p, f in zip(targets, forces):
        opacity = float(np.clip(f, 0.12, 1.0))
        group.add(
            VGroup(
                Line(to3(source_pt), to3(p), color=REPEL_COLOR, stroke_width=3,
                     stroke_opacity=opacity),
                Dot(to3(p), radius=0.05, color=REPEL_COLOR, fill_opacity=opacity),
            )
        )
    return group


def force_of(distances):
    return soft_clip(repulsion_magnitude(np.asarray(distances, dtype=float)))


def polyline(axes, xs, ys, color, width=4):
    return VMobject(color=color, stroke_width=width).set_points_as_corners(
        [axes.c2p(x, y) for x, y in zip(xs, ys)]
    )


# ---------------------------------------------------------------------------
# The slide class
# ---------------------------------------------------------------------------
class HardNegatives(UMAPSlide):
    def construct(self):
        self.data = np.load(ASSETS / "negatives.npz")
        self.meta = json.loads((ASSETS / "negatives.json").read_text())
        self.setup_toy()

        self.end_section_wipe(
            SECTION_TITLES["negatives"],
            next_slide_prep=lambda: None,
            notes=NOTES["s01_sum"],
        )
        self.stage_uniform()
        self.stage_ideal()
        self.stage_trick()
        self.stage_geometry()
        self.stage_shape()
        self.stage_calibration()
        self.stage_mnist()
        self.stage_takeaway()

    # -- toy world ------------------------------------------------------------
    def setup_toy(self):
        d = self.data
        Y = d["toy_Y"].astype(float)
        self.toy_labels = d["toy_labels"]
        _, self.toy_scale, to_scene = fit_box(Y, **TOY_BOX)
        self.toy_scene = np.array([to_scene(p) for p in Y])
        self.toy_data = Y
        self.src = int(d["toy_source"])
        self.angle = float(d["toy_angle"])
        self.window_size = len(d["toy_window"])
        box = TOY_BOX
        self.toy_clip = Rectangle(
            width=box["width"] + 0.3, height=box["height"] + 0.3
        ).move_to(to3(box["center"]))

    def make_toy_cloud(self, keyframes=None):
        kf = self.toy_scene[None] if keyframes is None else keyframes
        return EmbeddingCloud(
            kf,
            labels_to_rgb(self.toy_labels, digit_rgb()),
            fit="none",
            point_px=7.0,
        )

    # -- stage 1: uniform negatives are wasted ----------------------------------
    def stage_uniform(self):
        d = self.data
        self.set_title("Repulsion is a sum over every point")
        self.cloud = self.make_toy_cloud()
        self.add(self.cloud)
        self.play(PMFadeIn(self.cloud), run_time=1.2)
        s_pt = self.toy_scene[self.src]
        self.marker = source_marker(s_pt)
        self.play(FadeIn(self.marker, scale=1.6))

        negs = d["toy_uniform_neg"]
        dists = np.linalg.norm(self.toy_data[negs] - self.toy_data[self.src], axis=1)
        forces = force_of(dists)
        lines = negative_lines(s_pt, self.toy_scene[negs], forces)
        self.play(PMDim(self.cloud, 0.6))
        self.bring_to_front(self.marker)
        self.play(LaggedStart(*[Create(l) for l in lines], lag_ratio=0.08), run_time=1.5)
        self.bring_to_front(self.marker)

        chart = styled_axes(
            x_range=[0, 20, 5],
            y_range=[0, 1, 0.25],
            x_label="distance",
            y_label="repulsive force",
            x_length=3.8,
            y_length=2.3,
            y_decimal_places=2,
        )
        chart.move_to(to3((4.95, 1.05)))
        axes = chart[0]
        axes.x_axis.add_numbers([5, 10, 15, 20], font_size=16)
        xs = np.geomspace(0.01, 20, 300)
        curve = polyline(axes, xs, force_of(xs), REPEL_COLOR, 3)
        dots = VGroup(
            *[Dot(axes.c2p(x, y), radius=0.06, color=REPEL_COLOR) for x, y in zip(dists, forces)]
        )
        note = caption_stack(["Most uniform draws land far", "away and barely push at all."])
        note.next_to(chart, DOWN, buff=0.35).align_to(chart, LEFT)
        self.play(FadeIn(chart), Create(curve))
        self.play(LaggedStart(*[FadeIn(dt, scale=1.5) for dt in dots], lag_ratio=0.05))
        self.play(FadeIn(note))
        self.stage1 = VGroup(lines, chart, curve, dots, note)
        self.marked_next_slide(notes=NOTES["s02_ideal"])

    # -- stage 2: the ideal, and why it is out of reach ---------------------------
    def stage_ideal(self):
        self.play(FadeOut(self.stage1))
        self.set_title("The ideal: sample by force")
        s_pt = self.toy_scene[self.src]
        grid = np.geomspace(0.05, 10, 400)
        f = force_of(grid)
        halo = VGroup()
        for level in (0.03, 0.07, 0.15, 0.3, 0.5, 0.8):
            radius = grid[np.flatnonzero(f >= level).max()] * self.toy_scale
            halo.add(
                Circle(radius=radius, stroke_width=0, fill_color=REPEL_COLOR,
                       fill_opacity=0.08).move_to(to3(s_pt))
            )
        col = Column()
        formula = col.place(MathTex(r"q^\ast(j)\;\propto\;f(d_{ij})", font_size=44))
        explain = col.place(
            caption_stack(["Draw each point in proportion", "to the force it exerts."]),
            buff=0.25,
        )
        self.play(FadeIn(halo, lag_ratio=0.2), run_time=1.5)
        self.bring_to_front(self.marker)
        self.play(Write(formula))
        self.play(FadeIn(explain))
        self.marked_next_slide(notes=NOTES["s03_cost"])

        problem = caption_stack(
            ["Needs a nearest-neighbour", "search in an embedding", "that moves every epoch."],
            font_size=20,
        )
        frame = SurroundingRectangle(problem, buff=0.2, color=SECONDARY_COLOR, stroke_width=1.5)
        boxed = col.place(VGroup(frame, problem), buff=0.5)
        cross = Cross(boxed, stroke_color=REPEL_COLOR, stroke_width=5)
        self.play(FadeIn(boxed))
        self.play(Create(cross))
        self.stage2 = VGroup(halo, formula, explain, boxed, cross)
        self.marked_next_slide(notes=NOTES["s04_project"])

    # -- stage 3: project, sort, window ------------------------------------------
    def stage_trick(self):
        self.play(FadeOut(self.stage2))
        self.set_title("The trick: project, sort, take a window")
        S, s = self.toy_scene, self.src
        s_pt = S[s]
        u = direction(self.angle)
        window, t, t_lo, t_hi = window_for(S, s, self.angle, self.window_size)

        # Keyframes: 2D layout -> on the direction line -> sorted strip.
        k_line = s_pt + t[:, None] * u
        # Sorting: lay the points out evenly by rank along a horizontal strip.
        order = np.argsort(t, kind="stable")
        rank = np.empty(len(S), dtype=np.int64)
        rank[order] = np.arange(len(S))
        strip_center = np.array(TOY_BOX["center"])
        step = STRIP_LENGTH / (len(S) - 1)
        strip_left = strip_center[0] - STRIP_LENGTH / 2
        jitter = np.random.default_rng(0).normal(scale=0.045, size=len(S))
        k_strip = np.c_[strip_left + rank * step, strip_center[1] + jitter]
        keyframes = np.stack([S, k_line, k_strip])

        cloud = self.make_toy_cloud(keyframes)
        cloud.rgbas[:, :3] = self.cloud.rgbas[:, :3]
        cloud.base_rgb = self.cloud.base_rgb.copy()
        self.remove(self.cloud)
        self.add(cloud)
        self.bring_to_front(self.marker)
        self.cloud = cloud
        frame = ValueTracker(0.0)
        cloud.track(frame)

        col = Column()
        steps = [
            caption_stack(["1. Project every point onto", "a random direction."]),
            caption_stack(["2. Sort by projection."]),
            caption_stack(["3. Take the R points around", "the source's rank."]),
            caption_stack(["Back in 2D, the window is", "a slab through the source."]),
        ]
        for mob in steps:
            col.place(mob)

        a_end, b_end = line_in_box(s_pt, u, TOY_BOX["center"], TOY_BOX["width"], TOY_BOX["height"])
        dir_line = DashedLine(
            to3(a_end), to3(b_end), color=STRUCTURE_COLOR, stroke_width=2.5, dash_length=0.12,
        )
        self.play(Create(dir_line), FadeIn(steps[0]))
        self.play(frame.animate.set_value(1), run_time=2)
        self.marked_next_slide(notes=NOTES["s05_sort"])

        self.play(FadeOut(dir_line), FadeOut(self.marker))
        self.play(frame.animate.set_value(2), FadeIn(steps[1]), run_time=2)
        strip_src = k_strip[s]
        strip_marker = source_marker(strip_src)
        self.play(FadeIn(strip_marker, scale=1.6))
        self.marked_next_slide(notes=NOTES["s06_window"])

        left_x = strip_left + rank[window].min() * step
        right_x = strip_left + rank[window].max() * step
        brace = Brace(
            Line(to3((left_x, strip_center[1] - 0.12)), to3((right_x, strip_center[1] - 0.12))),
            DOWN, color=STRUCTURE_COLOR,
        )
        brace_label = caption("R points around the source's rank", 20).next_to(brace, DOWN, buff=0.1)
        self.play(
            PMRecolor(cloud, cloud.target_rgb(window, STRUCTURE_COLOR, dim_rest=0.78)),
            GrowFromCenter(brace),
            FadeIn(brace_label),
            FadeIn(steps[2]),
        )
        self.marked_next_slide(notes=NOTES["s07_slab"])

        self.play(FadeOut(brace), FadeOut(brace_label), FadeOut(strip_marker))
        self.play(frame.animate.set_value(1), run_time=1.5)
        self.play(frame.animate.set_value(0), run_time=1.5)
        self.marker = source_marker(s_pt)
        band = slab_band(s_pt, self.angle, t_lo, t_hi, self.toy_clip)
        self.add(band)
        self.bring_to_front(cloud)
        self.play(FadeIn(band), FadeIn(self.marker, scale=1.6), FadeIn(steps[3]))
        self.bring_to_front(self.marker)
        self.marked_next_slide(notes=NOTES["s08_closer"])

        # Clear the step captions completely before drawing in their place.
        self.play(FadeOut(VGroup(*steps)))
        negs = self.data["toy_window_neg"]
        uni = self.data["toy_uniform_neg"]
        data_src = self.toy_data[s]
        d_win = np.linalg.norm(self.toy_data[negs] - data_src, axis=1)
        d_uni = np.linalg.norm(self.toy_data[uni] - data_src, axis=1)
        lines = negative_lines(s_pt, S[negs], force_of(d_win))

        chart = styled_axes(
            x_range=[0, 15, 5], y_range=[0, 1, 0.25],
            x_label="distance", y_label="repulsive force",
            x_length=3.6, y_length=2.2, y_decimal_places=2,
        )
        chart.move_to(to3((4.95, 1.0)))
        axes = chart[0]
        axes.x_axis.add_numbers([5, 10, 15], font_size=16)
        xs = np.geomspace(0.01, 15, 300)
        curve = polyline(axes, xs, force_of(xs), REPEL_COLOR, 3)
        uni_dots = VGroup(*[Dot(axes.c2p(min(x, 15), y), radius=0.055, color=SECONDARY_COLOR)
                            for x, y in zip(d_uni, force_of(d_uni))])
        win_dots = VGroup(*[Dot(axes.c2p(min(x, 15), y), radius=0.065, color=REPEL_COLOR)
                            for x, y in zip(d_win, force_of(d_win))])
        key = VGroup(
            VGroup(Dot(radius=0.055, color=SECONDARY_COLOR), caption("uniform draws", 18)).arrange(RIGHT, buff=0.12),
            VGroup(Dot(radius=0.065, color=REPEL_COLOR), caption("slab draws", 18)).arrange(RIGHT, buff=0.12),
        ).arrange(RIGHT, buff=0.4).next_to(chart, DOWN, buff=0.25).align_to(chart, LEFT)
        pop_win = force_of(np.linalg.norm(self.toy_data[window] - data_src, axis=1)[window != s]).mean()
        pop_uni = force_of(np.delete(np.linalg.norm(self.toy_data - data_src, axis=1), s)).mean()
        closer = caption_stack(["Slab negatives are closer:",
                                f"on average they push ×{pop_win / pop_uni:.0f} harder."])
        closer.next_to(key, DOWN, buff=0.3).align_to(key, LEFT)

        self.play(FadeIn(chart), Create(curve))
        self.play(FadeIn(uni_dots, lag_ratio=0.05), FadeIn(key[0]))
        self.play(LaggedStart(*[Create(l) for l in lines], lag_ratio=0.06),
                  FadeIn(win_dots, lag_ratio=0.05), FadeIn(key[1]), run_time=1.5)
        self.bring_to_front(self.marker)
        self.play(FadeIn(closer))
        self.marked_next_slide(notes=NOTES["s09_epoch"], auto_next=True)

        # A fresh direction every epoch: clear the chart, then say so.
        self.play(FadeOut(VGroup(lines, chart, curve, uni_dots, win_dots, key, closer)))
        every = caption_stack(["A new random direction", "every epoch."])
        every.move_to(to3(COLUMN_TOP_LEFT), aligned_edge=UL)
        self.play(FadeIn(every))
        theta = ValueTracker(self.angle)
        cloud.clear_updaters()
        cloud.set_frame(0)

        def recolor(m):
            win, *_ = window_for(S, s, theta.get_value(), self.window_size)
            m.rgbas[:, :3] = m.target_rgb(win, STRUCTURE_COLOR, dim_rest=0.78)

        def draw_band():
            win, _, lo, hi = window_for(S, s, theta.get_value(), self.window_size)
            return slab_band(s_pt, theta.get_value(), lo, hi, self.toy_clip)

        self.remove(band)
        live_band = always_redraw(draw_band)
        self.add(live_band)
        cloud.add_updater(recolor)
        self.bring_to_front(cloud, self.marker)
        self.start_loop(notes=NOTES["s10_loop"])
        self.play(theta.animate.set_value(self.angle + PI), run_time=5, rate_func=linear)
        self.end_loop(notes=NOTES["s11_geometry"])
        cloud.clear_updaters()
        live_band.clear_updaters()

    # -- stage 4: the arc formula ------------------------------------------------
    def stage_geometry(self):
        self.clear_slide()
        self.set_title("Why a slab picks the right points")
        w = 1.0
        src = np.array([-3.6, -0.3])
        half_h = 2.65
        band = Rectangle(
            width=w, height=2 * half_h, stroke_width=0, fill_color=SECONDARY_COLOR, fill_opacity=0.18
        ).move_to(to3(src))
        edges = VGroup(
            *[
                DashedLine(to3(src + [sx * w / 2, -half_h]), to3(src + [sx * w / 2, half_h]),
                           color=SECONDARY_COLOR, stroke_width=2, dash_length=0.1)
                for sx in (-1, 1)
            ]
        )
        w_brace = BraceBetweenPoints(to3(src + [-w / 2, -half_h]), to3(src + [w / 2, -half_h]), DOWN)
        w_label = MathTex("w", font_size=32).next_to(w_brace, DOWN, buff=0.05)
        marker = source_marker(src)

        chart = styled_axes(
            x_range=[0, 3, 0.5],
            y_range=[0, 1, 0.25],
            x_label="distance r  (in slab widths)",
            y_label="chance of being in slab",
            x_length=4.6,
            y_length=2.9,
            y_decimal_places=2,
        )
        chart.move_to(to3((3.75, 0.85)))
        axes = chart[0]
        axes.x_axis.add_numbers([0.5, 1, 1.5, 2, 2.5], font_size=16)

        r = ValueTracker(0.25)

        def arcs():
            radius = r.get_value()
            ring = Circle(radius=radius, color=SECONDARY_COLOR, stroke_width=1.5).move_to(to3(src))
            if radius <= w / 2:
                inside = Circle(radius=radius, color=STRUCTURE_COLOR, stroke_width=5).move_to(to3(src))
                return VGroup(ring, inside)
            a0 = np.arccos(w / (2 * radius))
            span = PI - 2 * a0
            top = Arc(radius=radius, start_angle=a0, angle=span, arc_center=to3(src),
                      color=STRUCTURE_COLOR, stroke_width=5)
            bottom = Arc(radius=radius, start_angle=PI + a0, angle=span, arc_center=to3(src),
                         color=STRUCTURE_COLOR, stroke_width=5)
            return VGroup(ring, top, bottom)

        def trace():
            xs = np.linspace(0.02, r.get_value() / w, 200)
            ys = slab_probability(xs, 1.0)
            return polyline(axes, xs, ys, STRUCTURE_COLOR, 4)

        def dot():
            x = r.get_value() / w
            return Dot(axes.c2p(x, slab_probability(x, 1.0)), radius=0.07, color=SOURCE_COLOR,
                       stroke_color=STRUCTURE_COLOR, stroke_width=2)

        circle = always_redraw(arcs)
        curve = always_redraw(trace)
        tip = always_redraw(dot)
        explain = caption_stack(
            ["By symmetry: the chance a point at distance r",
             "falls in a random slab is the share of its",
             "circle inside a fixed one."],
            font_size=20,
        )
        explain.next_to(chart, DOWN, buff=0.35).align_to(chart, LEFT)

        self.play(FadeIn(band), Create(edges), GrowFromCenter(w_brace), FadeIn(w_label))
        self.play(FadeIn(marker, scale=1.6))
        self.play(FadeIn(chart))
        self.add(circle, curve, tip)
        self.bring_to_front(marker)
        self.play(FadeIn(explain))
        self.marked_next_slide(notes=NOTES["s12_arc"])
        self.play(r.animate.set_value(2.55), run_time=6, rate_func=linear)
        for mob in (circle, curve, tip):
            mob.clear_updaters()

        formula = MathTex(
            r"P(r)\;=\;\frac{2}{\pi}\arcsin\frac{w}{2r}\;\approx\;\frac{w}{\pi r}",
            font_size=40,
        )
        formula.next_to(explain, DOWN, buff=0.35).align_to(explain, LEFT)
        self.play(Write(formula))
        self.marked_next_slide(notes=NOTES["s13_shape"])

    # -- stage 5: why 1/r is the right shape ------------------------------------
    def stage_shape(self):
        self.clear_slide()
        self.set_title("Why that shape is right")
        w = 2.0  # slab width in embedding units, for these illustrative curves
        xs = np.linspace(0.005, 6, 600)
        uniform = xs / xs.max()
        slab = 4 * xs * np.arcsin(np.minimum(1.0, w / (2 * xs)))
        slab = slab / slab.max()
        ideal = xs * force_of(xs)
        ideal = ideal / ideal.max()

        left = styled_axes(
            x_range=[0, 6, 1], y_range=[0, 1.1, 0.25],
            x_label="distance from source",
            y_label="samples per unit distance",
            x_length=5.0, y_length=3.2, y_decimal_places=2,
        )
        left.move_to(to3((-3.35, 0.6)))
        la = left[0]
        la.x_axis.add_numbers([1, 2, 3, 4, 5], font_size=16)
        shade = Polygon(
            *[la.c2p(x, y) for x, y in zip(xs, ideal)], la.c2p(xs[-1], 0), la.c2p(xs[0], 0),
            stroke_width=0, fill_color=REPEL_COLOR, fill_opacity=0.15,
        )
        c_uniform = DashedVMobject(polyline(la, xs, uniform, SECONDARY_COLOR, 3), num_dashes=40)
        c_slab = polyline(la, xs, slab, STRUCTURE_COLOR, 4)
        c_ideal = polyline(la, xs, ideal, REPEL_COLOR, 4)
        def at(x, ys):
            return la.c2p(x, ys[int(np.argmin(np.abs(xs - x)))])

        t_uniform = VGroup(caption("uniform", 18, color=SECONDARY_COLOR),
                           MathTex(r"\propto r", font_size=28, color=SECONDARY_COLOR)
                           ).arrange(RIGHT, buff=0.12).next_to(at(5.5, uniform), UL, buff=0.1)
        t_slab = caption("slab: flat", 18, color=STRUCTURE_COLOR).next_to(
            at(4.6, slab), DOWN, buff=0.12)
        peak_x = xs[int(np.argmax(ideal))]
        t_ideal = VGroup(caption("ideal", 18, color=REPEL_COLOR),
                         MathTex(r"\propto r\,f(r)", font_size=28, color=REPEL_COLOR)
                         ).arrange(RIGHT, buff=0.12).next_to(la.c2p(peak_x, 1.0), RIGHT, buff=0.35)
        left_note = caption_stack(
            ["The slab sits between the two:", "far fewer distant samples than", "uniform, heavier tail than ideal."],
            font_size=20,
        ).next_to(left, DOWN, buff=0.3).align_to(left, LEFT)

        self.play(FadeIn(left))
        self.play(Create(c_uniform), FadeIn(t_uniform))
        self.play(FadeIn(shade), Create(c_ideal), FadeIn(t_ideal))
        self.play(Create(c_slab), FadeIn(t_slab))
        self.play(FadeIn(left_note))
        self.marked_next_slide(notes=NOTES["s14_weights"])

        weight = force_of(xs) / slab_probability(xs, w)
        y_top = max(1.25, np.ceil(weight.max() * 4) / 4 + 0.25)
        right = styled_axes(
            x_range=[0, 6, 1], y_range=[0, y_top, 0.25],
            x_label="distance from source",
            y_label="importance weight  f / P",
            x_length=5.0, y_length=3.2, y_decimal_places=2,
        )
        right.move_to(to3((3.55, 0.6)))
        ra = right[0]
        ra.x_axis.add_numbers([1, 2, 3, 4, 5], font_size=16)
        c_weight = polyline(ra, xs, weight, STRUCTURE_COLOR, 4)
        cap = DashedLine(ra.c2p(0, 1), ra.c2p(6, 1), color=REPEL_COLOR, stroke_width=2, dash_length=0.1)
        cap_label = MathTex(r"\gamma", font_size=30, color=REPEL_COLOR).next_to(ra.c2p(6, 1), RIGHT, buff=0.1)
        right_note = caption_stack(
            ["Bounded everywhere: the soft clip", "caps it at γ inside the slab, and it", "falls away beyond."],
            font_size=20,
        ).next_to(right, DOWN, buff=0.3).align_to(right, LEFT)
        cost = caption("Cost: one sort per epoch.", 24, color=STRUCTURE_COLOR)
        cost.to_edge(DOWN, buff=0.35)

        self.play(FadeIn(right))
        self.play(Create(cap), FadeIn(cap_label))
        self.play(Create(c_weight), run_time=1.5)
        self.play(FadeIn(right_note))
        self.play(FadeIn(cost))
        self.marked_next_slide(notes=NOTES["s15_balance"])

    # -- stage 6: calibration ----------------------------------------------------
    def stage_calibration(self):
        self.clear_slide()
        self.set_title("Keeping repulsion in balance")
        pivot = np.array([-2.6, -0.2])
        half = 2.6
        scale = ValueTracker(1.0)  # global repulsion scale, 1 = untouched
        settled = 0.4

        def tilt():
            return np.radians(11.0) * (scale.get_value() - settled) / (1.0 - settled)

        def ends():
            a = tilt()
            offset = half * np.array([np.cos(a), np.sin(a)])  # left end dips for a > 0
            return pivot - offset, pivot + offset

        def draw_balance():
            left, right = ends()
            beam = Line(to3(left), to3(right), color=STRUCTURE_COLOR, stroke_width=6)
            parts = [beam]
            for end in (left, right):
                hang = Line(to3(end), to3(end + [0, -1.1]), color=SECONDARY_COLOR, stroke_width=2)
                pan = Line(to3(end + [-0.75, -1.1]), to3(end + [0.75, -1.1]),
                           color=STRUCTURE_COLOR, stroke_width=6)
                parts += [hang, pan]
            return VGroup(*parts)

        stand = VGroup(
            Triangle(color=STRUCTURE_COLOR, fill_opacity=1).scale(0.25).move_to(to3(pivot + [0, -0.22])),
            Line(to3(pivot + [0, -0.4]), to3(pivot + [0, -2.6]), color=STRUCTURE_COLOR, stroke_width=5),
            Line(to3(pivot + [-0.9, -2.6]), to3(pivot + [0.9, -2.6]), color=STRUCTURE_COLOR, stroke_width=5),
        )
        balance = always_redraw(draw_balance)

        def pan_label(side, text, color):
            def draw():
                end = ends()[side]
                return caption(text, 20, color=color).move_to(to3(end + [0, -1.45]))
            return always_redraw(draw)

        left_label = pan_label(0, "slab negatives", STRUCTURE_COLOR)
        right_label = pan_label(1, "all points", SECONDARY_COLOR)

        slider_line = Line(to3((-4.8, -3.5)), to3((-0.4, -3.5)), color=SECONDARY_COLOR, stroke_width=3)
        knob = always_redraw(
            lambda: Dot(slider_line.point_from_proportion(scale.get_value()),
                        radius=0.12, color=SOURCE_COLOR, stroke_color=STRUCTURE_COLOR, stroke_width=2)
        )
        slider_label = caption("global scale on slab repulsion", 18).next_to(slider_line, UP, buff=0.15)

        col = Column((2.3, 2.3))
        notes_a = col.place(caption_stack(["Nearby samples push harder,", "so the total comes out too big."]))
        notes_b = col.place(caption_stack(["One global factor scales it back,",
                                           "adjusted gently in the first half", "of training."]))
        notes_c = col.place(caption_stack(["Where the repulsion lands stays",
                                           "local, by design. That matters", "at the end of the talk."]))

        self.play(FadeIn(stand), FadeIn(balance), FadeIn(left_label), FadeIn(right_label))
        self.play(FadeIn(notes_a))
        self.marked_next_slide(notes=NOTES["s16_scale"])
        self.play(Create(slider_line), FadeIn(slider_label), FadeIn(knob))
        self.play(scale.animate.set_value(settled), FadeIn(notes_b), run_time=2.5)
        self.play(FadeIn(notes_c))
        for mob in (balance, knob, left_label, right_label):
            mob.clear_updaters()
        self.marked_next_slide(notes=NOTES["s17_mnist"])

    # -- stage 7: MNIST ------------------------------------------------------------
    def stage_mnist(self):
        self.clear_slide()
        title = "Checking it on MNIST"
        if self.meta.get("synthetic"):
            title += " (stand-in data)"
        self.set_title(title)
        d, meta = self.data, self.meta
        Y = d["mnist_Y"].astype(float)
        labels = d["mnist_labels"]
        box = dict(width=7.4, height=5.6, center=(-2.4, -0.35))
        _, _, to_scene = fit_box(Y, **box)
        P = to_scene(Y)
        cloud = EmbeddingCloud(P, labels_to_rgb(labels, digit_rgb()), fit="none", point_px=3.0)
        src = int(meta["example_source"])
        angle = float(meta["example_angle"])
        window, _, lo, hi = window_for(P, src, angle, int(meta["window"]))
        clip = Rectangle(width=box["width"] + 0.3, height=box["height"] + 0.3).move_to(to3(box["center"]))
        band = slab_band(P[src], angle, lo, hi, clip, opacity=0.12)
        marker = source_marker(P[src], radius=0.12)

        col = Column((2.6, 2.3))
        facts = col.place(caption_stack([f"{meta['n']:,} points;", f"window of {meta['window']:,}."], font_size=24))
        param = col.place(crisp_text(f"negative_selection_range={meta['window']}", font=MONO_FONT, font_size=18))

        self.add(cloud)
        self.play(PMFadeIn(cloud), FadeIn(facts), run_time=1.5)
        self.add(band)
        self.bring_to_front(cloud)
        dimmed = cloud.target_rgb(window, np.array([0.0, 0.0, 0.0]), dim_rest=0.8)
        keep = cloud._rank[window]
        dimmed[keep] = cloud.base_rgb[keep]
        self.play(FadeIn(band), PMRecolor(cloud, dimmed), FadeIn(marker, scale=1.6), FadeIn(param))
        self.marked_next_slide(notes=NOTES["s18_hit"])

        self.play(PMFadeOut(cloud), FadeOut(band), FadeOut(marker), FadeOut(facts), FadeOut(param))

        r_max = float(np.ceil(meta["r_max"] / 5.0) * 5.0)
        hit = styled_axes(
            x_range=[0, r_max, 5], y_range=[0, 1, 0.25],
            x_label="distance from source", y_label="chance of being in window",
            x_length=4.8, y_length=3.0, y_decimal_places=2,
        )
        hit.move_to(to3((-3.4, 0.75)))
        ha = hit[0]
        ha.x_axis.add_numbers(list(np.arange(5, r_max + 1, 5)), font_size=16)
        pred = polyline(ha, d["hit_r"], d["hit_pred"], REPEL_COLOR, 3)
        measured = VGroup(*[Dot(ha.c2p(x, y), radius=0.045, color=STRUCTURE_COLOR)
                            for x, y in zip(d["hit_r"], d["hit_rate"])])
        legend = VGroup(
            VGroup(Dot(radius=0.06, color=STRUCTURE_COLOR), caption("measured", 18)).arrange(RIGHT, buff=0.15),
            VGroup(Line(ORIGIN, RIGHT * 0.4, color=REPEL_COLOR, stroke_width=3),
                   caption("arc formula, each window's extent", 18)).arrange(RIGHT, buff=0.15),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        legend.next_to(hit, DOWN, buff=0.3).align_to(hit, LEFT)

        hist = styled_axes(
            x_range=[0, r_max, 5], y_range=[0, 1, 0.25],
            x_label="distance to negative", y_label="share of samples",
            x_length=4.8, y_length=3.0, y_decimal_places=2,
        )
        hist.move_to(to3((3.5, 0.75)))
        hb = hist[0]
        hb.x_axis.add_numbers(list(np.arange(5, r_max + 1, 5)), font_size=16)
        edges = np.linspace(0, r_max, 31)

        def step_curve(values, color, width):
            counts, _ = np.histogram(values, bins=edges)
            share = counts / counts.sum()
            peak = max(np.histogram(d["dist_uniform"], bins=edges)[0].max() / len(d["dist_uniform"]),
                       np.histogram(d["dist_window"], bins=edges)[0].max() / len(d["dist_window"]))
            share = share / peak * 0.95
            pts = []
            for a, b, h in zip(edges[:-1], edges[1:], share):
                pts += [hb.c2p(a, h), hb.c2p(b, h)]
            return VMobject(color=color, stroke_width=width).set_points_as_corners(
                [hb.c2p(0, 0)] + pts + [hb.c2p(edges[-1], 0)]
            )

        h_uni = step_curve(d["dist_uniform"], SECONDARY_COLOR, 3)
        h_win = step_curve(d["dist_window"], STRUCTURE_COLOR, 4)
        hist_legend = VGroup(
            VGroup(Line(ORIGIN, RIGHT * 0.4, color=SECONDARY_COLOR, stroke_width=3),
                   caption("uniform negatives", 18)).arrange(RIGHT, buff=0.15),
            VGroup(Line(ORIGIN, RIGHT * 0.4, color=STRUCTURE_COLOR, stroke_width=4),
                   caption("window negatives", 18)).arrange(RIGHT, buff=0.15),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        hist_legend.next_to(hist, DOWN, buff=0.3).align_to(hist, LEFT)

        same_u, same_w = meta["same_label_uniform"], meta["same_label_window"]
        ratio = meta["mean_force_window"] / meta["mean_force_uniform"]
        readouts = VGroup(
            caption(f"same digit as the source: {same_u:.0%} → {same_w:.0%}", 22, color=STRUCTURE_COLOR),
            caption(f"average push per sample: ×{ratio:.1f}", 22, color=STRUCTURE_COLOR),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        readouts.next_to(hist_legend, DOWN, buff=0.3).align_to(hist_legend, LEFT)

        self.play(FadeIn(hit))
        self.play(Create(pred), FadeIn(legend[1]))
        self.play(LaggedStart(*[FadeIn(m) for m in measured], lag_ratio=0.03), FadeIn(legend[0]))
        self.marked_next_slide(notes=NOTES["s19_hist"])
        self.play(FadeIn(hist))
        self.play(Create(h_uni), Create(h_win), FadeIn(hist_legend))
        self.play(FadeIn(readouts))
        self.marked_next_slide(notes=NOTES["s20_takeaway"])

    # -- takeaway -------------------------------------------------------------
    def stage_takeaway(self):
        self.clear_slide()
        self.takeaway("A random projection and a sort", "buy you hard negatives.")
        self.start_section_wipe(SECTION_TITLES["recursive"], auto_next=True)


# ---------------------------------------------------------------------------
# Speaker notes
# ---------------------------------------------------------------------------
NOTES = {
    "s01_sum": (
        "The repulsive part of each point's gradient is a sum over every other "
        "point, and UMAP estimates it with a few negative samples drawn "
        "uniformly. But repulsion falls off steeply with distance, roughly as "
        "the power 2.8, so most uniform draws land far away and push with "
        "essentially no force. Most of the sampling effort is wasted."
    ),
    "s02_ideal": (
        "Importance sampling says the variance-minimising choice is to draw each "
        "point in proportion to the force it exerts, which here means nearby points."
    ),
    "s03_cost": (
        "But finding nearby points means a nearest-neighbour search, and the "
        "embedding moves every epoch. We need something far cheaper."
    ),
    "s04_project": "Pick a random direction, and project every point onto it.",
    "s05_sort": "Sort the points by that projection.",
    "s06_window": (
        "Take a window of R points around the source's position in that order. "
        "The window is a count of points, not a distance."
    ),
    "s07_slab": (
        "Lift it back into two dimensions, and the window is a slab through the "
        "source: every nearby point, plus some distant ones along the slab."
    ),
    "s08_closer": (
        "Negatives drawn from the slab are much closer to the source, and push "
        "much harder."
    ),
    "s09_epoch": "And every epoch uses a fresh random direction.",
    "s10_loop": (
        "So the slab sweeps around the source, and no direction is favoured. "
        "Press on when ready."
    ),
    "s11_geometry": (
        "Why does a slab pick the right points? Take a point at distance r. The "
        "direction is random, so by symmetry, the chance that point lands in the "
        "slab equals the share of its circle that lies inside a fixed slab."
    ),
    "s12_arc": (
        "Inside half a slab width, that share is one: certain. Beyond it, only "
        "two arcs remain inside, and they shrink as the circle grows. The share "
        "is two over pi, arcsine of w over 2r: about w over pi r. Slab sampling "
        "favours nearby points roughly as one over distance."
    ),
    "s13_shape": (
        "Count samples per unit of distance. Uniform sampling grows linearly with "
        "distance, because there is more area further out. The ideal falls with "
        "the force. The slab is flat: a strip samples distance evenly, where a "
        "disc samples area evenly. It sits between the two."
    ),
    "s14_weights": (
        "Its tail is heavier than the ideal's, which is what keeps it safe as an "
        "importance sampler: the weight, force over sampling probability, is "
        "capped by the soft clip near the source and falls away beyond it. The "
        "whole cost is one sort per epoch: a random projection is the simplest "
        "locality-sensitive hash."
    ),
    "s15_balance": (
        "One subtlety. The kernel doesn't reweight each sample. Nearby samples "
        "push harder, so left alone the total repulsion would come out too big."
    ),
    "s16_scale": (
        "So it applies one global scale, adjusted gently during the first half of "
        "training by comparing forces from slab negatives with forces from uniform "
        "ones. Where the repulsion lands stays deliberately local, and we'll use "
        "that at the end of the talk."
    ),
    "s17_mnist": (
        "On MNIST, with a window of twenty thousand of the seventy thousand "
        "points, the slab looks like this."
    ),
    "s18_hit": (
        "Measured over many random sources and directions, the chance of landing "
        "in the window tracks the arc formula, computed from each window's actual "
        "extent on either side of the source."
    ),
    "s19_hist": (
        "And the negatives it produces are much closer, far more often the same "
        "digit as the source, and each pushes several times harder."
    ),
    "s20_takeaway": "A random projection and a sort buy you hard negatives.",
}
