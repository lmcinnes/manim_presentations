"""
Intuition slide: why noise orthogonal to the data breaks nearest neighbours.

A circle of N points (radius 1 in data units) with its 1-NN edges. Each
point gets a fixed noise value z_i ~ N(0, 1), applied along the single
direction orthogonal to the circle and scaled by sigma. As sigma grows the
circle is drawn out into a long thin cylinder, and once it is long enough
some points' nearest neighbour is a point on the far side of the circle
that happens to sit at nearly the same height.

(In high dimensions the noise accrues across many orthogonal directions;
here it is all packed into one so it can be seen.)

    manim-slides render circle_noise_intuition.py CircleNoiseIntuition

Display: the noise axis is world x (roughly screen-horizontal after the
camera tilt) and the circle lies in the y-z plane. The whole picture is
rescaled uniformly as the cylinder grows so it stays in frame; uniform
scaling does not change which point is nearest to which.
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

apply_defaults()

import numpy as np

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
N_POINTS = 50  # must be even (diametric partners at i + N/2)
NOISE_SEED = 0
ANGLE_JITTER = 0.35  # fraction of the spacing; breaks 1-NN ties

# Engineered noise (this is an intuition slide, designed to look natural).
# Each entry (i, h, d) makes points i and i + N/2 nearly opposite each other
# (partner angle offset by d radians from exactly opposite) with almost the
# same noise value h (difference up to PAIR_HEIGHT_JITTER). The heights sit
# out in the tails of the other points' noise, so each pair ends up nearly
# alone at its height: once the cylinder is long enough the two become each
# other's nearest neighbours, giving an edge (almost) straight across the
# circle. Directions (~22, 72, 158 degrees) and heights are deliberately
# irregular. With these values and NOISE_SEED the three edges appear at
# sigma ~ 1.3, 1.5 and 2.0, after a few "natural" shortcuts near sigma ~ 1.
DIAMETER_PAIRS = [(3, -2.95, 0.05), (10, 2.85, -0.04), (22, 3.8, 0.07)]
PAIR_HEIGHT_JITTER = 0.03
BULK_STD = 1.1  # other points: N(0, BULK_STD) truncated ...
BULK_CAP = 2.05  # ... to |z| <= BULK_CAP
SIGMA_MAX = 2.2  # in units of the circle radius
SHORTCUT_ANGLE = PI  # 3 * PI / 4  # 1-NN edge spanning more than this = "shortcut"

RADIUS_DISPLAY = 2.0  # scene units at sigma = 0
MAX_LENGTH_DISPLAY = 11.0  # cylinder length is capped at this (uniform rescale)

SWEEP_SECONDS = 16.0
LOOP_SECONDS = 12.0
SPIN_DURING_SWEEP = 0.25  # radians / second, about the cylinder axis

DOT_RADIUS = 0.07
EDGE_WIDTH = 3
SHORTCUT_WIDTH = 6
EDGE_COLOR = DEFAULT_COLOR
SHORTCUT_COLOR = HIGHLIGHT_COLOR
NUMBER_COLOR = DEFAULT_COLOR
GUIDE_COLOR = ACCENT_COLOR  # reference circle and noise "stems"
GUIDE_OPACITY = 0.45
HUD_FONT = 34

# camera: face-on to the circle (looking down the noise axis), then oblique
FACE_ON = dict(phi=90 * DEGREES, theta=0 * DEGREES)
# for the final face-on view: a long focal distance makes the projection
# nearly orthographic, so points at every depth land on the same circle
FAR_FOCAL = 400.0
OBLIQUE = dict(phi=72 * DEGREES, theta=-58 * DEGREES)

# point colours: cyclic map from the palette, in hue order (as in the
# embedding slides, so "colour = position along the loop" carries over)
T_CYCLE = [COLOR_CYCLE[i] for i in (0, 4, 3, 1, 2, 5)]


def cyclic_color(u):
    """u in [0, 1) -> ManimColor interpolated around T_CYCLE."""
    x = (u % 1.0) * len(T_CYCLE)
    i = int(np.floor(x))
    return interpolate_color(T_CYCLE[i], T_CYCLE[(i + 1) % len(T_CYCLE)], x - i)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
class NoisyCircle:
    def __init__(self, n=N_POINTS, seed=NOISE_SEED):
        rng = np.random.default_rng(seed)
        self.n = n
        # evenly spaced angles are all tied between two neighbours, so
        # jitter them slightly to make each 1-NN well defined
        self.theta = (
            2 * np.pi * (np.arange(n) + ANGLE_JITTER * rng.uniform(-1, 1, n)) / n
        )
        # bulk noise: truncated Gaussian
        self.z = np.empty(n)
        for k in range(n):
            while True:
                v = rng.standard_normal() * BULK_STD
                if abs(v) <= BULK_CAP:
                    self.z[k] = v
                    break
        # engineered (near-)diametric pairs out in the tails
        self.diameter_pairs = []
        for i, h, d in DIAMETER_PAIRS:
            j = (i + n // 2) % n
            self.theta[j] = self.theta[i] + np.pi + d
            self.z[i] = h
            self.z[j] = h + rng.uniform(-PAIR_HEIGHT_JITTER, PAIR_HEIGHT_JITTER)
            self.diameter_pairs.append((min(i, j), max(i, j)))
        self.z_span = float(self.z.max() - self.z.min())
        self.zoom_fn = lambda: 1.0  # extra display zoom (set by the scene)

    def data_points(self, sigma):
        """(noise, y, z) in data units: noise along the first axis."""
        return np.c_[sigma * self.z, np.cos(self.theta), np.sin(self.theta)]

    def nn_edges(self, sigma):
        """Undirected 1-NN edges as sorted index pairs."""
        P = self.data_points(sigma)
        D = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(D, np.inf)
        j = D.argmin(1)
        return sorted({(min(i, int(k)), max(i, int(k))) for i, k in enumerate(j)})

    def is_shortcut(self, i, j):
        """Returns True if the angular distance across the circle exceeds SHORTCUT_ANGLE."""
        d_theta = abs(self.theta[i] - self.theta[j]) % (2 * np.pi)
        d_theta = min(d_theta, 2 * np.pi - d_theta)
        return d_theta > SHORTCUT_ANGLE

    def display_scale(self, sigma):
        length = sigma * self.z_span * RADIUS_DISPLAY
        return (
            RADIUS_DISPLAY
            * self.zoom_fn()
            * min(1.0, MAX_LENGTH_DISPLAY / max(length, 1e-9))
        )

    def display_points(self, sigma, spin):
        P = self.data_points(sigma) * self.display_scale(sigma)
        c, s = np.cos(spin), np.sin(spin)
        y, z = P[:, 1].copy(), P[:, 2].copy()
        P[:, 1], P[:, 2] = c * y - s * z, s * y + c * z  # spin about noise axis
        return P


# ---------------------------------------------------------------------------
class CircleNoiseIntuition(ThreeDTIMCSlide):

    def pin_to_screen(self, mobs):
        """Integer/DecimalNumber rebuild their glyphs on set_value, so keep
        re-registering them as fixed-in-frame (scene updaters run last)."""
        self.add_fixed_in_frame_mobjects(*mobs)
        self.add_updater(lambda dt: self.camera.add_fixed_in_frame_mobjects(*mobs))

    def build_guides(self):
        """Faint reference circle (zero noise) plus a stem from each point's
        original position to its displaced one, parallel to the noise axis.
        Together they trace out the cylinder."""
        sig, spin = self.sigma.get_value(), self.spin.get_value()
        P = self.circle.display_points(sig, spin)
        r = self.circle.display_scale(sig)
        ring = Circle(
            radius=r, color=GUIDE_COLOR, stroke_width=1.5, stroke_opacity=GUIDE_OPACITY
        ).rotate(PI / 2, axis=UP)
        stems = VGroup(
            *[
                Line(
                    [0.0, p[1], p[2]],
                    p,
                    color=GUIDE_COLOR,
                    stroke_width=1.5,
                    stroke_opacity=GUIDE_OPACITY,
                )
                for p in P
                if abs(p[0]) > 1e-3
            ]
        )
        return VGroup(ring, stems)

    def build_edges(self):
        P = self.circle.display_points(self.sigma.get_value(), self.spin.get_value())
        edges = self.circle.nn_edges(self.sigma.get_value())
        edge_lines = []

        for i, j in edges:
            # Calculate actual normalized angular distance across circle [0 = adjacent, 1 = diametrically opposite]
            d_theta = abs(self.circle.theta[i] - self.circle.theta[j]) % (2 * np.pi)
            d_theta = min(d_theta, 2 * np.pi - d_theta)
            d = np.clip(d_theta / SHORTCUT_ANGLE, 0.0, 1.0)

            if d > 0.5:
                t = np.sqrt(2 * (d - 0.5))
                color = interpolate_color(COLOR_CYCLE[3], SHORTCUT_COLOR, t)
                width = interpolate(EDGE_WIDTH * 1.5, SHORTCUT_WIDTH, t)
            else:
                t = np.sqrt(2 * d)
                color = interpolate_color(EDGE_COLOR, COLOR_CYCLE[3], t)
                width = interpolate(EDGE_WIDTH, EDGE_WIDTH * 1.5, t)

            edge_lines.append(Line(P[i], P[j], color=color, stroke_width=width))

        # Return a flat VGroup so Manim updates submobjects cleanly every frame
        return VGroup(*edge_lines)

    def construct(self):
        self.circle = circ = NoisyCircle()
        self.sigma = ValueTracker(0.0)
        self.spin = ValueTracker(0.0)
        self.view_zoom = ValueTracker(1.0)
        circ.zoom_fn = self.view_zoom.get_value
        self.set_camera_orientation(**FACE_ON)

        # ---- points and edges (3D) ----
        P0 = circ.display_points(0.0, 0.0)
        # flat dots kept facing the camera (much faster than Dot3D spheres)
        dots = VGroup(
            *[
                Dot(point=P0[i], radius=DOT_RADIUS, color=cyclic_color(i / circ.n))
                for i in range(circ.n)
            ]
        )

        def move_dots(group):
            P = circ.display_points(self.sigma.get_value(), self.spin.get_value())
            for d, p in zip(group, P):
                d.move_to(p)

        edges = self.build_edges()

        # ---- overlay (screen space) ----
        title = (
            Text("Why noise creates shortcuts", color=DEFAULT_COLOR)
            .scale(0.6)
            .to_edge(UP, buff=0.35)
        )
        sub = (
            Text(
                "a circle, its nearest-neighbour edges, and noise in "
                "one direction orthogonal to it",
                color=ACCENT_COLOR,
            )
            .scale(0.32)
            .next_to(title, DOWN, buff=0.15)
        )

        sig_lab = Text("noise / radius", color=DEFAULT_COLOR, font_size=0.72 * HUD_FONT)
        sig_eq = MathTex("=", color=DEFAULT_COLOR, font_size=HUD_FONT)
        sig_num = DecimalNumber(
            0.0, num_decimal_places=2, font_size=HUD_FONT, color=NUMBER_COLOR
        )
        cut_lab = Text(
            "edges across the circle", color=DEFAULT_COLOR, font_size=0.72 * HUD_FONT
        )
        cut_eq = MathTex("=", color=DEFAULT_COLOR, font_size=HUD_FONT)
        cut_num = Integer(0, font_size=HUD_FONT, color=SHORTCUT_COLOR)
        rows = [VGroup(sig_lab, sig_eq, sig_num), VGroup(cut_lab, cut_eq, cut_num)]
        for lab, eq, num in rows:
            eq.next_to(lab, RIGHT, buff=0.12)
            num.next_to(eq, RIGHT, buff=0.12)
        hud = VGroup(*rows).arrange(DOWN, buff=0.22)
        x_eq = max(r[1].get_x() for r in rows)
        for r in rows:
            r.shift(RIGHT * (x_eq - r[1].get_x()))
        hud.to_corner(DL, buff=0.45)

        sig_num.add_updater(lambda m: m.set_value(self.sigma.get_value()))
        sig_num.add_updater(lambda m: m.next_to(sig_eq, RIGHT, buff=0.12))
        cut_num.add_updater(
            lambda m: m.set_value(
                sum(
                    circ.is_shortcut(i, j)
                    for i, j in circ.nn_edges(self.sigma.get_value())
                )
            )
        )
        cut_num.add_updater(lambda m: m.next_to(cut_eq, RIGHT, buff=0.12))

        key_local = VGroup(
            Line(ORIGIN, RIGHT * 0.5, color=EDGE_COLOR, stroke_width=EDGE_WIDTH),
            Text("nearest neighbour, along the circle", color=ACCENT_COLOR).scale(0.3),
        ).arrange(RIGHT, buff=0.15)
        key_short = VGroup(
            Line(
                ORIGIN, RIGHT * 0.5, color=SHORTCUT_COLOR, stroke_width=SHORTCUT_WIDTH
            ),
            Text("nearest neighbour, across the circle", color=ACCENT_COLOR).scale(0.3),
        ).arrange(RIGHT, buff=0.15)
        key = (
            VGroup(key_local, key_short)
            .arrange(DOWN, aligned_edge=LEFT, buff=0.12)
            .next_to(hud, RIGHT, buff=0.9)
            .align_to(hud, DOWN)
        )

        caption = (
            Text(
                "in high dimensions the noise spreads over many "
                "orthogonal directions; here it is packed into one",
                color=ACCENT_COLOR,
            )
            .scale(0.3)
            .next_to(sub, DOWN, buff=0.12)
        )

        # ---- slide 1: the circle ----
        self.add_fixed_in_frame_mobjects(title, sub)
        self.play(FadeIn(title), FadeIn(sub))
        self.add_fixed_orientation_mobjects(*dots)
        self.play(
            LaggedStart(
                *[FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.03, run_time=1.5
            )
        )
        self.add(dots)  # register the group itself so its updater runs
        self.marked_next_slide()

        # ---- slide 2: 1-NN edges ----
        self.add_fixed_in_frame_mobjects(key_local)
        self.play(Create(edges, lag_ratio=0.05), FadeIn(key_local), run_time=1.5)
        self.marked_next_slide()

        # ---- slide 3: tilt to show the noise direction ----
        self.move_camera(**OBLIQUE, run_time=2.5)
        self.pin_to_screen([hud, caption])
        self.add_fixed_in_frame_mobjects(key_short)
        self.play(FadeIn(hud), FadeIn(caption), FadeIn(key_short))

        # live updates from here on (edges and guides rebuilt every frame)
        guides = self.build_guides()
        self.play(FadeIn(guides))
        guides.add_updater(lambda m: m.become(self.build_guides()))
        dots.add_updater(move_dots)
        edges.add_updater(lambda m: m.become(self.build_edges()))
        self.marked_next_slide()

        # ---- slide 4: the noise sweep ----
        self.play(
            self.sigma.animate.set_value(SIGMA_MAX),
            self.spin.animate.increment_value(SPIN_DURING_SWEEP * SWEEP_SECONDS),
            run_time=SWEEP_SECONDS,
            rate_func=smooth,
        )

        # ---- slide 5: looping spin about the cylinder axis ----
        self.marked_next_slide(loop=True)
        self.play(
            self.spin.animate.increment_value(TAU),
            run_time=LOOP_SECONDS,
            rate_func=linear,
        )

        # ---- slide 6: back to face-on, looking down the noise direction ----
        self.marked_next_slide()
        # zoom the picture back up so the circle has its original size
        restore = RADIUS_DISPLAY / circ.display_scale(SIGMA_MAX)
        self.move_camera(
            **FACE_ON,
            focal_distance=FAR_FOCAL,
            run_time=3,
            added_anims=[self.view_zoom.animate.set_value(restore)],
        )
        self.marked_next_slide()
