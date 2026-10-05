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
from sklearn.datasets import make_swiss_roll

apply_defaults()


# ---------------------------------------------------------------------------
# Geometry
#
# The swiss roll's centre line is the Archimedean spiral (t cos t, t sin t),
# t in [1.5 pi, 4.5 pi]; the third coordinate is just height.
#
# The unrolling peels the roll like a carpet: at unroll u everything up to
# arc length s_cut = (1 - u) * total is still spiralled, and past that the
# strip is straight. u = 0 is the spiral, u = 1 is a straight line, and
# every stage has the SAME arc length, so this is a genuine (isometric)
# unrolling that runs equally well in either direction. The whole thing is
# rotated so the flat part always lies along x (otherwise the peeling tail
# swings the picture around) and re-centred.
# ---------------------------------------------------------------------------
T_MIN, T_MAX = 1.5 * np.pi, 4.5 * np.pi
GRID = 4000  # resolution of the arc-length grid


def spiral_grid():
    """Arc length s and unwrapped tangent angle theta along the spiral."""
    t = np.linspace(T_MIN, T_MAX, GRID)
    speed = np.sqrt(1 + t**2)  # |d/dt (t cos t, t sin t)|
    s = np.concatenate([[0], np.cumsum(np.diff(t) * (speed[1:] + speed[:-1]) / 2)])
    theta = np.unwrap(np.arctan2(np.sin(t) + t * np.cos(t), np.cos(t) - t * np.sin(t)))
    return t, s, theta - theta[0]


def curve_at(u, s_grid, theta_grid):
    """Points of the partially unrolled centre line on the arc-length grid."""
    s_cut = (1.0 - u) * s_grid[-1]
    th = np.interp(np.minimum(s_grid, s_cut), s_grid, theta_grid)
    th = th - np.interp(s_cut, s_grid, theta_grid)  # flat part along +x
    ds = np.diff(s_grid)
    cx, cy = np.cos(th), np.sin(th)
    x = np.concatenate([[0], np.cumsum(ds * (cx[1:] + cx[:-1]) / 2)])
    y = np.concatenate([[0], np.cumsum(ds * (cy[1:] + cy[:-1]) / 2)])
    x -= x.mean()
    y -= y.mean()
    return x, y


# ---------------------------------------------------------------------------
class SwissRollUnroll(ThreeDTIMCSlide):
    """Swiss roll in 3D, a full turn around it, then a reversible unroll.

    Everything is driven by the `self.unroll` tracker (0 = rolled,
    1 = flat), so animating it in either direction rolls or unrolls, and
    anything attached with the helpers below follows along:

        dot  = self.follower(i)            # marker on sample i
        line = self.connection(i, j)       # straight line between two samples
        path = self.path_through([...])    # polyline through several samples

    Sample positions at the current stage are self.current_positions().
    """

    N_POINTS = 2000
    SEED = 0
    NOISE = 0.0  # sklearn's noise; keep 0 so the strip unrolls cleanly
    SCALE = 0.135  # scene units per data unit (flat strip ~ 12 wide)
    HEIGHT_SCALE = 0.6  # sklearn's roll is as tall as it is wide, which
    # hides the spiral; a shorter roll reads better
    POINT_SIZE = 2.5
    CMAP = "plasma"  # colour by position along the roll

    # looking down on the roll shows the spiral; looking from the side
    # shows the unrolled strip as a rectangle
    ROLLED_CAMERA = dict(phi=32 * DEGREES, theta=-60 * DEGREES)
    FLAT_CAMERA = dict(phi=78 * DEGREES, theta=-90 * DEGREES)
    SPIN_RATE = 0.5  # rad/s for the turn around the rolled data
    UNROLL_SECONDS = 6.0

    # ---- data ------------------------------------------------------------
    def setup_data(self):
        X, t = make_swiss_roll(self.N_POINTS, noise=self.NOISE, random_state=self.SEED)
        self.t = t  # position along the roll (= spiral t)
        self.height = self.HEIGHT_SCALE * X[:, 1]  # sklearn's 2nd coordinate
        _, s_grid, theta_grid = spiral_grid()
        self.s_grid, self.theta_grid = s_grid, theta_grid
        # arc length of each sample, by interpolation on the grid
        t_grid = np.linspace(T_MIN, T_MAX, GRID)
        self.s = np.interp(t, t_grid, s_grid)
        self.rgbas = np.array(
            [[*colormap_color(v, t.min(), t.max(), self.CMAP).to_rgb(), 1.0] for v in t]
        )

    def current_positions(self):
        """(N, 3) sample positions at the current unroll value."""
        u = self.unroll.get_value()
        x, y = curve_at(u, self.s_grid, self.theta_grid)
        px = np.interp(self.s, self.s_grid, x)
        py = np.interp(self.s, self.s_grid, y)
        # centre on the extent of the samples (they bunch towards the outer
        # end of the roll, so the mean would sit off-centre)
        mid = lambda a: (a.max() + a.min()) / 2
        return (
            self.SCALE
            * np.c_[px - mid(px), py - mid(py), self.height - mid(self.height)]
        )

    # ---- helpers for things attached to the data -------------------------
    def follower(self, i, color=HIGHLIGHT_COLOR, radius=0.07):
        """A dot that stays on sample i through the (un)rolling."""
        dot = Dot3D(self.current_positions()[i], radius=radius, color=color)
        dot.add_updater(lambda m: m.move_to(self.current_positions()[i]))
        return dot

    def connection(self, i, j, color=HIGHLIGHT_COLOR, stroke_width=4, **kw):
        """A straight line between samples i and j (ambient distance)."""
        P = self.current_positions()
        line = Line(P[i], P[j], color=color, stroke_width=stroke_width, **kw)
        line.add_updater(
            lambda m: m.put_start_and_end_on(*self.current_positions()[[i, j]])
        )
        return line

    def path_through(self, indices, color=ACCENT_COLOR, stroke_width=4, **kw):
        """A polyline through a list of samples (e.g. a geodesic path)."""
        idx = list(indices)
        path = VMobject(color=color, stroke_width=stroke_width, **kw)
        path.set_points_as_corners(self.current_positions()[idx])
        path.add_updater(
            lambda m: m.set_points_as_corners(self.current_positions()[idx])
        )
        return path

    # ---- scene -----------------------------------------------------------
    def construct(self):
        self.setup_data()
        self.unroll = ValueTracker(0.0)

        cloud = PMobject(stroke_width=self.POINT_SIZE)
        cloud.add_points(self.current_positions(), rgbas=self.rgbas)
        cloud.add_updater(lambda m: m.set(points=self.current_positions()))

        title = Text("The swiss roll", font_size=36).to_edge(UP, buff=0.35)
        self.add_fixed_in_frame_mobjects(title)

        self.set_camera_orientation(**self.ROLLED_CAMERA)
        self.play(FadeIn(title))
        self.play(FadeIn(cloud), run_time=1.5)
        self.add(cloud)

        # --- slide: one full turn around the rolled data (loops seamlessly)
        self.marked_next_slide(loop=True)
        self.begin_ambient_camera_rotation(rate=self.SPIN_RATE)
        self.wait(TAU / self.SPIN_RATE)
        self.stop_ambient_camera_rotation()

        # --- slide: unroll
        self.marked_next_slide()
        self.move_camera(
            **self.FLAT_CAMERA,
            added_anims=[self.unroll.animate.set_value(1.0)],
            run_time=self.UNROLL_SECONDS,
            rate_func=smooth,
        )
        self.marked_next_slide()

        # --- slide: roll back up (the same animation in reverse)
        self.move_camera(
            **self.ROLLED_CAMERA,
            added_anims=[self.unroll.animate.set_value(0.0)],
            run_time=self.UNROLL_SECONDS,
            rate_func=smooth,
        )
        self.marked_next_slide()

        # ------------------------------------------------------------------
        # Space for extras. Anything made with the helpers follows the data,
        # so it can be added before, during or after the (un)rolling, e.g.:
        #
        # i, j = 100, 1500                       # two samples to compare
        # near = self.connection(i, j)           # straight-line distance
        # ends = VGroup(self.follower(i), self.follower(j))
        # self.play(Create(near), FadeIn(ends))
        # self.marked_next_slide()
        # self.move_camera(**self.FLAT_CAMERA,
        #                  added_anims=[self.unroll.animate.set_value(1.0)],
        #                  run_time=self.UNROLL_SECONDS)
        #
        # Along-the-surface (geodesic) paths: pick samples ordered by t
        # between the two ends and draw a polyline through them, e.g.
        # order = np.argsort(self.t)
        # between = order[(self.t[order] >= self.t[i]) &
        #                 (self.t[order] <= self.t[j])][::25]
        # geo = self.path_through(between)
        # ------------------------------------------------------------------
