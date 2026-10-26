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
# The unrolling is pinned: the outer end of the sheet is fixed where it
# finally lies (the right-hand end of the flat strip); everything already
# unrolled lies flat on the "floor" and never moves; the rest keeps the
# roll's shape, attached at the cut and tangent to the floor, and rolls along
# as it unwinds. Arc length is preserved at every stage, so this is a genuine
# (isometric) unrolling and runs equally well in either direction.
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


def spiral_points(s_grid, theta_grid):
    """The rolled centre line, integrated from its tangent angle."""
    ds = np.diff(s_grid)
    c, sn = np.cos(theta_grid), np.sin(theta_grid)
    x = np.concatenate([[0], np.cumsum(ds * (c[1:] + c[:-1]) / 2)])
    y = np.concatenate([[0], np.cumsum(ds * (sn[1:] + sn[:-1]) / 2)])
    return x, y


def pinned_curve(u, s_grid, theta_grid, x0, y0, flip):
    """Centre line on the arc-length grid at unroll fraction u (0 rolled,
    1 flat). The outer end is pinned at the right-hand end of the final
    strip (centred on 0, lying along y = 0); flip mirrors the roll to the
    other side of the floor."""
    S = s_grid[-1]
    s_cut = (1 - u) * S
    th_c = np.interp(s_cut, s_grid, theta_grid)
    xc, yc = np.interp(s_cut, s_grid, x0), np.interp(s_cut, s_grid, y0)
    c, sn = np.cos(-th_c), np.sin(-th_c)
    dx, dy = x0 - xc, y0 - yc
    rx, ry = c * dx - sn * dy, sn * dx + c * dy  # tangent -> +x
    cut_x = S / 2 - (S - s_cut)
    flat = s_grid >= s_cut
    X = np.where(flat, S / 2 - (S - s_grid), cut_x + rx)
    Y = np.where(flat, 0.0, ry)
    return X, (-Y if flip else Y)


# ---------------------------------------------------------------------------
class SwissRollUnroll(ThreeDTIMCSlide):
    """Swiss roll in 3D, a full turn around it, then a reversible unroll.

    Everything is driven by two trackers: `self.place` (0 = roll centred on
    screen; 0 -> 0.5 settles it back onto the floor line, a small move in
    depth best done while the camera turns; 0.5 -> 1 slides it straight
    sideways to the pinned end, ready to unroll) and
    `self.unroll` (0 = rolled, 1 = flat). Animating them in either direction
    rolls or unrolls, and anything attached with the helpers below follows
    along:

        dot  = self.follower(i)            # marker on sample i
        line = self.connection(i, j)       # straight line between two samples
        path = self.path_through([...])    # polyline through several samples

    Sample positions at the current stage are self.current_positions().
    """

    N_POINTS = 2000
    SEED = 0
    NOISE = 0.0  # sklearn's noise; keep 0 so the strip unrolls cleanly
    SCALE = 0.12  # scene units per data unit (flat strip ~ 10.7 wide;
    # leaves room for the roll at the pinned end)
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
    PLACE_SECONDS = 2.0  # sliding the roll to the pinned end

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
        # pinned-unroll geometry: the rolled shape, which side of the floor
        # the roll sits on (away from the camera), and the offset that
        # centres the rolled-up roll on screen
        self._x0, self._y0 = spiral_points(s_grid, theta_grid)
        _, Y = pinned_curve(0.0, s_grid, theta_grid, self._x0, self._y0, False)
        self._flip = bool(np.interp(self.s, s_grid, Y).mean() < 0)
        X, Y = pinned_curve(0.0, s_grid, theta_grid, self._x0, self._y0, self._flip)
        xs, ys = np.interp(self.s, s_grid, X), np.interp(self.s, s_grid, Y)
        mid = lambda a: (a.max() + a.min()) / 2
        self._roll_offset = np.array([mid(xs), mid(ys)])
        self.rgbas = np.array(
            [[*colormap_color(v, t.min(), t.max(), self.CMAP).to_rgb(), 1.0] for v in t]
        )

    def current_positions(self):
        """(N, 3) sample positions at the current place / unroll values."""
        return self.sheet_positions(self.s, self.height)

    def sheet_positions(self, s_vals, h_vals):
        """3D positions of any sheet coordinates (arc length along the roll,
        height) at the current place / unroll values."""
        X, Y = pinned_curve(
            self.unroll.get_value(),
            self.s_grid,
            self.theta_grid,
            self._x0,
            self._y0,
            self._flip,
        )
        place = self.place.get_value() if hasattr(self, "place") else 1.0
        # first half: settle onto the floor line (depth); second half: slide
        # straight sideways to the pinned end
        depth = min(place / 0.5, 1.0)
        side = max((place - 0.5) / 0.5, 0.0)
        off = np.array(
            [(1 - side) * self._roll_offset[0], (1 - depth) * self._roll_offset[1]]
        )
        mid_h = (self.height.max() + self.height.min()) / 2
        return (
            self.SCALE
            * np.c_[
                np.interp(s_vals, self.s_grid, X) - off[0],
                np.interp(s_vals, self.s_grid, Y) - off[1],
                h_vals - mid_h,
            ]
        )

    # ---- helpers for things attached to the data -------------------------
    def follower(self, i, color=HIGHLIGHT_COLOR, radius=0.15):
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
        self.place = ValueTracker(0.0)

        cloud = PMobject(stroke_width=self.POINT_SIZE)
        cloud.add_points(self.current_positions(), rgbas=self.rgbas)
        cloud.add_updater(lambda m: m.set(points=self.current_positions()))

        title = make_heading("The swiss roll")
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

        # --- slide: unroll (slide the roll to the pinned end, then unroll)
        self.marked_next_slide()
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

        # --- slide: roll back up (the same animation in reverse)
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
        #                  added_anims=[self.place.animate.set_value(0.5)],
        #                  run_time=self.PLACE_SECONDS)
        # self.play(self.place.animate.set_value(1.0),
        #           run_time=self.PLACE_SECONDS)
        # self.play(self.unroll.animate.set_value(1.0),
        #           run_time=self.UNROLL_SECONDS)
        #
        # Along-the-surface (geodesic) paths: pick samples ordered by t
        # between the two ends and draw a polyline through them, e.g.
        # order = np.argsort(self.t)
        # between = order[(self.t[order] >= self.t[i]) &
        #                 (self.t[order] <= self.t[j])][::25]
        # geo = self.path_through(between)
        # ------------------------------------------------------------------
