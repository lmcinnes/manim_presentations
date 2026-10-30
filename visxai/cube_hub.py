"""
A point nudged towards the centre becomes everyone's near neighbour.

A diffuse cloud of points, each cloud its own colour, sits at each corner of
a cube. Every point is linked to its k nearest neighbours by thin edges. k is
a little larger than a cloud, so most links stay inside a cloud but every
point also reaches a few into the clouds around it. One point leaves its
cloud and slides towards the centre. As it does, points in other clouds start
to count it among their k nearest:
    the 3 adjacent clouds         part way in,
    the 3 face-diagonal clouds    close to the centre,
    the opposite cloud            at the centre.

    manim-slides render cube_hub.py CubeHubSlide
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
    install_round_points,
    res_px,
    PMFadeIn,
)

import itertools

import numpy as np

apply_defaults()
install_round_points()

HALF = 2.0  # half the cube's side in scene units (data: 1)
CLOUD_SIZE = 10  # points per corner
K = 13  # neighbours kept: > CLOUD_SIZE, so every point
# has at least K - CLOUD_SIZE + 1 = 4 outside links
CLOUD_SPREAD = 0.25  # sd per axis, data units (corners are 2 apart)
SEED = 11  # checked: the three waves below hold (see STAGES)
POINT_SIZE = 21  # PMobject point size, pixels at 1440p (scaled by res_px)
HUB_RADIUS = 0.085
HUB_COLOR = HIGHLIGHT_COLOR
EDGE_WIDTH = 1.0  # thin kNN edges
HUB_EDGE_WIDTH = 2.2
INSIDE_TINT = 0.35  # inside-cloud edges: cloud colour mixed this far
# towards the background
OUTSIDE_TINT = 0.6  # links between clouds: DEFAULT_COLOR mixed this
# far towards the background
SPIN_RATE = 0.12  # camera rotation, radians per second
STAGES = (0.6, 0.15, 0.0)  # the point moves to s * its start; at these s the
# new links come from adjacent clouds only, then
# also face-diagonal ones, then all seven clouds

CORNERS = np.array(list(itertools.product((-1, 1), repeat=3)), float)
HOME = int(np.flatnonzero((CORNERS == 1).all(1))[0])  # (1, 1, 1)
STEPS = (CORNERS != CORNERS[HOME]).sum(1)  # 1 adjacent, 2 face, 3 opposite


def segments(P, pairs):
    """Points for one VMobject drawing every (i, j) in pairs as a straight
    segment (4 points per segment)."""
    if not pairs:
        return np.zeros((0, 3))
    idx = np.array(pairs)
    a, b = HALF * P[idx[:, 0]], HALF * P[idx[:, 1]]
    return np.stack([a, a + (b - a) / 3, a + 2 * (b - a) / 3, b], axis=1).reshape(-1, 3)


def cloud_colours(n):
    """n colours from COLOR_CYCLE, in palette order, leaving out the one
    closest to HUB_COLOR (so the moving point stays distinct) and then the
    greyest (which reads as an edge, not a cloud)."""
    rgb = [np.array(ManimColor(c).to_rgb()) for c in COLOR_CYCLE]
    hub = np.array(ManimColor(HUB_COLOR).to_rgb())
    keep = list(range(len(COLOR_CYCLE)))
    keep.remove(min(keep, key=lambda c: np.linalg.norm(rgb[c] - hub)))
    while len(keep) > n:
        keep.remove(min(keep, key=lambda c: np.ptp(rgb[c])))  # chroma
    return [COLOR_CYCLE[c] for c in keep]


class CubeHubSlide(ThreeDTIMCSlide):

    def heading(self, title, subtitle):
        return make_heading(title, subtitle)

    def swap_heading(self, old, title, subtitle):
        new = self.heading(title, subtitle)
        self.add_fixed_in_frame_mobjects(new)
        self.play(FadeOut(old), FadeIn(new))
        self.remove_fixed_in_frame_mobjects(old)
        return new

    # ---- data ----
    def setup_data(self):
        m = CLOUD_SIZE
        rng = np.random.default_rng(SEED)
        self.P0 = np.repeat(CORNERS, m, 0) + rng.normal(
            scale=CLOUD_SPREAD, size=(8 * m, 3)
        )
        self.cloud = np.repeat(np.arange(8), m)
        # the point that moves: an ordinary member, nearest its cloud's mean
        own = np.flatnonzero(self.cloud == HOME)
        self.hub = int(
            own[np.argmin(np.linalg.norm(self.P0[own] - self.P0[own].mean(0), axis=1))]
        )
        self.s = ValueTracker(1.0)

    def positions(self, s=None):
        P = self.P0.copy()
        P[self.hub] = (self.s.get_value() if s is None else s) * self.P0[self.hub]
        return P

    def knn(self, s):
        P = self.positions(s)
        D = np.linalg.norm(P[:, None] - P[None], axis=-1)
        np.fill_diagonal(D, np.inf)
        return np.argsort(D, axis=1)[:, :K]

    def edges(self, s):
        """Undirected edges {i, j} where i keeps j among its K nearest. The
        moving point's own choices are left out: the slide is about who
        chooses it."""
        nn = self.knn(s)
        return {
            tuple(sorted((i, int(j))))
            for i in range(len(nn))
            if i != self.hub
            for j in nn[i]
        }

    def choosers(self, s):
        """Points in other clouds that keep the moving point among their K
        nearest."""
        nn = self.knn(s)
        return [
            i for i in range(len(nn)) if self.cloud[i] != HOME and self.hub in nn[i]
        ]

    # ---- edge layers: one VMobject per style ----
    def kind(self, e):
        i, j = e
        if self.hub in e:
            other = j if i == self.hub else i
            return "hub_home" if self.cloud[other] == HOME else "hub"
        if self.cloud[i] == self.cloud[j]:
            return int(self.cloud[i])
        return "out"

    def style(self, kind):
        if kind == "hub":
            return dict(
                stroke_color=HUB_COLOR, stroke_width=HUB_EDGE_WIDTH, stroke_opacity=1.0
            )
        if kind == "out":
            return dict(
                stroke_color=mixbox_interpolate_color(
                    DEFAULT_COLOR, BACKGROUND_COLOR, OUTSIDE_TINT
                ),
                stroke_width=EDGE_WIDTH,
                stroke_opacity=1.0,
            )
        c = HOME if kind == "hub_home" else kind
        return dict(
            stroke_color=mixbox_interpolate_color(
                self.colours[c], BACKGROUND_COLOR, INSIDE_TINT
            ),
            stroke_width=EDGE_WIDTH,
            stroke_opacity=1.0,
        )

    def layer(self, kind, pairs):
        mob = VMobject(fill_opacity=0, **self.style(kind))
        mob.pairs = sorted(pairs)
        mob.set_points(segments(self.positions(), mob.pairs))
        mob.set_z_index(0)
        if kind in ("hub", "hub_home"):  # these follow the moving point
            mob.add_updater(lambda l: l.set_points(segments(self.positions(), l.pairs)))
        return mob

    def rewire(self, new_edges):
        """Edges no longer kept fade out while new ones grow, together."""
        gone, born = self.E - new_edges, new_edges - self.E
        fades, grows = [], []
        for kind in {self.kind(e) for e in gone}:
            lay = self.layers[kind]
            lay.pairs = [e for e in lay.pairs if e not in gone]
            lay.set_points(segments(self.positions(), lay.pairs))
            fades.append(self.layer(kind, [e for e in gone if self.kind(e) == kind]))
        for kind in {self.kind(e) for e in born}:
            grows.append(self.layer(kind, [e for e in born if self.kind(e) == kind]))
        self.add(*fades)
        self.play(
            *[f.animate.set_stroke(opacity=0) for f in fades],
            *[Create(g) for g in grows],
            run_time=2,
        )
        self.remove(*fades, *grows)
        for e in born:
            kind = self.kind(e)
            if kind not in self.layers:
                self.layers[kind] = self.layer(kind, [])
                self.add(self.layers[kind])
            self.layers[kind].pairs.append(e)
        for lay in self.layers.values():
            lay.set_points(segments(self.positions(), lay.pairs))
        self.E = set(new_edges)

    def construct(self):
        self.setup_data()
        cloud, hub = self.cloud, self.hub
        self.colours = cloud_colours(8)

        # check the story still holds for these settings before drawing
        caught = [self.choosers(s) for s in STAGES]

        def reached(c, step):
            return {int(cloud[i]) for i in c if STEPS[cloud[i]] == step}

        pattern = [tuple(len(reached(c, t)) for t in (1, 2, 3)) for c in caught]
        if not (
            pattern[0] == (3, 0, 0)
            and pattern[1][1] >= 2
            and pattern[1][2] == 0
            and pattern[2] == (3, 3, 1)
        ):
            raise ValueError(
                f"clouds reached per stage {pattern}: the waves "
                "do not hold for these settings; try another "
                "SEED or STAGES"
            )

        self.set_camera_orientation(phi=65 * DEGREES, theta=-50 * DEGREES, zoom=0.78)
        self.begin_ambient_camera_rotation(rate=SPIN_RATE)

        # ---- mobjects ----
        cube = VGroup(
            *[
                Line(
                    HALF * CORNERS[a],
                    HALF * CORNERS[b],
                    color=DEFAULT_COLOR,
                    stroke_width=1.5,
                    stroke_opacity=0.15,
                )
                for a, b in itertools.combinations(range(8), 2)
                if np.isclose(np.linalg.norm(CORNERS[a] - CORNERS[b]), 2)
            ]
        )
        px = res_px(POINT_SIZE)
        clouds = []
        for c in range(8):
            idx = [i for i in np.flatnonzero(cloud == c) if i != hub]
            pm = PMobject(stroke_width=px)
            pm.add_points(HALF * self.P0[idx], color=self.colours[c])
            pm.set_z_index(1)
            clouds.append(pm)
        mover = Dot(HALF * self.P0[hub], radius=HUB_RADIUS, color=HUB_COLOR)
        mover.add_updater(lambda d: d.move_to(HALF * self.positions()[hub]))
        mover.set_z_index(2)
        ghost = Dot(
            HALF * self.P0[hub],
            radius=HUB_RADIUS,
            color=mixbox_interpolate_color(HUB_COLOR, BACKGROUND_COLOR, 0.7),
        ).set_z_index(1)
        self.add_fixed_orientation_mobjects(mover, ghost)
        self.remove(ghost)

        self.E = self.edges(1.0)
        by_kind = {}
        for e in self.E:
            by_kind.setdefault(self.kind(e), []).append(e)
        self.layers = {k: self.layer(k, v) for k, v in by_kind.items()}
        inside = [self.layers[k] for k in self.layers if k != "out"]

        directed = [
            (i, j) for i, nn in enumerate(self.knn(1.0)) if i != hub for j in nn
        ]
        share = np.mean([cloud[i] == cloud[j] for i, j in directed])

        # ---- slide 1: the clouds, and links inside them ----
        head = self.heading(
            "A diffuse cloud of points at each corner",
            f"each point is linked to its {K} nearest " "neighbours",
        )
        self.add_fixed_in_frame_mobjects(head)
        self.play(FadeIn(head), Create(cube))
        self.add(*clouds)
        self.play(
            LaggedStart(*[PMFadeIn(pm) for pm in clouds], lag_ratio=0.15),
            FadeIn(mover, scale=0.5),
            run_time=2,
        )
        self.play(*[Create(l) for l in inside], run_time=2)
        self.marked_next_slide()

        # ---- slide 2: ... and the links that reach outside ----
        head = self.swap_heading(
            head,
            f"Most links stay inside a cloud ({share:.0%})",
            f"but with {CLOUD_SIZE} points to a cloud and {K} neighbours, "
            "every point also reaches into the next clouds",
        )
        self.play(Create(self.layers["out"]), run_time=2)
        self.marked_next_slide()

        # ---- nudge the point in, in stages ----
        captions = [
            "part way in: {n} points in the 3 adjacent clouds now count it "
            "among their nearest",
            "close to the centre: the clouds across its faces join in "
            "({n} points in {c} clouds)",
            "at the centre: {n} points from all 7 other clouds count it "
            "among their nearest",
        ]
        for s, c, cap in zip(STAGES, caught, captions):
            head = self.swap_heading(
                head,
                "Nudge one point towards the centre",
                cap.format(n=len(c), c=len({int(cloud[i]) for i in c})),
            )
            if s == STAGES[0]:
                self.add(ghost)
            self.play(self.s.animate.set_value(s), run_time=2.5, rate_func=smooth)
            self.rewire(self.edges(s))
            self.marked_next_slide()

        # ---- what changes in high dimensions ----
        head = self.swap_heading(
            head,
            "In n dimensions",
            "each corner has n adjacent corners: even a tiny nudge makes the "
            "point a near neighbour of all n at once",
        )
        self.wait(4)
        self.marked_next_slide()

        self.new_section("Hubs in Practice?", auto_next=True)
