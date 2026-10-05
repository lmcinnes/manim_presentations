"""
A point nudged towards the centre becomes everyone's near neighbour.

A small cloud of points sits at each corner of a cube. Each point's k
nearest neighbours (k = cloud size) are its own cloud-mates plus the single
closest point of a neighbouring corner: that "outside" link is drawn. One
point leaves its cloud and slides towards the centre. As it does, points at
other corners replace their outside link with a link to it:
    the 3 adjacent corners       after a small nudge,
    the 3 face-diagonal corners  close to the centre,
    the opposite corner          at the centre.

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
)

import itertools

import numpy as np

apply_defaults()

HALF = 2.0                 # half the cube's side in scene units (data: 1)
CLOUD_SIZE = 5             # points per corner (also k, the neighbours kept)
CLOUD_SPREAD = 0.05        # in data units; larger blurs the three waves
SEED = 0
POINT_RADIUS = 0.06
HUB_COLOR = HIGHLIGHT_COLOR
CORNER_COLOR = ACCENT_COLOR
LINK_COLOR = DEFAULT_COLOR
SPIN_RATE = 0.12           # camera rotation, radians per second
STAGES = [                 # (s, caption): the point moves to s * its start
    (0.7, "nudged towards the centre: points at the 3 adjacent corners "
          "switch to it"),
    (0.12, "closer still: so do the 3 corners across its faces"),
    (0.0, "at the centre: every other corner links to it"),
]


class CubeHubSlide(ThreeDTIMCSlide):

    def heading(self, title, subtitle):
        t = Text(title, color=DEFAULT_COLOR).scale(0.6).to_edge(UP, buff=0.3)
        s = Text(subtitle, color=ACCENT_COLOR).scale(0.32).next_to(
            t, DOWN, buff=0.12)
        return VGroup(t, s)

    def swap_heading(self, old, title, subtitle):
        new = self.heading(title, subtitle)
        self.add_fixed_in_frame_mobjects(new)
        self.play(FadeOut(old), FadeIn(new))
        self.remove_fixed_in_frame_mobjects(old)
        return new

    def construct(self):
        # ---- data: a cloud at each corner; one point will move ----
        corners = np.array(list(itertools.product((-1, 1), repeat=3)), float)
        home = int(np.flatnonzero((corners == 1).all(1))[0])    # (1, 1, 1)
        m = CLOUD_SIZE
        rng = np.random.default_rng(SEED)
        P0 = np.repeat(corners, m, 0) + rng.normal(scale=CLOUD_SPREAD,
                                                   size=(8 * m, 3))
        cloud = np.repeat(np.arange(8), m)
        hub = home * m
        s = ValueTracker(1.0)

        def positions():
            P = P0.copy()
            P[hub] = s.get_value() * P0[hub]
            return P

        # each point's outside link: its closest point in another cloud
        # (the moving point excluded)
        D0 = np.linalg.norm(P0[:, None] - P0[None], axis=-1)
        outside = {}
        for i in range(8 * m):
            if i == hub:
                continue
            cand = np.flatnonzero((cloud != cloud[i])
                                  & (np.arange(8 * m) != hub))
            outside[i] = int(cand[np.argmin(D0[i, cand])])

        def captured():
            """Points (outside the moving point's own cloud) whose k nearest
            neighbours now include the moving point."""
            P = positions()
            D = np.linalg.norm(P[:, None] - P[None], axis=-1)
            np.fill_diagonal(D, np.inf)
            knn = np.argsort(D, axis=1)[:, :m]
            return {i for i in range(8 * m)
                    if cloud[i] != home and hub in knn[i]}

        self.set_camera_orientation(phi=65 * DEGREES, theta=-50 * DEGREES,
                                    zoom=0.9)
        self.begin_ambient_camera_rotation(rate=SPIN_RATE)

        # ---- mobjects ----
        cube = VGroup(*[
            Line(HALF * corners[a], HALF * corners[b], color=LINK_COLOR,
                 stroke_width=1.5, stroke_opacity=0.15)
            for a, b in itertools.combinations(range(8), 2)
            if np.isclose(np.linalg.norm(corners[a] - corners[b]), 2)])
        links = {i: Line(HALF * P0[i], HALF * P0[j], color=LINK_COLOR,
                         stroke_width=1.5, stroke_opacity=0.5)
                 for i, j in outside.items()}
        dots = VGroup(*[Dot(HALF * P0[i], radius=POINT_RADIUS,
                            color=CORNER_COLOR)
                        for i in range(8 * m) if i != hub])
        mover = Dot(HALF * P0[hub], radius=POINT_RADIUS * 1.4, color=HUB_COLOR)
        mover.add_updater(lambda d: d.move_to(HALF * positions()[hub]))
        ghost = Dot(HALF * P0[hub], radius=POINT_RADIUS * 1.4,
                    color=HUB_COLOR).set_opacity(0.25)
        self.add_fixed_orientation_mobjects(*dots, mover, ghost)
        self.remove(ghost)

        head = self.heading("A small cloud of points at each corner",
                            "each point links to its cloud and to the "
                            "closest point of a neighbouring corner")
        self.add_fixed_in_frame_mobjects(head)
        self.play(FadeIn(head), Create(cube))
        self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in [*dots, mover]],
                              lag_ratio=0.02))
        self.play(LaggedStart(*[Create(l) for l in links.values()],
                              lag_ratio=0.02), run_time=2)
        self.marked_next_slide()

        # ---- nudge the point in, in stages ----
        to_hub, done = {}, set()
        for target, caption in STAGES:
            head = self.swap_heading(head, "Nudge one point towards the centre",
                                     caption)
            if not done:
                self.add(ghost)
            self.play(s.animate.set_value(target), run_time=2.5,
                      rate_func=smooth)
            new = sorted(captured() - done)
            for i in new:
                ln = Line(HALF * P0[i], HALF * positions()[hub],
                          color=HUB_COLOR, stroke_width=2.5)
                ln.add_updater(lambda l, i=i: l.put_start_and_end_on(
                    HALF * P0[i], HALF * positions()[hub]))
                to_hub[i] = ln
            # the switch: old outside links fade as new links grow from each
            # captured point towards the moving point
            self.play(*[links[i].animate.set_stroke(opacity=0.0) for i in new],
                      LaggedStart(*[Create(to_hub[i]) for i in new],
                                  lag_ratio=0.08), run_time=2)
            done |= set(new)
            self.marked_next_slide()

        # ---- what changes in high dimensions ----
        head = self.swap_heading(
            head, "In n dimensions",
            "each corner has n adjacent corners: even a tiny nudge makes the "
            "point a near neighbour of all n at once")
        self.wait(4)
        self.marked_next_slide()
