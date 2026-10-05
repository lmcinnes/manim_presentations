"""
Shared slide for the nearest-neighbour edge showcases (MNIST, Quick, Draw!).

Subclasses set the data file and wording; see mnist_knn_slides.py and
quickdraw_knn_slides.py. The .npz comes from knn_edge_prep.analyse_and_save.

Slides:
    1. UMAP layout, coloured by class, with class badges
    2. all 1-NN edges (in the feature space) fade in
    3. short edges fade out; long (cross-cluster) edges remain
    4+. one slide per showcase edge: the edge is highlighted and the two
        images are shown beside the plot
    last. back to the long-edge overview
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

from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
PLOT_CENTER = np.array([0.0, -0.35, 0.0])
PLOT_WIDTH = 7.2  # the layout is fitted inside this box
PLOT_HEIGHT = 5.9
CLASS_COLORS = COLOR_CYCLE[:10]

EDGE_COLOR = DEFAULT_COLOR
DIM_OPACITY = 0.55  # white veil over the points when edges are shown

PANEL_X = 5.35  # centres of the image panels (left at -PANEL_X)
IMAGE_SIZE = 2.1
IMAGE_UPSCALE = 10  # 28px -> 280px, nearest-neighbour
INK_DARKEN = 0.5  # blend image ink toward DEFAULT_COLOR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fit_to_box(emb):
    lo, hi = emb.min(0), emb.max(0)
    s = min(PLOT_WIDTH / (hi[0] - lo[0]), PLOT_HEIGHT / (hi[1] - lo[1]))
    xy = (emb - (lo + hi) / 2) * s
    return np.c_[xy, np.zeros(len(xy))] + PLOT_CENTER


def segments_mobject(A, B, color, width, opacity):
    """Many disjoint line segments as ONE VMobject (far faster to render and
    animate than thousands of Line mobjects)."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    pts = np.empty((4 * len(A), 3))
    pts[0::4] = A
    pts[1::4] = A + (B - A) / 3
    pts[2::4] = A + 2 * (B - A) / 3
    pts[3::4] = B
    vm = VMobject(stroke_color=color, stroke_width=width, stroke_opacity=opacity)
    vm.set_points(pts)
    return vm


def image_mobject(pixels, color):
    """28x28 uint8 (ink = high) -> ImageMobject with the ink in `color` on a
    transparent background, sharp pixels."""
    a = np.asarray(pixels, np.uint8).reshape(28, 28)
    a = np.kron(a, np.ones((IMAGE_UPSCALE, IMAGE_UPSCALE), np.uint8))
    rgb = (np.array(ManimColor(color).to_rgb()) * 255).astype(np.uint8)
    rgba = np.empty(a.shape + (4,), np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = a
    img = ImageMobject(rgba)
    img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    img.height = IMAGE_SIZE
    return img


def darker(c, amount=0.35):
    return interpolate_color(ManimColor(c), DEFAULT_COLOR, amount)


def with_article(name):
    return ("an " if name[:1].lower() in "aeiou" else "a ") + name


# ---------------------------------------------------------------------------
class NearestNeighbourEdgeSlide(TIMCSlide):
    """Subclasses set DATA_FILE and override the text methods as needed."""

    DATA_FILE = None  # Path to the .npz from knn_edge_prep
    SHOWCASE_EDGES = None  # list of edge indices; None = prep's picks

    POINT_SIZE = 1.5
    EDGE_WIDTH = 0.4
    EDGE_OPACITY = 0.2
    LONG_EDGE_WIDTH = 0.8
    LONG_EDGE_OPACITY = 0.45
    BADGE_SCALE = 0.5

    # ---- wording (override in subclasses) --------------------------------
    def title_embedding(self):
        return ("UMAP", f"{self.n:,} items, coloured by class")

    def title_all_edges(self):
        return (
            f"Nearest neighbours in {self.space}",
            f"each item joined to its single closest item " f"({self.dim} dimensions)",
        )

    def title_long_edges(self):
        return (
            "Long nearest-neighbour edges",
            f"{int(self.is_long.sum())} of {len(self.is_long):,} edges "
            "jump between clusters",
        )

    def title_showcase(self, name_i, name_j):
        return (
            f"{with_article(name_i).capitalize()} and "
            f"{with_article(name_j)}: nearest neighbours",
            f"closest in {self.space}, from opposite ends of the map",
        )

    def caption(self, name):
        return with_article(name)

    # ---- data --------------------------------------------------------------
    def load(self):
        with np.load(self.DATA_FILE) as z:
            d = {k: z[k] for k in z.files}
        self.d = d
        self.P = fit_to_box(d["emb"])
        self.labels = d["labels"]
        self.n = len(self.labels)
        self.names = (
            [str(s) for s in d["class_names"]]
            if "class_names" in d
            else [str(i) for i in range(10)]
        )
        self.space = str(d.get("feature_space", "feature space"))
        self.dim = int(d["feature_dim"]) if "feature_dim" in d else d["images"].shape[1]
        pairs = d["pairs"]
        self.A, self.B = self.P[pairs[:, 0]], self.P[pairs[:, 1]]
        self.is_long = d["is_long"].astype(bool)
        edges = (
            self.SHOWCASE_EDGES if self.SHOWCASE_EDGES is not None else d["showcase"]
        )
        self.showcase = [int(e) for e in edges]

    # ---- building blocks -------------------------------------------------
    def heading(self, texts):
        title, subtitle = texts
        t = Text(title, color=DEFAULT_COLOR).scale(0.6).to_edge(UP, buff=0.3)
        s = Text(subtitle, color=ACCENT_COLOR).scale(0.32).next_to(t, DOWN, buff=0.12)
        return VGroup(t, s)

    def build_points(self):
        rgba = np.array(
            [[*ManimColor(CLASS_COLORS[l]).to_rgb(), 1.0] for l in self.labels]
        )
        pts = PMobject(stroke_width=self.POINT_SIZE)
        pts.add_points(self.P, rgbas=rgba)
        return pts

    def build_badges(self):
        badges = VGroup()
        for c, name in enumerate(self.names):
            members = self.P[self.labels == c]
            if not len(members):
                continue
            m = np.median(members, axis=0)
            t = Text(name, color=darker(CLASS_COLORS[c], 0.5), weight=BOLD)
            t.scale(self.BADGE_SCALE)
            back = RoundedRectangle(
                width=max(t.width + 0.22, 0.4),
                height=t.height + 0.18,
                corner_radius=0.18,
                stroke_color=CLASS_COLORS[c],
                stroke_width=2.5,
                fill_color=BACKGROUND_COLOR,
                fill_opacity=0.9,
            )
            badges.add(VGroup(back, t.move_to(back)).move_to(m))
        return badges

    def build_panel(self, point_index, side):
        """Image with frame and caption, left (-1) or right (+1) of the
        plot. Returns (group, frame)."""
        lab = int(self.labels[point_index])
        col = CLASS_COLORS[lab]
        img = image_mobject(self.d["images"][point_index], darker(col, INK_DARKEN))
        frame = SurroundingRectangle(
            img, buff=0.12, color=col, stroke_width=3, corner_radius=0.08
        )
        cap = Text(self.caption(self.names[lab]), color=darker(col, INK_DARKEN)).scale(
            0.45
        )
        cap.next_to(frame, DOWN, buff=0.15)
        panel = Group(frame, img, cap)
        panel.move_to([side * PANEL_X, PLOT_CENTER[1], 0])
        return panel, frame

    # ---- scene -----------------------------------------------------------
    def construct(self):
        self.load()

        points = self.build_points()
        badges = self.build_badges()
        veil = Rectangle(
            width=PLOT_WIDTH + 0.4,
            height=PLOT_HEIGHT + 0.4,
            stroke_width=0,
            fill_color=BACKGROUND_COLOR,
            fill_opacity=DIM_OPACITY,
        ).move_to(PLOT_CENTER)
        short_edges = segments_mobject(
            self.A[~self.is_long],
            self.B[~self.is_long],
            EDGE_COLOR,
            self.EDGE_WIDTH,
            self.EDGE_OPACITY,
        )
        long_edges = segments_mobject(
            self.A[self.is_long],
            self.B[self.is_long],
            EDGE_COLOR,
            self.EDGE_WIDTH,
            self.EDGE_OPACITY,
        )

        head = self.heading(self.title_embedding())

        # ---- slide 1: the embedding ----
        self.play(FadeIn(head))
        self.play(FadeIn(points), run_time=1.5)
        self.play(FadeIn(badges))
        self.marked_next_slide()

        # ---- slide 2: all 1-NN edges ----
        new = self.heading(self.title_all_edges())
        self.play(FadeIn(veil), FadeOut(badges), FadeTransform(head, new))
        head = new
        self.play(FadeIn(short_edges), FadeIn(long_edges), run_time=2)
        self.marked_next_slide()

        # ---- slide 3: keep the long edges ----
        new = self.heading(self.title_long_edges())
        self.play(
            FadeOut(short_edges),
            long_edges.animate.set_stroke(
                width=self.LONG_EDGE_WIDTH, opacity=self.LONG_EDGE_OPACITY
            ),
            FadeTransform(head, new),
            run_time=1.5,
        )
        head = new
        self.marked_next_slide()

        # ---- showcase slides ----
        pairs = self.d["pairs"]
        shown = None
        for e in self.showcase:
            i, j = int(pairs[e, 0]), int(pairs[e, 1])
            if self.P[i, 0] > self.P[j, 0]:
                i, j = j, i  # i is the left-hand endpoint
            pi, pj = self.P[i], self.P[j]
            li, lj = int(self.labels[i]), int(self.labels[j])

            edge = Line(pi, pj, color=HIGHLIGHT_COLOR, stroke_width=6)
            ends = VGroup(
                *[
                    Dot(
                        p,
                        radius=0.09,
                        color=CLASS_COLORS[l],
                        stroke_color=DEFAULT_COLOR,
                        stroke_width=2,
                    )
                    for p, l in ((pi, li), (pj, lj))
                ]
            )
            left, lframe = self.build_panel(i, -1)
            right, rframe = self.build_panel(j, +1)
            links = VGroup(
                DashedLine(
                    pi,
                    lframe.get_right(),
                    color=ACCENT_COLOR,
                    stroke_width=2,
                    dash_length=0.08,
                ),
                DashedLine(
                    pj,
                    rframe.get_left(),
                    color=ACCENT_COLOR,
                    stroke_width=2,
                    dash_length=0.08,
                ),
            )
            new = self.heading(self.title_showcase(self.names[li], self.names[lj]))

            if shown is None:
                out = [long_edges.animate.set_stroke(opacity=0.25)]
            else:
                out = [FadeOut(m) for m in shown]
            self.play(*out, FadeTransform(head, new), run_time=0.8)
            head = new
            self.play(Create(edge), FadeIn(ends), run_time=0.8)
            self.play(
                Create(links),
                FadeIn(left, shift=RIGHT * 0.3),
                FadeIn(right, shift=LEFT * 0.3),
                run_time=1,
            )
            shown = [edge, ends, left, right, links]
            self.marked_next_slide()

        # ---- back to the overview ----
        if shown is not None:
            new = self.heading(self.title_long_edges())
            self.play(
                *[FadeOut(m) for m in shown],
                long_edges.animate.set_stroke(opacity=self.LONG_EDGE_OPACITY),
                FadeTransform(head, new),
                run_time=1,
            )
        self.marked_next_slide()
