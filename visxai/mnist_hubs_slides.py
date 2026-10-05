"""
Good and bad hubs in noisy MNIST.

    python mnist_hubs_prep.py           # once: writes mnist_hubs.npz + sheets
    manim-slides render mnist_hubs_slides.py MNISTHubsSlide

A UMAP of noisy MNIST; then, for each (bad hub, good hub) pair, the points
that count each hub among their nearest neighbours reach out to it (edges
grow from those points towards the hub), and both hubs' pointers are shown
as digit images. Locally the two hubs look the same; only the labels (or a
global view) reveal that one gathers its own digit and the other gathers
everything.

Pick pairs from the contact sheets (mnist_hubs.candidates_NN.png) and list
them in PAIRS as (bad, good), using the indices printed above each panel.
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

import os
from pathlib import Path

import numpy as np

apply_defaults()

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
DATA_FILE = Path(os.environ.get("MNIST_HUBS", _HERE / "mnist_hubs.npz"))

PAIRS = [
    (15728, 25703),
    (17664, 32612),
    (63213, 2725),
    (36046, 4508),
    (22312, 11235),
]
# PAIRS = None             # e.g. [(17664, 32612), (28984, 28005)] as (bad, good);
# None = the prep's top N_PAIRS_DEFAULT pairs
N_PAIRS_DEFAULT = 3
SHOW_NOISY_DIGITS = False  # True: show the digits with their noise added

CLASS_COLORS = COLOR_CYCLE[:10]
GOOD_COLOR = COLOR_CYCLE[2]
BAD_COLOR = HIGHLIGHT_COLOR
OTHER_DIGIT_COLOR = HIGHLIGHT_COLOR  # labels of pointers of another digit

PLOT_CENTER = np.array([-3.65, -0.45, 0.0])
PLOT_WIDTH, PLOT_HEIGHT = 6.4, 6.0
POINT_SIZE = 1.2
VEIL_OPACITY = 0.6  # dims the map while edges are shown

N_SHOW = 10  # pointers shown as images per hub
IMAGE_SIZE = 0.52
IMAGE_GAP = 0.05
PANEL_LEFT = 0.0  # x where the image rows start
GOOD_ROW_Y, BAD_ROW_Y = 0.9, -1.9


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fit_to_box(emb):
    lo, hi = emb.min(0), emb.max(0)
    s = min(PLOT_WIDTH / (hi[0] - lo[0]), PLOT_HEIGHT / (hi[1] - lo[1]))
    xy = (emb - (lo + hi) / 2) * s
    return np.c_[xy, np.zeros(len(xy))] + PLOT_CENTER


def reverse_neighbours(idx, h):
    """The points that have h among their k nearest neighbours."""
    return np.flatnonzero((idx == h).any(1))


def digit_image(pixels, color=DEFAULT_COLOR, upscale=10):
    """28x28 uint8 (ink = high) -> sharp ImageMobject inked in `color`."""
    a = np.kron(
        np.asarray(pixels, np.uint8).reshape(28, 28),
        np.ones((upscale, upscale), np.uint8),
    )
    rgb = (np.array(ManimColor(color).to_rgb()) * 255).astype(np.uint8)
    rgba = np.empty(a.shape + (4,), np.uint8)
    rgba[..., :3], rgba[..., 3] = rgb, a
    img = ImageMobject(rgba)
    img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    img.height = IMAGE_SIZE
    return img


def darker(c, amount=0.4):
    return interpolate_color(ManimColor(c), DEFAULT_COLOR, amount)


# ---------------------------------------------------------------------------
class MNISTHubsSlide(TIMCSlide):

    def load(self):
        with np.load(DATA_FILE) as z:
            d = {k: z[k] for k in z.files}
        self.d = d
        self.y = d["labels"]
        self.P = fit_to_box(d["emb"])
        self.idx = d["idx"]
        self.images = d["images"]
        if SHOW_NOISY_DIGITS:
            noise = np.random.default_rng(int(d["noise_seed"])).standard_normal(
                (len(self.images), 784), dtype=np.float32
            )
            noisy = self.images / 255.0 + float(d["sigma"]) * noise
            self.images = (np.clip(noisy, 0, 1) * 255).astype(np.uint8)
        pairs = (
            PAIRS
            if PAIRS is not None
            else [tuple(p) for p in d["pairs"][:N_PAIRS_DEFAULT]]
        )
        self.pairs = [(int(b), int(g)) for b, g in pairs]

    # ---- building blocks -------------------------------------------------
    def heading(self, title, subtitle):
        t = Text(title, color=DEFAULT_COLOR).scale(0.6).to_edge(UP, buff=0.3)
        s = Text(subtitle, color=ACCENT_COLOR).scale(0.32).next_to(t, DOWN, buff=0.12)
        return VGroup(t, s)

    def build_points(self):
        rgba = np.array([[*ManimColor(CLASS_COLORS[l]).to_rgb(), 1.0] for l in self.y])
        pts = PMobject(stroke_width=POINT_SIZE)
        pts.add_points(self.P, rgbas=rgba)
        return pts

    def build_badges(self):
        badges = VGroup()
        for c in range(10):
            m = np.median(self.P[self.y == c], axis=0)
            t = Text(str(c), color=darker(CLASS_COLORS[c], 0.5), weight=BOLD).scale(
                0.45
            )
            disc = Circle(
                radius=0.18,
                stroke_color=CLASS_COLORS[c],
                stroke_width=2.5,
                fill_color=BACKGROUND_COLOR,
                fill_opacity=0.9,
            )
            badges.add(VGroup(disc, t.move_to(disc)).move_to(m))
        return badges

    def hub_marker(self, h, color):
        return Dot(
            self.P[h],
            radius=0.1,
            color=color,
            stroke_color=DEFAULT_COLOR,
            stroke_width=2.5,
        ).set_z_index(3)

    def reach_edges(self, h, color):
        """Dots at the pointers and lines from each pointer to the hub (so
        Create grows them from the pointer towards the hub)."""
        rev = reverse_neighbours(self.idx, h)
        dots = VGroup(
            *[Dot(self.P[j], radius=0.035, color=color) for j in rev]
        ).set_z_index(2)
        lines = VGroup(
            *[Line(self.P[j], self.P[h], color=color, stroke_width=1.8) for j in rev]
        ).set_z_index(1)
        return dots, lines

    def image_row(self, h, color, y_row, kind):
        """Header with the hub's local statistics, the hub's image (framed)
        and the images of its pointers, labelled with their digit."""
        rev = reverse_neighbours(self.idx, h)
        nk, tri = int(self.d["Nk"][h]), int(self.d["tri"][h])
        header = Text(
            f"{kind} hub: a nearest neighbour of {nk} digits " f"({tri} triangles)",
            color=darker(color, 0.35),
        ).scale(0.3)
        x = PANEL_LEFT
        hub_img = digit_image(self.images[h])
        hub_img.move_to([x + IMAGE_SIZE / 2, y_row, 0])
        frame = SurroundingRectangle(hub_img, buff=0.05, color=color, stroke_width=3)
        x += IMAGE_SIZE + 3 * IMAGE_GAP
        cells, labels = Group(), VGroup()
        for j in rev[:N_SHOW]:
            img = digit_image(self.images[j])
            img.move_to([x + IMAGE_SIZE / 2, y_row, 0])
            same = self.y[j] == self.y[h]
            lab = Text(
                str(self.y[j]),
                font_size=18,
                color=DEFAULT_COLOR if same else OTHER_DIGIT_COLOR,
            )
            lab.next_to(img, UP, buff=0.04)
            cells.add(img)
            labels.add(lab)
            x += IMAGE_SIZE + IMAGE_GAP
        header.next_to(frame, UP, buff=0.32).align_to(frame, LEFT)
        return header, Group(frame, hub_img), cells, labels

    # ---- scene -----------------------------------------------------------
    def construct(self):
        self.load()
        sigma = float(self.d["sigma"])

        points = self.build_points()
        badges = self.build_badges()
        veil = Rectangle(
            width=PLOT_WIDTH + 0.4,
            height=PLOT_HEIGHT + 0.4,
            stroke_width=0,
            fill_color=BACKGROUND_COLOR,
            fill_opacity=VEIL_OPACITY,
        ).move_to(PLOT_CENTER)

        head = self.heading(
            "Hubs in noisy MNIST",
            f"UMAP of {len(self.y):,} digits with pixel noise "
            f"(sigma = {sigma:g}), coloured by digit",
        )
        self.play(FadeIn(head))
        self.play(FadeIn(points), run_time=1.5)
        self.play(FadeIn(badges))
        self.marked_next_slide()

        self.play(FadeOut(badges), FadeIn(veil))
        for b, g in self.pairs:
            # --- the two hubs: similar locally ---
            new = self.heading(
                "Two hubs",
                "each is a nearest neighbour of many digits, with a similar "
                "local neighbourhood",
            )
            marks = VGroup(
                self.hub_marker(g, GOOD_COLOR), self.hub_marker(b, BAD_COLOR)
            )
            self.play(FadeTransform(head, new), FadeIn(marks, scale=1.5))
            head = new
            self.marked_next_slide()

            shown = [marks]
            for h, color, y_row, kind in (
                (g, GOOD_COLOR, GOOD_ROW_Y, "one"),
                (b, BAD_COLOR, BAD_ROW_Y, "another"),
            ):
                dots, lines = self.reach_edges(h, color)
                header, hub, cells, labels = self.image_row(
                    h, color, y_row, kind.capitalize()
                )
                self.play(
                    LaggedStart(
                        *[FadeIn(dd) for dd in dots], lag_ratio=0.02, run_time=0.8
                    ),
                    FadeIn(header),
                    FadeIn(hub),
                )
                self.play(
                    LaggedStart(
                        *[Create(l) for l in lines], lag_ratio=0.04, run_time=2.0
                    ),
                    LaggedStart(
                        *[FadeIn(c, shift=0.1 * DOWN) for c in cells],
                        lag_ratio=0.1,
                        run_time=2.0,
                    ),
                )
                self.play(
                    LaggedStart(
                        *[FadeIn(l) for l in labels], lag_ratio=0.05, run_time=0.8
                    )
                )
                shown += [dots, lines, header, hub, cells, labels]
                self.marked_next_slide()

            # --- the reveal: only labels / the global view tell them apart ---
            n_good = len(np.unique(self.y[reverse_neighbours(self.idx, g)]))
            n_bad = len(np.unique(self.y[reverse_neighbours(self.idx, b)]))
            new = self.heading(
                "Same local picture, different roles",
                f"the {self.y[g]} gathers {n_good} kind of digit; "
                f"the {self.y[b]} gathers {n_bad} kinds "
                "-- only labels or a global view tell them apart",
            )
            self.play(FadeTransform(head, new))
            head = new
            self.marked_next_slide()
            self.play(*[FadeOut(m) for m in shown])

        # --- take-home message ---
        new = self.heading(
            "Hubs will happen",
            "with even a little noise; good and bad hubs look alike locally",
        )
        self.play(FadeTransform(head, new), FadeOut(veil))
        self.marked_next_slide()
