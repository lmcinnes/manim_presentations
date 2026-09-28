"""Smoke test / demo for umap_talk_common.

    manim-slides render common_demo.py CommonDemo
    manim -s --resolution 2560,1440 common_demo.py PointSizeCheck
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

from umap_talk_common import *

apply_umap_defaults()


def synthetic_mnist_like(n=70_000, seed=0):
    """Three keyframes: noisy init -> clustered layout -> relaxed layout."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 10, n)
    angle = labels * (2 * np.pi / 10) + 0.3
    centres = np.c_[np.cos(angle), np.sin(angle)] * 6.0
    stretch = rng.normal(size=(n, 2)) * np.c_[np.full(n, 0.9), np.full(n, 0.45)]
    rot = angle * 1.7
    local = np.c_[
        stretch[:, 0] * np.cos(rot) - stretch[:, 1] * np.sin(rot),
        stretch[:, 0] * np.sin(rot) + stretch[:, 1] * np.cos(rot),
    ]
    k0 = rng.normal(size=(n, 2)) * 3.0 + 0.4 * centres
    k1 = centres + local
    k2 = 1.25 * centres + 0.6 * local + 0.1 * rng.normal(size=(n, 2))
    return np.stack([k0, k1, k2]), labels


class CommonDemo(UMAPSlide):
    def construct(self):
        keyframes, labels = synthetic_mnist_like()

        brand = EmbeddingCloud(
            keyframes,
            labels_to_rgb(labels, digit_rgb()),
            width=5.2,
            height=5.2,
            center=LEFT * 3.4 + DOWN * 0.4,
        )
        dark = EmbeddingCloud(
            keyframes,
            labels_to_rgb(labels, digit_rgb(min_contrast=3.0)),
            width=5.2,
            height=5.2,
            center=RIGHT * 3.4 + DOWN * 0.4,
        )
        t = ValueTracker(0)
        brand.track(t)
        dark.track(t)

        heads = VGroup(
            Text("COLOR_CYCLE", font_size=26).next_to(brand, UP, buff=0.3),
            Text("darkened to 3:1", font_size=26).next_to(dark, UP, buff=0.3),
        )
        self.add(brand, dark)
        self.play(PMFadeIn(brand), PMFadeIn(dark), FadeIn(heads))
        self.play(t.animate.set_value(2), run_time=3)
        self.marked_next_slide(notes="Keyframe morph, 2 x 70k points.")

        # Dim everything, then lift one class to the top in the highlight colour.
        chosen = np.flatnonzero(labels == 3)
        self.play(PMDim(brand, 0.85), PMDim(dark, 0.85))
        for cloud in (brand, dark):
            cloud.bring_to_front(chosen)
        self.play(
            PMRecolor(brand, brand.target_rgb(chosen, REPEL_COLOR, dim_rest=0.85)),
            PMRecolor(dark, dark.target_rgb(chosen, REPEL_COLOR, dim_rest=0.85)),
        )
        readout = MetricReadout("Trustworthiness", 0.900).to_edge(DOWN, buff=0.35)
        formula = MathTex(r"P(r)=\frac{2}{\pi}\arcsin\frac{w}{2r}").next_to(
            readout, UP, buff=0.3
        )
        self.play(FadeIn(readout), Write(formula))
        self.play(readout.animate_to(0.9731), run_time=1.5)
        self.marked_next_slide()

        # Looping slide: a rotating slab over the left cloud.
        slab = Rectangle(
            width=0.6,
            height=5.6,
            stroke_width=0,
            fill_color=STRUCTURE_COLOR,
            fill_opacity=0.15,
        ).move_to(brand.get_center())
        self.play(FadeIn(slab), PMUndim(brand), PMUndim(dark))
        self.start_loop(notes="Slab sweeps; press to continue.")
        self.play(
            Rotate(slab, PI, about_point=brand.get_center()),
            run_time=3,
            rate_func=linear,
        )
        self.end_loop()

        self.play(PlaceCloud(dark, center=RIGHT * 3.4 + DOWN * 0.4, zoom=0.6))
        self.clear_slide()


class PointSizeCheck(Scene):
    """Still frame for judging point size at the final render resolution.

    A plain Scene, since manim-slides cannot build a slide from a -s still.
    """

    def construct(self):
        keyframes, labels = synthetic_mnist_like()
        brand = EmbeddingCloud(
            keyframes[1], labels_to_rgb(labels, digit_rgb()), width=6, height=6,
            center=LEFT * 3.3,
        )
        dark = EmbeddingCloud(
            keyframes[1], labels_to_rgb(labels, digit_rgb(min_contrast=3.0)),
            width=6, height=6, center=RIGHT * 3.3,
        )
        self.add(brand, dark)
