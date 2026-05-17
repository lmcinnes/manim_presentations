from encodings.idna import dots

from manim import *

import sys

import matplotlib
import pynndescent

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
from data_generation import (
    CircleEmbedding,
    CurvyLoopEmbedding,
    TorusEmbedding,
    effective_resistance_distance_embedding,
)

import numpy as np
import colorcet
from ripser import ripser
import sklearn.decomposition
import sklearn.neighbors
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sklearn.datasets as datasets

import itertools as _it
import types as _types

apply_defaults()


def _make_circular_nudges(thickness):
    """Pixel-nudge offsets forming a filled disc (round PMobject points)."""
    thickness = int(thickness)
    _range = range(-thickness // 2 + 1, thickness // 2 + 1)
    r2 = (thickness / 2.0) ** 2
    nudges = [
        (dy, dx) for dy, dx in _it.product(_range, _range) if dx * dx + dy * dy <= r2
    ]
    return np.array(nudges) if nudges else np.array([[0, 0]])


class PMFadeOut(Animation):
    """FadeOut for PMobject (Cairo renderer).

    Cairo's ``display_point_cloud`` directly overwrites pixel values without
    alpha compositing, so adjusting ``rgbas[:, 3]`` has no visible effect —
    the object stays fully opaque until the remover drops it.

    The fix: interpolate the RGB channels towards the background colour
    (keeping alpha = 1 throughout) so the points smoothly blend into the
    background.  The mobject is removed from the scene when the animation
    ends (``remover=True``).
    """

    def __init__(self, mob: PMobject, **kwargs):
        from manim.utils.color import color_to_rgba
        from manim import config

        self._orig_rgbas = mob.rgbas.copy()
        bg = color_to_rgba(config.background_color)
        self._bg_rgb = np.array(bg[:3], dtype=np.float64)
        super().__init__(mob, remover=True, **kwargs)

    def interpolate_mobject(self, alpha: float) -> None:
        # Blend each point's RGB towards the background; alpha stays 1.
        self.mobject.rgbas[:, :3] = (1.0 - alpha) * self._orig_rgbas[
            :, :3
        ] + alpha * self._bg_rgb


class PMFadeIn(Animation):
    """FadeIn for PMobject (Cairo renderer).

    Mirrors PMFadeOut: blends each point's RGB from the background colour
    to its target colour over the animation duration.  The mobject must
    already be added to the scene before playing this animation (just like
    the built-in FadeIn).
    """

    def __init__(self, mob: PMobject, **kwargs):
        from manim.utils.color import color_to_rgba
        from manim import config

        self._target_rgbas = mob.rgbas.copy()
        bg = color_to_rgba(config.background_color)
        self._bg_rgb = np.array(bg[:3], dtype=np.float64)
        # Start the mob at background colour so it's invisible before play()
        mob.rgbas[:, :3] = self._bg_rgb
        super().__init__(mob, **kwargs)

    def interpolate_mobject(self, alpha: float) -> None:
        # Blend from background towards the target point colours.
        self.mobject.rgbas[:, :3] = (
            1.0 - alpha
        ) * self._bg_rgb + alpha * self._target_rgbas[:, :3]


def target_medoids(data: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Find the medoid (most central point) of each class in the dataset."""
    medoids = []
    for label in np.unique(labels):
        class_points = data[labels == label]
        if len(class_points) == 0:
            continue
        # Compute pairwise distances and find the point with minimal total distance
        dists = sklearn.metrics.pairwise_distances(class_points)
        medoid_idx = np.argmin(dists.sum(axis=1))
        medoids.append(class_points[medoid_idx])
    return np.array(medoids)


def make_legend(colors, labels, medoids, swatch_size=0.25):
    """Create a legend mapping colors to class labels with medoid markers."""
    legend_items = []
    for color, label, medoid in zip(colors, labels, medoids):
        swatch = Square(
            color=color, fill_color=color, fill_opacity=1.0, side_length=swatch_size
        )
        text = Text(str(label), font_size=18).next_to(swatch, RIGHT)
        img = (
            ImageMobject(medoid.reshape(28, 28), invert=True)
            .scale(1.5)
            .next_to(text, RIGHT)
        )
        legend_items.append(Group(swatch, text, img))
    return (
        Group(*legend_items).arrange(DOWN, aligned_edge=LEFT, buff=0.1).to_edge(RIGHT)
    )


def _normalize_to_axes(X):
    """Centre and scale so the cloud fits comfortably inside [-2.5, 2.5]^3."""
    X = X - X.mean(axis=0)
    X = X / (np.abs(X).max() * 0.5)
    return X


def make_eff_res_embedding(data, labels):
    cmap = plt.get_cmap("Spectral")
    norm = mcolors.Normalize(vmin=0, vmax=9)
    rgba_colors = cmap(norm(labels))

    embedding = effective_resistance_distance_embedding(data, target_dim=2)
    embedding = _normalize_to_axes(embedding)
    points = PMobject(stroke_width=1.0)
    points.add_points(
        np.hstack([embedding, np.zeros((embedding.shape[0], 1))]), rgbas=rgba_colors
    )
    points.to_edge(LEFT)
    return points


def make_spectral_embedding(data, labels):
    cmap = plt.get_cmap("Spectral")
    norm = mcolors.Normalize(vmin=0, vmax=9)
    rgba_colors = cmap(norm(labels))

    precomputed_knn = pynndescent.PyNNDescentTransformer(n_neighbors=15).fit_transform(
        data
    )
    precomputed_knn = (precomputed_knn > 0).astype(
        np.float64
    )  # binarize to unweighted graph
    precomputed_knn = precomputed_knn.maximum(precomputed_knn.T)  # symmetrize
    embedding = sklearn.manifold.SpectralEmbedding(
        n_components=2, affinity="precomputed"
    ).fit_transform(precomputed_knn)
    embedding = _normalize_to_axes(embedding)
    points = PMobject(stroke_width=1.0)
    points.add_points(
        np.hstack([embedding, np.zeros((embedding.shape[0], 1))]), rgbas=rgba_colors
    )
    points.to_edge(LEFT).shift(DOWN * 1.5)
    return points


mnist = sklearn.datasets.fetch_openml("mnist_784")
mnist_data = np.ascontiguousarray(mnist.data, dtype=np.uint8)
mnist_target = mnist.target.astype(np.uint8)
mnist_medoids = target_medoids(mnist_data, mnist_target)

fmnist = sklearn.datasets.fetch_openml("Fashion-MNIST")
fmnist_data = np.ascontiguousarray(fmnist.data, dtype=np.uint8)
fmnist_target = fmnist.target.astype(np.uint8)
fmnist_medoids = target_medoids(fmnist_data, fmnist_target)


class _AnimatedUMAPOptimizationBase(TIMCSlide):
    """
    Animated depiction of UMAP's stochastic optimization process.

    Left panel: 10 random points in 2-D, with a highlighted pair (red vs blue).
    Right panel: UMAP's low-dimensional embedding of those points, starting as
                 a random scatter and gradually morphing into the final layout.

    During the morph, the highlighted pair is tracked with red/blue dots in the
    embedding; they start far apart and are pulled together as the optimization
    proceeds, illustrating how UMAP's attractive forces operate on similar points.
    """

    def _load_animation_steps(self):
        raise NotImplementedError("Subclasses must implement _load_animation_steps()")

    def _load_base_data(self):
        raise NotImplementedError("Subclasses must implement _load_base_data()")

    def _load_target_labels(self):
        raise NotImplementedError("Subclasses must implement _load_target_labels()")

    def _get_legend(self, rgb_colors):
        raise NotImplementedError("Subclasses must implement _get_legend()")

    def construct(self):
        try:
            labels = self._load_target_labels()  # shape (70000,)
        except NotImplementedError:
            self.play(Write(Text("Base class -- no animations")))
            self.wait()
            return

        cmap = plt.get_cmap("Spectral")
        norm = mcolors.Normalize(vmin=0, vmax=9)
        rgba_colors = cmap(norm(labels))  # (70000, 4) float64, values in [0, 1]
        legend_colors = cmap(norm(np.arange(0, 10)))  # List of 10 RGBA tuples

        umap_data = self._load_animation_steps()  # shape (n_frames, 70000, 2)
        legend = self._get_legend(legend_colors)

        # Normalize UMAP coordinates to fit Manim's screen (~14 × 8 units).
        # Use the global min/max across all frames to prevent clipping during animation.
        data_min = umap_data.min(axis=(0, 1))  # (2,)
        data_max = umap_data.max(axis=(0, 1))  # (2,)
        vis_center = (data_min + data_max) / 2
        vis_scale = min(
            12.0 / (data_max[0] - data_min[0]),
            7.0 / (data_max[1] - data_min[1]),
        )

        self.camera.get_thickening_nudges = _types.MethodType(
            lambda cam, t: _make_circular_nudges(t), self.camera
        )

        def _to_manim(frames_2d: np.ndarray) -> np.ndarray:
            """Map raw 2-D UMAP coordinates to Manim scene coordinates."""
            xy = (frames_2d - vis_center) * vis_scale
            return np.hstack([xy, np.zeros((xy.shape[0], 1))])

        # Initialize the PointCloud with per-point colors via add_points(rgbas=...).
        # PMobject.set_color() only accepts a single colour; per-point colours must
        # be supplied through the rgbas parameter of add_points().
        STROKE_INITIAL = 6.0
        STROKE_FINAL = 2.0
        n_frames = len(umap_data)
        # Linearly fade stroke_width from STROKE_INITIAL → STROKE_FINAL over a
        # 50-frame window that ends exactly 500 frames before the last frame.
        STROKE_FADE_END = n_frames - 500
        STROKE_FADE_START = STROKE_FADE_END - 50

        points = PMobject(stroke_width=STROKE_INITIAL)
        points.add_points(_to_manim(umap_data[0]), rgbas=rgba_colors)
        # self.add(points)
        self.play(PMFadeIn(points), FadeIn(legend), run_time=2)

        # Track the "Time" or "Epoch"
        epoch_tracker = ValueTracker(0)

        def update_points(obj):
            current_epoch = epoch_tracker.get_value()
            idx = int(current_epoch)
            alpha = current_epoch % 1

            # Linear interpolation between pre-computed frames.
            # When tracker reaches the last frame idx == len-1, alpha == 0 and
            # the branch below would be skipped, leaving points at the penultimate
            # position.  The else clause handles that edge case explicitly.
            if idx < len(umap_data) - 1:
                new_pos = (1 - alpha) * umap_data[idx] + alpha * umap_data[idx + 1]
            else:
                new_pos = umap_data[-1]
            obj.set_points(_to_manim(new_pos))

            # Smoothly interpolate stroke_width down to STROKE_FINAL over the
            # transition window [STROKE_FADE_START, STROKE_FADE_END].
            t = (current_epoch - STROKE_FADE_START) / (
                STROKE_FADE_END - STROKE_FADE_START
            )
            t = max(0.0, min(1.0, t))
            obj.stroke_width = STROKE_INITIAL + (STROKE_FINAL - STROKE_INITIAL) * t

        points.add_updater(update_points)

        # Animate the tracker, not the dots
        self.play(
            epoch_tracker.animate.set_value(len(umap_data) - 1),
            run_time=45,
            rate_func=linear,
        )
        points.remove_updater(update_points)
        self.marked_next_slide()

        if self._show_spectral:

            eff_res_points = make_eff_res_embedding(
                self._load_base_data(), self._load_target_labels()
            ).scale(2.0)
            eff_res_title = Paragraph(
                "Effective Resistance\nDistance Embedding",
                font_size=16,
                alignment="center",
            ).next_to(eff_res_points, UP)
            self.play(points.animate.scale(0.66).shift(RIGHT * 2.4), run_time=2)
            self.play(PMFadeIn(eff_res_points), FadeIn(eff_res_title), run_time=2)

            # spectral_points = make_spectral_embedding(
            #     self._load_base_data(), self._load_target_labels()
            # )
            # spectral_title = Paragraph(
            #     "Spectral\nEmbedding", font_size=16, alignment="center"
            # ).next_to(spectral_points, DOWN)
            # self.play(PMFadeIn(spectral_points), FadeIn(spectral_title), run_time=2)

            self.marked_next_slide()

            self.play(
                PMFadeOut(points),
                FadeOut(legend),
                PMFadeOut(eff_res_points),
                FadeOut(eff_res_title),
                # PMFadeOut(spectral_points),
                # FadeOut(spectral_title),
                run_time=2,
            )
        else:
            self.play(
                PMFadeOut(points), FadeOut(legend), run_time=2
            )  # Fade out points and legend together


class AnimatedUMAPOptimizationMNIST1(_AnimatedUMAPOptimizationBase):

    _show_spectral = True

    def _load_animation_steps(self):
        return np.load("mnist_umap_animation_steps_1.npy")

    def _load_target_labels(self):
        return np.load("mnist_targets.npy")

    def _load_base_data(self):
        return mnist_data

    def _get_legend(self, rgb_colors):
        return make_legend(
            rgb_colors,
            ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            mnist_medoids,
        )


class AnimatedUMAPOptimizationMNIST2(_AnimatedUMAPOptimizationBase):

    _show_spectral = False

    def _load_animation_steps(self):
        return np.load("mnist_umap_animation_steps_2.npy")

    def _load_target_labels(self):
        return np.load("mnist_targets.npy")

    def _load_base_data(self):
        return mnist_data

    def _get_legend(self, rgb_colors):
        return make_legend(
            rgb_colors,
            ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            mnist_medoids,
        )


class AnimatedUMAPOptimizationMNIST3(_AnimatedUMAPOptimizationBase):

    _show_spectral = False

    def _load_animation_steps(self):
        return np.load("mnist_umap_animation_steps_3.npy")

    def _load_target_labels(self):
        return np.load("mnist_targets.npy")

    def _load_base_data(self):
        return mnist_data

    def _get_legend(self, rgb_colors):
        return make_legend(
            rgb_colors,
            ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            mnist_medoids,
        )


class AnimatedUMAPOptimizationFashionMNIST1(_AnimatedUMAPOptimizationBase):

    _show_spectral = True

    def _load_animation_steps(self):
        return np.load("fashion_mnist_umap_animation_steps_1.npy")

    def _load_target_labels(self):
        return np.load("fashion_mnist_targets.npy")

    def _load_base_data(self):
        return fmnist_data

    def _get_legend(self, rgb_colors):
        return make_legend(
            rgb_colors,
            [
                "T-shirt/Top",
                "Trousers",
                "Pullover",
                "Dress",
                "Coat",
                "Sandal",
                "Shirt",
                "Sneaker",
                "Bag",
                "Ankle boot",
            ],
            fmnist_medoids,
        )


class AnimatedUMAPOptimizationFashionMNIST2(_AnimatedUMAPOptimizationBase):

    _show_spectral = False

    def _load_animation_steps(self):
        return np.load("fashion_mnist_umap_animation_steps_2.npy")

    def _load_target_labels(self):
        return np.load("fashion_mnist_targets.npy")

    def _load_base_data(self):
        return fmnist_data

    def _get_legend(self, rgb_colors):
        return make_legend(
            rgb_colors,
            [
                "T-shirt/Top",
                "Trousers",
                "Pullover",
                "Dress",
                "Coat",
                "Sandal",
                "Shirt",
                "Sneaker",
                "Bag",
                "Ankle boot",
            ],
            fmnist_medoids,
        )


class AnimatedUMAPOptimizationFashionMNIST3(_AnimatedUMAPOptimizationBase):

    _show_spectral = False

    def _load_animation_steps(self):
        return np.load("fashion_mnist_umap_animation_steps_3.npy")

    def _load_target_labels(self):
        return np.load("fashion_mnist_targets.npy")

    def _load_base_data(self):
        return fmnist_data

    def _get_legend(self, rgb_colors):
        return make_legend(
            rgb_colors,
            [
                "T-shirt/Top",
                "Trousers",
                "Pullover",
                "Dress",
                "Coat",
                "Sandal",
                "Shirt",
                "Sneaker",
                "Bag",
                "Ankle boot",
            ],
            fmnist_medoids,
        )
