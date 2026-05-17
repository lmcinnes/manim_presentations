from encodings.idna import dots

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
from data_generation import CircleEmbedding, CurvyLoopEmbedding, TorusEmbedding

import numpy as np
import colorcet
from ripser import ripser
import sklearn.decomposition
import sklearn.neighbors
from scipy.stats import gaussian_kde

from manim_slides import Slide, ThreeDSlide

apply_defaults()

# ---------------------------------------------------------------------------
# Helpers shared by the Lissajous knot base classes
# ---------------------------------------------------------------------------
import itertools as _it
import types as _types


def _make_circular_nudges(thickness):
    """Pixel-nudge offsets forming a filled disc (round PMobject points)."""
    thickness = int(thickness)
    _range = range(-thickness // 2 + 1, thickness // 2 + 1)
    r2 = (thickness / 2.0) ** 2
    nudges = [
        (dy, dx) for dy, dx in _it.product(_range, _range) if dx * dx + dy * dy <= r2
    ]
    return np.array(nudges) if nudges else np.array([[0, 0]])


def _normalize_to_axes(X):
    """Centre and scale so the cloud fits comfortably inside [-2.5, 2.5]^3."""
    X = X - X.mean(axis=0)
    X = X / (np.abs(X).max() * 0.5)
    return X


def _axes_to_scene_matrix(axes):
    """Return (origin, U) for vectorised scene-coordinate transforms.

    Enables  scene_coords = origin + proj_3d @ U  without per-point c2p calls.

    origin : shape (3,)   — scene-space position of the axes origin
    U      : shape (3, 3) — rows are the three axis unit vectors in scene space
    """
    o = np.array(axes.c2p(0, 0, 0))
    x_unit = np.array(axes.c2p(1, 0, 0)) - o
    y_unit = np.array(axes.c2p(0, 1, 0)) - o
    z_unit = np.array(axes.c2p(0, 0, 1)) - o
    return o, np.vstack([x_unit, y_unit, z_unit])


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


# ---------------------------------------------------------------------------
# Cairo base class
# ---------------------------------------------------------------------------


class _LissajousBase(ThreeDTIMCSlide):
    """
    Cairo-renderer base for Lissajous-knot embedding scenes.

    Subclasses set class-level attributes to vary the dataset and reduction,
    and may override ``get_embedding`` for custom dimensionality reduction.

    If ``get_embedding`` returns a 3-column array the scene uses ambient
    camera rotation.  For >3 columns it runs a grand tour instead.
    """

    _n_samples: int = 2000
    _seed: int = 42
    _generator_kwargs: dict = dict(
        target_dim=4,
        radius=2.0,
        n_planes=2,
        noise=0.0,
        noise_hd=0.05,
        dataset_type="highdim_loop",
    )
    _rotation_wait: float = 12.0
    _stroke_width: int = 16
    _grand_tour_n_legs: int = 8
    _grand_tour_leg_time: float = 3.0

    def get_embedding(self, X_embedded: np.ndarray) -> np.ndarray:
        """Return an (N, k) array to plot.  k==3 → camera rotation; k>3 → grand tour."""
        return sklearn.decomposition.PCA(n_components=3).fit_transform(X_embedded)

    def construct(self):
        generator = CurvyLoopEmbedding(n_samples=self._n_samples, seed=self._seed)
        _, X_embedded, _ = generator.generate_dataset(**self._generator_kwargs)
        embedding = self.get_embedding(X_embedded)

        axes = ThreeDAxes(x_range=[-2.5, 2.5], y_range=[-2.5, 2.5], z_range=[-2.5, 2.5])
        self.camera.get_thickening_nudges = _types.MethodType(
            lambda cam, t: _make_circular_nudges(t), self.camera
        )

        if embedding.shape[1] == 3:
            self._run_camera_rotation3d(embedding, axes)
        else:
            self._run_grand_tour(embedding, axes)

    def _run_camera_rotation3d(self, embedding: np.ndarray, axes) -> None:
        points = self.create_points(embedding, axes)
        self.move_camera(phi=75 * DEGREES, theta=30 * DEGREES)
        self.play(Create(axes))
        self.wait(1)
        self.play(PMFadeIn(points))
        self.begin_ambient_camera_rotation(rate=0.75)
        self.wait(self._rotation_wait)
        self.stop_ambient_camera_rotation()

        self.marked_next_slide()
        self.begin_ambient_camera_rotation(rate=4 * PI / self._rotation_wait)
        self.wait(self._rotation_wait / 2)
        self.stop_ambient_camera_rotation()
        self.marked_next_slide()

        self.play(PMFadeOut(points), FadeOut(axes))
        self.clear_slide()

    def _run_grand_tour(self, embedding: np.ndarray, axes) -> None:
        d = embedding.shape[1]
        origin, U = _axes_to_scene_matrix(axes)

        # Initial frame: first three PCA components (most informative)
        P0 = (
            sklearn.decomposition.PCA(n_components=3).fit(embedding).components_.T
        )  # (d, 3)

        # Random Stiefel frames; deterministic via seed
        rng = np.random.default_rng(self._seed)
        rand_frames = [
            np.linalg.qr(rng.standard_normal((d, 3)))[0][:, :3]
            for _ in range(self._grand_tour_n_legs)
        ]
        # Append P0 so the tour closes back to the initial PCA view
        frames = [P0] + rand_frames + [P0]

        # Precompute fixed centering and global scale for smooth animation.
        # Max L2 norm of any point bounds every projection direction, so
        # scale * 0.5 guarantees projected values stay within [-2, 2] in axes coords.
        X_centered = embedding - embedding.mean(axis=0)
        global_scale = np.sqrt((X_centered**2).sum(axis=1).max()) * 0.5

        # Build PMobject; colors are fixed to the loop parameter throughout
        proj0 = (X_centered @ P0) / global_scale
        pm = self.create_points(proj0, axes)

        self.move_camera(phi=75 * DEGREES, theta=30 * DEGREES)
        self.play(Create(axes))
        self.wait(1)
        self.play(PMFadeIn(pm))

        # Animate each leg — no camera rotation during the tour
        for P_start, P_end in zip(frames[:-1], frames[1:]):
            t_tr = ValueTracker(0.0)

            def make_updater(P_s, P_e, tracker):
                def updater(mob):
                    t = tracker.get_value()
                    P_interp = (1.0 - t) * P_s + t * P_e
                    # Polar decomposition (nearest orthogonal matrix via SVD) gives
                    # a unique, smooth map from the linearly-interpolated frame to
                    # St(d,3), avoiding the sign-flip discontinuities of QR.
                    _Usvd, _, _Vt = np.linalg.svd(P_interp, full_matrices=False)
                    P_orth = _Usvd @ _Vt
                    proj = (X_centered @ P_orth) / global_scale
                    mob.points[:] = (origin + proj @ U).astype(np.float64)

                return updater

            updater = make_updater(P_start, P_end, t_tr)
            pm.add_updater(updater)
            self.play(
                t_tr.animate.set_value(1.0),
                run_time=self._grand_tour_leg_time,
                rate_func=linear,
            )
            pm.remove_updater(updater)

        # Closing beat: slow ambient camera rotation after the loop closes
        self.begin_ambient_camera_rotation(rate=0.75)
        self.wait(self._rotation_wait)
        self.stop_ambient_camera_rotation()

        self.marked_next_slide()
        self.begin_ambient_camera_rotation(rate=4 * PI / self._rotation_wait)
        self.wait(self._rotation_wait / 2)
        self.stop_ambient_camera_rotation()
        self.marked_next_slide()

        self.play(PMFadeOut(pm), FadeOut(axes))
        self.clear_slide()

    def create_points(self, data: np.ndarray, axes) -> PMobject:
        from manim.utils.color import color_to_rgba

        coords = np.array(
            [axes.c2p(*(x if len(x) == 3 else np.append(x, 0))) for x in data],
            dtype=np.float64,
        )
        ts = np.linspace(0, 1, len(data))
        rgbas = np.array(
            [
                color_to_rgba(colorcet.CET_C9[int(t * (len(colorcet.CET_C9) - 1))])
                for t in ts
            ]
        )
        pm = PMobject(stroke_width=self._stroke_width)
        pm.add_points(coords, rgbas=rgbas)
        return pm


# ---------------------------------------------------------------------------
# OpenGL base class  (use with --renderer=opengl)
# ---------------------------------------------------------------------------


class _LissajousBaseOpenGL(ThreeDSlide):
    """
    OpenGL-renderer base for Lissajous-knot embedding scenes.
    Render with:  manim-slides render slides.py <ClassName> --renderer=opengl

    If ``get_embedding`` returns a 3-column array the scene uses ambient
    camera rotation.  For >3 columns it runs a grand tour instead.
    """

    _n_samples: int = 2000
    _seed: int = 42
    _generator_kwargs: dict = dict(
        target_dim=4,
        radius=2.0,
        n_planes=2,
        noise=0.0,
        noise_hd=0.05,
        dataset_type="highdim_loop",
    )
    _rotation_wait: float = 8.0
    _n_color_groups: int = 64
    _dot_stroke_width: float = 4.0
    _grand_tour_n_legs: int = 8
    _grand_tour_leg_time: float = 3.0

    def get_embedding(self, X_embedded: np.ndarray) -> np.ndarray:
        """Return an (N, k) array to plot.  k==3 → camera rotation; k>3 → grand tour."""
        return sklearn.decomposition.PCA(n_components=3).fit_transform(X_embedded)

    def construct(self):
        generator = CurvyLoopEmbedding(n_samples=self._n_samples, seed=self._seed)
        _, X_embedded, _ = generator.generate_dataset(**self._generator_kwargs)
        embedding = self.get_embedding(X_embedded)

        axes = ThreeDAxes(x_range=[-2.5, 2.5], y_range=[-2.5, 2.5], z_range=[-2.5, 2.5])

        if embedding.shape[1] == 3:
            dots = self._run_camera_rotation3d(embedding, axes)
        else:
            dots = self._run_grand_tour(embedding, axes)

        self.play(PMFadeOut(dots), FadeOut(axes))

    def _run_camera_rotation3d(self, embedding: np.ndarray, axes) -> None:
        points = self.create_points(embedding, axes)
        self.move_camera(phi=75 * DEGREES, theta=30 * DEGREES)
        self.play(Create(axes))
        self.wait(1)
        self.play(FadeIn(points))
        self.begin_ambient_camera_rotation(rate=0.75)
        self.wait(self._rotation_wait)
        self.stop_ambient_camera_rotation()
        return points

    def _run_grand_tour(self, embedding: np.ndarray, axes) -> None:
        d = embedding.shape[1]
        origin, U = _axes_to_scene_matrix(axes)

        # Initial frame: first three PCA components (most informative)
        P0 = (
            sklearn.decomposition.PCA(n_components=3).fit(embedding).components_.T
        )  # (d, 3)

        # Random Stiefel frames; deterministic via seed
        rng = np.random.default_rng(self._seed)
        rand_frames = [
            np.linalg.qr(rng.standard_normal((d, 3)))[0][:, :3]
            for _ in range(self._grand_tour_n_legs)
        ]
        # Append P0 so the tour closes back to the initial PCA view
        frames = [P0] + rand_frames + [P0]

        # Precompute fixed centering and global scale for smooth animation.
        # Max L2 norm of any point bounds every projection direction, so
        # scale * 0.5 guarantees projected values stay within [-2, 2] in axes coords.
        X_centered = embedding - embedding.mean(axis=0)
        global_scale = np.sqrt((X_centered**2).sum(axis=1).max()) * 0.5

        # Build DotCloud VGroup; keep group_indices for the updater
        proj0 = (X_centered @ P0) / global_scale
        N = len(proj0)
        coords0 = (origin + proj0 @ U).astype(np.float32)
        group_indices = np.array_split(np.arange(N), self._n_color_groups)
        dot_clouds = VGroup()
        for i, indices in enumerate(group_indices):
            t_mid = i / max(self._n_color_groups - 1, 1)
            color = colorcet.CET_C9[int(t_mid * (len(colorcet.CET_C9) - 1))]
            dc = DotCloud(color=color, stroke_width=self._dot_stroke_width)
            dc.set_points(coords0[indices])
            dot_clouds.add(dc)

        self.move_camera(phi=75 * DEGREES, theta=30 * DEGREES)
        self.play(Create(axes))
        self.wait(1)
        self.play(FadeIn(dot_clouds))

        # Animate each leg — no camera rotation during the tour
        for P_start, P_end in zip(frames[:-1], frames[1:]):
            t_tr = ValueTracker(0.0)

            def make_updater(P_s, P_e, tracker, g_indices):
                def updater(mob):
                    t = tracker.get_value()
                    P_interp = (1.0 - t) * P_s + t * P_e
                    # Polar decomposition (nearest orthogonal matrix via SVD) gives
                    # a unique, smooth map from the linearly-interpolated frame to
                    # St(d,3), avoiding the sign-flip discontinuities of QR.
                    _Usvd, _, _Vt = np.linalg.svd(P_interp, full_matrices=False)
                    P_orth = _Usvd @ _Vt
                    proj = (X_centered @ P_orth) / global_scale
                    coords = (origin + proj @ U).astype(np.float32)
                    for dc, indices in zip(mob, g_indices):
                        dc.set_points(coords[indices])

                return updater

            updater = make_updater(P_start, P_end, t_tr, group_indices)
            dot_clouds.add_updater(updater)
            self.play(
                t_tr.animate.set_value(1.0),
                run_time=self._grand_tour_leg_time,
                rate_func=linear,
            )
            dot_clouds.remove_updater(updater)

        # Closing beat: slow ambient camera rotation after the loop closes
        self.begin_ambient_camera_rotation(rate=0.75)
        self.wait(self._rotation_wait)
        self.stop_ambient_camera_rotation()
        return dot_clouds[0]

    def create_points(self, data: np.ndarray, axes):
        from manim import config, RendererType

        coords = np.array(
            [axes.c2p(*(x if len(x) == 3 else np.append(x, 0))) for x in data],
        )

        if config.renderer == RendererType.OPENGL:
            coords = coords.astype(np.float32)
            n = len(data)
            group_indices = np.array_split(np.arange(n), self._n_color_groups)
            result = VGroup()
            for i, indices in enumerate(group_indices):
                t_mid = i / max(self._n_color_groups - 1, 1)
                color = colorcet.CET_C9[int(t_mid * (len(colorcet.CET_C9) - 1))]
                dc = DotCloud(color=color, stroke_width=self._dot_stroke_width)
                dc.set_points(coords[indices])
                result.add(dc)
            return result
        else:
            # Cairo fallback: PMobject with per-point colours
            from manim.utils.color import color_to_rgba

            coords = coords.astype(np.float64)
            ts = np.linspace(0, 1, len(data))
            rgbas = np.array(
                [
                    color_to_rgba(colorcet.CET_C9[int(t * (len(colorcet.CET_C9) - 1))])
                    for t in ts
                ]
            )
            pm = PMobject(stroke_width=self._dot_stroke_width)
            pm.add_points(coords, rgbas=rgbas)
            return pm


# ---------------------------------------------------------------------------
# Shared generator kwargs for the high-dimensional variants
# ---------------------------------------------------------------------------

_HIGH_D_KWARGS = dict(
    target_dim=512,
    radius=2.0,
    n_planes=5,
    noise=0.01,
    noise_hd=0.175,
    dataset_type="highdim_loop",
)

# ---------------------------------------------------------------------------
# Concrete scenes
# ---------------------------------------------------------------------------


class LissajousKnot(_LissajousBase):
    """Low-dimensional Lissajous knot, PCA to 3D (Cairo renderer)."""

    pass


class LissajousKnotOpenGL(_LissajousBaseOpenGL):
    """Low-dimensional Lissajous knot, PCA to 3D (OpenGL renderer)."""

    pass


class HighDLissajousPCA(_LissajousBase):
    """High-dimensional Lissajous knot projected with PCA (Cairo renderer)."""

    _generator_kwargs = _HIGH_D_KWARGS


class HighDLissajousPCAOpenGL(_LissajousBaseOpenGL):
    """High-dimensional Lissajous knot projected with PCA (OpenGL renderer)."""

    _generator_kwargs = _HIGH_D_KWARGS


class HighDLissajousUMAP(_LissajousBase):
    """High-dimensional Lissajous knot projected with UMAP (Cairo renderer)."""

    _generator_kwargs = _HIGH_D_KWARGS

    def get_embedding(self, X_embedded):
        import umap

        result = umap.UMAP(n_components=3, random_state=42).fit_transform(X_embedded)
        return _normalize_to_axes(result)


class HighDLissajousEffRes(_LissajousBase):
    """High-dimensional Lissajous knot, effective-resistance embedding (Cairo renderer)."""

    _generator_kwargs = _HIGH_D_KWARGS

    def get_embedding(self, X_embedded):
        from data_generation import effective_resistance_distance_embedding

        result = effective_resistance_distance_embedding(X_embedded)
        return _normalize_to_axes(result)


class HighDLissajousSpectral(_LissajousBase):
    """High-dimensional Lissajous knot projected with spectral embedding (Cairo renderer)."""

    _generator_kwargs = _HIGH_D_KWARGS

    def get_embedding(self, X_embedded):
        from sklearn.manifold import SpectralEmbedding

        result = SpectralEmbedding(
            n_components=3, n_neighbors=15, random_state=42
        ).fit_transform(X_embedded)
        return _normalize_to_axes(result)


_HARD_HIGH_D_KWARGS = dict(
    target_dim=512,
    radius=2.0,
    n_planes=7,
    noise=0.01,
    noise_hd=0.25,
    dataset_type="highdim_loop",
)


class HardHighDLissajousPCA(_LissajousBase):
    """High-dimensional Lissajous knot projected with PCA (Cairo renderer)."""

    _generator_kwargs = _HARD_HIGH_D_KWARGS


class HardHighDLissajousPCA5D(_LissajousBase):
    """High-dimensional Lissajous knot projected with PCA (Cairo renderer)."""

    _generator_kwargs = _HARD_HIGH_D_KWARGS

    def get_embedding(self, X_embedded):
        result = sklearn.decomposition.PCA(n_components=5).fit_transform(X_embedded)
        return _normalize_to_axes(result)


class HardHighDLissajousUMAP(_LissajousBase):
    """High-dimensional Lissajous knot projected with UMAP (Cairo renderer)."""

    _generator_kwargs = _HARD_HIGH_D_KWARGS

    def get_embedding(self, X_embedded):
        import umap

        result = umap.UMAP(n_components=3, random_state=42).fit_transform(X_embedded)
        return _normalize_to_axes(result)


class HardHighDLissajousUMAP5D(_LissajousBase):
    """High-dimensional Lissajous knot projected with UMAP (Cairo renderer)."""

    _generator_kwargs = _HARD_HIGH_D_KWARGS

    def get_embedding(self, X_embedded):
        import umap

        result = umap.UMAP(
            n_components=5, random_state=42, repulsion_strength=0.5
        ).fit_transform(X_embedded)
        return _normalize_to_axes(result)


class HardHighDLissajousEffRes(_LissajousBase):
    """High-dimensional Lissajous knot, effective-resistance embedding (Cairo renderer)."""

    _generator_kwargs = _HARD_HIGH_D_KWARGS

    def get_embedding(self, X_embedded):
        from data_generation import effective_resistance_distance_embedding

        result = effective_resistance_distance_embedding(X_embedded)
        return _normalize_to_axes(result)


class HardHighDLissajousEffRes5D(_LissajousBase):
    """High-dimensional Lissajous knot, effective-resistance embedding (Cairo renderer)."""

    _generator_kwargs = _HARD_HIGH_D_KWARGS

    def get_embedding(self, X_embedded):
        from data_generation import effective_resistance_distance_embedding

        result = effective_resistance_distance_embedding(X_embedded, target_dim=5)
        return _normalize_to_axes(result)


# ---------------------------------------------------------------------------
# Introductory scene: parametric form → 2D projections → 3D knot → high-D
# ---------------------------------------------------------------------------


class LissajousKnotIntroduction(ThreeDTIMCSlide):
    """
    Two-part introduction to Lissajous knots.

    Part 1: Parametric equations, three 2D projections in the top half, arrows
            pointing down to a rotating 3D curve in the bottom half.
    Part 2: Extension to higher dimensions – generalised formula and all
            plane-pair projections for a 4D example.
    """

    _n_3d = (3, 5, 7)
    _phases_3d = (0.1, 0.7, 0)
    _n_hd = (2, 3, 5, 7)
    _phases_hd = (0.0, PI / 4, PI / 2, 3 * PI / 4)
    _n_points = 1000
    _rotation_time = 12.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _curve_data(self, ns, phases):
        """Return an (d, N) array of cosine components."""
        t = np.linspace(0, 2 * PI, self._n_points)
        return np.array([np.cos(n * t + p) for n, p in zip(ns, phases)])

    def _projection_plot(self, data, i, j, dim_names, size=1.8, n_colors=64):
        """Small Axes with the (i, j) projection of *data* and axis labels."""
        ax = Axes(
            x_range=[-1.2, 1.2, 1],
            y_range=[-1.2, 1.2, 1],
            x_length=size,
            y_length=size,
            axis_config={
                "include_tip": False,
                "color": DEFAULT_COLOR,
                "stroke_width": 1.5,
            },
        )
        coords = np.array([ax.c2p(x, y) for x, y in zip(data[i], data[j])])
        n = len(coords)
        curve = VGroup()
        for k in range(n_colors):
            start_idx = int(k * n / n_colors)
            end_idx = min(int((k + 1) * n / n_colors) + 1, n)
            t_mid = k / max(n_colors - 1, 1)
            color = colorcet.CET_C9[int(t_mid * (len(colorcet.CET_C9) - 1))]
            seg = VMobject(stroke_width=2.5, color=color)
            seg.set_points_smoothly(coords[start_idx:end_idx])
            curve.add(seg)
        x_lbl = MathTex(dim_names[i], font_size=22, color=DEFAULT_COLOR)
        x_lbl.next_to(ax, DOWN, buff=0.15)
        y_lbl = MathTex(dim_names[j], font_size=22, color=DEFAULT_COLOR)
        y_lbl.next_to(ax, LEFT, buff=0.15)
        return VGroup(ax, curve, x_lbl, y_lbl)

    # ------------------------------------------------------------------
    # construct
    # ------------------------------------------------------------------

    def construct(self):
        self._part1()
        self.marked_next_slide()
        self.clear_slide()
        self._part2()
        self.marked_next_slide()

    # ------------------------------------------------------------------
    # Part 1 – parametric form + 2D projections + rotating 3D knot
    # ------------------------------------------------------------------

    def _part1(self):
        nx, ny, nz = self._n_3d
        phi_x, phi_y, phi_z = self._phases_3d
        data = self._curve_data(self._n_3d, self._phases_3d)

        # Parametric equations at the top (fixed-in-frame)
        title = Text("Lissajous knots", font_size=56).shift(UP * 1.5)
        eq = MathTex(
            r"x = \cos(n_x t + \phi_x)\\"
            r"y = \cos(n_y t + \phi_y)\\"
            r"z = \cos(n_z t + \phi_z)",
            font_size=56,
        ).next_to(title, DOWN, buff=0.5)
        self.play(Write(title))
        self.play(Write(eq))
        self.wait()
        self.marked_next_slide()
        self.clear_slide()

        # Three 2D projection plots arranged across the upper half
        dim_names_3d = ["x", "y", "z"]
        plot_xy = self._projection_plot(data, 0, 1, dim_names_3d)
        plot_xz = self._projection_plot(data, 0, 2, dim_names_3d)
        plot_yz = self._projection_plot(data, 1, 2, dim_names_3d)
        plots = VGroup(plot_xy, plot_xz, plot_yz).arrange(RIGHT, buff=1.0)

        self.play(
            LaggedStart(*[Create(p[0]) for p in plots], lag_ratio=0.5),
            LaggedStart(*[Write(p[2:]) for p in plots], lag_ratio=0.5),
        )
        self.play(
            LaggedStart(*[Create(p[1], run_time=2) for p in plots], lag_ratio=0.5)
        )

        self.wait()

        # Shift plots to top edge, then pin them as screen-space overlays
        # before the camera tilts so their positions are preserved correctly.
        self.play(plots.animate.to_edge(UP, buff=0.5))

        # Build the 3D axes first so we know where the origin lands in
        # screen space – we'll point the arrows precisely at it.
        axes_3d = ThreeDAxes(
            x_range=[-1.2, 1.2],
            y_range=[-1.2, 1.2],
            z_range=[-1.2, 1.2],
            x_length=3.5,
            y_length=3.5,
            z_length=3.5,
        )
        axes_3d.shift(DOWN * 1.8)
        origin_3d = axes_3d.get_origin()

        # Arcing arrows from each projection plot down to the 3D axes origin.
        # Each arrow stops `arrow_gap` units short of the origin so it doesn't
        # impinge on the 3D plot region. Positive angle arcs CCW, negative CW.
        # arrow_angles = [PI / 4, PI / 12, -PI / 4]
        arrow_angles = [PI / 4, 0, -PI / 4]
        arrow_gaps = [2.5, 2.0, 2.5]
        arrows = VGroup()
        for p, a, arrow_gap in zip(
            [plot_xy, plot_xz, plot_yz], arrow_angles, arrow_gaps
        ):
            start = p.get_bottom() + 0.1 * RIGHT + DOWN * 0.1
            direction = origin_3d - start
            direction = direction / np.linalg.norm(direction)
            end = origin_3d - direction * arrow_gap
            arrows.add(
                CurvedArrow(
                    start,
                    end,
                    angle=a,
                    color=ACCENT_COLOR,
                    stroke_width=8,
                    tip_length=0.2,
                )
            )
        self.play(*[Create(a) for a in arrows])

        # 3D knot coloured by parameter t using CET_C9
        n_seg = 64
        curve_3d = VGroup()
        for k in range(n_seg):
            t_start = 2 * PI * k / n_seg
            t_end = 2 * PI * (k + 1) / n_seg
            t_mid = (k + 0.5) / n_seg
            seg_color = colorcet.CET_C9[int(t_mid * (len(colorcet.CET_C9) - 1))]
            seg = ParametricFunction(
                lambda t, _nx=nx, _ny=ny, _nz=nz, _px=phi_x, _py=phi_y, _pz=phi_z: axes_3d.c2p(
                    np.cos(_nx * t + _px),
                    np.cos(_ny * t + _py),
                    np.cos(_nz * t + _pz),
                ),
                t_range=[t_start, t_end],
                color=seg_color,
                stroke_width=6,
            )
            curve_3d.add(seg)
        self.play(Create(axes_3d), Create(curve_3d), run_time=3)

        self.wait()

        self.play(
            Rotating(
                axes_3d,
                axis=UP,
                about_point=axes_3d.get_center(),
                run_time=self._rotation_time,
            ),
            Rotating(
                curve_3d,
                axis=UP,
                about_point=axes_3d.get_center(),
                run_time=self._rotation_time,
            ),
        )

        # # Rotate the object group rather than the camera so the screen-space
        # # overlays above are completely undisturbed.
        # knot_group = VGroup(axes_3d, curve_3d)
        # knot_group.add_updater(
        #     lambda m, dt: m.rotate(dt * 0.5, axis=UP, about_point=m.get_center())
        # )
        # self.wait(self._rotation_time)
        # knot_group.clear_updaters()

    # ------------------------------------------------------------------
    # Part 2 – higher-dimensional extension
    # ------------------------------------------------------------------

    def _part2(self):

        nd = len(self._n_hd)
        n_vals_str = ", ".join(str(n) for n in self._n_hd)

        title = Text("Higher-dimensional extension", font_size=56).shift(UP)
        eq = MathTex(
            r"x_i = \cos(n_i\, t + \phi_i), \quad i = 1, \ldots, d",
            font_size=72,
        ).next_to(title, DOWN, buff=0.5)
        self.play(Write(title))
        self.play(Write(eq))
        self.wait()
        self.marked_next_slide()
        self.clear_slide()

        example_title = Text(f"Example in {nd}D", font_size=56).shift(UP)
        note = VGroup(
            example_title,
            Text(
                f"n = ({n_vals_str})  — all pairwise coprime",
                font_size=48,
            ).next_to(example_title, DOWN, buff=0.25),
        )
        self.play(Write(note))
        self.wait()
        self.play(note.animate.to_edge(UP, buff=0.5))

        # All C(d, 2) = 6 plane-pair projections for the 4D example
        data = self._curve_data(self._n_hd, self._phases_hd)
        dim_names_hd = [f"x_{{{i + 1}}}" for i in range(nd)]
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        all_plots = VGroup(
            *[
                self._projection_plot(data, i, j, dim_names_hd, size=1.5)
                for i, j in pairs
            ]
        )
        row1 = VGroup(*list(all_plots)[:3]).arrange(RIGHT, buff=0.6)
        row2 = VGroup(*list(all_plots)[3:]).arrange(RIGHT, buff=0.6)
        grid = VGroup(row1, row2).arrange(DOWN, buff=0.5)
        grid.next_to(note, DOWN, buff=0.5)
        self.play(
            LaggedStart(*[Create(p[0]) for p in all_plots], lag_ratio=0.5),
            LaggedStart(*[Write(p[2:]) for p in all_plots], lag_ratio=0.5),
        )
        self.play(
            LaggedStart(*[Create(p[1], run_time=2) for p in all_plots], lag_ratio=0.5)
        )
        self.wait(1)
