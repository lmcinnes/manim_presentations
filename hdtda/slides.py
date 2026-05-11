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

apply_defaults()

import ot  # Python Optimal Transport library


def prepare_diagram_for_transport(dgm, epsilon=1e-10):
    """
    Prepare a persistence diagram for optimal transport.

    Removes infinite points and adds a small epsilon to zero persistence points.

    Parameters
    ----------
    dgm : ndarray of shape (n_points, 2)
        Persistence diagram with birth and death times
    epsilon : float
        Small value to add to diagonal points

    Returns
    -------
    dgm_clean : ndarray
        Cleaned diagram
    """
    # Remove infinite points
    # dgm_clean = dgm[np.isfinite(dgm).all(axis=1)]
    dgm_clean = dgm.copy()
    dgm_clean[~np.isfinite(dgm_clean)] = 1e8

    # Handle points on/near diagonal (birth ≈ death)
    persistence = dgm_clean[:, 1] - dgm_clean[:, 0]
    on_diagonal = persistence < epsilon
    dgm_clean[on_diagonal, 1] = dgm_clean[on_diagonal, 0] + epsilon

    return dgm_clean


def compute_transport_plan(dgm1, dgm2, homology_dim, reg=0.01, unbalanced_reg=0.1):
    """
    Compute partial/unbalanced optimal transport plan between two persistence diagrams.

    Uses entropic regularization and unbalanced OT to handle diagrams with
    different numbers of points.

    Parameters
    ----------
    dgm1, dgm2 : ndarray
        Persistence diagrams
    homology_dim : int
        Homology dimension (0 or 1)
    reg : float
        Entropic regularization parameter
    unbalanced_reg : float
        Unbalanced regularization parameter (smaller = more unbalanced)

    Returns
    -------
    plan : ndarray
        Transport plan matrix
    """
    dgm1_clean = prepare_diagram_for_transport(dgm1)
    dgm2_clean = prepare_diagram_for_transport(dgm2)
    print(
        f"Computing transport plan for H{homology_dim} with {len(dgm1_clean)} and {len(dgm2_clean)} points"
    )
    plan = ot.solve_sample(dgm1_clean, dgm2_clean).plan

    return plan, dgm1_clean, dgm2_clean


def interpolate_diagrams(dgm1, dgm2, plan, dgm1_clean, dgm2_clean, t):
    """
    Interpolate between two diagrams using a transport plan.

    Parameters
    ----------
    dgm1, dgm2 : ndarray
        Original persistence diagrams
    plan : ndarray
        Transport plan matrix
    dgm1_clean, dgm2_clean : ndarray
        Cleaned diagrams used for transport
    t : float
        Interpolation parameter in [0, 1]

    Returns
    -------
    dgm_interp : ndarray
        Interpolated diagram
    """
    if len(dgm1_clean) == 0 or len(dgm2_clean) == 0:
        # Handle edge cases
        if t < 0.5:
            return dgm1_clean if len(dgm1_clean) > 0 else dgm2_clean
        else:
            return dgm2_clean if len(dgm2_clean) > 0 else dgm1_clean

    # # Threshold the plan to get significant correspondences
    # threshold = plan.max() * 0.75
    # significant = plan > threshold
    max_per_point = plan.argmax(axis=1)
    significant = np.zeros_like(plan)
    for i in range(significant.shape[0]):
        significant[i, max_per_point[i]] = 1

    # Create interpolated points
    interp_points = []

    for i in range(len(dgm1_clean)):
        for j in range(len(dgm2_clean)):
            if significant[i, j]:
                # Interpolate along geodesic (straight line in diagram space)
                p1 = dgm1_clean[i]
                p2 = dgm2_clean[j]
                p_interp = (1 - t) * p1 + t * p2

                # Weight by transport plan
                weight = plan[i, j]

                # # Add multiple copies based on weight (discretized)
                # n_copies = max(1, int(weight))  # Scale for visibility
                # for _ in range(n_copies):
                interp_points.append(p_interp)

    if len(interp_points) == 0:
        # Fallback: just use endpoints
        return dgm1_clean if t < 0.5 else dgm2_clean

    return np.array(interp_points)


def create_interpolated_sequence(dgms, n_interp=3, min_dim=2, max_dim=256):
    """
    Create interpolated sequence between persistence diagrams.

    Parameters
    ----------
    dgms : list of lists
        List of persistence diagrams (each containing H0 and H1)
    n_interp : int
        Number of interpolation frames between each pair of diagrams

    Returns
    -------
    dgms_interp : list
        Extended list with interpolated diagrams
    dimensions : list
        Corresponding dimension values for each frame
    """
    dgms_interp = []
    dimensions = []

    n_dgms = len(dgms)

    for idx in range(n_dgms - 1):
        print(f"Interpolating between diagram {idx} and {idx + 1}")
        dgm1 = dgms[idx]
        dgm2 = dgms[idx + 1]

        dim1 = int((idx / (n_dgms - 1)) * (max_dim - min_dim) + min_dim)
        dim2 = int(((idx + 1) / (n_dgms - 1)) * (max_dim - min_dim) + min_dim)

        # Compute transport plans for each homology dimension
        plans = []
        dgms_clean = []

        for h_dim in range(len(dgm1)):  # H0 and H1
            plan, d1_clean, d2_clean = compute_transport_plan(
                dgm1[h_dim], dgm2[h_dim], h_dim, reg=0.01, unbalanced_reg=0.1
            )
            plans.append(plan)
            dgms_clean.append((d1_clean, d2_clean))

        # Create interpolated frames
        for i in range(n_interp + 1):
            t = i / n_interp

            # Interpolate each homology dimension
            dgm_interp = []
            for h_dim in range(len(dgm1)):
                plan = plans[h_dim]
                d1_clean, d2_clean = dgms_clean[h_dim]

                d_interp = interpolate_diagrams(
                    dgm1[h_dim], dgm2[h_dim], plan, d1_clean, d2_clean, t
                )
                dgm_interp.append(d_interp)

            dgms_interp.append(dgm_interp)

            # Linear interpolation of dimension label
            dim_interp = (1 - t) * dim1 + t * dim2
            dimensions.append(dim_interp)

    # Add the last diagram
    dgms_interp.append(dgms[-1])
    dimensions.append(
        int(((n_dgms - 1) / (n_dgms - 1)) * (max_dim - min_dim) + min_dim)
    )

    return dgms_interp, dimensions


def rotation_matrix_to_axis_angle(R):
    angle = np.arccos((np.trace(R) - 1) / 2)
    if np.isclose(angle, 0):
        return np.array([1, 0, 0]), 0

    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (
        2 * np.sin(angle)
    )

    return axis, angle


def _pd_optimal_step(max_val: float) -> float:
    """Pick a 'round' tick step for a persistence-diagram axis."""
    if max_val < 0.5:
        return 0.1
    if max_val < 1.0:
        return 0.2
    if max_val < 2.5:
        return 0.5
    if max_val < 5.0:
        return 1.0
    if max_val < 10.0:
        return 2.0
    if max_val < 20.0:
        return 5.0
    return 10.0


class PersistanceDiagramExpectations(TIMCSlide):
    def construct(self):
        # 1. Setup Data
        generator = CircleEmbedding(n_samples=300, seed=42)

        noise_hd = 0.16
        generator_kwds = {"radius": 1.0, "noise_2d": 0.0, "embedding": "linear"}

        dimension = 4
        X_2d, X_embedded, _, metadata = generator.generate_dataset(
            target_dim=dimension,
            noise_hd=noise_hd,
            **generator_kwds,
        )
        dataset = X_embedded
        diagram = ripser(X_embedded, maxdim=1)["dgms"]

        finite_points = np.concatenate(
            [d[np.isfinite(d).all(axis=1)] for d in diagram if len(d) > 0]
        )
        frame_max = np.max(finite_points) * 1.1 if finite_points.size > 0 else 1.0

        # --- CREATE NEW FRAME COMPONENTS ---
        # Fixed tick steps (e.g., steps of 0.5 or 1.0 depending on scale)
        tick_step = self.get_optimal_step(frame_max)

        new_axes = Axes(
            x_range=[0, frame_max, tick_step],
            y_range=[0, frame_max, tick_step],
            x_length=6,
            y_length=6,
            axis_config={
                "include_tip": False,
                "include_ticks": True,
                "numbers_to_exclude": [0],
            },
        ).add_coordinates()

        new_diag = Line(
            start=new_axes.c2p(0, 0),
            end=new_axes.c2p(frame_max, frame_max),
            color=GRAY,
            stroke_width=2,
        )

        # Infinity Line at the very top of the Y-axis
        inf_y = new_axes.c2p(0, frame_max)[1]
        new_inf_line = DashedLine(
            start=[new_axes.c2p(0, 0)[0], inf_y, 0],
            end=[new_axes.c2p(frame_max, 0)[0], inf_y, 0],
            color=GRAY_C,
        )

        # Generate dot groups for H0, H1, H2
        new_h_groups = self.get_homology_groups(diagram, new_axes, frame_max)
        self.play(
            Create(new_axes), Create(new_diag), Create(new_inf_line), run_time=1.0
        )
        for group in new_h_groups:
            if len(group) > 0:
                self.play(
                    LaggedStart(
                        *[FadeIn(dot, scale=1.5) for dot in group], lag_ratio=0.05
                    ),
                    run_time=0.5,
                )
            # self.play(Create(group))

        self.marked_next_slide()

        noise_arrow = Arrow(
            start=new_axes.c2p(0.5, 0.6),
            end=new_axes.c2p(1.0, 1.1),
            color=DEFAULT_COLOR,
            stroke_width=8,
        )
        self.play(GrowArrow(noise_arrow))

        signal_arrow = Arrow(
            start=new_axes.c2p(0.35, 1.16),
            end=new_axes.c2p(1.0, 1.16),
            color=ACCENT_COLOR,
            stroke_width=8,
        )
        self.play(GrowArrow(signal_arrow))

        self.marked_next_slide()

        noise_moves = []
        signal_moves = []
        for dot in new_h_groups[1]:
            if dot.get_center()[1] < new_axes.c2p(0.5, 0.5)[1] + 0.5:
                target = (
                    dot.get_center() + new_axes.c2p(0.66, 0.66) - new_axes.c2p(0, 0)
                )
                noise_moves.append(dot.animate.move_to(target))
            else:
                target = dot.get_center() + new_axes.c2p(0.66, 0.0) - new_axes.c2p(0, 0)
                signal_moves.append(dot.animate.move_to(target))

        self.play(
            FadeOut(noise_arrow),
            *noise_moves,
        )

        self.marked_next_slide()

        self.play(
            FadeOut(signal_arrow),
            *signal_moves,
        )

    def get_homology_groups(self, dgms, axes, frame_max):
        """Returns a list of VGroups, one for each homology dimension."""
        groups = []
        colors = [BLUE, ORANGE, GREEN]  # H0, H1, H2

        for dim, dgm in enumerate(dgms):
            vg = VGroup()
            if len(dgm) == 0:
                groups.append(vg)
                continue

            for point in dgm:
                birth, death = point
                # Handle Infinity Point (H0)
                if not np.isfinite(death):
                    dot = (
                        Triangle(color=colors[dim])
                        .scale(0.05)
                        .move_to(axes.c2p(birth, frame_max))
                    )
                else:
                    dot = Dot(
                        point=axes.c2p(birth, death),
                        color=colors[dim],
                        radius=0.04,
                        fill_opacity=0.7,
                        stroke_width=0.5,
                        stroke_color=WHITE,
                        stroke_opacity=0.8,
                    )
                vg.add(dot)
            groups.append(vg)

        # Ensure we always return 3 groups even if H2 is empty
        while len(groups) < 3:
            groups.append(VGroup())

        return groups

    def get_optimal_step(self, max_val):
        """Logic to keep ticks at 'round' intervals."""
        if max_val < 10:
            return 1.0
        if max_val < 20:
            return 2.0
        if max_val < 50:
            return 5.0
        if max_val < 100:
            return 10.0
        return 20.0


class DynamicPersistenceAnimation(TIMCSlide):
    def construct(self):
        # 1. Setup Data
        generator = CircleEmbedding(n_samples=300, seed=42)

        noise_hd = 0.16
        generator_kwds = {"radius": 1.0, "noise_2d": 0.0, "embedding": "linear"}

        dimensions = np.concatenate(
            [
                np.round(np.linspace(2, 64, 16)).astype(np.int32),
                np.round(
                    np.logspace(np.log2(64), int(round(np.log2(512))), 64, base=2)
                ).astype(np.int32),
            ]
        )
        # dimensions = np.round(np.linspace(2, np.cbrt(512), 256) ** 3).astype(np.int32)
        datasets = []
        diagrams = []
        for d in dimensions:
            print(f"Generating dataset for ambient dimension {d}...")
            X_2d, X_embedded, _, metadata = generator.generate_dataset(
                target_dim=int(round(d)),
                noise_hd=noise_hd,
                **generator_kwds,
            )
            datasets.append(X_embedded)
            dgms = ripser(X_embedded, maxdim=1)["dgms"]
            diagrams.append(dgms)

        # print("Creating interpolated sequence...")
        # dgms_interp, dimensions = create_interpolated_sequence(
        #     diagrams, n_interp=2, min_dim=2, max_dim=512
        # )
        # print(len(dgms_interp[0]))
        # print(f"Total frames: {len(dgms_interp)}")
        dgms_interp = diagrams
        axis_bounds = []
        current_max = 0
        for dgms in dgms_interp:
            finite_points = np.concatenate(
                [d[np.isfinite(d).all(axis=1)] for d in dgms if len(d) > 0]
            )
            frame_max = np.max(finite_points) * 1.1 if finite_points.size > 0 else 1.0
            if frame_max > current_max:
                current_max = frame_max
                axis_bounds.append(frame_max)
            else:
                axis_bounds.append(current_max)

        # Track the groups across frames
        self.current_h_groups = [VGroup() for _ in range(3)]  # H0, H1, H2
        self.current_axes = None
        self.current_diag = None
        self.current_inf_line = None
        self.current_dim_label = None

        for i, dgms in enumerate(dgms_interp):
            # Calculate max scale for this frame (excluding infinity)
            finite_points = np.concatenate(
                [d[np.isfinite(d).all(axis=1)] for d in dgms if len(d) > 0]
            )
            frame_max = np.max(finite_points) * 1.1 if finite_points.size > 0 else 1.0

            # --- CREATE NEW FRAME COMPONENTS ---
            # Fixed tick steps (e.g., steps of 0.5 or 1.0 depending on scale)
            tick_step = self.get_optimal_step(frame_max)

            new_axes = Axes(
                x_range=[0, axis_bounds[i], tick_step],
                y_range=[0, axis_bounds[i], tick_step],
                x_length=6,
                y_length=6,
                axis_config={
                    "include_tip": False,
                    "include_ticks": True,
                    "numbers_to_exclude": [0],
                },
            ).add_coordinates()

            new_diag = Line(
                start=new_axes.c2p(0, 0),
                end=new_axes.c2p(axis_bounds[i], axis_bounds[i]),
                color=GRAY,
                stroke_width=2,
            )

            # Infinity Line at the very top of the Y-axis
            inf_y = new_axes.c2p(0, axis_bounds[i])[1]
            new_inf_line = DashedLine(
                start=[new_axes.c2p(0, 0)[0], inf_y, 0],
                end=[new_axes.c2p(axis_bounds[i], 0)[0], inf_y, 0],
                color=GRAY_C,
            )

            # Generate dot groups for H0, H1, H2
            new_h_groups = self.get_homology_groups(dgms, new_axes, axis_bounds[i])

            dim_label = Text("Ambient Dimension", font_size=24).next_to(
                new_axes, RIGHT, buff=0.5
            )
            dim_label_value = Text(
                f"{dimensions[i]}", font_size=48, color=ACCENT_COLOR
            ).next_to(dim_label, DOWN, buff=0.25)

            # --- ANIMATION LOGIC ---
            if i == 0:
                self.add(new_axes, new_diag, new_inf_line, dim_label, dim_label_value)
                for group in new_h_groups:
                    self.add(group)
            else:
                self.play(
                    ReplacementTransform(self.current_axes, new_axes),
                    ReplacementTransform(self.current_diag, new_diag),
                    ReplacementTransform(self.current_inf_line, new_inf_line),
                    ReplacementTransform(self.current_dim_label_value, dim_label_value),
                    # Transform each homology group to its corresponding new version
                    *[
                        ReplacementTransform(self.current_h_groups[j], new_h_groups[j])
                        for j in range(3)
                    ],
                    run_time=0.5,
                )

            # Update references
            self.current_axes = new_axes
            self.current_diag = new_diag
            self.current_inf_line = new_inf_line
            self.current_h_groups = new_h_groups
            self.current_dim_label_value = dim_label_value
            self.wait(0.1)

    def get_homology_groups(self, dgms, axes, frame_max):
        """Returns a list of VGroups, one for each homology dimension."""
        groups = []
        colors = [BLUE, ORANGE, GREEN]  # H0, H1, H2

        for dim, dgm in enumerate(dgms):
            vg = VGroup()
            if len(dgm) == 0:
                groups.append(vg)
                continue

            for point in dgm:
                birth, death = point
                # Handle Infinity Point (H0)
                if not np.isfinite(death):
                    dot = (
                        Triangle(color=colors[dim])
                        .scale(0.05)
                        .move_to(axes.c2p(birth, frame_max))
                    )
                else:
                    dot = Dot(
                        point=axes.c2p(birth, death),
                        color=colors[dim],
                        radius=0.04,
                        fill_opacity=0.7,
                        stroke_width=0.5,
                        stroke_color=WHITE,
                        stroke_opacity=0.8,
                    )
                vg.add(dot)
            groups.append(vg)

        # Ensure we always return 3 groups even if H2 is empty
        while len(groups) < 3:
            groups.append(VGroup())

        return groups

    def get_optimal_step(self, max_val):
        """Logic to keep ticks at 'round' intervals."""
        if max_val < 10:
            return 1.0
        if max_val < 20:
            return 2.0
        if max_val < 50:
            return 5.0
        if max_val < 100:
            return 10.0
        return 20.0


class ExampleCircleEmbeddingConstruction(ThreeDTIMCSlide):

    def construct(self):
        generator = CircleEmbedding(n_samples=150, seed=42)
        X_2d, X_embedded, X_embedded_unnoised, metadata = generator.generate_dataset(
            target_dim=3, noise_hd=0.16, radius=1.0, noise_2d=0.0, embedding="linear"
        )
        X_2d_3d = np.hstack([X_2d, np.zeros((X_2d.shape[0], 1))])
        cov = X_2d_3d.T @ X_embedded_unnoised
        us, s, vt = np.linalg.svd(cov)
        # if np.linalg.det(us @ vt) < 0:
        #     us[:, -1] *= -1
        R = us @ vt

        X_embedded_unnoised = (R @ X_2d_3d.T).T
        X_embedded = (R @ X_2d_3d.T).T + np.random.normal(
            scale=0.16, size=X_2d_3d.shape
        )

        axes = ThreeDAxes(x_range=[-1.5, 1.5], y_range=[-1.5, 1.5], z_range=[-1.5, 1.5])
        twod_points = self.create_points(X_2d_3d, axes)
        # threed_points_unnoised = self.create_points(X_embedded_unnoised, axes)
        threed_points = self.create_points(X_embedded, axes)

        self.play(Create(axes))
        # self.play(LaggedStart(*[FadeIn(p) for p in twod_points], lag_ratio=0.01))
        self.play(Create(twod_points))

        self.wait()

        self.move_camera(phi=75 * DEGREES, theta=30 * DEGREES)

        self.wait()
        axis, angle = rotation_matrix_to_axis_angle(R)

        theta = ValueTracker(0)

        def update_dots(mob):
            t = theta.get_value()
            rot = rotation_matrix(axis=axis, angle=t)  # manim helper
            for i, d in enumerate(mob):
                d.move_to(axes.c2p(*(rot @ X_2d_3d[i])))

        twod_points.add_updater(update_dots)

        self.play(theta.animate.set_value(angle), run_time=2)
        twod_points.remove_updater(update_dots)

        self.wait()

        print(np.allclose((R @ X_2d_3d.T).T, X_embedded_unnoised))
        print((R @ X_2d_3d.T).T)
        print(X_embedded_unnoised)
        plan = ot.solve_sample((R @ X_2d_3d.T).T, X_embedded).plan

        geodesic_t = ValueTracker(0)

        def update_dots_transport(mob):
            t = geodesic_t.get_value()
            for i, d in enumerate(mob):
                target_node = plan[
                    i
                ].argmax()  # Get the index of the target point in X_embedded
                interp = (1 - t) * (R @ X_2d_3d.T)[:, i] + t * X_embedded[target_node]
                d.move_to(axes.c2p(*interp))

        twod_points.add_updater(update_dots_transport)

        self.play(geodesic_t.animate.set_value(1), run_time=2)
        twod_points.remove_updater(update_dots_transport)

        self.begin_ambient_camera_rotation(rate=0.75)
        self.wait(12)  # update to be longer as required
        self.stop_ambient_camera_rotation()

    def create_points(self, data, axes):
        points = VGroup()
        for x in data:
            if len(x) == 2:
                x = np.append(x, 0)  # Add z=0 for 2D points
            dot = Dot3D(point=axes.c2p(*x), color=ACCENT_COLOR, radius=0.05)
            points.add(dot)
        return points


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


class PMFadeOut(Animation):
    """FadeOut for PMobject.

    Manim's built-in FadeOut uses Mobject.fade(), which is a no-op for
    PMobject (it never touches rgbas[:,3]).  This subclass interpolates
    the alpha channel directly and sets remover=True so the scene removes
    the object automatically when the animation ends.
    """

    def __init__(self, mob: PMobject, **kwargs):
        self._orig_alphas = mob.rgbas[:, 3].copy()
        super().__init__(mob, remover=True, **kwargs)

    def interpolate_mobject(self, alpha: float) -> None:
        self.mobject.rgbas[:, 3] = self._orig_alphas * (1.0 - alpha)


# ---------------------------------------------------------------------------
# Cairo base class
# ---------------------------------------------------------------------------


class _LissajousBase(ThreeDTIMCSlide):
    """
    Cairo-renderer base for Lissajous-knot embedding scenes.

    Subclasses set class-level attributes to vary the dataset and reduction,
    and may override ``get_embedding`` for custom dimensionality reduction.
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

    def get_embedding(self, X_embedded: np.ndarray) -> np.ndarray:
        """Return an (N, 3) array to plot. Override to change reduction."""
        return sklearn.decomposition.PCA(n_components=3).fit_transform(X_embedded)

    def construct(self):
        generator = CurvyLoopEmbedding(n_samples=self._n_samples, seed=self._seed)
        _, X_embedded, _ = generator.generate_dataset(**self._generator_kwargs)
        embedding_3d = self.get_embedding(X_embedded)

        axes = ThreeDAxes(x_range=[-2.5, 2.5], y_range=[-2.5, 2.5], z_range=[-2.5, 2.5])
        points = self.create_points(embedding_3d, axes)

        self.camera.get_thickening_nudges = _types.MethodType(
            lambda cam, t: _make_circular_nudges(t), self.camera
        )

        self.move_camera(phi=75 * DEGREES, theta=30 * DEGREES)
        self.play(Create(axes))
        self.wait(1)
        self.play(FadeIn(points))
        self.begin_ambient_camera_rotation(rate=0.75)
        self.wait(self._rotation_wait)
        self.stop_ambient_camera_rotation()

        self.marked_next_slide()
        self.begin_ambient_camera_rotation(rate=4 * PI / self._rotation_wait)
        self.wait(self._rotation_wait / 2)
        self.stop_ambient_camera_rotation()
        self.marked_next_slide()

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


class _LissajousBaseOpenGL(ThreeDTIMCSlide):
    """
    OpenGL-renderer base for Lissajous-knot embedding scenes.
    Render with:  manim-slides render slides.py <ClassName> --renderer=opengl
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

    def get_embedding(self, X_embedded: np.ndarray) -> np.ndarray:
        """Return an (N, 3) array to plot. Override to change reduction."""
        return sklearn.decomposition.PCA(n_components=3).fit_transform(X_embedded)

    def construct(self):
        generator = CurvyLoopEmbedding(n_samples=self._n_samples, seed=self._seed)
        _, X_embedded, _ = generator.generate_dataset(**self._generator_kwargs)
        embedding_3d = self.get_embedding(X_embedded)

        axes = ThreeDAxes(x_range=[-2.5, 2.5], y_range=[-2.5, 2.5], z_range=[-2.5, 2.5])
        points = self.create_points(embedding_3d, axes)

        self.move_camera(phi=75 * DEGREES, theta=30 * DEGREES)
        self.play(Create(axes))
        self.wait(1)
        self.play(FadeIn(points))
        self.begin_ambient_camera_rotation(rate=0.75)
        self.wait(self._rotation_wait)
        self.stop_ambient_camera_rotation()

    def create_points(self, data: np.ndarray, axes):
        coords = np.array(
            [axes.c2p(*(x if len(x) == 3 else np.append(x, 0))) for x in data],
            dtype=np.float32,
        )
        n = len(data)
        group_indices = np.array_split(np.arange(n), self._n_color_groups)
        result = Group()
        for i, indices in enumerate(group_indices):
            t_mid = i / max(self._n_color_groups - 1, 1)
            color = colorcet.CET_C9[int(t_mid * (len(colorcet.CET_C9) - 1))]
            dc = DotCloud(color=color, stroke_width=self._dot_stroke_width)
            dc.set_points(coords[indices])
            result.add(dc)
        return result


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


class ShadedNNDistribution(TIMCSlide):

    def construct(self):
        # --- Parameters ---
        num_points = 5000
        dimensions = np.round(np.linspace(3, 64, 63)).astype(np.int32)
        x_range = [0, 3.0, 0.5]

        axes = Axes(
            x_range=x_range, y_range=[0, 20, 2], axis_config={"include_tip": False}
        ).add_coordinates()

        self.add(axes)

        # --- Helper to create Curve + Area ---
        def get_distribution_mobjects(d, color=YELLOW):
            # 1. Sample and calculate KDE
            points = np.random.uniform(0, 1, size=(num_points, d))
            nn_index = sklearn.neighbors.NearestNeighbors(n_neighbors=2).fit(points)
            dist, _ = nn_index.kneighbors(points)
            nn_distances = dist[:, 1]

            kde = gaussian_kde(nn_distances)

            # 2. Use a lambda to ensure the KDE returns a float, not an array
            curve = axes.plot(
                lambda x: float(kde.evaluate(x)[0]),
                x_range=[x_range[0], x_range[1], 0.01],
                color=color,
                stroke_width=2,
            )

            # 3. Create the shaded area
            area = axes.get_area(
                curve, x_range=(x_range[0], x_range[1]), color=color, opacity=0.3
            )

            vals = np.squeeze(kde.evaluate(np.linspace(x_range[0], x_range[1], 1024).T))
            y_loc = np.max(vals)
            x_loc = (
                np.argmax(vals) * (1.0 / 1024.0 * (x_range[1] - x_range[0]))
                + x_range[0]
            )

            return VGroup(area, curve), np.asarray((x_loc, y_loc))

        # --- Initial State ---
        dim_tracker = Integer(dimensions[0]).to_corner(UR)
        label = Text("dimension = ", font_size=24).next_to(dim_tracker, LEFT)

        # current_mobjects is a VGroup(area, curve)
        current_mobjects, current_loc = get_distribution_mobjects(dimensions[0])

        self.add(label, dim_tracker, current_mobjects)

        # --- Animation Loop ---
        for i, d in enumerate(dimensions):
            new_mobjects, new_loc = get_distribution_mobjects(
                d,
                color=colorcet.bmy[
                    int(i * (len(colorcet.bmy) - 1) / (len(dimensions) - 1))
                ],
            )

            if d in [4, 8, 16, 32, 64]:
                saved_density = new_mobjects.copy()

                density_label = Text(
                    f"dim={d}",
                    font_size=18,
                    color=colorcet.bmy[
                        int(i * (len(colorcet.bmy) - 1) / (len(dimensions) - 1))
                    ],
                ).move_to(axes.c2p(*new_loc) + np.array([0.0, 0.5, 0]))
                print(f"Adding label for dimension {d} at location {new_loc}")

                self.play(
                    current_mobjects.animate.become(new_mobjects),
                    dim_tracker.animate.set_value(d),
                    FadeIn(density_label),
                    run_time=0.25,
                    rate_func=smooth,
                )
                self.add(saved_density, density_label)
            else:
                self.play(
                    # .become() handles morphing the polygon and line points
                    current_mobjects.animate.become(new_mobjects),
                    dim_tracker.animate.set_value(d),
                    run_time=0.25,
                    rate_func=smooth,
                )

        self.wait()


def _make_vr_demo_cloud(
    n_clusters: int = 8,
    circle_radius: float = 1.0,
    rect_min_side: float = 0.35,
    rect_max_side: float = 0.55,
    n_connectors: int = 10,
    seed: int = 13,
) -> np.ndarray:
    """
    Build a point cloud that looks like a noisily sampled circle but has
    controlled topological properties:

    * One dominant H₁ loop (the circle) with long persistence.
    * Several short-persistence H₁ loops, one per rectangular cluster
      (points near the diagonal of the persistence diagram).
    * H₀ components that merge as the filtration grows.

    Construction
    ------------
    ``n_clusters`` small 2×2 rectangular grids of points are placed around a
    circle of ``circle_radius``.  Each rectangle has random side lengths drawn
    from [rect_min_side, rect_max_side] and a random rotation.  An additional
    ``n_connectors`` points are scattered near the circle arc between random
    adjacent cluster pairs to ensure clean connectivity between clusters.

    Topological rationale for the rectangles
    -----------------------------------------
    For a rectangle with sides w ≤ h the VR H₁ loop is born at r = h and dies
    at r = √(w²+h²).  With small sides (≪ inter-cluster gap) the persistence
    √(w²+h²) − h  is short, so these features sit close to the diagonal.
    """
    rng = np.random.RandomState(seed)
    points = []

    # Cluster centres: equally-spaced angles with small random jitter
    base_angles = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False)
    cluster_angles = base_angles + rng.uniform(-0.15, 0.15, n_clusters)

    for angle in cluster_angles:
        cx = circle_radius * np.cos(angle)
        cy = circle_radius * np.sin(angle)

        w = rng.uniform(rect_min_side, rect_max_side)
        h = rng.uniform(rect_min_side, rect_max_side)

        rot = rng.uniform(0, 2 * np.pi)
        cos_r, sin_r = np.cos(rot), np.sin(rot)
        R_mat = np.array([[cos_r, -sin_r], [sin_r, cos_r]])

        for gx in (-w / 2, w / 2):
            for gy in (-h / 2, h / 2):
                local = R_mat @ np.array([gx, gy]) + rng.normal(scale=0.05, size=2)
                points.append(np.array([cx + local[0], cy + local[1]]))

    # Connector points placed in the arc between randomly chosen adjacent clusters
    n = len(cluster_angles)
    sorted_angles = np.sort(cluster_angles)
    for _ in range(n_connectors):
        idx = rng.randint(0, n)
        a1 = sorted_angles[idx]
        a2 = sorted_angles[(idx + 1) % n]
        if a2 < a1:  # handle wrap-around
            a2 += 2 * np.pi
        a = a1 + rng.uniform(0.25, 0.75) * (a2 - a1)
        r = circle_radius + rng.uniform(-0.07, 0.07)
        points.append(np.array([r * np.cos(a), r * np.sin(a)]))

    return np.array(points)


class VietorisRipsExplanation(TIMCSlide):
    def construct(self):
        from scipy.spatial.distance import pdist, squareform

        # Build the point cloud and compute its distance matrix and persistence diagrams
        pts = _make_vr_demo_cloud()
        n_pts = len(pts)
        dist_mat = squareform(pdist(pts))
        max_r = np.max(dist_mat) * 1.1

        # ── 2.  VR Complex panel  (left, centred at x = -3.7) ─────────────────
        PANEL_CX = 0.0
        PANEL_CY = 0.0
        pts_c = pts - pts.mean(axis=0)
        vis_scale = 3.5 / (np.max(np.abs(pts_c)) + 1e-9)
        pts_vis = pts_c * vis_scale  # scene 2-D coords

        pts_manim = [np.array([PANEL_CX + p[0], PANEL_CY + p[1], 0.0]) for p in pts_vis]

        pt_dots = VGroup(
            *[Dot(point=p, color=DEFAULT_COLOR, radius=0.07) for p in pts_manim]
        )

        self.play(
            LaggedStart(*[FadeIn(dot, scale=2.0) for dot in pt_dots], lag_ratio=0.05)
        )

        self.marked_next_slide()

        circles = []
        for i in range(n_pts):
            circ = Circle(
                radius=0.001,  # start near-zero so set_radius has a valid base
                arc_center=pts_manim[i],
                color=GRAY,
                stroke_width=1.5,
                fill_opacity=0,
            ).set_stroke(
                opacity=0
            )  # invisible until animation begins
            circles.append(circ)

        edge_data = []  # [(r_threshold, Line)]
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                r_ij = dist_mat[i, j]
                if r_ij <= max_r:
                    line = Line(
                        pts_manim[i],
                        pts_manim[j],
                        color=GRAY,
                        stroke_width=1.5,
                    ).set_stroke(opacity=0)
                    edge_data.append((r_ij, line))
        edge_data.sort(key=lambda x: x[0])

        # Triangles — sorted by max-edge distance
        tri_data = []  # [(r_threshold, Polygon)]
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                for k in range(j + 1, n_pts):
                    r_tri = max(dist_mat[i, j], dist_mat[j, k], dist_mat[i, k])
                    if r_tri <= max_r:
                        poly = Polygon(
                            pts_manim[i],
                            pts_manim[j],
                            pts_manim[k],
                            fill_color=ACCENT_COLOR,
                            fill_opacity=0,
                            stroke_width=0,
                        )
                        tri_data.append((r_tri, poly))
        tri_data.sort(key=lambda x: x[0])
        all_edges = VGroup(*[e for _, e in edge_data])
        all_tris = VGroup(*[t for _, t in tri_data])
        all_circles = VGroup(*circles)
        self.add(all_tris, all_edges, all_circles)

        r_tracker = ValueTracker(0.0)
        self.add(r_tracker)

        edge_ptr = [0]
        tri_ptr = [0]

        def _reveal_edges(_mob):
            r = r_tracker.get_value()
            while edge_ptr[0] < len(edge_data) and edge_data[edge_ptr[0]][0] <= r:
                edge_data[edge_ptr[0]][1].set_stroke(opacity=0.5)
                edge_ptr[0] += 1

        def _reveal_tris(_mob):
            r = r_tracker.get_value()
            while tri_ptr[0] < len(tri_data) and tri_data[tri_ptr[0]][0] <= r:
                tri_data[tri_ptr[0]][1].set_fill(opacity=0.1)
                tri_ptr[0] += 1

        def _update_circles(_mob):
            r = r_tracker.get_value()
            target_r = max(0.001, r * vis_scale / 2)
            opacity = 0.5 if r > 1e-9 else 0.0
            for i, circ in enumerate(circles):
                # set_radius scales about the bounding-box centre (correct for
                # a centred circle); move_to then re-pins the centre to the data
                # point, eliminating any accumulated floating-point drift.
                # circ.set_radius(target_r)
                # circ.move_to(pts_manim[i])
                # circ.set_stroke(opacity=opacity)
                circ.become(
                    Circle(
                        radius=max(0.001, target_r),
                        arc_center=pts_manim[i],
                        color=GRAY,
                        stroke_width=1.5,
                        fill_opacity=0,
                    ).set_stroke(opacity=opacity)
                )

        all_edges.add_updater(_reveal_edges)
        all_tris.add_updater(_reveal_tris)
        all_circles.add_updater(_update_circles)

        self.play(
            r_tracker.animate.set_value(max_r / 6),
            run_time=4.0,
            rate_func=rate_functions.ease_out_quart,
        )

        self.marked_next_slide()

        self.play(
            r_tracker.animate.set_value(max_r / 3),
            run_time=6.0,
            rate_func=rate_functions.ease_in_quad,
        )

        self.marked_next_slide()

        self.play(FadeOut(all_circles), run_time=1.0)

        self.marked_next_slide()

        self.clear_slide(run_time=1.0)


class SimplicialHomologyExplanation(TIMCSlide):
    """
    Animated H₁ recap (ℤ₂ coefficients) on a hand-crafted planar complex K.

    Complex layout — 8 vertices in a 2×4 grid, middle column triangulated:

        v0 ─── v1 ─── v2 ─── v3
        │       │  ╲   │       │
        v4 ─── v5 ─── v6 ─── v7

    Edges: top/bottom rows, four verticals, plus diagonal v2–v5 (triangulation).
    2-cells: τ₁ = (v1,v2,v5),  τ₂ = (v2,v5,v6)  — fill the middle rectangle.

    Three 1-cycles highlighted in sequence:
        σ₁  left  loop  v0–v1–v5–v4   —  non-trivial  [σ₁] ≠ 0
        σ₂  right loop  v2–v3–v7–v6   —  non-trivial  [σ₂] ≠ 0, independent of σ₁
        σ₃  mid   loop  v1–v2–v6–v5   —  null-homologous: σ₃ = ∂₂(τ₁ + τ₂)

    Conclusion:  H₁(K; ℤ₂) ≅ ℤ₂ ⊕ ℤ₂
    """

    def construct(self):
        # ── 1.  Vertex layout ─────────────────────────────────────────────────
        # Three rectangular regions side-by-side; complex centred on screen.
        DX, DY = 1.8, 1.9  # horizontal / vertical spacing
        CX, CY = 0.0, 0.6  # centroid of the complex
        v = [
            np.array([CX - 1.5 * DX, CY + DY / 2, 0.0]),  # 0  top-left
            np.array([CX - 0.5 * DX, CY + DY / 2, 0.0]),  # 1  top-mid-L
            np.array([CX + 0.5 * DX, CY + DY / 2, 0.0]),  # 2  top-mid-R
            np.array([CX + 1.5 * DX, CY + DY / 2, 0.0]),  # 3  top-right
            np.array([CX - 1.5 * DX, CY - DY / 2, 0.0]),  # 4  bot-left
            np.array([CX - 0.5 * DX, CY - DY / 2, 0.0]),  # 5  bot-mid-L
            np.array([CX + 0.5 * DX, CY - DY / 2, 0.0]),  # 6  bot-mid-R
            np.array([CX + 1.5 * DX, CY - DY / 2, 0.0]),  # 7  bot-right
        ]
        # x ∈ [−2.7, 2.7],  y ∈ [−0.35, 1.55]  — fits comfortably on screen.

        # ── 2.  Base simplicial complex mobjects ──────────────────────────────
        EDGE_PAIRS = [
            (0, 1),
            (1, 2),
            (2, 3),  # top row
            (4, 5),
            (5, 6),
            (6, 7),  # bottom row
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),  # verticals
            (2, 5),  # triangulation diagonal
        ]
        base_edges = {
            (i, j): Line(v[i], v[j], color=DEFAULT_COLOR, stroke_width=2.0)
            for (i, j) in EDGE_PAIRS
        }
        base_dots = {i: Dot(v[i], color=DEFAULT_COLOR, radius=0.07) for i in range(8)}
        # Middle region: τ₁ = (1,2,5),  τ₂ = (2,5,6)
        tri1 = Polygon(
            v[1],
            v[2],
            v[5],
            fill_color=ACCENT_COLOR,
            fill_opacity=0.15,
            stroke_width=0,
        )
        tri2 = Polygon(
            v[2],
            v[5],
            v[6],
            fill_color=ACCENT_COLOR,
            fill_opacity=0.15,
            stroke_width=0,
        )
        all_tris = VGroup(tri1, tri2)
        all_edges_grp = VGroup(*base_edges.values())
        all_dots_grp = VGroup(*base_dots.values())

        # ── 3.  Slide 1: reveal the complex ───────────────────────────────────
        self.play(
            FadeIn(all_tris),
            LaggedStart(
                *[FadeIn(e) for e in all_edges_grp.submobjects], lag_ratio=0.05
            ),
            LaggedStart(
                *[GrowFromCenter(d) for d in all_dots_grp.submobjects], lag_ratio=0.05
            ),
            run_time=2.2,
        )
        self.marked_next_slide()

        # ── 4.  Title ─────────────────────────────────────────────────────────
        title = MathTex(
            r"\text{Simplicial Homology:}\\ H_1(K;\,\mathbb{Z}_2)"
            r" \;=\; \ker\partial_1 \;/\; \mathrm{im}\,\partial_2",
            font_size=42,
        ).to_edge(UP, buff=0.35)
        self.play(Write(title), run_time=2.0)
        self.marked_next_slide()

        # ── 5.  Helpers ───────────────────────────────────────────────────────
        def _cycle_overlay(pairs, color, width=5.5):
            """Thick coloured lines drawn over the given edge pairs."""
            return VGroup(
                *[
                    Line(v[i], v[j], color=color, stroke_width=width, z_index=2)
                    for (i, j) in pairs
                ]
            )

        # Interior centres of the three rectangular loops
        ctr_left = np.array([CX - DX, CY, 0.0])  # centre of left loop
        ctr_right = np.array([CX + DX, CY, 0.0])  # centre of right loop

        # Annotation rows build upward from the bottom of the frame
        ANNO_Y = -3.1  # y of the first annotation line
        ANNO_DY = 0.72  # step between annotation rows

        # ── 6.  Slide 2: σ₁ — left loop, non-trivial ─────────────────────────
        C1 = COLOR_CYCLE[0]
        hi1 = _cycle_overlay([(0, 1), (1, 5), (4, 5), (0, 4)], C1)
        inner1 = MathTex(r"\sigma_1", font_size=26, color=C1).move_to(ctr_left)
        anno1 = MathTex(
            r"\sigma_1 \;\in\; \ker\partial_1,\quad [\sigma_1] \neq 0",
            font_size=28,
            color=C1,
        ).move_to([0.0, ANNO_Y, 0.0])

        self.play(FadeIn(hi1), FadeIn(inner1), run_time=0.7)
        self.play(Write(anno1), run_time=1.1)
        self.marked_next_slide()

        # ── 7.  Slide 3: σ₂ — right loop, non-trivial ────────────────────────
        C2 = COLOR_CYCLE[1]
        hi2 = _cycle_overlay([(2, 3), (3, 7), (6, 7), (2, 6)], C2)
        inner2 = MathTex(r"\sigma_2", font_size=26, color=C2).move_to(ctr_right)
        anno2 = MathTex(
            r"\sigma_2 \;\in\; \ker\partial_1,\quad [\sigma_2] \neq 0,"
            r"\quad [\sigma_1] \neq [\sigma_2]",
            font_size=28,
            color=C2,
        ).move_to([0.0, ANNO_Y + ANNO_DY, 0.0])

        self.play(FadeIn(hi2), FadeIn(inner2), run_time=0.7)
        self.play(Write(anno2), run_time=1.1)
        self.marked_next_slide()

        # ── 8.  Slide 4: σ₃ — middle loop, null-homologous ───────────────────
        C3 = COLOR_CYCLE[2]
        hi3 = _cycle_overlay([(1, 2), (2, 6), (5, 6), (1, 5)], C3)

        # Brighten the two 2-cells to show σ₃ = ∂₂(τ₁ + τ₂)
        tri1_hi = Polygon(
            v[1], v[2], v[5], fill_color=C3, fill_opacity=0.42, stroke_width=0
        )
        tri2_hi = Polygon(
            v[2], v[5], v[6], fill_color=C3, fill_opacity=0.42, stroke_width=0
        )
        tau1_lbl = MathTex(r"\tau_1", font_size=22, color=C3).move_to(
            (v[1] + v[2] + v[5]) / 3
        )
        tau2_lbl = MathTex(r"\tau_2", font_size=22, color=C3).move_to(
            (v[2] + v[5] + v[6]) / 3
        )
        anno3 = MathTex(
            r"\sigma_3 \;=\; \partial_2(\tau_1 + \tau_2)"
            r"\;\Rightarrow\; [\sigma_3] = 0",
            font_size=28,
            color=C3,
        ).move_to([0.0, ANNO_Y + 2 * ANNO_DY, 0.0])

        self.play(FadeIn(hi3), run_time=0.7)
        self.play(
            Transform(tri1, tri1_hi),
            Transform(tri2, tri2_hi),
            FadeIn(tau1_lbl),
            FadeIn(tau2_lbl),
            run_time=0.8,
        )
        self.play(Write(anno3), run_time=1.1)
        self.marked_next_slide()

        # ── 9.  Slide 5: conclusion ────────────────────────────────────────────
        concl = MathTex(
            r"H_1(K;\,\mathbb{Z}_2) \;\cong\; \mathbb{Z}_2 \oplus \mathbb{Z}_2",
            font_size=44,
        )
        box = SurroundingRectangle(
            concl, color=HIGHLIGHT_COLOR, buff=0.18, corner_radius=0.1
        )
        VGroup(concl, box).move_to([0.0, ANNO_Y + 3 * ANNO_DY, 0.0])

        self.play(FadeOut(anno1), FadeOut(anno2), FadeOut(anno3), run_time=0.4)
        self.play(Write(concl), Create(box), run_time=1.5)
        self.marked_next_slide()

        self.clear_slide(run_time=1.0)


class PersistenceExplanation(TIMCSlide):
    """
    Animated introduction to persistence diagrams via the Vietoris-Rips filtration.

    Left panel  — VR simplicial complex growing on a noisy circle in 2-D.
    Right panel — Persistence diagram with a horizontal red tracker line (y = ε)
                  sweeping upward at the same filtration value ε.

    Feature dots are born on the birth=death diagonal and slide upward along the
    tracker; they freeze at their final (birth, death) position when the feature dies.
    """

    def construct(self):
        from scipy.spatial.distance import pdist, squareform

        # ── 1.  Build hand-crafted noisy-circle point cloud ───────────────────
        pts = _make_vr_demo_cloud()
        n_pts = len(pts)

        dist_mat = squareform(pdist(pts))

        dgms = ripser(pts, maxdim=1)["dgms"]
        dgm_h0 = dgms[0]  # all births = 0; one inf death
        dgm_h1 = dgms[1]

        all_finite = np.concatenate(
            [d[np.isfinite(d).all(axis=1)] for d in dgms if len(d) > 0]
        )
        max_r = float(np.max(all_finite)) * 1.1
        tick_step = _pd_optimal_step(max_r)

        # ── 2.  VR Complex panel  (left, centred at x = -3.7) ─────────────────
        PANEL_CX = -3.7
        PANEL_CY = 0.0
        pts_c = pts - pts.mean(axis=0)
        vis_scale = 2.5 / (np.max(np.abs(pts_c)) + 1e-9)
        pts_vis = pts_c * vis_scale  # scene 2-D coords

        pts_manim = [np.array([PANEL_CX + p[0], PANEL_CY + p[1], 0.0]) for p in pts_vis]

        pt_dots = VGroup(
            *[Dot(point=p, color=DEFAULT_COLOR, radius=0.07) for p in pts_manim]
        )
        vr_title = Text("Vietoris-Rips Complex", font_size=28).next_to(
            pt_dots, UP, buff=0.3
        )

        # ── 3.  Persistence Diagram panel  (right, axes shifted RIGHT*3) ───────
        pd_ax = Axes(
            x_range=[0, max_r, tick_step],
            y_range=[0, max_r, tick_step],
            x_length=5.0,
            y_length=5.0,
            axis_config={
                "include_tip": False,
                "include_ticks": True,
                "numbers_to_exclude": [0],
            },
        ).add_coordinates()
        pd_ax.shift(RIGHT * 3)

        diag_line = Line(
            pd_ax.c2p(0, 0),
            pd_ax.c2p(max_r, max_r),
            color=GRAY,
            stroke_width=2,
        )

        inf_y_px = pd_ax.c2p(0, max_r)[1]
        inf_line = DashedLine(
            start=[pd_ax.c2p(0, 0)[0], inf_y_px, 0],
            end=[pd_ax.c2p(max_r, 0)[0], inf_y_px, 0],
            color=GRAY_C,
        )
        inf_label = MathTex(r"\infty", font_size=24, color=GRAY_C).next_to(
            np.array([pd_ax.c2p(0, 0)[0] - 0.05, inf_y_px, 0]), LEFT, buff=0.08
        )

        pd_title = Text("Persistence Diagram", font_size=28).next_to(
            pd_ax, UP, buff=0.15
        )
        pd_xlabel = Text("birth", font_size=18, color=GRAY).next_to(
            pd_ax.get_x_axis(), DOWN, buff=0.1
        )
        pd_ylabel = (
            Text("death", font_size=18, color=GRAY)
            .rotate(PI / 2)
            .next_to(pd_ax.get_y_axis(), LEFT, buff=0.1)
        )

        # ── 4.  Tracker line + ε label (no LaTeX per frame) ───────────────────
        r_tracker = ValueTracker(0.0)

        tracker_line = always_redraw(
            lambda: Line(
                start=pd_ax.c2p(0, r_tracker.get_value()),
                end=pd_ax.c2p(max_r, r_tracker.get_value()),
                color=RED,
                stroke_width=2.5,
            )
        )

        eps_tex = MathTex(r"\varepsilon = ", font_size=36, color=RED)
        eps_num = DecimalNumber(0.0, num_decimal_places=3, font_size=36, color=RED)
        eps_group = VGroup(eps_tex, eps_num).arrange(RIGHT, buff=0.05)
        eps_group.to_edge(DOWN, buff=0.35)
        eps_num.add_updater(lambda m: m.set_value(r_tracker.get_value()))

        # Infinite H₀ marker: triangle at (0, ∞) — present from the start
        inf_h0_marker = (
            Triangle(color=BLUE, fill_color=BLUE, fill_opacity=0.0)
            .scale(0.065)
            .move_to(pd_ax.c2p(0, max_r))
        )

        # ── 5.  First slide: static skeleton ──────────────────────────────────
        self.add(
            pd_ax,
            diag_line,
            inf_line,
            inf_label,
            pd_title,
            pd_xlabel,
            pd_ylabel,
            pt_dots,
            vr_title,
            tracker_line,
            eps_group,
            inf_h0_marker,
        )
        self.next_slide()  # [Slide 1] empty panels — introduce the setup

        # ── 6.  Pre-build VR edges (initially invisible, sorted by distance) ──────
        edge_data = []  # [(r_threshold, Line)]
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                r_ij = dist_mat[i, j]
                if r_ij <= max_r:
                    line = Line(
                        pts_manim[i],
                        pts_manim[j],
                        color=GRAY,
                        stroke_width=1.5,
                    ).set_stroke(opacity=0)
                    edge_data.append((r_ij, line))
        edge_data.sort(key=lambda x: x[0])

        # Triangles — sorted by max-edge distance
        tri_data = []  # [(r_threshold, Polygon)]
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                for k in range(j + 1, n_pts):
                    r_tri = max(dist_mat[i, j], dist_mat[j, k], dist_mat[i, k])
                    if r_tri <= max_r:
                        poly = Polygon(
                            pts_manim[i],
                            pts_manim[j],
                            pts_manim[k],
                            fill_color=ACCENT_COLOR,
                            fill_opacity=0,
                            stroke_width=0,
                        )
                        tri_data.append((r_tri, poly))
        tri_data.sort(key=lambda x: x[0])

        # ── 7.  Pre-build H₀ PD dots (birth = 0 for all) ────────────────────────
        h0_fin_mask = np.isfinite(dgm_h0[:, 1])
        dgm_h0_fin = dgm_h0[h0_fin_mask]
        h0_deaths_sorted = sorted(float(pt[1]) for pt in dgm_h0_fin)
        h0_dots = []  # [(death_r, Dot)]
        for death_r in h0_deaths_sorted:
            dot = Dot(
                point=pd_ax.c2p(0, 0),
                color=BLUE,
                radius=0.055,
                fill_opacity=0,
                stroke_width=1.0,
                stroke_color=WHITE,
            )
            h0_dots.append((death_r, dot))

        # ── 8.  Pre-build H₁ PD dots ─────────────────────────────────────────────
        dgm_h1_fin = (
            dgm_h1[np.isfinite(dgm_h1[:, 1])] if len(dgm_h1) > 0 else np.empty((0, 2))
        )
        main_h1 = None
        if len(dgm_h1_fin) > 0:
            pers = dgm_h1_fin[:, 1] - dgm_h1_fin[:, 0]
            best = int(np.argmax(pers))
            main_h1 = (float(dgm_h1_fin[best, 0]), float(dgm_h1_fin[best, 1]))

        h1_dots = []  # [(birth_r, death_r, Dot)]
        for pt in dgm_h1_fin:
            b, d = float(pt[0]), float(pt[1])
            dot = Dot(
                point=pd_ax.c2p(b, b),
                color=ORANGE,
                radius=0.08,
                fill_opacity=0,
                stroke_width=1.0,
                stroke_color=WHITE,
            )
            h1_dots.append((b, d, dot))

        # ── 9.  Add all pre-built elements to the scene ───────────────────────────
        all_edges = VGroup(*[e for _, e in edge_data])
        all_tris = VGroup(*[t for _, t in tri_data])
        all_h0_dots = VGroup(*[d for _, d in h0_dots])
        all_h1_dots = VGroup(*[d for _, _, d in h1_dots])
        self.add(all_tris, all_edges, all_h0_dots, all_h1_dots)

        # ── 10.  Pointer-based updaters on r_tracker ──────────────────────────────
        # Each updater advances a pointer through a pre-sorted list, revealing
        # elements as r crosses their threshold — O(newly revealed) per frame.
        edge_ptr = [0]
        tri_ptr = [0]

        def _reveal_edges(_mob):
            r = r_tracker.get_value()
            while edge_ptr[0] < len(edge_data) and edge_data[edge_ptr[0]][0] <= r:
                edge_data[edge_ptr[0]][1].set_stroke(opacity=0.5)
                edge_ptr[0] += 1

        def _reveal_tris(_mob):
            r = r_tracker.get_value()
            while tri_ptr[0] < len(tri_data) and tri_data[tri_ptr[0]][0] <= r:
                tri_data[tri_ptr[0]][1].set_fill(opacity=0.1)
                tri_ptr[0] += 1

        # Same reasoning as VietorisRipsExplanation: attach to scene mobjects,
        # not to r_tracker, so updaters fire every frame during self.play().
        all_edges.add_updater(_reveal_edges)
        all_tris.add_updater(_reveal_tris)

        # Per-dot updaters for H₁ (position + visibility)
        for birth_r, death_r, dot in h1_dots:

            def _h1_upd(m, b=birth_r, d=death_r):
                r = r_tracker.get_value()
                if r < b:
                    m.set_fill(opacity=0.0)
                else:
                    m.set_fill(opacity=0.9)
                    m.move_to(pd_ax.c2p(b, min(r, d)))

            dot.add_updater(_h1_upd)

        # ── 11.  Show H₀ births (all at r = 0) before the sweep ──────────────────
        for _, dot in h0_dots:
            dot.set_fill(opacity=0.85)
        self.wait(0.5)
        self.next_slide()  # [Slide 2] H₀ components present — explain birth = 0

        # Attach H₀ sliding updaters after the birth pause
        for death_r, dot in h0_dots:

            def _h0_upd(m, dr=death_r):
                m.move_to(pd_ax.c2p(0.0, min(r_tracker.get_value(), dr)))

            dot.add_updater(_h0_upd)

        # ── 12.  Three continuous sweeps, pausing at dominant H₁ events ──────────
        h1_birth = main_h1[0] if main_h1 else max_r
        h1_death = main_h1[1] if main_h1 else max_r
        TOTAL_DURATION = 16.0

        # Segment A: 0 → h1_birth  (tracker sweeps; edges/tris pop in via updaters)
        t_a = max(0.5, h1_birth / max_r * TOTAL_DURATION)
        self.play(r_tracker.animate.set_value(h1_birth), run_time=t_a, rate_func=linear)
        self.next_slide()  # [Slide N] H₁ loop just born — explain the orange dot

        # Segment B: h1_birth → h1_death
        t_b = max(0.5, (h1_death - h1_birth) / max_r * TOTAL_DURATION)
        self.play(r_tracker.animate.set_value(h1_death), run_time=t_b, rate_func=linear)
        h1_main_dot = next(
            (
                d
                for b, dth, d in h1_dots
                if np.isclose(b, h1_birth) and np.isclose(dth, h1_death)
            ),
            None,
        )
        if h1_main_dot is not None:
            self.play(Flash(h1_main_dot, color=ORANGE, flash_radius=0.18), run_time=0.4)
        self.marked_next_slide()  # [Slide N+1] loop killed — explain persistence

        # Segment C: h1_death → max_r
        t_c = max(0.5, (max_r - h1_death) / max_r * TOTAL_DURATION)
        self.play(r_tracker.animate.set_value(max_r), run_time=t_c, rate_func=linear)

        # ── 13.  Final state ───────────────────────────────────────────────────────
        self.play(
            Flash(inf_h0_marker, color=BLUE, flash_radius=0.18),
            inf_h0_marker.animate.set_fill(opacity=0.85),
            run_time=0.5,
        )
        self.wait(1.0)
        self.marked_next_slide()  # [Final slide] complete persistence diagram


class EffectiveResistanceCommute(TIMCSlide):
    """
    Demonstrates effective resistance through the commute-time intuition.

    Key message: R_eff is governed by the *number* of paths between two nodes.
    Two nodes connected by many medium-length paths can have lower resistance
    than two nodes connected by a single short path if all other routes between
    the latter are extremely long.

    Slide flow
    ----------
    Phase 1 – Setup
      1. Title
      2. Reveal 3×4 grid; highlight s (row 1, left) and t (row 1, right)
      3. Commute-time definition + counter widget appears

    Phase 2 – Walks on full grid (many paths)
      4. Slow walk 1: step-by-step dot + trail (0.45s/hop)
      5. Faster walk 2: (0.2s/hop), rolling avg counter
      6. Fast flash walks × 10: polyline traces, counter settles low
         → caption "Many paths → low resistance"

    Phase 3 – Prune to bottleneck (single bridge)
      7. FadeOut most cross-edges; highlight one bridge; |E| updates
      8. Slow walk 1 on bottleneck: long detour, counter jumps high
      9. Fast flash walks × 8 on bottleneck: counter settles high
         → side-by-side annotation comparing both averages

    Phase 4 – Formulas
     10. Clear, show L definition, R_eff = (e_i−e_j)^T L+ (e_i−e_j),
         C(i,j) = 2|E|·R_eff(i,j)
    """

    # ------------------------------------------------------------------ #
    #  Grid helpers                                                        #
    # ------------------------------------------------------------------ #

    ROWS = 3
    COLS = 4
    DX = 2.1  # horizontal spacing
    DY = 2.0  # vertical spacing
    GRID_LEFT = -4.5

    def _node_pos(self, r, c):
        """Manim position of grid node (row r, col c)."""
        x = self.GRID_LEFT + c * self.DX
        y = (self.ROWS - 1) / 2 * self.DY - r * self.DY
        return np.array([x, y, 0.0])

    def _node_index(self, r, c):
        return r * self.COLS + c

    def _build_grid(self):
        """
        Return (positions, edge_list, dot_objects, edge_objects).
        edge_list : list of (i, j) node-index pairs (i < j)
        """
        N = self.ROWS * self.COLS
        positions = [
            self._node_pos(r, c) for r in range(self.ROWS) for c in range(self.COLS)
        ]

        edge_list = []
        for r in range(self.ROWS):
            for c in range(self.COLS):
                idx = self._node_index(r, c)
                if c + 1 < self.COLS:  # horizontal edge
                    edge_list.append((idx, self._node_index(r, c + 1)))
                if r + 1 < self.ROWS:  # vertical edge
                    edge_list.append((idx, self._node_index(r + 1, c)))

        dots = [
            Dot(positions[k], radius=0.13, color=DEFAULT_COLOR).set_z_index(2)
            for k in range(N)
        ]

        edges = {
            (i, j): Line(
                positions[i], positions[j], color=DEFAULT_COLOR, stroke_width=3
            )
            for (i, j) in edge_list
        }

        return positions, edge_list, dots, edges

    # ------------------------------------------------------------------ #
    #  Counter widget                                                      #
    # ------------------------------------------------------------------ #

    def _make_counter(self, n_edges):
        """
        Return (VGroup widget, DecimalNumber value_mob, Integer n_edges_mob).
        Counter displays:  avg_walk_length / |E| = ???
        """
        path_label = Text("Walk length", font_size=24)
        path_val = DecimalNumber(
            0,
            num_decimal_places=0,
            font_size=48,
            color=ACCENT_COLOR,
        )
        path_val.next_to(path_label, DOWN, buff=0.18)
        label = Tex(r"Average $\frac{\text{walk length}}{|E|}$", font_size=36)
        label.next_to(path_val, DOWN, buff=0.4)
        val = DecimalNumber(
            0.0,
            num_decimal_places=2,
            font_size=48,
            color=ACCENT_COLOR,
        )
        val.next_to(label, DOWN, buff=0.18)

        n_lbl = Text(f"|E| = {n_edges}", font_size=18).next_to(label, DOWN, buff=0.18)
        n_lbl.align_to(label, LEFT)

        box = VGroup(path_label, path_val, label, val)
        box.to_edge(RIGHT, buff=0.35).move_to([box.get_center()[0], 0, 0])
        return box, path_val, val

    # ------------------------------------------------------------------ #
    #  Random-walk helpers                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _adjacency(edge_list, n_nodes):
        adj = [[] for _ in range(n_nodes)]
        for i, j in edge_list:
            adj[i].append(j)
            adj[j].append(i)
        return adj

    @staticmethod
    def _random_walk_to(start, target, adj, rng, max_steps=2000):
        """Walk until we reach *target*; return list of node indices visited."""
        path = [start]
        cur = start
        for _ in range(max_steps):
            if cur == target:
                break
            nxt = rng.choice(adj[cur])
            path.append(nxt)
            cur = nxt
        return path

    def _round_trip(self, start, target, adj, rng):
        """s → t walk concatenated with t → s walk (share one endpoint)."""
        leg1 = self._random_walk_to(start, target, adj, rng)
        leg2 = self._random_walk_to(target, start, adj, rng)
        return leg1 + leg2[1:]  # avoid duplicating the target node

    # ------------------------------------------------------------------ #
    #  Animation primitives                                                #
    # ------------------------------------------------------------------ #

    def _animate_walk_slow(self, path, positions, dot, step_time, max_anim_steps=32):
        """
        Animate a dot moving step-by-step; each edge leaves a fading trail.
        If the path is longer than *max_anim_steps*, only the first
        max_anim_steps hops are shown step-by-step.  An ellipsis label
        appears briefly to signal that the walk continues off-screen.
        """
        display_path = path[: max_anim_steps + 1]
        truncated = len(path) - 1 > max_anim_steps

        trail_segs = []
        for k in range(1, len(display_path)):
            p0 = positions[display_path[k - 1]]
            p1 = positions[display_path[k]]
            seg = Line(p0, p1, color=dot.get_color(), stroke_width=8)
            self.play(
                dot.animate.move_to(p1),
                Create(seg),
                run_time=step_time,
                rate_func=linear,
            )
            # self.add(seg)
            trail_segs.append(seg)

        if truncated:
            ellipsis = Text("...", font_size=36, color=dot.get_color())
            ellipsis.next_to(dot, UP, buff=0.15)
            self.play(FadeIn(ellipsis, run_time=0.25))
            self.wait(0.5)
            # Snap the dot to the true endpoint without animation
            true_end = positions[path[-1]]
            self.play(FadeOut(ellipsis, run_time=0.2))
            self.play(dot.animate.move_to(true_end), run_time=0.3)

        # Fade all trail segments together
        if trail_segs:
            self.play(*[FadeOut(s) for s in trail_segs], run_time=1.2)

    def _animate_walk_flash(self, path, positions, color=None):
        """Flash the full path as a polyline then fade it out quickly."""
        if color is None:
            color = COLOR_CYCLE[4]
        pts = [positions[n] for n in path]
        poly = VMobject(color=color, stroke_width=8, stroke_opacity=0.8)
        poly.set_points_as_corners(pts)
        self.play(Create(poly, run_time=0.35, rate_func=linear))
        self.play(FadeOut(poly, run_time=0.5))

    def _animate_walk_medium(
        self, path, positions, dot, run_time, fade_run_time=0.75, color=None
    ):
        """
        Move *dot* along *path* in a single self.play call, with the dot
        leading and a trail growing behind it.

        UpdateFromAlphaFunc rebuilds the trail at each alpha tick using the
        exact node positions, lagging exactly one segment behind the dot.
        This is frame-rate independent — the trail always follows the grid
        edges exactly regardless of how many hops per second are animated.

        This produces exactly 2 self.play calls regardless of path length,
        keeping the total animation count low enough for manim-slides to
        concatenate without errors.
        """
        if color is None:
            color = dot.get_color()
        pts = [positions[n] for n in path]
        n_segs = len(pts) - 1

        # Geometric path for MoveAlongPath.
        walk_path = VMobject()
        walk_path.set_points_as_corners(pts)
        dot.move_to(pts[0])

        trail = VMobject(stroke_color=color, stroke_width=8, stroke_opacity=0.85)

        def trail_updater(mob, alpha):
            # Lag trail by exactly one segment so dot always leads.
            lag_alpha = max(0.0, alpha - 1.0 / n_segs)
            if lag_alpha < 1e-9:
                mob.set_points_as_corners([pts[0], pts[0]])
                return
            seg_float = lag_alpha * n_segs
            full_segs = int(seg_float)
            partial = seg_float - full_segs
            if full_segs >= n_segs:
                mob.set_points_as_corners(pts)
            else:
                mid = pts[full_segs] + partial * (pts[full_segs + 1] - pts[full_segs])
                mob.set_points_as_corners(list(pts[: full_segs + 1]) + [mid])

        self.add(trail)
        self.play(
            UpdateFromAlphaFunc(trail, trail_updater, rate_func=linear),
            MoveAlongPath(dot, walk_path, rate_func=linear),
            run_time=run_time,
        )
        self.play(FadeOut(trail), run_time=fade_run_time)

    # ------------------------------------------------------------------ #
    #  construct                                                           #
    # ------------------------------------------------------------------ #

    def construct(self):
        rng = np.random.default_rng(42)
        N = self.ROWS * self.COLS

        # s = row 1, col 0 (left-middle);  t = row 1, col 3 (right-middle)
        IDX_S = self._node_index(1, 0)
        IDX_T = self._node_index(1, 3)

        positions, full_edge_list, dots, edge_mobs = self._build_grid()

        NI = self._node_index  # shorthand

        # ── Canned walks for pedagogically clear slow illustrations. ──────────
        # Walk 1: go via the top row to t, return via the bottom row to s.
        canned_walk1 = [
            IDX_S,
            NI(0, 0),
            NI(0, 1),
            NI(0, 2),
            NI(0, 3),
            IDX_T,
            NI(2, 3),
            NI(2, 2),
            NI(2, 1),
            NI(2, 0),
            IDX_S,
        ]
        # Walk 2: brief detour up through col-1 before crossing, direct return.
        canned_walk2 = [
            IDX_S,
            NI(1, 1),
            NI(0, 1),
            NI(0, 2),
            NI(1, 2),
            IDX_T,
            NI(1, 2),
            NI(1, 1),
            IDX_S,
        ]
        # Bottleneck walk: bounces near the bridge before eventually crossing.
        canned_bot_walk = [
            IDX_S,
            NI(1, 1),
            NI(0, 1),
            NI(1, 1),
            NI(2, 1),
            NI(1, 1),
            NI(1, 2),
            NI(0, 2),
            NI(1, 2),
            IDX_T,
            NI(0, 3),
            NI(0, 2),
            NI(1, 2),
            NI(1, 1),
            IDX_S,
        ]

        # ============================================================
        # Slide 1 – Title
        # ============================================================
        title = Text("Effective Resistance & Commute Time", font_size=44)
        subtitle = Text("How many paths are there between two nodes?", font_size=28)
        subtitle.next_to(title, DOWN, buff=0.45)
        self.play(Write(title))
        self.play(FadeIn(subtitle, shift=UP * 0.15))
        self.marked_next_slide()
        self.play(FadeOut(title), FadeOut(subtitle))

        # ============================================================
        # Slide 2 – Reveal 4×4 grid
        # ============================================================
        all_edge_mobs = VGroup(*edge_mobs.values())
        all_dots = VGroup(*dots)

        self.play(LaggedStart(*[FadeIn(d, scale=1.4) for d in dots], lag_ratio=0.04))
        self.play(Create(all_edge_mobs, run_time=1.2, lag_ratio=0.04))

        # Colour s and t
        dot_s = dots[IDX_S]
        dot_t = dots[IDX_T]
        lbl_s = Text("source", font_size=28, color=COLOR_CYCLE[0]).next_to(
            dot_s, LEFT, buff=0.2
        )
        lbl_t = Text("target", font_size=28, color=COLOR_CYCLE[1]).next_to(
            dot_t, RIGHT, buff=0.2
        )
        self.play(
            dot_s.animate.set_color(COLOR_CYCLE[0]).scale(1.6).set_z_index(4),
            dot_t.animate.set_color(COLOR_CYCLE[1]).scale(1.6).set_z_index(4),
            FadeIn(lbl_s),
            FadeIn(lbl_t),
        )
        self.marked_next_slide()

        # ============================================================
        # Slide 3 – Commute-time definition + counter
        # ============================================================
        defn = Text(
            "Commute time  C(source, target)  =  expected steps for  source → target → source",
            font_size=24,
        ).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(defn, shift=UP * 0.1))

        n_edges_full = len(full_edge_list)
        counter_grp, path_val, counter_val = self._make_counter(n_edges_full)
        self.play(FadeIn(counter_grp))
        self.marked_next_slide()

        # ============================================================
        # Phase 2 – Walks on full grid
        # ============================================================
        adj_full = self._adjacency(full_edge_list, N)

        # -- Walk 1 (slow, canned: scenic route via top and bottom rows) ------
        walk_dot = Dot(positions[IDX_S], radius=0.08, color=COLOR_CYCLE[3])
        self.add(walk_dot)
        self._animate_walk_slow(canned_walk1, positions, walk_dot, step_time=0.42)

        commute_steps = [len(canned_walk1) - 1]  # steps = hops = len(path)-1
        rolling_avg = np.mean(commute_steps)
        self.play(ChangeDecimalToValue(path_val, commute_steps[-1]))
        self.play(ChangeDecimalToValue(counter_val, rolling_avg / n_edges_full))
        self.marked_next_slide()

        # -- Walk 2 (faster, canned: detour then direct return) ----------------
        self._animate_walk_slow(
            canned_walk2, positions, walk_dot, step_time=0.22, max_anim_steps=20
        )

        commute_steps.append(len(canned_walk2) - 1)
        rolling_avg = np.mean(commute_steps)
        self.play(ChangeDecimalToValue(path_val, commute_steps[-1]))
        self.play(ChangeDecimalToValue(counter_val, rolling_avg / n_edges_full))
        self.marked_next_slide()

        for k in range(16):
            walk = self._round_trip(IDX_S, IDX_T, adj_full, rng)
            self._animate_walk_medium(
                walk,
                positions,
                walk_dot,
                run_time=4.0 / (k + 1),
                fade_run_time=1.0 / np.sqrt(k + 1),
            )

            commute_steps.append(len(walk) - 1)
            rolling_avg = np.mean(commute_steps)
            self.play(
                ChangeDecimalToValue(path_val, commute_steps[-1]),
                ChangeDecimalToValue(counter_val, rolling_avg / n_edges_full),
                run_time=1.0 / np.sqrt(k + 1),
            )
        self.play(FadeOut(walk_dot))

        self.marked_next_slide()

        # -- Fast flash walks (10 more) ------------------------------
        # self.play(FadeOut(walk_dot))
        # for k in range(10):
        #     wk = self._round_trip(IDX_S, IDX_T, adj_full, rng)
        #     commute_steps.append(len(wk) - 1)
        #     self._animate_walk_flash(wk, positions, color=COLOR_CYCLE[3])
        #     rolling_avg = np.mean(commute_steps)
        #     self.play(
        #         counter_val.animate.set_value(rolling_avg / n_edges_full),
        #         run_time=(0.66 / np.sqrt(k + 1)),
        #     )

        caption_full = Text(
            "Many paths  →  short average commute  →  low  R_eff",
            font_size=24,
            color=COLOR_CYCLE[2],
        ).to_edge(DOWN, buff=0.5)
        self.play(FadeOut(defn), FadeIn(caption_full))
        full_avg_val = rolling_avg / n_edges_full
        self.marked_next_slide()

        # ============================================================
        # Phase 3 – Prune to bottleneck
        # ============================================================
        # Remove all horizontal edges crossing cols 1→2 except the one
        # at row 1 (the bridge row). Also remove most col 2→3 cross-edges
        # to really force the walker through the bottleneck.
        # Bridge = horizontal edge (row=1, col1→col2).
        bridge_idx_pair = (
            self._node_index(1, 1),
            self._node_index(1, 2),
        )

        # Edges to prune: all horizontal edges in cols 1→2 except the bridge.
        edges_to_prune = []
        for r in range(self.ROWS):
            for c in range(self.COLS - 1):
                i = self._node_index(r, c)
                j = self._node_index(r, c + 1)
                key = (min(i, j), max(i, j))
                # Keep only the bridge and the outermost column edges
                if c == 1 and r != 1:
                    edges_to_prune.append(key)

        # Also prune the direct horizontal edges cols 0→1 (rows 0 and 3)
        # so that the bridge is the dominant bottleneck.
        extra_prune = [
            (self._node_index(0, 0), self._node_index(0, 1)),
            (self._node_index(2, 0), self._node_index(2, 1)),
        ]
        for ep in extra_prune:
            edges_to_prune.append(ep)

        # Deduplicate
        edges_to_prune = list({(min(a, b), max(a, b)) for a, b in edges_to_prune})

        pruned_mobs = [edge_mobs[k] for k in edges_to_prune if k in edge_mobs]
        pruned_set = set(edges_to_prune)

        self.play(FadeOut(caption_full))
        prune_caption = Text("Prune to a single bridge", font_size=28).to_edge(
            DOWN, buff=0.5
        )
        self.play(Write(prune_caption))
        self.play(
            LaggedStart(*[FadeOut(m) for m in pruned_mobs], lag_ratio=0.12),
            run_time=1.2,
        )

        # # Highlight the bridge
        # bridge_key = (min(bridge_idx_pair), max(bridge_idx_pair))
        # bridge_mob = edge_mobs[bridge_key]
        # self.play(bridge_mob.animate.set_color(HIGHLIGHT_COLOR).set_stroke(width=6))

        # Rebuild adjacency on pruned graph
        pruned_edge_list = [
            (i, j)
            for (i, j) in full_edge_list
            if (min(i, j), max(i, j)) not in pruned_set
        ]
        n_edges_pruned = len(pruned_edge_list)
        adj_pruned = self._adjacency(pruned_edge_list, N)

        # Update |E| label
        # new_ne_lbl = Text(f"|E| = {n_edges_pruned}", font_size=18).move_to(
        #     counter_ne_lbl
        # )
        # new_ne_lbl.align_to(counter_ne_lbl, LEFT)
        # self.play(
        #     FadeOut(counter_ne_lbl),
        #     FadeIn(new_ne_lbl),
        #     run_time=0.5,
        # )
        # counter_ne_lbl = new_ne_lbl

        self.marked_next_slide()

        # -- Walk on bottleneck (canned: shows bouncing at bridge) ------------
        self.play(FadeOut(prune_caption))
        self.play(ChangeDecimalToValue(path_val, 0))
        self.play(ChangeDecimalToValue(counter_val, 0.0))
        commute_steps_bot = []

        walk_dot2 = Dot(positions[IDX_S], radius=0.08, color=COLOR_CYCLE[3])
        self.add(walk_dot2)
        self._animate_walk_slow(
            canned_bot_walk, positions, walk_dot2, step_time=0.32, max_anim_steps=22
        )

        commute_steps_bot.append(len(canned_bot_walk) - 1)
        rolling_avg_bot = np.mean(commute_steps_bot)
        self.play(ChangeDecimalToValue(path_val, commute_steps_bot[-1]))
        self.play(ChangeDecimalToValue(counter_val, rolling_avg_bot / n_edges_pruned))
        self.marked_next_slide()

        for k in range(16):
            walk = self._round_trip(IDX_S, IDX_T, adj_pruned, rng)
            self._animate_walk_medium(
                walk,
                positions,
                walk_dot2,
                run_time=4.0 / (k + 1),
                fade_run_time=1.0 / np.sqrt(k + 1),
            )

            commute_steps_bot.append(len(walk) - 1)
            rolling_avg_bot = np.mean(commute_steps_bot)
            self.play(ChangeDecimalToValue(path_val, commute_steps_bot[-1]))
            self.play(
                ChangeDecimalToValue(counter_val, rolling_avg_bot / n_edges_pruned),
                run_time=1.0 / np.sqrt(k + 1),
            )

        self.marked_next_slide()

        self.play(FadeOut(walk_dot2))

        # # -- Fast flash walks on bottleneck (8 more) -----------------
        # for k in range(8):
        #     wk = self._round_trip(IDX_S, IDX_T, adj_pruned, rng)
        #     commute_steps_bot.append(len(wk) - 1)
        #     self._animate_walk_flash(wk, positions, color=COLOR_CYCLE[4])
        #     rolling_avg_bot = np.mean(commute_steps_bot)
        #     self.play(
        #         counter_val.animate.set_value(rolling_avg_bot / n_edges_pruned),
        #         run_time=0.25,
        #     )

        bottleneck_avg_val = rolling_avg_bot / n_edges_pruned

        # Side-by-side comparison annotations
        ann_full = MathTex(
            rf"\text{{full graph: }}\approx {full_avg_val:.2f}",
            color=COLOR_CYCLE[2],
            font_size=26,
        ).to_edge(DOWN, buff=1.1)
        ann_bot = MathTex(
            rf"\text{{bottleneck: }}\approx {bottleneck_avg_val:.2f}",
            color=HIGHLIGHT_COLOR,
            font_size=26,
        ).to_edge(DOWN, buff=0.55)
        caption_bot = Text(
            "Fewer paths  →  longer average commute  →  higher  R_eff",
            font_size=24,
            color=HIGHLIGHT_COLOR,
        ).to_edge(DOWN, buff=0.5)

        self.play(FadeIn(ann_full), FadeIn(ann_bot))
        self.marked_next_slide()

        self.play(FadeOut(ann_full), FadeOut(ann_bot), FadeIn(caption_bot))
        self.marked_next_slide()

        # ============================================================
        # Phase 4 – Formulas
        # ============================================================
        everything = VGroup(
            all_edge_mobs,
            all_dots,
            lbl_s,
            lbl_t,
            counter_grp,
            caption_bot,
        )
        self.play(FadeOut(everything), run_time=0.8)

        form_title = Text("Effective Resistance", font_size=40)
        form_title.shift(UP * 2.6)

        lap_def = MathTex(
            r"L_{ij} = \begin{cases}"
            r"\deg(i) & i = j \\"
            r"-1 & \{i,j\} \in E \\"
            r"0 & \text{otherwise}"
            r"\end{cases}",
            font_size=34,
        ).next_to(form_title, DOWN, buff=0.5)

        reff_form = MathTex(
            r"R_{\mathrm{eff}}(i,j) \;=\; "
            r"(\mathbf{e}_i - \mathbf{e}_j)^\top \, L^+ \, (\mathbf{e}_i - \mathbf{e}_j)",
            font_size=34,
        ).next_to(lap_def, DOWN, buff=0.5)

        commute_form = MathTex(
            r"\frac{C(i,j)}{2\,|E|} \;=\; R_{\mathrm{eff}}(i,j)",
            font_size=34,
        ).next_to(reff_form, DOWN, buff=0.42)
        commute_box = SurroundingRectangle(
            commute_form, color=HIGHLIGHT_COLOR, buff=0.16, stroke_width=3
        )

        self.play(Write(form_title))
        self.play(Write(lap_def))
        self.marked_next_slide()

        self.play(Write(reff_form))
        self.play(Write(commute_form), Create(commute_box))
        self.marked_next_slide()


import networkx as nx


class ElectricResistanceGrid(TIMCSlide):
    """
    Visualises effective resistance on a 3×4 grid as an electric circuit.

    Slide flow
    ----------
    1. Reveal the grid – uniform grey nodes, uniform-thickness grey edges.
    2. Scale up source and target; show "source"/"target" labels;
       show a horizontal viridis colorbar along the bottom whose range
       covers both the full and pruned graphs.
    3. Wave animation: source node lights up first, then the rest of the
       nodes colour-in by potential (high→low order) and edges thicken by
       current (|Δv|).
    4. Shift graph left; reveal right-aligned, vertically-centred info
       box showing "Effective Resistance", source/target colour swatches,
       and the numeric R_eff value.
    5. Reset graph to grey/uniform (ground state), then fade out the
       pruned edges, then redo the wave animation on the pruned graph,
       and update R_eff in the info box.
    """

    # ── Grid geometry ────────────────────────────────────────────────── #
    ROWS, COLS = 3, 4
    DX, DY = 2.2, 1.7

    # ── Visual constants ─────────────────────────────────────────────── #
    NODE_R = 0.15  # default node radius
    NODE_R_HL = 0.22  # highlighted (s, t) radius
    EDGE_W_UNIFORM = 4.0  # uniform edge stroke before potential applied
    EDGE_W_MIN = 1.5  # thinnest edge after potential applied
    EDGE_W_MAX = 13.0  # thickest edge after potential applied
    GRAPH_SHIFT = 2.0  # leftward shift for info-box slide

    # ── Helpers ──────────────────────────────────────────────────────── #

    def _node_pos(self, r, c):
        x = -(self.COLS - 1) * self.DX / 2 + c * self.DX
        y = (self.ROWS - 1) * self.DY / 2 - r * self.DY
        return np.array([x, y, 0.0])

    def _potentials(self, G, nodes, node_idx, source, target):
        """Return potential array via pseudoinverse of Laplacian."""
        L = nx.laplacian_matrix(G, nodelist=nodes).toarray().astype(float)
        b = np.zeros(len(nodes))
        b[node_idx[source]] = 1.0
        b[node_idx[target]] = -1.0
        return np.linalg.pinv(L) @ b

    def _make_colorbar(self, cmap_fn, v_min, v_max):
        """Return a horizontal VGroup colorbar along the bottom of the frame."""
        N, total_w, height, y = 128, 10.0, 0.30, -3.25
        rw = total_w / N
        rects = VGroup(
            *[
                Rectangle(
                    width=rw + 0.005,
                    height=height,
                    fill_color=cmap_fn(i / (N - 1)),
                    fill_opacity=1.0,
                    stroke_width=0,
                ).move_to([-total_w / 2 + (i + 0.5) * rw, y, 0])
                for i in range(N)
            ]
        )
        border = Rectangle(
            width=total_w + 0.02,
            height=height + 0.01,
            stroke_width=1.5,
            stroke_color=DARK_GRAY,
            fill_opacity=0,
        ).move_to([0, y, 0])
        lbl_lo = Text(f"{v_min:.2f}", font_size=17).next_to(rects[0], LEFT, buff=0.12)
        lbl_hi = Text(f"{v_max:.2f}", font_size=17).next_to(rects[-1], RIGHT, buff=0.12)
        title = Text("Potential", font_size=19).next_to(border, DOWN, buff=0.10)
        return VGroup(rects, border, lbl_lo, lbl_hi, title)

    # ── construct ────────────────────────────────────────────────────── #

    def construct(self):
        import networkx as nx
        import matplotlib.cm as _cm

        # ── Graphs ────────────────────────────────────────────────────────
        G = nx.grid_2d_graph(self.ROWS, self.COLS)
        source, target = (1, 0), (1, 3)
        nodes = list(G.nodes())
        node_idx = {nd: i for i, nd in enumerate(nodes)}

        # Pruned graph: remove only the two middle-third edges (top and bottom
        # of columns 1→2), leaving the left-third edges intact.
        G2 = G.copy()
        for e in [((0, 1), (0, 2)), ((2, 1), (2, 2))]:
            G2.remove_edge(*e)

        # ── Potentials ────────────────────────────────────────────────────
        pot1 = self._potentials(G, nodes, node_idx, source, target)
        pot2 = self._potentials(G2, nodes, node_idx, source, target)

        # Colorbar range covers both graphs so colours are comparable.
        v_min = min(pot1.min(), pot2.min())
        v_max = max(pot1.max(), pot2.max())

        R1 = float(pot1[node_idx[source]] - pot1[node_idx[target]])
        R2 = float(pot2[node_idx[source]] - pot2[node_idx[target]])

        # ── Colormap (matplotlib viridis) ─────────────────────────────────
        _cm_obj = _cm.viridis

        def cmap_fn(t):
            r, g, b, _ = _cm_obj(float(np.clip(t, 0, 1)))
            return "#{:02x}{:02x}{:02x}".format(
                int(r * 255), int(g * 255), int(b * 255)
            )

        def pcol(val):
            return cmap_fn((val - v_min) / (v_max - v_min + 1e-9))

        # ── Edge stroke-width helper ──────────────────────────────────────
        def sw(drop, mx):
            frac = drop / (mx + 1e-9)
            return self.EDGE_W_MIN + (self.EDGE_W_MAX - self.EDGE_W_MIN) * frac

        mx1 = max(abs(pot1[node_idx[u]] - pot1[node_idx[v]]) for u, v in G.edges())
        mx2 = max(abs(pot2[node_idx[u]] - pot2[node_idx[v]]) for u, v in G2.edges())

        # ── Build initial mobjects (uniform grey style) ────────────────────
        npos = {nd: self._node_pos(*nd) for nd in nodes}
        dots = {nd: Dot(npos[nd], radius=self.NODE_R, color=GRAY) for nd in nodes}
        emobs = {
            (u, v): Line(npos[u], npos[v], stroke_width=self.EDGE_W_UNIFORM, color=GRAY)
            for u, v in G.edges()
        }

        def em(u, v):
            return emobs.get((u, v)) or emobs.get((v, u))

        all_e = VGroup(*emobs.values())
        all_n = VGroup(*dots.values())

        # ── SLIDE 1: reveal plain grey grid ──────────────────────────────
        self.play(Create(all_e), Create(all_n))
        self.marked_next_slide()

        # ── SLIDE 2: highlight s/t; "source"/"target" labels; colorbar ───
        hl = self.NODE_R_HL / self.NODE_R
        self.play(dots[source].animate.scale(hl), dots[target].animate.scale(hl))

        s_lbl = Text("source", font_size=24).next_to(dots[source], LEFT, buff=0.25)
        t_lbl = Text("target", font_size=24).next_to(dots[target], RIGHT, buff=0.25)
        self.play(Write(s_lbl), Write(t_lbl))

        cb = self._make_colorbar(cmap_fn, v_min, v_max)
        self.play(FadeIn(cb))
        self.marked_next_slide()

        # ── SLIDE 3: wave animation — potentials + edge thicknesses ───────
        def _wave_anim(pot, mx, graph_obj):
            """Animate a potential wave and return the animations."""
            src_anim = [dots[source].animate.set_color(pcol(pot[node_idx[source]]))]
            others = sorted(
                [nd for nd in nodes if nd != source],
                key=lambda nd: -pot[node_idx[nd]],
            )
            n_anims = [
                dots[nd].animate.set_color(pcol(pot[node_idx[nd]])) for nd in others
            ]
            e_anims = [
                em(u, v).animate.set_stroke(
                    width=sw(abs(pot[node_idx[u]] - pot[node_idx[v]]), mx),
                    color=DARK_GRAY,
                )
                for u, v in graph_obj.edges()
            ]
            return src_anim, n_anims, e_anims

        src_a1, n_a1, e_a1 = _wave_anim(pot1, mx1, G)
        self.play(*src_a1)
        self.play(LaggedStart(*n_a1, lag_ratio=0.12), *e_a1, run_time=2.5)
        self.marked_next_slide()

        # ── SLIDE 4: shift graph left; reveal right-aligned info box ──────
        graph_grp = VGroup(all_e, all_n, s_lbl, t_lbl)
        self.play(graph_grp.animate.shift(LEFT * self.GRAPH_SHIFT))

        i_title = Text("Effective Resistance", font_size=26, color=DEFAULT_COLOR)
        i_src = Dot(radius=0.13, color=pcol(pot1[node_idx[source]]))
        i_minus = Text("−", font_size=30, color=DEFAULT_COLOR)
        i_tgt = Dot(radius=0.13, color=pcol(pot1[node_idx[target]]))
        i_swatches = VGroup(i_src, i_minus, i_tgt).arrange(RIGHT, buff=0.18)
        reff_num = DecimalNumber(
            R1, num_decimal_places=3, font_size=40, color=ACCENT_COLOR
        )
        info = VGroup(i_title, i_swatches, reff_num).arrange(DOWN, buff=0.30)
        info.to_edge(RIGHT, buff=0.35)
        info.move_to([info.get_center()[0], 0, 0])  # centre vertically

        self.play(FadeIn(info))
        self.marked_next_slide()

        # ── SLIDE 5: reset → prune → reflow wave → update R_eff ──────────
        # Reset graph to ground state (grey nodes, uniform edges).
        # Also grey out the info-box swatches and zero the R_eff value.
        reset_n = [dots[nd].animate.set_color(GRAY) for nd in nodes]
        reset_e = [
            em(u, v).animate.set_stroke(width=self.EDGE_W_UNIFORM, color=GRAY)
            for u, v in G.edges()
        ]
        self.play(
            *reset_n,
            *reset_e,
            i_src.animate.set_color(GRAY),
            i_tgt.animate.set_color(GRAY),
            ChangeDecimalToValue(reff_num, 0.0),
            run_time=1.2,
        )

        # Fade out the two pruned middle-third edges.
        prune_edges = [((0, 1), (0, 2)), ((2, 1), (2, 2))]
        pm = [m for e in prune_edges if (m := em(*e)) is not None]
        self.play(LaggedStart(*[FadeOut(m) for m in pm], lag_ratio=0.12), run_time=0.9)

        # Redo the wave on the pruned graph.
        src_a2, n_a2, e_a2 = _wave_anim(pot2, mx2, G2)
        self.play(*src_a2)
        self.play(LaggedStart(*n_a2, lag_ratio=0.12), *e_a2, run_time=2.5)

        # Update info box: swatch colours and R_eff value.
        self.play(
            i_src.animate.set_color(pcol(pot2[node_idx[source]])),
            i_tgt.animate.set_color(pcol(pot2[node_idx[target]])),
            ChangeDecimalToValue(reff_num, R2),
        )
        self.marked_next_slide()


class ForceParameterization(TIMCSlide):
    def construct(self):

        axes = Axes(
            x_range=[0.0, 2.5, 0.5],
            y_range=[0.0, 1.25, 0.5],
            x_length=8,
            y_length=6,
            axis_config={"include_tip": False},
        ).add_coordinates()

        self.play(Create(axes))

        # f(x) = 1 / (1 + (max(0, x - shift) / scale)^p)^q
        BASE_SCALE = 0.75
        BASE_P = 2.0
        BASE_Q = 1.5

        def force(x, scale=BASE_SCALE, shift=0.0, p=BASE_P, q=BASE_Q):
            v = max(0.0, x - shift) / scale
            return 1.0 / (1.0 + v**p) ** q

        PLOT_STEP = 0.005
        graph = axes.plot(
            lambda x: force(x),
            x_range=[0.0, 2.5, PLOT_STEP],
            color=ACCENT_COLOR,
            stroke_width=9,
        )
        self.play(Create(graph))

        self.marked_next_slide()

        import matplotlib.cm as _cm

        def _plasma(t):
            """Sample the plasma colormap at t in [0, 1], return hex string."""
            r, g, b, _ = _cm.plasma(float(t))
            return "#{:02x}{:02x}{:02x}".format(
                int(r * 255), int(g * 255), int(b * 255)
            )

        def make_ghost_fan(param, values, **fixed):
            n = len(values)
            g = VGroup()
            for i, val in enumerate(values):
                kw = dict(fixed, **{param: val})
                color = _plasma(i / max(n - 1, 1))
                gc = axes.plot(
                    lambda x, kw=kw: force(x, **kw),
                    x_range=[0.0, 2.5, PLOT_STEP],
                    color=color,
                    stroke_width=2.5,
                )
                gc.set_stroke(opacity=0.55)
                g.add(gc)
            return g

        def smooth_sweep(waypoints, make_graph_fn, total_duration=3.0):
            """
            Continuously morphs the live curve through each waypoint value.
            Uses a ValueTracker + always_redraw so transitions are frame-continuous
            with no discrete steps or pauses between waypoints.
            """
            nonlocal graph
            n_segs = len(waypoints) - 1
            seg_dur = total_duration / n_segs
            param_t = ValueTracker(waypoints[0])
            live = always_redraw(lambda: make_graph_fn(param_t.get_value()))
            self.remove(graph)
            self.add(live)
            for target in waypoints[1:]:
                self.play(
                    param_t.animate.set_value(target),
                    run_time=seg_dur,
                    rate_func=smooth,
                )
            # Swap live graph for a static one so later transforms work normally
            final = make_graph_fn(waypoints[-1])
            self.remove(live)
            self.add(final)
            graph = final

        # ── Scale ─────────────────────────────────────────────────────────
        SCALE_GHOSTS = [0.2, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        ghosts = make_ghost_fan(
            "scale",
            SCALE_GHOSTS,
            shift=0.0,
            p=BASE_P,
            q=BASE_Q,
        )
        param_text = Text("Scale", font_size=22).move_to(axes.c2p(1.5, 0.9))
        self.play(Write(param_text), FadeIn(ghosts))
        smooth_sweep(
            [BASE_SCALE, 2.0, 0.2, BASE_SCALE],
            lambda sv: axes.plot(
                lambda x, sv=sv: force(x, scale=sv),
                x_range=[0.0, 2.5, PLOT_STEP],
                color=ACCENT_COLOR,
                stroke_width=9,
            ),
        )
        self.play(FadeOut(ghosts), FadeOut(param_text))

        self.marked_next_slide()

        # ── Shoulder width ─────────────────────────────────────────────────
        SHIFT_GHOSTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        ghosts = make_ghost_fan(
            "shift",
            SHIFT_GHOSTS,
            scale=BASE_SCALE,
            p=BASE_P,
            q=BASE_Q,
        )
        param_text = Text("Shoulder width", font_size=22).move_to(axes.c2p(1.5, 0.9))
        self.play(Write(param_text), FadeIn(ghosts))
        # Shift only has a positive direction (base = 0); sweep out and back
        smooth_sweep(
            [0.0, 0.8, 0.0],
            lambda sh: axes.plot(
                lambda x, sh=sh: force(x, shift=sh),
                x_range=[0.0, 2.5, PLOT_STEP],
                color=ACCENT_COLOR,
                stroke_width=9,
            ),
            total_duration=2.0,
        )
        self.play(FadeOut(ghosts), FadeOut(param_text))

        self.marked_next_slide()

        # ── Shoulder sharpness: vary p while keeping p·q = BASE_P·BASE_Q ─
        # This holds the large-x tail exponent constant so only the knee
        # shape changes, not the rate of decay in the tail.
        TAIL_CONST = BASE_P * BASE_Q  # = 3.0
        P_GHOSTS = [0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]

        def _sharpness_ghost_fan():
            n = len(P_GHOSTS)
            g = VGroup()
            for i, pv in enumerate(P_GHOSTS):
                qv = TAIL_CONST / pv
                color = _plasma(i / max(n - 1, 1))
                gc = axes.plot(
                    lambda x, pv=pv, qv=qv: force(x, p=pv, q=qv),
                    x_range=[0.0, 2.5, PLOT_STEP],
                    color=color,
                    stroke_width=2.5,
                )
                gc.set_stroke(opacity=0.55)
                g.add(gc)
            return g

        ghosts = _sharpness_ghost_fan()
        param_text = Text("Shoulder sharpness", font_size=22).move_to(
            axes.c2p(1.5, 0.9)
        )
        self.play(Write(param_text), FadeIn(ghosts))
        smooth_sweep(
            [BASE_P, 8.0, 0.4, BASE_P],
            lambda pv: axes.plot(
                lambda x, pv=pv, qv=TAIL_CONST / pv: force(x, p=pv, q=qv),
                x_range=[0.0, 2.5, PLOT_STEP],
                color=ACCENT_COLOR,
                stroke_width=9,
            ),
        )
        self.play(FadeOut(ghosts), FadeOut(param_text))

        self.marked_next_slide()

        # ── Tail decay (outer exponent q only; scale and p fixed) ─────────
        # q < 1: heavy tails, force persists at large distances
        # q >> 1: tight support, force drops to near-zero quickly
        Q_GHOSTS = [0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
        ghosts = make_ghost_fan(
            "q",
            Q_GHOSTS,
            scale=BASE_SCALE,
            shift=0.0,
            p=BASE_P,
        )
        param_text = Text("Tail decay", font_size=22).move_to(axes.c2p(1.5, 0.9))
        self.play(Write(param_text), FadeIn(ghosts))
        smooth_sweep(
            [BASE_Q, 5.0, 0.4, BASE_Q],
            lambda qv: axes.plot(
                lambda x, qv=qv: force(x, q=qv),
                x_range=[0.0, 2.5, PLOT_STEP],
                color=ACCENT_COLOR,
                stroke_width=9,
            ),
        )
        self.play(FadeOut(ghosts), FadeOut(param_text))


class AnimatedUMAPOptimization(TIMCSlide):
    """
    Animated depiction of UMAP's stochastic optimization process.

    Left panel: 10 random points in 2-D, with a highlighted pair (red vs blue).
    Right panel: UMAP's low-dimensional embedding of those points, starting as
                 a random scatter and gradually morphing into the final layout.

    During the morph, the highlighted pair is tracked with red/blue dots in the
    embedding; they start far apart and are pulled together as the optimization
    proceeds, illustrating how UMAP's attractive forces operate on similar points.
    """

    def construct(self):
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import sklearn.datasets as datasets

        mnist = datasets.fetch_openml("mnist_784", as_frame=False, parser="liac-arff")
        labels = mnist.target.astype(int)  # shape (70000,)

        cmap = plt.get_cmap("Spectral")
        norm = mcolors.Normalize(vmin=0, vmax=9)
        rgba_colors = cmap(norm(labels))  # (70000, 4) float64, values in [0, 1]

        umap_data = np.load("mnist_optimization_steps.npy")  # (n_frames, 70000, 2)

        # Normalize UMAP coordinates to fit Manim's screen (~14 × 8 units).
        # Use the global min/max across all frames to prevent clipping during animation.
        data_min = umap_data.min(axis=(0, 1))  # (2,)
        data_max = umap_data.max(axis=(0, 1))  # (2,)
        vis_center = (data_min + data_max) / 2
        vis_scale = min(
            12.0 / (data_max[0] - data_min[0]),
            7.0 / (data_max[1] - data_min[1]),
        )

        def _to_manim(frames_2d: np.ndarray) -> np.ndarray:
            """Map raw 2-D UMAP coordinates to Manim scene coordinates."""
            xy = (frames_2d - vis_center) * vis_scale
            return np.hstack([xy, np.zeros((xy.shape[0], 1))])

        # Initialize the PointCloud with per-point colors via add_points(rgbas=...).
        # PMobject.set_color() only accepts a single colour; per-point colours must
        # be supplied through the rgbas parameter of add_points().
        STROKE_INITIAL = 6.0
        STROKE_FINAL = 1.0
        n_frames = len(umap_data)
        # Linearly fade stroke_width from STROKE_INITIAL → STROKE_FINAL over a
        # 50-frame window that ends exactly 500 frames before the last frame.
        STROKE_FADE_END = n_frames - 500
        STROKE_FADE_START = STROKE_FADE_END - 50

        points = PMobject(stroke_width=STROKE_INITIAL)
        points.add_points(_to_manim(umap_data[0]), rgbas=rgba_colors)
        self.add(points)

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
            run_time=20,
            rate_func=linear,
        )
        points.remove_updater(update_points)
        self.marked_next_slide()


class HighDLissajousKnotUMAP2DWithEdges(TIMCSlide):

    _n_samples = 2000
    _seed = 42
    _generator_kwargs: dict = dict(
        target_dim=512,
        radius=2.0,
        n_planes=5,
        noise=0.01,
        noise_hd=0.175,
        dataset_type="highdim_loop",
    )
    _stroke_width = 16

    def construct(self):
        import umap

        generator = CurvyLoopEmbedding(n_samples=self._n_samples, seed=self._seed)
        _, X_embedded, _ = generator.generate_dataset(**self._generator_kwargs)
        mapper = umap.UMAP(
            random_state=42,
            n_components=3,
            n_neighbors=16,
            repulsion_strength=0.5,
            n_epochs=500,
        ).fit(X_embedded)
        embedding_2d = _normalize_to_axes(
            np.ascontiguousarray(mapper.embedding_[:, [0, 2]])
        )

        fade_point_indices = np.random.RandomState(42).choice(
            self._n_samples, size=int(self._n_samples * 0.30), replace=False
        )
        fade_point_embedding_2d = embedding_2d[fade_point_indices]
        fade_point_colors = [
            colorcet.CET_C9[int(t / self._n_samples * (len(colorcet.CET_C9) - 1))]
            for t in fade_point_indices
        ]

        axes = Axes(x_range=[-2.0, 2.0], y_range=[-2.0, 2.0])
        points = self.create_points(embedding_2d, axes)
        fade_points = [
            Dot(axes.c2p(*pt), radius=0.03, color=c)
            for pt, c in zip(fade_point_embedding_2d, fade_point_colors)
        ]

        self.camera.get_thickening_nudges = _types.MethodType(
            lambda cam, t: _make_circular_nudges(t), self.camera
        )

        self.play(
            LaggedStart(*[FadeIn(fp) for fp in fade_points], lag_ratio=0.01),
            run_time=1.0,
        )
        self.play(FadeIn(points))

        self.wait()
        self.marked_next_slide()

        # Build one VMobject per width bucket.
        # Cairo sets line width once per draw call, so true per-segment widths
        # require separate objects. Bucketing keeps the count to O(N_BUCKETS)
        # instead of O(N_edges) while still conveying edge strength visually.
        # Each edge A→B is packed as the degenerate cubic bezier [A, A, B, B].
        N_BUCKETS = 64
        EDGE_WIDTH_MIN = 0.01
        EDGE_WIDTH_MAX = 1.0

        graph = mapper.graph_.tocoo()
        all_coords = np.array([axes.c2p(*pt) for pt in embedding_2d], dtype=np.float64)
        src = all_coords[graph.row]  # (M, 3)
        dst = all_coords[graph.col]  # (M, 3)
        weights = np.asarray(graph.data, dtype=np.float64)
        weights = (weights - weights.min()) / (
            weights.max() - weights.min() + 1e-9
        )  # normalise 0→1
        bucket_idx = np.clip((weights * N_BUCKETS).astype(int), 0, N_BUCKETS - 1)

        edge_grp = VGroup()
        for b in range(N_BUCKETS):
            mask = bucket_idx == b
            if not mask.any():
                continue
            pts = np.empty((4 * mask.sum(), 3), dtype=np.float64)
            pts[0::4] = src[mask]
            pts[1::4] = src[mask]
            pts[2::4] = dst[mask]
            pts[3::4] = dst[mask]
            w = EDGE_WIDTH_MIN + (EDGE_WIDTH_MAX - EDGE_WIDTH_MIN) * (
                b / (N_BUCKETS - 1)
            )
            mob = VMobject(
                stroke_width=w, stroke_color=GRAY, fill_opacity=0, stroke_opacity=w
            )
            mob.set_points(pts)
            edge_grp.add(mob)

        edge_grp.set_z_index(-1)
        self.play(
            LaggedStart(
                *[FadeIn(mob) for mob in edge_grp],
                lag_ratio=0.5,
                run_time=2.0,
            )
        )

        self.wait()
        self.marked_next_slide()

        self.play(PMFadeOut(points))
        self.play(
            LaggedStart(*[FadeOut(fp) for fp in fade_points], lag_ratio=0.01),
            run_time=0.5,
        )
        self.play(
            edge_grp.animate.scale(2.75, about_point=(-3.0, -1.5, 0.0)),
        )

    def create_points(self, data: np.ndarray, axes) -> PMobject:
        from manim.utils.color import color_to_rgba

        coords = np.array(
            [axes.c2p(*x) for x in data],
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
