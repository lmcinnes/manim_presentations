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
        self.wait(8)  # update to be longer as required
        self.stop_ambient_camera_rotation()

    def create_points(self, data, axes):
        points = VGroup()
        for x in data:
            if len(x) == 2:
                x = np.append(x, 0)  # Add z=0 for 2D points
            dot = Dot3D(point=axes.c2p(*x), color=ACCENT_COLOR, radius=0.05)
            points.add(dot)
        return points


class LissajousKnot(ThreeDTIMCSlide):

    def construct(self):
        generator = CurvyLoopEmbedding(n_samples=200, seed=42)
        X_low, X_embedded, metadata = generator.generate_dataset(
            target_dim=4,
            radius=2.0,
            n_planes=2,
            noise=0.0,
            noise_hd=0.075,
            dataset_type="highdim_loop",
        )
        pca_embedded = sklearn.decomposition.PCA(n_components=3).fit_transform(
            X_embedded
        )
        axes = ThreeDAxes(x_range=[-2.5, 2.5], y_range=[-2.5, 2.5], z_range=[-2.5, 2.5])
        points = self.create_points(pca_embedded, axes)

        self.move_camera(phi=75 * DEGREES, theta=30 * DEGREES)

        self.play(Create(axes))
        self.wait(1)
        self.play(Create(points))
        self.begin_ambient_camera_rotation(rate=0.75)
        self.wait(8)
        self.stop_ambient_camera_rotation()

    def create_points(self, data, axes):
        points = VGroup()
        for x, t in zip(data, np.linspace(0, 1, len(data))):
            if len(x) == 2:
                x = np.append(x, 0)  # Add z=0 for 2D points
            color = colorcet.CET_C9[int(t * (len(colorcet.CET_C9) - 1))]
            dot = Dot3D(point=axes.c2p(*x), color=color, radius=0.05)
            points.add(dot)
        return points


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
            # Rebuild each circle in-place via become() so the centre is always
            # exactly pts_manim[i] and no set_radius drift can occur.
            r = r_tracker.get_value()
            ball_r_screen = r * vis_scale / 2
            opacity = 0.5 if r > 1e-9 else 0.0
            for i, circ in enumerate(circles):
                circ.become(
                    Circle(
                        radius=max(0.001, ball_r_screen),
                        arc_center=pts_manim[i],
                        color=GRAY,
                        stroke_width=1.5,
                        fill_opacity=0,
                    ).set_stroke(opacity=opacity)
                )

        # Attach updaters to scene mobjects (all_edges / all_tris / all_circles),
        # NOT to r_tracker.  Manim suspends r_tracker's own updaters while it is
        # the animated object (suspend_mobject_updating=True by default), so any
        # updater on r_tracker only fires once at the very end of the clip.
        all_edges.add_updater(_reveal_edges)
        all_tris.add_updater(_reveal_tris)
        all_circles.add_updater(_update_circles)

        self.play(
            r_tracker.animate.set_value(max_r / 6),
            run_time=4.0,
            rate_func=rate_functions.ease_in_out_quad,
        )

        self.marked_next_slide()

        self.play(
            r_tracker.animate.set_value(max_r / 3),
            run_time=6.0,
            rate_func=rate_functions.ease_in_quad,
        )

        self.wait()


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
