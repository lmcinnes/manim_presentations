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
from data_generation import CircleEmbedding, TorusEmbedding

import numpy as np
from ripser import ripser

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
            X_2d, X_embedded, metadata = generator.generate_dataset(
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
                    "include_tip": True,
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
        X_2d, X_embedded, metadata = generator.generate_dataset(
            target_dim=3, noise_hd=0.16, radius=1.0, noise_2d=0.0, embedding="linear"
        )
        X_2d_3d = np.hstack([X_2d, np.zeros((X_2d.shape[0], 1))])
        cov = X_embedded.T @ X_2d_3d
        us, s, vt = np.linalg.svd(cov)
        if np.linalg.det(us @ vt) < 0:
            us[:, -1] *= -1
        R = us @ vt
        X_embedded_unprojected = X_embedded @ R.T

        axes = ThreeDAxes(x_range=[-1.5, 1.5], y_range=[-1.5, 1.5], z_range=[-1.5, 1.5])
        twod_points = self.create_points(X_2d, axes)
        threed_points_unprojected = self.create_points(X_embedded_unprojected, axes)
        threed_points = self.create_points(X_embedded, axes)

        self.play(Create(axes))
        # self.play(LaggedStart(*[FadeIn(p) for p in twod_points], lag_ratio=0.01))
        self.play(Create(twod_points))

        self.wait()

        self.move_camera(phi=75 * DEGREES, theta=30 * DEGREES)

        self.wait()

        self.play(ReplacementTransform(twod_points, threed_points_unprojected))
        self.wait()
        axis, angle = rotation_matrix_to_axis_angle(R)

        theta = ValueTracker(0)

        def update_dots(mob):
            t = theta.get_value()
            rot = rotation_matrix(axis=axis, angle=t)  # manim helper
            for i, d in enumerate(mob):
                d.move_to(axes.c2p(*(rot @ X_embedded_unprojected[i])))

        threed_points_unprojected.add_updater(update_dots)

        self.play(theta.animate.set_value(angle), run_time=2)
        threed_points_unprojected.remove_updater(update_dots)

        self.begin_ambient_camera_rotation(rate=0.75)
        self.wait(9)
        self.stop_ambient_camera_rotation()

    def create_points(self, data, axes):
        points = VGroup()
        for x in data:
            if len(x) == 2:
                x = np.append(x, 0)  # Add z=0 for 2D points
            dot = Dot3D(point=axes.c2p(*x), color=ACCENT_COLOR, radius=0.05)
            points.add(dot)
        return points
