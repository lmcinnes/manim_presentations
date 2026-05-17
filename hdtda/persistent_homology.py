from encodings.idna import dots

from manim import *

import sys

import umap

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

        # Create label group once at a fixed position anchored to the right of the axes.
        # All axes share the same x_length/y_length so this position never changes.
        _ref_axes = Axes(x_range=[0, 1], y_range=[0, 1], x_length=6, y_length=6)
        self.current_dim_label = Text("Ambient Dimension", font_size=24).next_to(
            _ref_axes, RIGHT, buff=0.5
        )
        self.dim_label_value = DecimalNumber(
            dimensions[0], num_decimal_places=0, font_size=48, color=ACCENT_COLOR
        ).next_to(self.current_dim_label, DOWN, buff=0.25)

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

            # --- ANIMATION LOGIC ---
            if i == 0:
                self.add(
                    new_axes,
                    new_diag,
                    new_inf_line,
                    self.current_dim_label,
                    self.dim_label_value,
                )
                for group in new_h_groups:
                    self.add(group)
            else:
                self.play(
                    ReplacementTransform(self.current_axes, new_axes),
                    ReplacementTransform(self.current_diag, new_diag),
                    ReplacementTransform(self.current_inf_line, new_inf_line),
                    ChangeDecimalToValue(self.dim_label_value, dimensions[i]),
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
            # self.current_dim_label_value = dim_label_value
            self.wait(0.1)

        self.marked_next_slide()

        self.play(
            FadeOut(self.current_axes),
            FadeOut(self.current_diag),
            FadeOut(self.current_inf_line),
            FadeOut(self.current_dim_label),
            FadeOut(self.dim_label_value),
            *[FadeOut(group) for group in self.current_h_groups],
        )

        self.add_centered_text("Is there simply no signal left in high dimensions?")

        self.marked_next_slide()

        self.clear_slide()

        reduced_embedding = sklearn.decomposition.PCA(n_components=3).fit_transform(
            datasets[-1]
        )
        reduced_dgms = ripser(reduced_embedding, maxdim=1)["dgms"]
        reduced_h_groups = self.get_homology_groups(
            reduced_dgms, self.current_axes, axis_bounds[-1]
        )

        self.play(
            FadeIn(self.current_axes),
            FadeIn(self.current_diag),
            FadeIn(self.current_inf_line),
            FadeIn(self.current_dim_label),
            FadeIn(self.dim_label_value),
            *[FadeIn(group) for group in self.current_h_groups],
        )

        self.marked_next_slide()

        self.play(
            ReplacementTransform(self.current_h_groups[1], reduced_h_groups[1]),
            ReplacementTransform(self.current_h_groups[0], reduced_h_groups[0]),
            ReplacementTransform(
                self.current_dim_label,
                Paragraph(
                    "PCA-Reduced\nAmbient Dimension",
                    font_size=24,
                    color=ACCENT_COLOR,
                    alignment="center",
                ).next_to(self.current_axes, RIGHT, buff=0.25),
            ),
            ChangeDecimalToValue(
                self.dim_label_value.next_to(self.current_dim_label, DOWN, buff=0.75), 3
            ),
        )

        self.marked_next_slide()

        new_axes = Axes(
            x_range=[0, axis_bounds[0], tick_step],
            y_range=[0, axis_bounds[0], tick_step],
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
            end=new_axes.c2p(axis_bounds[0], axis_bounds[0]),
            color=GRAY,
            stroke_width=2,
        )

        # Infinity Line at the very top of the Y-axis
        inf_y = new_axes.c2p(0, axis_bounds[0])[1]
        new_inf_line = DashedLine(
            start=[new_axes.c2p(0, 0)[0], inf_y, 0],
            end=[new_axes.c2p(axis_bounds[0], 0)[0], inf_y, 0],
            color=GRAY_C,
        )

        new_h_groups = self.get_homology_groups(reduced_dgms, new_axes, axis_bounds[0])

        self.play(
            ReplacementTransform(self.current_axes, new_axes),
            ReplacementTransform(self.current_diag, new_diag),
            ReplacementTransform(self.current_inf_line, new_inf_line),
            *[
                ReplacementTransform(reduced_h_groups[j], new_h_groups[j])
                for j in range(3)
            ],
            run_time=0.5,
        )

        self.wait()

        h1_nonnoise_feature = [
            dot
            for dot in new_h_groups[1]
            if dot.get_center()[1] > new_axes.c2p(0.5, 1.0)[1]
        ][0]
        for i in range(3):
            self.play(
                # Create(
                #     Circle(color=HIGHLIGHT_COLOR, stroke_width=3).surround(
                #         h1_nonnoise_feature,
                #         buffer_factor=3,
                #     )
                # )
                Flash(
                    h1_nonnoise_feature,
                    color=HIGHLIGHT_COLOR,
                    flash_radius=0.2,
                ),
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


class DynamicPersistenceLissajousPCA(TIMCSlide):

    _n_samples: int = 2000
    _seed: int = 42
    _generator_kwargs: dict = dict(
        target_dim=512,
        radius=2.0,
        n_planes=7,
        noise=0.01,
        noise_hd=0.25,
        dataset_type="highdim_loop",
    )

    def construct(self):
        print("Starting DynamicPersistenceLissajousPCA construction...")
        import sys

        sys.stdout.flush()
        # 1. Setup Data
        generator = CurvyLoopEmbedding(n_samples=self._n_samples, seed=self._seed)
        _, X_embedded, _ = generator.generate_dataset(**self._generator_kwargs)

        dimensions = np.arange(2, 32)
        # dimensions = np.round(np.linspace(2, np.cbrt(512), 256) ** 3).astype(np.int32)
        datasets = []
        diagrams = []
        for d in dimensions:
            print(f"Computing PCA + PD for dimension {d}...")
            sys.stdout.flush()

            X_pca = sklearn.decomposition.PCA(n_components=d).fit_transform(X_embedded)
            datasets.append(X_pca)
            dgms = ripser(X_pca, maxdim=1)["dgms"]
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

        # Create label group once at a fixed position anchored to the right of the axes.
        # All axes share the same x_length/y_length so this position never changes.
        _ref_axes = Axes(x_range=[0, 1], y_range=[0, 1], x_length=6, y_length=6)
        self.current_dim_label = Text("Ambient Dimension", font_size=24).next_to(
            _ref_axes, RIGHT, buff=0.5
        )
        self.dim_label_value = DecimalNumber(
            dimensions[0], num_decimal_places=0, font_size=48, color=ACCENT_COLOR
        ).next_to(self.current_dim_label, DOWN, buff=0.25)

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

            # --- ANIMATION LOGIC ---
            if i == 0:
                self.add(
                    new_axes,
                    new_diag,
                    new_inf_line,
                    self.current_dim_label,
                    self.dim_label_value,
                )
                for group in new_h_groups:
                    self.add(group)
            else:
                self.play(
                    ReplacementTransform(self.current_axes, new_axes),
                    ReplacementTransform(self.current_diag, new_diag),
                    ReplacementTransform(self.current_inf_line, new_inf_line),
                    ChangeDecimalToValue(self.dim_label_value, dimensions[i]),
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
            # self.current_dim_label_value = dim_label_value
            self.wait(0.1)

        self.marked_next_slide()

        self.play(
            FadeOut(self.current_axes),
            FadeOut(self.current_diag),
            FadeOut(self.current_inf_line),
            FadeOut(self.current_dim_label),
            FadeOut(self.dim_label_value),
            *[FadeOut(group) for group in self.current_h_groups],
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


class CompareLissajousPersistence(TIMCSlide):

    _n_samples: int = 2000
    _seed: int = 42
    _generator_kwargs: dict = dict(
        target_dim=512,
        radius=2.0,
        n_planes=7,
        noise=0.01,
        noise_hd=0.25,
        dataset_type="highdim_loop",
    )

    _LEFT_POS = LEFT * 3
    _RIGHT_POS = RIGHT * 3

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

        while len(groups) < 3:
            groups.append(VGroup())

        return groups

    def _build_pd_plot(self, dgms, bounds, step, title_str, axes_center):
        """
        Build a persistence diagram plot group.

        Returns a (plot, axes) tuple where plot is a VGroup of all mobjects and
        axes is the Axes object (used to compute exact shift vectors in transitions).
        """
        axes = (
            Axes(
                x_range=[0, bounds, step],
                y_range=[0, bounds, step],
                x_length=4,
                y_length=4,
                axis_config={
                    "include_tip": False,
                    "include_ticks": True,
                    "numbers_to_exclude": [0],
                },
            )
            .add_coordinates()
            .move_to(axes_center)
        )
        diag_line = Line(
            start=axes.c2p(0, 0),
            end=axes.c2p(bounds, bounds),
            color=GRAY,
            stroke_width=2,
        )
        inf_y = axes.c2p(0, bounds)[1]
        inf_line = DashedLine(
            start=[axes.c2p(0, 0)[0], inf_y, 0],
            end=[axes.c2p(bounds, 0)[0], inf_y, 0],
            color=GRAY_C,
        )
        title = Text(title_str, font_size=28).next_to(axes, UP, buff=0.3)
        h_groups = self.get_homology_groups(dgms, axes, bounds)
        plot = VGroup(axes, diag_line, inf_line, title, *h_groups)
        return plot, axes

    def construct(self):
        print("Starting CompareLissajousPersistence construction...")
        import sys
        from data_generation import effective_resistance_distance_embedding

        sys.stdout.flush()
        # 1. Setup Data
        generator = CurvyLoopEmbedding(n_samples=self._n_samples, seed=self._seed)
        _, X_embedded, _ = generator.generate_dataset(**self._generator_kwargs)

        # Compute PCA and PD for the original embedded data
        X_pca = sklearn.decomposition.PCA(n_components=3).fit_transform(X_embedded)
        dgm_pca = ripser(X_pca, maxdim=1)["dgms"]

        # Compute PD for the original high-dimensional data (using a subset for speed)
        subset_size = min(1000, len(X_embedded))
        dgm_hd = ripser(X_embedded[:subset_size], maxdim=1)["dgms"]

        X_eff = effective_resistance_distance_embedding(X_embedded)
        dgm_eff = ripser(X_eff, maxdim=1)["dgms"]

        X_umap = umap.UMAP(n_components=3, random_state=self._seed).fit_transform(
            X_embedded
        )
        dgm_umap = ripser(X_umap, maxdim=1)["dgms"]

        def _finite_h1_max(dgm):
            fin = dgm[1][np.isfinite(dgm[1])]
            return fin.max() * 1.2 if len(fin) > 0 else 1.0

        pca_bounds = _finite_h1_max(dgm_pca)
        hd_bounds = _finite_h1_max(dgm_hd)
        eff_bounds = _finite_h1_max(dgm_eff)
        umap_bounds = _finite_h1_max(dgm_umap)

        print(eff_bounds, umap_bounds)

        pca_step = self.get_optimal_step(pca_bounds)
        hd_step = self.get_optimal_step(hd_bounds)
        eff_step = self.get_optimal_step(eff_bounds)
        umap_step = self.get_optimal_step(umap_bounds)

        # --- Build all plots ---
        pca_plot, axes_pca = self._build_pd_plot(
            dgm_pca, pca_bounds, pca_step, "PCA Projection", self._LEFT_POS
        )
        hd_plot, _axes_hd = self._build_pd_plot(
            dgm_hd, hd_bounds, hd_step, "High-Dimensional Data", self._RIGHT_POS
        )
        eff_plot, axes_eff = self._build_pd_plot(
            dgm_eff,
            eff_bounds,
            eff_step,
            "Effective Resistance Distance",
            self._LEFT_POS,
        )
        umap_plot, _axes_umap = self._build_pd_plot(
            dgm_umap, umap_bounds, umap_step, "UMAP Projection", self._LEFT_POS
        )

        # --- Slide 1: PCA (left) + HD (right) ---
        self.play(
            LaggedStart(FadeIn(hd_plot), FadeIn(pca_plot), lag_ratio=0.5), run_time=1.0
        )
        self.add(pca_plot, hd_plot)
        self.wait(1.0)
        self.marked_next_slide()

        # --- Transition 1: HD fades, PCA shifts right, eff resistance appears left ---
        self.play(FadeOut(hd_plot), run_time=0.5)
        self.wait(0.5)
        self.play(
            pca_plot.animate.shift(self._RIGHT_POS - axes_pca.get_center()),
            run_time=0.75,
        )
        self.play(FadeIn(eff_plot), run_time=0.5)
        self.wait(1.0)
        self.marked_next_slide()

        # --- Transition 2: PCA fades, eff shifts right, UMAP appears left ---
        self.play(FadeOut(pca_plot), run_time=0.5)
        self.wait(0.5)
        self.play(
            eff_plot.animate.shift(self._RIGHT_POS - axes_eff.get_center()),
            run_time=0.75,
        )
        self.play(FadeIn(umap_plot), run_time=0.5)
        self.wait(1.0)
        self.marked_next_slide()

        self.clear_slide(run_time=1.0)

    def get_optimal_step(self, max_val):
        """Logic to keep ticks at 'round' intervals."""
        if max_val < 0.05:
            return 0.01
        if max_val < 0.1:
            return 0.02
        if max_val < 0.5:
            return 0.1
        if max_val < 1:
            return 0.2
        if max_val < 2:
            return 0.25
        if max_val < 3:
            return 0.5
        return 1.0
