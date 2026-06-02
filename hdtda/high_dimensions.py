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

import ot

apply_defaults()


def rotation_matrix_to_axis_angle(R):
    angle = np.arccos((np.trace(R) - 1) / 2)
    if np.isclose(angle, 0):
        return np.array([1, 0, 0]), 0

    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (
        2 * np.sin(angle)
    )

    return axis, angle


def _make_vr_demo_cloud(
    n_clusters: int = 8,
    circle_radius: float = 1.0,
    rect_min_side: float = 0.35,
    rect_max_side: float = 0.55,
    n_connectors: int = 10,
    seed: int = 13,
    tilt_phi: float = np.pi / 5,
    tilt_theta: float = np.pi / 4,
    noise_xy_scale: float = 0.05,
    noise_z_scale: float = 0.15,
) -> np.ndarray:
    """
    Build a 3-D point cloud that looks like a noisily sampled circle but has
    controlled topological properties:

    * One dominant H₁ loop (the circle) with long persistence.
    * Several short-persistence H₁ loops, one per rectangular cluster
      (points near the diagonal of the persistence diagram).
    * H₀ components that merge as the filtration grows.

    The circle lies in a tilted plane inside ℝ³, controlled by ``tilt_phi``
    (polar tilt from the z-axis) and ``tilt_theta`` (azimuthal direction of
    the tilt).  Anisotropic noise is applied in the local circle frame before
    tilting: ``noise_xy_scale`` adds in-plane scatter, while ``noise_z_scale``
    (larger by default) lifts points off the plane to give the cloud clear
    3-D thickness.

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
                local = R_mat @ np.array([gx, gy])
                points.append(np.array([cx + local[0], cy + local[1], 0.0]))

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
        points.append(np.array([r * np.cos(a), r * np.sin(a), 0.0]))

    pts = np.array(points)

    # Apply anisotropic noise in the local circle frame (z=0) before tilting:
    # noise_xy_scale spreads points within the plane; noise_z_scale lifts them
    # off it, making the 3-D cloud thickness clearly visible.
    pts[:, :2] += rng.normal(scale=noise_xy_scale, size=(len(pts), 2))
    pts[:, 2] += rng.normal(scale=noise_z_scale, size=len(pts))

    # Tilt the plane: first rotate around the x-axis by tilt_phi, then around
    # the z-axis by tilt_theta.
    cp, sp = np.cos(tilt_phi), np.sin(tilt_phi)
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    ct, st = np.cos(tilt_theta), np.sin(tilt_theta)
    Rz = np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1]])
    pts = pts @ (Rz @ Rx).T

    return pts


class HighDimIntro(ThreeDTIMCSlide):
    def construct(self):
        self.add_centered_text(
            "This works well for 2D data\nWhat about higher dimensions?",
            max_width=0.66,
            font_size=56,
        )

        self.wait()
        self.marked_next_slide()
        self.clear_slide()

        from scipy.spatial.distance import pdist, squareform

        # Angled camera so the tilted cloud reads clearly as 3-D.
        self.set_camera_orientation(phi=65 * DEGREES, theta=-55 * DEGREES)

        # Build the 3-D point cloud and its distance matrix.
        pts = _make_vr_demo_cloud()
        n_pts = len(pts)
        dist_mat = squareform(pdist(pts))
        max_r = np.max(dist_mat) * 1.1

        # Scale to scene coordinates (all three dimensions).
        pts_c = pts - pts.mean(axis=0)
        vis_scale = 3.5 / (np.max(np.abs(pts_c)) + 1e-9)
        pts_vis = pts_c * vis_scale  # shape (n, 3)
        pts_manim = [pts_vis[i] for i in range(n_pts)]

        # ── Points ────────────────────────────────────────────────────────────
        pt_dots = VGroup(
            *[Dot3D(point=p, color=DEFAULT_COLOR, radius=0.07) for p in pts_manim]
        )

        self.play(
            LaggedStart(*[FadeIn(dot, scale=2.0) for dot in pt_dots], lag_ratio=0.05),
            run_time=1.5,
        )

        self.marked_next_slide()

        # ── Spheres (very translucent, low resolution for performance) ────────
        SPHERE_RES = (12, 12)
        INIT_R = 0.001
        spheres = [
            Sphere(radius=INIT_R, resolution=SPHERE_RES)
            .set_color(GRAY)
            .set_opacity(0.10)
            .move_to(p)
            for p in pts_manim
        ]
        spheres_group = VGroup(*spheres)
        # Track the current radius so we can scale incrementally (avoids
        # rebuilding Sphere geometry every frame).
        current_sphere_r = [INIT_R]

        # ── Edges — pre-sorted by birth radius ────────────────────────────────
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

        # ── Triangles — sorted by max-edge birth radius ────────────────────────
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

        # ── 3-Simplices — four triangular faces per tetrahedron, green ─────────
        # Only pre-build tetrahedra whose birth radius falls within the
        # animation range, keeping construction time manageable.
        tet_r_limit = max_r / 3
        tet_data = []  # [(r_threshold, [face0, face1, face2, face3])]
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                for k in range(j + 1, n_pts):
                    for l in range(k + 1, n_pts):
                        r_tet = max(
                            dist_mat[i, j],
                            dist_mat[i, k],
                            dist_mat[i, l],
                            dist_mat[j, k],
                            dist_mat[j, l],
                            dist_mat[k, l],
                        )
                        if r_tet <= tet_r_limit:
                            faces = [
                                Polygon(
                                    pts_manim[a],
                                    pts_manim[b],
                                    pts_manim[c],
                                    fill_color=GREEN_C,
                                    fill_opacity=0,
                                    stroke_width=0,
                                )
                                for a, b, c in [
                                    (i, j, k),
                                    (i, j, l),
                                    (i, k, l),
                                    (j, k, l),
                                ]
                            ]
                            tet_data.append((r_tet, faces))
        tet_data.sort(key=lambda x: x[0])

        all_edges = VGroup(*[e for _, e in edge_data])
        all_tris = VGroup(*[t for _, t in tri_data])
        all_tet_faces = VGroup(*[face for _, faces in tet_data for face in faces])
        self.add(all_tet_faces, all_tris, all_edges, spheres_group)

        r_tracker = ValueTracker(0.0)
        self.add(r_tracker)

        edge_ptr = [0]
        tri_ptr = [0]
        tet_ptr = [0]

        def _reveal_edges(_mob):
            r = r_tracker.get_value()
            while edge_ptr[0] < len(edge_data) and edge_data[edge_ptr[0]][0] <= r:
                edge_data[edge_ptr[0]][1].set_stroke(opacity=0.6)
                edge_ptr[0] += 1

        def _reveal_tris(_mob):
            r = r_tracker.get_value()
            while tri_ptr[0] < len(tri_data) and tri_data[tri_ptr[0]][0] <= r:
                tri_data[tri_ptr[0]][1].set_fill(opacity=0.12)
                tri_ptr[0] += 1

        def _reveal_tets(_mob):
            r = r_tracker.get_value()
            while tet_ptr[0] < len(tet_data) and tet_data[tet_ptr[0]][0] <= r:
                for face in tet_data[tet_ptr[0]][1]:
                    face.set_fill(opacity=0.20)
                tet_ptr[0] += 1

        def _update_spheres(_mob):
            r = r_tracker.get_value()
            target_r = max(INIT_R, r * vis_scale / 2)
            prev_r = current_sphere_r[0]
            if abs(target_r - prev_r) < 1e-9:
                return
            ratio = target_r / prev_r
            current_sphere_r[0] = target_r
            # Scale each sphere about its own centre — no geometry rebuild needed.
            for sph in spheres:
                sph.scale(ratio, about_point=sph.get_center())

        all_edges.add_updater(_reveal_edges)
        all_tris.add_updater(_reveal_tris)
        all_tet_faces.add_updater(_reveal_tets)
        spheres_group.add_updater(_update_spheres)

        # Slow ambient rotation lets the audience appreciate the 3-D structure.
        self.begin_ambient_camera_rotation(rate=0.2)
        self.play(
            r_tracker.animate.set_value(max_r / 4),
            run_time=8.0,
            rate_func=rate_functions.ease_out_quad,
        )
        self.stop_ambient_camera_rotation()

        self.marked_next_slide()

        self.play(FadeOut(spheres_group), run_time=1.0)

        self.marked_next_slide()
        self.begin_ambient_camera_rotation(rate=0.8)
        self.wait(8.0)
        self.stop_ambient_camera_rotation()
        self.marked_next_slide()

        self.clear_slide(run_time=1.0)
        self.set_camera_orientation(phi=0 * DEGREES, theta=-90 * DEGREES)

        self.add_centered_text(
            "We only need distances and volumes to be well behaved",
            font_size=56,
            max_width=0.75,
        )
        self.wait()
        self.marked_next_slide()
        self.clear_slide()

        self.add_centered_text(
            "And is there a need for very many dimensions in data anyway?",
            font_size=56,
        )
        self.wait()
        self.marked_next_slide()
        self.clear_slide()


class SphereShellVolume(ThreeDTIMCSlide):

    def construct(self):

        radius_arrow = Arrow(
            start=(0.0, 0.0, 0.0), end=(0.0, 2.0, 0.0), color=ACCENT_COLOR, buff=0.0
        )
        radius_label = MathTex(r"1", font_size=32, color=ACCENT_COLOR).next_to(
            radius_arrow, LEFT, buff=0.1
        )
        self.play(Create(radius_arrow), Write(radius_label))
        self.wait()

        ball = Circle(radius=2.0, color=ACCENT_COLOR, fill_opacity=1.0)
        self.play(DrawBorderThenFill(ball))

        width_arrow = Arrow(
            start=(0.0, 2.0, 0.0),
            end=(0.0, 2.3, 0.0),
            color=HIGHLIGHT_COLOR,
        )
        epsilon_label = MathTex(
            r"\varepsilon", font_size=32, color=HIGHLIGHT_COLOR
        ).next_to(width_arrow, LEFT, buff=0.1)
        self.play(Create(width_arrow), Write(epsilon_label))
        self.wait()

        shell = Circle(radius=2.3, color=HIGHLIGHT_COLOR, fill_opacity=0.5).set_z_index(
            -1
        )
        self.play(DrawBorderThenFill(shell))

        self.marked_next_slide()

        ratio_text = Text(
            "Volume ratio of shell to ball:",
            t2c={r"ball": ACCENT_COLOR, r"shell": HIGHLIGHT_COLOR},
            font_size=28,
        ).next_to(shell, DOWN, buff=0.25)
        ratio_value = MathTex(r"2\varepsilon + \varepsilon^2", font_size=48).next_to(
            ratio_text, DOWN, buff=0.1
        )
        self.play(Write(ratio_text), Write(ratio_value))
        self.wait()

        self.marked_next_slide()

        self.play(
            FadeOut(radius_arrow, radius_label, width_arrow, epsilon_label, ratio_text),
            ratio_value.animate.next_to(shell, DOWN, buff=0.5),
        )
        two_d_case = VGroup(
            ball,
            shell,
            ratio_value,
        )
        self.play(
            two_d_case.animate.scale(0.5).move_to(4 * LEFT),
        )
        self.marked_next_slide()

        sphere = (
            Sphere(radius=2.0, fill_opacity=1.0, resolution=64)  # resolution=64)
            .rotate_about_origin(90 * DEGREES, axis=RIGHT)
            .set_color(ACCENT_COLOR)
        )

        self.play(
            Create(sphere),
        )

        width_arrow = Arrow(
            start=(0.0, 2.0, 0.0),
            end=(0.0, 2.3, 0.0),
            color=HIGHLIGHT_COLOR,
        )
        epsilon_label = MathTex(
            r"\varepsilon", font_size=32, color=HIGHLIGHT_COLOR
        ).next_to(width_arrow, LEFT, buff=0.1)
        self.play(Create(width_arrow), Write(epsilon_label))
        self.wait()
        spherical_shell = (
            Sphere(
                radius=2.3,
                fill_opacity=0.25,
                stroke_opacity=0.25,
                resolution=64,
                # resolution=128,
            )
            .rotate_about_origin(90 * DEGREES, axis=RIGHT)
            .set_color(HIGHLIGHT_COLOR)
        )

        self.play(
            Create(spherical_shell),
        )

        self.marked_next_slide()

        ratio_text = Text(
            "Volume ratio of shell to ball:",
            t2c={r"ball": ACCENT_COLOR, r"shell": HIGHLIGHT_COLOR},
            font_size=28,
        ).next_to(spherical_shell, DOWN, buff=0.25)
        ratio_value_3d = MathTex(
            r"3\varepsilon + 3\varepsilon^2 + \varepsilon^3", font_size=48
        ).next_to(ratio_text, DOWN, buff=0.1)
        self.play(Write(ratio_text), Write(ratio_value_3d))
        self.wait()

        self.marked_next_slide()

        self.play(
            FadeOut(width_arrow, epsilon_label, ratio_text),
            ratio_value_3d.animate.next_to(spherical_shell, DOWN, buff=0.5),
        )
        three_d_case = VGroup(
            sphere,
            spherical_shell,
            ratio_value_3d,
        )
        self.play(
            two_d_case.animate.scale(0.5).move_to(5.5 * LEFT),
            three_d_case.animate.scale(0.5).move_to(3.5 * LEFT),
        )
        self.wait()

        n_sphere = Tex(
            r"$n$-ball of\\unit radius", color=ACCENT_COLOR, stroke_color=ACCENT_COLOR
        ).shift(UP * 0.75)
        n_shell = Tex(
            r"Shell of width $\varepsilon$\\around $n$-ball",
            color=HIGHLIGHT_COLOR,
            stroke_color=HIGHLIGHT_COLOR,
        ).next_to(n_sphere, DOWN, buff=0.5)

        self.play(LaggedStart(Write(n_sphere), Write(n_shell), lag_ratio=1.0))
        self.wait()
        self.marked_next_slide()
        ratio_text = Text(
            "Volume ratio of shell to ball:",
            t2c={r"ball": ACCENT_COLOR, r"shell": HIGHLIGHT_COLOR},
            font_size=28,
        ).next_to(n_shell, DOWN, buff=0.75)
        ratio_value_3d = MathTex(
            r"n\varepsilon + o(\varepsilon^2)", font_size=48
        ).next_to(ratio_text, DOWN, buff=0.1)
        self.play(LaggedStart(Write(ratio_text), Write(ratio_value_3d), lag_ratio=0.5))
        self.wait()

        self.marked_next_slide()

        self.play(
            FadeOut(ratio_text),
            ratio_value_3d.animate.next_to(n_shell, DOWN, buff=0.75),
        )
        self.wait()

        n_d_case = VGroup(
            n_sphere,
            n_shell,
            ratio_value_3d,
        )
        self.play(
            two_d_case.animate.scale(0.5).move_to(6.5 * LEFT),
            three_d_case.animate.scale(0.5).move_to(5.5 * LEFT),
            n_d_case.animate.scale(0.66).move_to(3.66 * LEFT),
        )

        explainer_text = Tex(
            r"For any $\varepsilon > 0$\\"
            r"there is some\\dimension $n$\\"
            r"such that\\the shell contains\\"
            r"at least\\as much volume\\"
            r"as the ball",
            font_size=48,
            tex_to_color_map={r"shell": HIGHLIGHT_COLOR, r"ball": ACCENT_COLOR},
        )
        frame_box = SurroundingRectangle(explainer_text, color=ACCENT_COLOR, buff=0.5)

        self.play(Write(explainer_text))
        self.play(
            Create(frame_box),
            # Circumscribe(
            #     explainer_text, color=HIGHLIGHT_COLOR, buff=0.75, fade_out=False
            # )
        )
        self.wait()
        self.marked_next_slide()
        # self.clear_slide()


class ShadedNNDistribution(TIMCSlide):

    def construct(self):
        # --- Parameters ---
        num_points = 5000
        dimensions = np.round(np.linspace(3, 64, 63)).astype(np.int32)
        x_range = [0, 3.0, 0.5]

        axes = Axes(
            x_range=x_range, y_range=[0, 20, 2], axis_config={"include_tip": False}
        ).add_coordinates()

        self.play(Create(axes), run_time=1.5)
        self.marked_next_slide()

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

        self.play(Create(label), Create(dim_tracker), Create(current_mobjects))
        self.wait()

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
        if i == 0:
            self.marked_next_slide()


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


import cv2


# --- Custom Video Mobject Implementation ---
class VideoMobject(ImageMobject):
    def __init__(self, filename, **kwargs):
        self.filename = filename

        # 1. Open temporarily to grab dimensions
        cap = cv2.VideoCapture(filename)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps

        # Extract metadata dimensions
        self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Could not read video file: {filename}")

        # CRITICAL FIX 1: Convert first frame to RGBA (4 channels) to initialize the parent correctly
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        super().__init__(frame, **kwargs)

        self.cap = None
        self.current_time = 0.0
        self.current_frame_idx = 0
        self.prev_frame_no = -1

        # Trigger updates on every timeline tick
        self.add_updater(lambda m, dt: m.update_frame(dt))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["cap"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.cap = None

    def __deepcopy__(self, memo):
        import copy

        # Create a clean, uninitialized instance of VideoMobject
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # Copy all properties over, but explicitly leave 'cap' out of it
        for k, v in self.__dict__.items():
            if k == "cap":
                result.cap = None  # The copy will instantiate its own fresh stream when it renders
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def update_frame(self, dt):
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.filename)
            self.current_frame_idx = 0
            self.prev_frame_no = -1

        self.current_time += dt
        frame_no = int(self.current_time * self.fps) % self.frame_count

        # PERFORMANCE FIX: Skip frame processing entirely if the timeline tick
        # hasn't shifted into a brand new video frame yet.
        if frame_no == self.prev_frame_no:
            return

        # PERFORMANCE FIX: Avoid using the expensive set() operation unless
        # the video loops or jumps out of sequential reading order.
        if frame_no != self.current_frame_idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

        ret, frame = self.cap.read()

        if ret:
            self.current_frame_idx = frame_no + 1
            self.prev_frame_no = frame_no

            # CRITICAL FIX 2: Convert streaming frames to RGBA to match Manim's 4-channel matrix specs
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

            # Ensure the array structure precisely matches original specs
            if (
                frame.shape[1] != self.video_width
                or frame.shape[0] != self.video_height
            ):
                frame = cv2.resize(frame, (self.video_width, self.video_height))

            # Safely replace pixel matrix texture data
            if hasattr(self, "set_pixel_array"):
                self.set_pixel_array(frame)
            else:
                self.pixel_array = frame


class HighDExampleUseCases(TIMCSlide):
    def construct(self):

        self.add_centered_text(
            "Activation spaces of deep Neural Networks",
            max_width=0.66,
        )
        self.wait()
        self.marked_next_slide()

        self.clear_slide()

        neural_video = VideoMobject("Neural network geometry.mp4")
        neural_video.scale_to_fit_height(config.frame_height * 0.8)
        # self.play(FadeIn(neural_video))
        self.add(neural_video)
        self.wait(neural_video.duration * 3)
        self.marked_next_slide()

        self.play(FadeOut(neural_video))
        self.add_centered_text(
            "Biology, especially single-cell genomics",
            max_width=0.75,
        )
        self.wait()
        self.marked_next_slide()
        self.clear_slide()
        sc_video = VideoMobject("zebrafish_scrna.mp4")
        sc_video.scale_to_fit_height(config.frame_height * 0.66)
        # self.play(FadeIn(sc_video))
        self.add(sc_video)
        self.wait(sc_video.duration * 2)
        self.marked_next_slide()

        self.play(FadeOut(sc_video))
        self.wait()
