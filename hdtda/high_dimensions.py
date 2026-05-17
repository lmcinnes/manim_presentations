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
            Sphere(radius=2.0, fill_opacity=1.0, resolution=8)  # resolution=64)
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
                resolution=8,
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
        self.clear_slide()


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
