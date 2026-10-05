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
from scipy.special import betaincinv, gammaln
from scipy.stats import norm

apply_defaults()


def rotation_matrix_to_axis_angle(R):
    angle = np.arccos((np.trace(R) - 1) / 2)
    if np.isclose(angle, 0):
        return np.array([1, 0, 0]), 0

    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (
        2 * np.sin(angle)
    )

    return axis, angle


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


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------
def ball_to_box_ratio(n):
    """Volume of the n-ball of diameter 1 over the unit n-cube:
    pi^(n/2) / (2^n Gamma(n/2 + 1))."""
    return float(np.exp(n / 2 * np.log(np.pi) - gammaln(n / 2 + 1) - n * np.log(2)))


def band_half_height(n, fraction):
    """Half-height h such that the band |x_n| <= h holds `fraction` of the
    surface of the unit sphere in R^n. For a uniform point on that sphere,
    x_n^2 ~ Beta(1/2, (n-1)/2)."""
    return float(np.sqrt(betaincinv(0.5, (n - 1) / 2, fraction)))


def sci_tex(x, digits=2):
    """0.785 -> '0.785', 0.00249 -> '0.0025', 2.46e-08 -> '2.5 x 10^{-8}' (TeX)."""
    if x >= 0.01:
        return f"{x:.3f}"
    if x >= 1e-3:
        return f"{x:.4f}"
    m, e = f"{x:.{digits - 1}e}".split("e")
    return rf"{m} \times 10^{{{int(e)}}}"


def value_table(header, rows, font_size=30):
    """Small right-aligned table of MathTex cells, header underlined."""
    cells = [[MathTex(h, font_size=font_size, color=ACCENT_COLOR) for h in header]]
    cells += [[MathTex(c, font_size=font_size) for c in row] for row in rows]
    n_cols = len(header)
    col_w = [max(r[j].width for r in cells) for j in range(n_cols)]
    row_h = max(c.height for r in cells for c in r) + 0.18
    grid = VGroup()
    for i, r in enumerate(cells):
        x = 0.0
        for j, c in enumerate(r):
            c.move_to([x + col_w[j], -i * row_h, 0], aligned_edge=RIGHT)
            x += col_w[j] + 0.5
            grid.add(c)
    rule = Line(
        grid.get_corner(UL), grid.get_corner(UR), color=ACCENT_COLOR, stroke_width=2
    )
    rule.shift(DOWN * (cells[0][0].height + 0.12))
    return VGroup(grid, rule)


# ---------------------------------------------------------------------------
class BallInBoxVolume(ThreeDTIMCSlide):
    """The ball inscribed in a unit box holds a vanishing share of its
    volume as the dimension grows: nearly everything is in the corners."""

    SIDE = 4.0  # display size of the unit box (ball radius = SIDE / 2)

    def construct(self):
        S, R = self.SIDE, self.SIDE / 2

        # ---- 2D: disk in a square ----
        square = Square(
            side_length=S, color=HIGHLIGHT_COLOR, fill_opacity=0.5
        ).set_z_index(-1)
        side_brace = Brace(square, DOWN, color=DEFAULT_COLOR)
        side_label = MathTex(r"1", font_size=32).next_to(side_brace, DOWN, buff=0.1)
        self.play(DrawBorderThenFill(square))
        self.play(GrowFromCenter(side_brace), Write(side_label))
        self.wait()

        disk = Circle(radius=R, color=ACCENT_COLOR, fill_opacity=1.0)
        self.play(DrawBorderThenFill(disk))

        radius_arrow = Arrow(
            ORIGIN, R * RIGHT, color=BACKGROUND_COLOR, buff=0.0
        ).set_z_index(2)
        radius_label = (
            MathTex(r"\tfrac{1}{2}", font_size=32, color=BACKGROUND_COLOR)
            .next_to(radius_arrow, UP, buff=0.05)
            .set_z_index(2)
        )
        corner_arrow = Arrow(
            ORIGIN, R * (LEFT + UP), color=DEFAULT_COLOR, buff=0.0
        ).set_z_index(2)
        corner_label = (
            MathTex(r"\tfrac{\sqrt{2}}{2} \approx 0.71", font_size=32)
            .next_to(corner_arrow.get_end(), UP, buff=0.15)
            .set_z_index(2)
        )
        self.play(Create(radius_arrow), Write(radius_label))
        self.play(Create(corner_arrow), Write(corner_label))
        self.wait()

        self.marked_next_slide()

        ratio_text = Text(
            "Volume ratio of ball to box:",
            t2c={r"ball": ACCENT_COLOR, r"box": HIGHLIGHT_COLOR},
            font_size=28,
        ).next_to(side_label, DOWN, buff=0.2)
        ratio_value = MathTex(
            rf"\tfrac{{\pi}}{{4}} \approx {ball_to_box_ratio(2):.3f}", font_size=48
        ).next_to(ratio_text, DOWN, buff=0.1)
        self.play(Write(ratio_text), Write(ratio_value))
        self.wait()

        self.marked_next_slide()

        self.play(
            FadeOut(
                radius_arrow,
                radius_label,
                corner_arrow,
                corner_label,
                side_brace,
                side_label,
                ratio_text,
            ),
            ratio_value.animate.next_to(square, DOWN, buff=0.5),
        )
        two_d_case = VGroup(square, disk, ratio_value)
        self.play(two_d_case.animate.scale(0.5).move_to(5.2 * LEFT))
        self.marked_next_slide()

        # ---- 3D: sphere in a cube (tilted so the cube reads as 3D) ----
        tilt = rotation_matrix(-35 * DEGREES, UP) @ rotation_matrix(25 * DEGREES, RIGHT)
        cube = Cube(
            side_length=S,
            fill_opacity=0.25,
            fill_color=HIGHLIGHT_COLOR,
            stroke_color=HIGHLIGHT_COLOR,
            stroke_width=1.5,
        )
        cube.apply_matrix(tilt)
        # The 3D camera depth-sorts faces by their centre, which makes the
        # cube's front faces cut patchily through the sphere. Take the
        # front faces (facing the camera, +z) out of the depth sort so they
        # are always drawn translucently over the sphere.
        for face in cube:
            if face.get_center()[2] > 0:
                face.shade_in_3d = False
                face.set_fill(opacity=0.1)
        sphere = (
            Sphere(radius=R, fill_opacity=1.0, resolution=48)
            .rotate_about_origin(90 * DEGREES, axis=RIGHT)
            .set_color(ACCENT_COLOR)
        )

        self.play(Create(cube))
        self.play(Create(sphere))

        corner3 = tilt @ np.array([-R, R, R])
        corner_arrow = Arrow(
            ORIGIN, corner3, color=DEFAULT_COLOR, buff=0.0
        ).set_z_index(2)
        corner_label = MathTex(
            r"\tfrac{\sqrt{3}}{2} \approx 0.87", font_size=32
        ).next_to(corner_arrow.get_end(), UP, buff=0.15)
        self.add_fixed_orientation_mobjects(corner_label)
        self.play(Create(corner_arrow), Write(corner_label))
        self.wait()

        self.marked_next_slide()

        ratio_text = Text(
            "Volume ratio of ball to box:",
            t2c={r"ball": ACCENT_COLOR, r"box": HIGHLIGHT_COLOR},
            font_size=28,
        ).next_to(cube, DOWN, buff=0.25)
        ratio_value_3d = MathTex(
            rf"\tfrac{{\pi}}{{6}} \approx {ball_to_box_ratio(3):.3f}", font_size=48
        ).next_to(ratio_text, DOWN, buff=0.1)
        self.play(Write(ratio_text), Write(ratio_value_3d))
        self.wait()

        self.marked_next_slide()

        self.play(
            FadeOut(corner_arrow, corner_label, ratio_text),
            ratio_value_3d.animate.next_to(cube, DOWN, buff=0.5),
        )
        three_d_case = VGroup(cube, sphere, ratio_value_3d)
        self.play(
            two_d_case.animate.scale(0.5).move_to(5.5 * LEFT),
            three_d_case.animate.scale(0.5).move_to(3.5 * LEFT),
        )
        self.wait()

        # ---- n dimensions ----
        n_ball = Tex(
            r"$n$-ball of\\diameter 1", color=ACCENT_COLOR, stroke_color=ACCENT_COLOR
        ).shift(UP * 0.75)
        n_box = Tex(
            r"$n$-cube of\\side length 1",
            color=HIGHLIGHT_COLOR,
            stroke_color=HIGHLIGHT_COLOR,
        ).next_to(n_ball, DOWN, buff=0.5)
        self.play(LaggedStart(Write(n_ball), Write(n_box), lag_ratio=1.0))
        self.wait()
        self.marked_next_slide()

        ratio_text = Text(
            "Volume ratio of ball to box:",
            t2c={r"ball": ACCENT_COLOR, r"box": HIGHLIGHT_COLOR},
            font_size=28,
        ).next_to(n_box, DOWN, buff=0.75)
        ratio_value_nd = MathTex(
            r"\frac{\pi^{n/2}}{2^n\,\Gamma\!\left(\frac{n}{2}+1\right)}", font_size=44
        ).next_to(ratio_text, DOWN, buff=0.1)
        self.play(LaggedStart(Write(ratio_text), Write(ratio_value_nd), lag_ratio=0.5))
        self.wait()

        self.marked_next_slide()

        self.play(
            FadeOut(ratio_text),
            ratio_value_nd.animate.next_to(n_box, DOWN, buff=0.75),
        )
        n_d_case = VGroup(n_ball, n_box, ratio_value_nd)
        self.play(
            two_d_case.animate.scale(0.5).move_to(6.5 * LEFT),
            three_d_case.animate.scale(0.5).move_to(5.5 * LEFT),
            n_d_case.animate.scale(0.66).move_to(3.66 * LEFT),
        )

        dims = [2, 3, 4, 5, 10, 20, 100]
        table = (
            value_table(
                [r"n", r"\text{ball / box}", r"\text{centre to corner}"],
                [
                    [str(n), sci_tex(ball_to_box_ratio(n)), f"{np.sqrt(n) / 2:.2f}"]
                    for n in dims
                ],
                font_size=26,
            )
            .to_edge(RIGHT, buff=0.35)
            .set_y(0)
        )
        self.play(
            FadeIn(table[1]),
            LaggedStart(*[FadeIn(c) for c in table[0]], lag_ratio=0.05),
        )
        self.wait()
        self.marked_next_slide()

        explainer_text = Tex(
            r"As $n$ grows\\"
            r"the ball fills\\"
            r"a vanishing\\fraction of the box:\\"
            r"almost all\\the volume\\"
            r"is in the corners",
            font_size=40,
            tex_to_color_map={
                r"ball": ACCENT_COLOR,
                r"box": HIGHLIGHT_COLOR,
                r"corners": HIGHLIGHT_COLOR,
            },
        ).move_to(RIGHT * 0.05)
        frame_box = SurroundingRectangle(explainer_text, color=ACCENT_COLOR, buff=0.35)
        self.play(Write(explainer_text))
        self.play(Create(frame_box))
        self.wait()
        self.marked_next_slide()


# ---------------------------------------------------------------------------
def sphere_patch(radius, lat_lo, lat_hi, color, opacity, resolution=(48, 8)):
    """Part of a sphere (poles on z) between two latitudes."""
    return Surface(
        lambda u, v: radius
        * np.array([np.cos(v) * np.cos(u), np.cos(v) * np.sin(u), np.sin(v)]),
        u_range=[0, TAU],
        v_range=[lat_lo, lat_hi],
        resolution=resolution,
        checkerboard_colors=False,
        fill_color=color,
        fill_opacity=opacity,
        stroke_color=color,
        stroke_width=0.4,
        stroke_opacity=opacity,
    )


def latitude_circle(radius, lat, color, stroke_width=2):
    return ParametricFunction(
        lambda u: radius
        * np.array([np.cos(lat) * np.cos(u), np.cos(lat) * np.sin(u), np.sin(lat)]),
        t_range=[0, TAU],
        color=color,
        stroke_width=stroke_width,
    )


class EquatorBandArea(ThreeDTIMCSlide):
    """How tall must a band around the equator be to hold most of the
    sphere's surface? Tall in low dimensions, vanishingly thin in high
    dimensions -- and it works for any equator."""

    FRACTION = 0.90  # 0.99 also works, but then the 2D and 3D bands cover
    # essentially the whole circle / sphere (h = 0.9999,
    # 0.99), so the low-dimensional pictures show nothing
    RADIUS = 2.0
    TILT = -72 * DEGREES  # tip the sphere's pole axis towards the viewer
    SCHEMATIC_H = 0.14  # band half-height (radius units) for part two

    def construct(self):
        R, p = self.RADIUS, self.FRACTION
        pct = f"{round(100 * p)}\\%"
        pct_text = f"{round(100 * p)}%"

        # ---- 2D: circle and a horizontal band ----
        h2 = band_half_height(2, p)
        a2 = np.arcsin(h2)
        circle = Circle(radius=R, color=ACCENT_COLOR, stroke_width=6)
        equator = DashedLine(
            LEFT * (R + 0.5), RIGHT * (R + 0.5), color=DEFAULT_COLOR, stroke_width=2
        )
        eq_label = Text("equator", font_size=22, color=DEFAULT_COLOR).next_to(
            equator, LEFT, buff=0.1
        )
        self.play(Create(circle))
        self.play(Create(equator), Write(eq_label))
        self.wait()

        bounds = VGroup(
            *[
                DashedLine(
                    [-R - 0.5, s * h2 * R, 0],
                    [R + 0.5, s * h2 * R, 0],
                    color=HIGHLIGHT_COLOR,
                    stroke_width=2,
                )
                for s in (1, -1)
            ]
        )
        band_arcs = VGroup(
            Arc(radius=R, start_angle=-a2, angle=2 * a2),
            Arc(radius=R, start_angle=PI - a2, angle=2 * a2),
        ).set_stroke(HIGHLIGHT_COLOR, width=9)
        height_arrow = DoubleArrow(
            [R + 0.8, -h2 * R, 0],
            [R + 0.8, h2 * R, 0],
            color=HIGHLIGHT_COLOR,
            buff=0.0,
            stroke_width=3,
            tip_length=0.2,
        )
        height_label = MathTex(r"2h", font_size=32, color=HIGHLIGHT_COLOR).next_to(
            height_arrow, RIGHT, buff=0.1
        )
        self.play(Create(bounds), Create(band_arcs))
        self.play(GrowFromCenter(height_arrow), Write(height_label))
        self.wait()

        self.marked_next_slide()

        ratio_text = Text(
            f"Band holding {pct_text} of the circumference:",
            t2c={"Band": HIGHLIGHT_COLOR, "circumference": ACCENT_COLOR},
            font_size=28,
        ).next_to(circle, DOWN, buff=0.35)
        h_value = MathTex(rf"h = {h2:.3f}", font_size=48).next_to(
            ratio_text, DOWN, buff=0.1
        )
        self.play(Write(ratio_text), Write(h_value))
        self.wait()

        self.marked_next_slide()

        self.play(
            FadeOut(equator, eq_label, height_arrow, height_label, ratio_text),
            h_value.animate.next_to(circle, DOWN, buff=0.5),
        )
        two_d_case = VGroup(circle, band_arcs, bounds, h_value)
        self.play(two_d_case.animate.scale(0.5).move_to(4 * LEFT))
        self.marked_next_slide()

        # ---- 3D: sphere and an equatorial belt ----
        h3 = band_half_height(3, p)
        a3 = np.arcsin(h3)
        caps = VGroup(
            sphere_patch(R, a3, PI / 2, ACCENT_COLOR, 0.35),
            sphere_patch(R, -PI / 2, -a3, ACCENT_COLOR, 0.35),
        )
        band = sphere_patch(R, -a3, a3, HIGHLIGHT_COLOR, 0.85, resolution=(48, 12))
        edges = VGroup(
            latitude_circle(R, a3, HIGHLIGHT_COLOR),
            latitude_circle(R, -a3, HIGHLIGHT_COLOR),
        )
        eq3 = DashedVMobject(latitude_circle(R, 0, DEFAULT_COLOR, 2), num_dashes=40)
        sphere3 = VGroup(caps, band, edges, eq3).rotate(
            self.TILT, axis=RIGHT, about_point=ORIGIN
        )
        whole = sphere_patch(
            R, -PI / 2, PI / 2, ACCENT_COLOR, 0.85, resolution=(48, 24)
        ).rotate(self.TILT, axis=RIGHT, about_point=ORIGIN)

        self.play(Create(whole))
        self.play(Create(eq3))
        self.wait()
        self.play(FadeOut(whole), FadeIn(caps), FadeIn(band), Create(edges))

        ys = [
            (rotation_matrix(self.TILT, RIGHT) @ np.array([0, 0, s * h3 * R]))[1]
            for s in (1, -1)
        ]
        height_arrow = DoubleArrow(
            [R + 0.8, ys[1], 0],
            [R + 0.8, ys[0], 0],
            color=HIGHLIGHT_COLOR,
            buff=0.0,
            stroke_width=3,
            tip_length=0.2,
        )
        height_label = MathTex(r"2h", font_size=32, color=HIGHLIGHT_COLOR).next_to(
            height_arrow, RIGHT, buff=0.1
        )
        self.play(GrowFromCenter(height_arrow), Write(height_label))
        self.wait()

        self.marked_next_slide()

        ratio_text = Text(
            f"Band holding {pct_text} of the surface area:",
            t2c={"Band": HIGHLIGHT_COLOR, "surface area": ACCENT_COLOR},
            font_size=28,
        ).next_to(sphere3, DOWN, buff=0.25)
        h_value_3d = MathTex(rf"h = {h3:.3f}", font_size=48).next_to(
            ratio_text, DOWN, buff=0.1
        )
        self.play(Write(ratio_text), Write(h_value_3d))
        self.wait()

        self.marked_next_slide()

        self.play(
            FadeOut(height_arrow, height_label, ratio_text),
            h_value_3d.animate.next_to(sphere3, DOWN, buff=0.5),
        )
        three_d_case = VGroup(sphere3, h_value_3d)
        self.play(
            two_d_case.animate.scale(0.5).move_to(5.5 * LEFT),
            three_d_case.animate.scale(0.5).move_to(3.5 * LEFT),
        )
        self.wait()

        # ---- n dimensions ----
        n_sphere = Tex(
            r"Unit sphere\\in $n$ dimensions",
            color=ACCENT_COLOR,
            stroke_color=ACCENT_COLOR,
        ).shift(UP * 0.75)
        n_band = Tex(
            r"Band $|x_n| \le h$\\around the equator",
            color=HIGHLIGHT_COLOR,
            stroke_color=HIGHLIGHT_COLOR,
        ).next_to(n_sphere, DOWN, buff=0.5)
        self.play(LaggedStart(Write(n_sphere), Write(n_band), lag_ratio=1.0))
        self.wait()
        self.marked_next_slide()

        z = norm.ppf((1 + p) / 2)
        ratio_text = Text(
            f"Band holding {pct_text} of the surface area:",
            t2c={"Band": HIGHLIGHT_COLOR, "surface area": ACCENT_COLOR},
            font_size=28,
        ).next_to(n_band, DOWN, buff=0.75)
        h_value_nd = MathTex(
            rf"h \approx \frac{{{z:.3f}}}{{\sqrt{{n}}}}", font_size=48
        ).next_to(ratio_text, DOWN, buff=0.1)
        self.play(LaggedStart(Write(ratio_text), Write(h_value_nd), lag_ratio=0.5))
        self.wait()

        self.marked_next_slide()

        self.play(
            FadeOut(ratio_text),
            h_value_nd.animate.next_to(n_band, DOWN, buff=0.75),
        )
        n_d_case = VGroup(n_sphere, n_band, h_value_nd)
        self.play(
            two_d_case.animate.scale(0.5).move_to(6.5 * LEFT),
            three_d_case.animate.scale(0.5).move_to(5.5 * LEFT),
            n_d_case.animate.scale(0.66).move_to(3.3 * LEFT),
        )

        dims = [2, 3, 5, 10, 100, 1000, 10000]
        table = (
            value_table(
                [r"n", r"h \text{ for " + pct + r"}"],
                [
                    [f"{n:,}".replace(",", r"{,}"), f"{band_half_height(n, p):.3f}"]
                    for n in dims
                ],
                font_size=30,
            )
            .move_to(RIGHT * 4.8)
            .set_y(0)
        )
        self.play(
            FadeIn(table[1]),
            LaggedStart(*[FadeIn(c) for c in table[0]], lag_ratio=0.05),
        )
        self.wait()
        self.marked_next_slide()

        explainer_text = Tex(
            r"For any $h > 0$\\"
            r"there is some\\dimension $n$\\"
            r"such that\\a band of height $h$\\"
            r"around the equator\\"
            rf"holds {pct}\\of the surface",
            font_size=40,
            tex_to_color_map={r"band": HIGHLIGHT_COLOR, r"surface": ACCENT_COLOR},
        ).move_to(RIGHT * 0.6)
        frame_box = SurroundingRectangle(explainer_text, color=ACCENT_COLOR, buff=0.35)
        self.play(Write(explainer_text))
        self.play(Create(frame_box))
        self.wait()
        self.marked_next_slide()

        # ---- part two: which equator? ----
        self.play(
            FadeOut(
                two_d_case, three_d_case, n_d_case, table, explainer_text, frame_box
            )
        )

        title = Text("Which equator?", font_size=40).to_edge(UP, buff=0.4)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))

        Rb, hs = 2.3, self.SCHEMATIC_H
        ab = np.arcsin(hs)
        tilt = rotation_matrix(self.TILT, RIGHT)
        ball = sphere_patch(
            Rb, -PI / 2, PI / 2, ACCENT_COLOR, 0.3, resolution=(48, 24)
        ).rotate(self.TILT, axis=RIGHT, about_point=ORIGIN)

        def make_band(color):
            return VGroup(
                sphere_patch(Rb, -ab, ab, color, 0.9, resolution=(64, 2)),
                latitude_circle(Rb, ab, color, 1.5),
                latitude_circle(Rb, -ab, color, 1.5),
            ).rotate(self.TILT, axis=RIGHT, about_point=ORIGIN)

        band_s = make_band(HIGHLIGHT_COLOR)
        note = Text(
            "schematic: a thin band, as in high dimensions",
            font_size=20,
            color=ACCENT_COLOR,
        ).to_edge(DOWN, buff=0.9)
        self.add_fixed_in_frame_mobjects(note)
        self.play(Create(ball), FadeIn(note))
        self.play(FadeIn(band_s))
        self.wait()
        self.marked_next_slide()

        point = Text(
            "We never said which equator:\n"
            f"the same band about any equator holds {pct_text}",
            font_size=26,
            line_spacing=0.8,
            t2c={"any": HIGHLIGHT_COLOR},
        ).to_edge(DOWN, buff=0.35)
        self.add_fixed_in_frame_mobjects(point)
        self.play(FadeOut(note), Write(point))

        # rotate the band to a few arbitrary equators, leaving copies behind
        rng = np.random.default_rng(3)
        colors = [COLOR_CYCLE[i] for i in (0, 2, 3, 5, 4)]
        ghosts = VGroup()
        for k, col in enumerate(colors):
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            angle = rng.uniform(0.5, 1.3) * (1 if k % 2 else -1)
            ghost = band_s.copy()
            self.add(ghost)
            ghosts.add(ghost)
            self.play(
                Rotate(band_s, angle=angle, axis=axis, about_point=ORIGIN),
                ghost[0]
                .animate.set_fill(col, opacity=0.35)
                .set_stroke(col, opacity=0.35),
                ghost[1:].animate.set_stroke(col, opacity=0.6),
                run_time=1.6,
            )
        self.wait()
        self.marked_next_slide()

        conclusion = Text(
            "So in high dimensions almost all of the surface\n"
            "lies close to every equator at once:\n"
            "two random directions are almost always nearly orthogonal",
            font_size=24,
            line_spacing=0.8,
            t2c={"every": HIGHLIGHT_COLOR, "nearly orthogonal": HIGHLIGHT_COLOR},
        ).to_edge(DOWN, buff=0.3)
        self.add_fixed_in_frame_mobjects(conclusion)
        self.play(FadeOut(point), Write(conclusion))
        self.wait()
        self.marked_next_slide()
