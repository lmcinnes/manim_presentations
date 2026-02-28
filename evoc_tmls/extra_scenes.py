"""Extra scene classes not used in the main EVoC presentation but preserved for reuse."""

from manim import *
from manim_slides import Slide

import sys

sys.path.append("..")

from config import (
    apply_defaults,
    COLOR_CYCLE,
    DEFAULT_COLOR,
    ACCENT_COLOR,
    HIGHLIGHT_COLOR,
    BACKGROUND_COLOR,
    add_logo_to_background,
    TIMCSlide,
    ThreeDTIMCSlide,
)

import numpy as np
import sklearn.preprocessing

from data_manifest import (
    BASE_DATA as BASE_DATA_PATH,
    DATA_COLORMAP as DATA_COLORMAP_PATH,
)

base_data = (np.load(BASE_DATA_PATH) + 0.5) * 10
data_colormap = np.load(DATA_COLORMAP_PATH)

# Compute fly_away: points radiate outward from center
fly_away = base_data - (5, 5)
sklearn.preprocessing.normalize(fly_away, copy=False)
fly_away *= 30

apply_defaults()


class DistortionLenses(TIMCSlide):

    def _move_points(self, new_locations, run_time=4.0):
        animations = []
        for i, dot in enumerate(self.dots):
            new_point = self.graph.coords_to_point(*new_locations[i].tolist())
            animations.append(dot.animate.move_to(new_point))

        self.play(*animations, run_time=run_time)

    def construct(self):
        add_logo_to_background(self)

        self.graph = Axes(
            x_range=[-2, 12, 1],
            y_range=[-2, 12, 1],
            x_length=9,
            y_length=6,
            axis_config={"include_tip": True, "color": DEFAULT_COLOR},
        )
        labels = self.graph.get_axis_labels()
        # self.add(graph, labels)

        x_data, y_data = base_data.T

        # Create dots for each point
        self.dots = VGroup()
        for x, y in zip(x_data, y_data):
            # Convert data coordinates to screen coordinates
            point = self.graph.coords_to_point(x, y)
            dot = Dot(
                point=point,
                radius=0.025,  # Adjust size
                color=ACCENT_COLOR,
                stroke_width=0.5,
                stroke_color=BACKGROUND_COLOR,
            )
            self.dots.add(dot)

        # self.add(graph, labels)
        init_text = Text("Suppose we have some data...")
        self.play(Write(init_text))
        self.play(
            FadeOut(init_text, run_time=1.0),
            LaggedStart(
                *[FadeIn(dot, scale=0.25) for dot in self.dots], lag_ratio=0.001
            ),
        )

        self.marked_next_slide()

        text1 = self.add_title_text(
            "Add colour so we can track points...", font_size=32
        )

        animations = []
        for i, dot in enumerate(self.dots):
            animations.append(dot.animate.set_fill_color(data_colormap[i]))

        self.play(LaggedStart(*animations, lag_ratio=0.001), run_time=1.0)
        self.play(FadeOut(text1))

        self.marked_next_slide()

        # TODO: gmm_distortion and kmeans_distortion data not yet generated
        # text2 = self.add_title_text("As viewed through the lens of a GMM", font_size=40)
        # self._move_points(gmm_distortion)
        # self.play(FadeOut(text2))
        #
        # self.marked_next_slide()
        # self._move_points(base_data)
        #
        # self.marked_next_slide()
        #
        # text3 = self.add_title_text(
        #     "As viewed through the lens of K-Means", font_size=40
        # )
        # self._move_points(kmeans_distortion)
        # self.play(FadeOut(text3))
        #
        # self.marked_next_slide()
        # self._move_points(base_data)

        self.marked_next_slide()
        self._move_points(fly_away, run_time=1.0)

        self.add_centered_text(
            "All clustering algorithms distort data based on their internal view of the data!"
        )


class OpenClusteringBox(ThreeDTIMCSlide):

    def construct(self):
        add_logo_to_background(self)
        self.new_section("Data Distortion")
        self.set_camera_orientation(phi=75 * DEGREES, theta=70 * DEGREES)

        # 2D labels that stay fixed
        title = Text("Opening up a clustering algorithm", font_size=40)
        # title.to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)

        # Animate
        self.play(Write(title))
        self.marked_next_slide()

        self.play(title.animate.move_to(UP * 3))

        # 3D cube
        cube = Cube(side_length=2, fill_opacity=1, stroke_width=2)
        cube.set_fill(DARK_GRAY)
        cube.set_stroke(WHITE)

        # Create label
        label = Paragraph(
            "Clustering\nAlgorithm",
            font_size=32,
            color=WHITE,
            stroke_color="white",
            alignment="center",
        )

        # Position on the front face and rotate to align with face
        label.rotate(PI / 2, axis=RIGHT)  # Rotate to be upright in 3D
        label.rotate(PI, axis=OUT)  # Face forward
        label.move_to(cube.get_center() + UP)  # Position in front

        input_arrow = Arrow(
            start=LEFT * (5 - 7 / 8),
            end=LEFT * (1.5 + 7 / 8),
            color=ACCENT_COLOR,
            stroke_width=30,
            max_tip_length_to_length_ratio=25.0,
            max_stroke_width_to_length_ratio=10,
            buff=0,
        ).scale(2, scale_tips=True)
        input_label = Text("Input", font_size=32)
        input_label.next_to(input_arrow, UP)
        self.add_fixed_in_frame_mobjects(input_arrow, input_label)
        self.play(GrowArrow(input_arrow), Write(input_label))

        self.play(GrowFromCenter(cube))
        self.play(Write(label))

        output_arrow = Arrow(
            start=RIGHT * (1.5 + 7 / 8),
            end=RIGHT * (5 - 7 / 8),
            color=ACCENT_COLOR,
            stroke_width=30,
            max_tip_length_to_length_ratio=15,
            max_stroke_width_to_length_ratio=10.0,
            buff=0,
        ).scale(2, scale_tips=True)
        output_label = Text("Output", font_size=32)
        output_label.next_to(output_arrow, UP)
        self.add_fixed_in_frame_mobjects(output_arrow, output_label)
        self.play(GrowArrow(output_arrow), Write(output_label))
        self.marked_next_slide()

        left_half = Prism(dimensions=[1, 2, 2])
        left_half.set_fill(DARK_GRAY, opacity=1)
        left_half.set_stroke(WHITE, width=2)
        left_half.move_to(LEFT * 0.5)

        right_half = Prism(dimensions=[1, 2, 2])
        right_half.set_fill(DARK_GRAY, opacity=1)
        right_half.set_stroke(WHITE, width=2)
        right_half.move_to(RIGHT * 0.5)

        self.play(FadeOut(label))
        self.remove(cube)
        self.add(left_half, right_half)

        # New arrows
        input_arrow_short = Arrow(
            start=LEFT * (5 - 5 / 8),
            end=LEFT * (2.5 + 5 / 8),
            color=ACCENT_COLOR,
            stroke_width=30,
            max_tip_length_to_length_ratio=25.0,
            max_stroke_width_to_length_ratio=10,
            buff=0,
        ).scale(2, scale_tips=True)
        input_label_short = Text("Input", font_size=32)
        input_label_short.next_to(input_arrow_short, UP).shift(LEFT * 0.25)
        output_arrow_short = Arrow(
            start=RIGHT * (2.5 + 5 / 8),
            end=RIGHT * (5 - 5 / 8),
            color=ACCENT_COLOR,
            stroke_width=30,
            max_tip_length_to_length_ratio=15,
            max_stroke_width_to_length_ratio=10.0,
            buff=0,
        ).scale(2, scale_tips=True)
        output_label_short = Text("Output", font_size=32)
        output_label_short.next_to(output_arrow, UP).shift(RIGHT * 0.25)
        self.play(
            left_half.animate.shift(LEFT * 0.8),
            right_half.animate.shift(RIGHT * 0.8),
            Transform(input_arrow, input_arrow_short),
            Transform(input_label, input_label_short),
            Transform(output_arrow, output_arrow_short),
            Transform(output_label, output_label_short),
            run_time=2,
        )

        # Add curved annotation arrow pointing to the gap
        annotation_text = Paragraph(
            "Visualize a canonical example\nof the internal representation",
            font_size=32,
            alignment="center",
            color=COLOR_CYCLE[3],
            stroke_color=COLOR_CYCLE[3],
        )
        annotation_text.to_edge(DOWN).shift(RIGHT * 1.5)
        curve_points = [
            annotation_text.get_top() + UP * 0.2,
            annotation_text.get_top() + UP * 1.5 + LEFT * 0.5,
            DOWN + LEFT * 0.1,
            ORIGIN,
        ]

        # Use VMobject to create custom curve
        curved_path = VMobject(color=COLOR_CYCLE[3], stroke_width=8)
        curved_path.set_points_smoothly(curve_points)

        # Add tip
        tip = StealthTip(color=COLOR_CYCLE[3]).scale(2)
        tip.move_to(curve_points[-1])
        # Orient the tip
        direction = curve_points[-1] - curve_points[-2]
        angle = np.arctan2(direction[1], direction[0]) - 18 * DEGREES
        tip.rotate(angle)

        annotation = VGroup(curved_path, tip)

        self.add_fixed_in_frame_mobjects(annotation)
        self.play(Create(curved_path), GrowFromCenter(tip))
        self.add_fixed_in_frame_mobjects(annotation_text)
        self.play(Write(annotation_text))
        self.marked_next_slide()
