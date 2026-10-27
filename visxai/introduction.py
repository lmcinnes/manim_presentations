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
    TIMCSlide,
    make_heading,
    SmallText,
    mixbox_interpolate_color,
    cyclic_color,
)

import numpy as np

apply_defaults()


class Introduction(TIMCSlide):

    def construct(self):
        title = self.add_centered_text(
            "The Challenges of Visualizing High Dimensional Data",
            font="Marcellus SC",
            line_spacing=0.9,
            animate=False,
        ).shift(UP)
        sub_title = Text(
            "VISxAI 2026", font="Marcellus SC", font_size=48, color=ACCENT_COLOR
        ).next_to(title, DOWN, buff=1.0)
        author = Text(
            "Leland McInnes", font="Marcellus SC", font_size=48, color=ACCENT_COLOR
        ).next_to(sub_title, DOWN, buff=0.5)
        self.add(title, sub_title, author)

        self.wait(2)
        self.marked_next_slide()

        self.clear_slide()
        self.add_centered_text("How hard can it be?")

        self.new_section("Visualization")
