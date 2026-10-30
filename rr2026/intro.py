from manim import *

import sys

sys.path.append("..")  # Add parent directory to path to import config

from umap_talk_common import SECTION_TITLES

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

apply_defaults()


class TitleAndIntro(TIMCSlide):

    def construct(self):
        ## TITLE SLIDE
        logo = (
            SVGMobject("umap_logo_horizontal_converted_text.svg")
            .scale(1.0)
            .shift(UP * 1.5)
        )
        venue = Text(
            "TIMC Research Review 2026",
            font_size=72,
            font="Marcellus SC",
            # "RC Meeting 2026, Toronto Canada",
            # font_size=42,
            # font="Marcellus SC",
        ).next_to(logo, DOWN, buff=1)
        speaker = Text(
            "Leland McInnes",
            color=ACCENT_COLOR,
            font_size=40,
            font="Marcellus SC",
        ).next_to(venue, DOWN)

        self.add(logo, venue, speaker)
        self.wait(3)

        self.marked_next_slide()
        self.clear_slide()

        self.add_centered_text("UMAP has been very successful over the years")

        self.marked_next_slide()
        self.clear_slide()

        self.add_centered_text("Work is ongoing with come recent new developments")

        self.marked_next_slide()

        self.start_section_wipe(SECTION_TITLES["optimizers"])
