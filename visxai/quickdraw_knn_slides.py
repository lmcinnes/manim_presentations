"""
UMAP of Quick, Draw! doodles embedded with DINOv2, with the first-nearest-
neighbour graph (in DINOv2 embedding space) overlaid.

    python quickdraw_knn_prep.py        # once: writes quickdraw_knn.npz
    manim-slides render quickdraw_knn_slides.py QuickDrawNearestNeighbourSlide

Slide logic lives in knn_edge_slides.py. To hand-pick showcase edges, look
at quickdraw_knn.candidates.png and list edge indices in SHOWCASE_EDGES.
"""

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

apply_defaults()

import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.append(str(_HERE))  # shared module lives next to this file

from knn_edge_slides import NearestNeighbourEdgeSlide


class QuickDrawNearestNeighbourSlide(NearestNeighbourEdgeSlide):
    DATA_FILE = Path(os.environ.get("QUICKDRAW_KNN", _HERE / "quickdraw_knn.npz"))
    SHOWCASE_EDGES = [1924, 1571, 2284]
    POINT_SIZE = 2.5  # fewer points than MNIST: draw them larger
    EDGE_WIDTH = 0.6
    EDGE_OPACITY = 0.3
    LONG_EDGE_WIDTH = 1.2
    LONG_EDGE_OPACITY = 0.6
    BADGE_SCALE = 0.36

    def title_embedding(self):
        return (
            "Quick, Draw! doodles, embedded with DINOv2",
            f"UMAP of {self.n:,} doodles from {len(self.names)} "
            "classes, coloured by class",
        )

    def title_all_edges(self):
        return (
            "Nearest neighbours in DINOv2 space",
            f"each doodle joined to its single closest doodle "
            f"({self.dim}-dimensional embedding)",
        )

    def title_showcase(self, name_i, name_j):
        return (
            f"{self.caption(name_i).capitalize()} and "
            f"{self.caption(name_j)}: nearest neighbours",
            "closest doodle in DINOv2 space, from opposite ends of the map",
        )
