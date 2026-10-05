"""
UMAP of MNIST with the pixel-space first-nearest-neighbour graph overlaid.

    python mnist_knn_prep.py            # once: writes mnist_knn.npz
    manim-slides render mnist_knn_slides.py MNISTNearestNeighbourSlide

Slide logic lives in knn_edge_slides.py. To hand-pick showcase edges, look
at mnist_knn.candidates.png and list edge indices in SHOWCASE_EDGES.
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


class MNISTNearestNeighbourSlide(NearestNeighbourEdgeSlide):
    DATA_FILE = Path(os.environ.get("MNIST_KNN", _HERE / "mnist_knn.npz"))
    SHOWCASE_EDGES = None  # e.g. [17342, 35211, 24108, 19899, 56425]
    BADGE_SCALE = 0.5

    def title_embedding(self):
        return ("UMAP of MNIST", f"{self.n:,} handwritten digits, coloured by label")

    def title_all_edges(self):
        return (
            "Nearest neighbours in pixel space",
            f"each digit joined to its single closest image "
            f"({self.dim} dimensions)",
        )

    def title_showcase(self, name_i, name_j):
        return (
            f"A {name_i} and a {name_j}: nearest neighbours",
            "closest image in pixel space, from opposite ends of the map",
        )
