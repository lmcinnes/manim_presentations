from manim import *

import sys

sys.path.append("..")  # Add parent directory to path to import config

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

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import pickle
from scipy.signal import find_peaks
import os
import json

from data_manifest import (
    BASE_DATA as BASE_DATA_PATH,
    DATA_COLORMAP as DATA_COLORMAP_PATH,
    BARCODE_BARS as BARCODE_BARS_PATH,
    PERSISTENCE_SCORES_TRACE as PERSISTENCE_TRACE_PATH,
    CLUSTER_DENSITY_PROFILES as CLUSTER_DENSITY_PROFILES_PATH,
    CLUSTER_SIZES as CLUSTER_SIZES_PATH,
    CLUSTER_BINARY_TREE as CLUSTER_BINARY_TREE_PATH,
    IMAGE_DIR,
    EMBEDDING_CARTOON_RAW,
    EMBEDDING_CARTOON_EDGES,
    EVOC_LOGO_DATA,
    EVOC_LOGO_COLORS,
    ICON_DELUGE_SIMULATION,
    SCALED_CTREE,
    SCALED_POINTS_IN_PDF_ORDER,
    SCALED_DENSITY_VALUES,
    BENCHMARK_DATASETS,
    BENCHMARK_METRICS,
    BENCHMARK_ALGORITHMS,
    benchmark_file,
    benchmark_ylim_file,
    benchmark_yticks_file,
)

from density_and_barcodes import (
    density_profile_for_cluster,
    lambda_to_density,
)
from density_and_barcodes import ensure_data as _ensure_density_data
from falling_icon_simulation import ensure_data as _ensure_simulation_data

# ── Ensure precomputed data exists ───────────────────────────────────────────
_ensure_density_data()
_ensure_simulation_data()

# ── Module-level data loading ────────────────────────────────────────────────

base_data = (np.load(BASE_DATA_PATH) + 0.5) * 10
data_colormap = np.load(DATA_COLORMAP_PATH)

order = np.arange(base_data.shape[0])
rng_state = np.random.RandomState(42)
rng_state.shuffle(order)
base_data = base_data[order]
data_colormap = data_colormap[order]
SCALED_BASE_DATA = base_data * (12, 9)

_ALGO_DISPLAY_NAMES = {
    "kmeans": "K-Means",
    "umap_hdbscan": "UMAP+HDBSCAN",
    "EVoC": "EVoC",
}

benchmark_data = {}
for _ds in BENCHMARK_DATASETS:
    benchmark_data[_ds] = {}
    for _metric in BENCHMARK_METRICS:
        benchmark_data[_ds][_metric] = {
            "swarms": {
                _ALGO_DISPLAY_NAMES[algo]: np.load(benchmark_file(_ds, _metric, algo))
                for algo in BENCHMARK_ALGORITHMS
            },
            "ylim": np.load(benchmark_ylim_file(_ds, _metric)),
            "yticks": np.load(benchmark_yticks_file(_ds, _metric)),
        }

SWARM_COLORS = {
    "K-Means": DEFAULT_COLOR,
    "UMAP+HDBSCAN": ACCENT_COLOR,
    "EVoC": COLOR_CYCLE[1],
}

barcode_bars = np.load(BARCODE_BARS_PATH)
persistence_trace = np.load(PERSISTENCE_TRACE_PATH)

cluster_density_profiles = np.load(CLUSTER_DENSITY_PROFILES_PATH)
cluster_sizes = np.load(CLUSTER_SIZES_PATH)
cluster_binary_tree = np.load(CLUSTER_BINARY_TREE_PATH)

knn_graph_embedding_raw_data = np.load(EMBEDDING_CARTOON_RAW)
knn_graph_embedding_raw_data -= knn_graph_embedding_raw_data.mean(axis=0)
knn_graph_embedding_edges = pd.read_csv(EMBEDDING_CARTOON_EDGES)

# Precomputed HDBSCAN on SCALED_BASE_DATA
with open(SCALED_CTREE, "rb") as _f:
    scaled_ctree = pickle.load(_f)
scaled_points_in_pdf_order = np.load(SCALED_POINTS_IN_PDF_ORDER)
scaled_pdf_order_of_points = np.argsort(scaled_points_in_pdf_order)
scaled_density_values = np.load(SCALED_DENSITY_VALUES)

# ---------------------------------------------------------------------------
# Presentation constants
# ---------------------------------------------------------------------------
MAX_BARCODE_TIME = 350  # Upper bound for barcode x-axis / minimum-cluster-size sweep
COOLING_WINDOW = 100  # Trailing-gradient window for score trace


def is_active_cluster(cluster_num, binary_tree, cluster_sizes, minimum_cluster_size):

    if binary_tree[cluster_num, 0] == 0 and binary_tree[cluster_num, 1] == 0:
        return True

    left, right = binary_tree[cluster_num]

    if (
        cluster_sizes[left] < minimum_cluster_size
        and cluster_sizes[right] < minimum_cluster_size
    ):
        return True
    elif cluster_sizes[left] < minimum_cluster_size:
        return is_active_cluster(
            right, binary_tree, cluster_sizes, minimum_cluster_size
        )
    elif cluster_sizes[right] < minimum_cluster_size:
        return is_active_cluster(left, binary_tree, cluster_sizes, minimum_cluster_size)
    else:
        return False


def _create_curved_annotation(
    scene, text_content, text_shift, arrow_mob, mid_offsets, tip_angle_offset
):
    """Create a curved annotation arrow pointing from text at bottom to an arrow mob.

    Parameters
    ----------
    scene : Scene
        The manim scene (for add_fixed_in_frame_mobjects / play).
    text_content : str
        Paragraph text content (may include newlines).
    text_shift : np.ndarray
        Shift applied after ``.to_edge(DOWN)``.
    arrow_mob : Mobject
        The arrow whose centre the curve points toward.
    mid_offsets : tuple[np.ndarray, np.ndarray]
        ``(text_top_mid_offset, arrow_center_mid_offset)`` – the two
        middle control-point offsets for the smooth curve.
    tip_angle_offset : float
        Angle adjustment for the StealthTip (radians).
    """
    color = COLOR_CYCLE[3]
    text = Paragraph(
        text_content,
        font_size=24,
        alignment="center",
        color=color,
        stroke_color=color,
    )
    text.to_edge(DOWN).shift(text_shift)

    curve_points = [
        text.get_top() + UP * 0.2,
        text.get_top() + mid_offsets[0],
        arrow_mob.get_center() + mid_offsets[1],
        arrow_mob.get_center() + DOWN * 0.25,
    ]

    curved_path = VMobject(color=color, stroke_width=4)
    curved_path.set_points_smoothly(curve_points)

    tip = StealthTip(color=color)
    tip.move_to(curve_points[-1])
    direction = curve_points[-1] - curve_points[-2]
    angle = np.arctan2(direction[1], direction[0]) + tip_angle_offset
    tip.rotate(angle)

    arrow_group = VGroup(curved_path, tip)

    scene.add_fixed_in_frame_mobjects(arrow_group)
    scene.play(Create(curved_path), GrowFromCenter(tip))
    scene.add_fixed_in_frame_mobjects(text)
    scene.play(Write(text))

    return text, arrow_group


def _build_density_polygons(
    axes, density_profiles, cluster_sizes, x_values, color_func, positive_only_min=False
):
    """Build filled polygons from cluster density profiles.

    Parameters
    ----------
    axes : Axes
        Manim axes for coordinate conversion.
    density_profiles : np.ndarray
        2-D array  (n_clusters, n_points).
    cluster_sizes : array-like
        Size of each cluster.
    x_values : np.ndarray
        X-axis values matching the profile columns.
    color_func : callable(int) -> ManimColor
        Maps cluster index to a fill colour.
    positive_only_min : bool
        If True, compute y_min from positive values only.

    Returns
    -------
    polygons : list[Polygon]
    polygon_sizes : list
    """
    polygons = []
    polygon_sizes = []
    for i, y_values in enumerate(density_profiles):
        y_min = (
            np.min(y_values[y_values > 0]) if positive_only_min else np.min(y_values)
        )
        if np.sum(y_values > y_min):
            indices = np.where(y_values > y_min)[0]
            start, end = indices[0], indices[-1]
            curve_pts = [
                axes.c2p(x_values[j], y_values[j]) for j in range(start, end + 1)
            ]
            bottom_right = axes.c2p(x_values[end], y_min)
            bottom_left = axes.c2p(x_values[start], y_min)
            polygons.append(
                Polygon(
                    *curve_pts,
                    bottom_right,
                    bottom_left,
                    stroke_width=0,
                    fill_opacity=1.0,
                    color=color_func(i),
                )
            )
            polygon_sizes.append(cluster_sizes[i])
        else:
            polygons.append(Polygon((0, 0, 0), (0, 0, 0), (0, 0, 0)))
            polygon_sizes.append(cluster_sizes[i])
    return polygons, polygon_sizes


class PersistenceBarcode(VGroup):
    def __init__(self, axes, data, weights, stroke_width=8, **kwargs):
        """
        A generalizable Barcode Mobject.

        Args:
            axes: The Manim Axes object to plot on.
            data: Nx2 array of [birth, death] values.
            weights: Array of weight values for coloring.
        """
        super().__init__(**kwargs)
        self.axes = axes
        self.data = data
        self.weights = weights
        self.bars = VGroup()
        self.scanner = None
        self.stroke_width = stroke_width

        # Initialize bars (hidden or static depending on usage)
        self._create_bars()
        self.add(self.bars)

    def _create_bars(self):
        for i, (start, end) in enumerate(self.data):
            # Default to full length
            p1 = self.axes.c2p(start, i + 0.5)
            p2 = self.axes.c2p(end, i + 0.5)
            bar = Line(p1, p2, stroke_width=self.stroke_width)

            # Attach properties to mobject for updaters
            bar.birth = start
            bar.death = end
            bar.weight = self.weights[i]

            # Initial static color
            bar.set_color(self._get_static_color(bar.weight))
            self.bars.add(bar)

    def _get_static_color(self, weight):
        """Standard inactive color based on weight."""
        return interpolate_color(WHITE, ACCENT_COLOR, weight)

    def _get_active_color(self, weight):
        """Standard active/highlight color."""
        # You can adjust this logic to match your preference
        return interpolate_color(WHITE, COLOR_CYCLE[1], min(1, weight**2 + 0.1))

    def add_scanner(self, tracker, color=RED):
        """Adds a vertical scanning line tied to the tracker."""
        self.scanner = Line(UP, DOWN, color=color, stroke_width=2)

        def update_scanner(line):
            t = tracker.get_value()
            # Spans from bottom of axes to top of data
            start_point = self.axes.c2p(t, 0)
            end_point = self.axes.c2p(t, len(self.data))
            line.put_start_and_end_on(start_point, end_point)

        self.scanner.add_updater(update_scanner)
        self.add(self.scanner)
        return self.scanner

    def add_highlight_updater(self, tracker):
        """
        Mode 1: Static bars that 'light up' when scanner passes over.
        Used in PersistenceScoring.
        """

        def update(bars):
            t = tracker.get_value()
            for bar in bars:
                if bar.birth <= t <= bar.death:
                    bar.set_color(self._get_active_color(bar.weight))
                else:
                    bar.set_color(self._get_static_color(bar.weight))

        self.bars.add_updater(update)

    def add_growing_updater(self, tracker):
        """
        Mode 2: Bars that 'grow' from birth to death as scanner moves.
        Used in ClusterExtraction.
        """

        def update(bars):
            t = tracker.get_value()
            for i, bar in enumerate(bars):
                # Calculate effective endpoint
                endpoint = min(t, bar.death)

                if endpoint > bar.birth:
                    # Bar is visible (at least partially)
                    bar.set_stroke(opacity=1.0)

                    # Update geometry to stretch bar
                    p1 = self.axes.c2p(bar.birth, i + 0.5)
                    p2 = self.axes.c2p(endpoint, i + 0.5)
                    bar.put_start_and_end_on(p1, p2)

                    # Color logic: Active (growing) vs Finished
                    if t < bar.death:
                        bar.set_color(COLOR_CYCLE[1])  # Active color
                    else:
                        bar.set_color(
                            self._get_static_color(bar.weight)
                        )  # Finished color
                else:
                    # Bar hasn't started yet
                    bar.set_stroke(opacity=0.0)

        self.bars.add_updater(update)


apply_defaults()


class Benchmarks(TIMCSlide):

    def construct(self):
        self.load_state("benefits")

        self.new_section("EVōC Performance")

        # 1. Configuration for the 3 distinct datasets
        datasets = [
            {
                "name_lines": [
                    "Clustering",
                    "results for",
                    "CIFAR-100",
                    "Embedded with",
                    "CLIP",
                ],
                "data": {
                    "ARI": (
                        benchmark_data["cifar"]["ari"]["swarms"],
                        benchmark_data["cifar"]["ari"]["yticks"],
                    ),
                    "Score": (
                        benchmark_data["cifar"]["cs"]["swarms"],
                        benchmark_data["cifar"]["ari"]["yticks"],
                    ),
                    "Time": (
                        benchmark_data["cifar"]["time"]["swarms"],
                        benchmark_data["cifar"]["time"]["yticks"],
                    ),
                },
            },
            {
                "name_lines": [
                    "Clustering",
                    "results for",
                    "20-Newsgroups",
                    "Embedded with",
                    "mpnet-base-v2",
                ],
                "data": {
                    "ARI": (
                        benchmark_data["news"]["ari"]["swarms"],
                        benchmark_data["news"]["ari"]["yticks"],
                    ),
                    "Score": (
                        benchmark_data["news"]["cs"]["swarms"],
                        benchmark_data["news"]["ari"]["yticks"],
                    ),
                    "Time": (
                        benchmark_data["news"]["time"]["swarms"],
                        benchmark_data["news"]["time"]["yticks"],
                    ),
                },
            },
            {
                "name_lines": [
                    "Clustering",
                    "results for",
                    "BirdCLEF-2023",
                    "Embedded with",
                    "Google",
                    "Bird Vocalization",
                    "Classifier",
                ],
                "data": {
                    "ARI": (
                        benchmark_data["bird"]["ari"]["swarms"],
                        benchmark_data["bird"]["ari"]["yticks"],
                    ),
                    "Score": (
                        benchmark_data["bird"]["cs"]["swarms"],
                        benchmark_data["bird"]["ari"]["yticks"],
                    ),
                    "Time": (
                        benchmark_data["bird"]["time"]["swarms"],
                        benchmark_data["bird"]["time"]["yticks"],
                    ),
                },
            },
        ]

        categories = ["K-Means", "UMAP+HDBSCAN", "EVoC"]

        # Define the layout box for the title (visual reference, not added to scene)
        title_box = Rectangle(height=6, width=4).to_edge(RIGHT, buff=0.75)

        # Trackers for objects that persist across loops
        current_plot_group = VGroup()
        current_dots = VGroup()
        current_title = None

        for dataset in datasets:
            # --- 1. Title Sequence ---
            title_text = self._create_title(dataset["name_lines"], title_box)
            self.play(Write(title_text))
            self.marked_next_slide()

            current_title = title_text

            # --- 2. Iterate Metrics (ARI -> Score -> Time) ---
            for i, (metric_name, (swarm_data, y_ticks)) in enumerate(
                dataset["data"].items()
            ):
                y_min = y_ticks[0]
                y_max = y_ticks[-1]
                y_step = y_ticks[1] - y_ticks[0]

                # Create the axes using the config helper
                new_plot_group = create_styled_axes(
                    x_range=[0, 4, 1],
                    y_range=[y_min, y_max, y_step],
                    x_label_tex="Clustering Algorithm",
                    y_label_tex=self._get_metric_label(metric_name),
                    y_decimal_places=0 if metric_name == "Time" else 2,
                    x_tick_labels=categories,
                ).shift(UP * 0.5 + LEFT)
                new_axes = new_plot_group[0]

                # --- 3. Handle Transitions ---
                if i == 0:
                    # First metric: Fade everything IN
                    self.play(
                        current_title.animate.move_to(title_box.get_center()),
                        Create(new_plot_group),
                    )

                    # Create Dots (Only needed once per dataset)
                    current_dots = self._create_swarm_dots(
                        categories, swarm_data, new_axes
                    )
                    self.play(
                        LaggedStart(
                            *[FadeIn(dot, scale=2.0) for dot in current_dots],
                            lag_ratio=3.0 / len(current_dots),
                        ),
                        run_time=1.0,
                    )

                else:
                    # Subsequent metrics: Transform Axes and Move Dots
                    dot_animations = self._get_dot_move_animations(
                        categories,
                        swarm_data,
                        current_dots,
                        new_axes,
                        metric_name,
                    )

                    self.play(
                        ReplacementTransform(current_plot_group, new_plot_group),
                        *dot_animations,
                    )

                current_plot_group = new_plot_group
                self.marked_next_slide()

            # --- 4. Cleanup before next dataset ---
            if not "Classifier" in dataset["name_lines"]:
                self.play(
                    FadeOut(current_plot_group),
                    FadeOut(current_dots),
                    FadeOut(current_title),
                )
            else:
                print(f"Skipping cleanup for {dataset}")

        self.save_state("evoc_performance")

    def _create_title(self, lines, target_box):
        """Creates the title paragraph, scaled to fit the target box width."""
        title = Paragraph(*lines, alignment="center", line_spacing=0.5)
        title.scale_to_fit_width(target_box.width - 0.5)
        return title

    def _get_metric_label(self, metric_name):
        """Returns the Tex string for axis labels."""
        if metric_name == "ARI":
            return "Adjusted Rand Index"
        elif metric_name == "Score":
            return "Clustering Score"
        elif metric_name == "Time":
            return "Time (s)"
        return metric_name

    def _create_swarm_dots(self, categories, swarm_data, axes):
        """Generates the VGroup of dots for the initial swarm plot."""
        dots_group = VGroup()
        # Create a substructure to easily find dots by category later
        dots_group.category_map = {cat: VGroup() for cat in categories}

        for i, category in enumerate(categories):
            points_list = swarm_data.get(category, [])
            for pos in points_list:
                x_coord = 1 + i + (pos[0] - i) * 4
                y_coord = pos[1]

                dot = Dot(
                    axes.c2p(x_coord, y_coord),
                    radius=0.05,
                    color=SWARM_COLORS[category],
                    stroke_width=0.5,
                    stroke_color=BACKGROUND_COLOR,
                )
                dots_group.add(dot)
                dots_group.category_map[category].add(dot)

        return dots_group

    def _get_dot_move_animations(
        self,
        categories,
        target_swarm_data,
        current_dots_group,
        target_axes,
        metric_name,
    ):
        """Calculates animations to move existing dots to new positions."""
        animations = []

        for i, category in enumerate(categories):
            target_positions = target_swarm_data.get(category, [])
            current_category_dots = current_dots_group.category_map[category]

            # We assume len(dots) == len(target_positions) because it's the same dataset
            for j, dot in enumerate(current_category_dots):
                if j < len(target_positions):
                    pos = target_positions[j]

                    move_scale = 3.0 if metric_name == "Time" else 4.0

                    x_coord = 1 + i + (pos[0] - i) * move_scale
                    y_coord = pos[1]

                    animations.append(
                        dot.animate.move_to(target_axes.c2p(x_coord, y_coord))
                    )
        return animations


class PersistenceScoring(TIMCSlide):

    def construct(self):
        max_time = MAX_BARCODE_TIME
        barcode_data = barcode_bars[:, :2]
        barcode_weights = barcode_bars.T[2]
        barcode_data = np.minimum(barcode_data, max_time)
        n_bars = barcode_data.shape[0]
        x_values, y_values = persistence_trace.T
        score_max = max(y_values)
        score_min = min(y_values)

        # Shared configuration
        x_range = [0, max_time, 25]
        plot_width = 10
        plot_height = 2

        # Top Axes: Barcode
        # y-range is just the index of the bar (0 to n_bars)
        ax_top = Axes(
            x_range=x_range,
            y_range=[0, n_bars, n_bars // 10],
            x_length=plot_width,
            y_length=plot_height * 2,
            y_axis_config={"stroke_width": 0, "include_tip": False},
            x_axis_config={"include_numbers": False, "tip_shape": StealthTip},
        )

        # Bottom Axes: Score
        ax_bottom = Axes(
            x_range=x_range,
            y_range=[score_min - 2, score_max + 2, (score_max - score_min) // 5],
            x_length=plot_width,
            y_length=plot_height * 0.5,
            y_axis_config={"include_numbers": False, "include_tip": False},
            x_axis_config={
                "tip_shape": StealthTip,
                "include_numbers": True,
                "font_size": 16,
            },
        )

        # Group and arrange them vertically
        plots = VGroup(ax_top, ax_bottom).arrange(DOWN, buff=0.5)
        y_label = ax_bottom.get_y_axis_label(
            Paragraph("Persistence\nscore", font_size=36, alignment="center")
            .scale(0.5)
            .rotate(PI / 2),
            edge=LEFT,
            direction=LEFT,
        )
        x_label = ax_bottom.get_x_axis_label(
            Text("Minimum cluster size", font_size=36).scale(0.5),
            edge=DOWN,
            direction=DOWN,
            buff=0.2,
        )
        bottom_labels = VGroup(y_label, x_label)
        # --- A. The Barcode Lines ---
        barcode = PersistenceBarcode(ax_top, barcode_data, barcode_weights)
        bars = barcode.bars

        self.add(bars)
        self.add(ax_top)
        self.play(Create(ax_bottom), FadeIn(bottom_labels))

        # ---------------------------------------------------------
        # 3. CREATE VISUAL ELEMENTS
        # ---------------------------------------------------------

        self.marked_next_slide()

        # --- B. The Vertical Scanner Line ---
        # A tall line spanning the top plot
        scanner_line = Line(
            start=ax_bottom.c2p(0, y_values[0]),
            end=ax_top.c2p(0, n_bars + 1),
            color=RED,
            stroke_width=2,
        )
        self.play(GrowFromCenter(scanner_line))

        # # --- C. The Bottom Score Curve ---
        # # We use a Dot that follows the data, and a TracedPath to draw the line
        start_score = y_values[0]
        score_dot = Dot(color=COLOR_CYCLE[1], radius=0.1)
        score_dot.move_to(ax_bottom.c2p(0, start_score))
        self.add(score_dot)

        tracker = ValueTracker(0)

        def update_scanner(line):
            t = tracker.get_value()
            current_score_y = np.interp(t, x_values, y_values)
            line.set_points_by_ends(
                ax_bottom.c2p(t, current_score_y), ax_top.c2p(t, n_bars + 1)
            )

        scanner_line.add_updater(update_scanner)

        def update_bars(bar_group):
            t = tracker.get_value()
            for bar in bar_group:
                # Check intersection: birth <= t <= death
                if bar.birth <= t <= bar.death:
                    bar.set_color(
                        interpolate_color(
                            WHITE, COLOR_CYCLE[1], min(1, bar.weight**2 + 0.1)
                        )
                    )
                    # bar.set_stroke(width=6)
                else:
                    bar.set_color(
                        interpolate_color(
                            WHITE, ACCENT_COLOR, min(1, bar.weight**2 + 0.1)
                        )
                    )

        bars.add_updater(update_bars)

        self.history_points = [ax_bottom.c2p(0, y_values[0])]
        trailing_line = VMobject()
        self.add(trailing_line)

        def update_score_dot(dot):
            t = tracker.get_value()
            current_y = np.interp(t, x_values, y_values)
            new_pos = ax_bottom.c2p(t, current_y)
            dot.move_to(new_pos)
            self.history_points.append(new_pos)

            if len(self.history_points) > 1:
                trailing_line.set_points_as_corners(self.history_points)
                cooling_window = COOLING_WINDOW

                num_points = len(self.history_points)
                colors = []
                for i in range(num_points):
                    points_from_end = num_points - 1 - i

                    if points_from_end >= cooling_window:
                        colors.append(COLOR_CYCLE[0])
                    else:
                        alpha = 1.0 - (points_from_end / cooling_window)
                        colors.append(
                            interpolate_color(COLOR_CYCLE[0], COLOR_CYCLE[1], alpha)
                        )

                trailing_line.set_stroke(color=colors[::-1], width=4)

        score_dot.add_updater(update_score_dot)

        self.play(tracker.animate.set_value(max_time), run_time=10, rate_func=linear)

        self.next_slide()

        self.history_points = []
        line_plot_xs = []
        line_plot_ys = []
        for t in np.linspace(0, max_time, int(config.frame_rate * 10)):
            current_y = np.interp(t, x_values, y_values)
            new_pos = ax_bottom.c2p(t, current_y)
            line_plot_xs.append(t)
            line_plot_ys.append(current_y)
            self.history_points.append(new_pos)
        colors = []
        num_points = len(self.history_points)
        for i in range(num_points):
            points_from_end = num_points - 1 - i
            if points_from_end >= COOLING_WINDOW:
                colors.append(COLOR_CYCLE[0])
            else:
                alpha = 1.0 - (points_from_end / COOLING_WINDOW)
                colors.append(interpolate_color(COLOR_CYCLE[0], COLOR_CYCLE[1], alpha))
        trailing_line.set_points_as_corners(self.history_points)
        trailing_line.set_stroke(color=colors[::-1], width=4)
        self.play(
            FadeOut(scanner_line),
            FadeOut(score_dot),
            trailing_line.animate.set_stroke(color=COLOR_CYCLE[0], width=4),
        )
        self.play(FadeOut(ax_top, bars))
        new_ax_bottom = Axes(
            x_range=x_range,  # Keep same x-range
            y_range=[score_min - 2, score_max + 2, (score_max - score_min) // 5],
            x_length=plot_width,  # Same width
            y_length=plot_height
            * 1.8,  # Make it 1.8 times as tall (or whatever factor)
            y_axis_config={"include_numbers": False, "include_tip": False},
            x_axis_config={
                "tip_shape": StealthTip,
                "include_numbers": True,
                "font_size": 16,
            },
        )
        new_y_label = new_ax_bottom.get_y_axis_label(
            Paragraph("Persistence\nscore", font_size=36, alignment="center")
            .scale(0.5)
            .rotate(PI / 2),
            edge=LEFT,
            direction=LEFT,
        )
        new_x_label = new_ax_bottom.get_x_axis_label(
            Text("Minimum cluster size", font_size=36).scale(0.5),
            edge=DOWN,
            direction=DOWN,
            buff=0.2,
        )
        new_bottom_labels = VGroup(new_y_label, new_x_label)

        # Create the new line on the new axes
        new_points = [
            new_ax_bottom.c2p(x, y) for x, y in zip(line_plot_xs, line_plot_ys)
        ]
        new_trailing_line = VMobject()
        new_trailing_line.set_points_as_corners(new_points)
        new_trailing_line.set_stroke(color=COLOR_CYCLE[0], width=4)

        # Animate the transition
        self.play(
            Transform(ax_bottom, new_ax_bottom),
            Transform(trailing_line, new_trailing_line),
            Transform(bottom_labels, new_bottom_labels),
        )

        self.marked_next_slide()

        peak_idxs = find_peaks(persistence_trace.T[1])[0]
        peak_sizes = persistence_trace.T[0][peak_idxs]
        peak_scores = persistence_trace.T[1][peak_idxs]

        maxima_idx = np.argmax(peak_scores)
        maxima_size = peak_sizes[maxima_idx]
        maxima_score = peak_scores[maxima_idx]
        maxima_annotation_arrow = Arrow(
            start=new_ax_bottom.c2p(maxima_size, maxima_score) + UP,
            end=new_ax_bottom.c2p(maxima_size, maxima_score),
            buff=0.1,
            color=COLOR_CYCLE[3],
            stroke_width=8,
            max_tip_length_to_length_ratio=0.3,
            max_stroke_width_to_length_ratio=10,
        )
        maxima_annotation_text = Paragraph(
            "Maximal\npersistence\nscore",
            font_size=24,
            color=COLOR_CYCLE[3],
            stroke_color=COLOR_CYCLE[3],
            alignment="center",
        ).next_to(maxima_annotation_arrow, UP, buff=0.2)
        self.play(Create(maxima_annotation_arrow), Write(maxima_annotation_text))
        self.marked_next_slide()

        green = COLOR_CYCLE[2]  # interpolate_color(WHITE, COLOR_CYCLE[3], 0.66)
        local_maxima_arrows = []
        for peak_size, peak_score in zip(peak_sizes, peak_scores):
            if peak_score == maxima_score:
                continue
            local_maxima_arrows.append(
                Arrow(
                    start=new_ax_bottom.c2p(peak_size, peak_score) + UP,
                    end=new_ax_bottom.c2p(peak_size, peak_score),
                    buff=0.1,
                    color=green,
                    stroke_width=4,
                    max_tip_length_to_length_ratio=0.2,
                    max_stroke_width_to_length_ratio=10,
                )
            )
        local_maxima_text = Text(
            "Other local maxima",
            color=green,
            stroke_color=green,
            font_size=24,
        ).shift(LEFT * 2)
        self.play(
            *[Create(arrow) for arrow in local_maxima_arrows], Write(local_maxima_text)
        )

        self.marked_next_slide()

        self.wait(1)
        self.play(
            *[
                mob.animate.shift(RIGHT * 15)
                for mob in self.mobjects
                if mob != self.logo
            ],
            run_time=1,
        )
        self.clear()
        add_logo_to_background(self)

        line1 = Text(
            "The maximum score defines",
            font_size=64,
            t2c={"maximum score": COLOR_CYCLE[3]},
        )
        line3 = Text("the default cluster resolution", font_size=64)

        paragraph1 = VGroup(line1, line3).arrange(DOWN, buff=0.2)

        line4 = Text("Other local maxima", font_size=64, t2c={"local maxima": green})
        line5 = Text("provide alternative layers", font_size=64)
        line6 = Text("of cluster resolution", font_size=64)
        line5.next_to(line4, DOWN, buff=0.15)
        line6.next_to(line5, DOWN, buff=-0.05)

        paragraph2 = VGroup(line4, line5, line6)

        final_text = VGroup(paragraph1, paragraph2).arrange(DOWN, buff=1.25)

        self.play(Write(final_text))

        self.save_state("persistence_scores")


class ClusterExtraction(TIMCSlide):

    def construct(self):

        self.end_section_wipe("Cluster Extraction")

        self.add_centered_text(
            "The cluster hierarchy sorts the points and provides a density estimate"
        )
        self.marked_next_slide()
        self.clear_slide()

        self.density_axes = Axes(
            x_range=[0, cluster_density_profiles.shape[1], 250],
            y_range=[0, 0.66, 0.1],
            x_length=10,
            y_length=6,
            axis_config={
                "include_tip": False,
                "color": DEFAULT_COLOR,
                "include_numbers": True,
                "font_size": 18,
            },
        )
        y_axis_label = self.density_axes.get_y_axis_label(
            Text("Estimated density").scale(0.33).rotate(PI / 2),
            edge=LEFT,
            direction=LEFT,
        )
        x_axis_label = self.density_axes.get_x_axis_label(
            Text("Point index").scale(0.33),
            edge=DOWN,
            direction=DOWN,
        )
        self.add(self.density_axes, y_axis_label, x_axis_label)

        n_points = cluster_density_profiles.shape[1]

        polygons = []
        polygon_sizes = []
        x_values = np.arange(n_points)
        polygon_color_gradient = color_gradient(
            [DEFAULT_COLOR, ACCENT_COLOR, WHITE], 100
        )

        def _gradient_color(idx):
            scaled = 1.0 - np.power(cluster_sizes[idx] / x_values.shape[0], 0.2)
            cidx = max(0, min(int(scaled * 99), 99))
            return polygon_color_gradient[cidx]

        polygons, polygon_sizes = _build_density_polygons(
            self.density_axes,
            cluster_density_profiles,
            cluster_sizes,
            x_values,
            _gradient_color,
        )

        self.play(
            LaggedStart(
                *[DrawBorderThenFill(polygon) for polygon in polygons], lag_ratio=0.001
            )
        )

        min_cluster_size_scale = NumberLine(
            [0, MAX_BARCODE_TIME, 50],
            length=6,
            rotation=PI / 2,
            include_numbers=True,
            font_size=12,
            label_direction=LEFT,
        )
        min_cluster_size_label = Text("Minimum cluster size", font_size=18).rotate(
            PI / 2
        )
        min_cluster_size_scale.next_to(self.density_axes)
        min_cluster_size_label.next_to(min_cluster_size_scale, LEFT, buff=0.1)
        size_marker = (
            Triangle()
            .rotate(-PI / 6)
            .scale(0.15)
            .set_fill(color=COLOR_CYCLE[1], opacity=1.0)
            .set_stroke(color=COLOR_CYCLE[1])
        )
        size_marker.move_to(min_cluster_size_scale.n2p(0) + RIGHT * 0.15)

        self.play(
            Create(min_cluster_size_scale),
            Write(min_cluster_size_label),
            FadeIn(size_marker),
        )

        self.marked_next_slide()

        tracker = ValueTracker(0)

        def update_size_marker(marker):
            t = tracker.get_value()
            size_marker.move_to(min_cluster_size_scale.n2p(t) + RIGHT * 0.15)

        size_marker.add_updater(update_size_marker)
        # Create the decimal label
        size_label = DecimalNumber(
            tracker.get_value(), num_decimal_places=0, include_sign=False, font_size=18
        ).set_color(COLOR_CYCLE[1])

        # Position it relative to the marker
        size_label.add_updater(lambda d: d.set_value(tracker.get_value()))
        size_label.add_updater(lambda d: d.next_to(size_marker, RIGHT, buff=0.1))

        self.add(size_label)
        polygon_group = VGroup(*polygons)
        self.add(polygon_group)

        def update_polygons(group):
            current_threshold = tracker.get_value()
            for i, (poly, size) in enumerate(zip(group, polygon_sizes)):
                if (
                    is_active_cluster(
                        i, cluster_binary_tree, cluster_sizes, current_threshold
                    )
                    and cluster_sizes[i] >= current_threshold
                ):
                    cluster_fill_color = COLOR_CYCLE[1]
                else:
                    # cluster_fill_color = interpolate_color(
                    #     ACCENT_COLOR,
                    #     DEFAULT_COLOR,
                    #     np.power(cluster_sizes[i] / x_values.shape[0], 0.25),
                    # )
                    scaled_color_val = 1.0 - np.power(
                        cluster_sizes[i] / x_values.shape[0], 0.2
                    )
                    color_idx = int(scaled_color_val * 99)
                    color_idx = max(0, min(color_idx, 99))
                    cluster_fill_color = polygon_color_gradient[color_idx]
                if size < current_threshold:
                    opacity = clip(0.1 + (size - current_threshold) / 100, 0.1, 1.0)
                    poly.set_fill(color=cluster_fill_color, opacity=opacity)
                    # poly.set_fill(color=cluster_fill_color, opacity=0.1)
                    poly.set_stroke(width=1, opacity=0.5)
                else:
                    poly.set_fill(color=cluster_fill_color, opacity=1.0)
                    poly.set_stroke(width=1, opacity=0.5)

        polygon_group.add_updater(update_polygons)

        def safe_wiggle(t):
            return np.exp(-t * 2) * (1.0 + np.sin(t * 7 * PI - PI / 2)) / 2

        self.play(
            tracker.animate.set_value(MAX_BARCODE_TIME),
            run_time=10,
            rate_func=safe_wiggle,
        )
        self.play(
            tracker.animate.set_value(15),
            run_time=3,
            rate_func=linear,
        )

        self.marked_next_slide()

        new_density_axes = Axes(
            x_range=[0, cluster_density_profiles.shape[1], 250],
            y_range=[0, 0.66, 0.1],
            x_length=10,
            y_length=3,
            axis_config={
                "include_tip": False,
                "color": DEFAULT_COLOR,
                "include_numbers": True,
                "font_size": 18,
            },
        ).shift(UP * 2)
        new_y_axis_label = new_density_axes.get_y_axis_label(
            Text("Estimated density").scale(0.33).rotate(PI / 2),
            edge=LEFT,
            direction=LEFT,
        )
        new_x_axis_label = new_density_axes.get_x_axis_label(
            Text("Point index").scale(0.33),
            edge=DOWN,
            direction=DOWN,
        )
        new_polygons, _ = _build_density_polygons(
            new_density_axes,
            cluster_density_profiles,
            cluster_sizes,
            np.arange(n_points),
            _gradient_color,
        )

        new_polygon_group = VGroup(*new_polygons)
        update_polygons(new_polygon_group)

        barcode_data = barcode_bars[:, :2]
        barcode_weights = barcode_bars.T[2]
        barcode_data = np.minimum(barcode_data, MAX_BARCODE_TIME)
        barcode_axis = Axes(
            x_range=[0, MAX_BARCODE_TIME, 25],
            y_range=[0, barcode_data.shape[0] + 1, 10],
            x_length=10,
            y_length=3,
            y_axis_config={"stroke_width": 0, "include_tip": False},
            x_axis_config={
                "include_numbers": True,
                "tip_shape": StealthTip,
                "font_size": 12,
            },
        ).shift(DOWN * 1.5)
        new_barcode_axis_label = barcode_axis.get_x_axis_label(
            Text("Minimum cluster size", font_size=18),
            edge=DOWN,
            direction=DOWN,
            buff=0.2,
        )

        swing_group = VGroup(min_cluster_size_scale, min_cluster_size_label)
        barcode_group = VGroup(barcode_axis, new_barcode_axis_label)

        # 2. Initialize our new Generalized Barcode
        barcode = PersistenceBarcode(
            barcode_axis, barcode_data, barcode_weights, stroke_width=6
        )
        barcode.bars.set_stroke(opacity=0)  # Start invisible for the growth effect
        self.add(barcode)

        barcode.add_growing_updater(tracker)

        size_marker.remove_updater(update_size_marker)
        size_label.clear_updaters()

        pivot_point = barcode_axis.get_origin()

        self.play(
            FadeOut(min_cluster_size_label),
            size_label.animate.set_opacity(0.0),
            run_time=0.25,
        )
        self.play(
            # STAGE 1: Physical Rigid Movement
            # We move and rotate the old axis as a solid piece
            min_cluster_size_scale.animate.rotate(-PI / 2)
            .move_to(pivot_point + RIGHT * 4.1)  # Align center roughly
            .scale(1.33)
            .set_opacity(0.75),  # Slight fade helps the transition
            # Move the marker along with it
            size_marker.animate.rotate(-PI / 2).move_to(
                barcode_axis.c2p(15, 0) + DOWN * 0.15
            ),
            size_label.animate.next_to(size_marker, DOWN, buff=0.1),
            # Simultaneous density plot transforms
            ReplacementTransform(self.density_axes, new_density_axes),
            ReplacementTransform(y_axis_label, new_y_axis_label),
            ReplacementTransform(x_axis_label, new_x_axis_label),
            ReplacementTransform(polygon_group, new_polygon_group),
            run_time=1.2,
            rate_func=bezier([0, 0, 0.5, 1]),  # Start slow, finish with momentum
        )
        self.play(
            # STAGE 2: The Hand-off
            # Now that it's in the right orientation, we morph into the real Axes object
            ReplacementTransform(min_cluster_size_scale, barcode_axis.get_x_axis()),
            FadeIn(new_barcode_axis_label),  # Fade in the final text labels
            run_time=0.8,
            rate_func=bezier([0.5, 0, 1, 1]),  # Take over the momentum and snap
        )

        # self.play(
        #     # Swing the barcode axis around
        #     ReplacementTransform(
        #         min_cluster_size_scale, barcode_axis, path_arc=-PI / 2
        #     ),
        #     # --- 2. THE TYPOGRAPHY (The "Cross-Fade") ---
        #     # Fade out old text while fading in new text at the final location
        #     FadeOut(min_cluster_size_label),
        #     FadeIn(new_barcode_axis_label),
        #     ReplacementTransform(self.density_axes, new_density_axes),
        #     ReplacementTransform(y_axis_label, new_y_axis_label),
        #     ReplacementTransform(x_axis_label, new_x_axis_label),
        #     ReplacementTransform(polygon_group, new_polygon_group),
        #     # *[
        #     #     ReplacementTransform(old_polygon, new_polygon)
        #     #     for old_polygon, new_polygon in zip(polygons, new_polygons)
        #     # ],
        #     # ReplacementTransform(min_cluster_size_scale, barcode_axis),
        #     # ReplacementTransform(min_cluster_size_label, new_barcode_axis_label),
        #     size_marker.animate,
        #     size_marker.animate.rotate(-PI / 2).move_to(
        #         barcode_axis.c2p(15, 0) + DOWN * 0.15
        #     ),
        #     size_label.animate.next_to(size_marker, DOWN, buff=0.1),
        # )

        self.marked_next_slide()

        def new_update_size_marker(marker):
            t = tracker.get_value()
            size_marker.move_to(barcode_axis.c2p(t, 0) + DOWN * 0.15)

        size_marker.add_updater(new_update_size_marker)
        size_label.add_updater(lambda d: d.set_value(tracker.get_value()))
        size_label.add_updater(lambda d: d.next_to(size_marker, DOWN, buff=0.1))

        self.play(size_label.animate.set_opacity(1.0), run_time=0.5)
        new_polygon_group.add_updater(update_polygons)

        self.play(
            tracker.animate.set_value(MAX_BARCODE_TIME),
            run_time=15,
            rate_func=lambda x: x**3,
        )

        self.marked_next_slide()

        n_bars = barcode_data.shape[0]
        x_range = [0, MAX_BARCODE_TIME, 25]

        # Top Axes: Barcode
        # y-range is just the index of the bar (0 to n_bars)
        ax_top = Axes(
            x_range=x_range,
            y_range=[0, n_bars, n_bars // 10],
            x_length=10,
            y_length=4,
            y_axis_config={"stroke_width": 0, "include_tip": False},
            x_axis_config={"include_numbers": False, "tip_shape": StealthTip},
        ).move_to([0.0, 0.93294936, 0.0])

        new_barcode = PersistenceBarcode(ax_top, barcode_data, barcode_weights)

        self.play(
            FadeOut(
                new_density_axes,
                new_polygon_group,
                new_x_axis_label,
                new_y_axis_label,
                new_barcode_axis_label,
                size_marker,
                size_label,
            )
        )
        self.play(
            ReplacementTransform(barcode_axis, ax_top),
            ReplacementTransform(barcode, new_barcode),
        )


class ManifoldLearning(ThreeDTIMCSlide):

    def construct(self):

        self.end_section_wipe("Manifold Learning")

        self.add_centered_text(
            "We need to build a clusterable representation", max_width=0.66
        )
        self.marked_next_slide()
        self.clear_slide(run_time=1)

        # Setup 3D axes
        axes = ThreeDAxes(
            x_range=[-15, 15, 4],
            y_range=[-15, 15, 4],
            z_range=[-15, 15, 4],
            x_length=10,
            y_length=10,
            z_length=8,
        )

        # Set camera orientation for 3D view
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)

        # Animate the axes appearing
        self.play(Create(axes))
        self.wait(0.5)

        # Create scatter points as Dot3D objects
        # Store them in a list so we can reference them later for edges
        scatter_dots = []
        for point in knn_graph_embedding_raw_data:
            dot = Dot3D(
                point=axes.c2p(*point),  # Convert coordinates to scene position
                color=ACCENT_COLOR,
                radius=0.1,
                resolution=(16, 16),
            )
            scatter_dots.append(dot)

        # Animate points appearing one by one (or all at once)
        # Option 1: All at once
        self.play(*[GrowFromCenter(dot) for dot in scatter_dots])

        # Option 2: One by one (uncomment to use instead)
        # for dot in scatter_dots:
        #     self.play(GrowFromCenter(dot), run_time=0.3)

        self.marked_next_slide()

        # Slowly rotate the camera to show 3D nature
        self.begin_ambient_camera_rotation(rate=0.75)
        self.wait(3)
        self.stop_ambient_camera_rotation()

        edges = []
        for _, row in knn_graph_embedding_edges.iterrows():
            source_idx = int(row["source"])
            target_idx = int(row["target"])
            weight = row["weight"]

            # Create line between dots
            edge = Line3D(
                start=scatter_dots[source_idx].get_center(),
                end=scatter_dots[target_idx].get_center(),
                resolution=16,
                color=GRAY,
                thickness=weight**2 * 0.02,
            )
            edges.append(edge)

        self.play(
            *[GrowFromPoint(edge, edge.get_start()) for edge in edges], run_time=1.5
        )
        self.bring_to_front(*scatter_dots)

        self.wait(1)
        self.play(FadeOut(axes))
        self.marked_next_slide()

        vertices = list(range(len(scatter_dots)))

        # Build edge list from your dataframe
        edge_list = [
            (int(row["source"]), int(row["target"]))
            for _, row in knn_graph_embedding_edges.iterrows()
        ]

        layout = {i: point.get_center() for i, point in enumerate(scatter_dots)}

        # Create the 2D Graph with a layout
        graph_2d = Graph(
            vertices=vertices,
            edges=edge_list,
            layout=layout,  # "random",
            layout_scale=3,
            vertex_config={"radius": 0.1, "color": ACCENT_COLOR},
            edge_config={"color": GRAY},
        )

        # Optional: apply edge weights to thickness
        for i, (_, row) in enumerate(knn_graph_embedding_edges.iterrows()):
            graph_2d.edges[edge_list[i]].set_stroke(width=row["weight"] * 5)

        # First, transition camera to 2D view
        self.move_camera(phi=0, theta=-90 * DEGREES, run_time=2)

        # Create thin 2D lines at current 3D positions
        temp_lines = [
            Line(
                edge.get_start(),
                edge.get_end(),
                color=GRAY,
                stroke_width=knn_graph_embedding_edges["weight"].values[i] * 5,
            )
            for i, edge in enumerate(edges)
        ]

        # Fade out 3D edges, fade in thin lines
        line_replacements = [FadeOut(edge) for edge in edges], *[
            FadeIn(line) for line in temp_lines
        ]

        # Transform 3D objects into 2D graph
        # Match up dots with graph vertices and edges with graph edges
        vertex_transforms = [
            ReplacementTransform(scatter_dots[i], graph_2d.vertices[i])
            for i in vertices
        ]

        edge_transforms = [
            ReplacementTransform(temp_lines[i], graph_2d.edges[edge_list[i]])
            for i in range(len(edge_list))
        ]

        self.play(*vertex_transforms, *line_replacements, run_time=0.5)
        self.play(edge_transforms, run_time=0.25)

        # Create 2D axes (using Axes, not ThreeDAxes)
        axes_2d = Axes(
            x_range=[0, 10, 2],
            y_range=[0, 10, 2],
            x_length=6,
            y_length=6,
            axis_config={
                "include_tip": True,
                "color": DEFAULT_COLOR,
                "tip_shape": StealthTip,
            },
            y_axis_config={
                "numbers_to_include": [2, 4, 6, 8],
                "decimal_number_config": {"num_decimal_places": 0},
                "font_size": 18,
                "stroke_width": 1,
                "include_tip": True,
            },
            x_axis_config={
                "numbers_to_include": [2, 4, 6, 8],
                "decimal_number_config": {"num_decimal_places": 0},
                "font_size": 18,
                "stroke_width": 1,
                "include_tip": True,
            },
        )

        self.wait(0.1)
        self.marked_next_slide()

        # Transition old 3D axes to 2D axes
        self.play(
            Create(axes_2d),
        )

        # self.marked_next_slide()

        graph_2d_new = Graph(
            vertices=vertices,
            edges=edge_list,
            layout="kamada_kawai",
            layout_scale=3,
            vertex_config={"radius": 0.1, "color": ACCENT_COLOR},
            edge_config={"color": GRAY},
        )
        # Optional: apply edge weights to thickness
        for i, (_, row) in enumerate(knn_graph_embedding_edges.iterrows()):
            graph_2d_new.edges[edge_list[i]].set_stroke(width=row["weight"] * 5)

        self.play(ReplacementTransform(graph_2d, graph_2d_new))

        self.wait(2)

        self.play(
            *[
                graph_2d_new.edges[edge_list[i]].animate.set_stroke(width=0)
                for i in range(len(edge_list))
            ]
        )

        self.marked_next_slide()

        self.play(
            axes_2d.animate.set_opacity(0.0),
            *[
                vertex.animate.set_opacity(0.0)
                for vertex in graph_2d_new.vertices.values()
            ],
        )

        text = self.add_centered_text(
            "We can now think about clustering", max_width=0.5, font_size=48
        )

        self.marked_next_slide()
        self.play(FadeOut(text))

        self.play(
            axes_2d.animate.set_opacity(1.0),
            *[
                vertex.animate.set_opacity(1.0)
                for vertex in graph_2d_new.vertices.values()
            ],
        )
        # self.marked_next_slide()

        cluster0_animation = [
            graph_2d_new.vertices[idx].animate.set_color(ORANGE)
            for idx in (2, 8, 11, 14, 16, 17)
        ]
        cluster1_animation = [
            graph_2d_new.vertices[idx].animate.set_color(GREEN)
            for idx in (3, 4, 6, 7, 10, 15)
        ]
        cluster2_animation = [
            graph_2d_new.vertices[idx].animate.set_color(DARK_BLUE)
            for idx in (0, 1, 5, 9, 12, 13, 18, 19)
        ]

        self.play(
            *cluster0_animation, *cluster1_animation, *cluster2_animation, run_time=2
        )

        self.play(FadeOut(axes_2d))

        self.marked_next_slide()

        axes = ThreeDAxes(
            x_range=[-15, 15, 4],
            y_range=[-15, 15, 4],
            z_range=[-15, 15, 4],
            x_length=10,
            y_length=10,
            z_length=8,
        )

        # Set camera orientation for 3D view
        # self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)

        # Animate the axes appearing
        self.play(Create(axes))
        self.wait(0.5)

        # Create scatter points as Dot3D objects
        # Store them in a list so we can reference them later for edges
        colors = [
            DARK_BLUE,
            DARK_BLUE,
            ORANGE,
            GREEN,
            GREEN,
            DARK_BLUE,
            GREEN,
            GREEN,
            ORANGE,
            DARK_BLUE,
            GREEN,
            ORANGE,
            DARK_BLUE,
            DARK_BLUE,
            ORANGE,
            GREEN,
            ORANGE,
            ORANGE,
            DARK_BLUE,
            DARK_BLUE,
        ]
        scatter_dots = []
        for i, point in enumerate(knn_graph_embedding_raw_data):
            dot = Dot3D(
                point=axes.c2p(*point),  # Convert coordinates to scene position
                color=colors[i],
                radius=0.1,
                resolution=(16, 16),
            )
            scatter_dots.append(dot)

        vertex_transforms = [
            ReplacementTransform(graph_2d_new.vertices[i], scatter_dots[i])
            for i in vertices
        ]

        self.play(vertex_transforms)
        # Set camera orientation for 3D view
        self.move_camera(phi=75 * DEGREES, theta=30 * DEGREES, run_time=2)
        self.wait(1)
        # Slowly rotate the camera to show 3D nature
        self.begin_ambient_camera_rotation(rate=0.75)
        self.wait(3)
        self.stop_ambient_camera_rotation()

        # self.marked_next_slide()
        self.wait(0.1)

        self.marked_next_slide()

        self.clear_slide()
        self.move_camera(phi=0, theta=-90 * DEGREES)
        self.add_centered_text(
            "We can optimize every part of this to the specific task of clustering Embedding Vectors",
            max_width=0.75,
            max_height=0.66,
            t2c={
                "optimize": ACCENT_COLOR,
                "specific": ACCENT_COLOR,
                "task": ACCENT_COLOR,
                "clustering": ACCENT_COLOR,
                "Embedding": ACCENT_COLOR,
                "Vectors": ACCENT_COLOR,
            },
        )

        self.save_state("knn_embedding")


class DensityEstimation(ThreeDTIMCSlide):

    def _data_to_screen_distance(self, data_dist):
        point1 = self.graph.c2p(0, 0)
        point2 = self.graph.c2p(data_dist, 0)
        return np.linalg.norm(point2 - point1)

    def _dist(self, from_idx, to_idx):
        displacement = self.dots[from_idx].get_center() - self.dots[to_idx].get_center()
        return np.sqrt(np.sum(displacement**2))

    def construct(self):

        self.end_section_wipe("Density Clustering")

        self.add_centered_text("First we need density estimates!")

        self.marked_next_slide()
        self.clear_slide(run_time=1)

        scaled_base_data = SCALED_BASE_DATA

        self.graph = Axes(
            x_range=[-2, 12, 1],
            y_range=[-2, 12, 1],
            x_length=12,
            y_length=9,
            axis_config={"include_tip": True, "color": DEFAULT_COLOR},
        )
        labels = self.graph.get_axis_labels()
        self.graph.shift(0.25 * DOWN)
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

        self.play(
            LaggedStart(
                *[FadeIn(dot, scale=0.25) for dot in self.dots], lag_ratio=0.001
            ),
        )

        self.marked_next_slide()

        dmat = pairwise_distances(scaled_base_data)
        neighbors = np.argsort(dmat, axis=1)
        core_neighbors = neighbors[:, 5]
        chosen_example = np.argsort(order)[2296]  # np.argsort(core_distances)[-89]
        chosen_example_loc = self.graph.c2p(*base_data[chosen_example])
        chosen_distances = dmat[chosen_example]
        chosen_neighbors = np.argsort(chosen_distances)[:6]

        self.play(FadeIn(self.dots[chosen_example].scale(3), scale=3))
        self.play(self.logo.animate.set_opacity(0.0), run_time=0.25)
        self.move_camera(
            frame_center=np.array(self.dots[chosen_example].get_center()),
            zoom=3,
            run_time=1.0,
            rate_func=smooth,
        )

        vt = ValueTracker(1)

        def get_polar_point():
            val = vt.get_value()
            idx_start = int(val)
            idx_end = min(idx_start + 1, 5)
            alpha = val % 1  # The progress between neighbor i and i+1

            # Get positions relative to the center (chosen_example_loc)
            p1 = (
                self.graph.c2p(*base_data[chosen_neighbors[idx_start]])
                - chosen_example_loc
            )
            p2 = (
                self.graph.c2p(*base_data[chosen_neighbors[idx_end]])
                - chosen_example_loc
            )

            # Calculate Polar Components for p1
            r1 = np.linalg.norm(p1)
            theta1 = np.arctan2(p1[1], p1[0])

            # Calculate Polar Components for p2
            r2 = np.linalg.norm(p2)
            theta2 = np.arctan2(p2[1], p2[0])

            # Handle the "shortest path" for rotation to avoid 360-degree flips
            if theta2 - theta1 > np.pi:
                theta2 -= 2 * np.pi
            elif theta2 - theta1 < -np.pi:
                theta2 += 2 * np.pi

            # Interpolate radius and angle
            r_now = interpolate(r1, r2, alpha)
            theta_now = interpolate(theta1, theta2, alpha)

            # Convert back to Cartesian and add the center offset
            return chosen_example_loc + np.array(
                [r_now * np.cos(theta_now), r_now * np.sin(theta_now), 0]
            )

        # 2. Define the dynamic arrow
        neighbor_arrow = always_redraw(
            lambda: Arrow(
                start=chosen_example_loc,
                end=get_polar_point(),
                color=DEFAULT_COLOR,
                buff=0,
                stroke_width=3,
            )
        )

        # 3. Define the dynamic circle
        chosen_circle = always_redraw(
            lambda: Circle(
                # We calculate radius directly from the current polar point distance
                radius=np.linalg.norm(get_polar_point() - chosen_example_loc),
                color=DEFAULT_COLOR,
                stroke_width=2,
            ).move_to(chosen_example_loc)
        )

        self.add(neighbor_arrow, chosen_circle)

        # 4. Animate the transition
        for n in range(1, 6):
            self.play(
                vt.animate.set_value(n),
                run_time=1,
                rate_func=smooth,  # Use linear for smooth continuous movement
            )
            self.wait(0.25)

        self.wait(1)

        self.play(
            FadeOut(neighbor_arrow),
            chosen_circle.animate.set_stroke(width=0.5),
            self.dots[chosen_example].animate.scale(0.33),
        )
        chosen_circle.set_stroke(width=0.5)
        self.move_camera(
            frame_center=np.array([0, 0, 0]),
            zoom=1,
            run_time=1.0,
            rate_func=smooth,
        )
        self.play(
            self.logo.animate.set_opacity(1.0),
            chosen_circle.animate.set_stroke(width=0.5),
            run_time=0.25,
        )

        self.marked_next_slide()

        core_distances_all = []
        for i in range(len(self.dots)):
            # if i == 2258:
            #     continue
            j = neighbors[i, 6]
            core_distances_all.append(self._dist(i, j))

        # Normalize distances for colormap (0 to 1)
        min_dist = np.min(core_distances_all)
        max_dist = np.max(core_distances_all)

        core_circles = VGroup()
        for i, dot in enumerate(self.dots):
            # if i == 2258:
            #     continue
            j = neighbors[i, 6]
            circle = Circle(
                radius=self._dist(i, j),
                color=DEFAULT_COLOR,
                stroke_width=0.5,
            ).move_to(dot.get_center())
            core_circles.add(circle)

        self.play(
            FadeOut(chosen_circle),
            LaggedStart(
                *[GrowFromCenter(circ) for circ in core_circles],
                lag_ratio=0.0005,
            ),
        )

        self.marked_next_slide()  # or self.wait() if you want

        # Thicken the strokes
        self.play(
            *[
                circle.animate.set_stroke(
                    width=2,
                    color=colormap_color(
                        circle.get_radius(), min_dist, max_dist, power=4.0, invert=True
                    ),
                )
                for circle in core_circles
            ],
            run_time=1,
        )

        self.marked_next_slide()  # or self.wait()

        # Fade out circles while transitioning dot colors
        dot_color_anims = []
        circle_fade_anims = []

        idx = 0
        for i, dot in enumerate(self.dots):
            # if i == 2258:
            #     circle = chosen_circle
            # else:
            circle = core_circles[idx]
            idx += 1
            dot_color_anims.append(dot.animate.set_color(circle.get_stroke_color()))
            circle_fade_anims.append(FadeOut(circle))

        self.play(*dot_color_anims, *circle_fade_anims, run_time=1.5)

        self.marked_next_slide()

        self.play(self.dots.animate.set_opacity(0.0))

        text = self.add_centered_text(
            'Now we need organize density by "region"', max_width=0.66
        )

        self.marked_next_slide()

        self.play(FadeOut(text))

        self.wait(0.5)
        self.play(self.dots.animate.set_opacity(1.0))
        self.wait(0.5)

        ctree = scaled_ctree
        points_in_pdf_order = scaled_points_in_pdf_order
        pdf_order_of_points = scaled_pdf_order_of_points
        density_values = scaled_density_values

        density_axes = Axes(
            x_range=[0, 1],
            y_range=[0, 1],
            x_length=12,
            y_length=5,
            axis_config={"include_tip": True, "color": DEFAULT_COLOR},
        ).shift(UP)

        self.play(
            *[
                self.dots[dot_idx]
                .animate.move_to(density_axes.c2p(idx / base_data.shape[0], 0))
                .scale(0.1)
                for idx, dot_idx in enumerate(points_in_pdf_order)
            ]
        )

        density_axes.shift(DOWN)
        self.play(self.dots.animate.shift(DOWN))

        lines = VGroup()

        for idx in range(base_data.shape[0]):
            x_pos = idx / base_data.shape[0]
            line = Line(
                start=density_axes.c2p(x_pos, 0),
                end=density_axes.c2p(x_pos, density_values[idx]),
                stroke_width=0.5,
                color=self.dots[points_in_pdf_order[idx]].get_color(),
            )
            lines.add(line)

        self.play(*[Create(line) for line in lines])

        self.marked_next_slide()

        self.play(
            Write(
                tmp_text := Text(
                    "We get a density profile\ngrouped by dense regions"
                ).next_to(density_axes, UP, buff=-0.5)
            )
        )

        cluster_tree = ctree[ctree["child_size"] > 1]
        clusters = np.unique(np.hstack([cluster_tree["parent"], cluster_tree["child"]]))

        profiles = []
        sizes = []
        for c in clusters:
            profile, size = density_profile_for_cluster(
                ctree, c, np.array(points_in_pdf_order), lambda_scale=1.0
            )
            profiles.append(profile)
            sizes.append(size)

        local_cluster_density_profiles = np.vstack(profiles)
        local_cluster_sizes = np.asarray(sizes)

        polygons_list, polygon_sizes = _build_density_polygons(
            density_axes,
            local_cluster_density_profiles,
            local_cluster_sizes,
            np.linspace(0, 1, base_data.shape[0]),
            lambda idx: colormap_color(
                local_cluster_sizes[idx],
                2,
                base_data.shape[0],
                power=8.0,
                invert=True,
            ),
            positive_only_min=True,
        )
        polygons = VGroup(*polygons_list)

        self.marked_next_slide()
        self.play(FadeOut(tmp_text))
        self.play(
            Write(
                tmp_text := Paragraph(
                    "We can organize these\ninto hierarchical clusters",
                    alignment="center",
                ).next_to(density_axes, UP, buff=-0.5)
            )
        )

        self.marked_next_slide()
        self.play(FadeTransform(lines, polygons))

        self.save_state("sorting_density")


class EVoCLogoReveal(ThreeDTIMCSlide):

    def construct(self):

        self.load_state("use_case")
        self.wait(0.5)
        self.clear_slide()

        evoc_logo = SVGMobject(IMAGE_DIR / "evoc_logo_only.svg").scale(4)
        logo_data = np.load(EVOC_LOGO_DATA)
        labels = np.load(EVOC_LOGO_COLORS)

        logo_data[:, 0] /= logo_data.T[0].max() - logo_data.T[0].min()
        logo_data[:, 1] /= logo_data.T[1].max() - logo_data.T[1].min()
        logo_data -= logo_data.mean(axis=0)
        logo_data *= 3.5
        logo_data[:, 1] -= 0.05
        logo_data[labels == "#1f77b4", 1] -= 0.05
        logo_data[labels == "#2ca02c", 0] -= 0.05
        logo_data *= 2

        dots = VGroup()
        for loc in logo_data:
            pos = np.hstack([loc, np.array([0])])
            dots.add(Dot(pos, fill_opacity=0.75, radius=0.05, color=DEFAULT_COLOR))

        self.play(Create(dots), run_time=1)
        self.wait(1)
        # self.marked_next_slide()
        self.play(
            AnimationGroup(
                *[
                    DrawBorderThenFill(part, stroke_color=part.get_fill_color())
                    for part in evoc_logo
                ],
                lag_ratio=1.05,
            ),
            run_time=2,
        )
        self.play(FadeOut(dots), run_time=0.1)
        self.wait(1)
        self.play(evoc_logo.animate.scale(0.33).move_to((-2.5, 0, 0)))

        title = Text("EVōC", font_size=176).next_to(evoc_logo)
        self.play(Write(title))
        self.marked_next_slide()

        # Step 2: Create individual letter mobjects
        letters_data = [
            ("E", "mbedding"),
            ("V", "ector"),
            ("ō", "riented"),
            ("C", "lustering"),
        ]

        # Create separate letter objects
        E = Text("E", font_size=104)
        V = Text("V", font_size=104)
        o = Text("ō", font_size=104)
        C = Text("C", font_size=104)

        letters = VGroup(E, V, o, C).arrange(RIGHT, buff=0.5)
        letters.move_to(ORIGIN)
        o.shift(LEFT * 0.05)

        # Step 3: Stack vertically
        vertical_spacing = 1.75
        target_positions = [
            [0, vertical_spacing * 1.5, 0],
            [0, vertical_spacing * 0.5, 0],
            [0, -vertical_spacing * 0.5, 0],
            [0, -vertical_spacing * 1.5, 0],
        ]

        animations = []
        for letter, pos in zip(letters, target_positions):
            animations.append(letter.animate.move_to(pos))

        # Transform title into separate letters
        self.play(FadeOut(evoc_logo), ReplacementTransform(title, letters))
        self.play(*animations, run_time=1.5)
        self.wait(0.5)

        # Step 4: Create full words to get proper alignment reference
        # We'll create the full words, then extract positioning
        full_E = Text("Embedding", font_size=104)
        full_V = Text("Vector", font_size=104)
        full_o = Text("ōriented", font_size=104)
        full_C = Text("Clustering", font_size=104)

        # Create just the trailing parts
        embedding_rest = Text("mbedding", font_size=104)
        vector_rest = Text("ector", font_size=104)
        oriented_rest = Text("riented", font_size=104)
        clustering_rest = Text("lustering", font_size=104)

        # Calculate the shift needed to center
        temp_group = VGroup(
            full_E,
            full_V,
            full_o,
            full_C,
        )
        shift_left = -temp_group.get_left()[0] * LEFT

        # Animate everything together
        self.play(
            E.animate.shift(shift_left),
            V.animate.shift(shift_left),
            o.animate.shift(shift_left),
            C.animate.shift(shift_left),
        )
        # Position the rest aligned to the bottom (baseline) of the letters
        # Use align_to to match the bottom edge
        embedding_rest.next_to(E, RIGHT, buff=0.075)
        embedding_rest.align_to(E, UP)

        vector_rest.next_to(V, RIGHT, buff=-0.05)
        vector_rest.align_to(V, DOWN)

        oriented_rest.next_to(o, RIGHT, buff=0.075)
        oriented_rest.align_to(o, DOWN)

        clustering_rest.next_to(C, RIGHT, buff=0.075)
        clustering_rest.align_to(C, UP)

        self.play(
            Write(embedding_rest),
            Write(vector_rest),
            Write(oriented_rest),
            Write(clustering_rest),
            run_time=2,
        )
        # self.marked_next_slide()

        self.save_state("logo_intro")


def get_sorting_animations(icons, floor_y=-3.0, spacing=0.5, animate=True):
    # 1. Filter for icons that stayed on the floor
    good_objects = [mob for mob in icons if mob.get_center()[1] >= floor_y]

    # 2. Group by type
    # (Assumes you assigned icon.data_type during creation)
    groups = {"text": [], "img": [], "video": [], "audio": []}
    for mob in good_objects:
        groups[mob.data_type].append(mob)

    animations = []
    current_x = -5.5  # Starting x-position for the first group
    group_gap = 1.0  # Extra space between different types

    rng = np.random.RandomState(42)

    icon_h = 0.5

    for itype in ["text", "img", "video", "audio"]:
        mobs = groups[itype]
        if not mobs:
            continue

        # Calculate how many columns this specific type needs
        # e.g., 14 icons with limit 6 = 3 columns (6, 6, 2)
        num_mobs = len(mobs)

        column_height = rng.randint(4, 12)
        already_sorted = 0
        column = 0
        row = -1

        for i, mob in enumerate(mobs):
            if i - already_sorted <= column_height:
                row += 1
            else:
                column += 1
                row = 0
                already_sorted = i
                column_height = rng.randint(4, 12)

            target_x = current_x + (column * spacing)
            target_y = floor_y + (row * spacing) + (icon_h / 2)

            if animate:
                animations.append(
                    mob.animate.move_to([target_x, target_y, 0]).rotate(
                        -mob.current_angle
                    )
                )
            else:
                animations.append(
                    mob.move_to([target_x, target_y, 0]).rotate(-mob.current_angle)
                )
            mob.current_angle = 0.0

        # Move the cursor for the next group
        current_x += (column * spacing) + group_gap

    return animations


class TitleAndMotivation(ThreeDTIMCSlide):

    def construct(self):
        ## TITLE SLIDE
        logo = (
            SVGMobject(IMAGE_DIR / "evoc_logo_horizontal.svg")
            .scale(2.0)
            .shift(UP * 0.5)
        )
        venue = Text(
            "TMLS 2026, Toronto Canada", font_size=48, font="Marcellus SC"
        ).next_to(logo, DOWN, buff=1)
        speaker = Text(
            "Leland McInnes", color=ACCENT_COLOR, font_size=40, font="Marcellus SC"
        ).next_to(venue, DOWN)

        self.add(logo, venue, speaker)

        # EMbedding use case

        self.new_section("The Problem")

        self.add_centered_text("Modern data is not neatly organized database tables")

        self.marked_next_slide()

        self.clear_slide(run_time=1)

        # Create the intro layout
        intro_icons = Group()
        intro_labels = VGroup()
        types = ["text", "img", "video", "audio"]
        labels = ["Text Documents", "Image Files", "Video Clips", "Audio Tracks"]
        # Assign types for later sorting
        colors = {"text": YELLOW, "img": BLUE, "video": RED, "audio": GREEN}

        # Using a RoundedRect as a generic icon base
        icon_assets = {
            # icon_type: RoundedRectangle(
            #     corner_radius=0.1,
            #     height=0.5,
            #     width=0.4,
            #     fill_opacity=1,
            #     color=colors[icon_type],
            # )
            icon_type: SVGMobject(IMAGE_DIR / f"icons/{icon_type}.svg")
            for icon_type in types
        }

        for i, (itype, txt) in enumerate(zip(types, labels)):
            # Grab a copy of your existing icon assets
            icon = icon_assets[itype].copy()
            icon.data_type = itype
            icon.current_angle = 0.0
            label = Text(txt, font_size=32).next_to(icon, RIGHT, buff=0.5)

            # Arrange in a vertical stack
            group = VGroup(icon, label)
            group.shift(UP * (1.5 - i * 0.75)).shift(LEFT * 2)  # Spacing them out

            intro_icons.add(icon)
            intro_labels.add(label)

        # 1. Show the labels
        self.play(FadeIn(intro_icons), Write(intro_labels))
        self.marked_next_slide()

        # 2. Fade out labels and prepare icons for the drop
        self.play(FadeOut(intro_labels))

        with open(ICON_DELUGE_SIMULATION, "r") as f:
            sim_data = json.load(f)

        icons = VGroup()
        for icon in intro_icons:
            icons.add(icon)
        for i in range(len(sim_data) - 4):
            icon_type = np.random.choice(
                ["text", "img", "video", "audio"], p=[0.5, 0.25, 0.125, 0.125]
            )

            icon = SVGMobject(IMAGE_DIR / f"icons/{icon_type}.svg")
            icon.data_type = icon_type
            icon.current_angle = 0.0
            icon.set_opacity(0)  # Hide initially
            icons.add(icon)

        self.add(icons)
        playhead = ValueTracker(0)

        def update_icons(mobs):
            current_frame = int(playhead.get_value())
            for i, mob in enumerate(mobs):
                if i < 4:
                    # They are always visible
                    mob.set_opacity(1)
                    # Use path data. If the deluge hasn't started (frame 0),
                    # they stay at their intro positions precomputed in Pymunk.
                    path_idx = current_frame
                    if path_idx < len(sim_data[i]["path"]):
                        x, y, angle, _ = sim_data[i]["path"][path_idx]
                        mob.move_to([x / 100, y / 100, 0])
                        angle_diff = angle - mob.current_angle
                        mob.rotate(angle_diff)
                        mob.current_angle = angle
                else:
                    spawn_f = sim_data[i]["spawn_frame"]

                    if current_frame < spawn_f:
                        mob.set_opacity(0)
                    else:
                        mob.set_opacity(1)
                        # Index into the path using (current_frame - spawn_frame)
                        path_idx = current_frame - spawn_f
                        if path_idx < len(sim_data[i]["path"]):
                            x, y, angle, _ = sim_data[i]["path"][path_idx]
                            mob.move_to([x / 100, y / 100, 0])
                            angle_diff = angle - mob.current_angle
                            mob.rotate(angle_diff)
                            mob.current_angle = angle

        icons.add_updater(update_icons)

        n_frames = np.max([len(item["path"]) for item in sim_data])
        # Adjust run_time to match the duration in the physics script
        self.play(
            playhead.animate.set_value(n_frames),
            run_time=n_frames / 30,
            rate_func=linear,
        )
        icons.remove_updater(update_icons)

        self.marked_next_slide()

        text = Paragraph("Embeddings provide similarities among items").shift(UP * 1.5)
        self.play(Write(text))

        self.marked_next_slide()

        self.play(FadeOut(text))

        text = Paragraph(
            "We can use that to sort\nthe mess into organized piles...",
            alignment="center",
        ).shift(UP * 1.5)
        self.play(Write(text))

        self.marked_next_slide()

        self.play(FadeOut(text))

        good_objects = [mob for mob in icons if mob.get_center()[1] >= -3.0]

        print(len(good_objects))
        for mob in icons:
            if mob.data_type == "text":
                mob.set_z_index(4)
            elif mob.data_type == "img":
                mob.set_z_index(3)
            elif mob.data_type == "video":
                mob.set_z_index(2)
            elif mob.data_type == "audio":
                mob.set_z_index(1)

        sort_anims = get_sorting_animations(icons, animate=True)

        self.play(
            LaggedStart(*sort_anims, lag_ratio=0.05),
            run_time=3,
            rate_func=bezier([0, 0, 1, 1]),  # Starts slow, ends snappy
        )
        self.marked_next_slide()

        self.play(FadeOut(icons))

        self.add_centered_text("And now we can browse the piles for insights")

        self.marked_next_slide()

        self.clear_slide()

        self.add_centered_text(
            "But how do we make piles?\nWhat is the right scale of pile?\nDoes everything belong in a pile?",
            font_size=60,
        )

        # self.next_slide()
        self.save_state("use_case")


class PipelineOverview(ThreeDTIMCSlide):

    def construct(self):

        self.load_state("logo_intro")

        self.new_section("Clustering\nEmbedding\nVectors")

        self.add_centered_text("Embedding Vectors are very high dimensional")

        self.marked_next_slide()

        self.clear_slide(run_time=1.0)

        self.add_centered_text("Clustering high dimensional data is difficult")

        self.marked_next_slide()

        self.clear_slide(run_time=1.0)

        self.play(Write(Text("The Process", font_size=56).shift(UP * 2.5)))

        self.marked_next_slide()

        # --- 1. THE DATA STRUCTURES ---
        # A. Hypercube Embeddings
        # --- Setup Layout ---
        # Create placeholders to define where each stage lives on screen
        stage_1_placeholder = VectorizedPoint().shift(LEFT * 4.5)

        # --- Create Stage 1 ---
        cube = Cube(side_length=1.5)
        cube.set_stroke(GRAY, 1)
        cube.set_fill(WHITE, opacity=0.05)

        rng = np.random.RandomState(42)

        dots_cube = VGroup(
            *[
                Dot3D(
                    point=[rng.uniform(-0.6, 0.6) for _ in range(3)],
                    radius=0.03,
                    color=DEFAULT_COLOR,
                ).set_opacity(
                    rng.uniform(0.3, 1.0)
                )  # Simplified depth hint
                for _ in range(40)
            ]
        )

        stage_1 = VGroup(cube, dots_cube)
        stage_1.move_to(stage_1_placeholder)  # Fixes it in the "left slot"

        # B. UMAP Blobs
        manifold_axes = Axes(
            x_range=[0, 1.2, 1],
            y_range=[0, 1, 1],
            x_length=2,
            y_length=1.5,
            axis_config={
                "include_tip": True,
                "include_ticks": False,
                "tip_shape": StealthTip,
                "tip_height": 0.15,
                "tip_width": 0.15,
            },
        )
        blob_centers = [
            LEFT * 0.3 + UP * 0.2,
            RIGHT * 0.4,
            DOWN * 0.6,
            LEFT + 0.4 + DOWN * 0.3,
        ]
        dots_blobs = VGroup(
            *[
                Dot(
                    rng.normal(c, 0.1),
                    radius=0.04,
                    color=ACCENT_COLOR,
                ).set_opacity(0.75)
                for c in blob_centers
                for _ in range(15)
            ]
        )
        stage_2 = VGroup(manifold_axes, dots_blobs)

        # C. Reachability Plot (Cartoon version)
        axes = Axes(
            x_range=[0, 1.2, 1],
            y_range=[0, 1, 1],
            x_length=2,
            y_length=1.5,
            axis_config={
                "include_tip": True,
                "include_ticks": False,
                "tip_shape": StealthTip,
                "tip_height": 0.15,
                "tip_width": 0.15,
            },
        )
        ctree = scaled_ctree
        points_in_pdf_order = scaled_points_in_pdf_order
        pdf_order_of_points = scaled_pdf_order_of_points
        density_values = scaled_density_values

        lines = VGroup()

        max_density = density_values.max()
        min_density = density_values.min()

        for idx in range(base_data.shape[0]):
            x_pos = idx / base_data.shape[0]
            line = Line(
                start=axes.c2p(x_pos, 0),
                end=axes.c2p(x_pos, density_values[idx]),
                stroke_width=0.5,
                color=colormap_color(density_values[idx], min_density, max_density),
            ).set_opacity(0.0)
            lines.add(line)

        stage_3 = VGroup(axes, lines)

        # 1. Adjust the Matrix for a "flatter" look
        # Lowering the [1][1] value or stretching the Y axis down makes it flatter
        shear_matrix = [[1, 1.2, 0], [0, 0.6, 0], [0, 0, 1]]

        # 2. Define persistent centers (to ensure hierarchy/no overlaps)
        # These are the "seeds" for the clusters
        base_centers = [
            [-0.7, -0.2, 0],
            [0.7, 0.2, 0],  # Group A
            [-0.1, 0.3, 0],
            [0.2, -0.3, 0],  # Group B
            [0.8, -0.2, 0],  # Group C
        ]

        layers = VGroup()
        for i in range(3):
            # Create the base sheet
            sheet = Rectangle(width=3.0, height=1.5)
            sheet.set_stroke(GRAY, 1)
            sheet.set_fill(interpolate_color(WHITE, GRAY, 0.15), opacity=0.9)

            # Determine which centers to keep for this "resolution" layer
            # Layer 0: all, Layer 1: subset, Layer 2: one large "global" cluster
            if i == 0:
                current_centers = base_centers
                r = 0.15
            elif i == 1:
                current_centers = [base_centers[0], base_centers[2], base_centers[4]]
                r = 0.3
            else:
                current_centers = [[0, 0, 0]]
                r = 0.6

            clusters = VGroup(
                *[
                    Circle(
                        radius=r,
                        color=interpolate_color(ACCENT_COLOR, DEFAULT_COLOR, i / 2.5),
                        fill_opacity=0.4,
                        stroke_width=2,
                    ).move_to(loc)
                    for loc in current_centers
                ]
            )

            layer = VGroup(sheet, clusters)

            # Apply the flattening transformation
            layer.apply_matrix(shear_matrix)

            # Stack them with decreasing opacity to give "depth"
            layer.set_opacity(1.0 - (i * 0.15))
            # layer.move_to(stage_4_placeholder)
            # layer.shift(UP * i * 0.6)
            layers.add(layer)

        # 2. Create an invisible 'Space Holder'
        # This should match the total height of your FINAL spread (e.g., sheet height + total shift)
        proxy_box = Rectangle(
            width=3.0,
            height=1.0 + (2 * 0.33),  # Sheet height + total UP shift
            stroke_opacity=0,
            fill_opacity=0,
        )
        proxy_box.align_to(layers[0], DOWN)
        stage_4 = VGroup(proxy_box, layers.scale(0.6))

        # --- 2. LAYOUT ---
        pipeline = VGroup(stage_1, stage_2, stage_3, stage_4).arrange(RIGHT, buff=1.5)
        pipeline.set_width(config.frame_width * 0.9)

        # Labels
        labels = VGroup(
            Text("Embeddings", font_size=20).next_to(stage_1, DOWN),
            Text("Manifold", font_size=20).next_to(stage_2, DOWN),
            Text("Density", font_size=20).next_to(stage_3, DOWN),
            Text("Multiscale Clusters", font_size=20).next_to(stage_4, DOWN),
        )

        # --- 3. ANIMATION STEPS ---

        # --- Animation ---
        self.play(FadeIn(stage_1), run_time=1.0)

        # Rotate around its own center, not the screen origin
        self.play(
            Rotate(
                stage_1,
                angle=2 * PI,
                axis=np.array([1, 1, 1]),  # Diagonal axis for a "dynamic" spin
                about_point=stage_1.get_center(),
                run_time=1.5,
                rate_func=smooth,
            ),
            Write(labels[0]),
        )
        self.marked_next_slide()

        # Reveal next stage with an arrow
        arrow1 = Arrow(stage_1.get_right(), stage_2.get_left(), buff=0.1)
        self.play(
            GrowArrow(arrow1),
            ReplacementTransform(stage_1[1].copy(), dots_blobs.scale(1.1)),
            Write(labels[1]),
        )
        self.marked_next_slide()

        arrow2 = Arrow(stage_2.get_right(), stage_3.get_left(), buff=0.1)
        self.play(GrowArrow(arrow2), Create(axes), Write(labels[2]))
        for line in lines:
            line.set_opacity(1.0)
        self.play(*[Create(line) for line in lines])
        self.marked_next_slide()

        arrow3 = Arrow(stage_3.get_right(), stage_4.get_left(), buff=0.1)
        self.play(GrowArrow(arrow3), FadeIn(stage_4))
        self.play(
            layers[1].animate.shift(UP * 0.33).set_rate_func(rush_into),
            layers[2].animate.shift(UP * 0.66).set_rate_func(rush_into),
            Write(labels[3]),
        )

        self.wait(0.1)
        self.marked_next_slide()

        # Curved annotation arrows
        _create_curved_annotation(
            self,
            "Manifold learning\n(e.g. UMAP)",
            LEFT * 4.5,
            arrow1,
            (UP * 1.5 + LEFT * 0.5, DOWN + RIGHT * 0.5),
            18 * DEGREES,
        )
        self.marked_next_slide()

        _create_curved_annotation(
            self,
            "Density Clustering\n(e.g. HDBSCAN)",
            LEFT * 0.1 + UP * 0.5,
            arrow2,
            (UP * 1 + RIGHT * 0.15, DOWN + LEFT * 0.15),
            -18 * DEGREES,
        )
        self.marked_next_slide()

        _create_curved_annotation(
            self,
            "Resolution Persistence\n(e.g. PLSCAN)",
            RIGHT * 4.5,
            arrow3,
            (UP * 1.25 + LEFT * 0.25, DOWN + LEFT * 0.25),
            -18 * DEGREES,
        )

        stage_centers = {
            "embeddings": stage_1.get_center().tolist(),
            "manifold": stage_2.get_center().tolist(),
            "density": stage_3.get_center().tolist(),
            "clusters": stage_4.get_center().tolist(),
        }
        with open(os.path.join(self.state_dir, "stage_centers.json"), "w") as f:
            json.dump(stage_centers, f)

        self.save_state("overview")


class EVoCBenefits(TIMCSlide):

    def construct(self):

        self.load_state("persistence_scores")

        self.new_section("EVōC")

        self.add_centered_text("EVōC produces high quality clusters")

        self.marked_next_slide()

        self.clear_slide(run_time=1.0)

        self.add_centered_text("EVōC requires very little parameter tuning")

        self.marked_next_slide()

        self.clear_slide(run_time=1.0)

        self.add_centered_text("EVōC runs very quickly")

        self.save_state("benefits")


class Summary(TIMCSlide):

    def construct(self):

        self.load_state("evoc_performance")

        self.new_section("Summary")

        self.add_centered_text(
            "Fast\nEasy\nEffective\nClustering\nfor\nEmbedding Vectors"
        )

        self.marked_next_slide()

        self.clear_slide(run_time=1.0)

        logo = (
            SVGMobject(IMAGE_DIR / "evoc_logo_horizontal.svg")
            .scale(2.0)
            .shift(RIGHT * 3)
        )

        # self.play(Create(logo))
        print(list(logo))
        parts = logo[:3] + VGroup(logo[3:8]) + VGroup(logo[8:-2]) + logo[-2:]
        # self.play(
        #     AnimationGroup(
        #         *[
        #             DrawBorderThenFill(part, stroke_color=part.get_fill_color())
        #             for part in parts
        #         ],
        #         lag_ratio=1.05,
        #     ),
        #     run_time=6,
        # )
        self.play(
            AnimationGroup(
                *[
                    DrawBorderThenFill(part, stroke_color=part.get_fill_color())
                    for part in parts[:3]
                ],
                lag_ratio=1.05,
            ),
            run_time=2,
        )
        for part in parts[3:]:
            part.set_opacity(0.0)
        self.play(logo.animate.shift(LEFT * 3))
        parts[3].set_opacity(1.0)
        self.play(Write(parts[3]))
        parts[4].set_opacity(1.0)
        self.play(Write(parts[4]))
        for part in parts[5:]:
            part.set_opacity(1.0)
        self.play(
            DrawBorderThenFill(parts[5:], stroke_width=0.25, stroke_color=ACCENT_COLOR),
            run_time=1,
        )

        self.marked_next_slide()

        url = Text("https://github.com/TutteInstitute/evoc", font_size=32).next_to(
            logo, DOWN, buff=1
        )
        install = Text("pip install evoc", font_size=32).next_to(url, DOWN, buff=0.1)

        self.play(FadeIn(url))
        self.play(FadeIn(install))


class TransitionToManifold(ThreeDTIMCSlide, PhaseSlide):

    def construct(self):
        self.load_state("overview")
        self._pan_to_stages("manifold")
        self.start_section_wipe("Manifold Learning")


class TransitionToClusters(ThreeDTIMCSlide, PhaseSlide):

    def construct(self):
        self.load_state("sorting_density")
        self.transition_to_overview()
        self._pan_to_stages("clusters")
        self.start_section_wipe("Cluster Extraction")


class TransitionToDensity(ThreeDTIMCSlide, PhaseSlide):

    def construct(self):
        self.load_state("knn_embedding")
        self.transition_to_overview()
        self._pan_to_stages("density")
        self.start_section_wipe("Density Clustering")
