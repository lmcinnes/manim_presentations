"""Where next: the pivot to ongoing research, and noise in high dimensions.

python assets_where_next.py        # once, writes assets/where_next.*
manim-slides render where_next.py WhereNext
"""

from manim import *

import json
import sys
from pathlib import Path

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

from umap_talk_common import *

apply_umap_defaults()

ASSETS = Path(__file__).parent / "assets"
COLUMN_TOP_LEFT = (2.9, 2.45)

# Section-card titles for the research part. Newer copies of umap_talk_common
# carry these in SECTION_TITLES; this keeps the classes working without them.
RESEARCH_TITLES = {
    "future": "Where Next",
    "repair": "Repairing the Graph",
    "kernel": "The Kernel as Dials",
}


def section_title(key):
    return SECTION_TITLES.get(key, RESEARCH_TITLES[key])


GENUINE_COLOR = STRUCTURE_COLOR
NEAR_MISS_COLOR = darken_for_contrast(COLOR_CYCLE[5], 2.4)  # cyan, distinct from navy
WRONG_COLOR = REPEL_COLOR
LOOP_COLORS = [
    COLOR_CYCLE[i] for i in (0, 2, 1, 3, 4)
]  # colour = position along a loop
FAR_COLOR = "#c9d1e3"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def to3(p):
    return np.array([p[0], p[1], 0.0])


def caption(text, font_size=22, color=DEFAULT_COLOR, **kwargs):
    return crisp_text(text, font_size=font_size, color=color, **kwargs)


def caption_stack(lines, font_size=22, buff=0.1, color=DEFAULT_COLOR):
    return crisp_lines(lines, font_size, buff=buff, color=color)


class Column:
    """Stacks captions top-down in the right-hand column."""

    def __init__(self, top_left=COLUMN_TOP_LEFT, max_width=None):
        self.top_left = to3(top_left)
        self.max_width = max_width or (config.frame_width / 2 - 0.35 - self.top_left[0])
        self.items = []

    def place(self, mob, buff=0.35):
        if mob.width > self.max_width:
            mob.scale_to_fit_width(self.max_width)
        if not self.items:
            mob.move_to(self.top_left, aligned_edge=UL)
        else:
            mob.next_to(self.items[-1], DOWN, buff=buff, aligned_edge=LEFT)
        self.items.append(mob)
        return mob


def fit_points(P, width, height, center):
    """Map 2D data points into a box, keeping the aspect ratio."""
    P = np.asarray(P, dtype=float)
    lo, hi = P.min(0), P.max(0)
    scale = min(width / max(hi[0] - lo[0], 1e-9), height / max(hi[1] - lo[1], 1e-9))
    return (P - (lo + hi) / 2) * scale + np.asarray(center, dtype=float)[:2]


def segment_cloud(A, B, color, width, opacity):
    """Many separate line segments as one VMobject: far quicker to draw and
    animate than thousands of Line objects."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    if A.shape[1] == 2:
        A, B = np.c_[A, np.zeros(len(A))], np.c_[B, np.zeros(len(B))]
    pts = np.empty((4 * len(A), 3))
    pts[0::4], pts[1::4], pts[2::4], pts[3::4] = (
        A,
        A + (B - A) / 3,
        A + 2 * (B - A) / 3,
        B,
    )
    vm = VMobject(
        stroke_color=color, stroke_width=width, stroke_opacity=opacity, fill_opacity=0
    )
    if len(A):
        vm.set_points(pts)
    return vm


def digit_image(pixels, color, height=2.0, upscale=10):
    """A 28x28 greyscale digit as an image: ink in ``color``, clear background."""
    ink = np.kron(
        np.asarray(pixels, np.uint8).reshape(28, 28),
        np.ones((upscale, upscale), np.uint8),
    )
    rgba = np.zeros(ink.shape + (4,), np.uint8)
    rgba[..., :3] = (np.array(ManimColor(color).to_rgb()) * 255).astype(np.uint8)
    rgba[..., 3] = ink
    img = ImageMobject(rgba)
    img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    img.height = height
    return img


def loop_color(u):
    x = (u % 1.0) * len(LOOP_COLORS)
    i = int(np.floor(x))
    return interpolate_color(
        ManimColor(LOOP_COLORS[i]),
        ManimColor(LOOP_COLORS[(i + 1) % len(LOOP_COLORS)]),
        x - i,
    )


def angular_gap(a, b):
    d = np.abs(a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


def undirected(edges, kind=None):
    """Unique undirected pairs; with ``kind``, each pair keeps its worst kind."""
    pairs = np.sort(np.asarray(edges), axis=1)
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    pairs = pairs[order]
    keep = np.r_[True, np.any(pairs[1:] != pairs[:-1], axis=1)]
    if kind is None:
        return pairs[keep]
    kind = np.asarray(kind)[order]
    group = np.cumsum(keep) - 1
    worst = np.zeros(keep.sum(), dtype=int)
    np.maximum.at(worst, group, kind)
    return pairs[keep], worst


def darker(color, amount=0.4):
    return interpolate_color(ManimColor(color), ManimColor(DEFAULT_COLOR), amount)


# ---------------------------------------------------------------------------
# The slide class
# ---------------------------------------------------------------------------
class WhereNext(UMAPSlide):
    # Showcase pairs: edge numbers from assets/where_next_candidates.png, in the
    # order to show them. None uses the assets script's two default picks.
    SHOWCASE_EDGES = [19908, 257, 17350]

    def construct(self):
        self.data = np.load(ASSETS / "where_next.npz")
        self.meta = json.loads((ASSETS / "where_next.json").read_text())
        self.end_section_wipe(
            section_title("future"),
            next_slide_prep=lambda: None,
            notes=NOTES["w01_map"],
        )
        self.stage_mnist()
        self.stage_circle()
        self.stage_distances()
        self.stage_anatomy()
        self.stage_spiral()
        self.stage_four_kinds()
        self.stage_takeaway()

    # -- pivot: MNIST's pixel-space nearest neighbours ------------------------------
    def stage_mnist(self):
        d, m = self.data, self.meta["mnist"]
        labels = d["mnist_labels"]
        P = fit_points(d["mnist_layout"], 7.0, 5.6, (0.0, -0.4))
        cloud = EmbeddingCloud(
            P, labels_to_rgb(labels, digit_rgb()), fit="none", point_px=3.0
        )
        pairs = d["mnist_pairs"]
        long = d["mnist_long"].astype(bool)
        A, B = P[pairs[:, 0]], P[pairs[:, 1]]
        short_edges = segment_cloud(A[~long], B[~long], STRUCTURE_COLOR, 0.5, 0.22)
        long_edges = segment_cloud(A[long], B[long], STRUCTURE_COLOR, 0.5, 0.22)

        self.set_title("UMAP of MNIST")
        self.add(cloud)
        self.play(PMFadeIn(cloud), run_time=1.5)
        self.marked_next_slide(notes=NOTES["w02_edges"])

        self.set_title("Each digit's nearest neighbour in pixel space")
        stats = caption(
            f"{m['edges']:,} edges from {m['n']:,} digits", 20, color=SECONDARY_COLOR
        )
        stats.to_edge(DOWN, buff=0.3)
        self.play(
            PMDim(cloud, 0.7),
            FadeIn(short_edges),
            FadeIn(long_edges),
            FadeIn(stats),
            run_time=2,
        )
        self.marked_next_slide(notes=NOTES["w03_long"])

        self.set_title("Some of them jump across the map")
        stats_long = caption(
            f"{m['long']:,} long edges join different regions of the map",
            20,
            color=SECONDARY_COLOR,
        ).to_edge(DOWN, buff=0.3)
        self.play(FadeOut(stats), FadeOut(short_edges), run_time=0.8)
        self.play(
            long_edges.animate.set_stroke(width=1.1, opacity=0.6), FadeIn(stats_long)
        )
        self.marked_next_slide(notes=NOTES["w04_pair"])

        shown = None
        candidates = [int(c) for c in d["mnist_candidates"]]
        showcase = list(
            self.SHOWCASE_EDGES or [entry["edge"] for entry in m["showcase"]]
        )
        for e in showcase:
            if e not in candidates:
                raise ValueError(
                    f"edge {e} is not on the candidate sheet (assets/where_next_candidates.png)"
                )
        for n, e in enumerate(showcase):
            images = d["mnist_candidate_images"][candidates.index(e)]
            i, j = int(pairs[e, 0]), int(pairs[e, 1])
            ends = [(i, 0), (j, 1)]
            if P[i, 0] > P[j, 0]:
                ends = ends[::-1]
            if shown is not None:
                self.play(FadeOut(shown))
            names = [int(labels[idx]) for idx, _ in ends]

            def article(digit):
                return "an" if digit == 8 else "a"

            self.set_title(
                f"{article(names[0]).capitalize()} {names[0]} and {article(names[1])} {names[1]}, "
                "nearest neighbours"
            )
            edge = Line(
                to3(P[ends[0][0]]),
                to3(P[ends[1][0]]),
                color=HIGHLIGHT_COLOR,
                stroke_width=5,
            )
            parts = [edge]
            for side, (idx, end) in zip((-1, 1), ends):
                digit = int(labels[idx])
                colour = digit_rgb()[digit]
                dot = Dot(
                    to3(P[idx]),
                    radius=0.09,
                    color=ManimColor(colour),
                    stroke_color=STRUCTURE_COLOR,
                    stroke_width=2,
                )
                img = digit_image(images[end], darker(ManimColor(colour)))
                img.move_to(to3((side * 5.45, -0.3)))
                frame = SurroundingRectangle(
                    img, buff=0.12, color=ManimColor(colour), stroke_width=3
                )
                name = caption(
                    f"{'an' if digit == 8 else 'a'} {digit}",
                    24,
                    color=darker(ManimColor(colour)),
                ).next_to(frame, DOWN, buff=0.15)
                link = DashedLine(
                    to3(P[idx]),
                    frame.get_right() if side < 0 else frame.get_left(),
                    color=SECONDARY_COLOR,
                    stroke_width=2,
                    dash_length=0.08,
                )
                parts += [dot, link, Group(frame, img, name)]
            group = Group(*parts)
            self.play(Create(edge), FadeIn(parts[1]), FadeIn(parts[4]), run_time=0.8)
            self.play(
                Create(parts[2]),
                Create(parts[5]),
                FadeIn(parts[3], shift=RIGHT * 0.2),
                FadeIn(parts[6], shift=LEFT * 0.2),
            )
            shown = group
            self.marked_next_slide(
                notes=NOTES["w05_pair2" if n < len(showcase) - 1 else "w06_lesson"]
            )

        self.play(
            FadeOut(shown), FadeOut(stats_long), FadeOut(long_edges), PMFadeOut(cloud)
        )
        self.remove(cloud)
        self.set_title("The graph we lay out has wrong edges")
        lesson = VGroup(
            caption_stack(
                [
                    "Pixel-space neighbours can belong to different digits,",
                    "and the layout has already overruled some of them.",
                ],
                font_size=28,
                buff=0.14,
            ),
            caption(
                "How wrong is the graph, and can the layout repair it?",
                28,
                color=STRUCTURE_COLOR,
            ),
        ).arrange(DOWN, buff=0.6)
        lesson.move_to(to3((0, -0.2)))
        self.play(FadeIn(lesson))
        self.marked_next_slide(notes=NOTES["w07_circle"])

    # -- a circle with noise in many hidden directions --------------------------------
    def stage_circle(self):
        self.clear_slide()
        self.set_title("Noise off the data rewires the graph")
        d, meta = self.data, self.meta["circle"]
        theta, nn, across, sigmas = (
            d["circle_theta"],
            d["circle_nn"],
            d["circle_across"],
            d["circle_sigmas"],
        )
        centre, radius = np.array([-2.4, -0.35]), 2.45
        pos = centre + radius * np.c_[np.cos(theta), np.sin(theta)]
        order = np.argsort(theta)
        colours = {
            int(i): loop_color(rank / len(theta)) for rank, i in enumerate(order)
        }
        dots = VGroup(
            *[
                Dot(to3(pos[i]), radius=0.085, color=colours[i])
                for i in range(len(theta))
            ]
        )
        ghost = Circle(
            radius=radius, color=SECONDARY_COLOR, stroke_width=1.2, stroke_opacity=0.5
        ).move_to(to3(centre))
        step = ValueTracker(0.0)

        def edges_at():
            s = int(round(step.get_value()))
            local, far = [], []
            for a, b in {(min(i, int(j)), max(i, int(j))) for i, j in enumerate(nn[s])}:
                (far if angular_gap(theta[a], theta[b]) > PI / 2 else local).append(
                    (a, b)
                )
            group = VGroup(
                segment_cloud(
                    pos[[a for a, _ in local]],
                    pos[[b for _, b in local]],
                    STRUCTURE_COLOR,
                    3,
                    0.9,
                ),
                segment_cloud(
                    pos[[a for a, _ in far]],
                    pos[[b for _, b in far]],
                    WRONG_COLOR,
                    4,
                    0.95,
                ),
            )
            return group

        edges = always_redraw(edges_at)
        col = Column()
        intro = col.place(
            caption_stack(
                [
                    "48 points on a circle, with noise",
                    "in 400 extra directions.",
                    "Drawn at their true positions.",
                ]
            )
        )
        per_dir = col.place(
            MetricReadout(
                "noise per direction", 0.0, num_decimal_places=2, font_size=22
            ),
            buff=0.45,
        )
        total = col.place(
            MetricReadout(
                "total noise \u00f7 radius", 0.0, num_decimal_places=1, font_size=22
            ),
            buff=0.15,
        )
        count = col.place(
            MetricReadout(
                "edges across the circle",
                0,
                num_decimal_places=0,
                font_size=22,
                number_color=WRONG_COLOR,
            ),
            buff=0.15,
        )
        key = col.place(
            VGroup(
                VGroup(
                    Line(ORIGIN, RIGHT * 0.45, color=STRUCTURE_COLOR, stroke_width=3),
                    caption("nearest neighbour along the circle", 17),
                ).arrange(RIGHT, buff=0.12),
                VGroup(
                    Line(ORIGIN, RIGHT * 0.45, color=WRONG_COLOR, stroke_width=4),
                    caption("nearest neighbour across the circle", 17),
                ).arrange(RIGHT, buff=0.12),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.12),
            buff=0.4,
        )
        scale = np.sqrt(meta["noise_dims"])

        self.play(
            Create(ghost),
            LaggedStart(*[FadeIn(dt, scale=0.5) for dt in dots], lag_ratio=0.02),
            run_time=1.5,
        )
        self.add(edges)
        self.bring_to_front(dots)
        self.play(
            FadeIn(intro), FadeIn(key), FadeIn(per_dir), FadeIn(total), FadeIn(count)
        )
        self.marked_next_slide(notes=NOTES["w08_sweep"])

        per_dir.number.add_updater(
            lambda m: m.set_value(sigmas[int(round(step.get_value()))])
        )
        total.number.add_updater(
            lambda m: m.set_value(sigmas[int(round(step.get_value()))] * scale)
        )
        count.number.add_updater(
            lambda m: m.set_value(int(across[int(round(step.get_value()))]))
        )
        self.play(
            step.animate.set_value(len(sigmas) - 1), run_time=10, rate_func=linear
        )
        for mob in (edges, per_dir.number, total.number, count.number):
            mob.clear_updaters()
        punch = col.place(
            caption_stack(
                [
                    f"Each point's noise is {sigmas[-1] * scale:.0f}\u00d7 the radius in",
                    f"total, but only {sigmas[-1]:.1f}\u00d7 in any one direction.",
                ],
                color=STRUCTURE_COLOR,
            ),
            buff=0.4,
        )
        self.play(FadeIn(punch))
        self.circle_final = (pos, theta, nn[-1])
        self.marked_next_slide(notes=NOTES["w10_sweep"])

    # -- distances from one point as the dimension grows ---------------------------
    def stage_distances(self):
        self.clear_slide()
        self.set_title("Distances bunch up; their order survives")
        d = self.data
        dims, dist = d["dist_dims"], d["dist_values"]
        degrees = np.degrees(d["dist_true_gap"])
        contrast, agreement = d["dist_contrast"], d["dist_agreement"]
        x_max = float(np.ceil(dist.max()))
        chart = styled_axes(
            [0, x_max, 1],
            [0, 180, 45],
            "observed distance",
            "true separation (degrees)",
            x_length=7.2,
            y_length=4.6,
            y_decimal_places=0,
        )
        chart.move_to(to3((-2.2, -0.45)))
        axes = chart[0]
        for v in range(1, int(x_max) + 1):
            chart.add(caption(str(v), 16).next_to(axes.c2p(v, 0), DOWN, buff=0.18))
        step = ValueTracker(0.0)
        colours = [
            interpolate_color(
                ManimColor(STRUCTURE_COLOR), ManimColor(FAR_COLOR), g / 180
            )
            for g in degrees
        ]

        def dots_at():
            t = float(np.clip(step.get_value(), 0, len(dims) - 1))
            i = int(np.floor(t))
            j = min(i + 1, len(dims) - 1)
            f = t - i
            xs = (1 - f) * dist[i] + f * dist[j]
            return VGroup(
                *[
                    Dot(axes.c2p(x, g), radius=0.06, color=c)
                    for x, g, c in zip(xs, degrees, colours)
                ]
            )

        dots = always_redraw(dots_at)
        col = Column((3.0, 2.2))
        intro = col.place(
            caption_stack(
                [
                    "From one point on the circle,",
                    "the distance to every other point,",
                    "as noise spreads over more",
                    "directions (0.15 per direction).",
                ]
            )
        )
        n_dims = col.place(
            MetricReadout("noise directions", 1, num_decimal_places=0, font_size=22),
            buff=0.45,
        )
        ratio = col.place(
            MetricReadout(
                "farthest \u00f7 nearest",
                float(contrast[0]),
                num_decimal_places=2,
                font_size=22,
            ),
            buff=0.15,
        )
        ranks = col.place(
            MetricReadout(
                "rank agreement",
                float(agreement[0]),
                num_decimal_places=2,
                font_size=22,
            ),
            buff=0.15,
        )

        self.play(FadeIn(chart), FadeIn(intro))
        self.add(dots)
        self.play(FadeIn(n_dims), FadeIn(ratio), FadeIn(ranks))
        self.marked_next_slide(notes=NOTES["w11_anatomy"])

        def index():
            return int(np.clip(round(step.get_value()), 0, len(dims) - 1))

        n_dims.number.add_updater(lambda m: m.set_value(int(dims[index()])))
        ratio.number.add_updater(lambda m: m.set_value(float(contrast[index()])))
        ranks.number.add_updater(lambda m: m.set_value(float(agreement[index()])))
        self.play(step.animate.set_value(len(dims) - 1), run_time=8, rate_func=smooth)
        for mob in (dots, n_dims.number, ratio.number, ranks.number):
            mob.clear_updaters()
        verdict = col.place(
            caption_stack(
                [
                    "Every distance looks alike,",
                    "but the ordering keeps",
                    "much of its information.",
                ],
                color=STRUCTURE_COLOR,
            ),
            buff=0.45,
        )
        self.play(FadeIn(verdict))
        self.marked_next_slide(notes=NOTES["w12_cancel"])

    # -- anatomy of a noisy distance ----------------------------------------------------
    def stage_anatomy(self):
        self.clear_slide()
        self.set_title("Anatomy of a noisy distance")
        colours = [
            STRUCTURE_COLOR,
            COLOR_CYCLE[6],
            COLOR_CYCLE[4],
            COLOR_CYCLE[4],
            REPEL_COLOR,
        ]
        eq = MathTex(
            r"\lVert x_i-x_j\rVert^2",
            "=",
            r"\lVert\delta_{ij}+\sigma P(\varepsilon_i-\varepsilon_j)\rVert^2",
            "+",
            r"2\sigma^2(D-d)",
            "+",
            "a_i",
            "+",
            "a_j",
            "+",
            r"\xi_{ij}",
            font_size=40,
        )
        if eq.width > 12.6:
            eq.scale_to_fit_width(12.6)
        eq.move_to(to3((0, 1.75)))
        terms = [eq[2], eq[4], eq[6], eq[8], eq[10]]
        for term, colour in zip(terms, colours):
            term.set_color(colour)

        def tag_for(lines, colour, x, y):
            tag = caption_stack(
                lines, 17, buff=0.05, color=darker(ManimColor(colour), 0.2)
            )
            for line in tag:
                line.set_x(x)
            tag.move_to(to3((x, y)), aligned_edge=UP)
            return tag

        row1 = eq.get_bottom()[1] - 0.3
        row2 = row1 - 0.85
        fluct_x = min(terms[4].get_x(), config.frame_width / 2 - 1.2)
        per_point_x = (terms[2].get_x() + terms[3].get_x()) / 2
        tags = VGroup(
            tag_for(
                ["within the data", "real blur"], colours[0], terms[0].get_x(), row1
            ),
            tag_for(
                ["common offset", "grows like D"], colours[1], terms[1].get_x(), row1
            ),
            tag_for(
                ["per-point effects", "(a\u2c7c makes hubs)"],
                colours[2],
                per_point_x,
                row2,
            ),
            tag_for(
                ["pair fluctuation", "grows like \u221aD"], colours[4], fluct_x, row1
            ),
        )
        pointer = Line(
            to3((per_point_x, terms[2].get_bottom()[1] - 0.08)),
            tags[2].get_top() + UP * 0.05,
            color=darker(ManimColor(colours[2]), 0.2),
            stroke_width=1.5,
        )
        self.play(Write(eq[:3]), run_time=1.2)
        self.play(FadeIn(tags[0]))
        self.play(Write(eq[3:5]), FadeIn(tags[1]), run_time=0.7)
        self.play(Write(eq[5:9]), Create(pointer), FadeIn(tags[2]), run_time=0.9)
        self.play(Write(eq[9:]), FadeIn(tags[3]), run_time=0.7)
        self.marked_next_slide(notes=NOTES["w13_rule"])

        strikes = VGroup(
            *[
                Line(
                    t.get_corner(DL) + LEFT * 0.05,
                    t.get_corner(UR) + RIGHT * 0.05,
                    color=HIGHLIGHT_COLOR,
                    stroke_width=4,
                )
                for t in (terms[1], terms[2])
            ]
        )
        note = caption_stack(
            [
                "When point i ranks its candidates, the offset and its own a\u1d62 shift",
                "every candidate equally, so they cancel. Only a\u2c7c and the pair",
                "fluctuation are left to reorder its neighbours.",
            ],
            font_size=22,
        )
        note.move_to(to3((0, -1.75)))
        self.play(Create(strikes))
        self.play(FadeIn(note))
        self.marked_next_slide(notes=NOTES["w14_spiral"])

        rule = caption_stack(
            [
                "Statistics that aggregate many comparisons survive the noise;",
                "a single local distance ratio does not.",
            ],
            font_size=24,
            color=STRUCTURE_COLOR,
        )
        rule.move_to(to3((0, -3.0)))
        self.play(FadeIn(rule))
        self.marked_next_slide(notes=NOTES["w15_denser"])

    # -- the graph still breaks: the density paradox ---------------------------------------
    def spiral_panel(self, n, centre, size=(5.4, 4.6), point_px=7.0):
        """Points coloured by the share of their k neighbours that are true
        (navy all, rose none): edges along a curve overlap, so colouring
        edges would show only whichever layer is drawn last."""
        d = self.data
        P = fit_points(d[f"spiral_{n}_points"], size[0], size[1], centre)
        directed, kind = d[f"spiral_{n}_edges"], d[f"spiral_{n}_kind"]
        k = self.meta["spiral_params"]["k"]
        true_share = np.bincount(directed[:, 0], weights=(kind == 0), minlength=n) / k
        pairs = undirected(directed)
        underlay = segment_cloud(
            P[pairs[:, 0]], P[pairs[:, 1]], SECONDARY_COLOR, 0.5, 0.25
        )
        colours = np.array(
            [
                ManimColor(
                    interpolate_color(
                        ManimColor(WRONG_COLOR), ManimColor(GENUINE_COLOR), f
                    )
                ).to_rgb()
                for f in true_share
            ]
        )
        points = EmbeddingCloud(
            P, colours, fit="none", point_px=point_px, shuffle_seed=None
        )
        return P, underlay, points

    def stage_spiral(self):
        self.clear_slide()
        self.set_title("The graph still breaks, predictably")
        stats = self.meta["spiral"]
        panels = {}
        for n, x in ((400, -3.45), (1600, 3.45)):
            P, layers, points = self.spiral_panel(n, (x, -0.75), size=(5.2, 4.1))
            head = caption(f"{n:,} points", 26).move_to(to3((x, 2.5)))
            score = MetricReadout(
                "overlap with true neighbours",
                stats[str(n)]["overlap"],
                num_decimal_places=2,
                font_size=20,
            ).move_to(to3((x, 1.95)))
            panels[n] = (P, layers, points, head, score)
        ramp = VGroup(
            *[
                Rectangle(
                    width=0.16,
                    height=0.18,
                    stroke_width=0,
                    fill_opacity=1,
                    fill_color=interpolate_color(
                        ManimColor(WRONG_COLOR), ManimColor(GENUINE_COLOR), t / 11
                    ),
                )
                for t in range(12)
            ]
        ).arrange(RIGHT, buff=0)
        key = (
            VGroup(
                caption("each point's share of true neighbours:", 18),
                caption("none", 18),
                ramp,
                caption("all", 18),
            )
            .arrange(RIGHT, buff=0.15)
            .to_edge(DOWN, buff=0.3)
        )
        for n in (400, 1600):
            P, layers, points, head, score = panels[n]
            self.add(points)
            self.play(
                PMFadeIn(points),
                FadeIn(head),
                FadeIn(layers),
                FadeIn(score),
                run_time=1.2,
            )
            if n == 400:
                self.play(FadeIn(key))
                self.marked_next_slide(notes=NOTES["w16_law"])
        self.marked_next_slide(notes=NOTES["w17_kinds"])

        self.play(FadeOut(key))
        law = (
            VGroup(
                MathTex(
                    r"\tilde\tau \;\propto\; \sigma^2\sqrt{D}\,\Big(\frac{n}{k}\Big)^{2/d}",
                    font_size=40,
                ),
                caption_stack(
                    [
                        "More points at the same k ask for finer",
                        "distinctions at the same noise.",
                    ],
                    font_size=20,
                ),
            )
            .arrange(RIGHT, buff=0.5)
            .to_edge(DOWN, buff=0.2)
        )
        self.play(Write(law[0]), FadeIn(law[1]))
        self.spiral_1600 = panels[1600][0]
        self.marked_next_slide(notes=NOTES["w18_incoherent"])

    # -- four kinds of wrong edge ------------------------------------------------------------
    def stage_four_kinds(self):
        self.clear_slide()
        self.set_title("Four kinds of wrong edge")
        d, meta = self.data, self.meta
        cells = [(-3.45, 1.05), (3.45, 1.05), (-3.45, -2.15), (3.45, -2.15)]
        vis_w, vis_h = 2.5, 2.3

        def vis_centre(cell):
            return np.array([cell[0] - 1.95, cell[1]])

        def cell_text(name, lines, cell):
            text = VGroup(
                caption(name, 24, color=STRUCTURE_COLOR),
                caption_stack(lines, 17, buff=0.06),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
            if text.width > 3.85:
                text.scale_to_fit_width(3.85)
            text.move_to(to3((cell[0] - 0.55, cell[1])), aligned_edge=LEFT)
            return text

        # 1. near misses: a stretch of the dense spiral, straightened, edges as arcs
        n = len(d["spiral_1600_points"])
        edges, kind = undirected(d["spiral_1600_edges"], d["spiral_1600_kind"])
        lo, hi = int(0.60 * n), int(0.60 * n) + 36
        sel = (edges[:, 0] >= lo) & (edges[:, 1] < hi) & (kind <= 1)
        c = vis_centre(cells[0])
        xs = np.linspace(c[0] - vis_w / 2, c[0] + vis_w / 2, hi - lo)
        base_y = c[1] - 0.75
        arcs = VGroup()
        for (a, b), k in zip(edges[sel], kind[sel]):
            arcs.add(
                ArcBetweenPoints(
                    to3((xs[a - lo], base_y)),
                    to3((xs[b - lo], base_y)),
                    angle=-PI * 0.9,
                    color=GENUINE_COLOR if k == 0 else NEAR_MISS_COLOR,
                    stroke_width=1.0 if k == 0 else 1.6,
                    stroke_opacity=0.5 if k == 0 else 0.9,
                )
            )
        near = VGroup(
            arcs,
            VGroup(
                *[
                    Dot(to3((x, base_y)), radius=0.025, color=STRUCTURE_COLOR)
                    for x in xs
                ]
            ),
            caption(
                "a stretch of the curve, straightened", 14, color=SECONDARY_COLOR
            ).move_to(to3((c[0], base_y - 0.25))),
        )
        text1 = cell_text(
            "Near misses",
            [
                "a true neighbour just outside the top k",
                "swaps with one just inside: harmless,",
                "gains and losses balance",
            ],
            cells[0],
        )

        # 2. incoherent shortcuts: the circle from before
        pos, theta, nn = self.circle_final
        c = vis_centre(cells[1])
        cpos = fit_points(pos, vis_h, vis_h, c)
        pairs = {(min(i, int(j)), max(i, int(j))) for i, j in enumerate(nn)}
        far = [(a, b) for a, b in pairs if angular_gap(theta[a], theta[b]) > PI / 2]
        loc = [(a, b) for a, b in pairs if angular_gap(theta[a], theta[b]) <= PI / 2]
        circ = VGroup(
            segment_cloud(
                cpos[[a for a, _ in loc]],
                cpos[[b for _, b in loc]],
                STRUCTURE_COLOR,
                1.5,
                0.8,
            ),
            segment_cloud(
                cpos[[a for a, _ in far]],
                cpos[[b for _, b in far]],
                WRONG_COLOR,
                2.2,
                0.9,
            ),
            VGroup(*[Dot(to3(p), radius=0.035, color=STRUCTURE_COLOR) for p in cpos]),
        )
        text2 = cell_text(
            "Incoherent shortcuts",
            [
                "isolated edges between unrelated",
                "neighbourhoods: repairable,",
                "as we'll see next",
            ],
            cells[1],
        )

        # 3. hubs: how many neighbour lists each point appears in
        indeg = d["spiral_1600_indegree"]
        c = vis_centre(cells[2])
        counts = np.bincount(indeg)
        fixed_counts = np.bincount(
            d["spiral_1600_indegree_corrected"], minlength=len(counts)
        )[: len(counts)]
        tallest = max(counts.max(), fixed_counts.max())  # one scale for both histograms
        bar_w = vis_w / len(counts)
        base = np.array([c[0] - vis_w / 2, c[1] - vis_h / 2 + 0.3])
        bars = VGroup(
            *[
                Rectangle(
                    width=bar_w * 0.85,
                    height=max(1e-3, (vis_h - 0.7) * h / tallest),
                    stroke_width=0,
                    fill_color=STRUCTURE_COLOR,
                    fill_opacity=0.75,
                ).move_to(
                    to3(base + [bar_w * (v + 0.5), (vis_h - 0.7) * h / tallest / 2])
                )
                for v, h in enumerate(counts)
                if h
            ]
        )
        top = int(indeg.max())
        hub_mark = VGroup(
            Dot(
                to3(base + [bar_w * (top + 0.5), 0.12]),
                radius=0.06,
                color=SOURCE_COLOR,
                stroke_color=STRUCTURE_COLOR,
                stroke_width=1.5,
            ),
            caption(f"{top}", 14, color=SOURCE_COLOR).move_to(
                to3(base + [bar_w * (top + 0.5), 0.35])
            ),
        )
        axis = VGroup(
            Line(
                to3(base),
                to3(base + [vis_w, 0]),
                color=STRUCTURE_COLOR,
                stroke_width=1.5,
            ),
            caption("lists each point appears in", 14, color=SECONDARY_COLOR).move_to(
                to3(base + [vis_w / 2, -0.22])
            ),
        )
        scale_h = (vis_h - 0.7) / tallest
        after = VGroup(
            *[
                Rectangle(
                    width=bar_w * 0.85,
                    height=max(1e-3, scale_h * h),
                    stroke_color=ATTRACT_COLOR,
                    stroke_width=1.5,
                    fill_opacity=0,
                ).move_to(to3(base + [bar_w * (v + 0.5), scale_h * h / 2]))
                for v, h in enumerate(fixed_counts)
                if h
            ]
        )
        hubs = VGroup(bars, axis, hub_mark)
        corr = meta["spiral"]["1600"]["corrected"]
        spread = meta["spiral"]["1600"]["indegree_sd"]
        text3 = cell_text(
            "Hubs",
            [
                f"low-noise points land in many lists",
                f"(up to {top}; about 10 is typical).",
                "Estimating each point's noise locally",
                f"and removing it cuts the spread from",
                f"{spread:.1f} to {corr['indegree_sd']:.1f} (blue outline)",
            ],
            cells[2],
        )
        hubs.add(after)

        # 4. coherent bridges: the two strands, and the PCA split
        c = vis_centre(cells[3])
        S, lab = d["strands_points"], d["strands_labels"]
        Ps = fit_points(
            np.c_[S[:, 0], S[:, 1] * 3.0], vis_w, 0.8, c + [0, 0.65]
        )  # gap drawn enlarged
        se = d["strands_edges"]
        bridge = lab[se[:, 0]] != lab[se[:, 1]]
        strands = VGroup(
            segment_cloud(Ps[se[bridge, 0]], Ps[se[bridge, 1]], WRONG_COLOR, 0.8, 0.3),
            VGroup(
                *[
                    Dot(
                        to3(p),
                        radius=0.018,
                        color=(COLOR_CYCLE[0] if l == 0 else COLOR_CYCLE[1]),
                    )
                    for p, l in zip(Ps[::2], lab[::2])
                ]
            ),
        )
        pc = d["strands_pc"]
        bins = np.linspace(pc.min(), pc.max(), 25)
        h0 = np.histogram(pc[lab == 0], bins)[0]
        h1 = np.histogram(pc[lab == 1], bins)[0]
        top_h = max(h0.max(), h1.max())
        hbase = np.array([c[0] - vis_w / 2, c[1] - vis_h / 2 + 0.25])
        width = vis_w / (len(bins) - 1)
        hist = VGroup()
        for k in range(len(bins) - 1):
            for h, colour in ((h0[k], COLOR_CYCLE[0]), (h1[k], COLOR_CYCLE[1])):
                if h:
                    hist.add(
                        Rectangle(
                            width=width * 0.9,
                            height=0.8 * h / top_h,
                            stroke_width=0,
                            fill_color=colour,
                            fill_opacity=0.65,
                        ).move_to(to3(hbase + [width * (k + 0.5), 0.4 * h / top_h]))
                    )
        hist.add(
            caption("PCA of the raw data", 14, color=SECONDARY_COLOR).move_to(
                to3(hbase + [vis_w / 2, -0.2])
            )
        )
        st = meta["strands"]
        text4 = cell_text(
            "Coherent bridges",
            [
                f"{st['bridging']:.0%} of edges join two strands,",
                f"yet PCA of the raw data splits",
                f"them {st['pca_accuracy']:.0%}: the graph discards",
                "what the data still holds",
            ],
            cells[3],
        )

        steps = [
            (near, text1, "w19_hubs"),
            (circ, text2, "w20_coherent"),
            (hubs, text3, "w21_takeaway"),
            (VGroup(strands, hist), text4, "w22_end"),
        ]
        for visual, text, notes in steps:
            self.play(FadeIn(visual), FadeIn(text))
            self.marked_next_slide(notes=NOTES[notes])

    def stage_takeaway(self):
        self.clear_slide()
        self.takeaway("Rankings help,", "but the graph still needs repair.")
        self.start_section_wipe(section_title("repair"), auto_next=True)


# ---------------------------------------------------------------------------
# Speaker notes
# ---------------------------------------------------------------------------
NOTES = {
    "w01_map": (
        "Everything so far takes the neighbour graph as given and lays it out better and "
        "faster. The research question for what comes next: is the graph itself right? "
        "Here's UMAP's map of MNIST."
    ),
    "w02_edges": "Join every digit to its single nearest neighbour in pixel space.",
    "w03_long": (
        "Most of those edges are short: the layout agrees with them. But some jump right "
        "across the map, between regions belonging to different digits."
    ),
    "w04_pair": "Here is one of them.",
    "w05_pair2": "And another. In pixel space these really are each other's nearest neighbours.",
    "w06_lesson": (
        "Two lessons. The graph we lay out has wrong edges in it. And the layout has "
        "already overruled some of them: UMAP placed these pairs far apart even though "
        "each was the other's closest image."
    ),
    "w07_circle": (
        "Where do wrong edges come from? Start with a circle of 48 points, each joined to "
        "its nearest neighbour, and add noise in 400 extra directions. We draw every point "
        "at its true position, so any edge across the circle is a mistake."
    ),
    "w08_sweep": (
        "As the noise grows, nearest neighbours start to jump across the circle. This is "
        "a genuine random draw, not an engineered one. The striking number: the noise is "
        "only 0.3 of the radius in any one direction, but six times the radius in total, "
        "because it adds up over 400 directions."
    ),
    "w10_sweep": (
        "Take one point and look at its distance to every other point, as the noise spreads "
        "over more directions. Vertically, how far apart they really are; horizontally, "
        "the distance we observe."
    ),
    "w11_anatomy": (
        "Every distance slides to the right and they bunch together: the farthest ends up "
        "only about twenty percent further than the nearest. But the vertical order is "
        "mostly kept: rank agreement stays around 0.8."
    ),
    "w12_cancel": (
        "Why? Split the squared distance into parts. There's real blur within the data. "
        "There's a common offset, the same for every pair, which grows with the dimension. "
        "There's an effect of each point's own noise, and there's a small fluctuation per "
        "pair, which grows only like the square root of the dimension."
    ),
    "w13_rule": (
        "When one point ranks its candidates, the offset and its own effect shift them all "
        "equally, so they cancel. That's why nearest-neighbour graphs work at all in high "
        "dimensions. What's left, the candidate's effect and the pair fluctuation, still "
        "reorders neighbours."
    ),
    "w14_spiral": (
        "The general rule: anything that aggregates many comparisons survives; anything "
        "that reads a single local distance ratio does not, as the collapsing contrast showed."
    ),
    "w15_denser": (
        "Rankings help, but the graph still breaks, and predictably. Here's a spiral with "
        "noise in 256 dimensions, 400 points, ten neighbours each. Each point is coloured by "
        "the share of its neighbours that are true: navy is all of them, rose none."
    ),
    "w16_law": (
        "Now sample it four times more densely, same noise, same k. The graph gets much "
        "worse: overlap with the true neighbours falls from 0.90 to 0.44."
    ),
    "w17_kinds": (
        "The research's analysis gives a law for this: corruption depends on a noise scale "
        "that grows with the ambient dimension and with n over k. More points at the same k "
        "ask for finer distinctions at the same noise."
    ),
    "w18_incoherent": (
        "Wrong edges come in four kinds, with different remedies. Near misses: a true "
        "neighbour just outside the top k swaps with one just inside. Harmless."
    ),
    "w19_hubs": (
        "Incoherent shortcuts, like the ones across the circle: isolated, unrelated to "
        "their surroundings. These the layout can repair, as we'll see."
    ),
    "w20_coherent": (
        "Hubs: points whose noise happens to be small look close to everything. In this "
        "simulation, estimating each point's noise effect from its local neighbourhood and "
        "removing it cuts the spread of list counts from about 6.6 to 2.4."
    ),
    "w21_takeaway": (
        "And coherent bridges: two parallel strands joined by forty percent of the graph's "
        "edges. No graph-only method can tell them apart, yet a principal-component split "
        "of the raw data separates them. The information is in the data; the graph discards it."
    ),
    "w22_end": "So rankings help, but the graph still needs repair. Can the layout do it?",
}
