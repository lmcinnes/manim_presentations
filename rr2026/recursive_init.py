"""Class 3: recursive initialization.

    python assets_recursive.py            # once (add --synthetic to test offline)
    manim-slides render recursive_init.py RecursiveInit
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

TOY_BOX = dict(width=8.0, height=5.6, center=(-2.4, -0.35))
COLUMN_TOP_LEFT = (2.35, 2.45)
FINE_RADIUS = 0.075


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def to3(p):
    return np.array([p[0], p[1], 0.0])


def caption(text, font_size=22, color=DEFAULT_COLOR, **kwargs):
    return crisp_text(text, font_size=font_size, color=color, **kwargs)


def caption_stack(lines, font_size=22, buff=0.1, color=DEFAULT_COLOR):
    group = VGroup(*[caption(line, font_size, color=color) for line in lines])
    return group.arrange(DOWN, aligned_edge=LEFT, buff=buff)


class Column:
    """Stacks captions top-down in the right-hand column."""

    def __init__(self, top_left=COLUMN_TOP_LEFT, max_width=None):
        self.top_left = to3(top_left)
        # Keep everything inside the frame (with a margin before the right edge).
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


def box_transform(point_sets, width, height, center):
    """One affine map fitting the union of several layouts into a box."""
    allp = np.concatenate([np.asarray(p).reshape(-1, 2) for p in point_sets])
    lo, hi = allp.min(0), allp.max(0)
    mid, span = 0.5 * (lo + hi), np.maximum(hi - lo, 1e-9)
    scale = min(width / span[0], height / span[1])
    center = np.asarray(center, dtype=float)
    return lambda p: center + (np.asarray(p) - mid) * scale


def greedy_colors(n_parts, pairs, palette):
    """Colour parts so that parts joined by an edge differ where possible."""
    neighbours = [set() for _ in range(n_parts)]
    for p, q in pairs:
        if p != q:
            neighbours[p].add(q)
            neighbours[q].add(p)
    order = sorted(range(n_parts), key=lambda k: -len(neighbours[k]))
    chosen, usage = {}, np.zeros(len(palette), dtype=int)
    for k in order:
        used = {chosen[j] for j in neighbours[k] if j in chosen}
        free = [c for c in range(len(palette)) if c not in used] or list(range(len(palette)))
        pick = min(free, key=lambda c: (usage[c], c))  # spread the palette evenly
        chosen[k] = pick
        usage[pick] += 1
    return [palette[chosen[k]] for k in range(n_parts)]


def polyline(axes, xs, ys, color, width=4):
    return VMobject(color=color, stroke_width=width).set_points_as_corners(
        [axes.c2p(x, y) for x, y in zip(xs, ys)]
    )


class ToyGraph(VGroup):
    """Graph whose edge lines follow their end nodes."""

    def __init__(self, positions, edges, weights, node_color=STRUCTURE_COLOR,
                 radius=FINE_RADIUS, edge_color=SECONDARY_COLOR):
        self.edge_list = [tuple(map(int, e)) for e in edges]
        self.weights = np.asarray(weights, dtype=float)
        self.nodes = VGroup(*[Dot(to3(p), radius=radius, color=node_color) for p in positions])
        self.edges = VGroup()
        for (i, j), w in zip(self.edge_list, self.weights):
            line = Line(to3(positions[i]), to3(positions[j]), color=edge_color,
                        stroke_width=0.8 + 2.2 * w, stroke_opacity=0.25 + 0.45 * w)
            line.add_updater(
                lambda l, i=i, j=j: l.put_start_and_end_on(
                    self.nodes[i].get_center(), self.nodes[j].get_center()
                )
            )
            self.edges.add(line)
        super().__init__(self.edges, self.nodes)


# ---------------------------------------------------------------------------
# The slide class
# ---------------------------------------------------------------------------
class RecursiveInit(UMAPSlide):
    def construct(self):
        self.data = np.load(ASSETS / "recursive.npz")
        self.meta = json.loads((ASSETS / "recursive.json").read_text())
        self.setup_toy()

        self.end_section_wipe(
            SECTION_TITLES["recursive"],
            next_slide_prep=lambda: None,
            notes=NOTES["s01_wave"],
        )
        self.stage_problem()
        self.stage_pyramid()
        self.stage_coarsen()
        self.stage_coarse_edges()
        self.stage_back_up()
        self.stage_details()
        self.stage_mnist_morph()
        self.stage_compare()
        self.stage_takeaway()

    # -- toy data ---------------------------------------------------------------
    def setup_toy(self):
        d = self.data
        self.labels = d["toy_labels"].astype(int)
        self.n_parts = int(self.labels.max()) + 1
        layouts = [d["toy_pos"], d["toy_centroids"], d["toy_coarse_layout"],
                   d["toy_expanded"], d["toy_settled"]]
        to_scene = box_transform(layouts, **TOY_BOX)
        self.P0 = to_scene(d["toy_pos"])
        self.centroids = to_scene(d["toy_centroids"])
        self.coarse_layout = to_scene(d["toy_coarse_layout"])
        self.expanded = to_scene(d["toy_expanded"])
        self.settled = to_scene(d["toy_settled"])
        self.edges = d["toy_edges"]
        self.weights = d["toy_weights"]
        pairs = [(self.labels[i], self.labels[j]) for i, j in self.edges]
        self.part_colors = greedy_colors(self.n_parts, pairs, COLOR_CYCLE[:10])
        self.part_sizes = np.bincount(self.labels, minlength=self.n_parts)

    def make_toy_graph(self, positions):
        return ToyGraph(positions, self.edges, self.weights)

    def title_text(self, text):
        return text + (" (stand-in data)" if self.meta.get("synthetic") else "")

    # -- stage 1: local forces spread slowly -----------------------------------------
    def stage_problem(self):
        self.set_title("Local forces spread slowly")
        graph = self.make_toy_graph(self.P0)
        self.play(FadeIn(graph.edges), FadeIn(graph.nodes, lag_ratio=0.01), run_time=1.2)

        # Hop distances from the leftmost node (breadth-first search).
        n = len(self.P0)
        adj = [[] for _ in range(n)]
        for i, j in self.edges:
            adj[i].append(j)
            adj[j].append(i)
        start = int(np.argmin(self.P0[:, 0]))
        hops = np.full(n, -1)
        hops[start] = 0
        frontier = [start]
        while frontier:
            nxt = []
            for i in frontier:
                for j in adj[i]:
                    if hops[j] < 0:
                        hops[j] = hops[i] + 1
                        nxt.append(j)
            frontier = nxt
        max_hops = int(hops.max())

        ring = Circle(radius=0.16, color=SOURCE_COLOR, stroke_width=4).move_to(graph.nodes[start])
        counter = MetricReadout("epochs", 0, num_decimal_places=0, font_size=26)
        col = Column()
        col.place(counter)
        self.play(Create(ring), graph.nodes[start].animate.set_color(SOURCE_COLOR), FadeIn(counter))
        for h in range(1, max_hops + 1):
            reached = [graph.nodes[i] for i in np.flatnonzero(hops == h)]
            counter.number.set_value(h)
            self.play(*[m.animate.set_color(ATTRACT_COLOR) for m in reached], run_time=0.28)
        note = col.place(caption_stack(
            ["Each epoch, a point only feels", "its neighbours. A nudge at one",
             f"end takes {max_hops} epochs to cross."]
        ))
        self.play(FadeIn(note))
        self.marked_next_slide(notes=NOTES["s02_options"])

        spectral = col.place(caption_stack(
            ["Spectral init solves the global", "arrangement directly, with an",
             "eigensolver on the whole graph."]
        ))
        idea = col.place(caption_stack(
            ["Recursive init: lay out a small", "version of the graph first."],
            color=STRUCTURE_COLOR,
        ))
        self.play(FadeIn(spectral))
        self.play(FadeIn(idea))
        self.marked_next_slide(notes=NOTES["s03_pyramid"])

    # -- stage 2: the pyramid -------------------------------------------------------
    def stage_pyramid(self):
        self.clear_slide()
        self.set_title("A pyramid of smaller graphs")
        sizes = self.meta["level_sizes"]  # full graph first
        bars = VGroup()
        for k, size in enumerate(sizes):
            width = 7.0 * (size / sizes[0]) ** 0.25
            bar = Rectangle(width=width, height=0.5, stroke_width=0,
                            fill_color=interpolate_color(ManimColor(SECONDARY_COLOR),
                                                         ManimColor(STRUCTURE_COLOR),
                                                         k / max(len(sizes) - 1, 1)),
                            fill_opacity=1)
            noun = "points" if k == 0 else "parts"
            label = caption(f"{size:,} {noun}", 22, color=WHITE if width > 2.4 else STRUCTURE_COLOR)
            if width > 2.4:
                label.move_to(bar)
            else:
                label.next_to(bar, RIGHT, buff=0.2)
            bars.add(VGroup(bar, label))
        bars.arrange(UP, buff=0.22).move_to(to3((0, -0.45)))
        pca = caption("placed by PCA", 22, color=REPEL_COLOR).next_to(bars[-1], UP, buff=0.2)
        up = Arrow(bars[0].get_left() + LEFT * 0.4, bars[-1].get_left() + LEFT * 1.8 + DOWN * 0.1,
                   buff=0, color=SECONDARY_COLOR, stroke_width=4)
        down = Arrow(bars[-1].get_right() + RIGHT * 1.8 + DOWN * 0.1, bars[0].get_right() + RIGHT * 0.4,
                     buff=0, color=STRUCTURE_COLOR, stroke_width=4)
        up_label = caption("coarsen", 22, color=SECONDARY_COLOR).next_to(up, LEFT, buff=0.15)
        down_label = caption_stack(["lay out,", "then expand"], color=STRUCTURE_COLOR).next_to(down, RIGHT, buff=0.15)

        self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.15) for b in bars], lag_ratio=0.25))
        self.play(GrowArrow(up), FadeIn(up_label))
        self.play(FadeIn(pca))
        self.play(GrowArrow(down), FadeIn(down_label))
        self.marked_next_slide(notes=NOTES["s04_seeds"])

    # -- stage 3: coarsening ------------------------------------------------------------
    def stage_coarsen(self):
        self.clear_slide()
        self.set_title("Coarsening: grow parts from hubs")
        d = self.data
        graph = self.make_toy_graph(self.P0)
        self.graph = graph
        self.play(FadeIn(graph.edges), FadeIn(graph.nodes, lag_ratio=0.01), run_time=1.2)

        col = Column()
        c_seeds = col.place(caption_stack(["The best-connected quarter", "of nodes become seeds."]))
        seeds = d["toy_seeds"]
        rings = VGroup(*[Circle(radius=0.15, color=self.part_colors[self.labels[s]], stroke_width=3)
                         .move_to(graph.nodes[s]) for s in seeds])
        self.play(FadeIn(c_seeds))
        self.play(
            *[graph.nodes[s].animate.set_color(self.part_colors[self.labels[s]]) for s in seeds],
            FadeIn(rings, scale=1.4),
        )
        self.marked_next_slide(notes=NOTES["s05_spread"])

        c_spread = col.place(caption_stack(
            ["Labels spread: a node joins a", "label once at least one unit of",
             "edge weight pulls it in."]
        ))
        self.play(FadeOut(rings), FadeIn(c_spread))
        passes = d["toy_passes"]
        for k in range(1, len(passes)):
            new = np.flatnonzero((passes[k] >= 0) & (passes[k - 1] < 0))
            if len(new):
                self.play(*[graph.nodes[i].animate.set_color(self.part_colors[passes[k][i]]) for i in new],
                          run_time=0.5)
        self.marked_next_slide(notes=NOTES["s06_stragglers"])

        c_strag = col.place(caption_stack(["Stragglers join the nearest", "labelled region."]))
        rest = np.flatnonzero(passes[-1] < 0)
        self.play(FadeIn(c_strag))
        self.play(*[graph.nodes[i].animate.set_color(self.part_colors[self.labels[i]]) for i in rest],
                  run_time=0.8)
        self.marked_next_slide(notes=NOTES["s07_collapse"])

        c_parts = col.place(caption_stack(
            ["Each part becomes one node:",
             f"{len(self.P0)} nodes, {self.n_parts} parts."]
        ))
        inner = VGroup(*[e for e, (i, j) in zip(graph.edges, graph.edge_list)
                         if self.labels[i] == self.labels[j]])
        self.supernodes = VGroup(*[
            Dot(to3(self.centroids[k]), radius=0.06 + 0.03 * np.sqrt(self.part_sizes[k]),
                color=self.part_colors[k])
            for k in range(self.n_parts)
        ])
        self.play(FadeIn(c_parts), FadeOut(inner))
        for e in inner:
            e.clear_updaters()
        graph.edges.remove(*inner)
        # Keep the edge bookkeeping in step with the edge mobjects that remain.
        keep = [self.labels[i] != self.labels[j] for i, j in graph.edge_list]
        graph.edge_list = [ij for ij, k in zip(graph.edge_list, keep) if k]
        graph.weights = graph.weights[np.array(keep, dtype=bool)]
        self.play(*[node.animate.move_to(to3(self.centroids[self.labels[i]]))
                    for i, node in enumerate(graph.nodes)], run_time=1.6)
        self.play(FadeIn(self.supernodes), FadeOut(graph.nodes), run_time=0.6)
        self.captions = VGroup(c_seeds, c_spread, c_strag, c_parts)
        self.marked_next_slide(notes=NOTES["s08_union"])

    # -- stage 4: fuzzy-union coarse edges ------------------------------------------------
    def stage_coarse_edges(self):
        d = self.data
        graph = self.graph
        self.play(FadeOut(self.captions))
        self.set_title("Coarse edges: a fuzzy union")

        # The pair of parts joined by the most fine edges.
        between = {}
        for e, (i, j) in zip(graph.edges, graph.edge_list):
            key = tuple(sorted((self.labels[i], self.labels[j])))
            between.setdefault(key, []).append(e)
        pair = max(between, key=lambda k: len(between[k]))
        highlighted = VGroup(*between[pair])

        coarse_edges = VGroup()
        self.coarse_edge_list = []
        for (p, q), w in zip(d["toy_coarse_edges"], d["toy_coarse_weights"]):
            p, q = int(p), int(q)
            line = Line(self.supernodes[p].get_center(), self.supernodes[q].get_center(),
                        color=STRUCTURE_COLOR, stroke_width=1.2 + 4.0 * w,
                        stroke_opacity=0.35 + 0.5 * w)
            line.add_updater(lambda l, p=p, q=q: l.put_start_and_end_on(
                self.supernodes[p].get_center(), self.supernodes[q].get_center()))
            coarse_edges.add(line)
            self.coarse_edge_list.append((p, q))
        target = coarse_edges[self.coarse_edge_list.index(tuple(sorted(pair)))] \
            if tuple(sorted(pair)) in self.coarse_edge_list else None

        col = Column()
        formula = col.place(MathTex(
            r"w_{PQ} \;=\; 1-\prod_{i\in P,\;j\in Q}\left(1-w_{ij}\right)", font_size=38))
        explain = col.place(caption_stack(
            ["The chance that at least one", "edge links the two parts: the",
             "same fuzzy union UMAP uses", "to build its graph."]
        ), buff=0.3)
        count = caption(f"{len(highlighted)} fine edges → 1 coarse edge", 20, color=HIGHLIGHT_COLOR)
        col.place(count, buff=0.3)

        for e in highlighted:
            e.clear_updaters()
        self.play(highlighted.animate.set_stroke(color=HIGHLIGHT_COLOR, width=4, opacity=1))
        self.play(Write(formula))
        self.play(FadeIn(explain), FadeIn(count))
        self.marked_next_slide(notes=NOTES["s09_merge"])

        others = VGroup(*[e for e in graph.edges if e not in highlighted])
        anims = [FadeOut(others), FadeIn(VGroup(*[c for c in coarse_edges if c is not target]))]
        # Morph into a static copy: transforming a group of lines into the live
        # edge would strip the edge of its own points and break its updater.
        proxy = target.copy().clear_updaters() if target is not None else None
        anims.append(ReplacementTransform(highlighted, proxy) if proxy is not None else FadeOut(highlighted))
        self.play(*anims, run_time=1.4)
        if proxy is not None:
            self.remove(proxy)
        # FadeOut restores faded mobjects after removal; drop the fine edges for good.
        self.remove(graph.edges)
        self.add(coarse_edges)
        self.bring_to_front(self.supernodes)
        self.coarse_edges = coarse_edges

        base = col.place(caption_stack(
            ["At the coarsest level, parts", "start at their members'", "average PCA position."],
            color=STRUCTURE_COLOR,
        ), buff=0.45)
        self.play(FadeIn(base))
        self.stage4_captions = VGroup(formula, explain, count, base)
        self.marked_next_slide(notes=NOTES["s10_layout"])

    # -- stage 5: coming back up ------------------------------------------------------
    def stage_back_up(self):
        self.play(FadeOut(self.stage4_captions))
        self.set_title("Coming back up")
        col = Column()
        c_layout = col.place(caption_stack(
            ["A short Adam layout of the", "coarse graph: strong repulsion,",
             "one negative per edge."]
        ))
        self.play(FadeIn(c_layout))
        self.play(*[s.animate.move_to(to3(p)) for s, p in zip(self.supernodes, self.coarse_layout)],
                  run_time=2)
        self.marked_next_slide(notes=NOTES["s11_expand"])

        formula = col.place(MathTex(
            r"x_i \;=\; \tfrac{1}{2}\,x_{p(i)} \;+\; \tfrac{1}{2}\,"
            r"\frac{\sum_j w_{ij}\,x_{p(j)}}{\sum_j w_{ij}}", font_size=34))
        c_expand = col.place(caption_stack(
            ["Each node starts halfway between", "its own part and its neighbours'", "parts."]
        ), buff=0.25)
        self.play(Write(formula))
        self.play(FadeIn(c_expand))

        # Children appear on their part, then move to the expanded positions.
        start = np.array([self.coarse_layout[self.labels[i]] for i in range(len(self.P0))])
        start = start + np.random.default_rng(0).normal(scale=1e-3, size=start.shape)
        children = self.make_toy_graph(start)
        for i, node in enumerate(children.nodes):
            node.set_color(self.part_colors[self.labels[i]])
        for e in children.edges:
            e.set_stroke(opacity=0)
        self.add(children.edges, children.nodes)
        self.bring_to_front(children.nodes)
        self.play(
            *[node.animate.move_to(to3(p)) for node, p in zip(children.nodes, self.expanded)],
            # Fade edges in without suspending the updaters that keep them on their nodes.
            *[UpdateFromAlphaFunc(e, lambda m, a, w=w: m.set_stroke(opacity=a * (0.25 + 0.45 * w)))
              for e, w in zip(children.edges, children.weights)],
            FadeOut(self.supernodes),
            FadeOut(self.coarse_edges),
            run_time=2.2,
        )
        self.marked_next_slide(notes=NOTES["s12_settle"])

        c_settle = col.place(caption_stack(
            ["The fine graph then needs only", "local refinement."], color=STRUCTURE_COLOR,
        ))
        self.play(FadeIn(c_settle))
        self.play(*[node.animate.move_to(to3(p)) for node, p in zip(children.nodes, self.settled)],
                  run_time=2)
        self.marked_next_slide(notes=NOTES["s13_details"])

    # -- stage 6: kernel continuation and orientation ------------------------------------
    def stage_details(self):
        self.clear_slide()
        self.set_title("Two quiet details")
        curves = sorted(self.meta["coarse_curves"], key=lambda c: c["n"])  # coarsest first
        params = [(c["a"], c["b"]) for c in curves] + [
            (self.meta["final_curve"]["a"], self.meta["final_curve"]["b"])]
        chart = styled_axes(
            x_range=[0, 3, 0.5], y_range=[0, 1, 0.25],
            x_label="distance", y_label="closeness",
            x_length=5.4, y_length=3.4, y_decimal_places=2,
        )
        chart.move_to(to3((-3.4, 0.35)))
        axes = chart[0]
        axes.x_axis.add_numbers([0.5, 1, 1.5, 2, 2.5], font_size=16)
        xs = np.linspace(0.001, 3, 200)
        lines = VGroup()
        for k, (a, b) in enumerate(params):
            color = interpolate_color(ManimColor(SECONDARY_COLOR), ManimColor(STRUCTURE_COLOR),
                                      k / max(len(params) - 1, 1))
            lines.add(polyline(axes, xs, 1 / (1 + a * xs ** (2 * b)), color, 2.5 if k < len(params) - 1 else 5))
        coarse_tag = caption("coarsest", 18, color=SECONDARY_COLOR).next_to(axes.c2p(1.6, 0.34), UR, buff=0.05)
        fine_tag = caption("full size", 18, color=STRUCTURE_COLOR).next_to(axes.c2p(0.9, 0.35), DL, buff=0.05)
        kernel_note = caption_stack(
            ["Coarse levels use a softer,", "nearly Cauchy-shaped kernel.",
             "It tightens level by level to", "your min_dist curve."], font_size=20,
        ).next_to(chart, DOWN, buff=0.3).align_to(chart, LEFT)

        col = Column((2.3, 1.9))
        orient = col.place(caption_stack(
            ["Every level is rotated to match", "the data's PCA, so results come",
             "out in a stable orientation,", "run after run."], font_size=24,
        ))
        self.play(FadeIn(chart))
        self.play(LaggedStart(*[Create(l) for l in lines], lag_ratio=0.3), run_time=2)
        self.play(FadeIn(coarse_tag), FadeIn(fine_tag), FadeIn(kernel_note))
        self.play(FadeIn(orient))
        self.marked_next_slide(notes=NOTES["s14_morph"])

    # -- stage 7: MNIST coarse to fine ----------------------------------------------------
    def stage_mnist_morph(self):
        self.clear_slide()
        self.set_title(self.title_text("MNIST, coarse to fine"))
        d, meta = self.data, self.meta
        cloud = EmbeddingCloud(
            d["morph"], labels_to_rgb(d["labels"], digit_rgb()),
            width=6.2, height=5.8, center=(-2.0, -0.4), fit="union", point_px=3.0,
        )
        frame = ValueTracker(0.0)
        cloud.track(frame)
        names = meta["morph_frames"]

        col = Column((3.0, 2.2))
        nodes = col.place(MetricReadout("nodes", int(names[0].split(":")[0]),
                                        num_decimal_places=0, font_size=30))
        nodes.number.set_value(int(names[0].split(":")[0]))
        status = col.place(caption("start: PCA of each part", 24, color=STRUCTURE_COLOR), buff=0.3)

        self.add(cloud)
        self.play(PMFadeIn(cloud), FadeIn(nodes), FadeIn(status), run_time=1.5)
        self.marked_next_slide(notes=NOTES["s15_morph_run"])

        def set_status(text):
            nonlocal status
            new = caption(text, 24, color=STRUCTURE_COLOR).move_to(status, aligned_edge=UL)
            self.play(FadeOut(status), run_time=0.25)
            self.play(FadeIn(new), run_time=0.3)
            status = new

        for k in range(1, len(names) - 1):
            size_now = int(names[k].split(":")[0])
            kind = names[k].split(":")[1]
            if kind == "out":
                set_status("lay out this level")
            else:
                set_status("expand to the next level")
            self.play(frame.animate.set_value(k), run_time=1.3)
            nodes.number.set_value(size_now)
        self.marked_next_slide(notes=NOTES["s16_morph_final"])

        set_status("then the full optimization")
        self.play(frame.animate.set_value(len(names) - 1), run_time=2.5)
        cloud.clear_updaters()
        self.marked_next_slide(notes=NOTES["s17_timing"])

    # -- stage 8: recursive vs spectral ------------------------------------------------------
    def stage_compare(self):
        self.clear_slide()
        self.set_title(self.title_text("Recursive vs spectral initialization"))
        d, meta = self.data, self.meta
        secs = meta["init_seconds"]
        entries = [("spectral init", secs["spectral"], OLD_STACK_COLOR),
                   ("recursive init", secs["recursive"], NEW_STACK_COLOR)]
        longest = max(v for _, v, _ in entries)
        bars = VGroup()
        for i, (label, value, color) in enumerate(entries):
            bar = Rectangle(width=max(6.0 * value / longest, 0.05), height=0.45,
                            fill_color=color, fill_opacity=1, stroke_width=0)
            bar.move_to(to3((-1.4, 0.8 - 0.8 * i)), aligned_edge=LEFT)
            name = caption(label, 24).next_to(bar, LEFT, buff=0.3)
            val = caption(f"{value:.1f} s", 24).next_to(bar, RIGHT, buff=0.25)
            bars.add(VGroup(name, bar, val))
        timing_note = caption("warm timings, same graph", 20, color=SECONDARY_COLOR).next_to(bars, DOWN, buff=0.4)
        self.play(*[GrowFromEdge(row[1], LEFT) for row in bars], *[FadeIn(row[0]) for row in bars],
                  FadeIn(timing_note))
        self.play(*[FadeIn(row[2]) for row in bars])
        self.marked_next_slide(notes=NOTES["s18_starts"])

        self.play(FadeOut(bars), FadeOut(timing_note))
        colors = labels_to_rgb(d["labels"], digit_rgb())
        clouds, tracker = {}, ValueTracker(0.0)
        heads, readouts = VGroup(), {}
        m = meta["metrics"]
        for key, name, x in (("spec", "spectral", -3.5), ("rec", "recursive", 3.3)):
            cloud = EmbeddingCloud(d[f"{key}_run"], colors, width=5.4, height=4.3,
                                   center=(x, -0.05), fit="union", point_px=2.6)
            cloud.track(tracker)
            clouds[key] = cloud
            heads.add(caption(name, 28).move_to(to3((x, 2.55))))
            metrics = m[f"{'spectral' if key == 'spec' else 'recursive'}_init"]
            group = VGroup(
                caption("start", 20, color=SECONDARY_COLOR),
                MetricReadout("trustworthiness", metrics["trustworthiness"], font_size=22),
                MetricReadout("kNN accuracy", metrics["knn_accuracy"], font_size=22),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
            group.move_to(to3((x, -3.05)))
            readouts[key] = group
        self.add(clouds["spec"], clouds["rec"])
        self.play(PMFadeIn(clouds["spec"]), PMFadeIn(clouds["rec"]), FadeIn(heads),
                  FadeIn(readouts["spec"]), FadeIn(readouts["rec"]), run_time=1.5)
        self.marked_next_slide(notes=NOTES["s19_run"])

        self.play(tracker.animate.set_value(d["rec_run"].shape[0] - 1), run_time=7, rate_func=linear)
        for cloud in clouds.values():
            cloud.clear_updaters()
        self.play(FadeOut(readouts["spec"]), FadeOut(readouts["rec"]))
        finals = VGroup()
        for key, x in (("spec", -3.5), ("rec", 3.3)):
            name = "spectral" if key == "spec" else "recursive"
            metrics = m[f"{name}_final"]
            group = VGroup(
                caption(f"points moved {meta['travel'][name]:.2f} layout radii on average", 20,
                        color=STRUCTURE_COLOR),
                MetricReadout("final trustworthiness", metrics["trustworthiness"], font_size=22),
                MetricReadout("final kNN accuracy", metrics["knn_accuracy"], font_size=22),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
            group.move_to(to3((x, -3.05)))
            finals.add(group)
        self.play(FadeIn(finals))
        self.marked_next_slide(notes=NOTES["s20_takeaway"])

    # -- takeaway -------------------------------------------------------------------------
    def stage_takeaway(self):
        self.clear_slide()
        self.takeaway("Lay out a small graph first,", "then refine.")
        self.start_section_wipe(SECTION_TITLES["payoff"], auto_next=True)


# ---------------------------------------------------------------------------
# Speaker notes
# ---------------------------------------------------------------------------
NOTES = {
    "s01_wave": (
        "Where the optimizer starts matters. Each epoch, a point only feels its "
        "neighbours along the graph, so news about the global arrangement "
        "travels about one hop per epoch. On this little graph, a nudge at one "
        "end takes this many epochs to reach the other; on real data, far more."
    ),
    "s02_options": (
        "Spectral initialization solves for the global arrangement directly, but "
        "it needs an eigensolver on the whole graph. Recursive initialization "
        "takes a different route: lay out a small version of the graph first."
    ),
    "s03_pyramid": (
        "It coarsens the graph repeatedly, about four times smaller each time, "
        "until there are a few hundred nodes. Those are placed by PCA. Then it "
        "works back up: lay out each level, expand it into the next."
    ),
    "s04_seeds": (
        "Coarsening, one level. The best-connected quarter of the nodes, by total "
        "edge weight, become seeds."
    ),
    "s05_spread": (
        "Labels spread outward in parallel passes. A node joins a label only when "
        "at least one full unit of edge weight pulls it in, so parts stay tight."
    ),
    "s06_stragglers": (
        "Nodes that never reach that bar take the label of the nearest labelled "
        "node along the graph."
    ),
    "s07_collapse": (
        "Each part becomes a single node. Parts average about four nodes, so this "
        "is closer to the aggregation step in algebraic multigrid than to "
        "clustering."
    ),
    "s08_union": (
        "Edges between parts need weights. Here several fine edges join two parts."
    ),
    "s09_merge": (
        "The coarse weight is the probability that at least one of those edges "
        "exists: one minus the product of one minus each weight. That is the same "
        "fuzzy union UMAP already uses to symmetrise its graph, so every coarse "
        "graph is a genuine UMAP graph over super-points. At the very bottom, "
        "parts are placed at the average PCA position of their members, with no "
        "eigensolver."
    ),
    "s10_layout": (
        "On the way back up, each coarse graph gets a short Adam layout, with "
        "strong but sparse repulsion: four times the usual strength, one negative "
        "sample per edge, and the hard negatives from before."
    ),
    "s11_expand": (
        "Then every node is placed halfway between its own part and the weighted "
        "average of its neighbours' parts, so children spread toward the parts "
        "they connect to instead of stacking up."
    ),
    "s12_settle": (
        "The next level down then only needs local refinement."
    ),
    "s13_details": (
        "Two quieter details. The force kernel starts soft and nearly Cauchy-"
        "shaped at coarse levels and tightens level by level to your min_dist "
        "curve. And every level is rotated to match the data's PCA, so the "
        "orientation is stable from run to run."
    ),
    "s14_morph": (
        "Here is the whole thing on MNIST. Every one of the seventy thousand "
        "points starts at its coarsest ancestor: a few hundred clumps."
    ),
    "s15_morph_run": (
        "Each level is laid out, then expanded into the next, four times smaller "
        "parts each time."
    ),
    "s16_morph_final": (
        "The full-size initialization then goes to the main optimizer, which only "
        "has to refine it."
    ),
    "s17_timing": (
        "How does it compare with spectral initialization? First, time: both "
        "initializers on the same graph, warm."
    ),
    "s18_starts": (
        "Then the starting layouts themselves, with the quality of each start."
    ),
    "s19_run": (
        "Now both run the same Adam optimizer. Recursive init also selects a gentler "
        "learning-rate schedule, which is part of the package. Watch how far each "
        "has to move."
    ),
    "s20_takeaway": "Lay out a small graph first, then refine.",
}
