"""Class 1: the new layout optimizers (node loop, Adam, soft clipping).

    python assets_optimizers.py                       # once, writes assets/
    manim-slides render new_optimizers.py NewOptimizers
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

ARROW_SCALE = 0.8  # scene units per unit of UMAP force
OLD_STEP_ALPHA = 0.1  # learning rate used for the illustrated old-kernel edge updates
THREAD_COLORS = (COLOR_CYCLE[4], COLOR_CYCLE[8])
TOY_COLORS = (COLOR_CYCLE[4], COLOR_CYCLE[8], COLOR_CYCLE[6], COLOR_CYCLE[5])
PANEL_TOP_LEFT = (3.0, 2.35)  # right-hand caption column beside the tiny graph


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def to3(p):
    return np.array([p[0], p[1], 0.0])


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v * 0


def force_arrow(start, vec, color, stroke_width=5):
    start = to3(start)
    return Arrow(
        start,
        start + to3(vec) * ARROW_SCALE,
        buff=0,
        color=color,
        stroke_width=stroke_width,
        max_tip_length_to_length_ratio=0.3,
        max_stroke_width_to_length_ratio=12,
    )


def caption(text, font_size=24, color=DEFAULT_COLOR, **kwargs):
    return crisp_text(text, font_size=font_size, color=color, **kwargs)


def caption_stack(lines, font_size=22, buff=0.12, color=DEFAULT_COLOR):
    group = VGroup(*[caption(line, font_size, color=color) for line in lines])
    return group.arrange(DOWN, aligned_edge=LEFT, buff=buff)


class Column:
    """Stacks captions top-down in the right-hand column."""

    def __init__(self, top_left=PANEL_TOP_LEFT):
        self.top_left = to3(top_left)
        self.items = []

    def place(self, mob, buff=0.35):
        if not self.items:
            mob.move_to(self.top_left, aligned_edge=UL)
        else:
            mob.next_to(self.items[-1], DOWN, buff=buff, aligned_edge=LEFT)
        self.items.append(mob)
        return mob


def fit_mapping(points, width, height, center, quantile=0.0):
    """Affine map from data coordinates into a width x height box."""
    flat = points.reshape(-1, 2)
    lo = np.quantile(flat, quantile, axis=0)
    hi = np.quantile(flat, 1 - quantile, axis=0)
    mid, span = 0.5 * (lo + hi), np.maximum(hi - lo, 1e-9)
    scale = min(width / span[0], height / span[1])
    center = np.asarray(center, dtype=float)[:2]
    return lambda p: center + (np.asarray(p) - mid) * scale


def check_mark(color=NEW_STACK_COLOR, size=0.32):
    mark = VMobject(color=color, stroke_width=6)
    mark.set_points_as_corners(
        [np.array([-0.5, 0.0, 0]), np.array([-0.15, -0.4, 0]), np.array([0.55, 0.5, 0])]
    )
    return mark.scale(size)


class TinyGraph(VGroup):
    """Hand-placed toy graph whose edges follow its nodes."""

    def __init__(self, pos, edges, weights, node_radius=0.13):
        self.pos0 = np.asarray(pos, dtype=float)
        self.edge_list = [tuple(e) for e in np.asarray(edges)]
        self.weights = np.asarray(weights, dtype=float)
        self.nodes = VGroup(
            *[
                Dot(to3(p), radius=node_radius, color=STRUCTURE_COLOR)
                for p in self.pos0
            ]
        )
        self.edges = VGroup()
        for (i, j), w in zip(self.edge_list, self.weights):
            line = Line(
                self.nodes[i].get_center(),
                self.nodes[j].get_center(),
                color=SECONDARY_COLOR,
                stroke_width=1.5 + 4.0 * w,
                stroke_opacity=0.35 + 0.55 * w,
            )
            line.add_updater(
                lambda l, i=i, j=j: l.put_start_and_end_on(
                    self.nodes[i].get_center(), self.nodes[j].get_center()
                )
            )
            self.edges.add(line)
        super().__init__(self.edges, self.nodes)

    def pos(self):
        return np.array([n.get_center()[:2] for n in self.nodes])

    def neighbors(self, i):
        return [b if a == i else a for a, b in self.edge_list if i in (a, b)]

    def edge_mob(self, i, j):
        key = (min(i, j), max(i, j))
        return self.edges[self.edge_list.index(key)]

    def attraction(self, i, j, emb_scale, pos=None):
        pos = self.pos() if pos is None else pos
        delta = pos[j] - pos[i]
        return unit(delta) * attraction_magnitude(np.linalg.norm(delta) * emb_scale)

    def repulsion(self, i, k, emb_scale, pos=None):
        pos = self.pos() if pos is None else pos
        delta = pos[i] - pos[k]
        raw = repulsion_magnitude(np.linalg.norm(delta) * emb_scale)
        return unit(delta) * soft_clip(raw)


# ---------------------------------------------------------------------------
# The slide class
# ---------------------------------------------------------------------------
class NewOptimizers(UMAPSlide):
    def construct(self):
        self.data = np.load(ASSETS / "optimizers.npz")
        self.meta = json.loads((ASSETS / "optimizers.json").read_text())
        self.emb = self.meta["tiny_emb_scale"]
        self.src = self.meta["tiny_source"]

        self.end_section_wipe(
            SECTION_TITLES["optimizers"],
            next_slide_prep=self.build_tiny_graph,
            notes=NOTES["setup"],
        )
        self.stage_setup()
        self.stage_old_loop()
        self.stage_node_loop()
        self.stage_adam()
        self.stage_clipping()
        self.stage_recap()

    # -- stage 1 ----------------------------------------------------------
    def build_tiny_graph(self):
        d = self.data
        self.graph = TinyGraph(d["tiny_pos"], d["tiny_edges"], d["tiny_weights"])
        self.graph.shift(DOWN * 0.3)
        self.add(self.graph)

    def source_forces(self, pos=None):
        """Attraction to each neighbour and repulsion from two negatives."""
        g, s = self.graph, self.src
        pos = g.pos() if pos is None else pos
        attract = [(j, g.attraction(s, j, self.emb, pos)) for j in g.neighbors(s)]
        repel = [(k, g.repulsion(s, k, self.emb, pos)) for k in (5, 6)]
        return attract, repel

    def stage_setup(self):
        g, s = self.graph, self.src
        self.set_title("Layout: forces on a graph")
        source_ring = Dot(
            g.nodes[s].get_center(),
            radius=0.19,
            color=SOURCE_COLOR,
            stroke_color=STRUCTURE_COLOR,
            stroke_width=3,
        )
        attract, repel = self.source_forces()
        start = g.nodes[s].get_center()[:2]
        a_arrows = VGroup(*[force_arrow(start, v, ATTRACT_COLOR) for _, v in attract])
        links = VGroup(
            *[
                DashedLine(
                    g.nodes[k].get_center(),
                    g.nodes[s].get_center(),
                    color=REPEL_COLOR,
                    stroke_width=2,
                    dash_length=0.08,
                    stroke_opacity=0.7,
                )
                for k, _ in repel
            ]
        )
        r_arrows = VGroup(*[force_arrow(start, v, REPEL_COLOR) for _, v in repel])

        legend = VGroup(
            VGroup(
                Arrow(ORIGIN, RIGHT * 0.6, buff=0, color=ATTRACT_COLOR, stroke_width=5),
                caption("attraction along graph edges", 22),
            ).arrange(RIGHT, buff=0.2),
            VGroup(
                Arrow(ORIGIN, RIGHT * 0.6, buff=0, color=REPEL_COLOR, stroke_width=5),
                caption("repulsion from random negative samples", 22),
            ).arrange(RIGHT, buff=0.2),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        legend.to_corner(DL, buff=0.45)

        self.play(FadeIn(source_ring, scale=1.5))
        self.play(LaggedStart(*[GrowArrow(a) for a in a_arrows], lag_ratio=0.2))
        self.play(Create(links), LaggedStart(*[GrowArrow(a) for a in r_arrows]))
        self.play(FadeIn(legend, shift=UP * 0.2))
        self.stage1 = VGroup(source_ring, a_arrows, links, r_arrows, legend)
        self.marked_next_slide(notes=NOTES["old_sweep"])

    # -- stage 2 ----------------------------------------------------------
    def old_edge_update(self, i, j, alpha=OLD_STEP_ALPHA):
        """Compatibility-kernel update for one edge: both endpoints move now."""
        pos = self.graph.pos()
        diff = (pos[i] - pos[j]) * self.emb
        d2 = diff @ diff
        coeff = -2 * UMAP_A * UMAP_B * d2 ** (UMAP_B - 1) / (UMAP_A * d2**UMAP_B + 1)
        grad = np.clip(coeff * diff, -4.0, 4.0) * alpha / self.emb
        return grad, -grad

    def stage_old_loop(self):
        g = self.graph
        self.play(FadeOut(self.stage1))
        self.set_title("Before: one edge at a time")
        col = Column()
        notes = col.place(
            caption_stack(["Walk the edge list.", "Move both endpoints", "immediately."])
        )
        self.play(FadeIn(notes[0]))
        sweep = [(0, 1), (2, 3), (4, 5), (7, 8), (6, 9), (0, 3), (8, 9)]
        for n, (i, j) in enumerate(sweep):
            gi, gj = self.old_edge_update(i, j)
            # A highlight that tracks the endpoints while they move (animating
            # the edge itself would suspend its updater mid-move).
            glow = Line(
                g.nodes[i].get_center(),
                g.nodes[j].get_center(),
                color=HIGHLIGHT_COLOR,
                stroke_width=9,
                stroke_opacity=0,
            )
            glow.add_updater(
                lambda l, i=i, j=j: l.put_start_and_end_on(
                    g.nodes[i].get_center(), g.nodes[j].get_center()
                )
            )
            self.bring_to_back(glow)
            anims = [
                UpdateFromAlphaFunc(
                    glow, lambda m, a: m.set_stroke(opacity=there_and_back(a))
                ),
                g.nodes[i].animate.shift(to3(gi)),
                g.nodes[j].animate.shift(to3(gj)),
            ]
            if n == 1:
                anims.append(FadeIn(notes[1:]))
            self.play(*anims, run_time=0.55 if n < 3 else 0.35)
            self.remove(glow)
        self.marked_next_slide(notes=NOTES["old_threads"])

        # Two threads touch edges that share node 5 at the same moment.
        shared, others = 5, (4, 9)
        tags, pushes = VGroup(), VGroup()
        for color, other, label in zip(THREAD_COLORS, others, ("thread 1", "thread 2")):
            edge = g.edge_mob(shared, other)
            push = force_arrow(
                g.nodes[shared].get_center()[:2],
                g.attraction(shared, other, self.emb),
                color,
                stroke_width=6,
            )
            tag = caption(label, 20, color=color).next_to(
                edge.get_center(), UP if other == 9 else LEFT, buff=0.15
            )
            tags.add(tag)
            pushes.add(push)
            self.play(
                edge.animate.set_stroke(color=color, width=8, opacity=1),
                FadeIn(tag),
                run_time=0.4,
            )
        clash = col.place(
            caption_stack(["Two threads can write", "the same point at once."])
        )
        self.play(*[GrowArrow(p) for p in pushes], FadeIn(clash))
        self.play(Wiggle(g.nodes[shared], scale_value=1.4, rotation_angle=0.05 * TAU))
        self.marked_next_slide(notes=NOTES["old_runs"])

        # Same seed, two interleavings: different layouts.
        d = self.data
        runs = np.stack([d["tiny_run_1"], d["tiny_run_2"]])
        mapping = fit_mapping(runs, 1.05, 1.05, ORIGIN, quantile=0.0)
        minis = VGroup()
        for k, run in enumerate(runs):
            mini = TinyGraph(
                np.array([mapping(p) for p in run]),
                d["tiny_edges"],
                d["tiny_weights"],
                node_radius=0.05,
            )
            frame = SurroundingRectangle(
                mini, buff=0.15, color=SECONDARY_COLOR, stroke_width=1.5
            )
            label = caption(f"run {k + 1}", 20).next_to(frame, DOWN, buff=0.12)
            minis.add(VGroup(frame, mini, label))
        minis.arrange(RIGHT, buff=0.7)
        neq = MathTex(r"\neq", font_size=48)
        same_seed = caption("Same seed, two runs:", 22)
        col.place(same_seed, buff=0.45)
        col.place(minis, buff=0.2)
        neq.move_to(minis.get_center())
        verdict = col.place(
            caption_stack(["So a seeded fit has to", "use a single thread."]), buff=0.3
        )
        self.play(FadeIn(same_seed), FadeIn(minis[0]), FadeIn(minis[1]), Write(neq))
        self.play(FadeIn(verdict, shift=UP * 0.1))
        self.stage2 = VGroup(notes, tags, pushes, clash, minis, neq, same_seed, verdict)
        self.marked_next_slide(notes=NOTES["node_gather"])

    # -- stage 3 ----------------------------------------------------------
    def stage_node_loop(self):
        g, s = self.graph, self.src
        for e, w in zip(g.edges, g.weights):
            e.set_stroke(
                color=SECONDARY_COLOR, width=1.5 + 4.0 * w, opacity=0.35 + 0.55 * w
            )
        self.play(
            FadeOut(self.stage2),
            *[
                n.animate.move_to(to3(p) + DOWN * 0.3)
                for n, p in zip(g.nodes, self.data["tiny_pos"])
            ],
        )
        self.set_title("Now: one node at a time")

        frozen = g.pos()
        ghosts = VGroup(
            *[
                Dot(to3(p), radius=0.2, color=SECONDARY_COLOR, fill_opacity=0.25)
                for p in frozen
            ]
        )
        col = Column()
        notes = VGroup(
            *[
                col.place(caption_stack(lines), buff=0.3)
                for lines in (
                    ["Positions frozen", "for the epoch."],
                    ["Each node sums", "its own forces."],
                    ["Then all nodes", "step together."],
                )
            ]
        )
        source_ring = Dot(
            g.nodes[s].get_center(),
            radius=0.19,
            color=SOURCE_COLOR,
            stroke_color=STRUCTURE_COLOR,
            stroke_width=3,
        )
        source_ring.add_updater(lambda m: m.move_to(g.nodes[s].get_center()))
        self.play(FadeIn(ghosts), FadeIn(notes[0]), FadeIn(source_ring, scale=1.5))
        self.bring_to_front(g.nodes[s])

        # Tip-to-tail sum for the source node.
        attract, repel = self.source_forces(frozen)
        vectors = [v for _, v in attract] + [v for _, v in repel]
        colors = [ATTRACT_COLOR] * len(attract) + [REPEL_COLOR] * len(repel)
        start = frozen[s]
        parts = VGroup(*[force_arrow(start, v, c) for v, c in zip(vectors, colors)])
        self.play(LaggedStart(*[GrowArrow(a) for a in parts], lag_ratio=0.15))
        tip = start.copy()
        moves = []
        for arrow, v in zip(parts, vectors):
            moves.append(arrow.animate.shift(to3(tip - start)))
            tip = tip + v * ARROW_SCALE
        self.play(LaggedStart(*moves, lag_ratio=0.35), run_time=2)
        total = np.sum(vectors, axis=0)
        resultant = force_arrow(start, total, STRUCTURE_COLOR, stroke_width=7)
        self.play(GrowArrow(resultant))
        self.play(FadeOut(parts))
        self.marked_next_slide(notes=NOTES["node_parallel"])

        # Every node, in parallel.
        rng = np.random.default_rng(7)
        resultants = VGroup()
        totals = []
        for i in range(len(frozen)):
            if i == s:
                totals.append(total)
                continue
            neigh = set(g.neighbors(i)) | {i}
            negs = rng.choice([k for k in range(len(frozen)) if k not in neigh], 2, False)
            vec = sum(g.attraction(i, j, self.emb, frozen) for j in g.neighbors(i))
            vec = vec + sum(g.repulsion(i, k, self.emb, frozen) for k in negs)
            totals.append(vec)
            resultants.add(force_arrow(frozen[i], vec, STRUCTURE_COLOR, stroke_width=7))
        self.play(
            LaggedStart(*[GrowArrow(a) for a in resultants], lag_ratio=0.05),
            FadeIn(notes[1]),
        )
        step_scale = 0.15 * ARROW_SCALE
        self.play(
            *[
                n.animate.shift(to3(v * step_scale))
                for n, v in zip(g.nodes, totals)
            ],
            FadeOut(resultants),
            FadeOut(resultant),
            FadeIn(notes[2]),
            run_time=1.5,
        )
        self.play(FadeOut(ghosts))

        owns = col.place(
            caption_stack(
                ["Each node writes only", "to itself: no races,", "any number of threads."],
                color=STRUCTURE_COLOR,
            ),
            buff=0.5,
        )
        code = VGroup(
            crisp_text(">>> np.array_equal(run_1, run_2)", font=MONO_FONT, font_size=20),
            crisp_text("True", font=MONO_FONT, font_size=20, weight=BOLD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        code.to_corner(DL, buff=0.6)
        box = SurroundingRectangle(code, buff=0.18, color=SECONDARY_COLOR, stroke_width=1.5)
        self.play(FadeIn(owns))
        self.play(Create(box), Write(code[0]))
        self.play(FadeIn(code[1], shift=RIGHT * 0.1))
        self.marked_next_slide(notes=NOTES["adam"])

    # -- stage 4 ----------------------------------------------------------
    def stage_adam(self):
        self.clear_slide()
        self.set_title("Per-node forces make Adam possible")
        d = self.data
        positions = d["toy_positions"].astype(float)
        labels = d["toy_labels"]
        h = self.meta["toy_node"]
        grads, steps, alphas = d["toy_node_grad"], d["toy_node_step"], d["toy_alpha"]
        n_epochs = len(alphas)

        mapping = fit_mapping(positions, 5.3, 5.3, (-3.5, -0.35), quantile=0.002)
        scene_pos = np.stack([mapping(p) for p in positions])  # (E+1, N, 2)

        epoch = ValueTracker(0.0)

        def pos_at(e):
            e = float(np.clip(e, 0, n_epochs))
            i = int(np.floor(e))
            j = min(i + 1, n_epochs)
            f = e - i
            return (1 - f) * scene_pos[i] + f * scene_pos[j]

        dots = VGroup(
            *[
                Dot(to3(p), radius=0.045, color=TOY_COLORS[l], fill_opacity=0.6)
                for p, l in zip(scene_pos[0], labels)
            ]
        )
        dots.add_updater(
            lambda m: [dot.move_to(to3(p)) for dot, p in zip(m, pos_at(epoch.get_value()))]
        )
        tracked = Dot(
            to3(scene_pos[0, h]),
            radius=0.1,
            color=SOURCE_COLOR,
            stroke_color=STRUCTURE_COLOR,
            stroke_width=2.5,
        )
        tracked.add_updater(lambda m: m.move_to(to3(pos_at(epoch.get_value())[h])))

        def path_points(e):
            k = int(np.floor(e))
            pts = [scene_pos[i, h] for i in range(k + 1)] + [pos_at(e)[h]]
            return [to3(p) for p in pts]

        def draw_path():
            path = VMobject(color=STRUCTURE_COLOR, stroke_width=3)
            pts = path_points(epoch.get_value())
            if len(pts) < 2:
                pts = pts + pts
            path.set_points_as_corners(pts)
            return path

        def draw_force_arrow():
            k = min(int(np.floor(epoch.get_value())), n_epochs - 1)
            g = grads[k]
            if np.linalg.norm(g) == 0:
                return VMobject()
            start = pos_at(epoch.get_value())[h]
            return Arrow(
                to3(start),
                to3(start + 0.6 * unit(g)),
                buff=0,
                color=REPEL_COLOR,
                stroke_width=5,
                max_tip_length_to_length_ratio=0.3,
            )

        path = always_redraw(draw_path)
        force = always_redraw(draw_force_arrow)

        # Charts: raw force on the tracked point, and the Adam step it takes.
        force_chart = styled_axes(
            x_range=[0, n_epochs, 50],
            y_range=[0, 5, 1],
            x_label="epoch",
            y_label="force on point",
            x_length=5.0,
            y_length=1.8,
            y_decimal_places=0,
        )
        step_chart = styled_axes(
            x_range=[0, n_epochs, 50],
            y_range=[0, 2.5, 0.5],
            x_label="epoch",
            y_label="Adam step",
            x_length=5.0,
            y_length=1.8,
            y_decimal_places=1,
        )
        force_chart.move_to(to3((3.7, 1.45)))
        step_chart.move_to(to3((3.7, -1.3)))
        for chart in (force_chart, step_chart):
            chart[0].x_axis.add_numbers([50, 100, 150, 200], font_size=16)

        f_axes, s_axes = force_chart[0], step_chart[0]
        force_norm = np.minimum(np.linalg.norm(grads, axis=1), 5.0)
        step_norm = np.linalg.norm(steps, axis=1)

        def partial_line(axes, values, color, width=2.5):
            def draw():
                k = int(np.clip(np.floor(epoch.get_value()), 0, n_epochs - 1))
                line = VMobject(color=color, stroke_width=width)
                pts = [axes.c2p(i, values[i]) for i in range(k + 1)]
                if len(pts) < 2:
                    pts = pts + pts
                line.set_points_as_corners(pts)
                return line

            return always_redraw(draw)

        force_line = partial_line(f_axes, force_norm, REPEL_COLOR, 2)
        step_line = partial_line(s_axes, step_norm, STRUCTURE_COLOR, 2.5)
        alpha_line = DashedVMobject(
            VMobject(color=SECONDARY_COLOR, stroke_width=3).set_points_as_corners(
                [s_axes.c2p(i, alphas[i]) for i in range(n_epochs)]
            ),
            num_dashes=70,
        )
        alpha_label = MathTex(r"\alpha", color=SECONDARY_COLOR, font_size=34).next_to(
            s_axes.c2p(8, alphas[8]), RIGHT, buff=0.12
        )

        self.play(
            FadeIn(dots, lag_ratio=0.002),
            FadeIn(tracked, scale=1.5),
            run_time=1.5,
        )
        self.play(FadeIn(force_chart), FadeIn(step_chart))
        self.play(Create(alpha_line), FadeIn(alpha_label))
        self.add(path, force, force_line, step_line, tracked)
        self.marked_next_slide(notes=NOTES["adam_run"])

        self.play(epoch.animate.set_value(n_epochs), run_time=9, rate_func=linear)
        for mob in (dots, tracked):
            mob.clear_updaters()
        for mob in (path, force, force_line, step_line):
            mob.clear_updaters()

        summary = caption_stack(
            [
                "Adam averages each point's force",
                "and its size, so steps stay under α,",
                "measured in embedding units.",
            ],
            font_size=20,
            buff=0.08,
        )
        summary.next_to(step_chart, DOWN, buff=0.25).align_to(step_chart, LEFT)
        momentum = caption(
            'optimizer="momentum": carry half of the last step forward', 18,
            color=SECONDARY_COLOR,
        ).to_corner(DL, buff=0.35)
        self.play(FadeIn(summary), FadeIn(momentum))
        self.marked_next_slide(notes=NOTES["clipping"])

    # -- stage 5 ----------------------------------------------------------
    def stage_clipping(self):
        self.clear_slide()
        self.set_title("Clipping: smooth, and only where needed")

        chart = styled_axes(
            x_range=[0, 3, 0.5],
            y_range=[0, 6, 1],
            x_label="distance",
            y_label="force magnitude",
            x_length=6.2,
            y_length=4.4,
            y_decimal_places=0,
        )
        chart.move_to(to3((-3.2, -0.5)))
        axes = chart[0]
        axes.x_axis.add_numbers([0.5, 1, 1.5, 2, 2.5], font_size=18)

        d_grid = np.geomspace(1e-4, 3.0, 800)
        rep = repulsion_magnitude(d_grid)
        right_six = d_grid[(rep > 6.0)].max()

        def curve(f, lo, color, dashed=False, width=4):
            xs = d_grid[d_grid >= lo]
            line = VMobject(color=color, stroke_width=width).set_points_as_corners(
                [axes.c2p(x, f(x)) for x in xs]
            )
            return DashedVMobject(line, num_dashes=45) if dashed else line

        attr = curve(lambda x: attraction_magnitude(x), 1e-4, ATTRACT_COLOR)
        raw = curve(lambda x: repulsion_magnitude(x), right_six, REPEL_COLOR, dashed=True, width=3)
        old = curve(lambda x: min(repulsion_magnitude(x), 4.0), 1e-4, OLD_STACK_COLOR, dashed=True, width=3)
        new = curve(lambda x: soft_clip(repulsion_magnitude(x)), 1e-4, REPEL_COLOR, width=5)
        gamma_line = DashedLine(
            axes.c2p(0, 1), axes.c2p(3, 1), color=SECONDARY_COLOR, stroke_width=2, dash_length=0.1
        )
        gamma_tag = MathTex(r"\gamma", color=SECONDARY_COLOR, font_size=32).next_to(
            axes.c2p(3, 1), RIGHT, buff=0.1
        )
        peak = VGroup(
            Arrow(axes.c2p(right_six, 5.2), axes.c2p(right_six, 6.2), buff=0,
                  color=REPEL_COLOR, stroke_width=3, max_tip_length_to_length_ratio=0.35),
            caption("raw repulsion peaks near 28", 18, color=REPEL_COLOR),
        )
        peak[1].next_to(peak[0], UP, buff=0.08).align_to(peak[0], LEFT).shift(LEFT * 0.1)

        def legend_row(color, text, dashed=False, width=4):
            sample = Line(ORIGIN, RIGHT * 0.5, color=color, stroke_width=width)
            if dashed:
                sample = DashedLine(ORIGIN, RIGHT * 0.5, color=color,
                                    stroke_width=width, dash_length=0.07)
            return VGroup(sample, caption(text, 18, color=color)).arrange(RIGHT, buff=0.15)

        attr_tag = legend_row(ATTRACT_COLOR, "attraction: bounded, never clipped")
        raw_tag = legend_row(REPEL_COLOR, "raw repulsion", dashed=True, width=3)
        old_tag = legend_row(OLD_STACK_COLOR, "old: each coordinate clamped at 4", dashed=True, width=3)
        new_tag = legend_row(REPEL_COLOR, "new: saturates smoothly at γ", width=5)
        legend = VGroup(attr_tag, raw_tag, old_tag, new_tag).arrange(
            DOWN, aligned_edge=LEFT, buff=0.16
        )
        legend.move_to(axes.c2p(0.95, 5.6), aligned_edge=UL)

        self.play(FadeIn(chart))
        self.play(Create(attr), FadeIn(attr_tag))
        self.play(Create(raw), GrowArrow(peak[0]), FadeIn(peak[1]), FadeIn(raw_tag))
        self.play(Create(old), FadeIn(old_tag))
        self.marked_next_slide(notes=NOTES["clip_new"])
        self.play(Create(new), Create(gamma_line), FadeIn(gamma_tag), FadeIn(new_tag))
        self.marked_next_slide(notes=NOTES["clip_direction"])

        # Square clamp rotates the vector; round clamp keeps its direction.
        raw_vec = np.array([10.0, 3.0])
        panels = VGroup()
        for k, title in enumerate(("old: square clamp", "new: round clamp")):
            center = np.array([2.35 + 2.75 * k, 0.9])
            origin = to3(center)
            axes_cross = VGroup(
                Line(origin + LEFT * 1.1, origin + RIGHT * 1.1, color=SECONDARY_COLOR, stroke_width=1),
                Line(origin + DOWN * 1.1, origin + UP * 1.1, color=SECONDARY_COLOR, stroke_width=1),
            )
            if k == 0:
                bound = Square(side_length=1.6, color=OLD_STACK_COLOR, stroke_width=2.5).move_to(origin)
                clipped = np.clip(raw_vec, -4, 4) * 0.2
            else:
                bound = Circle(radius=0.8, color=REPEL_COLOR, stroke_width=2.5).move_to(origin)
                clipped = unit(raw_vec) * 0.8 * np.tanh(np.linalg.norm(raw_vec))
            raw_line = DashedLine(
                origin, origin + to3(unit(raw_vec) * 1.45),
                color=REPEL_COLOR, stroke_width=2.5, dash_length=0.08,
            ).set_opacity(0.6)
            vec = Arrow(origin, origin + to3(clipped), buff=0, color=REPEL_COLOR,
                        stroke_width=5, max_tip_length_to_length_ratio=0.25)
            angle_deg = np.degrees(np.arctan2(clipped[1], clipped[0]))
            arc = Arc(radius=0.45, start_angle=0, angle=np.radians(angle_deg),
                      arc_center=origin, color=STRUCTURE_COLOR, stroke_width=2)
            angle_txt = caption(f"{angle_deg:.0f}°", 18).next_to(arc, RIGHT, buff=0.08).shift(UP * 0.05)
            head = caption(title, 20).next_to(axes_cross, UP, buff=0.2)
            panels.add(VGroup(axes_cross, bound, raw_line, vec, arc, angle_txt, head))
        formula = MathTex(
            r"\tilde F \;=\; \gamma\,\tanh\!\left(\frac{\lVert F\rVert}{\gamma}\right)"
            r"\frac{F}{\lVert F\rVert}",
            font_size=38,
        ).move_to(to3((3.7, -1.7)))
        raw_note = caption("dashed: raw force direction (17°)", 18, color=SECONDARY_COLOR)
        raw_note.next_to(panels, DOWN, buff=0.25)
        self.play(FadeIn(panels[0]))
        self.play(FadeIn(panels[1]), FadeIn(raw_note))
        self.play(Write(formula))
        self.marked_next_slide(notes=NOTES["recap"])

    # -- stage 6 ----------------------------------------------------------
    def stage_recap(self):
        self.clear_slide()
        self.set_title("What the node loop buys")
        rows = VGroup(
            *[
                VGroup(check_mark(), caption(text, 30)).arrange(RIGHT, buff=0.35)
                for text in (
                    "Reproducible with a fixed seed",
                    "Fully parallel",
                    "Stable, bounded steps",
                )
            ]
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        rows.move_to(to3((-2.6, 1.3)), aligned_edge=LEFT)
        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.2) for r in rows], lag_ratio=0.4))

        t = self.meta["timings"]
        entries = [
            ("compatibility · seeded · 1 thread", t["compatibility_seconds"], OLD_STACK_COLOR),
            ("Adam · seeded · all threads", t["adam_seconds"], NEW_STACK_COLOR),
        ]
        longest = max(v for _, v, _ in entries)
        bars = VGroup()
        for label, value, color in entries:
            bar = Rectangle(
                width=6.0 * value / longest, height=0.42,
                fill_color=color, fill_opacity=1, stroke_width=0,
            )
            name = caption(label, 22).next_to(bar, LEFT, buff=0.25)
            val = caption(f"{value:.1f} s", 22).next_to(bar, RIGHT, buff=0.2)
            bars.add(VGroup(name, bar, val))
        bars.arrange(DOWN, buff=0.3)
        for row in bars:
            row[1].align_to(bars[0][1], LEFT)
            row[0].next_to(row[1], LEFT, buff=0.25)
            row[2].next_to(row[1], RIGHT, buff=0.2)
        bars.move_to(to3((0.4, -1.55)))
        source = caption("MNIST, full fit, warm timings", 18, color=SECONDARY_COLOR)
        source.next_to(bars, DOWN, buff=0.3)
        self.play(
            *[GrowFromEdge(row[1], LEFT) for row in bars],
            *[FadeIn(row[0]) for row in bars],
            FadeIn(source),
        )
        self.play(*[FadeIn(row[2]) for row in bars])
        self.marked_next_slide(notes=NOTES["takeaway"])

        self.clear_slide()
        self.takeaway("Loop over nodes, not edges.", font_size=64)
        self.start_section_wipe(SECTION_TITLES["negatives"], auto_next=True)


# ---------------------------------------------------------------------------
# Speaker notes (shown by manim-slides present / html / pptx)
# ---------------------------------------------------------------------------
NOTES = {
    "setup": (
        "After building the graph, UMAP lays it out by simulating forces. "
        "Edges pull neighbours together; randomly chosen negative samples push "
        "points apart."
    ),
    "old_sweep": (
        "The classic optimizer walks the edge list. Each edge moves both of its "
        "endpoints immediately, and so do its negative samples."
    ),
    "old_threads": (
        "In parallel, threads work through different edges, but edges share "
        "points. Two threads can update the same point at the same moment, and "
        "the result depends on timing."
    ),
    "old_runs": (
        "Same data, same seed, different thread timing: different layouts. "
        "That is why a seeded fit has always fallen back to a single thread."
    ),
    "node_gather": (
        "The new kernels turn the loop inside out. Positions are frozen for the "
        "epoch. Each node gathers every force acting on it into its own buffer: "
        "attraction along its edges, repulsion from its negative samples."
    ),
    "node_parallel": (
        "Every node does this at once, reading frozen positions and writing only "
        "to itself. Then all nodes take their step together. Negative samples "
        "come from a deterministic hash into a seeded shuffle, so two runs with "
        "the same seed are bit-for-bit identical, on any number of threads."
    ),
    "adam": (
        "There is a second payoff. A per-node force for each epoch is a gradient, "
        "and that is exactly what Adam needs."
    ),
    "adam_run": (
        "Red is the raw force on this one point, epoch by epoch: noisy, because "
        "which edges and negatives fire changes every epoch. Adam keeps a running "
        "average of that force and of its square, and steps by their ratio. So "
        "the step, bottom, stays under the learning rate alpha, in embedding "
        "units. The learning rate is now a distance. The momentum optimizer is "
        "the simpler alternative: it carries half of the previous step forward."
    ),
    "clipping": (
        "Clipping changed too. Attraction is naturally bounded, so it is no longer "
        "clipped at all. Raw repulsion spikes for close points. The old code "
        "clamped every coordinate at four."
    ),
    "clip_new": (
        "The new code clips the length of each repulsive force with a smooth "
        "tanh, saturating at gamma, the repulsion strength. Gamma now sets the "
        "cap, so it can be turned up safely; more on that at the end."
    ),
    "clip_direction": (
        "Clamping coordinates separately also turns the force: this vector at 17 "
        "degrees comes out at 37. Clipping the length keeps the direction. This "
        "matters more under Adam, which normalises the sum of a node's forces: "
        "one near-collision would otherwise decide which way the whole step goes."
    ),
    "recap": (
        "So: reproducible with a seed, fully parallel, and stable bounded steps. "
        "On MNIST, a seeded fit goes from about 42 seconds on one thread to "
        "under 6 on all of them. That includes the new initialization as well."
    ),
    "takeaway": "One idea carries this section: loop over nodes, not edges.",
}
