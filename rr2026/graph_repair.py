"""Repairing the graph: fitting a layout jointly, with a heavy-tailed kernel,
weighs every edge against the whole graph.

    python assets_graph_repair.py      # once, writes assets/graph_repair.*
    manim-slides render graph_repair.py GraphRepair

Also reads assets/where_next.* (for the MNIST and strands callbacks).
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

from where_next import (section_title, Column, angular_gap, caption, caption_stack, fit_points, loop_color, segment_cloud,
                        to3, undirected, WRONG_COLOR, darker)

ASSETS = Path(__file__).parent / "assets"
RING_EDGE_COLOR = STRUCTURE_COLOR
SPRING_COLOR = COLOR_CYCLE[6]
HEAVY_COLOR = STRUCTURE_COLOR


def ring_points(n, radius, centre, offset=PI / 2):
    t = offset - 2 * PI * np.arange(n) / n
    return np.c_[radius * np.cos(t), radius * np.sin(t)] + np.asarray(centre)


class GraphRepair(UMAPSlide):
    def construct(self):
        self.data = np.load(ASSETS / "graph_repair.npz")
        self.meta = json.loads((ASSETS / "graph_repair.json").read_text())
        self.stats = self.meta["stats"]
        self.end_section_wipe(section_title("repair"), next_slide_prep=lambda: None, notes=NOTES["r01_local"])
        self.stage_local_vs_global()
        self.stage_likelihood()
        self.stage_test_graph()
        self.stage_race()
        self.stage_influence()
        self.stage_compliance()
        self.stage_mnist()
        self.stage_limits()
        self.stage_takeaway()

    # -- 1. local evidence can be fooled ------------------------------------------------------
    def stage_local_vs_global(self):
        self.set_title("Local evidence can be fooled")
        n, radius, centre = 24, 2.45, np.array([-2.6, -0.4])
        P = ring_points(n, radius, centre)
        ring = [(i, (i + s) % n) for i in range(n) for s in (1, 2)]
        chords = [(3, 15), (3, 16), (4, 15)]
        ring_lines = segment_cloud(P[[a for a, _ in ring]], P[[b for _, b in ring]], RING_EDGE_COLOR, 2, 0.7)
        chord_lines = segment_cloud(P[[a for a, _ in chords[1:]]], P[[b for _, b in chords[1:]]], WRONG_COLOR, 3, 0.9)
        focus = Line(to3(P[3]), to3(P[15]), color=HIGHLIGHT_COLOR, stroke_width=5)
        dots = VGroup(*[Dot(to3(p), radius=0.09, color=loop_color(i / n)) for i, p in enumerate(P)])
        shared = VGroup(*[Circle(radius=0.2, color=HIGHLIGHT_COLOR, stroke_width=3).move_to(to3(P[i])) for i in (4, 16)])

        col = Column()
        c1 = col.place(caption_stack(["Three wrong edges bridge the ring.", "The highlighted one's endpoints",
                                      "share two neighbours, exactly as", "a genuine ring edge's do."]))
        c2 = col.place(caption_stack(["Local consensus, counting shared", "neighbours, trusts it."],
                                     color=STRUCTURE_COLOR), buff=0.3)
        self.play(Create(ring_lines), LaggedStart(*[FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.03), run_time=1.5)
        self.play(Create(chord_lines), Create(focus))
        self.bring_to_front(dots)
        self.play(Create(shared), FadeIn(c1))
        self.play(FadeIn(c2))
        self.marked_next_slide(notes=NOTES["r03_global"])

        start = PI / 2 - 2 * PI * 3 / n
        paths = VGroup(*[Arc(radius=radius, start_angle=start, angle=sign * PI, color=ATTRACT_COLOR, stroke_width=9,
                             stroke_opacity=0.45).move_arc_center_to(to3(centre)) for sign in (-1, 1)])
        c3 = col.place(caption_stack(["Globally, every other path through", "the graph puts them far apart.",
                                      "Fitting a layout jointly weighs each", "edge against all of that evidence."]),
                       buff=0.45)
        self.play(FadeOut(shared), Create(paths), run_time=1.5)
        self.play(FadeIn(c3))
        self.marked_next_slide(notes=NOTES["r04_likelihood"])

    # -- 2. the objective ----------------------------------------------------------------------
    def stage_likelihood(self):
        self.clear_slide()
        self.set_title("A likelihood, not a heuristic")
        F = MathTex(r"F(Y)", "=", r"\sum_{\text{edges}} w\,\varphi(d_{ij})", "+",
                    r"\gamma\!\!\sum_{\text{all pairs}}\!\!\rho(d_{ij})", font_size=44)
        F[2].set_color(ATTRACT_COLOR)
        F[4].set_color(REPEL_COLOR)
        F.move_to(to3((0, 1.7)))
        defs = MathTex(r"\varphi = -\log q, \qquad \rho = -\log(1-q)", font_size=32).next_to(F, DOWN, buff=0.35)
        notes = caption_stack(["Each pair of points is a coin flip, linked with probability q(d),",
                               "which falls with their distance in the layout.",
                               "Edges attract, every pair repels, and \u03b3 sets the balance."], font_size=24)
        notes.next_to(defs, DOWN, buff=0.5)
        self.play(Write(F), run_time=1.5)
        self.play(FadeIn(defs), FadeIn(notes))
        self.marked_next_slide(notes=NOTES["r05_target"])

        target = MathTex(r"q^{*} \;=\; \frac{w}{w+\gamma}", font_size=44).move_to(to3((-3.8, -2.3)))
        box = SurroundingRectangle(target, buff=0.2, color=HIGHLIGHT_COLOR, stroke_width=2)
        tnote = caption_stack(["A single edge always settles at this probability, whatever",
                               "the link function q. The kernel only decides how far apart",
                               "that probability puts the points: more in the next section."], font_size=22)
        tnote.next_to(box, RIGHT, buff=0.5)
        self.play(Write(target), Create(box))
        self.play(FadeIn(tnote))
        self.marked_next_slide(notes=NOTES["r06_test"])

    # -- 3. the test graph ---------------------------------------------------------------------
    def stage_test_graph(self):
        self.clear_slide()
        self.set_title("A test: a circle with wrong and missing edges")
        d, meta = self.data, self.meta
        theta = d["theta"]
        P = fit_points(np.c_[np.cos(theta), np.sin(theta)], 5.4, 5.4, (-2.6, -0.45))
        kept, shortcuts, missing = d["kept"], d["shortcuts"], d["missing"]
        ring = segment_cloud(P[kept[:, 0]], P[kept[:, 1]], RING_EDGE_COLOR, 0.8, 0.35)
        gone = segment_cloud(P[missing[:, 0]], P[missing[:, 1]], HIGHLIGHT_COLOR, 3, 0.9)
        wrong = segment_cloud(P[shortcuts[:, 0]], P[shortcuts[:, 1]], WRONG_COLOR, 2.2, 0.9)
        dots = VGroup(*[Dot(to3(p), radius=0.04, color=loop_color(t / (2 * PI))) for p, t in zip(P, theta)])
        col = Column()
        c1 = col.place(caption_stack([f"{meta['n']} points on a circle, each joined", "to its 12 nearest along it."]))
        c2 = col.place(caption_stack([f"Then {meta['shortcuts']} random shortcuts are added", "across the circle..."],
                                     color=darker(ManimColor(WRONG_COLOR), 0.2)), buff=0.3)
        c3 = col.place(caption_stack([f"...and {meta['missing']} true edges removed", "(marked in orange)."],
                                     color=darker(ManimColor(HIGHLIGHT_COLOR), 0.3)), buff=0.3)
        c4 = col.place(caption_stack(["A layout sees only this graph.", "Can it get the circle back?"],
                                     color=STRUCTURE_COLOR), buff=0.45)
        self.play(FadeIn(dots), Create(ring), FadeIn(c1), run_time=1.2)
        self.play(Create(wrong), FadeIn(c2))
        self.play(FadeIn(gone), FadeIn(c3))
        self.play(FadeOut(gone, run_time=1.2), FadeIn(c4))
        self.marked_next_slide(notes=NOTES["r07_race"])

    # -- 3b. spring vs heavy tail ---------------------------------------------------------------
    def stage_race(self):
        self.clear_slide()
        self.set_title("A spring folds the circle; a heavy tail restores it")
        d, st = self.data, self.stats
        theta = d["theta"]
        colours = [loop_color(t / (2 * PI)) for t in theta]
        kept, shortcuts = d["kept"], d["shortcuts"]
        panels = {"spring": (-3.45, "spring attraction (Gaussian link)"),
                  "cauchy": (3.45, "heavy-tailed attraction (Cauchy link)")}
        step = ValueTracker(0.0)
        n_snaps = len(d["spring_snapshots"])
        mobs = []
        for kernel, (x, label) in panels.items():
            snaps = d[f"{kernel}_snapshots"]

            def draw(snaps=snaps, x=x):
                s = int(round(np.clip(step.get_value(), 0, n_snaps - 1)))
                P = fit_points(snaps[s], 5.0, 4.2, (x, -0.35))
                return VGroup(segment_cloud(P[kept[:, 0]], P[kept[:, 1]], RING_EDGE_COLOR, 0.6, 0.3),
                              segment_cloud(P[shortcuts[:, 0]], P[shortcuts[:, 1]], WRONG_COLOR, 1.8, 0.85),
                              VGroup(*[Dot(to3(p), radius=0.035, color=c) for p, c in zip(P, colours)]))

            panel = always_redraw(draw)
            head = caption(label, 22).move_to(to3((x, 2.5)))
            mobs.append((kernel, x, panel, head))
        start_note = caption("both start from the same spectral layout of the graph", 18,
                             color=SECONDARY_COLOR).to_edge(DOWN, buff=0.3)
        for kernel, x, panel, head in mobs:
            self.add(panel)
            self.play(FadeIn(head), run_time=0.5)
        self.play(FadeIn(start_note))
        self.marked_next_slide(notes=NOTES["r08_result"])

        self.play(step.animate.set_value(n_snaps - 1), run_time=9, rate_func=smooth)
        for _, _, panel, _ in mobs:
            panel.clear_updaters()
        readouts = VGroup()
        for kernel, x, _, _ in mobs:
            s = st[kernel]
            r = VGroup(MetricReadout("circle distortion", s["roundness"], num_decimal_places=2, font_size=20),
                       MetricReadout("shortcut \u00f7 ring edge", s["stretch"], num_decimal_places=1, font_size=20),
                       MetricReadout(f"missing edges restored (of {self.meta['missing']})", s["missing_restored"],
                                     num_decimal_places=0, font_size=20)).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
            r.move_to(to3((x, -3.35)))
            readouts.add(r)
        self.play(FadeOut(start_note), FadeIn(readouts))
        self.marked_next_slide(notes=NOTES["r09_influence"])

    # -- 3c. why: influence --------------------------------------------------------------------------
    def stage_influence(self):
        self.clear_slide()
        self.set_title("Bounded influence, not bounded cost")
        chart = styled_axes([0, 5, 1], [0, 2.5, 0.5], "edge length in the layout", "pull on the edge",
                            x_length=6.4, y_length=4.4, y_decimal_places=1)
        chart.move_to(to3((-2.6, -0.45)))
        axes = chart[0]
        for v in range(1, 6):
            chart.add(caption(str(v), 16).next_to(axes.c2p(v, 0), DOWN, buff=0.18))
        spring = axes.plot(lambda x: 2 * x, x_range=[0, 1.25], color=SPRING_COLOR, stroke_width=4)
        heavy = axes.plot(lambda x: 2 * x / (1 + x * x), x_range=[0, 5], color=HEAVY_COLOR, stroke_width=4)
        s_label = caption("spring: keeps growing", 20, color=darker(ManimColor(SPRING_COLOR), 0.2))
        s_label.next_to(axes.c2p(1.25, 2.5), RIGHT, buff=0.15)
        h_label = caption("heavy tail: fades like 1/d", 20).next_to(axes.c2p(3.2, 0.62), UP, buff=0.1)
        col = Column()
        c1 = col.place(caption_stack(["A spring pulls harder the longer", "an edge is stretched, so a far",
                                      "wrong edge wins."]))
        c2 = col.place(caption_stack(["A heavy tail pulls weakly on long", "edges, so the rest of the graph",
                                      "can hold them off."]), buff=0.3)
        chord = self.stats["chord"]
        c3 = col.place(caption_stack(["One wrong chord across a clean", "circle pulls its ends together by:"],
                                     color=STRUCTURE_COLOR), buff=0.45)
        r1 = col.place(MetricReadout("spring", 100 * chord["spring"], num_decimal_places=0, font_size=22,
                                     unit=r"\%"), buff=0.15)
        r2 = col.place(MetricReadout("heavy tail", 100 * chord["cauchy"], num_decimal_places=0, font_size=22,
                                     unit=r"\%"), buff=0.1)
        self.play(FadeIn(chart))
        self.play(Create(spring), FadeIn(s_label), FadeIn(c1))
        self.play(Create(heavy), FadeIn(h_label), FadeIn(c2))
        self.play(FadeIn(c3), FadeIn(r1), FadeIn(r2))
        self.marked_next_slide(notes=NOTES["r10_compliance"])

    # -- 4. compliance -------------------------------------------------------------------------------
    def stage_compliance(self):
        self.clear_slide()
        self.set_title("The rest of the graph pushes back")
        d = self.data
        theta, kept, shortcuts = d["theta"], d["kept"], d["shortcuts"]
        final = d["cauchy_snapshots"][-1]
        P = fit_points(final, 5.0, 5.0, (-3.3, -0.45))
        order = np.argsort(theta)
        # Draw the ring through a circular moving average of the points in
        # circle order: neighbouring points sit at slightly different radii,
        # so a line through the raw positions zigzags.
        window = 9
        ordered = P[order]
        padded = np.r_[ordered[-window:], ordered, ordered[:window]]
        kernel = np.ones(2 * window + 1) / (2 * window + 1)
        smooth = np.c_[np.convolve(padded[:, 0], kernel, "valid"), np.convolve(padded[:, 1], kernel, "valid")]
        smooth_of = {int(i): smooth[r] for r, i in enumerate(order)}
        ring = VMobject(stroke_color=RING_EDGE_COLOR, stroke_width=2, stroke_opacity=0.6)
        ring.set_points_smoothly([to3(p) for p in np.r_[smooth, smooth[:1]]])
        others = segment_cloud(P[shortcuts[:, 0]], P[shortcuts[:, 1]], WRONG_COLOR, 1.5, 0.5)
        lengths = np.linalg.norm(P[shortcuts[:, 0]] - P[shortcuts[:, 1]], axis=1)
        a, b = shortcuts[np.argmax(lengths)]
        focus = Line(to3(P[a]), to3(P[b]), color=HIGHLIGHT_COLOR, stroke_width=5)
        ia, ib = int(np.flatnonzero(order == a)[0]), int(np.flatnonzero(order == b)[0])
        lo, hi = min(ia, ib), max(ia, ib)
        arcs = VGroup()
        for path in (order[lo:hi + 1], np.r_[order[hi:], order[:lo + 1]]):
            arc = VMobject(stroke_color=ATTRACT_COLOR, stroke_width=9, stroke_opacity=0.45)
            arc.set_points_smoothly([to3(smooth_of[int(i)]) for i in path])
            arcs.add(arc)
        col = Column((1.2, 2.45))
        f1 = col.place(MathTex(r"\dot D \;=\; -\,w\,\psi(D)\,C", font_size=38))
        c1 = col.place(caption_stack(["C, the edge's compliance: how far its ends move per", "unit of pull once the whole layout adjusts. An",
                                      "effective resistance over every path, not just shared", "neighbours."]), buff=0.25)
        f2 = col.place(MathTex(r"R_{\mathrm{rej}} \;\approx\; \sqrt{2\,\nu p\,w\,C}", font_size=38), buff=0.45)
        c2 = col.place(caption_stack(["An edge cannot pull together a pair that the rest of", "the graph holds further apart than this. \u03bdp is set",
                                      "by the kernel: next section."]), buff=0.25)
        self.play(Create(ring), FadeIn(others), run_time=1.2)
        self.play(Create(focus))
        self.play(Create(arcs), Write(f1), FadeIn(c1), run_time=1.5)
        self.play(Write(f2), FadeIn(c2))
        self.marked_next_slide(notes=NOTES["r12_mnist"])

    # -- 6. UMAP already does some of this --------------------------------------------------------------
    def stage_mnist(self):
        self.clear_slide()
        self.set_title("UMAP already does some of this")
        w = np.load(ASSETS / "where_next.npz")
        labels = w["mnist_labels"]
        P = fit_points(w["mnist_layout"], 6.2, 5.4, (-3.0, -0.4))
        cloud = EmbeddingCloud(P, labels_to_rgb(labels, digit_rgb()), fit="none", point_px=3.0)
        pairs, long = w["mnist_pairs"], w["mnist_long"].astype(bool)
        edges = segment_cloud(P[pairs[long, 0]], P[pairs[long, 1]], STRUCTURE_COLOR, 1.0, 0.55)
        cross = float(np.mean(labels[pairs[long, 0]] != labels[pairs[long, 1]]))
        col = Column((1.2, 2.2))
        c1 = col.place(caption_stack(["MNIST's long nearest-neighbour edges again.", "Each is in UMAP's own graph, with the",
                                      "largest weight its point gives any neighbour."]))
        c2 = col.place(caption_stack(["Yet the layout put the two ends far apart:", "the rest of the graph overruled the edge.",
                                      f"{cross:.0%} of these edges join different digits."]), buff=0.3)
        c3 = col.place(caption_stack(["UMAP's kernel has a heavy tail. Built as a", "repair tool, the same mechanism can do more."],
                                     color=STRUCTURE_COLOR), buff=0.45)
        self.add(cloud)
        self.play(PMFadeIn(cloud), run_time=1.2)
        self.play(PMDim(cloud, 0.65), FadeIn(edges), FadeIn(c1))
        self.play(FadeIn(c2))
        self.play(FadeIn(c3))
        self.marked_next_slide(notes=NOTES["r13_limits"])

    # -- 7. limits ---------------------------------------------------------------------------------------
    def stage_limits(self):
        self.clear_slide()
        self.set_title("Where consensus fails: coherent bridges")
        d, st = self.data, self.stats
        xs = np.linspace(-5.5, 5.5, len(self.meta["bundles"]))
        panels = VGroup()
        for x, m in zip(xs, self.meta["bundles"]):
            Y = d[f"bundle_{m}"]
            chords = d[f"bundle_{m}_chords"]
            P = fit_points(Y, 2.1, 2.1, (x, 0.35))
            ring = VMobject(stroke_color=RING_EDGE_COLOR, stroke_width=2, stroke_opacity=0.7)
            ring.set_points_as_corners([to3(p) for p in np.r_[P, P[:1]]])
            bridges = segment_cloud(P[chords[:, 0]], P[chords[:, 1]], WRONG_COLOR, 1.6, 0.9)
            label = caption(f"{m} aligned bridge" + ("s" if m > 1 else ""), 18).move_to(to3((x, -1.2)))
            value = caption(f"ends pulled {st['bundles'][str(m)]:.0%} closer", 16,
                            color=SECONDARY_COLOR).next_to(label, DOWN, buff=0.1)
            panels.add(VGroup(ring, bridges, label, value))
        note = caption_stack(["Aligned wrong edges reinforce each other.",
                              "Past about ten, they win: the circle folds."], font_size=24, color=STRUCTURE_COLOR)
        note.to_edge(DOWN, buff=0.45)
        self.play(LaggedStart(*[FadeIn(p) for p in panels], lag_ratio=0.25), run_time=2)
        self.play(FadeIn(note))
        self.marked_next_slide(notes=NOTES["r14_data"])

        self.clear_slide()
        self.set_title("The graph can't tell, but the data can")
        w = np.load(ASSETS / "where_next.npz")
        wm = json.loads((ASSETS / "where_next.json").read_text())["strands"]
        pc, lab = w["strands_pc"], w["strands_labels"]
        bins = np.linspace(pc.min(), pc.max(), 31)
        h0, h1 = np.histogram(pc[lab == 0], bins)[0], np.histogram(pc[lab == 1], bins)[0]
        top = max(h0.max(), h1.max())
        base = np.array([-6.0, -2.4])
        width = 6.0 / (len(bins) - 1)
        hist = VGroup()
        for k in range(len(bins) - 1):
            for h, colour in ((h0[k], COLOR_CYCLE[0]), (h1[k], COLOR_CYCLE[1])):
                if h:
                    hist.add(Rectangle(width=width * 0.9, height=4.0 * h / top, stroke_width=0, fill_color=colour,
                                       fill_opacity=0.65).move_to(to3(base + [width * (k + 0.5), 2.0 * h / top])))
        axis = VGroup(Line(to3(base), to3(base + [6.0, 0]), color=STRUCTURE_COLOR, stroke_width=2),
                      caption("a principal component of the raw data", 20).move_to(to3(base + [3.0, -0.35])))
        col = Column((1.2, 2.2))
        c1 = col.place(caption_stack([f"The two strands from earlier: {wm['bridging']:.0%} of the", "graph's edges join them, so no graph-only",
                                      "method can separate them."]))
        c2 = col.place(caption_stack([f"A principal-component split of the raw data", f"separates them {wm['pca_accuracy']:.0%} of the time."]),
                       buff=0.3)
        c3 = col.place(caption_stack(["Repairing coherent errors needs information", "from the data, not just the graph:",
                                      "the leading open direction."], color=STRUCTURE_COLOR), buff=0.45)
        self.play(FadeIn(axis), LaggedStart(*[GrowFromEdge(b, DOWN) for b in hist], lag_ratio=0.01), FadeIn(c1),
                  run_time=1.5)
        self.play(FadeIn(c2))
        self.play(FadeIn(c3))
        self.marked_next_slide(notes=NOTES["r15_takeaway"])

    def stage_takeaway(self):
        self.clear_slide()
        self.takeaway("Fit jointly, with a heavy tail,", "and the layout repairs the graph.")
        self.start_section_wipe(section_title("kernel"), auto_next=True)


# ---------------------------------------------------------------------------
# Speaker notes
# ---------------------------------------------------------------------------
NOTES = {
    "r01_local": (
        "The obvious repair is local consensus: trust an edge if its endpoints share neighbours. "
        "It helps, but it only sees each point's immediate neighbourhood. Here three wrong edges "
        "bridge the ring. The highlighted one's endpoints share two neighbours, exactly as a "
        "genuine ring edge's do, so local consensus trusts it."
    ),
    "r03_global": (
        "Globally, though, every other path puts the two ends far apart. The idea: fit a layout "
        "jointly, so each edge is weighed against the evidence of the whole graph."
    ),
    "r04_likelihood": (
        "The objective is derived, not chosen. Treat each pair of points as a coin flip, linked "
        "with a probability that falls with distance. With case-control weighting and label "
        "smoothing, the negative log-likelihood is exactly this: edges attract, every pair repels."
    ),
    "r05_target": (
        "A notable fact, and a two-line calculation: a single edge always settles at probability "
        "w over w plus gamma, whatever the link function. The kernel decides how far apart that "
        "puts the points, not what probability the edge targets. We'll come back to that."
    ),
    "r06_test": (
        "A toy test. Two hundred points on a circle, each joined to its twelve nearest along the "
        "circle. Add twelve random shortcuts across it, and remove forty-eight true edges. The "
        "layout only ever sees this graph."
    ),
    "r07_race": (
        "Two layouts of that same graph, from the same spectral start. On the left, spring "
        "attraction, a Gaussian link. On the right, heavy-tailed attraction, a Cauchy link. "
        "Everything else is identical."
    ),
    "r08_result": (
        "The spring gives in to the shortcuts and folds the circle into a figure eight. The heavy "
        "tail keeps it round, stretches every shortcut straight across, and restores most of the "
        "missing edges. This is one toy graph, but it shows the mechanism working."
    ),
    "r09_influence": (
        "Why? Look at how hard an edge pulls as a function of its length. A spring pulls harder "
        "the more it's stretched, so a long wrong edge wins. A heavy tail's pull fades like one "
        "over d, so the rest of the graph can hold it off. The criterion is bounded influence, not "
        "bounded cost: the Cauchy cost itself is unbounded. In this toy, one wrong chord across a "
        "clean circle pulls its ends 78 percent closer with a spring, and 3 percent with the "
        "heavy tail."
    ),
    "r10_compliance": (
        "How hard the rest of the graph pushes back is the edge's compliance: an effective "
        "resistance over every path, not just shared neighbours. The theory gives a rejection "
        "radius: an edge cannot pull together a pair the rest of the graph holds further apart "
        "than about this. The factor nu p comes from the kernel, which is the next section."
    ),
    "r12_mnist": (
        "UMAP already does some of this. These are MNIST's long nearest-neighbour edges from "
        "earlier. Each is in UMAP's graph with full weight, yet the layout placed the two ends "
        "far apart. UMAP's kernel has a heavy tail; built deliberately as a repair tool, the same "
        "mechanism can do more."
    ),
    "r13_limits": (
        "The limit: coherent errors. Aligned wrong edges reinforce each other. In this toy, one "
        "is rejected; by about ten to twelve they win and the circle folds."
    ),
    "r14_data": (
        "This is a limit of graph-only methods, not of the problem. The two strands from earlier "
        "are joined by forty percent of the graph's edges, but a principal-component split of the "
        "raw data separates them. Using the data itself to repair coherent errors is the leading "
        "open direction."
    ),
    "r15_takeaway": (
        "So: fit jointly, with a heavy-tailed kernel, and the layout repairs the graph, at least "
        "for incoherent errors. The kernel's tail is doing the work, which brings us to the kernel."
    ),
}
