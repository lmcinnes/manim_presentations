"""Class 3, transition-coarsening version: recursive initialization where
coarse-graph edges are built from each part's share of its connections.

Sits alongside recursive_init.py. It reuses that module's RecursiveInit and
replaces only the coarse-edge stage (and its speaker notes); everything else
is the same story, drawn from assets computed with transition coarsening.

    python assets_recursive.py --coarsening transition   # writes assets/recursive_transition.*
    manim-slides render recursive_init_transition.py RecursiveInitTransition

The assets need the label_prop transition-coarsening patch.
"""

import sys

sys.path.append("..")  # Add parent directory to path to import config
sys.path.append(".")

import recursive_init as base
from recursive_init import *  # noqa: F401,F403  (manim, config names, helpers, RecursiveInit)

NOTES_TRANSITION = {
    "t1_shares": (
        "Edges between parts need weights. Here several fine edges join two parts, "
        "P and Q. Add up their weights to get the total weight between the parts. "
        "Then look at it from each side: f P Q is the share of P's connections "
        "to other parts that go to Q, and f Q P the share of Q's that go to P."
    ),
    "t2_union": (
        "The two directions combine by the same fuzzy union UMAP uses to "
        "symmetrise its own graph. Because each part is measured against its own "
        "connections, a large part doesn't swamp the rest, and a weight only "
        "approaches one when most of a part's connections go to a single other "
        "part. So the coarse graph keeps track of how strongly parts are connected."
    ),
    "t3_levels": (
        "Deeper levels grow their parts the same way, voting with these shares: a "
        "part joins the label holding the largest share of its connections, as "
        "long as that is at least ten percent. That keeps coarse parts much purer. "
        "At the very bottom, parts start at the average PCA position of their members."
    ),
}

# The boundary that ends the coarsening stage (inherited from RecursiveInit)
# carries the notes for the first coarse-edge slide, so point it at the new
# text. This only affects this process, where only this class is rendered.
base.NOTES = dict(base.NOTES, s08_union=NOTES_TRANSITION["t1_shares"])


class RecursiveInitTransition(RecursiveInit):
    ASSET_STEM = "recursive_transition"

    def construct(self):
        self.data = np.load(ASSETS / f"{self.ASSET_STEM}.npz")
        self.meta = json.loads((ASSETS / f"{self.ASSET_STEM}.json").read_text())
        self.setup_toy()

        self.end_section_wipe(
            SECTION_TITLES["recursive"],
            next_slide_prep=lambda: None,
            notes=base.NOTES["s01_wave"],
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

    # -- the one stage that changes: coarse edges from shares ---------------------
    def stage_coarse_edges(self):
        d = self.data
        graph = self.graph
        self.play(FadeOut(self.captions))
        self.set_title("Coarse edges: shares of connection")

        # Total weight between parts, and each part's share of its connections.
        k = self.n_parts
        W = np.zeros((k, k))
        for (i, j), w in zip(self.edges, self.weights):
            p, q = self.labels[i], self.labels[j]
            if p != q:
                W[p, q] += w
                W[q, p] += w
        share = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-12)

        # The pair of parts joined by the most fine edges.
        between = {}
        for e, (i, j) in zip(graph.edges, graph.edge_list):
            key = tuple(sorted((int(self.labels[i]), int(self.labels[j]))))
            between.setdefault(key, []).append(e)
        # A representative pair: the most-connected one whose shares are both
        # informative (not near 0 or 1) and whose parts are visibly apart.
        def distance(key):
            return np.linalg.norm(self.centroids[key[0]] - self.centroids[key[1]])

        candidates = [key for key in between if key[0] != key[1]
                      and 0.15 < share[key[0], key[1]] < 0.85 and 0.15 < share[key[1], key[0]] < 0.85
                      and distance(key) > 0.8]
        pool = candidates or [key for key in between if key[0] != key[1]]
        pair = max(pool, key=lambda key: len(between[key]))
        P, Q = pair
        highlighted = VGroup(*between[pair])
        f_pq, f_qp = share[P, Q], share[Q, P]
        w_pq = f_pq + f_qp - f_pq * f_qp

        coarse_edges = VGroup()
        self.coarse_edge_list = []
        for (p, q), w in zip(d["toy_coarse_edges"], d["toy_coarse_weights"]):
            p, q = int(p), int(q)
            line = Line(self.supernodes[p].get_center(), self.supernodes[q].get_center(),
                        color=STRUCTURE_COLOR, stroke_width=1.2 + 5.0 * w,
                        stroke_opacity=0.35 + 0.55 * min(1.0, 1.5 * w))
            line.add_updater(lambda l, p=p, q=q: l.put_start_and_end_on(
                self.supernodes[p].get_center(), self.supernodes[q].get_center()))
            coarse_edges.add(line)
            self.coarse_edge_list.append((p, q))
        target = (coarse_edges[self.coarse_edge_list.index(pair)]
                  if pair in self.coarse_edge_list else None)

        names = VGroup()
        for part, other, symbol in ((P, Q, "P"), (Q, P, "Q")):
            here, there = self.supernodes[part].get_center(), self.supernodes[other].get_center()
            away = (here - there) / max(np.linalg.norm(here - there), 1e-9)
            names.add(MathTex(symbol, font_size=34, color=HIGHLIGHT_COLOR).move_to(here + 0.38 * away))

        # Slide 1: totals and shares.
        col = Column()
        f_total = col.place(MathTex(r"W_{PQ} \;=\; \sum_{i\in P,\;j\in Q} w_{ij}", font_size=34))
        c_total = col.place(caption_stack(["The total weight between", "the two parts."]), buff=0.2)
        f_share = col.place(MathTex(r"f_{PQ} \;=\; \frac{W_{PQ}}{\sum_{Q'\neq P} W_{PQ'}}", font_size=34), buff=0.35)
        c_share = col.place(caption_stack(["P's membership to Q: the share", "of P's connections that go to Q."]), buff=0.2)
        values = col.place(MathTex(rf"f_{{PQ}} = {f_pq:.2f},\qquad f_{{QP}} = {f_qp:.2f}",
                                   font_size=30, color=HIGHLIGHT_COLOR), buff=0.3)
        count = col.place(crisp_text(f"{len(highlighted)} fine edges between P and Q", 20,
                                     color=HIGHLIGHT_COLOR), buff=0.15)

        for e in highlighted:
            e.clear_updaters()
        self.play(highlighted.animate.set_stroke(color=HIGHLIGHT_COLOR, width=4, opacity=1),
                  FadeIn(names))
        self.play(Write(f_total), FadeIn(c_total))
        self.play(Write(f_share), FadeIn(c_share))
        self.play(FadeIn(values), FadeIn(count))
        self.marked_next_slide(notes=NOTES_TRANSITION["t2_union"])

        # Slide 2: combine the directions, merge into one coarse edge.
        self.play(FadeOut(VGroup(f_total, c_total, f_share, c_share, values, count)))
        col = Column()
        f_union = col.place(MathTex(r"w_{PQ} \;=\; f_{PQ} + f_{QP} - f_{PQ}\,f_{QP}", font_size=34))
        c_union = col.place(caption_stack(["Both directions combine by the", "fuzzy union UMAP uses for", "its own graph."]), buff=0.2)
        merged = col.place(MathTex(rf"w_{{PQ}} = {w_pq:.2f}", font_size=30, color=HIGHLIGHT_COLOR), buff=0.3)
        self.play(Write(f_union), FadeIn(c_union))
        others = VGroup(*[e for e in graph.edges if e not in highlighted])
        anims = [FadeOut(others), FadeIn(VGroup(*[c for c in coarse_edges if c is not target]))]
        # Morph into a static copy: transforming a group of lines into the live
        # edge would strip the edge of its own points and break its updater.
        proxy = target.copy().clear_updaters() if target is not None else None
        anims.append(ReplacementTransform(highlighted, proxy) if proxy is not None else FadeOut(highlighted))
        self.play(*anims, FadeIn(merged), run_time=1.4)
        if proxy is not None:
            self.remove(proxy)
        # FadeOut restores faded mobjects after removal; drop the fine edges for good.
        self.remove(graph.edges)
        self.add(coarse_edges)
        self.bring_to_front(self.supernodes, names)
        self.coarse_edges = coarse_edges
        why = col.place(caption_stack(
            ["Each part is measured against its", "own connections, so large parts",
             "don't swamp the rest, and a weight", "only nears 1 when most of a part's",
             "connections go to one other part."], color=STRUCTURE_COLOR,
        ), buff=0.4)
        self.play(FadeIn(why))
        self.marked_next_slide(notes=NOTES_TRANSITION["t3_levels"])

        # Slide 3: the same shares drive the deeper levels; the base is PCA.
        self.play(FadeOut(VGroup(f_union, c_union, merged, why)))
        col = Column()
        deeper = col.place(caption_stack(
            ["Deeper levels grow parts the", "same way: a part joins the label",
             "holding the largest share (at", "least 10%) of its connections."]))
        bottom = col.place(caption_stack(
            ["At the coarsest level, parts", "start at their members'", "average PCA position."],
            color=STRUCTURE_COLOR), buff=0.45)
        self.play(FadeIn(deeper))
        self.play(FadeIn(bottom))
        self.stage4_captions = VGroup(deeper, bottom, names)
        self.marked_next_slide(notes=base.NOTES["s10_layout"])
