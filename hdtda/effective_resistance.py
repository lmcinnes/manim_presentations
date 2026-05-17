from encodings.idna import dots

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
from data_generation import CircleEmbedding, CurvyLoopEmbedding, TorusEmbedding

import numpy as np
import colorcet
from ripser import ripser
import sklearn.decomposition
import sklearn.neighbors
from scipy.stats import gaussian_kde

apply_defaults()


class EffectiveResistanceCommute(TIMCSlide):
    """
    Demonstrates effective resistance through the commute-time intuition.

    Key message: R_eff is governed by the *number* of paths between two nodes.
    Two nodes connected by many medium-length paths can have lower resistance
    than two nodes connected by a single short path if all other routes between
    the latter are extremely long.

    Slide flow
    ----------
    Phase 1 – Setup
      1. Title
      2. Reveal 3×4 grid; highlight s (row 1, left) and t (row 1, right)
      3. Commute-time definition + counter widget appears

    Phase 2 – Walks on full grid (many paths)
      4. Slow walk 1: step-by-step dot + trail (0.45s/hop)
      5. Faster walk 2: (0.2s/hop), rolling avg counter
      6. Fast flash walks × 10: polyline traces, counter settles low
         → caption "Many paths → low resistance"

    Phase 3 – Prune to bottleneck (single bridge)
      7. FadeOut most cross-edges; highlight one bridge; |E| updates
      8. Slow walk 1 on bottleneck: long detour, counter jumps high
      9. Fast flash walks × 8 on bottleneck: counter settles high
         → side-by-side annotation comparing both averages

    Phase 4 – Formulas
     10. Clear, show L definition, R_eff = (e_i−e_j)^T L+ (e_i−e_j),
         C(i,j) = 2|E|·R_eff(i,j)
    """

    # ------------------------------------------------------------------ #
    #  Grid helpers                                                        #
    # ------------------------------------------------------------------ #

    ROWS = 3
    COLS = 4
    DX = 2.1  # horizontal spacing
    DY = 2.0  # vertical spacing
    GRID_LEFT = -4.5

    def _node_pos(self, r, c):
        """Manim position of grid node (row r, col c)."""
        x = self.GRID_LEFT + c * self.DX
        y = (self.ROWS - 1) / 2 * self.DY - r * self.DY
        return np.array([x, y, 0.0])

    def _node_index(self, r, c):
        return r * self.COLS + c

    def _build_grid(self):
        """
        Return (positions, edge_list, dot_objects, edge_objects).
        edge_list : list of (i, j) node-index pairs (i < j)
        """
        N = self.ROWS * self.COLS
        positions = [
            self._node_pos(r, c) for r in range(self.ROWS) for c in range(self.COLS)
        ]

        edge_list = []
        for r in range(self.ROWS):
            for c in range(self.COLS):
                idx = self._node_index(r, c)
                if c + 1 < self.COLS:  # horizontal edge
                    edge_list.append((idx, self._node_index(r, c + 1)))
                if r + 1 < self.ROWS:  # vertical edge
                    edge_list.append((idx, self._node_index(r + 1, c)))

        dots = [
            Dot(positions[k], radius=0.13, color=DEFAULT_COLOR).set_z_index(2)
            for k in range(N)
        ]

        edges = {
            (i, j): Line(
                positions[i], positions[j], color=DEFAULT_COLOR, stroke_width=3
            )
            for (i, j) in edge_list
        }

        return positions, edge_list, dots, edges

    # ------------------------------------------------------------------ #
    #  Counter widget                                                      #
    # ------------------------------------------------------------------ #

    def _make_counter(self, n_edges):
        """
        Return (VGroup widget, DecimalNumber value_mob, Integer n_edges_mob).
        Counter displays:  avg_walk_length / |E| = ???
        """
        path_label = Text("Walk length", font_size=24)
        path_val = DecimalNumber(
            0,
            num_decimal_places=0,
            font_size=48,
            color=ACCENT_COLOR,
        )
        path_val.next_to(path_label, DOWN, buff=0.18)
        label = Tex(r"Average $\frac{\text{walk length}}{|E|}$", font_size=36)
        label.next_to(path_val, DOWN, buff=0.4)
        val = DecimalNumber(
            0.0,
            num_decimal_places=2,
            font_size=48,
            color=ACCENT_COLOR,
        )
        val.next_to(label, DOWN, buff=0.18)

        n_lbl = Text(f"|E| = {n_edges}", font_size=18).next_to(label, DOWN, buff=0.18)
        n_lbl.align_to(label, LEFT)

        box = VGroup(path_label, path_val, label, val)
        box.to_edge(RIGHT, buff=0.35).move_to([box.get_center()[0], 0, 0])
        return box, path_val, val

    # ------------------------------------------------------------------ #
    #  Random-walk helpers                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _adjacency(edge_list, n_nodes):
        adj = [[] for _ in range(n_nodes)]
        for i, j in edge_list:
            adj[i].append(j)
            adj[j].append(i)
        return adj

    @staticmethod
    def _random_walk_to(start, target, adj, rng, max_steps=2000):
        """Walk until we reach *target*; return list of node indices visited."""
        path = [start]
        cur = start
        for _ in range(max_steps):
            if cur == target:
                break
            nxt = rng.choice(adj[cur])
            path.append(nxt)
            cur = nxt
        return path

    def _round_trip(self, start, target, adj, rng):
        """s → t walk concatenated with t → s walk (share one endpoint)."""
        leg1 = self._random_walk_to(start, target, adj, rng)
        leg2 = self._random_walk_to(target, start, adj, rng)
        return leg1 + leg2[1:]  # avoid duplicating the target node

    # ------------------------------------------------------------------ #
    #  Animation primitives                                                #
    # ------------------------------------------------------------------ #

    def _animate_walk_slow(self, path, positions, dot, step_time, max_anim_steps=32):
        """
        Animate a dot moving step-by-step; each edge leaves a fading trail.
        If the path is longer than *max_anim_steps*, only the first
        max_anim_steps hops are shown step-by-step.  An ellipsis label
        appears briefly to signal that the walk continues off-screen.
        """
        display_path = path[: max_anim_steps + 1]
        truncated = len(path) - 1 > max_anim_steps

        trail_segs = []
        for k in range(1, len(display_path)):
            p0 = positions[display_path[k - 1]]
            p1 = positions[display_path[k]]
            seg = Line(p0, p1, color=dot.get_color(), stroke_width=8)
            self.play(
                dot.animate.move_to(p1),
                Create(seg),
                run_time=step_time,
                rate_func=linear,
            )
            # self.add(seg)
            trail_segs.append(seg)

        if truncated:
            ellipsis = Text("...", font_size=36, color=dot.get_color())
            ellipsis.next_to(dot, UP, buff=0.15)
            self.play(FadeIn(ellipsis, run_time=0.25))
            self.wait(0.5)
            # Snap the dot to the true endpoint without animation
            true_end = positions[path[-1]]
            self.play(FadeOut(ellipsis, run_time=0.2))
            self.play(dot.animate.move_to(true_end), run_time=0.3)

        # Fade all trail segments together
        if trail_segs:
            self.play(*[FadeOut(s) for s in trail_segs], run_time=1.2)

    def _animate_walk_flash(self, path, positions, color=None):
        """Flash the full path as a polyline then fade it out quickly."""
        if color is None:
            color = COLOR_CYCLE[4]
        pts = [positions[n] for n in path]
        poly = VMobject(color=color, stroke_width=8, stroke_opacity=0.8)
        poly.set_points_as_corners(pts)
        self.play(Create(poly, run_time=0.35, rate_func=linear))
        self.play(FadeOut(poly, run_time=0.5))

    def _animate_walk_medium(
        self, path, positions, dot, run_time, fade_run_time=0.75, color=None
    ):
        """
        Move *dot* along *path* in a single self.play call, with the dot
        leading and a trail growing behind it.

        UpdateFromAlphaFunc rebuilds the trail at each alpha tick using the
        exact node positions, lagging exactly one segment behind the dot.
        This is frame-rate independent — the trail always follows the grid
        edges exactly regardless of how many hops per second are animated.

        This produces exactly 2 self.play calls regardless of path length,
        keeping the total animation count low enough for manim-slides to
        concatenate without errors.
        """
        if color is None:
            color = dot.get_color()
        pts = [positions[n] for n in path]
        n_segs = len(pts) - 1

        # Geometric path for MoveAlongPath.
        walk_path = VMobject()
        walk_path.set_points_as_corners(pts)
        dot.move_to(pts[0])

        trail = VMobject(stroke_color=color, stroke_width=8, stroke_opacity=0.85)

        def trail_updater(mob, alpha):
            # Lag trail by exactly one segment so dot always leads.
            lag_alpha = max(0.0, alpha - 1.0 / n_segs)
            if lag_alpha < 1e-9:
                mob.set_points_as_corners([pts[0], pts[0]])
                return
            seg_float = lag_alpha * n_segs
            full_segs = int(seg_float)
            partial = seg_float - full_segs
            if full_segs >= n_segs:
                mob.set_points_as_corners(pts)
            else:
                mid = pts[full_segs] + partial * (pts[full_segs + 1] - pts[full_segs])
                mob.set_points_as_corners(list(pts[: full_segs + 1]) + [mid])

        self.add(trail)
        self.play(
            UpdateFromAlphaFunc(trail, trail_updater, rate_func=linear),
            MoveAlongPath(dot, walk_path, rate_func=linear),
            run_time=run_time,
        )
        self.play(FadeOut(trail), run_time=fade_run_time)

    # ------------------------------------------------------------------ #
    #  construct                                                           #
    # ------------------------------------------------------------------ #

    def construct(self):
        rng = np.random.default_rng(42)
        N = self.ROWS * self.COLS

        # s = row 1, col 0 (left-middle);  t = row 1, col 3 (right-middle)
        IDX_S = self._node_index(1, 0)
        IDX_T = self._node_index(1, 3)

        positions, full_edge_list, dots, edge_mobs = self._build_grid()

        NI = self._node_index  # shorthand

        # ── Canned walks for pedagogically clear slow illustrations. ──────────
        # Walk 1: go via the top row to t, return via the bottom row to s.
        canned_walk1 = [
            IDX_S,
            NI(0, 0),
            NI(0, 1),
            NI(0, 2),
            NI(0, 3),
            IDX_T,
            NI(2, 3),
            NI(2, 2),
            NI(2, 1),
            NI(2, 0),
            IDX_S,
        ]
        # Walk 2: brief detour up through col-1 before crossing, direct return.
        canned_walk2 = [
            IDX_S,
            NI(1, 1),
            NI(0, 1),
            NI(0, 2),
            NI(1, 2),
            IDX_T,
            NI(1, 2),
            NI(1, 1),
            IDX_S,
        ]
        # Bottleneck walk: bounces near the bridge before eventually crossing.
        canned_bot_walk = [
            IDX_S,
            NI(1, 1),
            NI(0, 1),
            NI(1, 1),
            NI(2, 1),
            NI(1, 1),
            NI(1, 2),
            NI(0, 2),
            NI(1, 2),
            IDX_T,
            NI(0, 3),
            NI(0, 2),
            NI(1, 2),
            NI(1, 1),
            IDX_S,
        ]

        # ============================================================
        # Slide 1 – Title
        # ============================================================
        title = Text("Effective Resistance & Commute Time", font_size=44)
        subtitle = Text("How many paths are there between two nodes?", font_size=28)
        subtitle.next_to(title, DOWN, buff=0.45)
        self.play(Write(title))
        self.play(FadeIn(subtitle, shift=UP * 0.15))
        self.marked_next_slide()
        self.play(FadeOut(title), FadeOut(subtitle))

        # ============================================================
        # Slide 2 – Reveal 4×4 grid
        # ============================================================
        all_edge_mobs = VGroup(*edge_mobs.values())
        all_dots = VGroup(*dots)

        self.play(LaggedStart(*[FadeIn(d, scale=1.4) for d in dots], lag_ratio=0.04))
        self.play(Create(all_edge_mobs, run_time=1.2, lag_ratio=0.04))

        # Colour s and t
        dot_s = dots[IDX_S]
        dot_t = dots[IDX_T]
        lbl_s = Text("source", font_size=28, color=COLOR_CYCLE[0]).next_to(
            dot_s, LEFT, buff=0.2
        )
        lbl_t = Text("target", font_size=28, color=COLOR_CYCLE[1]).next_to(
            dot_t, RIGHT, buff=0.2
        )
        self.play(
            dot_s.animate.set_color(COLOR_CYCLE[0]).scale(1.6).set_z_index(4),
            dot_t.animate.set_color(COLOR_CYCLE[1]).scale(1.6).set_z_index(4),
            FadeIn(lbl_s),
            FadeIn(lbl_t),
        )
        self.marked_next_slide()

        # ============================================================
        # Slide 3 – Commute-time definition + counter
        # ============================================================
        defn = Text(
            "Commute time  C(source, target)  =  expected steps for  source → target → source",
            font_size=24,
        ).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(defn, shift=UP * 0.1))

        n_edges_full = len(full_edge_list)
        counter_grp, path_val, counter_val = self._make_counter(n_edges_full)
        self.play(FadeIn(counter_grp))
        self.marked_next_slide()

        # ============================================================
        # Phase 2 – Walks on full grid
        # ============================================================
        adj_full = self._adjacency(full_edge_list, N)

        # -- Walk 1 (slow, canned: scenic route via top and bottom rows) ------
        walk_dot = Dot(positions[IDX_S], radius=0.08, color=COLOR_CYCLE[3])
        self.add(walk_dot)
        self._animate_walk_slow(canned_walk1, positions, walk_dot, step_time=0.42)

        commute_steps = [len(canned_walk1) - 1]  # steps = hops = len(path)-1
        rolling_avg = np.mean(commute_steps)
        self.play(ChangeDecimalToValue(path_val, commute_steps[-1]))
        self.play(ChangeDecimalToValue(counter_val, rolling_avg / n_edges_full))
        self.marked_next_slide()

        # -- Walk 2 (faster, canned: detour then direct return) ----------------
        self._animate_walk_slow(
            canned_walk2, positions, walk_dot, step_time=0.22, max_anim_steps=20
        )

        commute_steps.append(len(canned_walk2) - 1)
        rolling_avg = np.mean(commute_steps)
        self.play(ChangeDecimalToValue(path_val, commute_steps[-1]))
        self.play(ChangeDecimalToValue(counter_val, rolling_avg / n_edges_full))
        self.marked_next_slide()

        for k in range(16):
            walk = self._round_trip(IDX_S, IDX_T, adj_full, rng)
            self._animate_walk_medium(
                walk,
                positions,
                walk_dot,
                run_time=4.0 / (k + 1),
                fade_run_time=1.0 / np.sqrt(k + 1),
            )

            commute_steps.append(len(walk) - 1)
            rolling_avg = np.mean(commute_steps)
            self.play(
                ChangeDecimalToValue(path_val, commute_steps[-1]),
                ChangeDecimalToValue(counter_val, rolling_avg / n_edges_full),
                run_time=1.0 / np.sqrt(k + 1),
            )
        self.play(FadeOut(walk_dot))

        self.marked_next_slide()

        # -- Fast flash walks (10 more) ------------------------------
        # self.play(FadeOut(walk_dot))
        # for k in range(10):
        #     wk = self._round_trip(IDX_S, IDX_T, adj_full, rng)
        #     commute_steps.append(len(wk) - 1)
        #     self._animate_walk_flash(wk, positions, color=COLOR_CYCLE[3])
        #     rolling_avg = np.mean(commute_steps)
        #     self.play(
        #         counter_val.animate.set_value(rolling_avg / n_edges_full),
        #         run_time=(0.66 / np.sqrt(k + 1)),
        #     )

        caption_full = Text(
            "Many paths  →  short average commute  →  low  R_eff",
            font_size=24,
            color=COLOR_CYCLE[2],
        ).to_edge(DOWN, buff=0.5)
        self.play(FadeOut(defn), FadeIn(caption_full))
        full_avg_val = rolling_avg / n_edges_full
        self.marked_next_slide()

        # ============================================================
        # Phase 3 – Prune to bottleneck
        # ============================================================
        # Remove all horizontal edges crossing cols 1→2 except the one
        # at row 1 (the bridge row). Also remove most col 2→3 cross-edges
        # to really force the walker through the bottleneck.
        # Bridge = horizontal edge (row=1, col1→col2).
        bridge_idx_pair = (
            self._node_index(1, 1),
            self._node_index(1, 2),
        )

        # Edges to prune: all horizontal edges in cols 1→2 except the bridge.
        edges_to_prune = []
        for r in range(self.ROWS):
            for c in range(self.COLS - 1):
                i = self._node_index(r, c)
                j = self._node_index(r, c + 1)
                key = (min(i, j), max(i, j))
                # Keep only the bridge and the outermost column edges
                if c == 1 and r != 1:
                    edges_to_prune.append(key)

        # Also prune the direct horizontal edges cols 0→1 (rows 0 and 3)
        # so that the bridge is the dominant bottleneck.
        extra_prune = [
            (self._node_index(0, 0), self._node_index(0, 1)),
            (self._node_index(2, 0), self._node_index(2, 1)),
        ]
        for ep in extra_prune:
            edges_to_prune.append(ep)

        # Deduplicate
        edges_to_prune = list({(min(a, b), max(a, b)) for a, b in edges_to_prune})

        pruned_mobs = [edge_mobs[k] for k in edges_to_prune if k in edge_mobs]
        pruned_set = set(edges_to_prune)

        self.play(FadeOut(caption_full))
        prune_caption = Text("Prune to a single bridge", font_size=28).to_edge(
            DOWN, buff=0.5
        )
        self.play(Write(prune_caption))
        self.play(
            LaggedStart(*[FadeOut(m) for m in pruned_mobs], lag_ratio=0.12),
            run_time=1.2,
        )

        # # Highlight the bridge
        # bridge_key = (min(bridge_idx_pair), max(bridge_idx_pair))
        # bridge_mob = edge_mobs[bridge_key]
        # self.play(bridge_mob.animate.set_color(HIGHLIGHT_COLOR).set_stroke(width=6))

        # Rebuild adjacency on pruned graph
        pruned_edge_list = [
            (i, j)
            for (i, j) in full_edge_list
            if (min(i, j), max(i, j)) not in pruned_set
        ]
        n_edges_pruned = len(pruned_edge_list)
        adj_pruned = self._adjacency(pruned_edge_list, N)

        # Update |E| label
        # new_ne_lbl = Text(f"|E| = {n_edges_pruned}", font_size=18).move_to(
        #     counter_ne_lbl
        # )
        # new_ne_lbl.align_to(counter_ne_lbl, LEFT)
        # self.play(
        #     FadeOut(counter_ne_lbl),
        #     FadeIn(new_ne_lbl),
        #     run_time=0.5,
        # )
        # counter_ne_lbl = new_ne_lbl

        self.marked_next_slide()

        # -- Walk on bottleneck (canned: shows bouncing at bridge) ------------
        self.play(FadeOut(prune_caption))
        self.play(ChangeDecimalToValue(path_val, 0))
        self.play(ChangeDecimalToValue(counter_val, 0.0))
        commute_steps_bot = []

        walk_dot2 = Dot(positions[IDX_S], radius=0.08, color=COLOR_CYCLE[3])
        self.add(walk_dot2)
        self._animate_walk_slow(
            canned_bot_walk, positions, walk_dot2, step_time=0.32, max_anim_steps=22
        )

        commute_steps_bot.append(len(canned_bot_walk) - 1)
        rolling_avg_bot = np.mean(commute_steps_bot)
        self.play(ChangeDecimalToValue(path_val, commute_steps_bot[-1]))
        self.play(ChangeDecimalToValue(counter_val, rolling_avg_bot / n_edges_pruned))
        self.marked_next_slide()

        for k in range(16):
            walk = self._round_trip(IDX_S, IDX_T, adj_pruned, rng)
            self._animate_walk_medium(
                walk,
                positions,
                walk_dot2,
                run_time=4.0 / (k + 1),
                fade_run_time=1.0 / np.sqrt(k + 1),
            )

            commute_steps_bot.append(len(walk) - 1)
            rolling_avg_bot = np.mean(commute_steps_bot)
            self.play(ChangeDecimalToValue(path_val, commute_steps_bot[-1]))
            self.play(
                ChangeDecimalToValue(counter_val, rolling_avg_bot / n_edges_pruned),
                run_time=1.0 / np.sqrt(k + 1),
            )

        self.marked_next_slide()

        self.play(FadeOut(walk_dot2))

        # # -- Fast flash walks on bottleneck (8 more) -----------------
        # for k in range(8):
        #     wk = self._round_trip(IDX_S, IDX_T, adj_pruned, rng)
        #     commute_steps_bot.append(len(wk) - 1)
        #     self._animate_walk_flash(wk, positions, color=COLOR_CYCLE[4])
        #     rolling_avg_bot = np.mean(commute_steps_bot)
        #     self.play(
        #         counter_val.animate.set_value(rolling_avg_bot / n_edges_pruned),
        #         run_time=0.25,
        #     )

        bottleneck_avg_val = rolling_avg_bot / n_edges_pruned

        # Side-by-side comparison annotations
        ann_full = MathTex(
            rf"\text{{full graph: }}\approx {full_avg_val:.2f}",
            color=COLOR_CYCLE[2],
            font_size=26,
        ).to_edge(DOWN, buff=1.1)
        ann_bot = MathTex(
            rf"\text{{bottleneck: }}\approx {bottleneck_avg_val:.2f}",
            color=HIGHLIGHT_COLOR,
            font_size=26,
        ).to_edge(DOWN, buff=0.55)
        caption_bot = Text(
            "Fewer paths  →  longer average commute  →  higher  R_eff",
            font_size=24,
            color=HIGHLIGHT_COLOR,
        ).to_edge(DOWN, buff=0.5)

        self.play(FadeIn(ann_full), FadeIn(ann_bot))
        self.marked_next_slide()

        self.play(FadeOut(ann_full), FadeOut(ann_bot), FadeIn(caption_bot))
        self.marked_next_slide()

        # ============================================================
        # Phase 4 – Formulas
        # ============================================================
        everything = VGroup(
            all_edge_mobs,
            all_dots,
            lbl_s,
            lbl_t,
            counter_grp,
            caption_bot,
        )
        self.play(FadeOut(everything), run_time=0.8)

        form_title = Text("Effective Resistance", font_size=40)
        form_title.shift(UP * 2.6)

        lap_def = MathTex(
            r"L_{ij} = \begin{cases}"
            r"\deg(i) & i = j \\"
            r"-1 & \{i,j\} \in E \\"
            r"0 & \text{otherwise}"
            r"\end{cases}",
            font_size=34,
        ).next_to(form_title, DOWN, buff=0.5)

        reff_form = MathTex(
            r"R_{\mathrm{eff}}(i,j) \;=\; "
            r"(\mathbf{e}_i - \mathbf{e}_j)^\top \, L^+ \, (\mathbf{e}_i - \mathbf{e}_j)",
            font_size=34,
        ).next_to(lap_def, DOWN, buff=0.5)

        commute_form = MathTex(
            r"\frac{C(i,j)}{2\,|E|} \;=\; R_{\mathrm{eff}}(i,j)",
            font_size=34,
        ).next_to(reff_form, DOWN, buff=0.42)
        commute_box = SurroundingRectangle(
            commute_form, color=HIGHLIGHT_COLOR, buff=0.16, stroke_width=3
        )

        self.play(Write(form_title))
        self.play(Write(lap_def))
        self.marked_next_slide()

        self.play(Write(reff_form))
        self.play(Write(commute_form), Create(commute_box))
        self.marked_next_slide()


import networkx as nx


class ElectricResistanceGrid(TIMCSlide):
    """
    Visualises effective resistance on a 3×4 grid as an electric circuit.

    Slide flow
    ----------
    1. Reveal the grid – uniform grey nodes, uniform-thickness grey edges.
    2. Scale up source and target; show "source"/"target" labels;
       show a horizontal viridis colorbar along the bottom whose range
       covers both the full and pruned graphs.
    3. Wave animation: source node lights up first, then the rest of the
       nodes colour-in by potential (high→low order) and edges thicken by
       current (|Δv|).
    4. Shift graph left; reveal right-aligned, vertically-centred info
       box showing "Effective Resistance", source/target colour swatches,
       and the numeric R_eff value.
    5. Reset graph to grey/uniform (ground state), then fade out the
       pruned edges, then redo the wave animation on the pruned graph,
       and update R_eff in the info box.
    """

    # ── Grid geometry ────────────────────────────────────────────────── #
    ROWS, COLS = 3, 4
    DX, DY = 2.2, 1.7

    # ── Visual constants ─────────────────────────────────────────────── #
    NODE_R = 0.15  # default node radius
    NODE_R_HL = 0.22  # highlighted (s, t) radius
    EDGE_W_UNIFORM = 4.0  # uniform edge stroke before potential applied
    EDGE_W_MIN = 1.5  # thinnest edge after potential applied
    EDGE_W_MAX = 13.0  # thickest edge after potential applied
    GRAPH_SHIFT = 2.0  # leftward shift for info-box slide

    # ── Helpers ──────────────────────────────────────────────────────── #

    def _node_pos(self, r, c):
        x = -(self.COLS - 1) * self.DX / 2 + c * self.DX
        y = (self.ROWS - 1) * self.DY / 2 - r * self.DY
        return np.array([x, y, 0.0])

    def _potentials(self, G, nodes, node_idx, source, target):
        """Return potential array via pseudoinverse of Laplacian."""
        L = nx.laplacian_matrix(G, nodelist=nodes).toarray().astype(float)
        b = np.zeros(len(nodes))
        b[node_idx[source]] = 1.0
        b[node_idx[target]] = -1.0
        return np.linalg.pinv(L) @ b

    def _make_colorbar(self, cmap_fn, v_min, v_max):
        """Return a horizontal VGroup colorbar along the bottom of the frame."""
        N, total_w, height, y = 128, 10.0, 0.30, -3.25
        rw = total_w / N
        rects = VGroup(
            *[
                Rectangle(
                    width=rw + 0.005,
                    height=height,
                    fill_color=cmap_fn(i / (N - 1)),
                    fill_opacity=1.0,
                    stroke_width=0,
                ).move_to([-total_w / 2 + (i + 0.5) * rw, y, 0])
                for i in range(N)
            ]
        )
        border = Rectangle(
            width=total_w + 0.02,
            height=height + 0.01,
            stroke_width=1.5,
            stroke_color=DARK_GRAY,
            fill_opacity=0,
        ).move_to([0, y, 0])
        lbl_lo = Text(f"{v_min:.2f}", font_size=17).next_to(rects[0], LEFT, buff=0.12)
        lbl_hi = Text(f"{v_max:.2f}", font_size=17).next_to(rects[-1], RIGHT, buff=0.12)
        title = Text("Potential", font_size=19).next_to(border, DOWN, buff=0.10)
        return VGroup(rects, border, lbl_lo, lbl_hi, title)

    # ── construct ────────────────────────────────────────────────────── #

    def construct(self):
        import networkx as nx
        import matplotlib.cm as _cm

        # ── Graphs ────────────────────────────────────────────────────────
        G = nx.grid_2d_graph(self.ROWS, self.COLS)
        source, target = (1, 0), (1, 3)
        nodes = list(G.nodes())
        node_idx = {nd: i for i, nd in enumerate(nodes)}

        # Pruned graph: remove only the two middle-third edges (top and bottom
        # of columns 1→2), leaving the left-third edges intact.
        G2 = G.copy()
        for e in [((0, 1), (0, 2)), ((2, 1), (2, 2))]:
            G2.remove_edge(*e)

        # ── Potentials ────────────────────────────────────────────────────
        pot1 = self._potentials(G, nodes, node_idx, source, target)
        pot2 = self._potentials(G2, nodes, node_idx, source, target)

        # Colorbar range covers both graphs so colours are comparable.
        v_min = min(pot1.min(), pot2.min())
        v_max = max(pot1.max(), pot2.max())

        R1 = float(pot1[node_idx[source]] - pot1[node_idx[target]])
        R2 = float(pot2[node_idx[source]] - pot2[node_idx[target]])

        # ── Colormap (matplotlib viridis) ─────────────────────────────────
        _cm_obj = _cm.viridis

        def cmap_fn(t):
            r, g, b, _ = _cm_obj(float(np.clip(t, 0, 1)))
            return "#{:02x}{:02x}{:02x}".format(
                int(r * 255), int(g * 255), int(b * 255)
            )

        def pcol(val):
            return cmap_fn((val - v_min) / (v_max - v_min + 1e-9))

        # ── Edge stroke-width helper ──────────────────────────────────────
        def sw(drop, mx):
            frac = drop / (mx + 1e-9)
            return self.EDGE_W_MIN + (self.EDGE_W_MAX - self.EDGE_W_MIN) * frac

        mx1 = max(abs(pot1[node_idx[u]] - pot1[node_idx[v]]) for u, v in G.edges())
        mx2 = max(abs(pot2[node_idx[u]] - pot2[node_idx[v]]) for u, v in G2.edges())

        # ── Build initial mobjects (uniform grey style) ────────────────────
        npos = {nd: self._node_pos(*nd) for nd in nodes}
        dots = {nd: Dot(npos[nd], radius=self.NODE_R, color=GRAY) for nd in nodes}
        emobs = {
            (u, v): Line(npos[u], npos[v], stroke_width=self.EDGE_W_UNIFORM, color=GRAY)
            for u, v in G.edges()
        }

        def em(u, v):
            return emobs.get((u, v)) or emobs.get((v, u))

        all_e = VGroup(*emobs.values())
        all_n = VGroup(*dots.values())

        # ── SLIDE 1: reveal plain grey grid ──────────────────────────────
        self.play(Create(all_e), Create(all_n))
        self.marked_next_slide()

        # ── SLIDE 2: highlight s/t; "source"/"target" labels; colorbar ───
        hl = self.NODE_R_HL / self.NODE_R
        self.play(dots[source].animate.scale(hl), dots[target].animate.scale(hl))

        s_lbl = Text("source", font_size=24).next_to(dots[source], LEFT, buff=0.25)
        t_lbl = Text("target", font_size=24).next_to(dots[target], RIGHT, buff=0.25)
        self.play(Write(s_lbl), Write(t_lbl))

        cb = self._make_colorbar(cmap_fn, v_min, v_max)
        self.play(FadeIn(cb))
        self.marked_next_slide()

        # ── SLIDE 3: wave animation — potentials + edge thicknesses ───────
        def _wave_anim(pot, mx, graph_obj):
            """Animate a potential wave and return the animations."""
            src_anim = [dots[source].animate.set_color(pcol(pot[node_idx[source]]))]
            others = sorted(
                [nd for nd in nodes if nd != source],
                key=lambda nd: -pot[node_idx[nd]],
            )
            n_anims = [
                dots[nd].animate.set_color(pcol(pot[node_idx[nd]])) for nd in others
            ]
            e_anims = [
                em(u, v).animate.set_stroke(
                    width=sw(abs(pot[node_idx[u]] - pot[node_idx[v]]), mx),
                    color=DARK_GRAY,
                )
                for u, v in graph_obj.edges()
            ]
            return src_anim, n_anims, e_anims

        src_a1, n_a1, e_a1 = _wave_anim(pot1, mx1, G)
        self.play(*src_a1)
        self.play(LaggedStart(*n_a1, lag_ratio=0.12), *e_a1, run_time=2.5)
        self.marked_next_slide()

        # ── SLIDE 4: shift graph left; reveal right-aligned info box ──────
        graph_grp = VGroup(all_e, all_n, s_lbl, t_lbl)
        self.play(graph_grp.animate.shift(LEFT * self.GRAPH_SHIFT))

        i_title = Text("Effective Resistance", font_size=26, color=DEFAULT_COLOR)
        i_src = Dot(radius=0.13, color=pcol(pot1[node_idx[source]]))
        i_minus = Text("−", font_size=30, color=DEFAULT_COLOR)
        i_tgt = Dot(radius=0.13, color=pcol(pot1[node_idx[target]]))
        i_swatches = VGroup(i_src, i_minus, i_tgt).arrange(RIGHT, buff=0.18)
        reff_num = DecimalNumber(
            R1, num_decimal_places=3, font_size=40, color=ACCENT_COLOR
        )
        info = VGroup(i_title, i_swatches, reff_num).arrange(DOWN, buff=0.30)
        info.to_edge(RIGHT, buff=0.35)
        info.move_to([info.get_center()[0], 0, 0])  # centre vertically

        self.play(FadeIn(info))
        self.marked_next_slide()

        # ── SLIDE 5: reset → prune → reflow wave → update R_eff ──────────
        # Reset graph to ground state (grey nodes, uniform edges).
        # Also grey out the info-box swatches and zero the R_eff value.
        reset_n = [dots[nd].animate.set_color(GRAY) for nd in nodes]
        reset_e = [
            em(u, v).animate.set_stroke(width=self.EDGE_W_UNIFORM, color=GRAY)
            for u, v in G.edges()
        ]
        self.play(
            *reset_n,
            *reset_e,
            i_src.animate.set_color(GRAY),
            i_tgt.animate.set_color(GRAY),
            ChangeDecimalToValue(reff_num, 0.0),
            run_time=1.2,
        )

        # Fade out the two pruned middle-third edges.
        prune_edges = [((0, 1), (0, 2)), ((2, 1), (2, 2))]
        pm = [m for e in prune_edges if (m := em(*e)) is not None]
        self.play(LaggedStart(*[FadeOut(m) for m in pm], lag_ratio=0.12), run_time=0.9)

        # Redo the wave on the pruned graph.
        src_a2, n_a2, e_a2 = _wave_anim(pot2, mx2, G2)
        self.play(*src_a2)
        self.play(LaggedStart(*n_a2, lag_ratio=0.12), *e_a2, run_time=2.5)

        # Update info box: swatch colours and R_eff value.
        self.play(
            i_src.animate.set_color(pcol(pot2[node_idx[source]])),
            i_tgt.animate.set_color(pcol(pot2[node_idx[target]])),
            ChangeDecimalToValue(reff_num, R2),
        )
        self.marked_next_slide()
