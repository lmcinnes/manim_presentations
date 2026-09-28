"""The kernel as dials: the Burr family, what its parameters control, and the
closing of the ongoing-research section.

    python assets_kernel_dials.py      # once, writes assets/kernel_dials.*
    manim-slides render kernel_dials.py KernelDials

Also reads assets/graph_repair.npz (the test circle).
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

from where_next import Column, caption, caption_stack, darker, fit_points, loop_color, segment_cloud, to3, WRONG_COLOR

ASSETS = Path(__file__).parent / "assets"
P_COLORS = {1.0: COLOR_CYCLE[0], 1.5: COLOR_CYCLE[2], 2.0: STRUCTURE_COLOR, 4.0: COLOR_CYCLE[3]}


def burr(d, sigma, p, nu):
    return (1 + (np.asarray(d) / sigma) ** p) ** (-nu)


def influence(d, sigma, p, nu):
    u = (np.asarray(d) / sigma) ** p
    return nu * p * np.asarray(d) ** (p - 1) / sigma ** p / (1 + u)


def small_axes(x_range, y_range, x_label, y_label, x_length, y_length, centre, x_ticks=None, y_ticks=None):
    chart = styled_axes(x_range, y_range, x_label, y_label, x_length=x_length, y_length=y_length,
                        y_decimal_places=1)
    chart.move_to(to3(centre))
    axes = chart[0]
    for v in (x_ticks or []):
        chart.add(caption(f"{v:g}", 16).next_to(axes.c2p(v, y_range[0]), DOWN, buff=0.18))
    return chart, axes


class KernelDials(UMAPSlide):
    def construct(self):
        self.data = np.load(ASSETS / "kernel_dials.npz")
        self.meta = json.loads((ASSETS / "kernel_dials.json").read_text())
        self.repair = np.load(ASSETS / "graph_repair.npz")
        self.end_section_wipe(SECTION_TITLES["kernel"], next_slide_prep=lambda: None, notes=NOTES["k01_family"])
        self.stage_family()
        self.stage_mixture()
        self.stage_target()
        self.stage_dials()
        self.stage_denoise()
        self.stage_compact()
        self.stage_no_best()
        self.stage_closing()

    # -- 1. one family ---------------------------------------------------------------------------
    def stage_family(self):
        self.set_title("One family of kernels")
        formula = MathTex(r"q(d) \;=\; \Big(1 + \big(d/\sigma\big)^{p}\Big)^{-\nu}", font_size=48).move_to(to3((0, 2.0)))
        roles = VGroup(caption("\u03c3: a length scale", 22), caption("p: how sharply attraction switches on", 22),
                       caption("\u03bd: how heavy the tail is", 22)).arrange(RIGHT, buff=0.6)
        roles.next_to(formula, DOWN, buff=0.35)
        rows = [
            ("t-SNE", r"\frac{1}{1+d^2}", r"\sigma=1,\; p=2,\; \nu=1"),
            ("heavy-tailed t-SNE", r"\Big(1+\frac{d^2}{\alpha}\Big)^{-\frac{\alpha+1}{2}}",
             r"\sigma=\sqrt{\alpha},\; p=2,\; \nu=\tfrac{\alpha+1}{2}"),
            ("UMAP", r"\frac{1}{1+a\,d^{2b}}", r"\nu=1,\; p=2b,\; \sigma=a^{-1/(2b)}"),
        ]
        table = VGroup()
        for name, kernel, params in rows:
            table.add(VGroup(caption(name, 24), MathTex(kernel, font_size=34), MathTex(params, font_size=30)))
        for row in table:
            row[0].move_to(to3((-4.3, 0)), aligned_edge=LEFT)
            row[1].move_to(to3((-0.6, 0)))
            row[2].move_to(to3((2.2, 0)), aligned_edge=LEFT)
        table.arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        for row in table:
            row[0].set_x(-5.2 + row[0].width / 2)
            row[1].set_x(-0.6)
            row[2].set_x(1.9 + row[2].width / 2)
        table.move_to(to3((0, -1.35)))
        self.play(Write(formula), run_time=1.2)
        self.play(FadeIn(roles))
        self.play(LaggedStart(*[FadeIn(r) for r in table], lag_ratio=0.3), run_time=1.5)
        self.marked_next_slide(notes=NOTES["k02_map"])

        # the (p, nu) map
        self.clear_slide()
        self.set_title("Existing methods sit at points in it")
        chart, axes = small_axes([0, 4.5, 0.5], [0, 3.5, 0.5], "p  (onset sharpness)", "\u03bd  (tail weight)",
                                 7.0, 4.6, (-2.3, -0.4), x_ticks=[1, 2, 3, 4])
        curves = VGroup()
        for c in (0.5, 2.0, 4.0):
            curve = axes.plot(lambda x, c=c: c / x, x_range=[max(c / 3.5, 0.15), 4.5], color=SECONDARY_COLOR,
                              stroke_width=1.5, stroke_opacity=0.6)
            label = caption(f"\u03bdp = {c:g}", 16, color=SECONDARY_COLOR).next_to(axes.c2p(4.5, c / 4.5), RIGHT, buff=0.1)
            curves.add(VGroup(curve, label))
        student = Line(axes.c2p(2, 0.5), axes.c2p(2, 3.5), color=COLOR_CYCLE[4], stroke_width=4)
        tsne = Dot(axes.c2p(2, 1), radius=0.1, color=COLOR_CYCLE[4], stroke_color=STRUCTURE_COLOR, stroke_width=2)
        p_umap = self.meta["umap_p"]
        umap_dot = Dot(axes.c2p(p_umap, 1), radius=0.1, color=HIGHLIGHT_COLOR, stroke_color=STRUCTURE_COLOR,
                       stroke_width=2)
        labels = VGroup(
            caption("t-SNE", 20).next_to(tsne, RIGHT, buff=0.12),
            caption("heavy-tailed t-SNE", 18, color=darker(ManimColor(COLOR_CYCLE[4]), 0.2))
            .next_to(axes.c2p(2, 3.2), RIGHT, buff=0.12),
            caption("UMAP default", 20, color=darker(ManimColor(HIGHLIGHT_COLOR), 0.3))
            .next_to(umap_dot, DOWN + LEFT, buff=0.08),
        )
        col = Column((2.9, 2.2))
        c1 = col.place(caption_stack(["Each method explores one or two", "directions of a three-parameter family."]))
        c2 = col.place(caption_stack([f"UMAP's default curve (min_dist 0.1)", f"is p \u2248 {p_umap:.2f}, \u03bd = 1,",
                                      f"\u03c3 \u2248 {self.meta['umap_sigma']:.2f}: a heavy tail."],
                                     color=STRUCTURE_COLOR), buff=0.35)
        c3 = col.place(caption_stack(["The grey curves hold \u03bdp fixed:", "they will matter shortly."],
                                     color=SECONDARY_COLOR), buff=0.35)
        self.play(FadeIn(chart), FadeIn(curves))
        self.play(Create(student), FadeIn(tsne), FadeIn(labels[0]), FadeIn(labels[1]), FadeIn(c1))
        self.play(FadeIn(umap_dot, scale=1.5), FadeIn(labels[2]), FadeIn(c2))
        self.play(FadeIn(c3))
        self.map_parts = (chart, axes)
        self.marked_next_slide(notes=NOTES["k03_mixture"])

    # -- 2. a Gaussian with an uncertain scale ---------------------------------------------------------
    def stage_mixture(self):
        self.clear_slide()
        self.set_title("A Gaussian with an uncertain scale")
        d, meta = self.data, self.meta
        grid, samples = d["mix_grid"], d["mix_samples"]
        chart, axes = small_axes([0, 4, 1], [0, 1, 0.25], "distance d", "link probability",
                                 6.4, 4.4, (-2.6, -0.45), x_ticks=[1, 2, 3, 4])
        count = ValueTracker(0)
        faint = VGroup(*[axes.plot_line_graph(grid, row, add_vertex_dots=False, line_color=ATTRACT_COLOR,
                                              stroke_width=1.2, stroke_opacity=0.35) for row in samples])

        def average():
            k = max(1, int(round(count.get_value())))
            return axes.plot_line_graph(grid, samples[:k].mean(0), add_vertex_dots=False, line_color=STRUCTURE_COLOR,
                                        stroke_width=4)

        target = DashedVMobject(axes.plot(lambda x: 1 / (1 + x * x), x_range=[0, 4], color=HIGHLIGHT_COLOR,
                                          stroke_width=3), num_dashes=40)
        avg = always_redraw(average)
        col = Column((1.2, 2.45))
        f1 = col.place(MathTex(r"\Big(1+\frac{d^2}{\sigma^2}\Big)^{-\nu} \;=\; \mathbb{E}_{\tau}\big[e^{-\tau d^2}\big],"
                               r"\quad \tau\sim\mathrm{Gamma}(\nu,\ \sigma^2)", font_size=30))
        c1 = col.place(caption_stack(["For p = 2 the kernel is exactly an average of Gaussians",
                                      "whose precision \u03c4 is unknown. Thin curves: Gaussians",
                                      "with sampled \u03c4. Bold: their running average. Dashed:",
                                      "the t-SNE kernel (\u03bd = 1, \u03c3 = 1)."]), buff=0.3)
        c2 = col.place(caption_stack(["The heavy tail is uncertainty about local scale, which",
                                      "noisy distances cannot pin down. \u03bd is the prior's",
                                      "concentration: small \u03bd, heavier tail."], color=STRUCTURE_COLOR), buff=0.35)
        check = col.place(caption(f"checked numerically: agreement to {meta['mixture']['max_error']:.0e}", 18,
                                  color=SECONDARY_COLOR), buff=0.35)
        self.play(FadeIn(chart), Write(f1), FadeIn(c1))
        self.add(avg)
        self.play(LaggedStart(*[Create(f) for f in faint], lag_ratio=0.12), count.animate.set_value(len(samples)),
                  run_time=5, rate_func=linear)
        avg.clear_updaters()
        self.play(Create(target), FadeIn(c2), FadeIn(check))
        self.marked_next_slide(notes=NOTES["k04_target"])

    # -- 3. geometry, not target -----------------------------------------------------------------------
    def stage_target(self):
        self.clear_slide()
        self.set_title("The kernel sets geometry, not the target")
        chart, axes = small_axes([0, 4, 1], [0, 1, 0.25], "distance d", "link probability",
                                 6.4, 4.4, (-2.6, -0.45), x_ticks=[1, 2, 3, 4])
        q_star = 0.5
        kernels = [("t-SNE", 1.0, 2.0, 1.0, COLOR_CYCLE[4]),
                   ("UMAP default", self.meta["umap_sigma"], self.meta["umap_p"], 1.0, HIGHLIGHT_COLOR),
                   ("p = 1, \u03bd = 2", 1.0, 1.0, 2.0, COLOR_CYCLE[0]),
                   ("p = 4, \u03bd = 0.5", 1.0, 4.0, 0.5, COLOR_CYCLE[3])]
        curves, dots = VGroup(), VGroup()
        for name, sigma, p, nu, colour in kernels:
            curves.add(axes.plot(lambda x, s=sigma, p=p, n=nu: float(burr(x, s, p, n)), x_range=[0.001, 4],
                                 color=colour, stroke_width=3))
            d_star = sigma * ((1 / q_star) ** (1 / nu) - 1) ** (1 / p)
            dots.add(Dot(axes.c2p(d_star, q_star), radius=0.08, color=colour, stroke_color=STRUCTURE_COLOR,
                         stroke_width=1.5))
        level = DashedLine(axes.c2p(0, q_star), axes.c2p(4, q_star), color=STRUCTURE_COLOR, stroke_width=2)
        key = VGroup(*[VGroup(Line(ORIGIN, RIGHT * 0.4, color=c, stroke_width=3), caption(n, 18))
                       .arrange(RIGHT, buff=0.1) for n, _, _, _, c in kernels]).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        key.next_to(axes.c2p(4, 1), DL, buff=0.1).shift(LEFT * 0.2)
        col = Column((1.2, 2.45))
        f1 = col.place(MathTex(r"q^{*} \;=\; \frac{w}{w+\gamma}", font_size=40))
        c1 = col.place(caption_stack(["Every edge settles at the same probability,", "whatever the kernel. Here q* = 1/2.",
                                      "The kernels reach it at different distances."]), buff=0.3)
        c2 = col.place(caption_stack(["So choosing a kernel chooses a geometry,", "not a different statistical target."],
                                     color=STRUCTURE_COLOR), buff=0.35)
        check = col.place(caption(f"checked by direct minimisation: agreement to {self.meta['target']['max_error']:.0e}",
                                  18, color=SECONDARY_COLOR), buff=0.35)
        self.play(FadeIn(chart), Create(curves), FadeIn(key), run_time=1.5)
        self.play(Create(level), Write(f1), FadeIn(c1))
        self.play(LaggedStart(*[FadeIn(dot, scale=1.6) for dot in dots], lag_ratio=0.2), FadeIn(c2), FadeIn(check))
        self.marked_next_slide(notes=NOTES["k05_dials"])

    # -- 4. three dials ----------------------------------------------------------------------------------
    def stage_dials(self):
        self.clear_slide()
        self.set_title("Three dials, read off the pull of an edge")
        chart, axes = small_axes([0, 4, 1], [0, 2.5, 0.5], "edge length d", "pull on the edge",
                                 6.4, 4.4, (-2.6, -0.45), x_ticks=[1, 2, 3, 4])
        nup = 2.0
        curves = VGroup()
        for p in (1.0, 1.5, 2.0, 4.0):
            nu = nup / p
            sigma = nu ** (1 / p)
            curves.add(axes.plot(lambda x, s=sigma, p=p, n=nu: float(min(influence(max(x, 1e-3), s, p, n), 2.5)),
                                 x_range=[0.02, 4], color=P_COLORS[p], stroke_width=3))
        tail = DashedVMobject(axes.plot(lambda x: nup / x, x_range=[0.85, 4], color=STRUCTURE_COLOR, stroke_width=2),
                              num_dashes=30)
        tail_label = caption("every curve \u2192 \u03bdp / d", 18).next_to(axes.c2p(3.0, nup / 3.0), UP, buff=0.12)
        key = VGroup(*[VGroup(Line(ORIGIN, RIGHT * 0.4, color=P_COLORS[p], stroke_width=3), caption(f"p = {p:g}", 18))
                       .arrange(RIGHT, buff=0.1) for p in (1.0, 1.5, 2.0, 4.0)]).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        key.next_to(axes.c2p(4, 2.5), DL, buff=0.1)
        col = Column((1.2, 2.45))
        rows = [
            (r"\sigma_0 = \sigma\,\nu^{-1/p}", ["onset scale: where attraction", "switches on"]),
            (r"\psi(d) \sim \nu p\,d^{\,p-1}/\sigma^{p}", ["short range: p sets how neighbours", "behave once they're close"]),
            (r"\psi(d) \to \nu p / d", ["long range: \u03bdp caps how hard a long", "edge can pull: denoising strength"]),
        ]
        items = []
        for k, (tex, lines) in enumerate(rows):
            f = col.place(MathTex(tex, font_size=32), buff=0.3 if k else 0)
            c = col.place(caption_stack(lines, 20), buff=0.12)
            items.append((f, c))
        note = col.place(caption_stack(["Below p = 1 the short-range pull is", "unbounded, so p should be at least 1."],
                                       color=SECONDARY_COLOR, font_size=18), buff=0.35)
        self.play(FadeIn(chart), Create(curves), FadeIn(key), run_time=1.5)
        self.play(Create(tail), FadeIn(tail_label))
        for f, c in items:
            self.play(Write(f), FadeIn(c), run_time=0.8)
        self.play(FadeIn(note))
        self.marked_next_slide(notes=NOTES["k06_denoise"])

    # -- 5. nu*p: the denoising dial -----------------------------------------------------------------------
    def stage_denoise(self):
        self.clear_slide()
        self.set_title("\u03bdp is the denoising dial")
        d, meta = self.data, self.meta
        layouts = d["denoise_layouts"]
        values = meta["denoise_nup"]
        theta, kept, shortcuts = self.repair["theta"], self.repair["kept"], self.repair["shortcuts"]
        colours = [loop_color(t / (2 * PI)) for t in theta]
        step = ValueTracker(values.index(1.0))

        def draw():
            t = float(np.clip(step.get_value(), 0, len(values) - 1))
            i = int(np.floor(t)); j = min(i + 1, len(values) - 1); f = t - i
            Y = (1 - f) * layouts[i] + f * layouts[j]
            P = fit_points(Y, 5.0, 4.8, (-2.8, -0.45))
            return VGroup(segment_cloud(P[kept[:, 0]], P[kept[:, 1]], STRUCTURE_COLOR, 0.6, 0.3),
                          segment_cloud(P[shortcuts[:, 0]], P[shortcuts[:, 1]], WRONG_COLOR, 1.8, 0.85),
                          VGroup(*[Dot(to3(p), radius=0.035, color=c) for p, c in zip(P, colours)]))

        panel = always_redraw(draw)
        col = Column((1.6, 2.45))
        intro = col.place(caption_stack(["The test circle from before: 12 shortcuts", "added, 48 true edges removed. Same",
                                         "graph, same start; only \u03bdp changes."]))

        def stat(key):
            return lambda: meta["denoise"][str(values[int(round(np.clip(step.get_value(), 0, len(values) - 1)))])][key]

        r0 = col.place(MetricReadout("\u03bdp", 1.0, num_decimal_places=1, font_size=24), buff=0.4)
        r1 = col.place(MetricReadout("circle distortion", stat("roundness")(), num_decimal_places=2, font_size=22), buff=0.15)
        r2 = col.place(MetricReadout("overlap with true neighbours", stat("overlap")(), num_decimal_places=2,
                                     font_size=22), buff=0.1)
        r3 = col.place(MetricReadout("missing edges restored (of 48)", stat("restored")(), num_decimal_places=0,
                                     font_size=22), buff=0.1)
        # true edges torn: stretched beyond three times the median true-edge length
        torn = []
        for Y in layouts:
            L = np.linalg.norm(Y[kept[:, 0]] - Y[kept[:, 1]], axis=1)
            torn.append(100 * float(np.mean(L > 3 * np.median(L))))
        r4 = col.place(MetricReadout("true edges torn (%)", torn[values.index(1.0)], num_decimal_places=1,
                                     font_size=22), buff=0.1)
        self.add(panel)
        self.play(FadeIn(intro), FadeIn(r0), FadeIn(r1), FadeIn(r2), FadeIn(r3), FadeIn(r4))
        self.marked_next_slide(notes=NOTES["k07_sweep"])

        index = lambda: int(round(np.clip(step.get_value(), 0, len(values) - 1)))
        r0.number.add_updater(lambda m: m.set_value(values[index()]))
        r1.number.add_updater(lambda m: m.set_value(stat("roundness")()))
        r2.number.add_updater(lambda m: m.set_value(stat("overlap")()))
        r3.number.add_updater(lambda m: m.set_value(stat("restored")()))
        r4.number.add_updater(lambda m: m.set_value(torn[index()]))
        self.play(step.animate.set_value(len(values) - 1), run_time=5, rate_func=smooth)
        self.wait(0.5)
        self.play(step.animate.set_value(0), run_time=5, rate_func=smooth)
        for mob in (panel, r0.number, r1.number, r2.number, r3.number, r4.number):
            mob.clear_updaters()
        verdict = col.place(caption_stack(["Too high: long edges win and the circle folds.",
                                           "Too low: true edges are rejected too, and the",
                                           "circle tears into clumps. Both extremes fail."],
                                          color=STRUCTURE_COLOR), buff=0.4)
        self.play(FadeIn(verdict))
        self.marked_next_slide(notes=NOTES["k08_compact"])

    # -- 6. p: the compactness dial ------------------------------------------------------------------------
    def stage_compact(self):
        self.clear_slide()
        self.set_title("p is the cluster-compactness dial")
        d, meta = self.data, self.meta
        layouts, labels = d["compact_layouts"], d["compact_labels"]
        colours = labels_to_rgb(labels, digit_rgb())
        xs = (-4.5, 0.0, 4.5)
        panels = []
        for x, p, Y in zip(xs, meta["compact_p"], layouts):
            lo, hi = np.quantile(Y, 0.01, axis=0), np.quantile(Y, 0.99, axis=0)
            scale = min(3.6 / (hi[0] - lo[0]), 3.6 / (hi[1] - lo[1]))
            P = np.clip((Y - (lo + hi) / 2) * scale, -1.8, 1.8) + np.array([x, 0.35])  # 1-99% box; outliers clamped
            cloud = EmbeddingCloud(P, colours, fit="none", point_px=4.0)
            head = caption(f"p = {p:g}", 26, color=darker(ManimColor(P_COLORS[p]), 0.2)).move_to(to3((x, 2.45)))
            s = meta["compact"][str(p)]
            r = VGroup(MetricReadout("silhouette", s["silhouette"], num_decimal_places=2, font_size=20),
                       MetricReadout("k-means ARI", s["ari"], num_decimal_places=2, font_size=20))
            r.arrange(DOWN, aligned_edge=LEFT, buff=0.08).move_to(to3((x, -1.95)))
            panels.append((cloud, head, r))
        note = caption_stack(["1,500 MNIST digits, \u03bdp = 2 throughout; only the short-range",
                              "shape changes. With p = 1 the pull stays on as points",
                              "approach, so clusters stay compact."],
                             font_size=20, color=STRUCTURE_COLOR).to_edge(DOWN, buff=0.3)
        for cloud, head, r in panels:
            self.add(cloud)
            self.play(PMFadeIn(cloud), FadeIn(head), FadeIn(r), run_time=0.9)
        self.play(FadeIn(note))
        self.marked_next_slide(notes=NOTES["k09_no_best"])

    # -- 7. no single best kernel ----------------------------------------------------------------------------
    def stage_no_best(self):
        self.clear_slide()
        self.set_title("No single best kernel")
        # a curve with a shortcut, and a sheet with a near-miss edge
        t = np.linspace(0, 2 * PI, 60, endpoint=False)
        ring = np.c_[np.cos(t), np.sin(t)] * 1.15 + [-4.6, 0.6]
        curve = VGroup(VGroup(*[Dot(to3(p), radius=0.035, color=STRUCTURE_COLOR) for p in ring]),
                       Line(to3(ring[5]), to3(ring[33]), color=WRONG_COLOR, stroke_width=4))
        gx, gy = np.meshgrid(np.linspace(-1.1, 1.1, 9), np.linspace(-1.1, 1.1, 9))
        sheet_pts = np.c_[gx.ravel(), gy.ravel()] + [-1.4, 0.6]
        sheet = VGroup(VGroup(*[Dot(to3(p), radius=0.035, color=STRUCTURE_COLOR) for p in sheet_pts]),
                       Line(to3(sheet_pts[30]), to3(sheet_pts[48]), color=WRONG_COLOR, stroke_width=4))
        curve_note = caption_stack(["On a curve, a wrong edge", "is a shortcut: reject hard", "(lower \u03bdp)."],
                                   font_size=17).next_to(curve, DOWN, buff=0.3)
        sheet_note = caption_stack(["On a sheet, most wrong", "edges are near misses:", "reject gently (higher \u03bdp)."],
                                   font_size=17).next_to(sheet, DOWN, buff=0.3)
        col = Column((1.6, 2.45))
        c1 = col.place(caption_stack(["The right setting depends on the structure", "present and on what you want preserved.",
                                      "Neighbourhoods and clusters can pull", "in different directions."]))
        c2 = col.place(caption_stack(["Current working defaults in the research:", "p \u2248 1.5, \u03bd = 1 as an all-rounder;",
                                      "p \u2248 1 for clusters; lower \u03bdp for cycles", "and thin features."],
                                     color=STRUCTURE_COLOR), buff=0.35)
        c3 = col.place(caption_stack(["That is the argument for keeping the", "whole family, not one kernel."],
                                     color=SECONDARY_COLOR), buff=0.35)
        self.play(FadeIn(curve), FadeIn(curve_note))
        self.play(FadeIn(sheet), FadeIn(sheet_note))
        self.play(FadeIn(c1))
        self.play(FadeIn(c2))
        self.play(FadeIn(c3))
        self.marked_next_slide(notes=NOTES["k10_spine"])

    # -- 8. closing ------------------------------------------------------------------------------------------
    def stage_closing(self):
        self.clear_slide()
        self.set_title("One idea, three times: aggregate")
        rows = [
            ("Noise", "Rankings cancel the noise every candidate shares."),
            ("Repair", "Joint fitting weighs each edge against every path."),
            ("Kernel", "The heavy tail averages over an unknown local scale."),
        ]
        table = VGroup()
        for name, text in rows:
            table.add(VGroup(caption(name, 28, color=STRUCTURE_COLOR), caption(text, 26)))
        for row in table:
            row[1].next_to(row[0], RIGHT, buff=0.5)
        table.arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        for row in table:
            row[1].set_x(-2.3 + row[1].width / 2)
            row[0].set_x(-4.9 + row[0].width / 2)
        table.move_to(to3((0, 0.3)))
        rule = caption("Statistics that aggregate survive the noise; single local readings do not.", 24,
                       color=SECONDARY_COLOR).to_edge(DOWN, buff=0.7)
        self.play(LaggedStart(*[FadeIn(r) for r in table], lag_ratio=0.4), run_time=2)
        self.play(FadeIn(rule))
        self.marked_next_slide(notes=NOTES["k11_remedies"])

        self.clear_slide()
        self.set_title("Wrong edges, and what fixes them")
        remedies = [
            ("near misses", "nothing needed: gains and losses balance", SECONDARY_COLOR),
            ("hubs", "correct each point's noise before building the graph", COLOR_CYCLE[4]),
            ("incoherent shortcuts", "joint fitting with a heavy tail, tuned by \u03bdp", ATTRACT_COLOR),
            ("coherent bridges", "information from the raw data: the open problem", WRONG_COLOR),
        ]
        rows = VGroup()
        for name, fix, colour in remedies:
            rows.add(VGroup(Dot(radius=0.09, color=colour), caption(name, 26, color=darker(ManimColor(colour), 0.25)),
                            caption(fix, 24)))
        for row in rows:
            row[1].next_to(row[0], RIGHT, buff=0.2)
        rows.arrange(DOWN, buff=0.55, aligned_edge=LEFT)
        for row in rows:
            row[0].set_x(-5.6)
            row[1].set_x(-5.35 + row[1].width / 2)
            row[2].set_x(-1.2 + row[2].width / 2)
        rows.move_to(to3((0, -0.2)))
        self.play(LaggedStart(*[FadeIn(r) for r in rows], lag_ratio=0.35), run_time=2)
        self.marked_next_slide(notes=NOTES["k12_engineering"])

        self.clear_slide()
        self.set_title("The engineering we just shipped points the same way")
        pairs = [
            ("sampled repulsion, exactly corrected", "negative sampling, hard negatives"),
            ("normalised per-point steps", "Adam"),
            ("small coarse levels first", "recursive initialization"),
            ("coarsen via corroborated edges", "transition-weight coarsening"),
        ]
        heads = VGroup(caption("what a principled version needs", 22, color=SECONDARY_COLOR),
                       caption("UMAP 0.6", 22, color=SECONDARY_COLOR))
        heads[0].move_to(to3((-5.6, 1.9)), aligned_edge=LEFT)
        heads[1].move_to(to3((1.6, 1.9)), aligned_edge=LEFT)
        body = VGroup()
        for k, (need, have) in enumerate(pairs):
            y = 1.1 - 0.85 * k
            a = caption(need, 22).move_to(to3((-5.6, y)), aligned_edge=LEFT)
            arrow = Arrow(to3((0.55, y)), to3((1.4, y)), buff=0, color=SECONDARY_COLOR, stroke_width=3,
                          max_tip_length_to_length_ratio=0.3)
            b = caption(have, 22, color=STRUCTURE_COLOR).move_to(to3((1.6, y)), aligned_edge=LEFT)
            body.add(VGroup(a, arrow, b))
        self.play(FadeIn(heads))
        self.play(LaggedStart(*[FadeIn(r) for r in body], lag_ratio=0.35), run_time=2)
        self.marked_next_slide(notes=NOTES["k13_end"])

        self.clear_slide()
        self.takeaway("The layout is an estimator", "of the latent graph.")
        self.marked_next_slide(notes=NOTES["k14_thanks"])


# ---------------------------------------------------------------------------
# Speaker notes (each note narrates the slide it is attached to)
# ---------------------------------------------------------------------------
NOTES = {
    "k01_family": (
        "The repair depended on the kernel's tail. So consider a family of kernels with three "
        "parameters: a length scale, an onset sharpness p and a tail weight nu. The familiar "
        "kernels are slices of it: t-SNE, heavy-tailed t-SNE and UMAP."
    ),
    "k02_map": (
        "Put them on a map of p against nu. t-SNE is a point; heavy-tailed t-SNE is a line; "
        "UMAP's default curve sits at p about 1.8 and nu one. The grey curves hold nu times p "
        "fixed, and that product will turn out to matter."
    ),
    "k03_mixture": (
        "A probabilistic reading. For p equal to 2, the kernel is exactly a Gaussian affinity "
        "whose precision is unknown and averaged out, with a Gamma prior. Watch Gaussians with "
        "randomly drawn widths average into the heavy-tailed kernel. The heavy tail is "
        "uncertainty about local scale, which is exactly what the noise section said we can't "
        "measure from distances. We checked the identity numerically."
    ),
    "k04_target": (
        "Recall that each edge targets probability w over w plus gamma, whatever the kernel. "
        "Draw that level across four kernels: they reach it at different distances. So "
        "choosing a kernel is choosing a geometry, not a different statistical target."
    ),
    "k05_dials": (
        "Reorganise the parameters by what they control, using the pull an edge exerts at "
        "length d. Here nu p is held at 2 and only p varies. At long range every curve falls "
        "like nu p over d: nu p caps how hard a long edge can pull. At short range p decides how "
        "neighbours behave once they're close. And sigma times nu to the minus one over p is the "
        "scale where attraction switches on."
    ),
    "k06_denoise": (
        "Nu p is the denoising dial. Here is the test circle from the repair section: same "
        "graph, same start, only nu p changes. We start at nu p equal to one."
    ),
    "k07_sweep": (
        "Turn it up: long edges pull harder, and by nu p of about four the circle folds, as the "
        "spring did. Turn it down to a half: rejection gets so aggressive that true edges are "
        "rejected too, and the circle tears into clumps. Both extremes fail."
    ),
    "k08_compact": (
        "P is the cluster-compactness dial. Fifteen hundred MNIST digits, nu p fixed at 2, only "
        "the short-range shape changing. With p equal to one, the pull stays on as points "
        "approach, so clusters stay compact."
    ),
    "k09_no_best": (
        "There's no single best kernel. On a curve, a wrong edge is a shortcut that changes the "
        "shape, so rejecting hard pays. On a sheet, most wrong edges are near misses within the "
        "sheet, so rejecting them costs real neighbours. The right setting depends on the "
        "structure and on what you want to preserve, which is the argument for keeping the "
        "whole family."
    ),
    "k10_spine": (
        "To pull the three threads together: one idea keeps recurring. Rankings survive noise "
        "because they cancel what candidates share. Joint fitting repairs the graph because each "
        "edge is weighed against every path. And the heavy tail is an average over an unknown "
        "scale. Aggregate, and the noise loses."
    ),
    "k11_remedies": (
        "Each kind of wrong edge has its own remedy. Near misses need nothing. Hubs: correct "
        "each point's noise before building the graph. Incoherent shortcuts: joint fitting with "
        "a heavy tail, tuned by nu p. Coherent bridges need information from the raw data, and "
        "that's the open problem."
    ),
    "k12_engineering": (
        "And the engineering in UMAP 0.6 points the same way. A principled version needs "
        "sampled repulsion with a correction, per-point normalised steps, coarse levels solved "
        "first, and coarsening that only trusts corroborated edges. We have negative sampling "
        "and hard negatives, Adam, recursive initialization, and, if it holds up in testing, "
        "transition-weight coarsening."
    ),
    "k13_end": "The layout isn't just a picture of the graph: it's an estimator of the latent graph.",
    "k14_thanks": "Thank you.",
}
