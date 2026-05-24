from manim import *

import sys

sys.path.append("..")
sys.path.append(".")

from config import (
    apply_defaults,
    COLOR_CYCLE,
    DEFAULT_COLOR,
    ACCENT_COLOR,
    HIGHLIGHT_COLOR,
    BACKGROUND_COLOR,
    add_logo_to_background,
    TIMCSlide,
    colormap_color,
)

import numpy as np
import sklearn.neighbors

apply_defaults()


class UMAPExplanation(TIMCSlide):
    """
    Visual introduction to UMAP for an expert audience unfamiliar with the
    algorithm.  The goal is intuition, not rigour.

    Slide flow
    ----------
    Beat 1 – Two-phase overview
      UMAP = (I) build adaptive weighted k-NN graph  →  (II) optimise low-D layout

    Beat 2 – Phase I: k-NN graph construction
      Scatter of ~30 two-density points; edges to k nearest neighbours appear.

    Beat 3 – Adaptive bandwidth (key visual)
      Circles of radius σᵢ (= k-th-neighbour distance) grow around each point.
      Dense region → small circles; sparse region → large circles.

    Beat 4a – Edge weight formula
      Edges replayed with opacity ∝ wᵢⱼ = exp(-(d - ρᵢ)/σᵢ).
      Formula displayed above scatter.

    Beat 4b – Symmetrisation
      Fuzzy-union symmetrisation formula and brief note added below weight eq.

    Beat 5 – Low-D similarity kernel
      Axes showing q(d) = (1 + a d^{2b})^{-1}.

    Beat 6a – Cross-entropy loss (write)
    Beat 6b – Attraction term annotated
    Beat 6c – Repulsion term annotated
    """

    K = 5  # number of nearest neighbours

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _make_data(self):
        """Points noisily on a unit circle with inhomogeneous density.

        Sampling rate is proportional to
            p(θ) = α + (1-α) · (sin θ + 1) / 2
        so the top of the circle (θ ≈ π/2) is dense and the bottom
        (θ ≈ -π/2) is sparse.  Small radial Gaussian noise gives the
        'noisy circle' look consistent with other slides in this talk.
        """
        rng = np.random.default_rng(42)

        alpha = 0.125  # density floor: bottom ≈ 12.5 % of top rate
        n_target = 40  # total number of points
        radial_noise = 0.1  # std of radial jitter

        # Rejection sampling of angles
        thetas: list[float] = []
        while len(thetas) < n_target:
            batch = rng.uniform(0.0, 2.0 * np.pi, 4 * n_target)
            p_accept = alpha + (1.0 - alpha) * (np.sin(batch) + 1.0) / 2.0
            accepted = batch[rng.random(len(batch)) < p_accept]
            thetas.extend(accepted.tolist())
        thetas = np.asarray(thetas[:n_target])

        radii = 1.0 + rng.normal(0.0, radial_noise, n_target)
        data = np.column_stack([radii * np.cos(thetas), radii * np.sin(thetas)])

        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=self.K + 1).fit(data)
        dists, indices = nbrs.kneighbors(data)
        rho = dists[:, 1]  # nearest-neighbour distance
        sigma = dists[:, self.K]  # k-th-neighbour distance (bandwidth)
        return data, rho, sigma, indices.astype(int)

    @staticmethod
    def _to_scene(data, cx=0.0, cy=0.0, scale=5.0):
        """Uniformly normalise data to Manim scene coordinates."""
        mn, mx = data.min(axis=0), data.max(axis=0)
        spread = float((mx - mn).max()) or 1.0
        norm = (data - (mn + mx) * 0.5) / spread
        return np.column_stack(
            [
                norm[:, 0] * scale + cx,
                norm[:, 1] * scale + cy,
                np.zeros(len(data)),
            ]
        )

    # ------------------------------------------------------------------ #
    #  construct                                                           #
    # ------------------------------------------------------------------ #

    def construct(self):

        # ── Pre-compute everything ──────────────────────────────────── #

        data, rho, sigma, indices = self._make_data()
        n = len(data)

        spread = float((data.max(axis=0) - data.min(axis=0)).max()) or 1.0
        SCALE = 4.5
        pts = self._to_scene(data, cx=0.0, cy=0.1, scale=SCALE)

        # scene-unit versions of ρ and σ (same linear map as _to_scene)
        rho_s = rho / spread * SCALE
        sigma_s = sigma / spread * SCALE

        def w_sym(i, j):
            """Fuzzy-union (OR) symmetric edge weight, clipped to [0, 1]."""
            d = float(np.linalg.norm(data[i] - data[j]))
            wi = float(np.exp(-(d - rho[i]) / max(sigma[i], 1e-9)))
            wj = float(np.exp(-(d - rho[j]) / max(sigma[j], 1e-9)))
            wi = float(np.clip(wi, 0.0, 1.0))
            wj = float(np.clip(wj, 0.0, 1.0))
            return wi + wj - wi * wj

        # Undirected k-NN edge list (sorted pairs, no duplicates)
        edge_set = set()
        for i in range(n):
            for j in indices[i, 1:]:
                edge_set.add((min(int(i), int(j)), max(int(i), int(j))))
        edge_list = sorted(edge_set)

        # ── Beat 1: Two-phase overview ──────────────────────────────── #

        title = Text("UMAP", font_size=56).to_edge(UP, buff=0.45)

        phase1 = (
            VGroup(
                Text("Phase I", font_size=34, color=ACCENT_COLOR),
                Text("Build weighted k-NN graph", font_size=28),
                Text("adaptive local metric", font_size=22, color=ACCENT_COLOR),
            )
            .arrange(DOWN, buff=0.25)
            .shift(LEFT * 3.6 + DOWN * 0.3)
        )

        phase2 = (
            VGroup(
                Text("Phase II", font_size=34, color=ACCENT_COLOR),
                Text("Optimise low-D layout", font_size=28),
                Text("cross-entropy loss", font_size=22, color=ACCENT_COLOR),
            )
            .arrange(DOWN, buff=0.25)
            .shift(RIGHT * 3.6 + DOWN * 0.3)
        )

        arrow_ov = Arrow(
            phase1.get_right() + RIGHT * 0.1,
            phase2.get_left() - RIGHT * 0.1,
            color=HIGHLIGHT_COLOR,
            buff=0.0,
            stroke_width=6,
        )

        self.play(Write(title))
        self.play(FadeIn(phase1, shift=RIGHT * 0.3))
        self.play(GrowArrow(arrow_ov))
        self.play(FadeIn(phase2, shift=LEFT * 0.3))
        self.marked_next_slide()

        self.play(FadeOut(phase1), FadeOut(arrow_ov), FadeOut(phase2))

        # ── Beat 2: k-NN graph construction ────────────────────────── #

        dots = VGroup(
            *[
                Dot(
                    pts[i],
                    radius=0.075,
                    color=DEFAULT_COLOR,
                    stroke_color=WHITE,
                    stroke_width=1.0,
                ).set_z_index(2)
                for i in range(n)
            ]
        )

        knn_lines = VGroup(
            *[
                Line(
                    pts[a],
                    pts[b],
                    color=ACCENT_COLOR,
                    stroke_width=2.5,
                    stroke_opacity=0.65,
                )
                for (a, b) in edge_list
            ]
        )

        cap2 = Tex(
            r"Connect each point to its $k$ nearest neighbours \quad ($k = 5$)",
            font_size=28,
        ).to_edge(DOWN, buff=0.5)

        self.play(
            LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.05),
            run_time=1.5,
        )
        self.play(Write(cap2), run_time=0.7)
        self.play(
            LaggedStart(*[Create(e) for e in knn_lines], lag_ratio=0.025),
            run_time=3.0,
        )
        self.wait()
        self.marked_next_slide()

        # ── Beat 3: Adaptive bandwidth ──────────────────────────────── #

        s_min = float(sigma_s.min())
        s_max = float(sigma_s.max())

        circles = VGroup(
            *[
                Circle(
                    radius=float(sigma_s[i]),
                    color=colormap_color(
                        float(sigma_s[i]), s_min, s_max, cmap_name="plasma"
                    ),
                    stroke_width=4.0,
                    stroke_opacity=0.82,
                    fill_color=colormap_color(
                        float(sigma_s[i]), s_min, s_max, cmap_name="plasma"
                    ),
                    fill_opacity=0.125,
                ).move_to(pts[i])
                for i in range(n)
            ]
        )

        i_dense = int(np.argmin(sigma_s))
        i_sparse = int(np.argmax(sigma_s))

        lbl_dense = Text(
            "σ small\n(dense)", font_size=36, color=COLOR_CYCLE[0]
        ).next_to([0, max([dot.get_center()[1] for dot in dots]), 0], LEFT, buff=3.5)
        arr_dense = Arrow(
            lbl_dense.get_center() + RIGHT,
            [-0.5, max([dot.get_center()[1] for dot in dots]), 0],
            color=COLOR_CYCLE[0],
            stroke_width=3,
            buff=0.5,
            max_tip_length_to_length_ratio=0.18,
        )

        lbl_sparse = Text(
            "σ large\n(sparse)", font_size=36, color=COLOR_CYCLE[3]
        ).next_to([0, min([dot.get_center()[1] for dot in dots]), 0], RIGHT, buff=3.5)
        arr_sparse = Arrow(
            lbl_sparse.get_center() + LEFT,
            [0.5, min([dot.get_center()[1] for dot in dots]), 0],
            color=COLOR_CYCLE[3],
            stroke_width=3,
            buff=0.5,
            max_tip_length_to_length_ratio=0.18,
        )

        cap3 = (
            Tex(
                r"{\large$\sigma_i$}\\\phantom{line of text}\\distance to\\$k$-th neighbour\\\phantom{line of text}\\adapts to\\local density",
                font_size=36,
            )
            .to_edge(LEFT, buff=1)
            .shift(DOWN)
        )

        self.play(FadeOut(cap2))
        self.wait()
        self.play(
            LaggedStart(*[GrowFromCenter(c) for c in circles], lag_ratio=0.04),
            run_time=4.0,
        )
        self.wait()
        self.play(
            FadeIn(lbl_dense),
            GrowArrow(arr_dense),
            FadeIn(lbl_sparse),
            GrowArrow(arr_sparse),
            Write(cap3),
            run_time=2.5,
        )
        self.wait()
        self.marked_next_slide()

        # ── Beat 4a: Edge weight formula ────────────────────────────── #

        weights = np.array(
            [float(np.clip(w_sym(a, b), 0.01, 1.0)) for (a, b) in edge_list]
        )

        weighted_lines = VGroup(
            *[
                Line(
                    pts[a],
                    pts[b],
                    color=ACCENT_COLOR,
                    stroke_width=float(1.5 + weights[k] ** 2 * 4.0),
                    stroke_opacity=float(0.15 + weights[k] ** 2 * 0.75),
                )
                for k, (a, b) in enumerate(edge_list)
            ]
        )

        w_eq = (
            MathTex(
                r"w_{ij} = \exp\!\left(-\,\frac{d(i,j) - \rho_i}{\sigma_i}\right)",
                font_size=32,
            )
            .to_edge(RIGHT, buff=0.5)
            .shift(UP * 1.5)
        )

        cap4 = MathTex(
            r"\rho_i = \text{nearest-neighbour dist.}, \quad"
            r"\sigma_i = \text{bandwidth}",
            font_size=40,
        ).to_edge(DOWN, buff=0.5)

        self.play(
            FadeOut(lbl_dense),
            FadeOut(arr_dense),
            FadeOut(lbl_sparse),
            FadeOut(arr_sparse),
            FadeOut(circles),
            FadeOut(cap3),
            run_time=0.6,
        )
        self.play(FadeOut(knn_lines), FadeIn(weighted_lines), run_time=1.2)
        self.play(Write(w_eq), Write(cap4), run_time=1.5)
        self.marked_next_slide()

        # ── Beat 4b: Symmetrisation ─────────────────────────────────── #

        sym_eq = (
            MathTex(
                r"w^{\mathrm{sym}}_{ij} = w_{ij} + w_{ji} - w_{ij}\,w_{ji}",
                font_size=32,
            )
            .to_edge(LEFT, buff=0.5)
            .shift(DOWN * 1.5)
        )

        sym_note = Paragraph(
            "fuzzy union\n(can be built from a colimit over local edges)",
            font_size=16,
            color=ACCENT_COLOR,
            alignment="center",
        ).next_to(sym_eq, DOWN, buff=0.25)

        self.play(Write(sym_eq), run_time=1.0)
        self.play(FadeIn(sym_note), run_time=0.7)
        self.marked_next_slide()

        # ── Beat 5: Low-D similarity kernel ─────────────────────────── #

        self.play(
            FadeOut(dots),
            FadeOut(weighted_lines),
            FadeOut(w_eq),
            FadeOut(sym_eq),
            FadeOut(sym_note),
            FadeOut(cap4),
            run_time=0.7,
        )

        axes5 = Axes(
            x_range=[0.0, 3.2, 1.0],
            y_range=[0.0, 1.1, 0.5],
            x_length=6.5,
            y_length=4.2,
            axis_config={"include_tip": False, "color": DEFAULT_COLOR},
        ).shift(LEFT * 2.8 + DOWN * 0.5)

        x_lbl5 = MathTex(r"\|y_i - y_j\|", font_size=28).next_to(
            axes5.get_x_axis(), DOWN, buff=0.35
        )
        y_lbl5 = (
            MathTex(r"q_{ij}", font_size=28)
            .rotate(PI / 2)
            .next_to(axes5.get_y_axis(), LEFT, buff=0.35)
        )

        a_param, b_param = 1.929, 0.791  # UMAP default parameters
        curve5 = axes5.plot(
            lambda d: 1.0 / (1.0 + a_param * d ** (2 * b_param)),
            x_range=[0.0, 3.2, 0.02],
            color=ACCENT_COLOR,
            stroke_width=6,
        )

        kern_eq = MathTex(
            r"q_{ij} = \bigl(1 + a\,\|y_i - y_j\|^{2b}\bigr)^{\!-1}",
            font_size=48,
        ).shift(RIGHT * 3.2 + UP * 1.2)

        kern_note = (
            VGroup(
                Text("Smooth kernel on low-D distances", font_size=32),
                Tex(
                    r"(approximates $\exp\!\left(-\frac{\max(d - \rho, 0)}{\sigma}\right)$\\ for globally set $\sigma$ and $\rho$)",
                    font_size=18,
                    color=ACCENT_COLOR,
                ),
            )
            .arrange(DOWN, buff=0.2)
            .next_to(kern_eq, DOWN, buff=0.5)
        )

        self.play(Create(axes5), Write(x_lbl5), Write(y_lbl5), run_time=1.0)
        self.play(Create(curve5), run_time=1.2)
        self.play(Write(kern_eq), run_time=1.0)
        self.play(FadeIn(kern_note), run_time=0.8)
        self.marked_next_slide()

        # ── Beat 6: Cross-entropy loss ──────────────────────────────── #

        self.play(
            FadeOut(axes5),
            FadeOut(x_lbl5),
            FadeOut(y_lbl5),
            FadeOut(curve5),
            FadeOut(kern_eq),
            FadeOut(kern_note),
            run_time=0.7,
        )

        ce_title = Text("Cross-Entropy Loss", font_size=56, color=ACCENT_COLOR).to_edge(
            UP, buff=0.45
        )
        self.play(FadeOut(title), FadeIn(ce_title), run_time=0.6)

        # Four submobjects so we can colour [1] and [3] independently
        ce_loss = MathTex(
            r"\mathcal{L} \;=\; -\!\sum_{(i,j)}\! \Big[",
            r"w_{ij}\,\log q_{ij}",
            r"\;+\;",
            r"(1 - w_{ij})\,\log(1 - q_{ij})",
            r"\Big]",
            font_size=60,
        ).move_to(ORIGIN)

        self.play(Write(ce_loss), run_time=2.0)
        self.marked_next_slide()

        # Attraction term
        attr_color = COLOR_CYCLE[0]
        self.play(ce_loss[1].animate.set_color(attr_color))
        brace_a = Brace(ce_loss[1], DOWN, color=attr_color, buff=0.1)
        lbl_a = Text("attraction", font_size=28, color=attr_color).next_to(
            brace_a, DOWN, buff=0.15
        )
        sub_a = Text(
            "pulls connected pairs closer",
            font_size=16,
            color=attr_color,
        ).next_to(lbl_a, DOWN, buff=0.12)

        self.play(GrowFromCenter(brace_a), FadeIn(lbl_a), run_time=1.0)
        self.play(FadeIn(sub_a), run_time=0.7)
        self.marked_next_slide()

        # Repulsion term
        rep_color = COLOR_CYCLE[3]
        self.play(ce_loss[3].animate.set_color(rep_color))
        brace_r = Brace(ce_loss[3], DOWN, color=rep_color, buff=0.1)
        lbl_r = Text("repulsion", font_size=28, color=rep_color).next_to(
            brace_r, DOWN, buff=0.15
        )
        sub_r = Text(
            "pushes unconnected pairs apart",
            font_size=16,
            color=rep_color,
        ).next_to(lbl_r, DOWN, buff=0.12)

        self.play(GrowFromCenter(brace_r), FadeIn(lbl_r), run_time=1.0)
        self.play(FadeIn(sub_r), run_time=0.7)
        self.marked_next_slide()
