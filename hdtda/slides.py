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


class ForceParameterization(TIMCSlide):
    def construct(self):

        axes = Axes(
            x_range=[0.0, 2.5, 0.5],
            y_range=[0.0, 1.25, 0.5],
            x_length=8,
            y_length=6,
            axis_config={"include_tip": False},
        ).add_coordinates()

        self.play(Create(axes))

        # f(x) = 1 / (1 + (max(0, x - shift) / scale)^p)^q
        BASE_SCALE = 0.75
        BASE_P = 2.0
        BASE_Q = 1.5

        def force(x, scale=BASE_SCALE, shift=0.0, p=BASE_P, q=BASE_Q):
            v = max(0.0, x - shift) / scale
            return 1.0 / (1.0 + v**p) ** q

        PLOT_STEP = 0.005
        graph = axes.plot(
            lambda x: force(x),
            x_range=[0.0, 2.5, PLOT_STEP],
            color=ACCENT_COLOR,
            stroke_width=9,
        )
        self.play(Create(graph))

        self.marked_next_slide()

        import matplotlib.cm as _cm

        def _plasma(t):
            """Sample the plasma colormap at t in [0, 1], return hex string."""
            r, g, b, _ = _cm.plasma(float(t))
            return "#{:02x}{:02x}{:02x}".format(
                int(r * 255), int(g * 255), int(b * 255)
            )

        def make_ghost_fan(param, values, **fixed):
            n = len(values)
            g = VGroup()
            for i, val in enumerate(values):
                kw = dict(fixed, **{param: val})
                color = _plasma(i / max(n - 1, 1))
                gc = axes.plot(
                    lambda x, kw=kw: force(x, **kw),
                    x_range=[0.0, 2.5, PLOT_STEP],
                    color=color,
                    stroke_width=2.5,
                )
                gc.set_stroke(opacity=0.55)
                g.add(gc)
            return g

        def smooth_sweep(waypoints, make_graph_fn, total_duration=3.0):
            """
            Continuously morphs the live curve through each waypoint value.
            Uses a ValueTracker + always_redraw so transitions are frame-continuous
            with no discrete steps or pauses between waypoints.
            """
            nonlocal graph
            n_segs = len(waypoints) - 1
            seg_dur = total_duration / n_segs
            param_t = ValueTracker(waypoints[0])
            live = always_redraw(lambda: make_graph_fn(param_t.get_value()))
            self.remove(graph)
            self.add(live)
            for target in waypoints[1:]:
                self.play(
                    param_t.animate.set_value(target),
                    run_time=seg_dur,
                    rate_func=smooth,
                )
            # Swap live graph for a static one so later transforms work normally
            final = make_graph_fn(waypoints[-1])
            self.remove(live)
            self.add(final)
            graph = final

        # ── Scale ─────────────────────────────────────────────────────────
        SCALE_GHOSTS = [0.2, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        ghosts = make_ghost_fan(
            "scale",
            SCALE_GHOSTS,
            shift=0.0,
            p=BASE_P,
            q=BASE_Q,
        )
        param_text = Text("Scale", font_size=22).move_to(axes.c2p(1.5, 0.9))
        self.play(Write(param_text), FadeIn(ghosts))
        smooth_sweep(
            [BASE_SCALE, 2.0, 0.2, BASE_SCALE],
            lambda sv: axes.plot(
                lambda x, sv=sv: force(x, scale=sv),
                x_range=[0.0, 2.5, PLOT_STEP],
                color=ACCENT_COLOR,
                stroke_width=9,
            ),
        )
        self.play(FadeOut(ghosts), FadeOut(param_text))

        self.marked_next_slide()

        # ── Shoulder width ─────────────────────────────────────────────────
        SHIFT_GHOSTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        ghosts = make_ghost_fan(
            "shift",
            SHIFT_GHOSTS,
            scale=BASE_SCALE,
            p=BASE_P,
            q=BASE_Q,
        )
        param_text = Text("Shoulder width", font_size=22).move_to(axes.c2p(1.5, 0.9))
        self.play(Write(param_text), FadeIn(ghosts))
        # Shift only has a positive direction (base = 0); sweep out and back
        smooth_sweep(
            [0.0, 0.8, 0.0],
            lambda sh: axes.plot(
                lambda x, sh=sh: force(x, shift=sh),
                x_range=[0.0, 2.5, PLOT_STEP],
                color=ACCENT_COLOR,
                stroke_width=9,
            ),
            total_duration=2.0,
        )
        self.play(FadeOut(ghosts), FadeOut(param_text))

        self.marked_next_slide()

        # ── Shoulder sharpness: vary p while keeping p·q = BASE_P·BASE_Q ─
        # This holds the large-x tail exponent constant so only the knee
        # shape changes, not the rate of decay in the tail.
        TAIL_CONST = BASE_P * BASE_Q  # = 3.0
        P_GHOSTS = [0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]

        def _sharpness_ghost_fan():
            n = len(P_GHOSTS)
            g = VGroup()
            for i, pv in enumerate(P_GHOSTS):
                qv = TAIL_CONST / pv
                color = _plasma(i / max(n - 1, 1))
                gc = axes.plot(
                    lambda x, pv=pv, qv=qv: force(x, p=pv, q=qv),
                    x_range=[0.0, 2.5, PLOT_STEP],
                    color=color,
                    stroke_width=2.5,
                )
                gc.set_stroke(opacity=0.55)
                g.add(gc)
            return g

        ghosts = _sharpness_ghost_fan()
        param_text = Text("Shoulder sharpness", font_size=22).move_to(
            axes.c2p(1.5, 0.9)
        )
        self.play(Write(param_text), FadeIn(ghosts))
        smooth_sweep(
            [BASE_P, 8.0, 0.4, BASE_P],
            lambda pv: axes.plot(
                lambda x, pv=pv, qv=TAIL_CONST / pv: force(x, p=pv, q=qv),
                x_range=[0.0, 2.5, PLOT_STEP],
                color=ACCENT_COLOR,
                stroke_width=9,
            ),
        )
        self.play(FadeOut(ghosts), FadeOut(param_text))

        self.marked_next_slide()

        # ── Tail decay (outer exponent q only; scale and p fixed) ─────────
        # q < 1: heavy tails, force persists at large distances
        # q >> 1: tight support, force drops to near-zero quickly
        Q_GHOSTS = [0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
        ghosts = make_ghost_fan(
            "q",
            Q_GHOSTS,
            scale=BASE_SCALE,
            shift=0.0,
            p=BASE_P,
        )
        param_text = Text("Tail decay", font_size=22).move_to(axes.c2p(1.5, 0.9))
        self.play(Write(param_text), FadeIn(ghosts))
        smooth_sweep(
            [BASE_Q, 5.0, 0.4, BASE_Q],
            lambda qv: axes.plot(
                lambda x, qv=qv: force(x, q=qv),
                x_range=[0.0, 2.5, PLOT_STEP],
                color=ACCENT_COLOR,
                stroke_width=9,
            ),
        )
        self.play(FadeOut(ghosts), FadeOut(param_text))


class HighDLissajousKnotUMAP2DWithEdges(TIMCSlide):

    _n_samples = 2000
    _seed = 42
    _generator_kwargs: dict = dict(
        target_dim=512,
        radius=2.0,
        n_planes=5,
        noise=0.01,
        noise_hd=0.175,
        dataset_type="highdim_loop",
    )
    _stroke_width = 16

    def construct(self):
        import umap

        generator = CurvyLoopEmbedding(n_samples=self._n_samples, seed=self._seed)
        _, X_embedded, _ = generator.generate_dataset(**self._generator_kwargs)
        mapper = umap.UMAP(
            random_state=42,
            n_components=3,
            n_neighbors=16,
            repulsion_strength=0.5,
            n_epochs=500,
        ).fit(X_embedded)
        embedding_2d = _normalize_to_axes(
            np.ascontiguousarray(mapper.embedding_[:, [0, 2]])
        )

        fade_point_indices = np.random.RandomState(42).choice(
            self._n_samples, size=int(self._n_samples * 0.30), replace=False
        )
        fade_point_embedding_2d = embedding_2d[fade_point_indices]
        fade_point_colors = [
            colorcet.CET_C9[int(t / self._n_samples * (len(colorcet.CET_C9) - 1))]
            for t in fade_point_indices
        ]

        axes = Axes(x_range=[-2.0, 2.0], y_range=[-2.0, 2.0])
        points = self.create_points(embedding_2d, axes)
        fade_points = [
            Dot(axes.c2p(*pt), radius=0.03, color=c)
            for pt, c in zip(fade_point_embedding_2d, fade_point_colors)
        ]

        self.camera.get_thickening_nudges = _types.MethodType(
            lambda cam, t: _make_circular_nudges(t), self.camera
        )

        self.play(
            LaggedStart(*[FadeIn(fp) for fp in fade_points], lag_ratio=0.01),
            run_time=1.0,
        )
        self.play(FadeIn(points))

        self.wait()
        self.marked_next_slide()

        # Build one VMobject per width bucket.
        # Cairo sets line width once per draw call, so true per-segment widths
        # require separate objects. Bucketing keeps the count to O(N_BUCKETS)
        # instead of O(N_edges) while still conveying edge strength visually.
        # Each edge A→B is packed as the degenerate cubic bezier [A, A, B, B].
        N_BUCKETS = 64
        EDGE_WIDTH_MIN = 0.01
        EDGE_WIDTH_MAX = 1.0

        graph = mapper.graph_.tocoo()
        all_coords = np.array([axes.c2p(*pt) for pt in embedding_2d], dtype=np.float64)
        src = all_coords[graph.row]  # (M, 3)
        dst = all_coords[graph.col]  # (M, 3)
        weights = np.asarray(graph.data, dtype=np.float64)
        weights = (weights - weights.min()) / (
            weights.max() - weights.min() + 1e-9
        )  # normalise 0→1
        bucket_idx = np.clip((weights * N_BUCKETS).astype(int), 0, N_BUCKETS - 1)

        edge_grp = VGroup()
        for b in range(N_BUCKETS):
            mask = bucket_idx == b
            if not mask.any():
                continue
            pts = np.empty((4 * mask.sum(), 3), dtype=np.float64)
            pts[0::4] = src[mask]
            pts[1::4] = src[mask]
            pts[2::4] = dst[mask]
            pts[3::4] = dst[mask]
            w = EDGE_WIDTH_MIN + (EDGE_WIDTH_MAX - EDGE_WIDTH_MIN) * (
                b / (N_BUCKETS - 1)
            )
            mob = VMobject(
                stroke_width=w, stroke_color=GRAY, fill_opacity=0, stroke_opacity=w
            )
            mob.set_points(pts)
            edge_grp.add(mob)

        edge_grp.set_z_index(-1)
        self.play(
            LaggedStart(
                *[FadeIn(mob) for mob in edge_grp],
                lag_ratio=0.5,
                run_time=2.0,
            )
        )

        self.wait()
        self.marked_next_slide()

        self.play(PMFadeOut(points))
        self.play(
            LaggedStart(*[FadeOut(fp) for fp in fade_points], lag_ratio=0.01),
            run_time=0.5,
        )
        self.play(
            edge_grp.animate.scale(2.75, about_point=(-3.0, -1.5, 0.0)),
        )

    def create_points(self, data: np.ndarray, axes) -> PMobject:
        from manim.utils.color import color_to_rgba

        coords = np.array(
            [axes.c2p(*x) for x in data],
            dtype=np.float64,
        )
        ts = np.linspace(0, 1, len(data))
        rgbas = np.array(
            [
                color_to_rgba(colorcet.CET_C9[int(t * (len(colorcet.CET_C9) - 1))])
                for t in ts
            ]
        )
        pm = PMobject(stroke_width=self._stroke_width)
        pm.add_points(coords, rgbas=rgbas)
        return pm


class LEtoFDExplanation(TIMCSlide):
    def construct(self):

        loss = MathTex(
            r"\mathcal{L} = \mathop{tr}(Y^\top L Y)",
            font_size=56,
        )
        constrain_text = Text(
            "subject to",
            font_size=28,
            color=ACCENT_COLOR,
        ).next_to(loss, DOWN, buff=1.0)
        constraint = (
            VGroup(
                MathTex(r"Y^\top Y = I"),
                Text("and", font_size=36, color=ACCENT_COLOR),
                MathTex(r"Y^\top \mathbf{1} = 0"),
            )
            .arrange(buff=0.5, center=False, aligned_edge=DOWN)
            .next_to(constrain_text, DOWN, buff=0.5)
        )
        self.play(Write(loss))
        self.wait()
        self.play(Write(constrain_text))
        self.play(Write(constraint))
        self.marked_next_slide()

        new_loss = MathTex(
            r"\mathcal{L} = \frac{1}{2} \sum_{i,j} W_{ij} \|y_i - y_j\|^2",
            font_size=56,
        )
        self.play(Transform(loss, new_loss))
        self.marked_next_slide()

        # Build the final combined expression as a two-string MathTex so that
        # Manim compiles both parts in a single LaTeX run.  This means
        # final_loss[0] and final_loss[1] expose the *exact* centres those parts
        # occupy in the finished single-line rendering — not the centre-aligned
        # approximation that VGroup.arrange() would give.  We use those submobject
        # centres as animation targets so the fly-in lands pixel-perfectly.
        final_loss = MathTex(
            r"\mathcal{L} = \frac{1}{2} \sum_{i,j} W_{ij} \|y_i - y_j\|^2",
            r"+ \lambda \|Y^\top Y - I\|_F",
            font_size=56,
        ).move_to(ORIGIN)

        # Pre-position the Lagrange term at exactly where it will live in
        # final_loss so the ReplacementTransform lands without any jump.
        lagrange_term = MathTex(
            r"+ \lambda \|Y^\top Y - I\|_F",
            font_size=56,
        ).move_to(final_loss[1].get_center())

        self.play(
            # FadeOut(constrain_text),
            FadeOut(constraint[1]),  # "and"
            constraint[2].animate.move_to(
                constraint.get_center()
            ),  # Y^\top \mathbf{1} = 0
            loss.animate.move_to(final_loss[0].get_center()),
            ReplacementTransform(constraint[0], lagrange_term),
        )

        # Instantaneous swap: loss sits at final_loss[0]'s centre and
        # lagrange_term sits at final_loss[1]'s centre, so removing them and
        # adding final_loss is visually seamless.
        self.remove(loss, lagrange_term)
        self.add(final_loss)
        self.marked_next_slide()

        new_loss = MathTex(
            r"\mathcal{L} = \frac{1}{2} \sum_{i,j} W_{ij} \|y_i - y_j\|^2 - \lambda \left(\mathop{tr}(Y^\top Y)\right)",
            font_size=48,
        )
        self.play(Transform(final_loss, new_loss))

        self.marked_next_slide()
        new_loss = MathTex(
            r"\mathcal{L} = \frac{1}{2} \sum_{i,j} W_{ij} \|y_i - y_j\|^2 - \frac{\lambda}{2n} \sum_{i, j} \|y_i - y_j\|^2",
            font_size=48,
        )
        self.play(Transform(final_loss, new_loss))

        self.marked_next_slide()
        new_loss = MathTex(
            r"\mathcal{L} = \frac{1}{2} \sum_{i,j} W_{ij} \|y_i - y_j\|^2 - \frac{\lambda}{2n} \sum_{i, j} \varphi_{\text{rep}}(\|y_i - y_j\|)",
            font_size=48,
        )
        self.play(Transform(final_loss, new_loss))

        self.marked_next_slide()
        new_loss = MathTex(
            r"\mathcal{L} = \frac{1}{2} \sum_{i,j} W_{ij} \varphi_{\text{attr}}(\|y_i - y_j\|) - \frac{\lambda}{2n} \sum_{i, j} \varphi_{\text{rep}}(\|y_i - y_j\|)",
            font_size=48,
        )
        self.play(Transform(final_loss, new_loss))
        self.marked_next_slide()
