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


class FDComparisonToUMAP(TIMCSlide):
    def construct(self):
        self.add_centered_text(
            "We can specialise this to approximate UMAP's loss function",
            font_size=56,
        )
        self.wait()
        self.marked_next_slide()
        self.clear_slide()

        fd_loss = MathTex(
            r"\mathcal{L} = \frac{1}{2} \sum_{i,j} W_{ij} \varphi_{\text{attr}}(\|y_i - y_j\|) - \frac{\lambda}{2n} \sum_{i, j} \varphi_{\text{rep}}(\|y_i - y_j\|)",
            font_size=48,
        )
        umapish_loss = MathTex(
            r"\mathcal{L} = \frac{1}{2} \sum_{i,j} W_{ij} \log\left(\frac{1}{1 + a \|y_i - y_j\|^{2b}}\right) + \frac{\lambda}{2n} \sum_{i, j} \log\left(1 - \frac{1}{1 + a \|y_i - y_j\|^{2b}}\right)",
            font_size=36,
        )
        self.play(Write(fd_loss))

        set_forces = Tex(
            r"\[\varphi_{\text{attr}}(r) = \log\left(\frac{1}{1 + a r^{2b}}\right)\]",
            r"\[\varphi_{\text{rep}}(r) = \log\left(1 + \frac{1}{a r^{2b}}\right)\]",
            color=ACCENT_COLOR,
            stroke_color=ACCENT_COLOR,
            font_size=32,
        ).next_to(fd_loss, DOWN, buff=0.5)

        self.play(Write(set_forces))
        self.wait()
        self.marked_next_slide()

        self.play(FadeOut(set_forces), Transform(fd_loss, umapish_loss))
        self.wait()
        self.marked_next_slide()

        set_forces_new = Tex(
            r"\[q_{ij} = \frac{1}{1 + a \|y_i - y_j\|^{2b}}\]",
            r"Assume $W_{ij} \approx 0$ for most $i,j$",
            color=ACCENT_COLOR,
            stroke_color=ACCENT_COLOR,
            font_size=36,
        ).next_to(fd_loss, DOWN, buff=0.5)

        umapish_loss_new = MathTex(
            r"\mathcal{L} = \frac{1}{2} \sum_{i,j} W_{ij} \log\left(q_{ij}\right) + \frac{\lambda}{2n} \sum_{i, j} (1 - W_{ij})\log\left(1 - q_{ij}\right)",
            font_size=36,
        )

        self.play(Write(set_forces_new))
        self.wait()
        self.marked_next_slide()

        self.play(FadeOut(set_forces_new), Transform(fd_loss, umapish_loss_new))
        self.wait()
        self.marked_next_slide()

        set_forces_new2 = Tex(
            r"Let $\lambda = n$\\",
            r"Remove constants that don't affect optimization",
            color=ACCENT_COLOR,
            stroke_color=ACCENT_COLOR,
            font_size=36,
        ).next_to(fd_loss, DOWN, buff=0.5)

        umap_loss = MathTex(
            r"\mathcal{L_{\text{UMAP}}} = \sum_{i,j} W_{ij} \log\left(q_{ij}\right) + \sum_{i, j} (1 - W_{ij})\log\left(1 - q_{ij}\right)",
            font_size=36,
        )

        self.play(Write(set_forces_new2))
        self.wait()
        self.marked_next_slide()

        self.play(FadeOut(set_forces_new2), Transform(fd_loss, umap_loss))
        self.wait()
        self.marked_next_slide()

        self.clear_slide()

        self.add_centered_text(
            "We have a lot more flexibility with attraction and repulsion functions than just UMAP's specific choice",
            max_width=0.66,
            max_height=0.75,
            font_size=48,
        )
        self.wait()
        self.marked_next_slide()
        self.clear_slide()


class EffectiveResistanceEmbeddingExplanation(TIMCSlide):
    def construct(self):
        title = Text("Effective Resistance Embedding", font_size=56)
        self.play(Write(title))
        self.marked_next_slide()
        self.play(FadeOut(title))

        # Show the formula for effective resistance in terms of the pseudoinverse of the Laplacian
        formula = MathTex(
            r"R_{\text{eff}}(i, j) = (e_i - e_j)^\top L^+ (e_i - e_j)",
            font_size=56,
        )
        self.play(Write(formula))
        self.marked_next_slide()

        # Write the Laplacian as an eigendecomposition
        laplacian = MathTex(
            r"L = U \Sigma U^\top",
            font_size=56,
        ).next_to(formula, DOWN, buff=1.0)
        self.play(Write(laplacian))
        self.wait()
        self.marked_next_slide()

        # Substitute the eigendecomposition into the effective resistance formula
        substituted = MathTex(
            r"R_{\text{eff}}(i, j) = (e_i - e_j)^\top U \Sigma^+ U^\top (e_i - e_j)",
            font_size=56,
        ).move_to(formula.get_center())
        self.play(Transform(formula, substituted), FadeOut(laplacian))
        self.marked_next_slide()

        # Define the embedding coordinates as the rows of U \Sigma^{+1/2}
        embedding_def = MathTex(
            r"y_i = \Sigma^{+1/2} U^\top e_i",
            font_size=56,
        ).next_to(formula, DOWN, buff=1.0)
        self.play(Write(embedding_def))
        self.wait()
        self.marked_next_slide()

        # Show that the squared distance between embedding coordinates equals the effective resistance
        distance_formula = MathTex(
            r"\|y_i - y_j\|^2 = (e_i - e_j)^\top U \Sigma^+ U^\top (e_i - e_j) = R_{\text{eff}}(i, j)",
            font_size=56,
        ).move_to(formula.get_center())
        self.play(Transform(formula, distance_formula))
        self.wait()
        self.marked_next_slide()

        self.play(FadeOut(formula), embedding_def.animate.move_to(ORIGIN).scale(1.5))
        self.wait()
        self.marked_next_slide()

        self.play(FadeOut(embedding_def))

        self.add_centered_text(
            "We want the (scaled) eigenvectors corresponding to the smallest nonzero eigenvalues",
            max_width=0.75,
        )
        self.wait()
        self.marked_next_slide()
        self.clear_slide()


import cv2


# --- Custom Video Mobject Implementation ---
class VideoMobject(ImageMobject):
    def __init__(self, filename, **kwargs):
        self.filename = filename

        # 1. Open temporarily to grab dimensions
        cap = cv2.VideoCapture(filename)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps

        # Extract metadata dimensions
        self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Could not read video file: {filename}")

        # CRITICAL FIX 1: Convert first frame to RGBA (4 channels) to initialize the parent correctly
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        super().__init__(frame, **kwargs)

        self.cap = None
        self.current_time = 0.0
        self.current_frame_idx = 0
        self.prev_frame_no = -1

        # Trigger updates on every timeline tick
        self.add_updater(lambda m, dt: m.update_frame(dt))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["cap"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.cap = None

    def __deepcopy__(self, memo):
        import copy

        # Create a clean, uninitialized instance of VideoMobject
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # Copy all properties over, but explicitly leave 'cap' out of it
        for k, v in self.__dict__.items():
            if k == "cap":
                result.cap = None  # The copy will instantiate its own fresh stream when it renders
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def update_frame(self, dt):
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.filename)
            self.current_frame_idx = 0
            self.prev_frame_no = -1

        self.current_time += dt
        frame_no = int(self.current_time * self.fps) % self.frame_count

        # PERFORMANCE FIX: Skip frame processing entirely if the timeline tick
        # hasn't shifted into a brand new video frame yet.
        if frame_no == self.prev_frame_no:
            return

        # PERFORMANCE FIX: Avoid using the expensive set() operation unless
        # the video loops or jumps out of sequential reading order.
        if frame_no != self.current_frame_idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

        ret, frame = self.cap.read()

        if ret:
            self.current_frame_idx = frame_no + 1
            self.prev_frame_no = frame_no

            # CRITICAL FIX 2: Convert streaming frames to RGBA to match Manim's 4-channel matrix specs
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

            # Ensure the array structure precisely matches original specs
            if (
                frame.shape[1] != self.video_width
                or frame.shape[0] != self.video_height
            ):
                frame = cv2.resize(frame, (self.video_width, self.video_height))

            # Safely replace pixel matrix texture data
            if hasattr(self, "set_pixel_array"):
                self.set_pixel_array(frame)
            else:
                self.pixel_array = frame


class HighDExampleUseCases(TIMCSlide):
    def construct(self):

        self.add_centered_text(
            "Activation spaces of deep Neural Networks",
            max_width=0.66,
        )
        self.wait()
        self.marked_next_slide()

        self.clear_slide()

        neural_video = VideoMobject("Neural network geometry.mp4")
        neural_video.scale_to_fit_height(config.frame_height * 0.8)
        # self.play(FadeIn(neural_video))
        self.add(neural_video)
        self.wait(neural_video.duration * 3)
        self.marked_next_slide()

        self.play(FadeOut(neural_video))
        self.add_centered_text(
            "Biology, especially single-cell genomics",
            max_width=0.75,
        )
        self.wait()
        self.marked_next_slide()
        self.clear_slide()
        sc_video = VideoMobject("zebrafish_scrna.mp4")
        sc_video.scale_to_fit_height(config.frame_height * 0.66)
        # self.play(FadeIn(sc_video))
        self.add(sc_video)
        self.wait(sc_video.duration * 2)
        self.marked_next_slide()

        self.play(FadeOut(sc_video))
        self.wait()


class GeneralizedBetaPrimeDistribution(TIMCSlide):
    def construct(self):
        self.add_centered_text("Generalized Beta Prime Distribution", max_width=0.66)
        self.marked_next_slide()
        self.clear_slide()

        self.add_centered_text(
            "A probability distribution of a ratio of two Gamma-distributed variables",
            max_width=0.5,
        )
        self.marked_next_slide()
        self.clear_slide()

        # --- Ratio fraction: Text + Line so Marcellus font is used ---
        RATIO_SIZE = 64
        FRAC_BUFF = 0.12

        dist_label = Text("Distance", font_size=RATIO_SIZE)
        scale_label = Text("Scale", font_size=RATIO_SIZE)
        bar_w_init = max(dist_label.width, scale_label.width) + 0.5
        init_bar = Line(LEFT * bar_w_init / 2, RIGHT * bar_w_init / 2, stroke_width=2)
        VGroup(dist_label, init_bar, scale_label).arrange(DOWN, buff=FRAC_BUFF).move_to(
            ORIGIN
        )

        self.play(FadeIn(dist_label), Create(init_bar), FadeIn(scale_label))
        self.wait()
        self.marked_next_slide()

        # Build the full target layout to get precise final positions
        dist_label_t = Text("Distance", font_size=RATIO_SIZE)
        gamma_num = Text(
            " (Gamma distributed)",
            font_size=RATIO_SIZE,
            color=ACCENT_COLOR,
            stroke_color=ACCENT_COLOR,
        )
        scale_label_t = Text("Scale", font_size=RATIO_SIZE)
        gamma_den = Text(
            " (Gamma distributed)",
            font_size=RATIO_SIZE,
            color=ACCENT_COLOR,
            stroke_color=ACCENT_COLOR,
        )

        num_row = VGroup(dist_label_t, gamma_num).arrange(RIGHT, buff=0.2)
        den_row = VGroup(scale_label_t, gamma_den).arrange(RIGHT, buff=0.2)
        bar_w_final = max(num_row.width, den_row.width) + 0.5
        final_bar = Line(
            LEFT * bar_w_final / 2, RIGHT * bar_w_final / 2, stroke_width=2
        )
        VGroup(num_row, final_bar, den_row).arrange(DOWN, buff=FRAC_BUFF).move_to(
            ORIGIN
        )

        self.play(
            LaggedStart(
                AnimationGroup(
                    ReplacementTransform(dist_label, dist_label_t),
                    ReplacementTransform(scale_label, scale_label_t),
                    Transform(init_bar, final_bar),
                ),
                AnimationGroup(
                    Write(gamma_num),
                    Write(gamma_den),
                ),
                lag_ratio=0.6,
            )
        )
        self.wait()
        self.marked_next_slide()
        self.clear_slide()

        # One colour per parameter, used both in the formula and its label
        C_ALPHA = COLOR_CYCLE[0]  # blue
        C_BETA = COLOR_CYCLE[3]  # pink/red
        C_P = COLOR_CYCLE[2]  # green
        C_Q = COLOR_CYCLE[1]  # orange

        title = Text("Probability Density Function", font_size=48).to_edge(UP, buff=1.0)
        # Every parameter occurrence is its own submobject for colouring and
        # arrow targeting.  Index map:
        #   [1] / [9]  = \alpha  (param list / numerator exponent)    → C_ALPHA
        #   [3] / [16] = \beta   (param list / denominator exponent)  → C_BETA
        #   [5] / [11] = p       (param list / denominator interior)  → C_P
        #   [7] / [13] = q       (param list / denominator exponent)  → C_Q
        # Note: r")^{" + r"q" + r"}" braces the isolated q submobject so that
        # Manim's \special isolation markers don't appear between ^ and its
        # argument, which would cause a LaTeX "Missing {" error.
        formula = MathTex(
            r"\psi(x;\ ",
            r"\alpha",
            r",\ ",
            r"\beta",
            r",\ ",
            r"p",
            r",\ ",
            r"q",
            r") = \frac{1}{Z} \cdot \frac{x^{",
            r"\alpha",
            r" - 1}}{(1 + (x / ",
            r"p",
            r")^{",
            r"q",
            r"}",
            r")^{",
            r"\beta",
            r"}}",
            font_size=56,
        )
        for idx in (1, 9):
            formula[idx].set_color(C_ALPHA)
        for idx in (3, 16):
            formula[idx].set_color(C_BETA)
        for idx in (5, 11):
            formula[idx].set_color(C_P)
        for idx in (7, 13):
            formula[idx].set_color(C_Q)

        self.play(Write(title))
        self.play(Write(formula))
        self.wait()
        self.marked_next_slide()

        # Annotation labels, one per corner for maximum breathing room
        alpha_label = (
            Text(
                "Controls left tail", font_size=26, color=C_ALPHA, stroke_color=C_ALPHA
            )
            .to_corner(UL, buff=2.0)
            .shift(RIGHT * 2)
        )
        q_label = (
            Text("Shape parameter", font_size=26, color=C_Q, stroke_color=C_Q)
            .to_corner(UR, buff=1.75)
            .shift(RIGHT)
        )
        p_label = (
            Text("Scale parameter", font_size=26, color=C_P, stroke_color=C_P)
            .to_corner(DL, buff=2.0)
            .shift(RIGHT * 1.5)
        )
        beta_label = (
            Text("Controls right tail", font_size=26, color=C_BETA, stroke_color=C_BETA)
            .to_corner(DR, buff=0.5)
            .shift(UP * 0.5)
        )

        # Curved arrows from each label to the relevant symbol in the formula body
        alpha_arrow = CurvedArrow(
            alpha_label.get_edge_center(RIGHT) + RIGHT * 0.125,
            formula[9].get_center() + UL * 0.25,
            angle=-PI / 5,
            color=C_ALPHA,
            stroke_width=4,
            tip_length=0.2,
        )
        q_arrow = CurvedArrow(
            q_label.get_center() + DOWN * 0.2,
            formula[13].get_center() + UP * 0.25,
            angle=PI / 5,
            color=C_Q,
            stroke_width=4,
            tip_length=0.2,
        )
        p_arrow = CurvedArrow(
            p_label.get_edge_center(RIGHT) + RIGHT * 0.125,
            formula[11].get_bottom() + DL * 0.1,
            angle=PI / 5,
            color=C_P,
            stroke_width=4,
            tip_length=0.2,
        )
        beta_arrow = CurvedArrow(
            beta_label.get_center() + UL * 0.2,
            formula[16].get_bottom(),
            angle=PI / 5,
            color=C_BETA,
            stroke_width=4,
            tip_length=0.2,
        )

        self.play(
            Write(alpha_label),
            Create(alpha_arrow),
            Write(q_label),
            Create(q_arrow),
            Write(p_label),
            Create(p_arrow),
            Write(beta_label),
            Create(beta_arrow),
        )

        self.wait()
        self.marked_next_slide()

        # Place the new labels at their final positions up front so that
        # ReplacementTransform morphs the text AND moves to the target in one step.
        alpha_label_new = Text(
            "Optional", font_size=26, color=C_ALPHA, stroke_color=C_ALPHA
        ).next_to(alpha_arrow.get_start(), LEFT, buff=0.25)
        q_label_new = Text(
            "Shoulder sharpness", font_size=26, color=C_Q, stroke_color=C_Q
        ).next_to(q_arrow.get_start(), UP, buff=0.1)
        p_label_new = Text("Scale", font_size=26, color=C_P, stroke_color=C_P).next_to(
            p_arrow.get_start(), LEFT, buff=0.25
        )
        beta_label_new = Text(
            "Tail decay", font_size=26, color=C_BETA, stroke_color=C_BETA
        ).next_to(beta_arrow.get_start(), DOWN, buff=0.1)
        x_label_new = Paragraph(
            "Shoulder width\n(via offset)",
            font_size=26,
            color=COLOR_CYCLE[4],
            stroke_color=COLOR_CYCLE[4],
            alignment="center",
        ).move_to(formula[0].get_center() + DOWN * 1.5)
        x_arrow = CurvedArrow(
            x_label_new.get_top() + UP * 0.1,
            formula[0].get_center() + DR * 0.25,
            angle=-PI / 4,
            color=COLOR_CYCLE[4],
            stroke_width=4,
            tip_length=0.2,
        )
        self.play(
            ReplacementTransform(alpha_label, alpha_label_new),
            ReplacementTransform(q_label, q_label_new),
            ReplacementTransform(p_label, p_label_new),
            ReplacementTransform(beta_label, beta_label_new),
            Write(x_label_new),
            Create(x_arrow),
        )
        self.wait()
        self.marked_next_slide()

        self.clear_slide()


class DamrichHDPersistentHomologyPaper(TIMCSlide):

    def construct(self):

        self.add_centered_text(
            "Persistent Homology\nfor\nHigh-dimensional Data\nBased on Spectral Methods",
            font_size=56,
        )
        self.wait()
        authors = Text(
            "Sebastian Damrich, Phillip Berens and Dmitry Kobak",
            font_size=36,
            color=ACCENT_COLOR,
            stroke_color=ACCENT_COLOR,
        )
        venue = Text(
            "NeurIPS 2024",
            font_size=28,
            color=ACCENT_COLOR,
            stroke_color=ACCENT_COLOR,
        )
        citation = (VGroup(authors, venue).arrange(DOWN, buff=0.1)).to_edge(
            DOWN, buff=0.5
        )

        self.play(Write(citation))
        self.wait()
        self.marked_next_slide()

        self.clear_slide()

        poster = ImageMobject("damrich_poster.png").scale_to_fit_height(
            config.frame_height * 0.8
        )
        self.play(FadeIn(poster), run_time=2)
        self.wait()
        self.marked_next_slide()

        self.play(FadeOut(poster))

        main_text = Paragraph(
            "Uses Effective Resistance\nDistance over kNN-graph",
            font_size=56,
            alignment="center",
        )
        sub_text = Text(
            "(with von Luxburg correction)",
            font_size=36,
            color=ACCENT_COLOR,
            stroke_color=ACCENT_COLOR,
        )
        VGroup(main_text, sub_text).arrange(DOWN, buff=0.5).move_to(ORIGIN)
        self.play(Write(main_text))
        self.play(Write(sub_text))
        self.wait()
        self.marked_next_slide()

        self.clear_slide()


class WhatMakesItWork(TIMCSlide):

    def construct(self):

        self.add_centered_text(
            "What makes it work?\n\nWhy do other approaches fail?",
            font_size=56,
        )
        self.wait()
        self.marked_next_slide()
        self.clear_slide()

        text = self.add_centered_text(
            "kNN graphs to focus on local structure?",
            font_size=56,
        )
        self.wait()
        self.marked_next_slide()

        new_text = Text(
            "Path-distance fails!",
            font_size=56,
            color=ACCENT_COLOR,
            stroke_color=ACCENT_COLOR,
        ).shift(DOWN * 0.5)
        self.play(
            LaggedStart(text.animate.shift(UP * 1.5), Write(new_text), lag_ratio=0.5)
        )
        self.wait()
        self.marked_next_slide()
        self.clear_slide()

        self.add_centered_text('In high dimensions "noise" edges are common')
        self.wait()
        self.marked_next_slide()
        self.clear_slide()


class GradientOfLoss(TIMCSlide):
    def construct(self):

        self.add_centered_text(
            "Gradient of the loss function\nis a force field",
            font_size=56,
        )
        self.wait()
        self.marked_next_slide()
        self.clear_slide()

        # Show the formula for the loss function
        loss_formula = MathTex(
            r"\mathcal{L} = \frac{1}{2} \sum_{i,j} W_{ij} \varphi_{\text{attr}}(\|y_i - y_j\|) - \frac{\lambda}{2n} \sum_{i, j} \varphi_{\text{rep}}(\|y_i - y_j\|)",
            font_size=48,
        )
        self.play(Write(loss_formula))
        self.wait()
        self.marked_next_slide()

        # Show the gradient formula
        grad_formula = MathTex(
            r"\nabla_{y_i} \mathcal{L}",
            r" = \sum_j W_{ij} ",
            r"\varphi'_{\text{attr}}(\|y_i - y_j\|)",
            r" \cdot ",
            r"\frac{y_i - y_j}{\|y_i - y_j\|}",
            r" - \frac{\lambda}{n} \sum_{j} ",
            r"\varphi'_{\text{rep}}(\|y_i - y_j\|)",
            r" \cdot ",
            r"\frac{y_i - y_j}{\|y_i - y_j\|}",
            font_size=36,
        ).move_to(loss_formula.get_center())
        self.play(FadeOut(loss_formula, shift=UP), FadeIn(grad_formula, shift=UP))
        self.wait()
        self.marked_next_slide()

        attr = (
            Text(
                "Attraction force strength",
                font_size=28,
                color=COLOR_CYCLE[1],
                stroke_color=COLOR_CYCLE[1],
            )
            .next_to(grad_formula[2], DOWN, buff=1.5)
            .shift(LEFT * 0.5)
        )
        attr_arrow = CurvedArrow(
            attr.get_edge_center(UP) + UP * 0.1,
            grad_formula[2].get_edge_center(DOWN) + DOWN * 0.1,
            color=COLOR_CYCLE[1],
            stroke_width=4,
            angle=PI / 5,
            tip_length=0.15,
        )
        repr = (
            Text(
                "Repulsion force strength",
                font_size=28,
                color=COLOR_CYCLE[2],
                stroke_color=COLOR_CYCLE[2],
            )
            .next_to(grad_formula[6], UP, buff=1.5)
            .shift(RIGHT * 0.5)
        )
        repr_arrow = CurvedArrow(
            repr.get_edge_center(DOWN) + DOWN * 0.1,
            grad_formula[6].get_edge_center(UP) + UP * 0.1,
            color=COLOR_CYCLE[2],
            stroke_width=4,
            angle=PI / 5,
            tip_length=0.15,
        )

        self.play(
            grad_formula[2].animate.set_color(COLOR_CYCLE[1]).set_stroke_width(8),
            grad_formula[6].animate.set_color(COLOR_CYCLE[2]).set_stroke_width(8),
            FadeIn(attr, scale=0.25),
            FadeIn(repr, scale=0.25),
            Create(attr_arrow),
            Create(repr_arrow),
        )
        self.wait()

        vec = Text(
            "Direction of force",
            font_size=28,
            color=COLOR_CYCLE[3],
            stroke_color=COLOR_CYCLE[3],
        ).move_to(
            (grad_formula[4].get_center() + grad_formula[8].get_center()) / 2
            + DOWN * 2.5
        )
        vec_arrow1 = CurvedArrow(
            vec.get_edge_center(LEFT) + LEFT * 0.1,
            grad_formula[4].get_edge_center(DOWN) + DOWN * 0.25,
            color=COLOR_CYCLE[3],
            stroke_width=4,
            angle=-PI / 5,
            tip_length=0.15,
        )
        vec_arrow2 = CurvedArrow(
            vec.get_edge_center(RIGHT) + RIGHT * 0.1,
            grad_formula[8].get_edge_center(DOWN) + DOWN * 0.25,
            color=COLOR_CYCLE[3],
            stroke_width=4,
            angle=PI / 5,
            tip_length=0.15,
        )

        self.play(
            grad_formula[4].animate.set_color(COLOR_CYCLE[3]).set_stroke_width(8),
            grad_formula[8].animate.set_color(COLOR_CYCLE[3]).set_stroke_width(8),
            FadeIn(vec, scale=0.25),
            Create(vec_arrow1),
            Create(vec_arrow2),
        )
        self.wait()
        self.marked_next_slide()
        self.clear_slide()

        requirements = (
            VGroup(
                Tex(
                    r"$\varphi'_{\text{attr}}(r) \to 0$ as $r \to \infty$", font_size=48
                ),
                Tex(
                    r"$\varphi'_{\text{rep}}(r) \to 0$ as $r \to \infty$", font_size=48
                ),
                Tex(
                    r"$|\varphi'_{\text{rep}}(r)| > |\varphi'_{\text{attr}}(r)|$ for large $r$",
                    font_size=48,
                ),
            )
            .arrange(DOWN, buff=0.5)
            .move_to(ORIGIN)
        )

        self.play(Write(requirements))

        self.wait()
        self.marked_next_slide()
        self.clear_slide()

        self.add_centered_text(
            "The relative shapes of the derivatives of the attraction and repulsion functions control the geometry of the embedding",
            max_width=0.75,
            max_height=0.66,
            font_size=56,
            t2c={"relative shapes": ACCENT_COLOR},
        )

        self.wait()
        self.marked_next_slide()
        self.clear_slide()
