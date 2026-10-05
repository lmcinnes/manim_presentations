"""
manim-slides scenes: 3D embeddings of a noisy high-dimensional loop as
isotropic noise grows (fixed noise directions, scaled amplitude).

Frames come from compute_embeddings.py (run it next to this file):
    python curve_embedding_creation.py --umap
    manim-slides render curves.py.py SpectralNoiseSlide SpectralVsUMAPSlide
    manim-slides present SpectralNoiseSlide SpectralVsUMAPSlide

Slide structure (both scenes):
    1. build-in, then a looping spin at the lowest noise level
    2. the noise sweep
    3. a looping spin at the highest noise level
The camera stays fixed; clouds spin about their own vertical axis by exactly
2*pi in the looping slides, so each loop is seamless.
"""

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

apply_defaults()

import os
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
DATA_FILE = Path(os.environ.get("EMBED_FRAMES", _HERE / "embedding_frames.npz"))

SWEEP_SECONDS = 20.0  # duration of the noise ramp
LOOP_SECONDS = 12.0  # one full turn in the looping slides
POINT_SIZE = 3  # PMobject stroke width (pixel size of each point)
HUD_FONT = 34
NUMBER_COLOR = DEFAULT_COLOR  # HIGHLIGHT_COLOR is pale as text on white
PHI = 58 * DEGREES  # fixed camera elevation

# Temporal smoothing of UMAP trajectories (Gaussian, in frames) to suppress
# residual SGD jitter between refits. 0 disables it. Display-only.
UMAP_SMOOTH_FRAMES = 1.5

# Cyclic colour map for the curve parameter t, built from the palette in
# hue order (blue -> purple -> pink -> orange -> green -> cyan -> blue), so
# the loop reads as a closed colour wheel and stays legible on white.
T_CYCLE = [COLOR_CYCLE[i] for i in (0, 4, 3, 1, 2, 5)]


# ---------------------------------------------------------------------------
# Data / geometry helpers
# ---------------------------------------------------------------------------
def cyclic_rgba(t):
    stops = np.array([c.to_rgb() for c in T_CYCLE] + [T_CYCLE[0].to_rgb()])
    u = (np.asarray(t) % (2 * np.pi)) / (2 * np.pi) * len(T_CYCLE)
    i = np.floor(u).astype(int)
    a = (u - i)[:, None]
    rgb = (1 - a) * stops[i] + a * stops[i + 1]
    return np.hstack([rgb, np.ones((len(rgb), 1))])


def smooth_frames(E, sigma_frames):
    if sigma_frames <= 0:
        return E
    from scipy.ndimage import gaussian_filter1d

    return gaussian_filter1d(E, sigma_frames, axis=0, mode="nearest")


def face_up(E):
    """One fixed rotation (same for every frame) putting the first frame's
    least-variance axis along world z, the spin axis, so flat layouts
    (e.g. UMAP's ring) read as open ellipses rather than edge-on."""
    X0 = E[0] - E[0].mean(0)
    _, _, Vt = np.linalg.svd(X0, full_matrices=False)
    R = Vt.T
    if np.linalg.det(R) < 0:
        R[:, 1] *= -1
    return E @ R


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class FrameData:
    """Loads the .npz (into a plain dict: manim deep-copies mobjects and
    their updater closures, and an open NpzFile can't be deep-copied) and
    interpolates per-frame quantities at a fractional frame index."""

    def __init__(self, path=DATA_FILE):
        with np.load(path) as z:
            self.d = {key: z[key] for key in z.files}
        self.F = len(self.d["embeddings"])
        self.D, self.k = int(self.d["ambient_dim"]), int(self.d["k"])
        self.log_sigma = np.log(self.d["sigmas"])

    def interp(self, arr, f):
        i = int(np.clip(np.floor(f), 0, self.F - 2))
        a = float(np.clip(f - i, 0.0, 1.0))
        return (1 - a) * arr[i] + a * arr[i + 1]

    def sigma(self, f):
        # sigma is geometric along frames: interpolate in log space
        return float(np.exp(self.interp(self.log_sigma, f)))


# ---------------------------------------------------------------------------
# Shared slide logic
# ---------------------------------------------------------------------------
class NoiseSweepSlideBase(ThreeDTIMCSlide):
    """Subclasses implement build_panels() and build_overlay()."""

    SPIN_DURING_SWEEP = 0.2  # radians / second

    # -- overlay helpers ---------------------------------------------------
    def readout_rows(self):
        """sigma, sigma relative to d_typ/D^(1/4), noise offset relative to
        the clean kNN distance, and % shortcut edges."""
        data, f, d = self.data, self.frame.get_value, self.data.d
        specs = [
            (
                MathTex(r"\sigma", color=DEFAULT_COLOR, font_size=HUD_FONT),
                3,
                lambda: data.sigma(f()),
                "",
            ),
            (
                MathTex(
                    r"\sigma \,/\, (d_{\mathrm{typ}}/D^{1/4})",
                    color=DEFAULT_COLOR,
                    font_size=HUD_FONT,
                ),
                3,
                lambda: data.sigma(f()) / float(d["sigma_scale"]),
                "",
            ),
            (
                MathTex(
                    r"\sigma\sqrt{D}\,/\,d_{\mathrm{nn}}",
                    color=DEFAULT_COLOR,
                    font_size=HUD_FONT,
                ),
                1,
                lambda: float(data.interp(d["offset_over_dnn"], f())),
                "",
            ),
            (
                Text("shortcut edges", color=DEFAULT_COLOR, font_size=0.72 * HUD_FONT),
                1,
                lambda: 100 * float(data.interp(d["shortcut_frac"], f())),
                r"\%",
            ),
        ]
        rows = []
        for lab, places, getter, suffix in specs:
            eq = MathTex("=", color=DEFAULT_COLOR, font_size=HUD_FONT)
            eq.next_to(lab, RIGHT, buff=0.12)
            num = DecimalNumber(
                getter(),
                num_decimal_places=places,
                font_size=HUD_FONT,
                color=NUMBER_COLOR,
            )
            num.next_to(eq, RIGHT, buff=0.12)
            num.add_updater(lambda m, g=getter: m.set_value(g()))
            # keep the number left-anchored when its width changes
            num.add_updater(lambda m, eq=eq: m.next_to(eq, RIGHT, buff=0.12))
            parts = [lab, eq, num]
            if suffix:
                suf = MathTex(suffix, font_size=HUD_FONT, color=NUMBER_COLOR)
                suf.next_to(num, RIGHT, buff=0.06)
                suf.add_updater(lambda m, n=num: m.next_to(n, RIGHT, buff=0.06))
                parts.append(suf)
            rows.append(VGroup(*parts))
        return rows

    @staticmethod
    def stack_rows(rows, buff):
        """Stack readout rows vertically with their "=" signs aligned."""
        col = VGroup(*rows).arrange(DOWN, buff=buff)
        x_eq = max(r[1].get_x() for r in rows)
        for r in rows:
            r.shift(RIGHT * (x_eq - r[1].get_x()))
        return col

    def footnote(self, one_line=False):
        if one_line:
            return Text(
                "colour = true position along the loop   \u00b7   "
                "shortcut = kNN edge spanning > 5% of the loop",
                color=ACCENT_COLOR,
            ).scale(0.34)
        return (
            VGroup(
                Text("colour = true position along the loop", color=ACCENT_COLOR),
                Text(
                    "shortcut = kNN edge spanning > 5% of the loop", color=ACCENT_COLOR
                ),
            )
            .scale(0.34)
            .arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        )

    def make_cloud(self, E, offset):
        off = np.asarray(offset, dtype=float)
        m = PMobject(stroke_width=POINT_SIZE)
        m.add_points(E[0] + off, rgbas=self.rgba)
        m.add_updater(
            lambda mob: mob.set(
                points=(
                    self.data.interp(E, self.frame.get_value())
                    @ rot_z(self.spin.get_value()).T
                    + off
                )
            )
        )
        return m

    def pin_to_screen(self, mobs):
        """DecimalNumber rebuilds its glyph submobjects on set_value; the
        camera only knows those present when they were registered, so
        re-register every frame (scene updaters run after all mobject
        updaters, so freshly built glyphs are always included)."""
        self.add_fixed_in_frame_mobjects(*mobs)
        self.add_updater(lambda dt: self.camera.add_fixed_in_frame_mobjects(*mobs))

    # -- animation pieces --------------------------------------------------
    def spin_loop(self):
        self.play(
            self.spin.animate.increment_value(TAU),
            run_time=LOOP_SECONDS,
            rate_func=linear,
        )

    def sweep(self):
        self.play(
            self.frame.animate.set_value(self.data.F - 1),
            self.spin.animate.increment_value(self.SPIN_DURING_SWEEP * SWEEP_SECONDS),
            run_time=SWEEP_SECONDS,
            rate_func=linear,
        )

    # -- scene -------------------------------------------------------------
    def construct(self):
        self.data = FrameData()
        self.rgba = cyclic_rgba(self.data.d["t"])
        self.frame = ValueTracker(0.0)
        self.spin = ValueTracker(0.0)
        self.set_camera_orientation(phi=PHI, theta=-90 * DEGREES)

        clouds = self.build_panels()
        overlay = self.build_overlay()
        self.pin_to_screen(overlay)

        self.play(*[FadeIn(m) for m in overlay], run_time=1)
        self.add(*clouds)
        self.play(*[FadeIn(c) for c in clouds], run_time=1)

        self.marked_next_slide(loop=True)
        self.spin_loop()

        self.marked_next_slide()
        self.sweep()

        self.marked_next_slide(loop=True)
        self.spin_loop()

        self.marked_next_slide()


# ---------------------------------------------------------------------------
class SpectralNoiseSlide(NoiseSweepSlideBase):
    """Single spectral embedding, centre-right; readouts on the left."""

    SCALE = 1.9  # unit RMS radius -> scene units
    OFFSET = (1.8, 0.0, -0.2)

    def build_panels(self):
        E = face_up(self.data.d["embeddings"].astype(float)) * self.SCALE
        return [self.make_cloud(E, self.OFFSET)]

    def build_overlay(self):
        title = (
            Text("Spectral embedding of the kNN graph", color=DEFAULT_COLOR)
            .scale(0.6)
            .to_edge(UP, buff=0.35)
        )
        sub = (
            Text(
                f"multi-plane loop in {self.data.D} dimensions, "
                f"k = {self.data.k}, fixed noise directions",
                color=ACCENT_COLOR,
            )
            .scale(0.32)
            .next_to(title, DOWN, buff=0.15)
        )
        hud = self.stack_rows(self.readout_rows(), buff=0.3)
        hud.to_edge(LEFT, buff=0.6).shift(UP * 0.3)
        foot = self.footnote().to_corner(DL, buff=0.35)
        return [title, sub, hud, foot]


class SpectralVsUMAPSlide(NoiseSweepSlideBase):
    """Spectral (left) and UMAP (right) on the same X(sigma)."""

    SCALE = 1.35
    OFFSET_X = 3.5
    OFFSET_Z = 0.05

    def build_panels(self):
        if "umap_embeddings" not in self.data.d:
            raise RuntimeError(
                "no UMAP frames: rerun " "`python compute_embeddings.py --umap`"
            )
        S = face_up(self.data.d["embeddings"].astype(float)) * self.SCALE
        U = (
            face_up(
                smooth_frames(
                    self.data.d["umap_embeddings"].astype(float), UMAP_SMOOTH_FRAMES
                )
            )
            * self.SCALE
        )
        return [
            self.make_cloud(S, (-self.OFFSET_X, 0.0, self.OFFSET_Z)),
            self.make_cloud(U, (+self.OFFSET_X, 0.0, self.OFFSET_Z)),
        ]

    def build_overlay(self):
        title = (
            Text("Same noisy data, two embeddings", color=DEFAULT_COLOR)
            .scale(0.6)
            .to_edge(UP, buff=0.35)
        )
        sub = (
            Text(
                f"multi-plane loop in {self.data.D} dimensions, "
                f"k = {self.data.k} neighbours for both",
                color=ACCENT_COLOR,
            )
            .scale(0.32)
            .next_to(title, DOWN, buff=0.15)
        )
        lab_l = (
            Text("Spectral (kNN Laplacian)", color=DEFAULT_COLOR)
            .scale(0.45)
            .move_to([-self.OFFSET_X, 2.2, 0])
        )
        lab_r = (
            Text("UMAP", color=DEFAULT_COLOR)
            .scale(0.45)
            .move_to([self.OFFSET_X, 2.2, 0])
        )
        rows = self.readout_rows()
        col1 = self.stack_rows(rows[:2], buff=0.18)
        col2 = self.stack_rows(rows[2:], buff=0.18)
        hud = VGroup(col1, col2).arrange(RIGHT, buff=0.9, aligned_edge=UP)
        foot = self.footnote(one_line=True).to_edge(DOWN, buff=0.25)
        hud.next_to(foot, UP, buff=0.3)
        return [title, sub, lab_l, lab_r, hud, foot]
