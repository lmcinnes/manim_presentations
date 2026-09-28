"""Shared helpers for the UMAP 0.6 explainer slides.

Import after the usual config header in each scene module::

    from config import apply_defaults, ...
    apply_defaults()
    from umap_talk_common import *
    apply_umap_defaults()

Everything that may end up inside a pickled slide state (``save_state``) is
defined here rather than in a scene file, so states load from any scene.
"""

import colorsys
import itertools as _it
from functools import lru_cache

import numpy as np
from manim import *
from manim.camera.camera import Camera
from manim.utils.color import color_to_rgb, color_to_rgba

from config import (
    ACCENT_COLOR,
    COLOR_CYCLE,
    DEFAULT_COLOR,
    DEFAULT_TEX_TEMPLATE,
    HIGHLIGHT_COLOR,
    TIMCSlide,
    add_logo_to_background,
    create_styled_axes,
)

# Section titles, shared so each class can hand off to the next with a
# split section wipe (start_section_wipe at the end of one scene,
# end_section_wipe at the start of the next).
SECTION_TITLES = {
    "optimizers": "New Optimizers",
    "negatives": "Hard Negatives",
    "recursive": "Recursive Initialization",
    "payoff": "The Repulsion Dial",
    "future": "Where Next",
    "repair": "Repairing the Graph",
    "kernel": "The Kernel as Dials",
}

MONO_FONT = "DejaVu Sans Mono"  # e.g. "Menlo" on macOS

# ===========================================================================
# Crisp text
#
# Pango lays text out at the requested size, and at small sizes it rounds each
# glyph's advance, which gives Marcellus uneven letter spacing (Paragraph
# suffers the same way, since it is built from the same Text).  Laying text out
# large and scaling it down keeps the spacing even.  Use crisp_text() for all
# on-screen text.
# ===========================================================================
TEXT_RENDER_SIZE = 96


def crisp_text(text, font_size=DEFAULT_FONT_SIZE, **kwargs):
    """Text laid out at TEXT_RENDER_SIZE (or larger) and scaled to font_size."""
    render_size = max(float(font_size), TEXT_RENDER_SIZE)
    mob = Text(text, font_size=render_size, **kwargs)
    if render_size != font_size:
        mob.scale(font_size / render_size)
    return mob


def crisp_lines(lines, font_size=DEFAULT_FONT_SIZE, buff=0.1, aligned_edge=LEFT, **kwargs):
    """One crisp_text per line, stacked; never a single multi-line Text/Paragraph."""
    group = VGroup(*[crisp_text(line, font_size, **kwargs) for line in lines])
    return group.arrange(DOWN, aligned_edge=aligned_edge, buff=buff)


def styled_axes(x_range, y_range, x_label, y_label, x_tick_labels=None, tick_font_size=16,
                x_tick_label_buff=0.25, **kwargs):
    """create_styled_axes, with categorical x tick labels drawn as crisp text.

    create_styled_axes draws x_tick_labels with small Paragraphs; this draws
    them with crisp_text in the same positions ("+" still breaks a line).
    Returns the same VGroup(axes, x_label, y_label, tick_labels).
    """
    group = create_styled_axes(x_range, y_range, x_label, y_label, **kwargs)
    axes, old_x, old_y, ticks = group
    # create_styled_axes draws its titles at 48 pt scaled to 0.5 / 0.45; redraw crisply.
    x_label_mob = crisp_text(x_label, 24).move_to(old_x)
    y_label_mob = crisp_text(y_label, 21.6).rotate(PI / 2).move_to(old_y)
    group = VGroup(axes, x_label_mob, y_label_mob, ticks)
    if x_tick_labels:
        axis = axes.get_x_axis()
        for i, label in enumerate(x_tick_labels, 1):
            tick = crisp_lines(label.split("+"), tick_font_size, buff=0.05, aligned_edge=ORIGIN)
            x = axes.c2p(i, y_range[0])[0]
            tick.move_to([x, axis.get_top()[1], 0])
            tick.shift(DOWN * (tick.height / 2 + axis.tick_size + x_tick_label_buff))
            ticks.add(tick)
        x_label_mob.next_to(ticks, DOWN, buff=0.2)
    return group


# ===========================================================================
# Palette roles
# ===========================================================================
ATTRACT_COLOR = COLOR_CYCLE[0]  # blue: attractive forces / graph edges pulling
REPEL_COLOR = COLOR_CYCLE[3]  # rose: repulsive forces / negative samples
SOURCE_COLOR = HIGHLIGHT_COLOR  # the point being updated (use a navy outline)
STRUCTURE_COLOR = DEFAULT_COLOR  # nodes, edges, slabs, axes
SECONDARY_COLOR = ACCENT_COLOR  # de-emphasised structure
OLD_STACK_COLOR = COLOR_CYCLE[6]  # grey-olive: compatibility / legacy results
NEW_STACK_COLOR = DEFAULT_COLOR  # navy: new optimizer stack
MARKER_COLOR = COLOR_CYCLE[2]  # reserved: pause marker in the DR corner

UMAP_A, UMAP_B = 1.5769434602697652, 0.8950608778515733  # min_dist=0.1, spread=1


def contrast_vs_background(color, background=None):
    """WCAG contrast ratio of ``color`` against the background colour."""

    def luminance(rgb):
        rgb = np.where(rgb <= 0.03928, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]

    bg = config.background_color if background is None else background
    l1 = luminance(np.asarray(color_to_rgb(color)))
    l2 = luminance(np.asarray(color_to_rgb(bg)))
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def darken_for_contrast(color, min_contrast=3.0, step=0.02):
    """Keep hue and saturation, lower lightness until contrast >= min_contrast."""
    r, g, b = color_to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    out = ManimColor.from_rgb((r, g, b))
    while contrast_vs_background(out) < min_contrast and l > 0.05:
        l -= step
        out = ManimColor.from_rgb(colorsys.hls_to_rgb(h, l, s))
    return out


def digit_rgb(min_contrast=None):
    """(10, 3) RGB array for the ten MNIST digits, from COLOR_CYCLE.

    With ``min_contrast`` set, the low-contrast entries (orange, green, cyan,
    lavender, pink) are darkened for small-point legibility on white.
    """
    colors = COLOR_CYCLE[:10]
    if min_contrast is not None:
        colors = [darken_for_contrast(c, min_contrast) for c in colors]
    return np.array([color_to_rgb(c) for c in colors], dtype=np.float64)


def labels_to_rgb(labels, palette_rgb):
    """Per-point RGB array from integer labels and a (k, 3) palette."""
    palette_rgb = np.asarray(palette_rgb, dtype=np.float64)
    return palette_rgb[np.asarray(labels) % len(palette_rgb)]


def background_rgb():
    return np.asarray(color_to_rgba(config.background_color)[:3], dtype=np.float64)


# ===========================================================================
# UMAP force curves (for plots and toy-graph arrows)
# ===========================================================================
def attraction_magnitude(d, a=UMAP_A, b=UMAP_B):
    d = np.asarray(d, dtype=np.float64)
    return 2 * a * b * d ** (2 * b - 1) / (1 + a * d ** (2 * b))


def repulsion_magnitude(d, a=UMAP_A, b=UMAP_B, gamma=1.0, eps=1e-3):
    d = np.asarray(d, dtype=np.float64)
    return 2 * gamma * b * d / ((eps + d * d) * (1 + a * d ** (2 * b)))


def soft_clip(magnitude, gamma=1.0):
    """The new kernels' repulsion clip: gamma * tanh(|F| / gamma)."""
    return gamma * np.tanh(np.asarray(magnitude, dtype=np.float64) / gamma)


def slab_probability(r, w):
    """P(point at distance r lies in a random slab of width w through the source)."""
    r = np.maximum(np.asarray(r, dtype=np.float64), 1e-12)
    return (2 / np.pi) * np.arcsin(np.minimum(1.0, w / (2 * r)))


# ===========================================================================
# Point-cloud rendering (Cairo)
# ===========================================================================
_REF_PIXEL_HEIGHT = 1440  # resolution that point sizes are tuned for


def res_px(px_at_ref: float) -> float:
    """Scale a pixel size tuned at _REF_PIXEL_HEIGHT to the current resolution."""
    return px_at_ref * config.pixel_height / _REF_PIXEL_HEIGHT


_res_px = res_px  # backwards-compatible alias


@lru_cache(maxsize=None)
def _circular_nudges(thickness: int):
    """Pixel-nudge offsets forming a filled disc (round PMobject points)."""
    thickness = int(thickness)
    _range = range(-thickness // 2 + 1, thickness // 2 + 1)
    r2 = (thickness / 2.0) ** 2
    nudges = [
        (dy, dx) for dy, dx in _it.product(_range, _range) if dx * dx + dy * dy <= r2
    ]
    return np.array(nudges) if nudges else np.array([[0, 0]])


def _display_point_cloud_point_major(
    self, pmobject, points, rgbas, thickness, pixel_array
):
    """Drop-in replacement for Camera.display_point_cloud.

    Stock Cairo writes nudge-major (every point's first pixel, then every
    point's second pixel, ...), so where discs overlap the winner is whichever
    nudge came last, not whichever point came last.  Writing point-major gives
    true painter's order: later points fully cover earlier ones, which is what
    makes EmbeddingCloud.bring_to_front() work.
    """
    if len(points) == 0:
        return
    pixel_coords = self.points_to_pixel_coords(pmobject, points)
    nudges = self.get_thickening_nudges(thickness)
    n_nudges = len(nudges)
    pixel_coords = (pixel_coords[:, None, :] + nudges[None, :, :]).reshape(-1, 2)

    rgba_len = pixel_array.shape[2]
    rgbas = (self.rgb_max_val * rgbas).astype(self.pixel_array_dtype)
    rgbas = np.repeat(rgbas, n_nudges, axis=0)

    on_screen = self.on_screen_pixels(pixel_coords)
    pixel_coords = pixel_coords[on_screen]
    rgbas = rgbas[on_screen]

    ph, pw = self.pixel_height, self.pixel_width
    indices = (pixel_coords[:, 0] + pw * pixel_coords[:, 1]).astype("int")
    flat = pixel_array.reshape((ph * pw, rgba_len))
    flat[indices] = rgbas
    pixel_array[:, :] = flat.reshape((ph, pw, rgba_len))


_ORIGINAL_NUDGES = Camera.get_thickening_nudges
_ORIGINAL_DISPLAY_POINT_CLOUD = Camera.display_point_cloud


def install_point_cloud_patches(round_points=True, point_major=True):
    """Patch the Cairo camera (2D and 3D) for round, correctly layered points."""
    if round_points:
        Camera.get_thickening_nudges = lambda self, t: _circular_nudges(int(t))
    else:
        Camera.get_thickening_nudges = _ORIGINAL_NUDGES
    if point_major:
        Camera.display_point_cloud = _display_point_cloud_point_major
    else:
        Camera.display_point_cloud = _ORIGINAL_DISPLAY_POINT_CLOUD


_DEFAULTS_APPLIED = False


def apply_umap_defaults(round_points=True, point_major=True):
    """Call once per scene module, after config.apply_defaults()."""
    global _DEFAULTS_APPLIED
    if not _DEFAULTS_APPLIED:
        # Operator names (sin, arcsin) and \text{} pick up Marcellus; math
        # digits and letters stay Computer Modern.
        MathTex.set_default(tex_template=DEFAULT_TEX_TEMPLATE)
        _DEFAULTS_APPLIED = True
    install_point_cloud_patches(round_points, point_major)


# ===========================================================================
# Colour animations for PMobjects
#
# Cairo writes point colours straight into the pixel array with no alpha
# compositing, so opacity changes are invisible.  Instead every "fade" blends
# RGB towards the background with alpha held at 1.  All of these leave the
# mobject's updaters running (suspend_mobject_updating=False), so a cloud can
# morph and recolour in the same play() call.
# ===========================================================================
class PMRecolor(Animation):
    """Blend a PMobject's point colours from their current values to ``target_rgb``.

    ``target_rgb`` is a single colour, an (3,) array, or an (N, 3) array.
    """

    def __init__(self, mob: PMobject, target_rgb, **kwargs):
        target = target_rgb
        if not isinstance(target, np.ndarray):
            target = np.asarray(color_to_rgb(target), dtype=np.float64)
        self._target_rgb = np.broadcast_to(
            np.asarray(target, dtype=np.float64), (len(mob.rgbas), 3)
        ).copy()
        kwargs.setdefault("suspend_mobject_updating", False)
        super().__init__(mob, **kwargs)

    def create_starting_mobject(self):
        return Mobject()  # unused; avoids deep-copying large clouds

    def begin(self):
        self._start_rgb = self.mobject.rgbas[:, :3].copy()
        self.mobject.rgbas[:, 3] = 1.0
        super().begin()

    def interpolate_mobject(self, alpha: float) -> None:
        self.mobject.rgbas[:, :3] = (
            1.0 - alpha
        ) * self._start_rgb + alpha * self._target_rgb


class PMFadeOut(PMRecolor):
    """Fade a PMobject into the background, then remove it.

    Colours are restored after removal, so the mobject can be re-added later.
    """

    def __init__(self, mob: PMobject, **kwargs):
        kwargs.setdefault("remover", True)
        super().__init__(mob, background_rgb(), **kwargs)

    def clean_up_from_scene(self, scene):
        super().clean_up_from_scene(scene)
        self.mobject.rgbas[:, :3] = self._start_rgb


class PMFadeIn(PMRecolor):
    """Fade a PMobject in from the background.

    Add the mobject to the scene before playing (as with FadeIn); it is set
    to the background colour immediately so it does not pop in first.
    """

    def __init__(self, mob: PMobject, **kwargs):
        target = mob.rgbas[:, :3].copy()
        mob.rgbas[:, :3] = background_rgb()
        super().__init__(mob, target, **kwargs)


class PMDim(PMRecolor):
    """Push a PMobject's colours ``amount`` of the way towards the background.

    Measured from the cloud's base colours when it has them (EmbeddingCloud),
    so repeated dims do not compound.
    """

    def __init__(self, mob: PMobject, amount=0.8, **kwargs):
        base = getattr(mob, "base_rgb", mob.rgbas[:, :3].copy())
        target = (1.0 - amount) * base + amount * background_rgb()
        super().__init__(mob, target, **kwargs)


class PMUndim(PMRecolor):
    """Return an EmbeddingCloud to its base colours."""

    def __init__(self, mob, **kwargs):
        super().__init__(mob, mob.base_rgb, **kwargs)


# ===========================================================================
# EmbeddingCloud: keyframed 2D point cloud
# ===========================================================================
def _as_point3(p):
    """Accept (x, y) or (x, y, z) and return a 3D point."""
    p = np.asarray(p, dtype=np.float64).ravel()
    return np.r_[p, np.zeros(3 - len(p))] if len(p) < 3 else p[:3].copy()


class EmbeddingCloud(PMobject):
    """A PMobject that interpolates between precomputed 2D layouts.

    Parameters
    ----------
    keyframes : array (K, N, 2) or (N, 2)
        Layouts in data coordinates, one per keyframe.  Align them offline
        (Procrustes) so morphs read as structural change, not rotation.
    colors : array (N, 3) of RGB in [0, 1], or a single colour.
    width, height, center : placement box in scene units.
    fit : "union" fits the union bounding box of all keyframes into the box
        (use when scale is meaningful across keyframes); "each" fits every
        keyframe into the box independently (use when it is not, e.g.
        init vs final layouts); "none" takes keyframes as scene coordinates
        (offsets from ``center``, which defaults to the origin).
    quantile : bounding boxes use [q, 1-q] quantiles, so a few outliers do
        not shrink the whole cloud.
    point_px : point diameter in pixels at 1440p; rescaled via res_px().
    shuffle_seed : fixed random draw order so no class sits on top; None
        keeps input order.

    Indices taken by methods such as ``recolor`` and ``bring_to_front`` always
    refer to the original input order.
    """

    def __init__(
        self,
        keyframes,
        colors=DEFAULT_COLOR,
        width=6.0,
        height=6.0,
        center=ORIGIN,
        fit="union",
        quantile=0.002,
        point_px=4.0,
        shuffle_seed=0,
        **kwargs,
    ):
        kf = np.asarray(keyframes, dtype=np.float64)
        if kf.ndim == 2:
            kf = kf[None]
        if kf.ndim != 3 or kf.shape[2] != 2:
            raise ValueError("keyframes must have shape (K, N, 2) or (N, 2)")
        n_frames, n_points, _ = kf.shape

        if isinstance(colors, np.ndarray) and colors.ndim == 2:
            rgb = colors[:, :3].astype(np.float64)
        else:
            rgb = np.tile(np.asarray(color_to_rgb(colors)), (n_points, 1))
        if len(rgb) != n_points:
            raise ValueError("colors must have one row per point")

        if shuffle_seed is None:
            order = np.arange(n_points)
        else:
            order = np.random.default_rng(shuffle_seed).permutation(n_points)

        self._order = order  # internal position -> original index
        self._rank = np.empty(n_points, dtype=np.int64)
        self._rank[order] = np.arange(n_points)
        self._box = np.array([width, height], dtype=np.float64)
        self._v = self._normalise(kf, fit, quantile)[:, order, :]
        self.base_rgb = rgb[order].copy()
        self.n_frames = n_frames
        self.frame_value = 0.0
        self.placement_center = _as_point3(center)
        self.placement_zoom = 1.0

        super().__init__(stroke_width=res_px(point_px), **kwargs)
        self.add_points(
            np.zeros((n_points, 3)),
            rgbas=np.c_[self.base_rgb, np.ones(n_points)],
        )
        self.set_frame(0.0)

    # -- layout -----------------------------------------------------------
    def _normalise(self, kf, fit, q):
        """Map each keyframe to scene-unit offsets from the box centre."""
        lo = np.quantile(kf, q, axis=1)  # (K, 2)
        hi = np.quantile(kf, 1.0 - q, axis=1)
        if fit == "union":
            lo = np.broadcast_to(lo.min(axis=0), lo.shape)
            hi = np.broadcast_to(hi.max(axis=0), hi.shape)
        elif fit == "none":
            return kf.copy()  # keyframes already in scene coordinates
        elif fit != "each":
            raise ValueError("fit must be 'union', 'each' or 'none'")
        mid = 0.5 * (lo + hi)
        span = np.maximum(hi - lo, 1e-12)
        scale = np.min(self._box[None, :] / span, axis=1)  # (K,)
        return ((kf - mid[:, None, :]) * scale[:, None, None]).astype(np.float64)

    def set_frame(self, t):
        """Place points at keyframe position t (fractional t interpolates)."""
        t = float(np.clip(t, 0, self.n_frames - 1))
        i = int(np.floor(t))
        j = min(i + 1, self.n_frames - 1)
        f = t - i
        v = self._v[i] if f == 0.0 else (1.0 - f) * self._v[i] + f * self._v[j]
        self.points[:, :2] = self.placement_center[:2] + self.placement_zoom * v
        self.points[:, 2] = self.placement_center[2]
        self.frame_value = t
        return self

    def track(self, tracker: ValueTracker):
        """Follow a ValueTracker holding the (fractional) keyframe index."""
        self.add_updater(lambda m: m.set_frame(tracker.get_value()))
        return self

    def place(self, center=None, zoom=None):
        if center is not None:
            self.placement_center = _as_point3(center)
        if zoom is not None:
            self.placement_zoom = float(zoom)
        return self.set_frame(self.frame_value)

    def scene_positions(self, original_idx=None):
        """Current scene coordinates, optionally for given original indices."""
        if original_idx is None:
            out = np.empty_like(self.points)
            out[self._order] = self.points
            return out
        return self.points[self._rank[np.asarray(original_idx)]]

    # -- colours and layering ---------------------------------------------
    def recolor(self, original_idx, rgb, update_base=False):
        """Set colours for a subset (instantly; animate with PMRecolor)."""
        idx = self._rank[np.asarray(original_idx)]
        rgb = np.asarray(color_to_rgb(rgb) if not isinstance(rgb, np.ndarray) else rgb)
        self.rgbas[idx, :3] = rgb
        if update_base:
            self.base_rgb[idx] = rgb
        return self

    def reset_colors(self):
        self.rgbas[:, :3] = self.base_rgb
        return self

    def target_rgb(self, original_idx, rgb, dim_rest=None):
        """Colour array for PMRecolor: subset -> rgb, rest -> base (or dimmed)."""
        out = self.base_rgb.copy()
        if dim_rest is not None:
            out = (1.0 - dim_rest) * out + dim_rest * background_rgb()
        rgb = rgb if isinstance(rgb, np.ndarray) else np.asarray(color_to_rgb(rgb))
        out[self._rank[np.asarray(original_idx)]] = rgb
        return out

    def bring_to_front(self, original_idx):
        """Draw the given points last (on top).  Needs point_major patches.

        Reorders the cloud internally, so build any PMRecolor targets (e.g.
        with target_rgb) after calling this, not before.
        """
        chosen = np.zeros(len(self._order), dtype=bool)
        chosen[self._rank[np.asarray(original_idx)]] = True
        perm = np.r_[np.flatnonzero(~chosen), np.flatnonzero(chosen)]
        self._order = self._order[perm]
        self._rank[self._order] = np.arange(len(self._order))
        self._v = self._v[:, perm, :]
        self.base_rgb = self.base_rgb[perm]
        self.rgbas = self.rgbas[perm]
        self.points = self.points[perm]
        return self


class PlaceCloud(Animation):
    """Animate an EmbeddingCloud's placement (centre and/or zoom)."""

    def __init__(self, cloud: EmbeddingCloud, center=None, zoom=None, **kwargs):
        self._end_center = None if center is None else _as_point3(center)
        self._end_zoom = zoom
        kwargs.setdefault("suspend_mobject_updating", False)
        super().__init__(cloud, **kwargs)

    def create_starting_mobject(self):
        return Mobject()

    def begin(self):
        self._c0 = self.mobject.placement_center.copy()
        self._z0 = self.mobject.placement_zoom
        super().begin()

    def interpolate_mobject(self, alpha):
        c = self._c0 if self._end_center is None else (
            (1 - alpha) * self._c0 + alpha * self._end_center
        )
        z = self._z0 if self._end_zoom is None else (
            (1 - alpha) * self._z0 + alpha * self._end_zoom
        )
        self.mobject.place(center=c, zoom=z)


# ===========================================================================
# Readouts
# ===========================================================================
class MetricReadout(VGroup):
    """``label  0.973``: a crisp text label with a standard manim DecimalNumber.

    Update with ``readout.number.set_value(v)`` (for example in an updater: the
    DecimalNumber keeps its left edge fixed and redraws cleanly), or animate
    with ``readout.animate_to(v)``. ``unit`` is appended as TeX, e.g. r"\%".
    """

    def __init__(
        self,
        label,
        value=0.0,
        num_decimal_places=3,
        font_size=28,
        color=DEFAULT_COLOR,
        number_color=None,
        buff=0.2,
        unit=None,
    ):
        self.label = crisp_text(label, font_size, color=color)
        # TeX digits come out smaller than Marcellus at the same nominal size:
        # size the number so a digit is as tall as the label's digits.
        scale = crisp_text("0", font_size).height / DecimalNumber(0, num_decimal_places=0, font_size=font_size).height
        self.number = DecimalNumber(value, num_decimal_places=num_decimal_places, font_size=font_size * scale,
                                    color=number_color or color, group_with_commas=True, unit=unit)
        self.number.next_to(self.label, RIGHT, buff=buff)
        super().__init__(self.label, self.number)

    def animate_to(self, value, **kwargs):
        return ChangeDecimalToValue(self.number, value, **kwargs)


# ===========================================================================
# Slide base class for the talk
# ===========================================================================
class UMAPSlide(TIMCSlide):
    """TIMCSlide with point-cloud-aware clearing, titles and loop helpers."""

    def _build_section_slide(self, section_name):
        """TIMCSlide's section card, with the title in crisp text lines
        instead of a Paragraph (same background, font, size and colour)."""
        background = Rectangle(width=config.frame_width, height=config.frame_height,
                               fill_opacity=1, stroke_width=0)
        background.set_sheen_direction(UP)
        background.set_fill(color=[DEFAULT_COLOR, ACCENT_COLOR, WHITE], opacity=1)
        title = crisp_lines(section_name.split("\n"), 72, buff=0.25, aligned_edge=ORIGIN,
                            color=WHITE, font="Marcellus SC")
        return VGroup(background, title)

    # -- section wipes that accept next_slide options (notes, auto_next) ----
    def start_section_wipe(self, section_name, **slide_kwargs):
        """As TIMCSlide.start_section_wipe; kwargs go to the marked pause.

        Pass auto_next=True when this ends a scene, so the dropped card flows
        straight into the next scene's end_section_wipe (one keypress, not two).
        """
        self.marked_next_slide(**slide_kwargs)
        section_slide = self._build_section_slide(section_name)
        section_slide.move_to(UP * config.frame_height)
        self.play(section_slide.animate.shift(DOWN * config.frame_height), run_time=1)
        self.wait(0.1)

    def end_section_wipe(self, section_name, next_slide_prep=None, **slide_kwargs):
        """As TIMCSlide.end_section_wipe; kwargs (e.g. notes) go to the slide
        that reveals the new content."""
        section_slide = self._build_section_slide(section_name)
        self.add(section_slide)
        self.wait(0.1)
        self.next_slide(**slide_kwargs)
        self.clear()
        add_logo_to_background(self)
        if next_slide_prep is not None:
            next_slide_prep()
        self._title = None
        self.play(section_slide.animate.shift(UP * config.frame_height), run_time=1)
        self.remove(section_slide)

    def set_title(self, text, font_size=46, max_width=10.0):
        """Replace the current stage title (top centre) with ``text``.

        Built directly rather than via add_title_text, whose wrapping width is
        computed in pixels and so changes with render resolution.
        """
        new = crisp_text(text, font_size)
        if new.width > max_width:
            new.scale_to_fit_width(max_width)
        new.to_edge(UP, buff=0.4)
        old = getattr(self, "_title", None)
        if old is not None and old in self.mobjects:
            # Finish removing the old title before writing into the same spot.
            self.play(FadeOut(old), run_time=0.5)
        self.play(Write(new))
        self._title = new
        return new

    def takeaway(self, *lines, font_size=56, buff=0.3):
        """Centred takeaway card with explicit line breaks (no pixel-based wrap)."""
        group = crisp_lines(lines, font_size, buff=buff, aligned_edge=ORIGIN).move_to(ORIGIN)
        self.play(Write(group))
        return group

    def swap(self, old, new_anim, run_time=0.5):
        """Fade ``old`` out completely, then play ``new_anim``.

        Use whenever new text or graphics go where something else just was.
        """
        if old is not None:
            self.play(FadeOut(old), run_time=run_time)
        self.play(new_anim)

    def clear_slide(self, animation=FadeOut, run_time=1):
        anims, cloud_anims = [], []
        for mob in self.mobjects:
            if mob is self.logo:
                continue
            if isinstance(mob, PMobject):
                cloud_anims.append(PMFadeOut(mob))
                continue
            anims.append(animation(mob))
            # Clouds nested in groups: recolour after the group fade so the
            # colour blend is the last write each frame.
            cloud_anims += [
                PMRecolor(m, background_rgb())
                for m in mob.get_family()[1:]
                if isinstance(m, PMobject)
            ]
        if anims or cloud_anims:
            self.play(*anims, *cloud_anims, run_time=run_time)

    def start_loop(self, **kwargs):
        """Marked pause, then begin a looping slide.

        The pause marker stays on screen for the whole loop.  Everything
        played until end_loop() repeats, so it should end in the state it
        started in.
        """
        self.marked_next_slide(loop=True, **kwargs)
        self._loop_marker = Dot(radius=0.1, color=MARKER_COLOR).to_corner(
            DR, buff=0.2
        )
        self._add_overlay_mobject(self._loop_marker)

    def end_loop(self, **kwargs):
        """Close a looping slide with a plain boundary (no marker flash, no wait)."""
        self.next_slide(**kwargs)
        self.remove(self._loop_marker)
