from manim import *
from manim.utils.color.core import color_to_int_rgb
from manim_slides import Slide, ThreeDSlide
from functools import lru_cache
import json
import os
import pickle
from textwrap import wrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import mixbox

# Your color palette
DEFAULT_COLOR = ManimColor("#2c3e63")
ACCENT_COLOR = ManimColor("#7088b8")
HIGHLIGHT_COLOR = ManimColor("#fcaf3e")
BACKGROUND_COLOR = ManimColor("#FFFFFF")
COLOR_CYCLE = [
    ManimColor("#597ec9"),
    ManimColor("#fcaf3e"),
    ManimColor("#7ec959"),
    ManimColor("#c9597e"),
    ManimColor("#7e59c9"),
    ManimColor("#00d7e3"),
    ManimColor("#6d7561"),
    ManimColor("#c2badf"),
    ManimColor("#b66100"),
    ManimColor("#fba2a6"),
]

# ============================================================================
# TeX DEFAULTS
# ============================================================================
DEFAULT_TEX_TEMPLATE = TexTemplate()
DEFAULT_TEX_TEMPLATE.tex_compiler = "xelatex"
DEFAULT_TEX_TEMPLATE.output_format = ".xdv"
DEFAULT_TEX_TEMPLATE.add_to_preamble(r"\usepackage{fontspec}")
DEFAULT_TEX_TEMPLATE.add_to_preamble(r"\setmainfont{Marcellus}")

HEADING_FONT = "Marcellus SC"
HEADING_TEX_TEMPLATE = TexTemplate()
HEADING_TEX_TEMPLATE.tex_compiler = "xelatex"
HEADING_TEX_TEMPLATE.output_format = ".xdv"
HEADING_TEX_TEMPLATE.add_to_preamble(r"\usepackage{fontspec}")
HEADING_TEX_TEMPLATE.add_to_preamble(r"\setmainfont{" + HEADING_FONT + "}")

# ============================================================================
# AXES DEFAULTS
# ============================================================================
DEFAULT_AXES_CONFIG = {
    "axis_config": {
        "include_tip": False,
        "color": DEFAULT_COLOR,
    },
    "x_length": 9,
    "y_length": 6,
}

# ============================================================================
# FONT/TEXT SETTINGS
# ============================================================================
DEFAULT_FONT = "Marcellus"  # or "Helvetica", "Times New Roman", etc.
DEFAULT_FONT_SIZE = 36


# ============================================================================
# APPLY DEFAULTS FUNCTION
# ============================================================================
def apply_defaults():
    """Apply all custom defaults for the presentation."""
    # Set background
    config.background_color = BACKGROUND_COLOR

    # Set default colors
    Mobject.set_default(color=DEFAULT_COLOR)
    Text.set_default(color=DEFAULT_COLOR, font=DEFAULT_FONT)
    Tex.set_default(tex_template=DEFAULT_TEX_TEMPLATE, color=DEFAULT_COLOR)
    MathTex.set_default(tex_template=DEFAULT_TEX_TEMPLATE, color=DEFAULT_COLOR)

    print("Custom Manim configuration applied ✓")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_color(index):
    """Get color from cycle by index."""
    return COLOR_CYCLE[index % len(COLOR_CYCLE)]


def colormap_color(value, vmin, vmax, cmap_name="plasma", power=1.0, invert=False):
    """Map a scalar value to a manim color via a matplotlib colormap."""
    normalized = (value - vmin) / (vmax - vmin)
    if invert:
        normalized = 1.0 - normalized
    rgba = plt.get_cmap(cmap_name)(normalized**power)
    return rgb_to_color(rgba[:3])


def mixbox_interpolate_color(color1, color2, t):
    mix_color = mixbox.lerp(color_to_int_rgb(color1), color_to_int_rgb(color2), t)
    return ManimColor(mix_color)


# Cyclic colour wheel (blue -> purple -> pink -> orange -> green -> cyan),
# shared by every slide that colours points by position around a loop.
T_CYCLE = [COLOR_CYCLE[i] for i in (0, 4, 3, 1, 2, 5)]


def cyclic_color(u):
    """u in [0, 1) -> ManimColor mixed (mixbox) around T_CYCLE."""
    x = (u % 1.0) * len(T_CYCLE)
    i = int(np.floor(x)) % len(T_CYCLE)
    return mixbox_interpolate_color(T_CYCLE[i], T_CYCLE[(i + 1) % len(T_CYCLE)], x - i)


def cyclic_rgba(t):
    """Angles t (radians, array) -> (n, 4) RGBA floats mixed around T_CYCLE,
    for PMobject point colours."""
    stops = [c.to_rgb() for c in T_CYCLE]
    u = (np.asarray(t, float) % (2 * np.pi)) / (2 * np.pi) * len(T_CYCLE)
    i = np.floor(u).astype(int) % len(T_CYCLE)
    a = u - np.floor(u)
    rgb = np.array(
        [
            mixbox.lerp_float(stops[k], stops[(k + 1) % len(T_CYCLE)], w)
            for k, w in zip(i, a)
        ]
    ).reshape(-1, 3)
    return np.hstack([rgb, np.ones((len(rgb), 1))])


# ============================================================================
# POINT CLOUDS (PMobject, Cairo renderer)
# ============================================================================
# Cairo's Camera.display_point_cloud draws each point as a block of
# int(stroke_width) x int(stroke_width) pixels: a little square, at a size
# fixed in pixels whatever the render resolution. Call install_round_points()
# once per module (after apply_defaults()) to draw filled discs instead, and
# size points with res_px() so they look the same at -ql and -qh.
REF_PIXEL_HEIGHT = 1440  # resolution that res_px() sizes are tuned for


def res_px(px_at_ref: float) -> float:
    """Scale a pixel size tuned at REF_PIXEL_HEIGHT to the current resolution."""
    return px_at_ref * config.pixel_height / REF_PIXEL_HEIGHT


@lru_cache(maxsize=None)
def _circular_nudges(thickness: int) -> np.ndarray:
    c = (thickness - 1) / 2  # disc centre (on a half pixel for even sizes)
    k = np.arange(thickness)
    dy, dx = np.meshgrid(k, k, indexing="ij")
    keep = (dy - c) ** 2 + (dx - c) ** 2 <= (thickness / 2) ** 2
    return np.c_[dy[keep], dx[keep]] - (thickness - 1) // 2


def make_circular_nudges(thickness: float) -> np.ndarray:
    """Pixel-nudge offsets forming a filled disc (round PMobject points).

    Always at least one pixel, so tiny (res_px-scaled) points never vanish.
    """
    # the epsilon stops e.g. res_px(...) == 4.9999999 truncating to 4
    return _circular_nudges(max(int(thickness + 1e-6), 1))


def _display_point_cloud_point_major(
    self, pmobject, points, rgbas, thickness, pixel_array
):
    """Drop-in replacement for Camera.display_point_cloud.

    Stock Cairo writes nudge-major (every point's first pixel, then every
    point's second pixel, ...), so where discs overlap they interleave
    pixel by pixel. Writing point-major gives true painter's order: later
    points fully cover earlier ones.
    """
    if len(points) == 0:
        return
    pixel_coords = self.points_to_pixel_coords(pmobject, points)
    nudges = self.get_thickening_nudges(thickness)
    pixel_coords = (pixel_coords[:, None, :] + nudges[None, :, :]).reshape(-1, 2)

    rgba_len = pixel_array.shape[2]
    rgbas = (self.rgb_max_val * rgbas).astype(self.pixel_array_dtype)
    rgbas = np.repeat(rgbas, len(nudges), axis=0)

    on_screen = self.on_screen_pixels(pixel_coords)
    pixel_coords = pixel_coords[on_screen]
    rgbas = rgbas[on_screen]

    ph, pw = self.pixel_height, self.pixel_width
    indices = (pixel_coords[:, 0] + pw * pixel_coords[:, 1]).astype("int")
    flat = pixel_array.reshape((ph * pw, rgba_len))
    flat[indices] = rgbas
    pixel_array[:, :] = flat.reshape((ph, pw, rgba_len))


_ORIGINAL_THICKENING_NUDGES = Camera.get_thickening_nudges
_ORIGINAL_DISPLAY_POINT_CLOUD = Camera.display_point_cloud


def install_round_points(round_points=True, point_major=True):
    """Patch the Cairo camera (2D and 3D) to draw PMobject points as discs,
    layered in point order. Pass False to restore the stock behaviour."""
    Camera.get_thickening_nudges = (
        (lambda self, thickness: make_circular_nudges(thickness))
        if round_points
        else _ORIGINAL_THICKENING_NUDGES
    )
    Camera.display_point_cloud = (
        _display_point_cloud_point_major
        if point_major
        else _ORIGINAL_DISPLAY_POINT_CLOUD
    )


# Cairo writes PMobject colours straight into the pixel array with no alpha
# compositing, so FadeIn / FadeOut (which animate opacity) make clouds pop in
# and out. These blend RGB towards / from the background instead, and leave
# the cloud's updaters running so it can move while it fades.
class PMRecolor(Animation):
    """Blend a PMobject's point colours from their current values to
    ``target_rgb`` (a colour, a (3,) array, or an (N, 3) array)."""

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


def _background_rgb():
    return np.asarray(color_to_rgb(config.background_color), dtype=np.float64)


class PMFadeOut(PMRecolor):
    """FadeOut for a PMobject: blend it into the background, then remove it.
    Its colours are restored afterwards, so it can be added again later."""

    def __init__(self, mob: PMobject, **kwargs):
        kwargs.setdefault("remover", True)
        super().__init__(mob, _background_rgb(), **kwargs)

    def clean_up_from_scene(self, scene):
        super().clean_up_from_scene(scene)
        self.mobject.rgbas[:, :3] = self._start_rgb


class PMFadeIn(PMRecolor):
    """FadeIn for a PMobject: blend it in from the background. It is added to
    the scene when played, and set to the background colour straight away so
    it cannot pop in before the animation starts."""

    def __init__(self, mob: PMobject, **kwargs):
        target = mob.rgbas[:, :3].copy()
        mob.rgbas[:, :3] = _background_rgb()
        kwargs.setdefault("introducer", True)
        super().__init__(mob, target, **kwargs)


# ============================================================================
# SMALL TEXT
# ============================================================================
# Marcellus kerns badly when Pango renders it small, so Text is always built
# at TEXT_BUILD_SIZE and scaled down. TeX output is unaffected, but at equal
# font_size it comes out TEX_TO_TEXT times the size of Text.
TEXT_BUILD_SIZE = 48
TEX_TO_TEXT = 0.75

HEADING_TITLE_SIZE = 28.8
HEADING_SUBTITLE_SIZE = 15.36


def SmallText(text, font_size=TEXT_BUILD_SIZE, **kwargs):
    """Text, built at 48pt and scaled to font_size. Text with newlines
    becomes a centred Paragraph."""
    if "\n" in text:
        mob = Paragraph(
            *text.split("\n"), font_size=TEXT_BUILD_SIZE, alignment="center", **kwargs
        )
    else:
        mob = Text(text, font_size=TEXT_BUILD_SIZE, **kwargs)
    return mob.scale(font_size / TEXT_BUILD_SIZE)


def SmallTex(text, font_size=DEFAULT_FONT_SIZE, **kwargs):
    """Text with inline maths ($...$) in the slide font; newlines are centred
    line breaks (each line in an \\mbox so TeX does not re-wrap it).
    font_size is a Tex font size (see TEX_TO_TEXT)."""
    lines = [r"\mbox{" + line + "}" for line in text.split("\n")]
    return Tex(r"\\ ".join(lines), font_size=font_size, **kwargs)


def make_heading(
    title,
    subtitle=None,
    title_tex=False,
    subtitle_tex=False,
    title_color=DEFAULT_COLOR,
    subtitle_color=ACCENT_COLOR,
):
    """Standard slide heading: title (in HEADING_FONT) at the top, optional
    subtitle below. Use *_tex=True for strings containing inline maths."""

    def build(text, size, color, use_tex, is_title):
        if use_tex:
            kw = {"tex_template": HEADING_TEX_TEMPLATE} if is_title else {}
            return SmallTex(text, font_size=size / TEX_TO_TEXT, color=color, **kw)
        kw = {"font": HEADING_FONT} if is_title else {}
        return SmallText(text, font_size=size, color=color, **kw)

    t = build(title, HEADING_TITLE_SIZE, title_color, title_tex, True)
    t.to_edge(UP, buff=0.3)
    if subtitle is None:
        return VGroup(t)
    s = build(subtitle, HEADING_SUBTITLE_SIZE, subtitle_color, subtitle_tex, False)
    s.next_to(t, DOWN, buff=0.12)
    return VGroup(t, s)


def create_standard_axes(x_range=[-1, 10, 1], y_range=[-1, 10, 1], **kwargs):
    """Create axes with standard defaults."""
    axes_config = DEFAULT_AXES_CONFIG.copy()
    axes_config.update(kwargs)
    return Axes(x_range=x_range, y_range=y_range, **axes_config)


# ============================================================================
# LOGO SETTINGS
# ============================================================================
LOGO_PATH = (
    Path(__file__).parent / "tutte_logo_horizontal.png"
)  # Update this to your logo file
LOGO_SCALE = 0.1  # Adjust size as needed
LOGO_POSITION = UL  # Upper left (or UR, DL, DR for other corners)
LOGO_OPACITY = 1.0  # 0.0 to 1.0
LOGO_BUFFER = 0.2  # Distance from edge
USE_LOGO = True  # Toggle logo on/off globally


# ============================================================================
# LOGO HELPER FUNCTION
# ============================================================================
def create_logo():
    """Create a logo ImageMobject with standard settings."""
    if not USE_LOGO:
        return None

    if not LOGO_PATH.exists():
        print(f"Warning: Logo file not found at {LOGO_PATH}")
        return None

    try:
        logo = ImageMobject(LOGO_PATH)
        logo.scale(LOGO_SCALE)
        logo.set_opacity(LOGO_OPACITY)
        logo.to_corner(LOGO_POSITION, buff=LOGO_BUFFER)
        return logo
    except Exception as e:
        print(f"Error loading logo: {e}")
        return None


def add_logo_to_scene(scene):
    """
    Add logo to a scene in the foreground.

    Usage in Scene.construct():
        add_logo_to_scene(self)
    """
    logo = create_logo()
    if logo is not None:
        if isinstance(scene, ThreeDScene):
            if config.renderer == RendererType.OPENGL:
                # OpenGL renderer: ImageMobject doesn't support the fixed_in_frame
                # shader uniform used by add_fixed_in_frame_mobjects, so skip the logo.
                scene.logo = None
                return None
            scene.add_fixed_in_frame_mobjects(logo)
        else:
            scene.add(logo)
    scene.logo = logo
    return logo


def add_logo_to_background(scene):
    """
    Add logo to a scene in the background (won't be affected by camera).

    Usage in Scene.construct():
        add_logo_to_background(self)
    """
    logo = create_logo()
    if logo is not None:
        if isinstance(scene, ThreeDScene):
            if config.renderer == RendererType.OPENGL:
                # OpenGL renderer: ImageMobject doesn't support the fixed_in_frame
                # shader uniform used by add_fixed_in_frame_mobjects, so skip the logo.
                scene.logo = None
                return None
            scene.add_fixed_in_frame_mobjects(logo)
        else:
            scene.add_foreground_mobject(logo)
    scene.logo = logo
    return logo


def create_styled_axes(
    x_range,
    y_range,
    x_label_tex,
    y_label_tex,
    x_length=6,
    y_length=6,
    y_decimal_places=2,
    y_tick_font_size=18,
    x_tick_labels=None,
    x_tick_label_buff=0.25,
    include_y_axis=True,
):
    axes = Axes(
        x_range=x_range,
        y_range=y_range,
        x_length=x_length,
        y_length=y_length,
        axis_config={
            "include_tip": True,
            "color": DEFAULT_COLOR,
            "tip_shape": StealthTip,
            "tip_height": 0.1,
            "tip_width": 0.1,
        },
        y_axis_config={
            "numbers_to_include": (
                [round(x, 2) for x in np.arange(y_range[0], y_range[1], y_range[2])[1:]]
                if include_y_axis
                else []
            ),
            "decimal_number_config": {"num_decimal_places": y_decimal_places},
            "font_size": y_tick_font_size,
            "stroke_width": 1 if include_y_axis else 0,
            "include_tip": include_y_axis,
        },
    )

    # --- Y-LABEL ---
    y_label = Text(y_label_tex).scale(0.45).rotate(PI / 2)
    y_label.next_to(axes.get_y_axis(), LEFT, buff=0.3)

    # --- X-TICK LABELS ---
    tick_labels_group = VGroup()
    if x_tick_labels:
        # We find the physical start and end of the x-axis line
        x_start = axes.get_origin()
        x_end = axes.get_x_axis().get_end()

        for i, label_str in enumerate(x_tick_labels, 1):
            clean_text = label_str.replace("+", "\n")
            tick_label = Paragraph(
                clean_text, font_size=14, alignment="center", line_spacing=0.5
            )

            # THE FIX: Position relative to the physical X-axis line,
            # not the data coordinate (i, 0).
            # We determine the horizontal center for tick 'i'
            x_pos = axes.c2p(i, y_range[0])[0]  # Get only the X coordinate
            y_pos = axes.get_x_axis().get_top()[
                1
            ]  # Get the Y coordinate of the axis line

            tick_label.move_to([x_pos, y_pos, 0])
            tick_label.shift(
                DOWN
                * (
                    (tick_label.height / 2)
                    + axes.get_x_axis().tick_size
                    + x_tick_label_buff
                )
            )  # Manual vertical offset from the line

            tick_labels_group.add(tick_label)

    # --- X-AXIS TITLE ---
    x_label = Text(x_label_tex).scale(0.5)

    # Position relative to the bounding box of the ticks or the axis itself
    if x_tick_labels:
        x_label.next_to(tick_labels_group, DOWN, buff=0.2)
    else:
        x_label.next_to(axes.get_x_axis(), DOWN, buff=0.4)

    return VGroup(axes, x_label, y_label, tick_labels_group)


# ===========================================================================
# TIMC Slide Class
# ===========================================================================


# How far past the frame edge a section title wipes away: stopping exactly at
# the edge leaves an antialiased sliver of the title card on screen.
SECTION_WIPE_OVERSHOOT = 0.2


class _TIMCMixin:
    """
    Shared implementation for TIMCSlide (2D) and ThreeDTIMCSlide (3D).

    Subclasses must define ``_add_overlay_mobject(mob)`` to handle the
    difference between 2D (``add_foreground_mobject``) and 3D
    (``add_fixed_in_frame_mobjects``).
    """

    max_duration_before_split_reverse = None
    skip_reversing = True

    def _timc_init(self):
        """Common initialisation — call from each subclass __init__."""
        self.state_dir = "slide_states"
        os.makedirs(self.state_dir, exist_ok=True)
        add_logo_to_background(self)

    # ------------------------------------------------------------------
    # Overlay hook — overridden by each concrete class
    # ------------------------------------------------------------------
    def _add_overlay_mobject(self, mob):
        """Add *mob* so it renders in screen-space.  Override in 3-D."""
        self.add_foreground_mobject(mob)

    # ------------------------------------------------------------------
    # Slide navigation
    # ------------------------------------------------------------------
    def marked_next_slide(self, *args, **kwargs):
        marker = Dot(radius=0.1, color=COLOR_CYCLE[2]).to_corner(DR, buff=0.2)
        self._add_overlay_mobject(marker)
        self.wait(0.1)
        self.next_slide(*args, **kwargs)
        self.remove(marker)

    def text_slide(self, text, clear_run_time=1, **kwargs):
        """Convenience: show centered text, pause, then clear."""
        self.add_centered_text(text, **kwargs)
        self.marked_next_slide()
        self.clear_slide(run_time=clear_run_time)

    def _section_break(self, **kwargs):
        """marked_next_slide(**kwargs), unless nothing has played since the
        last break: that would make an empty slide (an extra click), so the
        options are applied to the slide already started instead.
        (Reads manim-slides' private slide-tracking state.)"""
        if self._current_animation > self._start_animation:
            self.marked_next_slide(**kwargs)
        else:
            for key, value in kwargs.items():
                setattr(self._base_slide_config, key, value)

    # ------------------------------------------------------------------
    # Camera state
    # ------------------------------------------------------------------
    def _get_camera_state(self):
        """Get a unified camera state that works for both 2D and 3D."""
        cam = self.camera
        is_3d = hasattr(cam, "get_phi")

        state = {
            "type": "3d" if is_3d else "2d",
            "center": (
                cam.frame.get_center().copy() if hasattr(cam, "frame") else ORIGIN
            ),
        }

        if is_3d:
            state.update(
                {
                    "phi": cam.get_phi(),
                    "theta": cam.get_theta(),
                    "focal_distance": cam.get_focal_distance(),
                    "gamma": cam.get_gamma(),
                    "zoom": cam.get_zoom(),
                    "frame_center": (
                        self.camera_target if hasattr(self, "camera_target") else ORIGIN
                    ),
                }
            )
        else:
            state["width"] = cam.frame.get_width() if hasattr(cam, "frame") else 14.0

        return state

    def _set_camera_state(self, state, animate=False, run_time=1.5):
        """Restore camera state with bi-directional fallbacks for 2D/3D transitions."""
        if state is None:
            return

        is_scene_3d = hasattr(self.camera, "get_phi")

        if is_scene_3d:
            phi = state.get("phi", 0)
            theta = state.get("theta", -90 * DEGREES)
            gamma = state.get("gamma", 0)
            focal_dist = state.get("focal_distance", 20.0)
            zoom = state.get("zoom", 14.22 / state.get("width", 14.22))
            center = state.get("frame_center", state.get("center", ORIGIN))

            if animate:
                self.move_camera(
                    phi=phi,
                    theta=theta,
                    gamma=gamma,
                    zoom=zoom,
                    focal_distance=focal_dist,
                    frame_center=center,
                    run_time=run_time,
                )
            else:
                self.set_camera_orientation(
                    phi=phi,
                    theta=theta,
                    gamma=gamma,
                    zoom=zoom,
                    focal_distance=focal_dist,
                )
                if hasattr(self.camera, "frame"):
                    self.camera.frame.move_to(center)
        else:
            center = state.get("center", state.get("frame_center", ORIGIN))
            if "width" in state:
                width = state["width"]
            elif "zoom" in state:
                width = 14.22 / state["zoom"]
            else:
                width = 14.22

            if hasattr(self.camera, "frame"):
                if animate:
                    self.play(
                        self.camera.frame.animate.move_to(center).set_width(width),
                        run_time=run_time,
                    )
                else:
                    self.camera.frame.move_to(center)
                    self.camera.frame.set_width(width)

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------
    def save_state(self, state_name):
        state_data = {
            "mobjects": [],
            "foreground_mobjects": [],
            "camera": self._get_camera_state(),
        }

        for mob in self.mobjects:
            if mob == self.logo:
                continue
            try:
                mob_copy = mob.copy()
                mob_copy.clear_updaters()
                for submob in mob_copy.get_family():
                    submob.clear_updaters()
                state_data["mobjects"].append(mob_copy)
            except Exception as e:
                print(f"Warning: Could not copy mobject {mob}: {e}")

        if hasattr(self, "foreground_mobjects"):
            for mob in self.foreground_mobjects:
                try:
                    state_data["foreground_mobjects"].append(mob.copy())
                except Exception as e:
                    print(f"Warning: Could not copy foreground mobject {mob}: {e}")

        self._save_extra_state(state_data)

        filepath = os.path.join(self.state_dir, f"{state_name}.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(state_data, f)
        print(f"State saved to {filepath} ({len(state_data['mobjects'])} mobjects)")

    def _save_extra_state(self, state_data):
        """Hook for subclasses to persist additional data (e.g. fixed-in-frame)."""

    def load_state(self, state_name):
        filepath = os.path.join(self.state_dir, f"{state_name}.pkl")

        if not os.path.exists(filepath):
            print(f"Warning: State file {filepath} not found. Skipping load.")
            return

        with open(filepath, "rb") as f:
            state_data = pickle.load(f)

        for mob in state_data["mobjects"]:
            self.add(mob)

        self._load_extra_state(state_data)

        if "foreground_mobjects" in state_data:
            for mob in state_data["foreground_mobjects"]:
                self.add_foreground_mobject(mob)

        if "camera" in state_data and state_data["camera"] is not None:
            self._set_camera_state(state_data["camera"], animate=False)
        elif state_data.get("camera_position") is not None and hasattr(
            self.camera, "frame"
        ):
            self.camera.frame.move_to(state_data["camera_position"])

        print(f"State loaded from {filepath} ({len(state_data['mobjects'])} mobjects)")

    def _load_extra_state(self, state_data):
        """Hook for subclasses to restore additional data (e.g. fixed-in-frame)."""

    # ------------------------------------------------------------------
    # Section transitions
    # ------------------------------------------------------------------
    def _build_section_slide(self, section_name):
        """Create the gradient-background + title VGroup used by section helpers."""
        background = Rectangle(
            width=config.frame_width,
            height=config.frame_height,
            fill_opacity=1,
            stroke_width=0,
        )
        background.set_sheen_direction(UP)
        background.set_fill(color=[DEFAULT_COLOR, ACCENT_COLOR, WHITE], opacity=1)
        section_title = Paragraph(
            section_name,
            font_size=72,
            color=WHITE,
            alignment="center",
            font="Marcellus SC",
        )
        return VGroup(background, section_title)

    def new_section(
        self, section_name, next_slide_prep=None, pause_at_title=True, auto_next=False
    ):
        """Wipe a section title down over the slide, then back up.

        auto_next: advance to the next slide on its own once the title has
        wiped away (at the end of a scene that flows into the next scene).
        """
        lead = {"auto_next": True} if auto_next and not pause_at_title else {}
        self._section_break(**lead)

        section_slide = self._build_section_slide(section_name)
        section_slide.move_to(UP * config.frame_height)

        self.play(
            section_slide.animate.shift(DOWN * config.frame_height),
            run_time=1,
        )
        self.wait(0.1)

        if pause_at_title:
            self.next_slide(auto_next=auto_next)
        else:
            self.wait()

        if next_slide_prep is None:
            self.clear()
            add_logo_to_background(self)
        else:
            next_slide_prep()

        self.play(
            section_slide.animate.shift(
                UP * (config.frame_height + SECTION_WIPE_OVERSHOOT)
            ),
            run_time=1,
        )

    def start_section_wipe(self, section_name):
        self.marked_next_slide()

        section_slide = self._build_section_slide(section_name)
        section_slide.move_to(UP * config.frame_height)

        self.play(
            section_slide.animate.shift(DOWN * config.frame_height),
            run_time=1,
        )
        self.wait(0.1)

    def end_section_wipe(self, section_name, next_slide_prep=None):
        section_slide = self._build_section_slide(section_name)

        self.add(section_slide)
        self.wait(0.1)

        self.next_slide()
        if next_slide_prep is None:
            self.clear()
            add_logo_to_background(self)
        else:
            next_slide_prep()

        self.play(
            section_slide.animate.shift(
                UP * (config.frame_height + SECTION_WIPE_OVERSHOOT)
            ),
            run_time=1,
        )

    # ------------------------------------------------------------------
    # Text helpers
    # ------------------------------------------------------------------
    def add_centered_text(
        self,
        text,
        max_width=0.85,
        max_height=0.7,
        font_size=72,
        line_spacing=1.2,
        t2c=None,
        animate=True,
        location=ORIGIN,
        **text_kwargs,
    ):
        """
        Add centered, wrapped text that automatically fits within a box.

        Args:
            text: The text string to display
            max_width: Maximum width as fraction of frame width (default 0.85)
            max_height: Maximum height as fraction of frame height (default 0.7)
            font_size: Starting font size (will be reduced if needed)
            line_spacing: Line spacing multiplier (1.0 = single space, 1.5 = 1.5x, etc.)
            t2c: Dict mapping words/phrases to colors (like Text's t2c)
            animate: Whether to animate the text appearance (default True)
            **text_kwargs: Additional arguments passed to Text()

        Returns:
            The text VGroup that was added
        """
        max_width_pixels = config.pixel_width * max_width
        max_height_pixels = config.pixel_height * max_height

        current_font_size = font_size
        text_group = None

        for attempt in range(10):
            lines = self._wrap_text_to_lines(
                text, current_font_size, max_width_pixels, t2c, **text_kwargs
            )
            text_group = self._position_lines_by_baseline(
                lines, current_font_size, line_spacing
            )

            if (
                text_group.width <= max_width_pixels
                and text_group.height <= max_height_pixels
            ):
                break

            current_font_size *= 0.9
            if current_font_size < 16:
                print("Warning: Text may not fit well even at minimum size")
                break

        if location is ORIGIN:
            text_group.move_to(ORIGIN)
        elif location is UP:
            text_group.to_edge(UP)
        elif location is DOWN:
            text_group.to_edge(DOWN)

        if animate:
            self.play(Write(text_group))
        else:
            self.add(text_group)

        return text_group

    def add_title_text(
        self,
        text,
        max_width=0.5,
        max_height=0.2,
        font_size=48,
        line_spacing=0.95,
        t2c=None,
        animate=True,
        location=UP,
        **text_kwargs,
    ):
        return self.add_centered_text(
            text=text,
            max_width=max_width,
            max_height=max_height,
            font_size=font_size,
            line_spacing=line_spacing,
            t2c=t2c,
            animate=animate,
            location=location,
            **text_kwargs,
        )

    def _position_lines_by_baseline(self, lines, font_size, line_spacing):
        """Position text lines using consistent baseline spacing."""
        if not lines:
            return VGroup()

        baseline_distance = font_size * line_spacing * 0.015
        text_group = VGroup(*lines)
        current_y = 0

        for i, line in enumerate(lines):
            if i == 0:
                line.move_to(UP * current_y)
            else:
                current_y -= baseline_distance
                line.move_to(UP * current_y)

        return text_group

    def _wrap_text_to_lines(self, text, font_size, max_width, t2c=None, **text_kwargs):
        """Wrap text into lines that fit within *max_width*."""
        paragraphs = text.split("\n")
        all_lines = []

        for paragraph in paragraphs:
            if not paragraph.strip():
                all_lines.append(Text(" ", font_size=font_size, **text_kwargs))
                continue

            chars_per_line = int(max_width / font_size)
            print(
                f"Initial estimated chars per line of {chars_per_line} at font size {font_size}"
            )
            print(f"Max width: {max_width}")
            chars_per_line = max(chars_per_line, 16)
            print(
                f"Estimated chars per line of {chars_per_line} at font size {font_size}"
            )

            wrapped_lines = wrap(
                paragraph, width=chars_per_line, break_long_words=False
            )

            for line in wrapped_lines:
                text_obj = Text(line, font_size=font_size, t2c=t2c, **text_kwargs)

                if text_obj.width > max_width:
                    words = line.split()
                    current_line = []

                    for word in words:
                        test_line = " ".join(current_line + [word])
                        test_obj = Text(
                            test_line, font_size=font_size, t2c=t2c, **text_kwargs
                        )

                        if test_obj.width <= max_width:
                            current_line.append(word)
                        else:
                            if current_line:
                                all_lines.append(
                                    Text(
                                        " ".join(current_line),
                                        font_size=font_size,
                                        t2c=t2c,
                                        **text_kwargs,
                                    )
                                )
                            current_line = [word]

                    if current_line:
                        all_lines.append(
                            Text(
                                " ".join(current_line),
                                font_size=font_size,
                                t2c=t2c,
                                **text_kwargs,
                            )
                        )
                else:
                    all_lines.append(text_obj)

        return (
            all_lines
            if all_lines
            else [Text(text, font_size=font_size, t2c=t2c, **text_kwargs)]
        )

    def clear_slide(self, animation=FadeOut, run_time=1):
        """Fade out everything but the logo. Point clouds (PMobjects), also
        inside groups, get PMFadeOut, as FadeOut cannot fade them in Cairo."""
        anims, emptied = [], []
        for mobject in self.mobjects:
            if mobject == self.logo:
                continue
            family = mobject.get_family()
            if animation is not FadeOut or not any(
                isinstance(m, PMobject) for m in family
            ):
                anims.append(animation(mobject))
            elif isinstance(mobject, PMobject) and len(family) == 1:
                anims.append(PMFadeOut(mobject))
            else:
                # fade the leaves one by one, then drop the emptied group
                for leaf in (m for m in family if not m.submobjects):
                    anims.append(
                        PMFadeOut(leaf) if isinstance(leaf, PMobject) else FadeOut(leaf)
                    )
                emptied.append(mobject)
        self.play(*anims, run_time=run_time)
        self.remove(*emptied)


# ===========================================================================
# Concrete slide classes
# ===========================================================================


class TIMCSlide(_TIMCMixin, Slide):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._timc_init()

    def _add_overlay_mobject(self, mob):
        self.add_foreground_mobject(mob)


class ThreeDTIMCSlide(_TIMCMixin, ThreeDSlide):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._timc_init()

    def _add_overlay_mobject(self, mob):
        self.add_fixed_in_frame_mobjects(mob)

    def marked_next_slide(self, *args, **kwargs):
        marker = Dot(radius=0.1, color=COLOR_CYCLE[2]).to_corner(DR, buff=0.2)
        self.add_fixed_in_frame_mobjects(marker)
        self.wait(0.1)
        super().next_slide(*args, **kwargs)
        self.remove(marker)

    # --- 3D-specific state hooks ---

    def _save_extra_state(self, state_data):
        state_data["fixed_in_frame_mobjects"] = []
        if hasattr(self, "fixed_in_frame_mobjects"):
            for mob in self.camera.fixed_in_frame_mobjects:
                try:
                    state_data["fixed_in_frame_mobjects"].append(mob.copy())
                except Exception as e:
                    print(f"Warning: Could not copy fixed_in_frame mobject {mob}: {e}")

    def _load_extra_state(self, state_data):
        if "fixed_in_frame_mobjects" in state_data:
            for mob in state_data["fixed_in_frame_mobjects"]:
                self.add_fixed_in_frame_mobjects(mob)

    # --- 3D-specific section wipes ---

    def new_section(
        self, section_name, next_slide_prep=None, pause_at_title=True, auto_next=False
    ):
        """See _TIMCMixin.new_section."""
        lead = {"auto_next": True} if auto_next and not pause_at_title else {}
        self._section_break(**lead)

        section_slide = self._build_section_slide(section_name)
        section_slide.move_to(UP * config.frame_height)

        self.add_fixed_in_frame_mobjects(section_slide)
        self.play(
            section_slide.animate.shift(DOWN * config.frame_height),
            run_time=1,
        )

        if pause_at_title:
            self.next_slide(auto_next=auto_next)
        else:
            self.wait()

        if next_slide_prep is None:
            self.clear()
            add_logo_to_background(self)
        else:
            next_slide_prep()

        self.play(
            section_slide.animate.shift(
                UP * (config.frame_height + SECTION_WIPE_OVERSHOOT)
            ),
            run_time=1,
        )

    def start_section_wipe(self, section_name):
        self.marked_next_slide()

        section_slide = self._build_section_slide(section_name)
        self.add_fixed_in_frame_mobjects(section_slide)

        # Position in screen-space then animate down
        section_slide.move_to(UP * config.frame_height)
        self.play(
            section_slide.animate.move_to(ORIGIN),
            run_time=1,
        )
        self.wait(0.1)


# ===========================================================================
# Phase-transition mixin (overview → zoom → pan → section wipe)
# ===========================================================================


class PhaseSlide:
    """Mixin for scenes that zoom into the pipeline overview and pan between stages."""

    def _pan_to_stages(self, *stage_keys, zoom=2.5, pan_run_time=1.5):
        """Zoom into embeddings then pan across to each subsequent stage."""
        with open(os.path.join(self.state_dir, "stage_centers.json")) as f:
            centers = json.load(f)

        self.play(self.logo.animate.set_opacity(0.0), run_time=0.25)
        self.move_camera(
            frame_center=np.array(centers["embeddings"]),
            zoom=zoom,
            run_time=pan_run_time,
            rate_func=smooth,
        )

        for key in stage_keys:
            self.move_camera(
                frame_center=np.array(centers[key]),
                run_time=pan_run_time,
                rate_func=smooth,
            )

    def transition_to_overview(self, run_time_fadeout=1.5, run_time_fadein=0.75):
        filepath = os.path.join(self.state_dir, "overview.pkl")
        with open(filepath, "rb") as f:
            overview_data = pickle.load(f)

        overview_camera = overview_data["camera"]
        overview_mobs = Group(
            *[
                obj
                for obj in overview_data["mobjects"]
                if not isinstance(obj, ImageMobject)
            ]
        )

        current_mobs = Group(
            *list(
                obj
                for obj in self.mobjects
                if not isinstance(obj, ImageMobject) and obj != self.logo
            )
        )

        self.move_camera(
            zoom=0.33,
            added_anims=[
                FadeOut(current_mobs),
            ],
            run_time=2,
        )
        self.move_camera(
            frame_center=overview_camera["frame_center"],
            zoom=1.0,
            phi=0,
            theta=-90 * DEGREES,
            run_time=0.1,
        )
        self.remove(current_mobs)

        self.add(overview_mobs)
        self.play(
            FadeIn(overview_mobs),
            run_time=run_time_fadein,
            rate_func=smooth,
        )
