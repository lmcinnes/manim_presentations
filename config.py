from manim import *
from manim_slides import Slide, ThreeDSlide
import os
import pickle
from textwrap import wrap
from pathlib import Path

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

    print("Custom Manim configuration applied ✓")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_color(index):
    """Get color from cycle by index."""
    return COLOR_CYCLE[index % len(COLOR_CYCLE)]


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


class TIMCSlide(Slide):

    max_duration_before_split_reverse = None
    skip_reversing = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_dir = "slide_states"
        os.makedirs(self.state_dir, exist_ok=True)
        add_logo_to_background(self)

    def marked_next_slide(self, *args, **kwargs):
        marker = Dot(radius=0.1, color=COLOR_CYCLE[2]).to_corner(DR, buff=0.2)
        self.add_foreground_mobject(marker)
        self.wait(0.1)
        self.next_slide(*args, **kwargs)
        self.remove(marker)

    # def _get_camera_state(self):
    #     """Get camera state regardless of camera type."""
    #     cam = self.camera
    #     if hasattr(cam, "frame"):
    #         # MovingCamera / 2D
    #         return {
    #             "type": "2d",
    #             "center": cam.frame.get_center().copy(),
    #             "width": cam.frame.get_width(),
    #         }
    #     else:
    #         try:
    #             # ThreeDCamera
    #             return {
    #                 "type": "3d",
    #                 "phi": cam.get_phi(),
    #                 "theta": cam.get_theta(),
    #                 "focal_distance": cam.get_focal_distance(),
    #                 "gamma": cam.get_gamma(),
    #                 "zoom": cam.get_zoom(),
    #                 "frame_center": (
    #                     self.camera_target  # the point the camera orbits around
    #                     if hasattr(self, "camera_target")
    #                     else ORIGIN
    #                 ),
    #             }
    #         except:
    #             return {}

    # def _set_camera_state(self, state, animate=False, run_time=1.5):
    #     """Restore camera state regardless of camera type."""
    #     if state is None:
    #         return
    #     if state.get("type") == "2d":
    #         if hasattr(self.camera, "frame"):
    #             if animate:
    #                 self.play(
    #                     self.camera.frame.animate.move_to(state["center"]).set_width(
    #                         state["width"]
    #                     ),
    #                     run_time=run_time,
    #                 )
    #             else:
    #                 self.camera.frame.move_to(state["center"])
    #                 self.camera.frame.set_width(state["width"])
    #     elif state.get("type") == "3d":
    #         try:
    #             # ThreeDCamera
    #             if animate:
    #                 self.move_camera(
    #                     phi=state["phi"],
    #                     theta=state["theta"],
    #                     focal_distance=state["focal_distance"],
    #                     gamma=state["gamma"],
    #                     zoom=state["zoom"],
    #                     frame_center=state["frame_center"],
    #                     run_time=run_time,
    #                 )
    #             else:
    #                 self.set_camera_orientation(
    #                     phi=state["phi"],
    #                     theta=state["theta"],
    #                     focal_distance=state["focal_distance"],
    #                     gamma=state["gamma"],
    #                     zoom=state["zoom"],
    #                 )
    #                 self.move_camera(frame_center=state["frame_center"], run_time=0.1)
    #         except:
    #             pass
    #     else:
    #         pass

    def _get_camera_state(self):
        """Get a unified camera state that works for both 2D and 3D."""
        cam = self.camera
        # Detect if we are in a 3D context
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
            # 2D cameras use width/height for zoom
            state["width"] = cam.frame.get_width() if hasattr(cam, "frame") else 14.0

        return state

    def _set_camera_state(self, state, animate=False, run_time=1.5):
        """Restore camera state with bi-directional fallbacks for 2D/3D transitions."""
        if state is None:
            return

        # Determine what the CURRENT scene is capable of
        is_scene_3d = hasattr(self.camera, "get_phi")

        # 1. LOADING INTO A 3D SCENE
        if is_scene_3d:
            # If the state was 2D, we provide 3D defaults
            phi = state.get("phi", 0)
            theta = state.get("theta", -90 * DEGREES)
            gamma = state.get("gamma", 0)
            focal_dist = state.get("focal_distance", 20.0)

            # Mapping Zoom: 2D uses 'width', 3D uses 'zoom'.
            # Default Manim width is 14.22 (16/9 * 8).
            zoom = state.get("zoom", 14.22 / state.get("width", 14.22))

            # Center: Use frame_center if exists, otherwise center
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
                # ThreeDCamera frame movement can be finicky; move_to center directly
                if hasattr(self.camera, "frame"):
                    self.camera.frame.move_to(center)

        # 2. LOADING INTO A 2D SCENE
        else:
            # If the state was 3D, we extract the 2D-compatible parts
            # 3D 'frame_center' maps to 2D 'center'
            center = state.get("center", state.get("frame_center", ORIGIN))

            # 3D 'zoom' maps back to 2D 'width'
            # width = default_width / zoom
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

    def save_state(self, state_name):
        state_data = {
            "mobjects": [],
            "foreground_mobjects": [],
            "camera": self._get_camera_state(),
        }

        # Get all mobjects (main list)
        for mob in self.mobjects:
            if mob == self.logo:
                continue
            try:
                # Make a copy and clear its updaters
                mob_copy = mob.copy()
                mob_copy.clear_updaters()

                # Recursively clear updaters from submobjects
                for submob in mob_copy.get_family():
                    submob.clear_updaters()

                state_data["mobjects"].append(mob_copy)
            except Exception as e:
                print(f"Warning: Could not copy mobject {mob}: {e}")

        # Also get foreground mobjects if they exist
        if hasattr(self, "foreground_mobjects"):
            for mob in self.foreground_mobjects:
                try:
                    state_data["foreground_mobjects"].append(mob.copy())
                except Exception as e:
                    print(f"Warning: Could not copy foreground mobject {mob}: {e}")

        # Save to file
        filepath = os.path.join(self.state_dir, f"{state_name}.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(state_data, f)

        print(f"State saved to {filepath} ({len(state_data['mobjects'])} mobjects)")

    def load_state(self, state_name):
        filepath = os.path.join(self.state_dir, f"{state_name}.pkl")

        if not os.path.exists(filepath):
            print(f"Warning: State file {filepath} not found. Skipping load.")
            return

        with open(filepath, "rb") as f:
            state_data = pickle.load(f)

        # Add all mobjects to the scene without animation
        for mob in state_data["mobjects"]:
            self.add(mob)

        # Add foreground mobjects if they exist
        if "foreground_mobjects" in state_data:
            for mob in state_data["foreground_mobjects"]:
                self.add_foreground_mobject(mob)

        # Restore camera position if saved
        if "camera" in state_data and state_data["camera"] is not None:
            self._set_camera_state(state_data["camera"], animate=False)
        elif state_data.get("camera_position") is not None and hasattr(
            self.camera, "frame"
        ):
            self.camera.frame.move_to(state_data["camera_position"])

        print(f"State loaded from {filepath} ({len(state_data['mobjects'])} mobjects)")

    def new_section(self, section_name, next_slide_prep=None, pause_at_title=True):

        self.marked_next_slide()

        # Create gradient background
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
        section_slide = VGroup(background, section_title).move_to(
            UP * config.frame_height
        )

        # Animate sliding down
        self.play(
            section_slide.animate.shift(DOWN * config.frame_height),
            run_time=1,
        )
        self.wait(0.1)

        if pause_at_title:
            self.next_slide()
        else:
            self.wait()

        if next_slide_prep is None:
            self.clear()
            add_logo_to_background(self)
        else:
            next_slide_prep()

        # Animate sliding up
        self.play(
            section_slide.animate.shift(UP * config.frame_height),
            run_time=1,
        )

    def start_section_wipe(self, section_name):
        self.marked_next_slide()

        # Create gradient background
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
        section_slide = VGroup(background, section_title).move_to(
            UP * config.frame_height
        )

        # Animate sliding down
        self.play(
            section_slide.animate.shift(DOWN * config.frame_height),
            run_time=1,
        )
        self.wait(0.1)

    def end_section_wipe(self, section_name, next_slide_prep=None):
        # Create gradient background
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
        section_slide = VGroup(background, section_title)

        self.add(section_slide)
        self.wait(0.1)

        self.next_slide()
        if next_slide_prep is None:
            self.clear()
            add_logo_to_background(self)
        else:
            next_slide_prep()

        # Animate sliding up
        self.play(
            section_slide.animate.shift(UP * config.frame_height),
            run_time=1,
        )

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

        # Try to fit text with the given font size
        current_font_size = font_size
        text_group = None

        # Iteratively reduce font size until text fits
        for attempt in range(10):  # Max 10 attempts
            # Create lines with current font size
            lines = self._wrap_text_to_lines(
                text, current_font_size, max_width_pixels, t2c, **text_kwargs
            )

            # Position lines manually using baseline spacing
            text_group = self._position_lines_by_baseline(
                lines, current_font_size, line_spacing
            )

            # Check if it fits within bounds
            if (
                text_group.width <= max_width_pixels
                and text_group.height <= max_height_pixels
            ):
                break

            # Reduce font size and try again
            current_font_size *= 0.9

            if current_font_size < 16:  # Minimum font size
                print("Warning: Text may not fit well even at minimum size")
                break

        # Center the text
        if location is ORIGIN:
            text_group.move_to(ORIGIN)
        elif location is UP:
            text_group.to_edge(UP)
        elif location is DOWN:
            text_group.to_edge(DOWN)

        # Add to scene
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
        """
        Position text lines using consistent baseline spacing instead of bounding boxes.

        Args:
            lines: List of Text objects
            font_size: Font size being used
            line_spacing: Multiplier for line spacing (1.0 = single, 1.5 = 1.5x, etc.)

        Returns:
            VGroup with properly positioned lines
        """
        if not lines:
            return VGroup()

        # Calculate line height based on font size
        # This is the distance between baselines
        baseline_distance = (
            font_size * line_spacing * 0.015
        )  # Adjust multiplier as needed

        text_group = VGroup(*lines)

        # Position first line at the top
        current_y = 0

        for i, line in enumerate(lines):
            if i == 0:
                # First line stays at origin, we'll center the whole group later
                line.move_to(UP * current_y)
            else:
                # Move each subsequent line down by baseline_distance
                current_y -= baseline_distance
                line.move_to(UP * current_y)

        return text_group

    def _wrap_text_to_lines(self, text, font_size, max_width, t2c=None, **text_kwargs):
        """
        Internal method to wrap text into lines that fit within max_width.

        Returns:
            List of Text objects, one per line
        """
        # First, try to split by existing newlines
        paragraphs = text.split("\n")
        all_lines = []

        for paragraph in paragraphs:
            if not paragraph.strip():
                # For blank lines, add a small invisible placeholder to maintain spacing
                all_lines.append(Text(" ", font_size=font_size, **text_kwargs))
                continue

            # Estimate characters per line based on font size
            # This is approximate - Manim doesn't give us exact metrics before rendering
            chars_per_line = int(max_width / font_size)
            print(
                f"Initial estimated chars per line of {chars_per_line} at font size {font_size}"
            )
            print(f"Max width: {max_width}")
            chars_per_line = max(chars_per_line, 16)  # Minimum 16 chars
            print(
                f"Estimated chars per line of {chars_per_line} at font size {font_size}"
            )

            # Wrap the paragraph
            wrapped_lines = wrap(
                paragraph, width=chars_per_line, break_long_words=False
            )

            # Create Text objects for each line
            for line in wrapped_lines:
                text_obj = Text(line, font_size=font_size, t2c=t2c, **text_kwargs)

                # If this single line is still too wide, try to force break it
                if text_obj.width > max_width:
                    # Force break into smaller chunks
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
        self.play(
            *[animation(mobject) for mobject in self.mobjects if mobject != self.logo],
            run_time=run_time,
        )


class ThreeDTIMCSlide(ThreeDSlide):

    max_duration_before_split_reverse = None
    skip_reversing = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_dir = "slide_states"
        os.makedirs(self.state_dir, exist_ok=True)
        add_logo_to_background(self)

    def marked_next_slide(self, *args, **kwargs):
        marker = Dot(radius=0.1, color=COLOR_CYCLE[2]).to_corner(DR, buff=0.2)
        self.add_fixed_in_frame_mobjects(marker)
        self.wait(0.1)
        super().next_slide(*args, **kwargs)
        self.remove(marker)

    # def _get_camera_state(self):
    #     """Get camera state regardless of camera type."""
    #     cam = self.camera
    #     if hasattr(cam, "frame"):
    #         # MovingCamera / 2D
    #         return {
    #             "type": "2d",
    #             "center": cam.frame.get_center().copy(),
    #             "width": cam.frame.get_width(),
    #         }
    #     else:
    #         # ThreeDCamera
    #         return {
    #             "type": "3d",
    #             "phi": cam.get_phi(),
    #             "theta": cam.get_theta(),
    #             "focal_distance": cam.get_focal_distance(),
    #             "gamma": cam.get_gamma(),
    #             "zoom": cam.get_zoom(),
    #             "frame_center": (
    #                 self.camera_target  # the point the camera orbits around
    #                 if hasattr(self, "camera_target")
    #                 else ORIGIN
    #             ),
    #         }

    # def _set_camera_state(self, state, animate=False, run_time=1.5):
    #     """Restore camera state regardless of camera type."""
    #     if state is None:
    #         return
    #     if state.get("type") == "2d":
    #         if hasattr(self.camera, "frame"):
    #             if animate:
    #                 self.play(
    #                     self.camera.frame.animate.move_to(state["center"]).set_width(
    #                         state["width"]
    #                     ),
    #                     run_time=run_time,
    #                 )
    #             else:
    #                 self.camera.frame.move_to(state["center"])
    #                 self.camera.frame.set_width(state["width"])
    #     else:
    #         # ThreeDCamera
    #         if animate:
    #             self.move_camera(
    #                 phi=state["phi"],
    #                 theta=state["theta"],
    #                 focal_distance=state["focal_distance"],
    #                 gamma=state["gamma"],
    #                 zoom=state["zoom"],
    #                 frame_center=state["frame_center"],
    #                 run_time=run_time,
    #             )
    #         else:
    #             self.set_camera_orientation(
    #                 phi=state["phi"],
    #                 theta=state["theta"],
    #                 focal_distance=state["focal_distance"],
    #                 gamma=state["gamma"],
    #                 zoom=state["zoom"],
    #             )
    #             self.move_camera(frame_center=state["frame_center"], run_time=0.1)

    def _get_camera_state(self):
        """Get a unified camera state that works for both 2D and 3D."""
        cam = self.camera
        # Detect if we are in a 3D context
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
            # 2D cameras use width/height for zoom
            state["width"] = cam.frame.get_width() if hasattr(cam, "frame") else 14.0

        return state

    def _set_camera_state(self, state, animate=False, run_time=1.5):
        """Restore camera state with bi-directional fallbacks for 2D/3D transitions."""
        if state is None:
            return

        # Determine what the CURRENT scene is capable of
        is_scene_3d = hasattr(self.camera, "get_phi")

        # 1. LOADING INTO A 3D SCENE
        if is_scene_3d:
            # If the state was 2D, we provide 3D defaults
            phi = state.get("phi", 0)
            theta = state.get("theta", -90 * DEGREES)
            gamma = state.get("gamma", 0)
            focal_dist = state.get("focal_distance", 20.0)

            # Mapping Zoom: 2D uses 'width', 3D uses 'zoom'.
            # Default Manim width is 14.22 (16/9 * 8).
            zoom = state.get("zoom", 14.22 / state.get("width", 14.22))

            # Center: Use frame_center if exists, otherwise center
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
                # ThreeDCamera frame movement can be finicky; move_to center directly
                if hasattr(self.camera, "frame"):
                    self.camera.frame.move_to(center)

        # 2. LOADING INTO A 2D SCENE
        else:
            # If the state was 3D, we extract the 2D-compatible parts
            # 3D 'frame_center' maps to 2D 'center'
            center = state.get("center", state.get("frame_center", ORIGIN))

            # 3D 'zoom' maps back to 2D 'width'
            # width = default_width / zoom
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

    def save_state(self, state_name):
        state_data = {
            "mobjects": [],
            "fixed_in_frame_mobjects": [],
            "foreground_mobjects": [],
            "camera": self._get_camera_state(),
        }

        # Get all mobjects (main list)
        for mob in self.mobjects:
            if mob == self.logo:
                continue
            try:
                # Make a copy and clear its updaters
                mob_copy = mob.copy()
                mob_copy.clear_updaters()

                # Recursively clear updaters from submobjects
                for submob in mob_copy.get_family():
                    submob.clear_updaters()

                state_data["mobjects"].append(mob_copy)
            except Exception as e:
                print(f"Warning: Could not copy mobject {mob}: {e}")

        # Also get foreground mobjects if they exist
        if hasattr(self, "foreground_mobjects"):
            for mob in self.foreground_mobjects:
                try:
                    state_data["foreground_mobjects"].append(mob.copy())
                except Exception as e:
                    print(f"Warning: Could not copy foreground mobject {mob}: {e}")

        if hasattr(self, "fixed_in_frame_mobjects"):
            for mob in self.camera.fixed_in_frame_mobjects:
                try:
                    state_data["fixed_in_frame_mobjects"].append(mob.copy())
                except Exception as e:
                    print(f"Warning: Could not copy fixed_in_frame mobject {mob}: {e}")

        # Save to file
        filepath = os.path.join(self.state_dir, f"{state_name}.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(state_data, f)

        print(f"State saved to {filepath} ({len(state_data['mobjects'])} mobjects)")

    def load_state(self, state_name):
        filepath = os.path.join(self.state_dir, f"{state_name}.pkl")

        if not os.path.exists(filepath):
            print(f"Warning: State file {filepath} not found. Skipping load.")
            return

        with open(filepath, "rb") as f:
            state_data = pickle.load(f)

        # Add all mobjects to the scene without animation
        for mob in state_data["mobjects"]:
            self.add(mob)

        if "fixed_in_frame_mobjects" in state_data:
            for mob in state_data["fixed_in_frame_mobjects"]:
                self.add_fixed_in_frame_mobjects(mob)

        # Add foreground mobjects if they exist
        if "foreground_mobjects" in state_data:
            for mob in state_data["foreground_mobjects"]:
                self.add_foreground_mobject(mob)

        # Restore camera position if saved
        if "camera" in state_data and state_data["camera"] is not None:
            self._set_camera_state(state_data["camera"], animate=False)
        elif state_data.get("camera_position") is not None and hasattr(
            self.camera, "frame"
        ):
            self.camera.frame.move_to(state_data["camera_position"])

        print(f"State loaded from {filepath} ({len(state_data['mobjects'])} mobjects)")

    def new_section(self, section_name, next_slide_prep=None, pause_at_title=True):

        self.marked_next_slide()

        # Create gradient background
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
        section_slide = VGroup(background, section_title).move_to(
            UP * config.frame_height
        )

        # Animate sliding down
        self.add_fixed_in_frame_mobjects(section_slide)
        self.play(
            section_slide.animate.shift(DOWN * config.frame_height),
            run_time=1,
        )

        if pause_at_title:
            self.next_slide()
        else:
            self.wait()

        if next_slide_prep is None:
            self.clear()
            add_logo_to_background(self)
        else:
            next_slide_prep()

        # Animate sliding up
        self.play(
            section_slide.animate.shift(UP * config.frame_height),
            run_time=1,
        )

    def start_section_wipe(self, section_name):
        self.marked_next_slide()

        # Create gradient background
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
        # section_slide = VGroup(background, section_title).move_to(
        #     UP * config.frame_height
        # )

        # # Animate sliding down
        # self.play(
        #     section_slide.animate.shift(DOWN * config.frame_height),
        #     run_time=1,
        # )
        # self.wait(0.1)
        section_slide = VGroup(background, section_title)
        self.add_fixed_in_frame_mobjects(section_slide)

        # 3. Position in screen-space (relative to the camera's "lens")
        # Even if panned, UP * frame_height now refers to the screen's top
        section_slide.move_to(UP * config.frame_height)

        # 4. Animate sliding down within the fixed frame
        self.play(
            section_slide.animate.move_to(ORIGIN),
            run_time=1,
        )
        self.wait(0.1)

    def end_section_wipe(self, section_name, next_slide_prep=None):
        # Create gradient background
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
        section_slide = VGroup(background, section_title)

        self.add(section_slide)
        self.wait(0.1)

        self.next_slide()
        if next_slide_prep is None:
            self.clear()
            add_logo_to_background(self)
        else:
            next_slide_prep()

        # Animate sliding up
        self.play(
            section_slide.animate.shift(UP * config.frame_height),
            run_time=1,
        )

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

        # Try to fit text with the given font size
        current_font_size = font_size
        text_group = None

        # Iteratively reduce font size until text fits
        for attempt in range(10):  # Max 10 attempts
            # Create lines with current font size
            lines = self._wrap_text_to_lines(
                text, current_font_size, max_width_pixels, t2c, **text_kwargs
            )

            # Position lines manually using baseline spacing
            text_group = self._position_lines_by_baseline(
                lines, current_font_size, line_spacing
            )

            # Check if it fits within bounds
            if (
                text_group.width <= max_width_pixels
                and text_group.height <= max_height_pixels
            ):
                break

            # Reduce font size and try again
            current_font_size *= 0.9

            if current_font_size < 16:  # Minimum font size
                print("Warning: Text may not fit well even at minimum size")
                break

        # Center the text
        if location is ORIGIN:
            text_group.move_to(ORIGIN)
        elif location is UP:
            text_group.to_edge(UP)
        elif location is DOWN:
            text_group.to_edge(DOWN)

        # Add to scene
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
        """
        Position text lines using consistent baseline spacing instead of bounding boxes.

        Args:
            lines: List of Text objects
            font_size: Font size being used
            line_spacing: Multiplier for line spacing (1.0 = single, 1.5 = 1.5x, etc.)

        Returns:
            VGroup with properly positioned lines
        """
        if not lines:
            return VGroup()

        # Calculate line height based on font size
        # This is the distance between baselines
        baseline_distance = (
            font_size * line_spacing * 0.015
        )  # Adjust multiplier as needed

        text_group = VGroup(*lines)

        # Position first line at the top
        current_y = 0

        for i, line in enumerate(lines):
            if i == 0:
                # First line stays at origin, we'll center the whole group later
                line.move_to(UP * current_y)
            else:
                # Move each subsequent line down by baseline_distance
                current_y -= baseline_distance
                line.move_to(UP * current_y)

        return text_group

    def _wrap_text_to_lines(self, text, font_size, max_width, t2c=None, **text_kwargs):
        """
        Internal method to wrap text into lines that fit within max_width.

        Returns:
            List of Text objects, one per line
        """
        # First, try to split by existing newlines
        paragraphs = text.split("\n")
        all_lines = []

        for paragraph in paragraphs:
            if not paragraph.strip():
                # For blank lines, add a small invisible placeholder to maintain spacing
                all_lines.append(Text(" ", font_size=font_size, **text_kwargs))
                continue

            # Estimate characters per line based on font size
            # This is approximate - Manim doesn't give us exact metrics before rendering
            chars_per_line = int(max_width / font_size)
            print(
                f"Initial estimated chars per line of {chars_per_line} at font size {font_size}"
            )
            print(f"Max width: {max_width}")
            chars_per_line = max(chars_per_line, 16)  # Minimum 16 chars
            print(
                f"Estimated chars per line of {chars_per_line} at font size {font_size}"
            )

            # Wrap the paragraph
            wrapped_lines = wrap(
                paragraph, width=chars_per_line, break_long_words=False
            )

            # Create Text objects for each line
            for line in wrapped_lines:
                text_obj = Text(line, font_size=font_size, t2c=t2c, **text_kwargs)

                # If this single line is still too wide, try to force break it
                if text_obj.width > max_width:
                    # Force break into smaller chunks
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
        self.play(
            *[animation(mobject) for mobject in self.mobjects if mobject != self.logo],
            run_time=run_time,
        )
