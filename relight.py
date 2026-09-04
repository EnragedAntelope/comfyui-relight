import logging
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

# SciPy is required for rim lighting, occlusion smoothing and mask blur
from scipy import ndimage

# Import the ComfyUI v3 node API. `comfy_api.v0_0_2` is a stable alias for the
# same surface; fall back to `latest` on builds that predate it, and fail with a
# readable message rather than a bare traceback if neither exists.
try:
    from comfy_api.v0_0_2 import ComfyExtension, io
except ImportError:
    try:
        from comfy_api.latest import ComfyExtension, io
    except ImportError as exc:  # pragma: no cover - depends on the ComfyUI build
        raise ImportError(
            "ComfyUI-ReLight requires the ComfyUI v3 node API (comfy_api), introduced in "
            "ComfyUI 0.3.48. Please update ComfyUI."
        ) from exc

logger = logging.getLogger("ComfyUI-ReLight")

# PIL's 'L' conversion weights, kept so contrast/saturation match the look this
# node had when those operations went through ImageEnhance.
_LUMA_WEIGHTS = (0.299, 0.587, 0.114)

# Shared by the gamma widgets' min/max and the strength-scaling clamp in
# _scaled_gamma, so the two can never drift apart.
GAMMA_MIN = 0.1
GAMMA_MAX = 5.0

# Single-entry pixel-coordinate cache; every light in a run shares one grid.
#
# Module level, deliberately: ComfyUI's v3 runtime does not call execute() on
# this class. It calls it on a *locked clone* whose metaclass raises
# AttributeError on any class-attribute write, so `cls._coord_cache = ...` blew
# up on the first mask of every run. Never store per-run state on the class.
_COORD_CACHE = {}

# The cast-shadow trace runs at or below this resolution and is upsampled.
# The result is blurred immediately afterwards, so tracing a 4K frame at full
# size buys nothing visible and costs 60x the work of tracing it at 512.
_SHADOW_TRACE_MAX = 512


def _supports_advanced():
    """Whether this ComfyUI build accepts `advanced=` on widget inputs.

    The flag arrived long after the v3 API itself. Passing it to an older build
    raises TypeError at import time, which takes the whole node pack down, so
    probe for it instead of assuming.
    """
    try:
        io.Boolean.Input("_relight_probe", default=False, advanced=True)
    except TypeError:
        return False
    return True


_ADVANCED_SUPPORTED = _supports_advanced()


def _wrap_debug_text(draw, text, font, max_width):
    """Greedy word wrap for the debug placeholder, in pixels not characters."""
    words, lines, line = text.split(), [], ""
    for word in words:
        candidate = f"{line} {word}".strip()
        try:
            too_wide = draw.textlength(candidate, font=font) > max_width
        except Exception:
            too_wide = len(candidate) * 7 > max_width
        if too_wide and line:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines


def _debug_font_size(height):
    """Type size for the debug overlays, as a fraction of the frame height.

    Fixed 13px type is legible on the 96x64 fixture the tests used and invisible
    on a real 1300px render - 1.7% of the height, which inside a ComfyUI preview
    thumbnail is a dark rectangle. That is exactly how the v3.1.2 placeholder
    shipped "working" and was reported as a black frame. Scale with the frame,
    with a floor so small images stay readable and a cap so a 4K render does not
    get billboard type.
    """
    return int(min(64, max(13, round(height * 0.035))))


def _load_debug_font(size):
    """The default font at `size`, falling back through older Pillow APIs."""
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        try:
            return ImageFont.load_default()
        except Exception as err:  # pragma: no cover - no usable font at all
            logger.debug(f"  Could not load a default font: {err}")
            return None


def _adv(**kwargs):
    """Pass input kwargs through, dropping `advanced` where unsupported."""
    if not _ADVANCED_SUPPORTED:
        kwargs.pop("advanced", None)
    return kwargs


class ReLight(io.ComfyNode):
    """
    Creates realistic lighting effects by applying color corrections or colored light
    to distinct areas of an image. Supports multiple light sources, colored lights,
    and 3D lighting simulation with subject occlusion. Requires SciPy.
    """

    # --- Mode vocabularies -------------------------------------------------
    # v4.0.0 replaced four booleans (use_colored_lights, use_gradient_mode,
    # apply_3d_lighting, show_debug_info) with these named choices. The option
    # strings are part of the saved-workflow format and are mirrored in
    # web/relight_migrate.js - change one and you must change the other.
    MODE_CORRECTION = "Color Correction"
    MODE_COLORED = "Colored Light"
    MODE_BOTH = "Both"
    LIGHTING_MODES = (MODE_CORRECTION, MODE_COLORED, MODE_BOTH)

    SHAPE_RADIAL = "Radial falloff"
    SHAPE_GRADIENT = "Directional gradient"
    MASK_SHAPES = (SHAPE_RADIAL, SHAPE_GRADIENT)

    SUBJECT_NONE = "None"
    SUBJECT_FRONT = "Light in front of subject"
    SUBJECT_RIM = "Light behind subject (rim)"
    SUBJECT_INTERACTIONS = (SUBJECT_NONE, SUBJECT_FRONT, SUBJECT_RIM)

    # Gamma follows the display convention used by Photoshop's Levels midtone
    # slider, ImageMagick's -gamma and ffmpeg's eq filter: output = input^(1/gamma),
    # so values above 1.0 brighten and values below 1.0 darken. Preset gammas below
    # are written in those terms - a shaded outer zone gets a gamma under 1.0.
    PRESETS = {
        "None": {},
        "Soft Window Light": {
            "light_position_x": 0.7, "light_position_y": 0.3, "inner_circle_radius": 0.45, "outer_circle_radius": 0.8,
            "inner_brightness": 10, "inner_contrast": 0, "inner_saturation": 0, "inner_temperature": -5, "inner_tint": 0, "inner_gamma": 1.0,
            "outer_brightness": -15, "outer_contrast": 0, "outer_saturation": -10, "outer_temperature": -5, "outer_tint": 0, "outer_gamma": 0.83,
            "mask_blur": 80, "rim_amplification": 1.0
        },
        "Dramatic Side Light": {
            "light_position_x": 0.1, "light_position_y": 0.5, "inner_circle_radius": 0.3, "outer_circle_radius": 0.6,
            "inner_brightness": 15, "inner_contrast": 10, "inner_saturation": 10, "inner_temperature": 0, "inner_tint": 0, "inner_gamma": 1.11,
            "outer_brightness": -30, "outer_contrast": 10, "outer_saturation": -20, "outer_temperature": 0, "outer_tint": 0, "outer_gamma": 0.77,
            "mask_blur": 50, "rim_amplification": 1.0
        },
        "Warm Sunset Glow": {
            "light_position_x": 0.9, "light_position_y": 0.4, "inner_circle_radius": 0.5, "outer_circle_radius": 0.8,
            "inner_brightness": 3, "inner_contrast": 5, "inner_saturation": 10, "inner_temperature": 25, "inner_tint": -5, "inner_gamma": 1.05,
            "outer_brightness": -10, "outer_contrast": 0, "outer_saturation": -5, "outer_temperature": 15, "outer_tint": -5, "outer_gamma": 0.91,
            "mask_blur": 75, "lighting_mode": MODE_BOTH, "light_color_r": 255, "light_color_g": 200, "light_color_b": 120,
            "light_intensity": 0.45, "rim_amplification": 1.0, "mask_shape": SHAPE_GRADIENT
        },
        "Cool Blue Moonlight": {
            "light_position_x": 0.8, "light_position_y": 0.2, "inner_circle_radius": 0.4, "outer_circle_radius": 0.7,
            "inner_brightness": -5, "inner_contrast": 5, "inner_saturation": -5, "inner_temperature": -20, "inner_tint": 0, "inner_gamma": 0.91,
            "outer_brightness": -20, "outer_contrast": 0, "outer_saturation": -10, "outer_temperature": -30, "outer_tint": 0, "outer_gamma": 0.83,
            "mask_blur": 60, "lighting_mode": MODE_BOTH, "light_color_r": 120, "light_color_g": 150, "light_color_b": 255,
            "light_intensity": 0.35, "rim_amplification": 1.0
        },
        "Studio Key Light": {
            "light_position_x": 0.4, "light_position_y": 0.3, "inner_circle_radius": 0.6, "outer_circle_radius": 0.9,
            "inner_brightness": 12, "inner_contrast": 5, "inner_saturation": 0, "inner_temperature": 0, "inner_tint": 0, "inner_gamma": 1.0,
            "outer_brightness": -5, "outer_contrast": 0, "outer_saturation": -5, "outer_temperature": 0, "outer_tint": 0, "outer_gamma": 0.91,
            "mask_blur": 90, "rim_amplification": 1.0
        },
        "Rim Light (Behind)": {
            "light_position_x": 0.5, "light_position_y": 0.1, "inner_circle_radius": 0.3, "outer_circle_radius": 0.6,
            "subject_interaction": SUBJECT_RIM, "lighting_mode": MODE_BOTH,
            # Neutral white. This shipped as (200, 255, 200) from v1.0, which put a
            # +20/255 green cast on the rim and the background glow - invisible while
            # the grading block was being discarded, obvious once inner_saturation
            # landed. Intensity drops 1.2 -> 1.0 so removing the tint does not also
            # make the preset brighter: mean lift stays within 2% of what shipped.
            "light_color_r": 255, "light_color_g": 255, "light_color_b": 255, "light_intensity": 1.0,
            "inner_brightness": 0, "inner_contrast": 8, "inner_saturation": 12, "inner_temperature": 0, "inner_tint": 0, "inner_gamma": 1.0,
            "outer_brightness": 0, "outer_contrast": 0, "outer_saturation": 0, "outer_temperature": 0, "outer_tint": 0, "outer_gamma": 1.0,
            "mask_blur": 25, "effect_strength": 1.5, "rim_amplification": 2.5,
            "shadow_strength": 0.7, "shadow_length": 0.45
        },
        "Spotlight": {
            "light_position_x": 0.5, "light_position_y": 0.4, "inner_circle_radius": 0.1, "outer_circle_radius": 0.25,
            "inner_brightness": 25, "inner_contrast": 15, "inner_saturation": -5, "inner_temperature": 0, "inner_tint": 0, "inner_gamma": 1.11,
            "outer_brightness": -40, "outer_contrast": 10, "outer_saturation": -20, "outer_temperature": 0, "outer_tint": 0, "outer_gamma": 0.77,
            "mask_blur": 30, "rim_amplification": 1.0, "effect_strength": 1.2
        },
        "Negative Light (Darken)": {
            "light_position_x": 0.5, "light_position_y": 0.5, "inner_circle_radius": 0.4, "outer_circle_radius": 0.7,
            "inner_brightness": -20, "inner_contrast": 5, "inner_saturation": -5, "inner_temperature": 0, "inner_tint": 0, "inner_gamma": 0.91,
            "outer_brightness": 0, "outer_contrast": 0, "outer_saturation": 0, "outer_temperature": 0, "outer_tint": 0, "outer_gamma": 1.0,
            "mask_blur": 60, "rim_amplification": 1.0, "effect_strength": 1.0
        }
    }

    # Three presets ("Warm Sunset Glow", "Cool Blue Moonlight", "Rim Light
    # (Behind)") set a colour *and* a grading block. Up to v3.1.2 the two were
    # mutually exclusive, so 12 values in each did nothing; they now run as
    # MODE_BOTH, and the colour intensities were pulled down to compensate for
    # the grade no longer being discarded.

    # Scaled by the user's widget rather than overridden outright; see _load_preset.
    STRENGTH_KEY = "effect_strength"

    # Keys a preset may not touch while `preserve_positioning` is on.
    GEOMETRY_KEYS = frozenset({
        "light_position_x", "light_position_y", "inner_circle_radius", "outer_circle_radius",
        "light2_position_x", "light2_position_y", "light2_inner_radius", "light2_outer_radius",
        "light3_position_x", "light3_position_y", "light3_inner_radius", "light3_outer_radius",
    })

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Define node schema for ComfyUI v3.

        Widget order is grouped for reading, not frozen: saved workflows store
        widget values *positionally*, and what keeps pre-v4 files loading is
        ``web/relight_migrate.js``, which remaps them by name. Any change to
        this list - adding, removing, renaming or reordering - must be paired
        with a check that the migration still produces the right mapping, and
        with the legacy order pinned in ``tests/test_relight.py``.
        """
        return io.Schema(
            node_id="ReLight",
            display_name="ReLight 💡",
            category="image/lighting",
            description="Creates realistic lighting effects with multiple light sources, colored lights, and 3D lighting simulation with subject occlusion",
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
            inputs=[
                # --- Core Inputs ---
                io.Image.Input("image", tooltip="The input image to apply lighting effects to"),
                io.Mask.Input("mask", optional=True, tooltip="Foreground mask (white=subject, black=background). Needed for subject interaction (in front of / behind the subject) and for remove_background compositing. Resized automatically if it does not match the image"),

                # --- Preset ---
                io.Combo.Input("preset", options=list(cls.PRESETS.keys()), default="None", tooltip="Select a preset or None for custom settings. NOTE: a preset overrides the widgets below - the values shown on the node are ignored for whatever the preset defines. The one exception is effect_strength, which scales the preset instead of being replaced by it"),
                io.Boolean.Input("preserve_positioning", **_adv(default=False, advanced=True, tooltip="Keep your own light positions and radii when a preset is selected, instead of letting the preset set them")),

                # --- Mode ---
                io.Combo.Input("lighting_mode", options=list(cls.LIGHTING_MODES), default=cls.MODE_CORRECTION, tooltip="What the light does. Color Correction grades an inner zone and the ring around it. Colored Light adds coloured light on top of the image. Both applies the coloured light first and then the grade"),
                io.Combo.Input("mask_shape", options=list(cls.MASK_SHAPES), default=cls.SHAPE_RADIAL, tooltip="Shape of the light mask. Radial falloff is a lamp; Directional gradient is light arriving from one side (sunset rays, window light)"),

                # --- Subject interaction ---
                io.Combo.Input("subject_interaction", options=list(cls.SUBJECT_INTERACTIONS), default=cls.SUBJECT_NONE, tooltip="How the light interacts with the masked subject. None lights the whole frame evenly. The other two need a mask connected"),
                io.Boolean.Input("remove_background", default=False, tooltip="Composite the lit result back over the untouched original using the mask, so only the subject is relit. Does not remove anything. Ignored when the light is in front of or behind the subject"),

                # --- Global modifiers ---
                io.Int.Input("num_light_sources", default=1, min=1, max=3, step=1, tooltip="Number of light sources (1-3). Lights 2 and 3 have their own position, radius and color, but in color-correction mode they reuse Light 1 correction settings"),
                io.Float.Input("effect_strength", default=1.0, min=0.0, max=5.0, step=0.1, tooltip="Overall intensity multiplier for lighting adjustments/colors, gamma included. 0.0 leaves the image untouched. With a preset active this scales the preset own strength, so 1.0 is the preset as designed. Does not scale rim_amplification or mask_blur - those have their own controls"),
                io.Float.Input("mask_blur", default=50.0, min=0.0, max=200.0, step=1.0, tooltip="Blur radius for light mask edges (smoother transitions)"),
                io.Float.Input("rim_amplification", default=2.0, min=0.0, max=10.0, step=0.1, tooltip="Intensity boost for the rim highlight along the subject edge. Only used when the light is behind the subject"),
                io.Float.Input("shadow_strength", default=0.6, min=0.0, max=1.0, step=0.05, tooltip="How dark the shadow the subject casts across the background is. 0.0 casts no shadow. Only used when the light is behind the subject"),
                io.Float.Input("shadow_length", default=0.35, min=0.0, max=1.0, step=0.01, tooltip="How far the cast shadow reaches, as a fraction of the image shorter side. Only used when the light is behind the subject"),
                io.Boolean.Input("debug_output_connected", **_adv(default=False, optional=True, advanced=True, tooltip="Managed by ReLight itself - the node UI hides this and keeps it in step with whether the debug_image output is wired to anything. It exists because a widget value is what ComfyUI hashes to decide a node needs re-running, so without it, wiring the debug output after a run would just replay the cached placeholder. Nothing to set by hand")),

                # --- Light 1: position and shape ---
                io.Float.Input("light_position_x", default=0.5, min=0.0, max=1.0, step=0.01, tooltip="Light 1: Horizontal position (0=left, 1=right)"),
                io.Float.Input("light_position_y", default=0.5, min=0.0, max=1.0, step=0.01, tooltip="Light 1: Vertical position (0=top, 1=bottom)"),
                io.Float.Input("inner_circle_radius", default=0.4, min=0.0, max=1.0, step=0.01, tooltip="Light 1: Inner radius (strongest effect area)"),
                io.Float.Input("outer_circle_radius", default=0.7, min=0.0, max=1.0, step=0.01, tooltip="Light 1: Outer radius (falloff area)"),

                # --- Light 1: colour (Colored Light / Both) ---
                io.Int.Input("light_color_r", default=255, min=0, max=255, step=1, tooltip="Light 1: Red color (Colored Light / Both)"),
                io.Int.Input("light_color_g", default=255, min=0, max=255, step=1, tooltip="Light 1: Green color (Colored Light / Both)"),
                io.Int.Input("light_color_b", default=255, min=0, max=255, step=1, tooltip="Light 1: Blue color (Colored Light / Both)"),
                io.Float.Input("light_intensity", default=1.0, min=0.0, max=3.0, step=0.1, tooltip="Light 1: Intensity (Colored Light / Both)"),

                # --- Light 1: grading, inner zone (Color Correction / Both) ---
                io.Float.Input("inner_brightness", default=10.0, min=-100.0, max=100.0, step=1.0, tooltip="Light 1: Inner area brightness (Color Correction / Both)"),
                io.Float.Input("inner_contrast", default=5.0, min=-100.0, max=100.0, step=1.0, tooltip="Light 1: Inner area contrast (Color Correction / Both)"),
                io.Float.Input("inner_saturation", default=5.0, min=-100.0, max=100.0, step=1.0, tooltip="Light 1: Inner area saturation (Color Correction / Both)"),
                io.Float.Input("inner_temperature", default=0.0, min=-100.0, max=100.0, step=1.0, tooltip="Light 1: Inner area temperature (-100=cool, 100=warm)"),
                io.Float.Input("inner_tint", default=0.0, min=-100.0, max=100.0, step=1.0, tooltip="Light 1: Inner area tint (-100=magenta, 100=green)"),
                io.Float.Input("inner_gamma", default=1.0, min=GAMMA_MIN, max=GAMMA_MAX, step=0.05, tooltip="Light 1: Inner area gamma. Above 1.0 brightens midtones, below 1.0 darkens them"),

                # --- Light 1: grading, outer ring (Color Correction / Both) ---
                io.Float.Input("outer_brightness", **_adv(default=-10.0, min=-100.0, max=100.0, step=1.0, advanced=True, tooltip="Light 1: Outer area brightness (Color Correction / Both)")),
                io.Float.Input("outer_contrast", **_adv(default=0.0, min=-100.0, max=100.0, step=1.0, advanced=True, tooltip="Light 1: Outer area contrast (Color Correction / Both)")),
                io.Float.Input("outer_saturation", **_adv(default=-10.0, min=-100.0, max=100.0, step=1.0, advanced=True, tooltip="Light 1: Outer area saturation (Color Correction / Both)")),
                io.Float.Input("outer_temperature", **_adv(default=0.0, min=-100.0, max=100.0, step=1.0, advanced=True, tooltip="Light 1: Outer area temperature")),
                io.Float.Input("outer_tint", **_adv(default=0.0, min=-100.0, max=100.0, step=1.0, advanced=True, tooltip="Light 1: Outer area tint")),
                io.Float.Input("outer_gamma", **_adv(default=0.91, min=GAMMA_MIN, max=GAMMA_MAX, step=0.05, advanced=True, tooltip="Light 1: Outer area gamma. Above 1.0 brightens midtones, below 1.0 darkens them")),

                # --- Light 2 Settings (Optional) ---
                io.Float.Input("light2_position_x", **_adv(default=0.8, min=0.0, max=1.0, step=0.01, optional=True, advanced=True, tooltip="Light 2: Horizontal position")),
                io.Float.Input("light2_position_y", **_adv(default=0.2, min=0.0, max=1.0, step=0.01, optional=True, advanced=True, tooltip="Light 2: Vertical position")),
                io.Float.Input("light2_inner_radius", **_adv(default=0.3, min=0.0, max=1.0, step=0.01, optional=True, advanced=True, tooltip="Light 2: Inner radius")),
                io.Float.Input("light2_outer_radius", **_adv(default=0.6, min=0.0, max=1.0, step=0.01, optional=True, advanced=True, tooltip="Light 2: Outer radius")),
                io.Int.Input("light2_color_r", **_adv(default=180, min=0, max=255, step=1, optional=True, advanced=True, tooltip="Light 2: Red color")),
                io.Int.Input("light2_color_g", **_adv(default=180, min=0, max=255, step=1, optional=True, advanced=True, tooltip="Light 2: Green color")),
                io.Int.Input("light2_color_b", **_adv(default=255, min=0, max=255, step=1, optional=True, advanced=True, tooltip="Light 2: Blue color")),
                io.Float.Input("light2_intensity", **_adv(default=0.7, min=0.0, max=3.0, step=0.1, optional=True, advanced=True, tooltip="Light 2: Intensity (Colored Light / Both)")),

                # --- Light 3 Settings (Optional) ---
                io.Float.Input("light3_position_x", **_adv(default=0.3, min=0.0, max=1.0, step=0.01, optional=True, advanced=True, tooltip="Light 3: Horizontal position")),
                io.Float.Input("light3_position_y", **_adv(default=0.8, min=0.0, max=1.0, step=0.01, optional=True, advanced=True, tooltip="Light 3: Vertical position")),
                io.Float.Input("light3_inner_radius", **_adv(default=0.25, min=0.0, max=1.0, step=0.01, optional=True, advanced=True, tooltip="Light 3: Inner radius")),
                io.Float.Input("light3_outer_radius", **_adv(default=0.5, min=0.0, max=1.0, step=0.01, optional=True, advanced=True, tooltip="Light 3: Outer radius")),
                io.Int.Input("light3_color_r", **_adv(default=255, min=0, max=255, step=1, optional=True, advanced=True, tooltip="Light 3: Red color")),
                io.Int.Input("light3_color_g", **_adv(default=150, min=0, max=255, step=1, optional=True, advanced=True, tooltip="Light 3: Green color")),
                io.Int.Input("light3_color_b", **_adv(default=120, min=0, max=255, step=1, optional=True, advanced=True, tooltip="Light 3: Blue color")),
                io.Float.Input("light3_intensity", **_adv(default=0.5, min=0.0, max=3.0, step=0.1, optional=True, advanced=True, tooltip="Light 3: Intensity (Colored Light / Both)")),
            ],
            outputs=[
                io.Image.Output("image", display_name="image", tooltip="The relit image"),
                io.Mask.Output("mask", display_name="mask", tooltip="Pass-through of the input mask, normalised to (batch, height, width) and resized to the image (black if none connected)"),
                io.Image.Output("debug_image", display_name="debug_image", tooltip="Visualization of light positions and masks. Drawn whenever this output is connected to something - there is no toggle to remember"),
            ]
        )

    # --- Utility Functions ---

    @classmethod
    def _load_preset(cls, preset_name, current_params):
        """Loads preset values, respecting 'preserve_positioning'."""
        if preset_name == "None" or preset_name not in cls.PRESETS:
            return current_params
        logger.debug(f"Applying preset: {preset_name}")
        updated_params = current_params.copy()
        preserve = current_params.get("preserve_positioning", False)
        for key, value in cls.PRESETS[preset_name].items():
            if key not in updated_params:
                continue
            if preserve and key in cls.GEOMETRY_KEYS:
                continue
            if key == cls.STRENGTH_KEY:
                # The master intensity stays under the user's control: the preset
                # supplies a baseline and the widget scales it. Overriding it
                # outright left effect_strength - and its documented "0.0 leaves
                # the image untouched" - dead under the presets that set it.
                updated_params[key] = value * current_params.get(key, 1.0)
                continue
            updated_params[key] = value
        if preserve:
            logger.debug("  - Preserving user-defined light positions and radii.")
        return updated_params

    @classmethod
    def _pixel_grid(cls, width, height):
        """Row/column coordinate grids for an image size, reused across lights.

        Broadcast grids - (height, 1) and (1, width) - rather than two full
        (height, width) arrays. Every expression here combines the two, so numpy
        materialises the same result, but the cached grids stay kilobytes instead
        of megabytes: a 4096x4096 image used to pin 268 MB for the life of the
        process, and each mask cost three full-size temporaries instead of one.
        """
        key = (height, width)
        cached = _COORD_CACHE.get(key)
        if cached is None:
            cached = np.ogrid[0:height, 0:width]
            # Single-entry cache: a workflow rarely alternates resolutions, and this
            # keeps grids from accumulating. Mutated in place - rebinding the name
            # would need a `global` and buys nothing.
            _COORD_CACHE.clear()
            _COORD_CACHE[key] = cached
        return cached

    @classmethod
    def create_circle_mask(cls, width, height, center_x, center_y, radius):
        """Create a circular mask using NumPy."""
        y_coords, x_coords = cls._pixel_grid(width, height)
        center_x_px, center_y_px = center_x * width, center_y * height
        radius_px = radius * min(width, height)
        if radius_px <= 0:
            return torch.zeros((height, width), dtype=torch.float32)
        dist_sq = (x_coords - center_x_px)**2 + (y_coords - center_y_px)**2
        mask = (dist_sq <= radius_px**2).astype(np.float32)
        return torch.from_numpy(mask)

    @classmethod
    def create_falloff_mask(cls, width, height, center_x, center_y, inner_radius, outer_radius):
        """Create a radial mask: 1.0 inside inner radius, linear falloff to 0.0 at outer radius."""
        y_coords, x_coords = cls._pixel_grid(width, height)
        center_x_px, center_y_px = center_x * width, center_y * height
        inner_px = max(inner_radius, 0.0) * min(width, height)
        outer_px = max(outer_radius, 0.0) * min(width, height)
        if outer_px <= 0:
            return torch.zeros((height, width), dtype=torch.float32)
        distances = np.sqrt((x_coords - center_x_px)**2 + (y_coords - center_y_px)**2)
        if outer_px <= inner_px:
            # Degenerate ring: hard circle at outer radius
            mask = (distances <= outer_px).astype(np.float32)
        else:
            mask = np.clip((outer_px - distances) / (outer_px - inner_px), 0.0, 1.0).astype(np.float32)
        return torch.from_numpy(mask)

    @classmethod
    def create_gradient_mask(cls, width, height, center_x, center_y, radius, direction_angle_deg=0):
        """Create a gradient mask with direction using NumPy."""
        y_coords, x_coords = cls._pixel_grid(width, height)
        center_x_px, center_y_px = center_x * width, center_y * height
        radius_px = radius * min(width, height)
        if radius_px <= 0:
            return torch.zeros((height, width), dtype=torch.float32)
        delta_x, delta_y = x_coords - center_x_px, y_coords - center_y_px
        distances = np.sqrt(delta_x**2 + delta_y**2)
        theta_rad = math.radians(direction_angle_deg)
        dir_x, dir_y = math.cos(theta_rad), math.sin(theta_rad)
        norm_dist = np.where(distances == 0, 1, distances)
        pos_norm_x, pos_norm_y = delta_x / norm_dist, delta_y / norm_dist
        directional_component = pos_norm_x * dir_x + pos_norm_y * dir_y
        gradient_intensity = (directional_component + 1) / 2
        falloff = np.clip(1 - distances / radius_px, 0, 1)
        mask = np.where(distances <= radius_px, falloff * gradient_intensity, 0).astype(np.float32)
        return torch.from_numpy(mask)

    @classmethod
    def _luma(cls, image_tensor):
        """Per-pixel luminance, keeping the channel dimension."""
        if image_tensor.shape[-1] == 1:
            return image_tensor
        weights = torch.tensor(_LUMA_WEIGHTS, device=image_tensor.device, dtype=image_tensor.dtype)
        return (image_tensor[..., :3] * weights).sum(dim=-1, keepdim=True)

    @classmethod
    def apply_color_correction(cls, image_tensor, brightness=0, contrast=0, saturation=0, temperature=0, tint=0, gamma=1.0):
        """Apply color correction adjustments in float32, on whatever device the image is on.

        Mirrors the PIL ImageEnhance semantics this node used previously (same
        luma weights, same blend-toward-degenerate maths, same per-stage clamp to
        0-1) without round-tripping through uint8. That round-trip made even an
        all-identity correction shift pixels by 1/255, and the error compounded
        across light sources and across chained ReLight nodes.
        """
        result = image_tensor
        is_single_channel = result.shape[-1] == 1

        if abs(brightness) > 0.1:
            result = (result * (1.0 + brightness / 100.0)).clamp(0.0, 1.0)

        if abs(contrast) > 0.1:
            # PIL blends against a flat grey of the image's mean luminance.
            mean = cls._luma(result).mean(dim=(1, 2, 3), keepdim=True)
            result = (mean + (result - mean) * (1.0 + contrast / 100.0)).clamp(0.0, 1.0)

        if abs(saturation) > 0.1 and not is_single_channel:
            gray = cls._luma(result)
            result = (gray + (result - gray) * (1.0 + saturation / 100.0)).clamp(0.0, 1.0)

        if (abs(temperature) > 0.1 or abs(tint) > 0.1) and not is_single_channel:
            gains = torch.tensor(
                [1.0 + temperature / 200.0, 1.0 + tint / 200.0, 1.0 - temperature / 200.0],
                device=result.device, dtype=result.dtype,
            )
            result = (result * gains).clamp(0.0, 1.0)

        if abs(gamma - 1.0) > 0.01:
            result = result.clamp(0.0, 1.0) ** (1.0 / max(gamma, 0.01))

        return result.clamp(0.0, 1.0)

    @staticmethod
    def _gaussian_kernel_1d(radius, sigma, device):
        offsets = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
        kernel = torch.exp(-(offsets * offsets) / (2.0 * sigma * sigma))
        return kernel / kernel.sum()

    @classmethod
    def apply_mask_blur(cls, mask_tensor, blur_amount):
        """Gaussian-blur a mask as a separable float32 convolution.

        This used to go through a uint8 PIL image, which quantised the mask to
        255 levels - visible as banding in smooth falloffs, since the mask *is*
        the effect here. Doing it in torch keeps full precision and, on a GPU
        workflow, avoids a device round-trip for every light source. The
        blur_amount -> sigma mapping is unchanged so existing workflows look the
        same.
        """
        if blur_amount <= 0.1:
            return mask_tensor
        sigma = blur_amount / 5.0
        if sigma <= 0:
            return mask_tensor

        was_2d = mask_tensor.dim() == 2
        work = mask_tensor.unsqueeze(0) if was_2d else mask_tensor
        work = work.to(dtype=torch.float32).unsqueeze(1)  # (batch, 1, height, width)
        height, width = work.shape[-2:]
        device = work.device

        # Truncate at 3 sigma, and never wider than the axis being blurred -
        # reflect padding cannot exceed the dimension it mirrors.
        radius = max(1, int(3.0 * sigma + 0.5))
        radius_x, radius_y = min(radius, width - 1), min(radius, height - 1)

        if radius_x > 0:
            kernel_x = cls._gaussian_kernel_1d(radius_x, sigma, device)
            work = F.conv2d(
                F.pad(work, (radius_x, radius_x, 0, 0), mode="reflect"),
                kernel_x.view(1, 1, 1, -1),
            )
        if radius_y > 0:
            kernel_y = cls._gaussian_kernel_1d(radius_y, sigma, device)
            work = F.conv2d(
                F.pad(work, (0, 0, radius_y, radius_y), mode="reflect"),
                kernel_y.view(1, 1, -1, 1),
            )

        work = work.squeeze(1).clamp(0.0, 1.0)
        if was_2d:
            work = work.squeeze(0)
        return work.to(device=mask_tensor.device, dtype=torch.float32)

    @classmethod
    def apply_colored_light(cls, image, mask, color_rgb, intensity=1.0):
        """Apply additive colored light using a mask."""
        if intensity <= 0:
            return image
        color_norm = torch.tensor([c / 255.0 for c in color_rgb], device=image.device, dtype=torch.float32)
        if image.shape[-1] == 1:
            # Single-channel input: add the light's luminance. Broadcasting the RGB
            # triple here would silently turn a 1-channel image into a 3-channel one.
            weights = torch.tensor(_LUMA_WEIGHTS, device=image.device, dtype=torch.float32)
            color_light = (color_norm * weights).sum().view(1, 1, 1, 1)
        else:
            color_light = color_norm.view(1, 1, 1, 3)
        if mask.dim() == 3:
            mask = mask.unsqueeze(-1)
        result = image + color_light * intensity * mask
        return torch.clamp(result, 0.0, 1.0)

    @classmethod
    def cast_shadow_mask(cls, fg_mask, light_position_x, light_position_y, shadow_length, steps=24):
        """How much of the subject stands between each pixel and the light.

        A light behind a subject should leave a shadow across the background,
        and up to v3.1.2 it left none at all: the subject blocked nothing, so
        the near side of a head was lit exactly as brightly as the far side.

        For every pixel, march back toward the light and take the largest
        foreground value found along the way, out to `shadow_length` of the
        frame's shorter side. A pixel whose path to the light crosses the
        subject comes back 1; one with a clear line of sight comes back 0.
        Marching stops at the light itself, so a pixel closer to the light than
        the subject is never shadowed by it.

        Vectorised through `grid_sample` rather than a per-pixel loop:
        deterministic, identical on CPU and GPU, and no SciPy. Returns a
        (batch, height, width) mask on the input's device.
        """
        batch, height, width = fg_mask.shape
        device = fg_mask.device
        if shadow_length <= 0 or steps < 1:
            return torch.zeros_like(fg_mask)

        # Trace small and upsample; the caller blurs the result anyway.
        scale = min(1.0, _SHADOW_TRACE_MAX / max(height, width))
        trace_h = max(8, int(round(height * scale)))
        trace_w = max(8, int(round(width * scale)))
        source = fg_mask.unsqueeze(1)
        if (trace_h, trace_w) != (height, width):
            source = F.interpolate(source, size=(trace_h, trace_w), mode="bilinear", align_corners=False)

        rows = torch.arange(trace_h, device=device, dtype=torch.float32).view(trace_h, 1)
        cols = torch.arange(trace_w, device=device, dtype=torch.float32).view(1, trace_w)
        light_x = light_position_x * trace_w
        light_y = light_position_y * trace_h
        to_light_x = light_x - cols
        to_light_y = light_y - rows
        distance = torch.sqrt(to_light_x * to_light_x + to_light_y * to_light_y).clamp(min=1e-6)
        step_x = to_light_x / distance
        step_y = to_light_y / distance
        reach = shadow_length * min(trace_w, trace_h)

        shadow = torch.zeros((batch, trace_h, trace_w), device=device, dtype=torch.float32)
        denom_x = max(trace_w - 1, 1)
        denom_y = max(trace_h - 1, 1)
        for step in range(1, steps + 1):
            travelled = torch.clamp(distance, max=reach * step / steps)
            sample_x = cols + step_x * travelled
            sample_y = rows + step_y * travelled
            grid = torch.stack(
                [
                    (sample_x / denom_x * 2.0 - 1.0).expand(trace_h, trace_w),
                    (sample_y / denom_y * 2.0 - 1.0).expand(trace_h, trace_w),
                ],
                dim=-1,
            ).unsqueeze(0).expand(batch, -1, -1, -1)
            sampled = F.grid_sample(
                source, grid, mode="bilinear", padding_mode="zeros", align_corners=True
            )
            shadow = torch.maximum(shadow, sampled.squeeze(1))

        if (trace_h, trace_w) != (height, width):
            shadow = F.interpolate(
                shadow.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False
            ).squeeze(1)
        return shadow.clamp(0.0, 1.0)

    @classmethod
    def calculate_rim_mask(cls, light_mask_np, fg_mask_np, light_position_x, light_position_y):
        """
        Calculates the raw (unblurred, unamplified) rim mask.

        This mask represents the light hitting the edges of the foreground subject,
        modulated by the direction of the light source relative to the edge normal.
        Requires SciPy for edge detection (ndimage.sobel).

        Args:
            light_mask_np (np.ndarray): The base light mask (e.g., outer circle)
                                        as a NumPy array, defining the potential
                                        area and intensity of the light source.
            fg_mask_np (np.ndarray): The foreground subject mask (1=subject, 0=background)
                                     as a NumPy array.
            light_position_x (float): Normalized horizontal light position (0-1).
            light_position_y (float): Normalized vertical light position (0-1).

        Returns:
            np.ndarray: The calculated raw rim mask as a NumPy array (float32, range 0-1).
                        Returns an empty (all zeros) mask if no edges are detected.
        """
        height, width = fg_mask_np.shape

        # 1. Edge Detection using Sobel filter
        edge_x = ndimage.sobel(fg_mask_np, axis=1)
        edge_y = ndimage.sobel(fg_mask_np, axis=0)
        grad_magnitude = np.sqrt(edge_x**2 + edge_y**2)

        max_edge = grad_magnitude.max()
        if max_edge > 1e-6:
            # Normalize to 0-1, then enhance edges slightly with a power function
            # (value < 1 thickens/brightens). The raw magnitude is reused below
            # for the edge normals, so it is computed once and never overwritten.
            edge_mask = np.power(grad_magnitude / max_edge, 0.7)
        else:
            # A uniform mask (all subject or all background) has no edge to rim.
            logger.debug("No edges detected for rim mask.")
            return np.zeros_like(light_mask_np)

        # 2. Base Rim Light: Modulate edge mask by the original light intensity
        # This ensures rim light only appears where the original light would hit the edge
        rim_light_raw = light_mask_np * edge_mask

        # 3. Directional Modulation: Make rim brighter where light hits edge from behind
        y_grid, x_grid = cls._pixel_grid(width, height)
        light_x_px, light_y_px = light_position_x * width, light_position_y * height
        light_dir_x, light_dir_y = x_grid - light_x_px, y_grid - light_y_px
        light_dist = np.sqrt(light_dir_x**2 + light_dir_y**2)
        light_dist = np.where(light_dist < 1e-6, 1, light_dist)
        light_dir_x = light_dir_x / light_dist
        light_dir_y = light_dir_y / light_dist
        grad_magnitude_norm = np.where(grad_magnitude < 1e-6, 1, grad_magnitude)
        normal_x, normal_y = edge_x / grad_magnitude_norm, edge_y / grad_magnitude_norm
        dot_product = light_dir_x * normal_x + light_dir_y * normal_y
        directional_factor = np.clip((-dot_product + 1) / 2, 0, 1)
        directional_factor = np.power(directional_factor, 1.5)
        final_mask_np = np.clip(rim_light_raw * directional_factor, 0.0, 1.0)

        logger.debug(f"    - Raw rim mask calculated: Max intensity = {final_mask_np.max():.3f}")
        return final_mask_np.astype(np.float32)

    @classmethod
    def create_debug_image(cls, original_image, all_inner_base_masks, all_outer_base_masks, light_sources, fg_mask=None):
        """Create a debug visualization showing base masks and light positions."""
        logger.debug("--- Creating Debug Image ---")
        try:
            img_tensor = original_image[0].cpu()
            fg_mask_tensor = fg_mask[0].cpu() if fg_mask is not None else None
            # Use base masks for the *first* light source for visualization
            inner_base_mask = all_inner_base_masks[0].cpu() if all_inner_base_masks else None
            outer_base_mask = all_outer_base_masks[0].cpu() if all_outer_base_masks else None

            img_np = (img_tensor.clamp(0, 1).numpy() * 255).astype(np.uint8)
            if img_np.shape[-1] == 1:
                # The debug view is always RGB; PIL cannot build an image from an
                # (H, W, 1) array, and a grayscale input used to land in the
                # fatal-error path and come back solid black.
                img_np = np.repeat(img_np, 3, axis=-1)
            pil_img = Image.fromarray(img_np).convert('RGBA')
            width, height = pil_img.size
            logger.debug(f"  Base image size: {width}x{height}")

            # --- Create Overlays ---
            inner_overlay_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            ring_overlay_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            fg_overlay_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))

            # Inner Mask Overlay (Red)
            try:
                if inner_base_mask is not None:
                    inner_alpha = (np.clip(inner_base_mask.numpy(), 0, 1) * 128).astype(np.uint8)
                    if np.max(inner_alpha) > 0:
                        inner_overlay_np = np.zeros((height, width, 4), dtype=np.uint8)
                        inner_overlay_np[..., 0] = 255  # R
                        inner_overlay_np[..., 3] = inner_alpha
                        inner_overlay_img = Image.fromarray(inner_overlay_np, 'RGBA')
            except Exception:
                logger.exception("  ERROR creating inner mask overlay")

            # Ring Mask Overlay (Blue)
            try:
                if inner_base_mask is not None and outer_base_mask is not None:
                    ring_np = np.clip(outer_base_mask.numpy() - inner_base_mask.numpy(), 0, 1)
                    ring_alpha = (ring_np * 128).astype(np.uint8)
                    if np.max(ring_alpha) > 0:
                        ring_overlay_np = np.zeros((height, width, 4), dtype=np.uint8)
                        ring_overlay_np[..., 2] = 255  # B
                        ring_overlay_np[..., 3] = ring_alpha
                        ring_overlay_img = Image.fromarray(ring_overlay_np, 'RGBA')
            except Exception:
                logger.exception("  ERROR creating ring mask overlay")

            # Foreground Mask Overlay (Green)
            try:
                if fg_mask_tensor is not None:
                    fg_alpha = (np.clip(fg_mask_tensor.numpy(), 0, 1) * 100).astype(np.uint8)
                    if np.max(fg_alpha) > 0:
                        fg_overlay_np = np.zeros((height, width, 4), dtype=np.uint8)
                        fg_overlay_np[..., 1] = 255
                        fg_overlay_np[..., 3] = fg_alpha
                        fg_overlay_img = Image.fromarray(fg_overlay_np, 'RGBA')
            except Exception:
                logger.exception("  ERROR creating FG mask overlay")

            # --- Composite Overlays ---
            debug_img = pil_img
            try:
                # Composite order: Base -> FG (Green) -> Inner (Red) -> Ring (Blue)
                debug_img = Image.alpha_composite(debug_img, fg_overlay_img)
                debug_img = Image.alpha_composite(debug_img, inner_overlay_img)
                debug_img = Image.alpha_composite(debug_img, ring_overlay_img)
            except Exception:
                logger.exception("  ERROR compositing overlays")

            # --- Draw Indicators & Legend ---
            draw_debug = ImageDraw.Draw(debug_img)
            # Same scaling rule as the placeholder: at 1344x768 the old fixed
            # 12px legend and 5px dots were unreadable in a preview thumbnail.
            font_size = _debug_font_size(height)
            font = _load_debug_font(font_size)
            marker_radius = max(4, font_size // 2)
            ring_width = max(1, font_size // 8)
            for i, light in enumerate(light_sources):
                try:
                    x, y = int(light["position_x"] * width), int(light["position_y"] * height)
                    color = tuple(light.get("color", [255, 255, 255])) + (220,)
                    inner_r_px = int(light["inner_radius"] * min(width, height))
                    outer_r_px = int(light["outer_radius"] * min(width, height))
                    draw_debug.ellipse((x-marker_radius, y-marker_radius, x+marker_radius, y+marker_radius), fill=color, outline=(0, 0, 0, 200), width=ring_width)
                    draw_debug.ellipse((x-inner_r_px, y-inner_r_px, x+inner_r_px, y+inner_r_px), outline=(255, 255, 0, 150), width=ring_width)
                    draw_debug.ellipse((x-outer_r_px, y-outer_r_px, x+outer_r_px, y+outer_r_px), outline=(0, 255, 255, 150), width=ring_width)
                    label = f"L{i+1}"
                    # The label sits to the right of the marker, but a light near
                    # the right edge (a rim light at x=0.9, say) would push it off
                    # the frame, so it flips to the left side instead. Both axes
                    # are then clamped, because a marker in a corner can still run
                    # a tall glyph past the top or bottom.
                    gap = marker_radius + ring_width * 2
                    label_w = draw_debug.textlength(label, font=font) if font else font_size
                    text_x = x + gap
                    if text_x + label_w > width:
                        text_x = x - gap - label_w
                    text_x = max(0, min(text_x, width - label_w))
                    text_y = max(0, min(y - font_size // 2 - 2, height - font_size))
                    text_pos = (text_x, text_y)
                    if font:
                        bbox = draw_debug.textbbox(text_pos, label, font=font)
                        draw_debug.rectangle(bbox, fill=(0, 0, 0, 180))
                        draw_debug.text(text_pos, label, fill=(255, 255, 255, 230), font=font)
                    else:
                        draw_debug.text(text_pos, label, fill=(255, 255, 255, 230))
                except Exception:
                    logger.exception(f"    ERROR drawing indicator for light {i+1}")
            try:
                legend_items = [("Inner Mask Area", (255, 0, 0, 128)), ("Outer Mask Area (Ring)", (0, 0, 255, 128))]
                if fg_mask_tensor is not None:
                    legend_items.append(("Foreground Mask", (0, 255, 0, 100)))
                pad = max(5, font_size // 2)
                swatch = font_size
                legend_x = legend_y = pad * 2
                line_height = int(font_size * 1.5)
                max_width = 0
                for text, _ in legend_items:
                    try:
                        text_w = draw_debug.textlength(text, font=font) if font else len(text) * font_size * 0.55
                    except Exception:
                        text_w = len(text) * font_size * 0.55
                    max_width = max(max_width, text_w)
                legend_box = (
                    legend_x - pad,
                    legend_y - pad,
                    legend_x + swatch + pad + max_width + pad,
                    legend_y + len(legend_items) * line_height,
                )
                draw_debug.rectangle(legend_box, fill=(0, 0, 0, 190))
                for text, color in legend_items:
                    draw_debug.rectangle((legend_x, legend_y, legend_x + swatch, legend_y + swatch), fill=color)
                    draw_debug.text((legend_x + swatch + pad, legend_y), text, fill=(255, 255, 255, 220), font=font)
                    legend_y += line_height
            except Exception:
                logger.exception("  ERROR drawing legend")

            debug_np = np.array(debug_img.convert('RGB')).astype(np.float32) / 255.0
            logger.debug("--- Debug Image Creation Finished ---")
            return torch.from_numpy(debug_np).unsqueeze(0).to(original_image.device)

        except Exception:
            logger.exception("--- FATAL ERROR in create_debug_image ---")
            return cls._blank_debug_image(
                original_image, "The debug view could not be drawn - see the ComfyUI console."
            )

    @staticmethod
    def _blank_debug_image(image, reason=None):
        """RGB placeholder matching the image's spatial size.

        A solid black frame is indistinguishable from a crash: users wire the
        debug output to a preview, see black, and reasonably conclude the node
        failed. So the placeholder states why it is empty and how to fill it.

        Everything here scales with the frame. v3.1.2 drew 13px type on a
        full-resolution canvas, which is legible at 96x64 and invisible at 1344
        wide, so the placeholder was reported as the very black frame it was
        written to replace. The inset border is the other half of that: it means
        even an unreadable thumbnail is visibly a deliberate panel rather than a
        dead output. Falls back to plain black only if the text cannot be drawn
        (no font, or an image too small to hold a line of type).
        """
        height, width = image.shape[1], image.shape[2]
        blank = torch.zeros((1, height, width, 3), device=image.device, dtype=torch.float32)
        if not reason or width < 64 or height < 24:
            return blank
        try:
            canvas = Image.new("RGB", (width, height), (24, 24, 28))
            draw = ImageDraw.Draw(canvas)
            font_size = _debug_font_size(height)
            font = _load_debug_font(font_size)

            border = max(2, font_size // 6)
            inset = max(4, font_size // 2)
            draw.rectangle(
                (inset, inset, width - 1 - inset, height - 1 - inset),
                outline=(96, 96, 112),
                width=border,
            )

            margin = inset + border + font_size
            lines = _wrap_debug_text(draw, reason, font, max(font_size, width - 2 * margin))
            line_height = int(font_size * 1.4)
            y = max(margin, (height - line_height * len(lines)) // 2)
            for line in lines:
                try:
                    text_width = draw.textlength(line, font=font)
                except Exception:
                    text_width = len(line) * font_size * 0.55
                draw.text(
                    (max(margin, int((width - text_width) // 2)), y),
                    line,
                    fill=(198, 198, 210),
                    font=font,
                )
                y += line_height
            placeholder = np.array(canvas).astype(np.float32) / 255.0
            return torch.from_numpy(placeholder).unsqueeze(0).to(image.device)
        except Exception:
            logger.debug("Could not render the debug placeholder text; using a black frame.")
            return blank

    #: Output slot the debug view is on. A downstream node consuming it appears
    #: in the submitted prompt as the pair ``[<this node's id>, 2]``.
    DEBUG_OUTPUT_SLOT = 2

    @classmethod
    def _debug_output_is_consumed(cls):
        """Does the submitted prompt wire anything to this node's debug output?

        The node's own `debug_output_connected` widget is what makes the UI feel
        immediate (see web/relight_debug.js - a widget value is what ComfyUI
        hashes to decide a node must re-run). This is the independent check for
        everything that never loads that file: an API caller posting to /prompt,
        or a UI where the frontend extension failed to load. Either signal is
        enough; neither is required.

        Deliberately total: hidden inputs are absent outside a running ComfyUI
        and empty in some internal calls, so every failure here means "no", not
        an exception on a path the user never asked about.
        """
        hidden = getattr(cls, "hidden", None)
        prompt = getattr(hidden, "prompt", None)
        unique_id = getattr(hidden, "unique_id", None)
        if not prompt or unique_id is None:
            return False
        wanted = str(unique_id)
        try:
            for node in prompt.values():
                for value in (node.get("inputs") or {}).values():
                    if (
                        isinstance(value, (list, tuple))
                        and len(value) == 2
                        and str(value[0]) == wanted
                        and value[1] == cls.DEBUG_OUTPUT_SLOT
                    ):
                        return True
        except (AttributeError, TypeError):
            # A prompt shape we do not recognise is not worth crashing a render.
            logger.debug("ReLight: could not read the prompt to check debug connectivity.")
        return False

    # --- Mask preparation ---

    @classmethod
    def _prepare_mask(cls, input_mask, batch_size, height, width, device):
        """Normalise an incoming MASK to a clean (batch, height, width) float tensor.

        Handles rank (2D/3D/4D and RGB/RGBA sources), resolution (a mask made
        before an upscale no longer crashes the node), value range and batch size.
        """
        fg_mask = input_mask.to(device=device, dtype=torch.float32)

        if fg_mask.dim() == 4:
            if fg_mask.shape[3] == 1:
                fg_mask = fg_mask.squeeze(-1)
            elif fg_mask.shape[3] == 4:
                fg_mask = fg_mask[..., 3]
            elif fg_mask.shape[3] == 3:
                fg_mask = fg_mask.mean(dim=-1)
            else:
                fg_mask = fg_mask[..., 0]
        elif fg_mask.dim() == 2:
            fg_mask = fg_mask.unsqueeze(0)
        elif fg_mask.dim() != 3:
            raise ValueError(f"ReLight: cannot interpret a mask with shape {tuple(input_mask.shape)}")

        if fg_mask.shape[-2:] != (height, width):
            logger.info(
                f"ReLight: resizing mask from {tuple(fg_mask.shape[-2:])} to {(height, width)} to match the image."
            )
            fg_mask = F.interpolate(
                fg_mask.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False
            ).squeeze(1)

        fg_mask = fg_mask.clamp(0.0, 1.0)

        if fg_mask.shape[0] != batch_size:
            if fg_mask.shape[0] != 1:
                logger.warning(
                    f"ReLight: mask batch ({fg_mask.shape[0]}) does not match image batch ({batch_size}); "
                    "using the first mask for every image."
                )
            fg_mask = fg_mask[0:1].expand(batch_size, -1, -1).contiguous()

        mean_val = fg_mask.mean().item()
        if mean_val > 0.9:
            logger.warning(
                f"ReLight: mask is {mean_val * 100:.0f}% white. If your subject appears unlit, the mask may be "
                "inverted (expected: white=subject, black=background)."
            )
        return fg_mask

    # --- Main Execution Function ---

    @classmethod
    def execute(cls, image: torch.Tensor, **kwargs) -> io.NodeOutput:
        """Applies relighting effects to the input image based on parameters."""
        start_time = time.time()
        image = image.to(dtype=torch.float32)
        if image.dim() != 4:
            raise ValueError(f"ReLight: expected an image of shape (batch, height, width, channels), got {tuple(image.shape)}")
        batch_size, height, width, channels = image.shape
        device = image.device

        # Keep any alpha channel out of the lighting maths and re-attach it at the
        # end; the correction path is RGB-only and used to crash on RGBA inputs.
        alpha = None
        if channels == 4:
            alpha = image[..., 3:]
            image = image[..., :3]
        elif channels not in (1, 3):
            raise ValueError(f"ReLight: expected a 1-, 3- or 4-channel image, got {channels} channels")

        logger.debug("\n--- ReLight Node ---")
        logger.debug(f"Input Image: {width}x{height}, Batch: {batch_size}, Device: {device}, Shape: {image.shape}")

        params = kwargs.copy()
        params = cls._load_preset(params.get('preset', 'None'), params)

        preset = params.get('preset', 'None')
        remove_background = params.get('remove_background', False)
        subject_interaction = params.get('subject_interaction', cls.SUBJECT_NONE)
        lighting_mode = params.get('lighting_mode', cls.MODE_CORRECTION)
        mask_shape = params.get('mask_shape', cls.SHAPE_RADIAL)
        debug_output_connected = bool(params.get('debug_output_connected', False)) or cls._debug_output_is_consumed()
        effect_strength = params.get('effect_strength', 1.0)
        rim_amplification = params.get('rim_amplification', 2.0)
        shadow_strength = params.get('shadow_strength', 0.6)
        shadow_length = params.get('shadow_length', 0.35)
        num_light_sources = params.get('num_light_sources', 1)
        mask_blur = params.get('mask_blur', 50.0)
        input_mask = params.get('mask', None)

        # The three lighting modes are two independent switches underneath: a
        # coloured additive pass and a two-zone grade. 'Both' runs the colour
        # first and grades the result, which is what the three presets that set
        # a colour *and* a full correction block always looked like they meant.
        apply_colored = lighting_mode in (cls.MODE_COLORED, cls.MODE_BOTH)
        apply_correction = lighting_mode in (cls.MODE_CORRECTION, cls.MODE_BOTH)
        use_gradient_mode = mask_shape == cls.SHAPE_GRADIENT

        logger.debug(f"Mode: Preset='{preset}', Lighting='{lighting_mode}', Shape='{mask_shape}', Subject='{subject_interaction}'")
        logger.debug(f"Settings: Strength={effect_strength:.2f}, Rim Amp={rim_amplification:.2f}, Mask Blur={mask_blur:.1f}, Debug={debug_output_connected}")

        # --- Mask Handling ---
        fg_mask = None
        if input_mask is not None:
            fg_mask = cls._prepare_mask(input_mask, batch_size, height, width, device)
            # The MASK output is always the normalised mask, regardless of which
            # features happen to be enabled.
            output_mask = fg_mask.clone()
        else:
            output_mask = torch.zeros((batch_size, height, width), device=device, dtype=torch.float32)

        occlusion_requested = subject_interaction != cls.SUBJECT_NONE
        if fg_mask is None and (occlusion_requested or remove_background):
            logger.info("ReLight: no mask connected - occlusion and background compositing are disabled for this run.")
        occlusion_active = occlusion_requested and fg_mask is not None
        composite_active = remove_background and fg_mask is not None

        # --- Define Light Sources ---
        light_sources = []
        for i in range(1, num_light_sources + 1):
            prefix = f"light{i}_" if i > 1 else "light_"
            pos_x_key, pos_y_key = ("light_position_x", "light_position_y") if i == 1 else (f"light{i}_position_x", f"light{i}_position_y")
            inner_r_key, outer_r_key = ("inner_circle_radius", "outer_circle_radius") if i == 1 else (f"light{i}_inner_radius", f"light{i}_outer_radius")
            color_r_key, color_g_key, color_b_key = f"{prefix}color_r", f"{prefix}color_g", f"{prefix}color_b"
            intensity_key = f"{prefix}intensity"
            if params.get(pos_x_key) is None or params.get(pos_y_key) is None:
                if i > 1:
                    continue
                raise ValueError("Missing essential parameters for Light 1")
            light = {
                "id": i,
                "position_x": params.get(pos_x_key, 0.5), "position_y": params.get(pos_y_key, 0.5),
                "inner_radius": params.get(inner_r_key, 0.3), "outer_radius": params.get(outer_r_key, 0.6),
                "color": [int(params.get(color_r_key, 255)), int(params.get(color_g_key, 255)), int(params.get(color_b_key, 255))],
                "intensity": params.get(intensity_key, 1.0),
            }
            light_sources.append(light)
            logger.debug(f"Defined Light {i}: Pos=({light['position_x']:.2f},{light['position_y']:.2f}), Radii=({light['inner_radius']:.2f},{light['outer_radius']:.2f}), Color={light['color']}, Intensity={light['intensity']:.2f}")

        # --- Initialize Result & Masks for Debug ---
        result_tensor = image.clone()
        all_inner_base_masks_for_debug = []
        all_outer_base_masks_for_debug = []

        # --- Process Each Light Source ---
        for light in light_sources:
            logger.debug(f"\nProcessing Light Source {light['id']}...")

            # --- Create Base Light Masks (position-only, shared across the batch) ---
            if use_gradient_mode:
                dx, dy = light["position_x"] - 0.5, light["position_y"] - 0.5
                angle = math.degrees(math.atan2(dy, dx))
                inner_mask_base = cls.create_gradient_mask(width, height, light["position_x"], light["position_y"], light["inner_radius"], angle).to(device)
                outer_mask_base = cls.create_gradient_mask(width, height, light["position_x"], light["position_y"], light["outer_radius"], angle).to(device)
            else:
                inner_mask_base = cls.create_circle_mask(width, height, light["position_x"], light["position_y"], light["inner_radius"]).to(device)
                outer_mask_base = cls.create_circle_mask(width, height, light["position_x"], light["position_y"], light["outer_radius"]).to(device)

            all_inner_base_masks_for_debug.append(inner_mask_base)
            all_outer_base_masks_for_debug.append(outer_mask_base)

            # --- MASK APPLICATION LOGIC ---
            # Light masks carry a leading dimension that is either 1 (identical for
            # every image) or batch_size (occlusion, which depends on each image's
            # own mask). Both broadcast against the (B, H, W, C) image.
            inner_zone_mask = None
            ring_zone_mask = None
            colored_light_mask = None
            correction_mask = None

            if occlusion_active and subject_interaction == cls.SUBJECT_RIM:
                logger.debug("  Processing as Behind Subject (Rim + Background Glow)...")
                light_mask_np = outer_mask_base.cpu().numpy()

                rim_per_image = []
                for b in range(batch_size):
                    raw_rim_mask_np = cls.calculate_rim_mask(
                        light_mask_np, fg_mask[b].cpu().numpy(), light["position_x"], light["position_y"]
                    )
                    rim_per_image.append(np.clip(raw_rim_mask_np * rim_amplification, 0.0, 1.0))
                amplified_rim = torch.from_numpy(np.stack(rim_per_image)).to(device)

                # The background glow gets real falloff. Up to v3.1.2 it was the
                # hard 0/1 outer disc, so "glow" was a flat slab with whatever
                # softness mask_blur happened to put on its rim.
                if use_gradient_mode:
                    background_base = outer_mask_base
                else:
                    background_base = cls.create_falloff_mask(
                        width, height, light["position_x"], light["position_y"],
                        light["inner_radius"], light["outer_radius"],
                    ).to(device)

                background_light_mask = background_base.unsqueeze(0).expand(batch_size, -1, -1)
                if shadow_strength > 0.0 and shadow_length > 0.0:
                    shadow = cls.cast_shadow_mask(
                        fg_mask, light["position_x"], light["position_y"], shadow_length
                    )
                    background_light_mask = background_light_mask * (1.0 - shadow * shadow_strength)

                # Blur the two halves SEPARATELY, and re-apply the silhouette to
                # the blurred background. v3.1.2 subtracted the subject before
                # the blur, so the blur smeared background light straight back
                # across the silhouette edge onto the face - the single most
                # direct cause of "Behind Subject does not occlude".
                blurred_rim = cls.apply_mask_blur(amplified_rim, mask_blur)
                blurred_background = cls.apply_mask_blur(background_light_mask, mask_blur) * (1.0 - fg_mask)
                final_light_mask = torch.clamp(blurred_rim + blurred_background, 0, 1)
                colored_light_mask = correction_mask = final_light_mask
                logger.debug(f"  Final Behind Subject Mask: Max={torch.max(final_light_mask):.3f}")

            elif occlusion_active and subject_interaction == cls.SUBJECT_FRONT:
                logger.debug("  Processing as Front Subject Light...")
                light_mask_np = outer_mask_base.cpu().numpy()
                fg_mask_np = fg_mask.cpu().numpy()

                enhance_subject_factor = 1.3
                reduce_background_factor = 0.8
                occlusion_factor_mask_np = fg_mask_np * enhance_subject_factor + (1.0 - fg_mask_np) * reduce_background_factor
                occlusion_factor_mask_np = ndimage.gaussian_filter(occlusion_factor_mask_np, sigma=(0, 2, 2))

                combined_mask_unblurred = torch.from_numpy(
                    np.clip(light_mask_np[None, ...] * occlusion_factor_mask_np, 0, 1).astype(np.float32)
                ).to(device)
                final_light_mask = cls.apply_mask_blur(combined_mask_unblurred, mask_blur)
                colored_light_mask = correction_mask = final_light_mask
                logger.debug(f"  Final Front Subject Mask: Max={torch.max(final_light_mask):.3f}")

            else:  # Standard (no subject interaction, or no mask to do it with)
                logger.debug("  Processing as Standard Light...")
                if apply_colored:
                    # Light area with real falloff: full strength inside the inner
                    # radius, fading to zero at the outer radius. Gradient masks
                    # already encode their own falloff.
                    if use_gradient_mode:
                        combined_mask_unblurred = outer_mask_base
                    else:
                        combined_mask_unblurred = cls.create_falloff_mask(
                            width, height, light["position_x"], light["position_y"],
                            light["inner_radius"], light["outer_radius"],
                        ).to(device)
                    colored_light_mask = cls.apply_mask_blur(combined_mask_unblurred, mask_blur).unsqueeze(0)
                if apply_correction:
                    # Two-zone color correction: inner_* params apply inside the
                    # inner mask, outer_* params in the ring between inner and outer.
                    # Both zones blur in one convolution - this is the hot path.
                    zones = cls.apply_mask_blur(
                        torch.stack([inner_mask_base, outer_mask_base]), mask_blur
                    )
                    inner_zone_mask, outer_zone_full = zones[0:1], zones[1:2]
                    ring_zone_mask = torch.clamp(outer_zone_full - inner_zone_mask, 0, 1)
                    correction_mask = torch.clamp(inner_zone_mask + ring_zone_mask, 0, 1)

            # --- Apply the lighting effect(s) this mode calls for ---
            # In 'Both' the colour goes on first and the grade is applied to the
            # result, so the grade sees the lit image rather than the original.
            if apply_colored and colored_light_mask is not None and torch.max(colored_light_mask) > 1e-4:
                logger.debug(f"  Applying colored light (RGB: {light['color']}, Intensity: {light['intensity']:.2f})...")
                effective_intensity = light['intensity'] * effect_strength
                result_tensor = cls.apply_colored_light(result_tensor, colored_light_mask.unsqueeze(-1), light['color'], effective_intensity)

            if apply_correction and correction_mask is not None and torch.max(correction_mask) > 1e-4:
                logger.debug("  Applying color correction...")
                corrected_inner = cls.apply_color_correction(
                    result_tensor,
                    params.get('inner_brightness', 0) * effect_strength,
                    params.get('inner_contrast', 0) * effect_strength,
                    params.get('inner_saturation', 0) * effect_strength,
                    params.get('inner_temperature', 0) * effect_strength,
                    params.get('inner_tint', 0) * effect_strength,
                    cls._scaled_gamma(params.get('inner_gamma', 1.0), effect_strength),
                )
                if inner_zone_mask is not None and ring_zone_mask is not None:
                    # Standard mode: two-zone correction. inner_* params in the
                    # inner area, outer_* params in the surrounding ring.
                    corrected_outer = cls.apply_color_correction(
                        result_tensor,
                        params.get('outer_brightness', 0) * effect_strength,
                        params.get('outer_contrast', 0) * effect_strength,
                        params.get('outer_saturation', 0) * effect_strength,
                        params.get('outer_temperature', 0) * effect_strength,
                        params.get('outer_tint', 0) * effect_strength,
                        cls._scaled_gamma(params.get('outer_gamma', 1.0), effect_strength),
                    )
                    # Blending as base + weighted *deltas* rather than
                    # base*(1-mask) + corrected*mask is algebraically identical but
                    # exact: where a correction is a no-op its delta is exactly
                    # zero, so identity settings leave the image untouched instead
                    # of accumulating rounding error across lights and chained nodes.
                    inner_expanded = inner_zone_mask.unsqueeze(-1)
                    ring_expanded = ring_zone_mask.unsqueeze(-1)
                    result_tensor = (
                        result_tensor
                        + (corrected_inner - result_tensor) * inner_expanded
                        + (corrected_outer - result_tensor) * ring_expanded
                    )
                else:
                    # Behind/Front modes: single combined mask with inner params
                    # (documented simplification).
                    result_tensor = result_tensor + (corrected_inner - result_tensor) * correction_mask.unsqueeze(-1)

        # --- Final Steps ---
        final_result = result_tensor

        if composite_active:
            # Gate on whether occlusion actually ran, not on the widget alone:
            # with no mask connected the lighting is plain, so it still needs
            # compositing even if subject_interaction asks for occlusion.
            if not occlusion_active:
                logger.debug("Compositing lit foreground onto original background...")
                final_result = image + (result_tensor - image) * fg_mask.unsqueeze(-1)
            else:
                logger.debug(f"Skipping final compositing for '{subject_interaction}' (lighting applied to FG/BG directly).")
        else:
            logger.debug("Skipping final compositing (remove_background=False or no mask).")

        # Debug Image Generation. There is no toggle: the debug view is drawn
        # whenever something is consuming this output, and skipped otherwise.
        if not debug_output_connected:
            debug_image = cls._blank_debug_image(
                image,
                "Nothing is connected to ReLight's debug_image output, so there was nothing to draw. "
                "Wire it to a preview to see light positions and mask zones.",
            )
        elif all_inner_base_masks_for_debug and all_outer_base_masks_for_debug:
            debug_image = cls.create_debug_image(
                image,
                all_inner_base_masks_for_debug,
                all_outer_base_masks_for_debug,
                light_sources,
                fg_mask,
            )
        else:
            logger.info("ReLight: no light masks were generated, so the debug view has nothing to draw.")
            debug_image = cls._blank_debug_image(
                image, "No light masks were generated - check the radius and position settings."
            )

        if not isinstance(debug_image, torch.Tensor) or debug_image.shape[1:] != (height, width, 3):
            logger.warning("ReLight: debug image was unusable; returning a placeholder.")
            debug_image = cls._blank_debug_image(image, "The debug view could not be drawn - see the ComfyUI console.")
        elif debug_image.shape[0] != 1:
            debug_image = debug_image[0:1]

        if alpha is not None:
            final_result = torch.cat([final_result, alpha], dim=-1)

        logger.debug(f"--- ReLight processing finished in {time.time() - start_time:.3f} seconds ---")
        return io.NodeOutput(final_result, output_mask, debug_image)

    @classmethod
    def _scaled_gamma(cls, gamma, effect_strength):
        """Fade gamma toward identity with effect_strength.

        Scaled in exponent space rather than linearly. Both forms agree exactly
        at strength 0 (identity) and 1 (the value as dialled in), but linear
        interpolation ran a dimming gamma straight through zero into negative
        territory above strength ~4 - where the safety clamp in
        apply_color_correction turned it into an exponent of 100 and crushed the
        zone to solid black. Clamped to the widget's own range so the master
        strength can never push gamma somewhere the user could not set by hand.
        """
        if gamma <= 0.0:
            return 1.0
        return min(max(gamma ** effect_strength, GAMMA_MIN), GAMMA_MAX)


# --- v3 Extension Registration ---
class ReLightExtension(ComfyExtension):
    """ComfyUI Extension for ReLight node."""

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        """Return list of nodes provided by this extension."""
        return [ReLight]


async def comfy_entrypoint() -> ReLightExtension:
    """Entry point for ComfyUI v3."""
    return ReLightExtension()
