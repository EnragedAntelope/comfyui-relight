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
            "mask_blur": 75, "use_colored_lights": True, "light_color_r": 255, "light_color_g": 200, "light_color_b": 120,
            "rim_amplification": 1.0, "use_gradient_mode": True
        },
        "Cool Blue Moonlight": {
            "light_position_x": 0.8, "light_position_y": 0.2, "inner_circle_radius": 0.4, "outer_circle_radius": 0.7,
            "inner_brightness": -5, "inner_contrast": 5, "inner_saturation": -5, "inner_temperature": -20, "inner_tint": 0, "inner_gamma": 0.91,
            "outer_brightness": -20, "outer_contrast": 0, "outer_saturation": -10, "outer_temperature": -30, "outer_tint": 0, "outer_gamma": 0.83,
            "mask_blur": 60, "use_colored_lights": True, "light_color_r": 120, "light_color_g": 150, "light_color_b": 255,
            "rim_amplification": 1.0
        },
        "Studio Key Light": {
            "light_position_x": 0.4, "light_position_y": 0.3, "inner_circle_radius": 0.6, "outer_circle_radius": 0.9,
            "inner_brightness": 12, "inner_contrast": 5, "inner_saturation": 0, "inner_temperature": 0, "inner_tint": 0, "inner_gamma": 1.0,
            "outer_brightness": -5, "outer_contrast": 0, "outer_saturation": -5, "outer_temperature": 0, "outer_tint": 0, "outer_gamma": 0.91,
            "mask_blur": 90, "rim_amplification": 1.0
        },
        "Rim Light (Behind)": {
            "light_position_x": 0.5, "light_position_y": 0.1, "inner_circle_radius": 0.3, "outer_circle_radius": 0.6,
            "apply_3d_lighting": True, "light_direction": "Behind Subject", "use_colored_lights": True,
            "light_color_r": 200, "light_color_g": 255, "light_color_b": 200, "light_intensity": 1.2,
            "inner_brightness": 0, "inner_contrast": 0, "inner_saturation": 0, "inner_temperature": 0, "inner_tint": 0, "inner_gamma": 1.0,
            "outer_brightness": 0, "outer_contrast": 0, "outer_saturation": 0, "outer_temperature": 0, "outer_tint": 0, "outer_gamma": 1.0,
            "mask_blur": 25, "effect_strength": 1.5, "rim_amplification": 2.5
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

    # Keys a preset may not touch while `preserve_positioning` is on.
    GEOMETRY_KEYS = frozenset({
        "light_position_x", "light_position_y", "inner_circle_radius", "outer_circle_radius",
        "light2_position_x", "light2_position_y", "light2_inner_radius", "light2_outer_radius",
        "light3_position_x", "light3_position_y", "light3_inner_radius", "light3_outer_radius",
    })

    # Single-entry pixel-coordinate cache; every light in a run shares one grid.
    _coord_cache = {}

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Define node schema for ComfyUI v3.

        NOTE: saved workflows store widget values *positionally*. Adding,
        removing or reordering an input silently corrupts every workflow already
        saved against this node. Append new inputs at the end.
        """
        return io.Schema(
            node_id="ReLight",
            display_name="ReLight 💡",
            category="image/lighting",
            description="Creates realistic lighting effects with multiple light sources, colored lights, and 3D lighting simulation with subject occlusion",
            inputs=[
                # --- Core Inputs ---
                io.Image.Input("image", tooltip="The input image to apply lighting effects to"),
                io.Mask.Input("mask", optional=True, tooltip="Foreground mask (white=subject, black=background). Needed for occlusion ('Behind Subject' / 'In Front of Subject') and for 'remove_background' compositing. Resized automatically if it does not match the image"),

                # --- Global Behavior ---
                io.Combo.Input("preset", options=list(cls.PRESETS.keys()), default="None", tooltip="Select a preset or 'None' for custom settings. NOTE: a preset overrides the widgets below - the values shown on the node are ignored for whatever the preset defines"),
                io.Int.Input("num_light_sources", default=1, min=1, max=3, step=1, tooltip="Number of light sources (1-3). Lights 2 and 3 have their own position, radius and color, but in color-correction mode they reuse Light 1's correction settings"),
                io.Boolean.Input("preserve_positioning", **_adv(default=False, advanced=True, tooltip="Keep your own light positions and radii when a preset is selected, instead of letting the preset set them")),
                io.Boolean.Input("show_debug_info", **_adv(default=False, advanced=True, tooltip="Output a debug visualization image (first image of the batch)")),

                # --- Lighting Mode & Occlusion ---
                io.Boolean.Input("use_colored_lights", default=False, tooltip="Use additive colored light instead of color correction?"),
                io.Boolean.Input("use_gradient_mode", default=False, tooltip="Use directional gradient masks instead of radial?"),
                io.Boolean.Input("apply_3d_lighting", **_adv(default=True, advanced=True, tooltip="Master switch for occlusion. Leave on and use 'light_direction' to choose the behaviour; turning this off forces 'No Occlusion'")),
                io.Combo.Input("light_direction", options=["Behind Subject", "In Front of Subject", "No Occlusion"], default="No Occlusion", tooltip="How light interacts with the subject. 'Behind'/'In Front' require a mask"),
                io.Boolean.Input("remove_background", default=False, tooltip="Composite the lit result back over the untouched original using the mask, so only the subject is relit. Does not remove anything. Ignored for 'Behind Subject' and 'In Front of Subject'"),

                # --- Global Modifiers ---
                io.Float.Input("effect_strength", default=1.0, min=0.0, max=5.0, step=0.1, tooltip="Overall intensity multiplier for lighting adjustments/colors. 0.0 leaves the image untouched"),
                io.Float.Input("mask_blur", default=50.0, min=0.0, max=200.0, step=1.0, tooltip="Blur radius for light mask edges (smoother transitions)"),
                io.Float.Input("rim_amplification", default=2.0, min=0.0, max=10.0, step=0.1, tooltip="Intensity boost specifically for rim light component (when 'Behind Subject')"),

                # --- Light 1 Settings ---
                # Position & Shape
                io.Float.Input("light_position_x", default=0.5, min=0.0, max=1.0, step=0.01, tooltip="Light 1: Horizontal position (0=left, 1=right)"),
                io.Float.Input("light_position_y", default=0.5, min=0.0, max=1.0, step=0.01, tooltip="Light 1: Vertical position (0=top, 1=bottom)"),
                io.Float.Input("inner_circle_radius", default=0.4, min=0.0, max=1.0, step=0.01, tooltip="Light 1: Inner radius (strongest effect area)"),
                io.Float.Input("outer_circle_radius", default=0.7, min=0.0, max=1.0, step=0.01, tooltip="Light 1: Outer radius (falloff area)"),
                # Colored Light Mode
                io.Int.Input("light_color_r", default=255, min=0, max=255, step=1, tooltip="Light 1: Red color (if 'Use Colored Lights' is True)"),
                io.Int.Input("light_color_g", default=255, min=0, max=255, step=1, tooltip="Light 1: Green color (if 'Use Colored Lights' is True)"),
                io.Int.Input("light_color_b", default=255, min=0, max=255, step=1, tooltip="Light 1: Blue color (if 'Use Colored Lights' is True)"),
                io.Float.Input("light_intensity", default=1.0, min=0.0, max=3.0, step=0.1, tooltip="Light 1: Intensity (if 'Use Colored Lights' is True)"),
                # Color Correction Mode (Inner Area)
                io.Float.Input("inner_brightness", default=10.0, min=-100.0, max=100.0, step=1.0, tooltip="Light 1: Inner area brightness (Color Correction mode)"),
                io.Float.Input("inner_contrast", default=5.0, min=-100.0, max=100.0, step=1.0, tooltip="Light 1: Inner area contrast (Color Correction mode)"),
                io.Float.Input("inner_saturation", default=5.0, min=-100.0, max=100.0, step=1.0, tooltip="Light 1: Inner area saturation (Color Correction mode)"),
                io.Float.Input("inner_temperature", default=0.0, min=-100.0, max=100.0, step=1.0, tooltip="Light 1: Inner area temperature (-100=cool, 100=warm)"),
                io.Float.Input("inner_tint", default=0.0, min=-100.0, max=100.0, step=1.0, tooltip="Light 1: Inner area tint (-100=magenta, 100=green)"),
                io.Float.Input("inner_gamma", default=1.0, min=0.1, max=5.0, step=0.05, tooltip="Light 1: Inner area gamma. Above 1.0 brightens midtones, below 1.0 darkens them"),
                # Color Correction Mode (Outer Area)
                io.Float.Input("outer_brightness", **_adv(default=-10.0, min=-100.0, max=100.0, step=1.0, advanced=True, tooltip="Light 1: Outer area brightness (Color Correction mode)")),
                io.Float.Input("outer_contrast", **_adv(default=0.0, min=-100.0, max=100.0, step=1.0, advanced=True, tooltip="Light 1: Outer area contrast (Color Correction mode)")),
                io.Float.Input("outer_saturation", **_adv(default=-10.0, min=-100.0, max=100.0, step=1.0, advanced=True, tooltip="Light 1: Outer area saturation (Color Correction mode)")),
                io.Float.Input("outer_temperature", **_adv(default=0.0, min=-100.0, max=100.0, step=1.0, advanced=True, tooltip="Light 1: Outer area temperature")),
                io.Float.Input("outer_tint", **_adv(default=0.0, min=-100.0, max=100.0, step=1.0, advanced=True, tooltip="Light 1: Outer area tint")),
                io.Float.Input("outer_gamma", **_adv(default=0.91, min=0.1, max=5.0, step=0.05, advanced=True, tooltip="Light 1: Outer area gamma. Above 1.0 brightens midtones, below 1.0 darkens them")),

                # --- Light 2 Settings (Optional) ---
                io.Float.Input("light2_position_x", **_adv(default=0.8, min=0.0, max=1.0, step=0.01, optional=True, advanced=True, tooltip="Light 2: Horizontal position")),
                io.Float.Input("light2_position_y", **_adv(default=0.2, min=0.0, max=1.0, step=0.01, optional=True, advanced=True, tooltip="Light 2: Vertical position")),
                io.Float.Input("light2_inner_radius", **_adv(default=0.3, min=0.0, max=1.0, step=0.01, optional=True, advanced=True, tooltip="Light 2: Inner radius")),
                io.Float.Input("light2_outer_radius", **_adv(default=0.6, min=0.0, max=1.0, step=0.01, optional=True, advanced=True, tooltip="Light 2: Outer radius")),
                io.Int.Input("light2_color_r", **_adv(default=180, min=0, max=255, step=1, optional=True, advanced=True, tooltip="Light 2: Red color")),
                io.Int.Input("light2_color_g", **_adv(default=180, min=0, max=255, step=1, optional=True, advanced=True, tooltip="Light 2: Green color")),
                io.Int.Input("light2_color_b", **_adv(default=255, min=0, max=255, step=1, optional=True, advanced=True, tooltip="Light 2: Blue color")),
                io.Float.Input("light2_intensity", **_adv(default=0.7, min=0.0, max=3.0, step=0.1, optional=True, advanced=True, tooltip="Light 2: Intensity (Colored mode)")),

                # --- Light 3 Settings (Optional) ---
                io.Float.Input("light3_position_x", **_adv(default=0.3, min=0.0, max=1.0, step=0.01, optional=True, advanced=True, tooltip="Light 3: Horizontal position")),
                io.Float.Input("light3_position_y", **_adv(default=0.8, min=0.0, max=1.0, step=0.01, optional=True, advanced=True, tooltip="Light 3: Vertical position")),
                io.Float.Input("light3_inner_radius", **_adv(default=0.25, min=0.0, max=1.0, step=0.01, optional=True, advanced=True, tooltip="Light 3: Inner radius")),
                io.Float.Input("light3_outer_radius", **_adv(default=0.5, min=0.0, max=1.0, step=0.01, optional=True, advanced=True, tooltip="Light 3: Outer radius")),
                io.Int.Input("light3_color_r", **_adv(default=255, min=0, max=255, step=1, optional=True, advanced=True, tooltip="Light 3: Red color")),
                io.Int.Input("light3_color_g", **_adv(default=150, min=0, max=255, step=1, optional=True, advanced=True, tooltip="Light 3: Green color")),
                io.Int.Input("light3_color_b", **_adv(default=120, min=0, max=255, step=1, optional=True, advanced=True, tooltip="Light 3: Blue color")),
                io.Float.Input("light3_intensity", **_adv(default=0.5, min=0.0, max=3.0, step=0.1, optional=True, advanced=True, tooltip="Light 3: Intensity (Colored mode)")),
            ],
            outputs=[
                io.Image.Output("image", display_name="image", tooltip="The relit image"),
                io.Mask.Output("mask", display_name="mask", tooltip="Pass-through of the input mask, normalised to (batch, height, width) and resized to the image (black if none connected)"),
                io.Image.Output("debug_image", display_name="debug_image", tooltip="Visualization of light positions and masks (enable 'show_debug_info')"),
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
            updated_params[key] = value
        if preserve:
            logger.debug("  - Preserving user-defined light positions and radii.")
        return updated_params

    @classmethod
    def _pixel_grid(cls, width, height):
        """Row/column coordinate grids for an image size, reused across lights."""
        key = (height, width)
        cached = cls._coord_cache.get(key)
        if cached is None:
            cached = np.mgrid[0:height, 0:width]
            # Single-entry cache: a workflow rarely alternates resolutions, and this
            # keeps large grids from accumulating.
            cls._coord_cache = {key: cached}
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
        color_light = color_norm.view(1, 1, 1, 3)
        if mask.dim() == 3:
            mask = mask.unsqueeze(-1)
        result = image + color_light * intensity * mask
        return torch.clamp(result, 0.0, 1.0)

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
        edge_magnitude = np.sqrt(edge_x**2 + edge_y**2)

        # Normalize edge magnitude to 0-1 range
        max_edge = edge_magnitude.max()
        if max_edge > 1e-6:
            edge_magnitude /= max_edge
        else:
            # A uniform mask (all subject or all background) has no edge to rim.
            logger.debug("No edges detected for rim mask.")
            return np.zeros_like(light_mask_np)

        # Enhance edges slightly using a power function (value < 1 thickens/brightens)
        edge_mask = np.power(edge_magnitude, 0.7)

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
        grad_magnitude_norm = np.sqrt(edge_x**2 + edge_y**2)
        grad_magnitude_norm = np.where(grad_magnitude_norm < 1e-6, 1, grad_magnitude_norm)
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
            font = None
            font_size = 10
            try:
                font = ImageFont.load_default(size=12)
                font_size = 12
            except Exception:
                try:
                    font = ImageFont.load_default()
                except Exception as font_err:
                    logger.debug(f"  Could not load default font: {font_err}")
            for i, light in enumerate(light_sources):
                try:
                    x, y = int(light["position_x"] * width), int(light["position_y"] * height)
                    color = tuple(light.get("color", [255, 255, 255])) + (220,)
                    inner_r_px = int(light["inner_radius"] * min(width, height))
                    outer_r_px = int(light["outer_radius"] * min(width, height))
                    draw_debug.ellipse((x-5, y-5, x+5, y+5), fill=color, outline=(0, 0, 0, 200))
                    draw_debug.ellipse((x-inner_r_px, y-inner_r_px, x+inner_r_px, y+inner_r_px), outline=(255, 255, 0, 150), width=1)
                    draw_debug.ellipse((x-outer_r_px, y-outer_r_px, x+outer_r_px, y+outer_r_px), outline=(0, 255, 255, 150), width=1)
                    label = f"L{i+1}"
                    text_pos = (x + 10, y - font_size // 2 - 2)
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
                legend_x, legend_y = 10, 10
                line_height = font_size + 6
                max_width = 0
                for text, _ in legend_items:
                    try:
                        text_w = draw_debug.textlength(text, font=font) if font else len(text) * 7
                    except Exception:
                        text_w = len(text) * 7
                    max_width = max(max_width, text_w)
                legend_box = (legend_x - 5, legend_y - 5, legend_x + max_width + 25, legend_y + len(legend_items) * line_height)
                draw_debug.rectangle(legend_box, fill=(0, 0, 0, 190))
                for text, color in legend_items:
                    draw_debug.rectangle((legend_x, legend_y, legend_x + 12, legend_y + 12), fill=color)
                    draw_debug.text((legend_x + 18, legend_y + 1), text, fill=(255, 255, 255, 220), font=font)
                    legend_y += line_height
            except Exception:
                logger.exception("  ERROR drawing legend")

            debug_np = np.array(debug_img.convert('RGB')).astype(np.float32) / 255.0
            logger.debug("--- Debug Image Creation Finished ---")
            return torch.from_numpy(debug_np).unsqueeze(0).to(original_image.device)

        except Exception:
            logger.exception("--- FATAL ERROR in create_debug_image ---")
            return torch.zeros_like(original_image[0:1])

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
        apply_3d_lighting = params.get('apply_3d_lighting', True)
        light_direction = params.get('light_direction', 'No Occlusion')
        effect_strength = params.get('effect_strength', 1.0)
        rim_amplification = params.get('rim_amplification', 2.0)
        num_light_sources = params.get('num_light_sources', 1)
        use_colored_lights = params.get('use_colored_lights', False)
        use_gradient_mode = params.get('use_gradient_mode', False)
        mask_blur = params.get('mask_blur', 50.0)
        show_debug_info = params.get('show_debug_info', False)
        input_mask = params.get('mask', None)

        logger.debug(f"Mode: Preset='{preset}', 3D Lighting={apply_3d_lighting}, Direction='{light_direction}', Colored={use_colored_lights}, Gradient={use_gradient_mode}")
        logger.debug(f"Settings: Strength={effect_strength:.2f}, Rim Amp={rim_amplification:.2f}, Mask Blur={mask_blur:.1f}, Debug={show_debug_info}")

        # --- Mask Handling ---
        fg_mask = None
        if input_mask is not None:
            fg_mask = cls._prepare_mask(input_mask, batch_size, height, width, device)
            # The MASK output is always the normalised mask, regardless of which
            # features happen to be enabled.
            output_mask = fg_mask.clone()
        else:
            output_mask = torch.zeros((batch_size, height, width), device=device, dtype=torch.float32)

        occlusion_requested = apply_3d_lighting and light_direction != "No Occlusion"
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

            if occlusion_active and light_direction == "Behind Subject":
                logger.debug("  Processing as Behind Subject (Rim + Background Glow)...")
                light_mask_np = outer_mask_base.cpu().numpy()

                rim_per_image = []
                for b in range(batch_size):
                    raw_rim_mask_np = cls.calculate_rim_mask(
                        light_mask_np, fg_mask[b].cpu().numpy(), light["position_x"], light["position_y"]
                    )
                    rim_per_image.append(np.clip(raw_rim_mask_np * rim_amplification, 0.0, 1.0))
                amplified_rim = torch.from_numpy(np.stack(rim_per_image)).to(device)

                background_light_mask = outer_mask_base.unsqueeze(0) * (1.0 - fg_mask)
                combined_mask_unblurred = torch.clamp(amplified_rim + background_light_mask, 0, 1)
                final_light_mask = cls.apply_mask_blur(combined_mask_unblurred, mask_blur)
                logger.debug(f"  Final Behind Subject Mask: Max={torch.max(final_light_mask):.3f}")

            elif occlusion_active and light_direction == "In Front of Subject":
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
                logger.debug(f"  Final Front Subject Mask: Max={torch.max(final_light_mask):.3f}")

            else:  # Standard (No Occlusion) or occlusion unavailable
                logger.debug("  Processing as Standard Light...")
                if use_colored_lights:
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
                    final_light_mask = cls.apply_mask_blur(combined_mask_unblurred, mask_blur).unsqueeze(0)
                else:
                    # Two-zone color correction: inner_* params apply inside the
                    # inner mask, outer_* params in the ring between inner and outer.
                    # Both zones blur in one convolution - this is the hot path.
                    zones = cls.apply_mask_blur(
                        torch.stack([inner_mask_base, outer_mask_base]), mask_blur
                    )
                    inner_zone_mask, outer_zone_full = zones[0:1], zones[1:2]
                    ring_zone_mask = torch.clamp(outer_zone_full - inner_zone_mask, 0, 1)
                    final_light_mask = torch.clamp(inner_zone_mask + ring_zone_mask, 0, 1)
                logger.debug(f"  Final Standard Mask: Max={torch.max(final_light_mask):.3f}")

            # --- Apply Lighting Effect using the single final_light_mask ---
            if torch.max(final_light_mask) <= 1e-4:
                logger.debug("  Skipping light application (final mask is near-empty).")
                continue

            final_mask_expanded = final_light_mask.unsqueeze(-1)
            if use_colored_lights:
                logger.debug(f"  Applying colored light (RGB: {light['color']}, Intensity: {light['intensity']:.2f})...")
                effective_intensity = light['intensity'] * effect_strength
                result_tensor = cls.apply_colored_light(result_tensor, final_mask_expanded, light['color'], effective_intensity)
            else:
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
                    result_tensor = result_tensor + (corrected_inner - result_tensor) * final_mask_expanded

        # --- Final Steps ---
        final_result = result_tensor

        if composite_active:
            # Gate on whether occlusion actually ran, not on light_direction alone:
            # with apply_3d_lighting off the lighting is plain, so it still needs
            # compositing even if light_direction is left on an occlusion mode.
            if not occlusion_active:
                logger.debug("Compositing lit foreground onto original background...")
                final_result = image + (result_tensor - image) * fg_mask.unsqueeze(-1)
            else:
                logger.debug(f"Skipping final compositing for '{light_direction}' mode (lighting applied to FG/BG directly).")
        else:
            logger.debug("Skipping final compositing (remove_background=False or no mask).")

        # Debug Image Generation
        debug_image = torch.zeros_like(image[0:1])
        if show_debug_info:
            if all_inner_base_masks_for_debug and all_outer_base_masks_for_debug:
                debug_image = cls.create_debug_image(
                    image,
                    all_inner_base_masks_for_debug,
                    all_outer_base_masks_for_debug,
                    light_sources,
                    fg_mask,
                )
            else:
                logger.debug("  Skipping debug image: No base masks were generated/collected.")

        if not isinstance(debug_image, torch.Tensor) or debug_image.shape[1:] != image.shape[1:]:
            logger.warning("ReLight: debug image was unusable; returning a black placeholder.")
            debug_image = torch.zeros_like(image[0:1])
        elif debug_image.shape[0] != 1:
            debug_image = debug_image[0:1]

        if alpha is not None:
            final_result = torch.cat([final_result, alpha], dim=-1)

        logger.debug(f"--- ReLight processing finished in {time.time() - start_time:.3f} seconds ---")
        return io.NodeOutput(final_result, output_mask, debug_image)

    @classmethod
    def _scaled_gamma(cls, gamma, effect_strength):
        """Fade gamma toward identity with effect_strength.

        Gamma was previously the one correction the master strength did not
        touch, so effect_strength=0 still altered the image.
        """
        return 1.0 + (gamma - 1.0) * effect_strength


# --- v3 Extension Registration ---
class ReLightExtension(ComfyExtension):
    """ComfyUI Extension for ReLight node."""

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        """Return list of nodes provided by this extension."""
        return [ReLight]


async def comfy_entrypoint() -> ReLightExtension:
    """Entry point for ComfyUI v3."""
    return ReLightExtension()
