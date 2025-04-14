import torch
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance, ImageDraw, ImageFont
import comfy.utils
# SciPy is now required for optimal rim lighting
from scipy import ndimage
import math
import json
import os
import time  # For debugging
import inspect # For mask finding hack
import traceback # For detailed error logging


class ReLight:
    """
    Creates realistic lighting effects by applying color corrections or colored light
    to distinct areas of an image. Supports multiple light sources, colored lights,
    and 3D lighting simulation with subject occlusion. Requires SciPy.
    """

    # Define lighting presets
    PRESETS = {
        "None": {},
        "Soft Window Light": {
            "light_position_x": 0.7, "light_position_y": 0.3, "inner_circle_radius": 0.45, "outer_circle_radius": 0.8,
            "inner_brightness": 10, "inner_contrast": 0, "inner_saturation": 0, "inner_temperature": -5, "inner_tint": 0, "inner_gamma": 1.0,
            "outer_brightness": -15, "outer_contrast": 0, "outer_saturation": -10, "outer_temperature": -5, "outer_tint": 0, "outer_gamma": 1.2,
            "mask_blur": 80, "rim_amplification": 1.0
        },
        "Dramatic Side Light": {
            "light_position_x": 0.1, "light_position_y": 0.5, "inner_circle_radius": 0.3, "outer_circle_radius": 0.6,
            "inner_brightness": 15, "inner_contrast": 10, "inner_saturation": 10, "inner_temperature": 0, "inner_tint": 0, "inner_gamma": 0.9,
            "outer_brightness": -30, "outer_contrast": 10, "outer_saturation": -20, "outer_temperature": 0, "outer_tint": 0, "outer_gamma": 1.3,
            "mask_blur": 50, "rim_amplification": 1.0
        },
        "Warm Sunset Glow": {
            "light_position_x": 0.9, "light_position_y": 0.4, "inner_circle_radius": 0.5, "outer_circle_radius": 0.8,
            "inner_brightness": 3, "inner_contrast": 5, "inner_saturation": 10, "inner_temperature": 25, "inner_tint": -5, "inner_gamma": 0.95, # Reduced brightness
            "outer_brightness": -10, "outer_contrast": 0, "outer_saturation": -5, "outer_temperature": 15, "outer_tint": -5, "outer_gamma": 1.1,
            "mask_blur": 75, "use_colored_lights": True, "light_color_r": 255, "light_color_g": 200, "light_color_b": 120,
            "rim_amplification": 1.0, "use_gradient_mode": True # Added gradient mode
        },
        "Cool Blue Moonlight": {
            "light_position_x": 0.8, "light_position_y": 0.2, "inner_circle_radius": 0.4, "outer_circle_radius": 0.7,
            "inner_brightness": -5, "inner_contrast": 5, "inner_saturation": -5, "inner_temperature": -20, "inner_tint": 0, "inner_gamma": 1.1,
            "outer_brightness": -20, "outer_contrast": 0, "outer_saturation": -10, "outer_temperature": -30, "outer_tint": 0, "outer_gamma": 1.2,
            "mask_blur": 60, "use_colored_lights": True, "light_color_r": 120, "light_color_g": 150, "light_color_b": 255,
            "rim_amplification": 1.0
        },
        "Studio Key Light": {
            "light_position_x": 0.4, "light_position_y": 0.3, "inner_circle_radius": 0.6, "outer_circle_radius": 0.9,
            "inner_brightness": 12, "inner_contrast": 5, "inner_saturation": 0, "inner_temperature": 0, "inner_tint": 0, "inner_gamma": 1.0,
            "outer_brightness": -5, "outer_contrast": 0, "outer_saturation": -5, "outer_temperature": 0, "outer_tint": 0, "outer_gamma": 1.1,
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
        "Spotlight": { # New Preset
            "light_position_x": 0.5, "light_position_y": 0.4, "inner_circle_radius": 0.1, "outer_circle_radius": 0.25,
            "inner_brightness": 25, "inner_contrast": 15, "inner_saturation": -5, "inner_temperature": 0, "inner_tint": 0, "inner_gamma": 0.9,
            "outer_brightness": -40, "outer_contrast": 10, "outer_saturation": -20, "outer_temperature": 0, "outer_tint": 0, "outer_gamma": 1.3,
            "mask_blur": 30, "rim_amplification": 1.0, "effect_strength": 1.2
        },
        "Negative Light (Darken)": { # New Preset
            "light_position_x": 0.5, "light_position_y": 0.5, "inner_circle_radius": 0.4, "outer_circle_radius": 0.7,
            "inner_brightness": -20, "inner_contrast": 5, "inner_saturation": -5, "inner_temperature": 0, "inner_tint": 0, "inner_gamma": 1.1,
            "outer_brightness": 0, "outer_contrast": 0, "outer_saturation": 0, "outer_temperature": 0, "outer_tint": 0, "outer_gamma": 1.0,
            "mask_blur": 60, "rim_amplification": 1.0, "effect_strength": 1.0
        }
    }

    @classmethod
    def INPUT_TYPES(cls):
        """Define node inputs for ComfyUI, ordered logically."""
        # Reordered for better UX
        return {
            "required": {
                # --- Core Inputs ---
                "image": ("IMAGE", {"tooltip": "The input image to apply lighting effects to"}),
                "mask": ("MASK", {"tooltip": "Foreground mask (white=subject, black=background). Required for occlusion ('Apply 3D Lighting') and optional compositing ('Remove Background')."}),

                # --- Global Behavior ---
                "preset": (list(cls.PRESETS.keys()), {"tooltip": "Select a preset or 'None' for custom settings"}),
                "num_light_sources": ("INT", {"default": 1, "min": 1, "max": 3, "step": 1, "tooltip": "Number of light sources (1-3)"}),
                "preserve_positioning": ("BOOLEAN", {"default": True, "tooltip": "Keep manual light positions when changing presets?"}),
                "show_debug_info": ("BOOLEAN", {"default": False, "tooltip": "Output a debug visualization image?"}),

                # --- Lighting Mode & Occlusion ---
                "use_colored_lights": ("BOOLEAN", {"default": False, "tooltip": "Use additive colored light instead of color correction?"}),
                "use_gradient_mode": ("BOOLEAN", {"default": False, "tooltip": "Use directional gradient masks instead of radial?"}),
                "apply_3d_lighting": ("BOOLEAN", {"default": True, "tooltip": "Simulate light occlusion by subject? Requires 'mask' input."}),
                "light_direction": (["Behind Subject", "In Front of Subject", "No Occlusion"], {"default": "No Occlusion", "tooltip": "How light interacts with subject (Requires 'mask' and 'Apply 3D Lighting')"}),
                "remove_background": ("BOOLEAN", {"default": True, "tooltip": "Composite final result using mask? (Note: Ignored for 'Behind Subject' & 'In Front of Subject' directions)"}), # Updated tooltip

                # --- Global Modifiers ---
                "effect_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "Overall intensity multiplier for lighting adjustments/colors"}),
                "mask_blur": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 200.0, "step": 1.0, "tooltip": "Blur radius for light mask edges (smoother transitions)"}),
                "rim_amplification": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Intensity boost specifically for rim light component (when 'Behind Subject')"}),

                # --- Light 1 Settings ---
                # Position & Shape
                "light_position_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Light 1: Horizontal position (0=left, 1=right)"}),
                "light_position_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Light 1: Vertical position (0=top, 1=bottom)"}),
                "inner_circle_radius": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Light 1: Inner radius (strongest effect area)"}),
                "outer_circle_radius": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Light 1: Outer radius (falloff area)"}),
                # Colored Light Mode
                "light_color_r": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1, "tooltip": "Light 1: Red color (if 'Use Colored Lights' is True)"}),
                "light_color_g": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1, "tooltip": "Light 1: Green color (if 'Use Colored Lights' is True)"}),
                "light_color_b": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1, "tooltip": "Light 1: Blue color (if 'Use Colored Lights' is True)"}),
                "light_intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1, "tooltip": "Light 1: Intensity (if 'Use Colored Lights' is True)"}),
                # Color Correction Mode (Inner Area)
                "inner_brightness": ("FLOAT", {"default": 10.0, "min": -100.0, "max": 100.0, "step": 1.0, "tooltip": "Light 1: Inner area brightness (Color Correction mode)"}),
                "inner_contrast": ("FLOAT", {"default": 5.0, "min": -100.0, "max": 100.0, "step": 1.0, "tooltip": "Light 1: Inner area contrast (Color Correction mode)"}),
                "inner_saturation": ("FLOAT", {"default": 5.0, "min": -100.0, "max": 100.0, "step": 1.0, "tooltip": "Light 1: Inner area saturation (Color Correction mode)"}),
                "inner_temperature": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0, "tooltip": "Light 1: Inner area temperature (-100=cool, 100=warm)"}),
                "inner_tint": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0, "tooltip": "Light 1: Inner area tint (-100=magenta, 100=green)"}),
                "inner_gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.05, "tooltip": "Light 1: Inner area gamma"}),
                # Color Correction Mode (Outer Area)
                "outer_brightness": ("FLOAT", {"default": -10.0, "min": -100.0, "max": 100.0, "step": 1.0, "tooltip": "Light 1: Outer area brightness (Color Correction mode)"}),
                "outer_contrast": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0, "tooltip": "Light 1: Outer area contrast (Color Correction mode)"}),
                "outer_saturation": ("FLOAT", {"default": -10.0, "min": -100.0, "max": 100.0, "step": 1.0, "tooltip": "Light 1: Outer area saturation (Color Correction mode)"}),
                "outer_temperature": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0, "tooltip": "Light 1: Outer area temperature"}),
                "outer_tint": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0, "tooltip": "Light 1: Outer area tint"}),
                "outer_gamma": ("FLOAT", {"default": 1.1, "min": 0.1, "max": 5.0, "step": 0.05, "tooltip": "Light 1: Outer area gamma"}),
            },
            "optional": {
                # --- Light 2 Settings ---
                "light2_position_x": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Light 2: Horizontal position"}),
                "light2_position_y": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Light 2: Vertical position"}),
                "light2_inner_radius": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Light 2: Inner radius"}),
                "light2_outer_radius": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Light 2: Outer radius"}),
                "light2_color_r": ("INT", {"default": 180, "min": 0, "max": 255, "step": 1, "tooltip": "Light 2: Red color"}),
                "light2_color_g": ("INT", {"default": 180, "min": 0, "max": 255, "step": 1, "tooltip": "Light 2: Green color"}),
                "light2_color_b": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1, "tooltip": "Light 2: Blue color"}),
                "light2_intensity": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 3.0, "step": 0.1, "tooltip": "Light 2: Intensity (Colored mode)"}),
                # --- Light 3 Settings ---
                "light3_position_x": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Light 3: Horizontal position"}),
                "light3_position_y": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Light 3: Vertical position"}),
                "light3_inner_radius": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Light 3: Inner radius"}),
                "light3_outer_radius": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Light 3: Outer radius"}),
                "light3_color_r": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1, "tooltip": "Light 3: Red color"}),
                "light3_color_g": ("INT", {"default": 150, "min": 0, "max": 255, "step": 1, "tooltip": "Light 3: Green color"}),
                "light3_color_b": ("INT", {"default": 120, "min": 0, "max": 255, "step": 1, "tooltip": "Light 3: Blue color"}),
                "light3_intensity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 3.0, "step": 0.1, "tooltip": "Light 3: Intensity (Colored mode)"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("image", "mask", "debug_image")
    FUNCTION = "relight_image"
    CATEGORY = "image/lighting"

    # --- Utility Functions ---

    def _load_preset(self, preset_name, current_params):
        """Loads preset values, respecting 'preserve_positioning'."""
        # (Unchanged)
        if preset_name == "None" or preset_name not in self.PRESETS: return current_params
        preset_data = self.PRESETS[preset_name]; print(f"Applying preset: {preset_name}")
        updated_params = current_params.copy(); original_pos = { k: current_params.get(k) for k in current_params if 'position' in k }
        for key, value in preset_data.items():
            if key in updated_params: updated_params[key] = value
        if updated_params.get("preserve_positioning", True):
            print("  - Preserving user-defined light positions.")
            for key, value in original_pos.items():
                if value is not None: updated_params[key] = value
        return updated_params

    def create_circle_mask(self, width, height, center_x, center_y, radius):
        """Create a circular mask using NumPy."""
        # (Unchanged)
        y_coords, x_coords = np.mgrid[0:height, 0:width]; center_x_px, center_y_px = center_x * width, center_y * height
        radius_px = radius * min(width, height)
        if radius_px <= 0: return torch.zeros((height, width), dtype=torch.float32)
        dist_sq = (x_coords - center_x_px)**2 + (y_coords - center_y_px)**2
        mask = (dist_sq <= radius_px**2).astype(np.float32); return torch.from_numpy(mask)

    def create_gradient_mask(self, width, height, center_x, center_y, radius, direction_angle_deg=0):
        """Create a gradient mask with direction using NumPy."""
        # (Unchanged)
        y_coords, x_coords = np.mgrid[0:height, 0:width]; center_x_px, center_y_px = center_x * width, center_y * height
        radius_px = radius * min(width, height)
        if radius_px <= 0: return torch.zeros((height, width), dtype=torch.float32)
        delta_x, delta_y = x_coords - center_x_px, y_coords - center_y_px; distances = np.sqrt(delta_x**2 + delta_y**2)
        theta_rad = math.radians(direction_angle_deg); dir_x, dir_y = math.cos(theta_rad), math.sin(theta_rad)
        norm_dist = np.where(distances == 0, 1, distances); pos_norm_x, pos_norm_y = delta_x / norm_dist, delta_y / norm_dist
        directional_component = pos_norm_x * dir_x + pos_norm_y * dir_y; gradient_intensity = (directional_component + 1) / 2
        falloff = np.clip(1 - distances / radius_px, 0, 1)
        mask = np.where(distances <= radius_px, falloff * gradient_intensity, 0).astype(np.float32); return torch.from_numpy(mask)

    def apply_color_correction(self, image_tensor, brightness=0, contrast=0, saturation=0, temperature=0, tint=0, gamma=1.0):
        """Apply color correction adjustments using PIL."""
        # (Unchanged)
        corrected_batch = []; image_tensor_cpu = image_tensor.cpu()
        for i in range(image_tensor_cpu.shape[0]):
            img_np = (image_tensor_cpu[i].numpy() * 255).astype(np.uint8); pil_mode = 'RGB'; pil_img = None
            if len(img_np.shape) == 3 and img_np.shape[2] == 1: pil_img = Image.fromarray(img_np.squeeze(-1), mode='L'); pil_mode = 'L'
            elif len(img_np.shape) == 3 and img_np.shape[2] >= 3: pil_img = Image.fromarray(img_np[..., :3], mode='RGB')
            elif len(img_np.shape) == 2: pil_img = Image.fromarray(img_np, mode='L'); pil_mode = 'L'
            else: print(f"Warn: Unexpected shape {img_np.shape}"); corrected_batch.append(image_tensor_cpu[i]); continue
            if abs(brightness) > 0.1: pil_img = ImageEnhance.Brightness(pil_img).enhance(1.0 + (brightness / 100.0))
            if abs(contrast) > 0.1: pil_img = ImageEnhance.Contrast(pil_img).enhance(1.0 + (contrast / 100.0))
            if abs(saturation) > 0.1 and pil_mode == 'RGB': pil_img = ImageEnhance.Color(pil_img).enhance(1.0 + (saturation / 100.0))
            if (abs(temperature) > 0.1 or abs(tint) > 0.1) and pil_mode == 'RGB':
                r, g, b = pil_img.split(); temp_r, temp_b = 1.0+(temperature/200.0), 1.0-(temperature/200.0); tint_g = 1.0+(tint/200.0)
                r,b,g=r.point(lambda x:np.clip(x*temp_r,0,255)),b.point(lambda x:np.clip(x*temp_b,0,255)),g.point(lambda x:np.clip(x*tint_g,0,255))
                pil_img = Image.merge('RGB', (r, g, b))
            if abs(gamma - 1.0) > 0.01:
                np_float = np.array(pil_img).astype(np.float32)/255.0; safe_gamma = max(gamma, 0.01)
                corrected = np.power(np.clip(np_float, 0.0, 1.0), 1.0 / safe_gamma)
                pil_img = Image.fromarray((np.clip(corrected, 0.0, 1.0) * 255).astype(np.uint8), mode=pil_img.mode)
            np_corrected = np.array(pil_img).astype(np.float32) / 255.0
            if pil_mode == 'L' and len(image_tensor_cpu[i].shape) == 3:
                 np_corrected = np.expand_dims(np_corrected, axis=-1)
                 if image_tensor_cpu[i].shape[-1] == 3: np_corrected = np.repeat(np_corrected, 3, axis=-1)
            corrected_batch.append(torch.from_numpy(np_corrected))
        return torch.stack(corrected_batch).to(image_tensor.device)

    def apply_mask_blur(self, mask_tensor, blur_amount):
        """Apply Gaussian blur to a mask tensor using PIL."""
        # (Unchanged)
        if blur_amount <= 0.1: return mask_tensor
        mask_np = mask_tensor.cpu().numpy(); blur_radius = blur_amount / 5.0
        if blur_radius <= 0: return mask_tensor
        mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
        try: mask_pil_blurred = mask_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        except ValueError: print(f"Warn: Blur failed (r={blur_radius})"); return mask_tensor
        mask_np_blurred = np.array(mask_pil_blurred).astype(np.float32) / 255.0
        return torch.from_numpy(mask_np_blurred).to(mask_tensor.device)

    def apply_colored_light(self, image, mask, color_rgb, intensity=1.0):
        """Apply additive colored light using a mask."""
        # (Unchanged)
        if intensity <= 0: return image
        color_norm = torch.tensor([c / 255.0 for c in color_rgb], device=image.device, dtype=torch.float32)
        color_light = color_norm.view(1, 1, 1, 3)
        if len(mask.shape) == 3: mask = mask.unsqueeze(-1)
        if mask.shape[-1] == 1: mask = mask.expand_as(image)
        result = image + color_light * intensity * mask
        return torch.clamp(result, 0.0, 1.0)

    def calculate_rim_mask(self, light_mask_np, fg_mask_np, light_position_x, light_position_y):
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
        # Removed SciPy optional check - now required
        height, width = fg_mask_np.shape

        # 1. Edge Detection using Sobel filter
        edge_x = ndimage.sobel(fg_mask_np, axis=1)
        edge_y = ndimage.sobel(fg_mask_np, axis=0)
        edge_magnitude = np.sqrt(edge_x**2 + edge_y**2)

        # Normalize edge magnitude to 0-1 range
        max_edge = edge_magnitude.max()
        if max_edge > 1e-6: # Avoid division by zero if mask is flat
             edge_magnitude /= max_edge
        else:
            # If no edges are found, return an empty mask
            print("Warn: No edges detected for rim mask.")
            return np.zeros_like(light_mask_np)

        # Enhance edges slightly using a power function (value < 1 thickens/brightens)
        edge_mask = np.power(edge_magnitude, 0.7)

        # 2. Base Rim Light: Modulate edge mask by the original light intensity
        # This ensures rim light only appears where the original light would hit the edge
        rim_light_raw = light_mask_np * edge_mask

        # 3. Directional Modulation: Make rim brighter where light hits edge from behind
        y_grid, x_grid = np.mgrid[0:height, 0:width]; light_x_px, light_y_px = light_position_x * width, light_position_y * height
        light_dir_x, light_dir_y = x_grid - light_x_px, y_grid - light_y_px; light_dist = np.sqrt(light_dir_x**2 + light_dir_y**2)
        light_dist = np.where(light_dist < 1e-6, 1, light_dist); light_dir_x /= light_dist; light_dir_y /= light_dist
        grad_magnitude_norm = np.sqrt(edge_x**2 + edge_y**2); grad_magnitude_norm = np.where(grad_magnitude_norm < 1e-6, 1, grad_magnitude_norm)
        normal_x, normal_y = edge_x / grad_magnitude_norm, edge_y / grad_magnitude_norm
        dot_product = light_dir_x * normal_x + light_dir_y * normal_y
        directional_factor = np.clip((-dot_product + 1) / 2, 0, 1); directional_factor = np.power(directional_factor, 1.5)
        rim_light_modulated = rim_light_raw * directional_factor; final_mask_np = np.clip(rim_light_modulated, 0.0, 1.0)

        print(f"    - Raw rim mask calculated: Max intensity = {final_mask_np.max():.3f}"); return final_mask_np


    def create_debug_image(self, original_image, all_inner_base_masks, all_outer_base_masks, light_sources, fg_mask=None):
        """Create a debug visualization showing base masks and light positions."""
        # Reverted to show base masks like reference image
        print("--- Creating Debug Image (v7 - Base Masks) ---") # Version updated
        try:
            img_tensor = original_image[0].cpu()
            if fg_mask is not None: fg_mask_tensor = fg_mask[0].cpu()
            else: fg_mask_tensor = None
            # Use base masks for the *first* light source for visualization
            inner_base_mask = all_inner_base_masks[0].cpu() if all_inner_base_masks else None
            outer_base_mask = all_outer_base_masks[0].cpu() if all_outer_base_masks else None

            img_np = (img_tensor.numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np).convert('RGBA')
            width, height = pil_img.size
            print(f"  Base image size: {width}x{height}")

            # --- Create Overlays ---
            inner_overlay_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            ring_overlay_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            fg_overlay_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))

            # Inner Mask Overlay (Red)
            try:
                if inner_base_mask is not None:
                    inner_np = inner_base_mask.numpy()
                    inner_alpha = (np.clip(inner_np, 0, 1) * 128).astype(np.uint8) # Alpha 0-128
                    if np.max(inner_alpha) > 0:
                        inner_overlay_np = np.zeros((height, width, 4), dtype=np.uint8)
                        inner_overlay_np[..., 0] = 255 # R
                        inner_overlay_np[..., 3] = inner_alpha
                        inner_overlay_img = Image.fromarray(inner_overlay_np, 'RGBA')
                        print(f"  Inner mask overlay created. Max Alpha: {np.max(inner_alpha)}")
                    else: print("  Skipping inner mask overlay (empty).")
                else: print("  Skipping inner mask overlay (no mask provided).")
            except Exception as e: print(f"  ERROR creating inner mask overlay: {e}"); traceback.print_exc()

            # Ring Mask Overlay (Blue)
            try:
                if inner_base_mask is not None and outer_base_mask is not None:
                    # Calculate ring mask: outer - inner
                    ring_np = np.clip(outer_base_mask.numpy() - inner_base_mask.numpy(), 0, 1)
                    ring_alpha = (ring_np * 128).astype(np.uint8) # Alpha 0-128
                    if np.max(ring_alpha) > 0:
                        ring_overlay_np = np.zeros((height, width, 4), dtype=np.uint8)
                        ring_overlay_np[..., 2] = 255 # B
                        ring_overlay_np[..., 3] = ring_alpha
                        ring_overlay_img = Image.fromarray(ring_overlay_np, 'RGBA')
                        print(f"  Ring mask overlay created. Max Alpha: {np.max(ring_alpha)}")
                    else: print("  Skipping ring mask overlay (empty).")
                else: print("  Skipping ring mask overlay (missing base masks).")
            except Exception as e: print(f"  ERROR creating ring mask overlay: {e}"); traceback.print_exc()

            # Foreground Mask Overlay (Green)
            try:
                if fg_mask_tensor is not None:
                    fg_mask_np = fg_mask_tensor.numpy()
                    fg_alpha = (fg_mask_np * 100).astype(np.uint8) # Alpha 0-100
                    if np.max(fg_alpha) > 0:
                        print(f"  FG mask stats: Min={np.min(fg_mask_np):.3f}, Max={np.max(fg_mask_np):.3f}, Mean={np.mean(fg_mask_np):.3f}")
                        fg_overlay_np = np.zeros((height, width, 4), dtype=np.uint8); fg_overlay_np[..., 1] = 255; fg_overlay_np[..., 3] = fg_alpha
                        fg_overlay_img = Image.fromarray(fg_overlay_np, 'RGBA'); print(f"  FG mask overlay created. Max Alpha: {np.max(fg_alpha)}")
                    else: print("  Skipping FG mask overlay (mask is empty).")
            except Exception as e: print(f"  ERROR creating FG mask overlay: {e}"); traceback.print_exc()

            # --- Composite Overlays ---
            debug_img = pil_img # Start with original
            try:
                # Composite order: Base -> FG (Green) -> Inner (Red) -> Ring (Blue)
                debug_img = Image.alpha_composite(debug_img, fg_overlay_img)
                debug_img = Image.alpha_composite(debug_img, inner_overlay_img)
                debug_img = Image.alpha_composite(debug_img, ring_overlay_img)
                print("  Overlays composited.")
            except Exception as e: print(f"  ERROR compositing overlays: {e}"); traceback.print_exc()

            # --- Draw Indicators & Legend ---
            draw_debug = ImageDraw.Draw(debug_img); font = None; font_size = 10
            try: font = ImageFont.load_default(size=12); font_size = 12
            except Exception:
                try: font = ImageFont.load_default(); font_size = 10
                except Exception as font_err: print(f"  Warn: Could not load default font: {font_err}")
            print(f"  Using font: {type(font)}, size: {font_size}")
            print(f"  Drawing indicators for {len(light_sources)} light source(s)...")
            for i, light in enumerate(light_sources):
                try:
                    x, y = int(light["position_x"] * width), int(light["position_y"] * height); color = tuple(light.get("color", [255, 255, 255])) + (220,)
                    # Draw center point and radii lines (like reference image)
                    inner_r_px = int(light["inner_radius"] * min(width, height))
                    outer_r_px = int(light["outer_radius"] * min(width, height))
                    draw_debug.ellipse((x-5, y-5, x+5, y+5), fill=color, outline=(0,0,0,200))
                    # Dashed lines for radii (simplified)
                    draw_debug.ellipse((x-inner_r_px, y-inner_r_px, x+inner_r_px, y+inner_r_px), outline=(255, 255, 0, 150), width=1) # Inner = Yellow dashed (approx)
                    draw_debug.ellipse((x-outer_r_px, y-outer_r_px, x+outer_r_px, y+outer_r_px), outline=(0, 255, 255, 150), width=1) # Outer = Cyan dashed (approx)
                    label = f"L{i+1}"; text_pos = (x + 10, y - font_size // 2 - 2)
                    if font: bbox = draw_debug.textbbox(text_pos, label, font=font); draw_debug.rectangle(bbox, fill=(0, 0, 0, 180)); draw_debug.text(text_pos, label, fill=(255, 255, 255, 230), font=font)
                    else: draw_debug.text(text_pos, label, fill=(255, 255, 255, 230))
                except Exception as draw_err: print(f"    ERROR drawing indicator/text for light {i+1}: {draw_err}"); traceback.print_exc()
            print("  Light indicators drawn.")
            try:
                legend_items = [ ("Inner Mask Area", (255, 0, 0, 128)), ("Outer Mask Area (Ring)", (0, 0, 255, 128)) ] # Red, Blue
                if fg_mask_tensor is not None: legend_items.append(("Foreground Mask", (0, 255, 0, 100))) # Green
                legend_x, legend_y = 10, 10; line_height = font_size + 6; max_width = 0
                for text, _ in legend_items:
                    try: text_w = draw_debug.textlength(text, font=font) if font else len(text)*7
                    except Exception: text_w = len(text) * 7
                    max_width = max(max_width, text_w)
                legend_box = (legend_x - 5, legend_y - 5, legend_x + max_width + 25, legend_y + len(legend_items) * line_height)
                draw_debug.rectangle(legend_box, fill=(0, 0, 0, 190))
                for text, color in legend_items:
                    draw_debug.rectangle((legend_x, legend_y, legend_x + 12, legend_y + 12), fill=color)
                    draw_debug.text((legend_x + 18, legend_y + 1), text, fill=(255, 255, 255, 220), font=font); legend_y += line_height
                print("  Legend drawn.")
            except Exception as legend_err: print(f"  ERROR drawing legend: {legend_err}"); traceback.print_exc()

            # Final conversion
            debug_img_rgb = debug_img.convert('RGB'); debug_np = np.array(debug_img_rgb).astype(np.float32) / 255.0
            # **** DEBUG: Check final numpy array shape ****
            print(f"  Final debug numpy array shape before tensor conversion: {debug_np.shape}")
            # **** END DEBUG ****
            print("--- Debug Image Creation Finished ---")
            # Ensure output tensor has batch dimension and correct device
            return torch.from_numpy(debug_np).unsqueeze(0).to(original_image.device)

        except Exception as e:
            print(f"--- FATAL ERROR in create_debug_image: {e} ---"); traceback.print_exc()
            return torch.zeros_like(original_image[0:1]) # Return black image on any major error

    # --- Main Execution Function ---

    def relight_image(self, image: torch.Tensor, **kwargs):
        """Applies relighting effects to the input image based on parameters."""
        start_time = time.time()
        image = image.to(dtype=torch.float32)
        batch_size, height, width, channels = image.shape
        device = image.device

        print(f"\n--- ReLight Node ---")
        print(f"Input Image: {width}x{height}, Batch: {batch_size}, Device: {device}, Shape: {image.shape}") # Log shape

        params = kwargs.copy()
        params = self._load_preset(params.get('preset', 'None'), params)

        # Extract parameters using the reordered structure
        preset=params.get('preset','None')
        remove_background=params.get('remove_background',True)
        apply_3d_lighting=params.get('apply_3d_lighting',True)
        light_direction=params.get('light_direction','No Occlusion')
        effect_strength=params.get('effect_strength',1.0)
        rim_amplification=params.get('rim_amplification',2.0) # Default reduced
        num_light_sources=params.get('num_light_sources',1)
        use_colored_lights=params.get('use_colored_lights',False)
        use_gradient_mode=params.get('use_gradient_mode',False)
        mask_blur=params.get('mask_blur',50.0)
        show_debug_info=params.get('show_debug_info',False)
        input_mask=params.get('mask',None) # Mask is now required, but keep checking None for safety

        print(f"Mode: Preset='{preset}', 3D Lighting={apply_3d_lighting}, Direction='{light_direction}', Colored={use_colored_lights}, Gradient={use_gradient_mode}")
        print(f"Settings: Strength={effect_strength:.2f}, Rim Amp={rim_amplification:.2f}, Mask Blur={mask_blur:.1f}, Debug={show_debug_info}")

        # --- Mask Handling ---
        fg_mask=None; original_input_mask=None
        # Mask is needed if applying 3D lighting or removing background
        needs_mask = apply_3d_lighting or remove_background

        if needs_mask:
            print("Processing mask input...")
            if input_mask is not None:
                print(f"  - Using mask. Shape:{input_mask.shape}, Type:{input_mask.dtype}")
                original_input_mask=input_mask.clone()
                fg_mask=input_mask.to(device=device,dtype=torch.float32)
                # Standardize mask shape
                if len(fg_mask.shape)==4:
                    if fg_mask.shape[3]==1: fg_mask=fg_mask.squeeze(-1)
                    elif fg_mask.shape[3]==4: fg_mask=fg_mask[...,3]
                    elif fg_mask.shape[3]==3: fg_mask=torch.mean(fg_mask,dim=-1)
                    else: fg_mask=fg_mask[...,0]
                elif len(fg_mask.shape)==2: fg_mask=fg_mask.unsqueeze(0)
                # Match batch size
                if fg_mask.shape[0]!=batch_size:
                    print(f"  - Warn: Mask batch mismatch. Repeating.")
                    fg_mask=fg_mask[0:1].repeat(batch_size,1,1)
                # Check inversion
                mean_val=torch.mean(fg_mask).item()
                if mean_val>0.8 and mean_val<0.999:
                    print(f"  - Mask inverted (mean={mean_val:.3f}).")
                    fg_mask=1.0-fg_mask
                print(f"  - Final fg_mask shape:{fg_mask.shape}, Min={torch.min(fg_mask):.3f}, Max={torch.max(fg_mask):.3f}, Mean={torch.mean(fg_mask):.3f}")
            else:
                # If mask is needed but not provided, disable features that require it
                print("  - Error: Mask input required for 'Apply 3D Lighting' or 'Remove Background'. Disabling these features.")
                apply_3d_lighting = False
                remove_background = False # Set remove_background to false as well
                original_input_mask=torch.zeros((batch_size,height,width),device=device,dtype=torch.float32)
        else:
            # If mask is not strictly needed, still assign a default black mask for output consistency
            original_input_mask=torch.zeros((batch_size,height,width),device=device,dtype=torch.float32)
            # Ensure apply_3d_lighting is off if no mask was provided, even if not strictly needed initially
            if input_mask is None:
                apply_3d_lighting = False


        # --- Define Light Sources ---
        # (Unchanged - Syntax fix already applied)
        light_sources = []
        for i in range(1, num_light_sources + 1):
            prefix = f"light{i}_" if i > 1 else "light_"
            pos_x_key,pos_y_key = ("light_position_x", "light_position_y") if i == 1 else (f"light{i}_position_x", f"light{i}_position_y")
            inner_r_key,outer_r_key = ("inner_circle_radius", "outer_circle_radius") if i == 1 else (f"light{i}_inner_radius", f"light{i}_outer_radius")
            color_r_key,color_g_key,color_b_key = f"{prefix}color_r", f"{prefix}color_g", f"{prefix}color_b"
            intensity_key = f"{prefix}intensity"
            if params.get(pos_x_key) is None or params.get(pos_y_key) is None:
                if i > 1: continue
                else: raise ValueError("Missing essential parameters for Light 1")
            light = { "id": i, "position_x": params.get(pos_x_key, 0.5), "position_y": params.get(pos_y_key, 0.5),
                      "inner_radius": params.get(inner_r_key, 0.3), "outer_radius": params.get(outer_r_key, 0.6),
                      "color": [int(params.get(color_r_key, 255)), int(params.get(color_g_key, 255)), int(params.get(color_b_key, 255))],
                      "intensity": params.get(intensity_key, 1.0) }
            light_sources.append(light); print(f"Defined Light {i}: Pos=({light['position_x']:.2f},{light['position_y']:.2f}), Radii=({light['inner_radius']:.2f},{light['outer_radius']:.2f}), Color={light['color']}, Intensity={light['intensity']:.2f}")


        # --- Initialize Result & Masks for Debug ---
        result_tensor = image.clone()
        all_inner_base_masks_for_debug = [] # Store base masks for debug
        all_outer_base_masks_for_debug = []

        # --- Process Each Light Source ---
        for light in light_sources:
            print(f"\nProcessing Light Source {light['id']}...")

            # --- Create Base Light Masks ---
            inner_mask_base = self.create_circle_mask(width, height, light["position_x"], light["position_y"], light["inner_radius"]).to(device)
            outer_mask_base = self.create_circle_mask(width, height, light["position_x"], light["position_y"], light["outer_radius"]).to(device)
            if use_gradient_mode:
                 center_x, center_y = 0.5, 0.5; dx = light["position_x"] - center_x; dy = light["position_y"] - center_y; angle = math.degrees(math.atan2(dy, dx))
                 inner_mask_base = self.create_gradient_mask(width, height, light["position_x"], light["position_y"], light["inner_radius"], angle).to(device)
                 outer_mask_base = self.create_gradient_mask(width, height, light["position_x"], light["position_y"], light["outer_radius"], angle).to(device)

            # Store base masks for potential debug use
            all_inner_base_masks_for_debug.append(inner_mask_base)
            all_outer_base_masks_for_debug.append(outer_mask_base)

            # --- MASK APPLICATION LOGIC ---
            final_light_mask = torch.zeros_like(inner_mask_base)
            occlusion_active = apply_3d_lighting and fg_mask is not None # Check if mask is actually available

            if occlusion_active and light_direction == "Behind Subject":
                print("  Processing as Behind Subject (Rim + Background Glow)...")
                fg_mask_np = fg_mask[0].cpu().numpy()
                light_mask_np = outer_mask_base.cpu().numpy() # Base for rim calculation

                # Calculate Raw Rim Mask
                raw_rim_mask_np = self.calculate_rim_mask(light_mask_np, fg_mask_np, light["position_x"], light["position_y"])
                # Amplify the raw rim mask component
                amplified_raw_rim_mask_np = np.clip(raw_rim_mask_np * rim_amplification, 0.0, 1.0)
                amplified_raw_rim_mask = torch.from_numpy(amplified_raw_rim_mask_np).to(device)
                print(f"    - Amplified raw rim mask: Max={torch.max(amplified_raw_rim_mask):.3f}")

                # Calculate Background Light Mask (occluded by subject)
                background_base_mask = outer_mask_base
                background_light_mask = background_base_mask * (1.0 - fg_mask[0])
                print(f"    - Background light mask calculated: Max={torch.max(background_light_mask):.3f}")

                # Combine Amplified Rim and Background masks (unblurred)
                combined_mask_unblurred = torch.clamp(amplified_raw_rim_mask + background_light_mask, 0, 1)
                print(f"    - Combined Rim(Amp)+BG mask (unblurred): Max={torch.max(combined_mask_unblurred):.3f}")

                # Apply blur to the combined mask
                final_light_mask = self.apply_mask_blur(combined_mask_unblurred, mask_blur) if mask_blur > 0.1 else combined_mask_unblurred
                print(f"  Final Behind Subject Mask (Blurred): Max={torch.max(final_light_mask):.3f}, Mean={torch.mean(final_light_mask):.3f}")

            elif occlusion_active and light_direction == "In Front of Subject":
                 print("  Processing as Front Subject Light...")
                 fg_mask_np = fg_mask[0].cpu().numpy()
                 light_mask_np = outer_mask_base.cpu().numpy() # Base light area

                 enhance_subject_factor = 1.3; reduce_background_factor = 0.8
                 occlusion_factor_mask_np = fg_mask_np * enhance_subject_factor + (1.0 - fg_mask_np) * reduce_background_factor
                 # Removed scipy check - now required
                 occlusion_factor_mask_np = ndimage.gaussian_filter(occlusion_factor_mask_np, sigma=2)

                 combined_mask_unblurred_np = light_mask_np * occlusion_factor_mask_np
                 combined_mask_unblurred = torch.from_numpy(np.clip(combined_mask_unblurred_np, 0, 1)).to(device)

                 final_light_mask = self.apply_mask_blur(combined_mask_unblurred, mask_blur) if mask_blur > 0.1 else combined_mask_unblurred
                 print(f"  Final Front Subject Mask: Max={torch.max(final_light_mask):.3f}, Mean={torch.mean(final_light_mask):.3f}")

            else: # Standard (No Occlusion) or Occlusion Inactive
                if occlusion_active: print("  Processing as Standard Light (Occlusion Direction != Behind/Front)...")
                else: print("  Processing as Standard Light (Occlusion Inactive)...")

                # Calculate standard inner/outer falloff mask
                ring_mask_base = torch.clamp(outer_mask_base - inner_mask_base, 0, 1)
                combined_mask_unblurred = torch.clamp(inner_mask_base + ring_mask_base, 0, 1) # Represents full light area with falloff
                final_light_mask = self.apply_mask_blur(combined_mask_unblurred, mask_blur) if mask_blur > 0.1 else combined_mask_unblurred
                print(f"  Final Standard Mask: Max={torch.max(final_light_mask):.3f}, Mean={torch.mean(final_light_mask):.3f}")

            # --- Apply Lighting Effect using the single final_light_mask ---
            if torch.max(final_light_mask) > 1e-4: # Only apply if mask has significant values
                final_mask_expanded = final_light_mask.unsqueeze(0).unsqueeze(-1).expand_as(result_tensor)
                if use_colored_lights:
                    print(f"  Applying colored light (RGB: {light['color']}, Intensity: {light['intensity']:.2f})...")
                    effective_intensity = light['intensity'] * effect_strength
                    result_tensor = self.apply_colored_light(result_tensor, final_mask_expanded, light['color'], effective_intensity)
                else: # Color Correction
                    print("  Applying color correction...")
                    # Use INNER parameters when applying a single combined mask
                    # This is a simplification for Behind/Front modes.
                    # Only "No Occlusion" mode uses outer params separately via ring_mask_base.
                    inner_brightness = params.get('inner_brightness',0)*effect_strength; inner_contrast=params.get('inner_contrast',0)*effect_strength
                    inner_saturation=params.get('inner_saturation',0)*effect_strength; inner_temperature=params.get('inner_temperature',0)*effect_strength
                    inner_tint=params.get('inner_tint',0)*effect_strength; inner_gamma=params.get('inner_gamma',1.0)
                    print(f"    Using Inner Corr Params: B={inner_brightness:.1f} C={inner_contrast:.1f} S={inner_saturation:.1f} ...")
                    corrected_image = self.apply_color_correction(result_tensor, inner_brightness, inner_contrast, inner_saturation, inner_temperature, inner_tint, inner_gamma)
                    result_tensor = result_tensor * (1.0 - final_mask_expanded) + corrected_image * final_mask_expanded
            else:
                 print("  Skipping light application (final mask is near-empty).")


        # --- Final Steps ---
        # Compositing (Conditional based on light_direction)
        final_result = result_tensor # Default to the fully processed tensor

        if remove_background and fg_mask is not None:
            # *** ADJUSTED COMPOSITING LOGIC ***
            if light_direction == "No Occlusion":
                print("Compositing lit foreground onto original background ('No Occlusion' mode)...")
                fg_mask_expanded = fg_mask.unsqueeze(-1).expand_as(image)
                final_result = result_tensor * fg_mask_expanded + image * (1.0 - fg_mask_expanded)
                print("  - Compositing complete.")
            else: # "Behind Subject" or "In Front of Subject"
                print(f"Skipping final compositing step for '{light_direction}' mode (lighting applied to FG/BG directly).")
                final_result = result_tensor # Keep the result which includes lit BG/FG interaction
            # *** END ADJUSTED COMPOSITING LOGIC ***
        else:
             # If remove_background is false, always keep the full result
             print("Skipping final compositing (remove_background=False or no mask).")
             final_result = result_tensor


        # Debug Image Generation
        debug_image = torch.zeros_like(image[0:1]) # Default black placeholder
        if show_debug_info:
            print("Attempting debug visualization...")
            if all_inner_base_masks_for_debug and all_outer_base_masks_for_debug:
                 fg_mask_for_debug = fg_mask if fg_mask is not None else None
                 debug_image = self.create_debug_image(
                     image,
                     all_inner_base_masks_for_debug, # Pass BASE inner masks
                     all_outer_base_masks_for_debug, # Pass BASE outer masks
                     light_sources,
                     fg_mask_for_debug
                 )
                 print(f"  - Debug image generation attempted.") # Log attempt
            else: print("  - Skipping debug image: No base masks were generated/collected.")
        else: print("Debug visualization disabled.")

        elapsed_time = time.time() - start_time
        print(f"--- ReLight processing finished in {elapsed_time:.3f} seconds ---")

        if original_input_mask is None: original_input_mask = torch.zeros((batch_size, height, width), device=device, dtype=torch.float32)
        if not isinstance(debug_image, torch.Tensor):
             print("Warn: Debug image was not a tensor. Returning black placeholder.")
             debug_image = torch.zeros_like(image[0:1])
        # Ensure debug image has correct shape and batch size before returning
        # *** DEBUG SHAPE CHECK FIX ***
        elif debug_image.shape[1:] != image.shape[1:]: # Compare (H, W, C) vs (H, W, C)
             print(f"Warn: Debug image shape mismatch {debug_image.shape[1:]} vs image {image.shape[1:]}. Returning black placeholder.")
             debug_image = torch.zeros_like(image[0:1])
        # *** END DEBUG SHAPE CHECK FIX ***
        elif debug_image.shape[0] != 1:
             print(f"Warn: Debug image has batch size {debug_image.shape[0]}. Taking first item.")
             debug_image = debug_image[0:1]

        return (final_result, original_input_mask, debug_image)

# --- Node Registration ---
NODE_CLASS_MAPPINGS = { "ReLight": ReLight }
NODE_DISPLAY_NAME_MAPPINGS = { "ReLight": "ReLight 💡" }