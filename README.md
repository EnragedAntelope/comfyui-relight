# ✨ ReLight Node for ComfyUI

![Platform](https://img.shields.io/badge/Platform-ComfyUI-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![ComfyUI v3](https://img.shields.io/badge/ComfyUI-v3%20Compatible-green)

> **Relight your images without re-generating them.**

ReLight is a single, self-contained ComfyUI node that adds up to 3 positionable light sources to any image — colored additive light or precise color correction, with presets, directional gradients, rim lighting, and mask-aware 3D occlusion. It's fast and deterministic: pure image processing, no diffusion pass, no models to download.

**Built on the ComfyUI v3 node schema.**
![ReLight Node Example](https://github.com/user-attachments/assets/34fa5b9f-65e6-4953-8bd4-65a349ed9455)

## 🌟 Features

### Powerful Lighting Control

- **Multiple Light Sources** - Place up to 3 independent light sources anywhere in your image
- **Dynamic Lighting Modes**:
  - 🎨 **Colored Lights** - Add RGB light with controllable intensity
  - 🔄 **Color Correction** - Apply precise adjustments to brightness, contrast, saturation, temperature, tint and gamma
- **Flexible Mask Shapes**:
  - 🔵 **Circular Falloff** - Natural radial lighting with inner/outer radius control
  - ↗️ **Gradient** - Directional lighting for effects like sunset rays or window light

### Advanced 3D Lighting Simulation

- **Subject Interaction** (when used with mask input):
  - 🔆 **Front Lighting** - Light illuminates the subject more strongly than background
  - ✨ **Rim Lighting** - Creates dramatic edge highlighting with background glow
  - 🌐 **Standard Lighting** - Traditional lighting without subject occlusion

### Production-Ready Features

- **Ready-to-Use Presets** for instant professional results:
  - "Soft Window Light" - Natural diffused lighting
  - "Dramatic Side Light" - Cinematic chiaroscuro effect
  - "Warm Sunset Glow" - Golden hour atmosphere
  - "Cool Blue Moonlight" - Mysterious night-time look
  - "Studio Key Light" - Professional portrait lighting
  - "Rim Light (Behind)" - Striking edge highlights
  - "Spotlight" - Focused dramatic lighting
  - "Negative Light (Darken)" - Creative darkening effects

- **Visual Debugging** - See exactly where your lights are positioned and how they interact
- **Fine-Tuning Controls** - Perfect your lighting with precision adjustments for blur, strength, and rim amplification

## 🔧 Installation

### Using ComfyUI-Manager (Recommended)

1. Open ComfyUI and navigate to the Manager
2. Search for "ReLight" in the available custom nodes
3. Click Install
4. Restart ComfyUI

### Manual Installation

```bash
# Navigate to your ComfyUI custom_nodes directory
cd path/to/ComfyUI/custom_nodes

# Clone this repository
git clone https://github.com/EnragedAntelope/comfyui-relight

# Install dependencies
pip install -r comfyui-relight/requirements.txt

# Restart ComfyUI
```

### Requirements

**ComfyUI 0.3.48 or newer.** ReLight is built on the ComfyUI v3 node schema (`comfy_api`), which first shipped in that release. On older builds the node will not load.

ReLight needs `numpy`, `Pillow`, and `scipy` (installed automatically from `requirements.txt`; `torch` is provided by ComfyUI itself).

ReLight works best with high-quality foreground masks. We recommend installing:

- **[ComfyUI Essentials](https://github.com/cubiq/ComfyUI_essentials)** - Provides enhanced mask generation and background removal tools

## 🚀 Quick Start Guide

1. **Add the ReLight 💡 node** to your workflow (found under category "image/lighting")
2. **Connect your source image**
3. **Connect a foreground mask** (white = subject, black = background) — optional, but required for occlusion ("Behind Subject" / "In Front of Subject") and for `remove_background` compositing. It is resized automatically if it does not match the image
4. **Select a preset** like "Rim Light (Behind)" or design your own lighting
5. **Adjust settings** to taste
6. **Preview your results** in real-time

> **Note on presets:** a preset overrides the widgets below it. The values shown on the node are ignored for whatever the preset defines, so don't be surprised when editing `inner_brightness` does nothing while a preset is active. Set `preset` back to `None` to tune by hand, or turn on `preserve_positioning` to keep your own light positions and radii while the preset supplies everything else. The one exception is `effect_strength`: it *scales* the preset rather than being replaced by it, so `1.0` is the preset as designed, `0.5` is half-strength and `0.0` is untouched.

### Sample Workflow

A ready-to-load workflow is included: [example_workflows/relight_basic.json](https://github.com/EnragedAntelope/comfyui-relight/blob/main/example_workflows/relight_basic.json)

The repository includes a sample workflow that demonstrates:

1. Loading an image
2. Removing the background using ComfyUI Essentials' RemBG nodes
3. Applying the ReLight node with "Warm Sunset Glow" preset in "Behind Subject" mode
4. Viewing the results through both standard preview and debug visualization

Simply load this workflow in ComfyUI to see ReLight in action!

## 📸 Examples

### Dramatic Three-Point RGB Lighting

![Before/After Three-Point RGB Lighting (_in case you really want to overdo things_)](https://github.com/user-attachments/assets/65024b82-2ba2-465e-9505-ca2025b93764)

This example uses three colored lights to create a purposely over the top striking RGB lighting setup:

- **Main Settings**: 
  - 3 light sources
  - "Behind Subject" light direction
  - 2.0 effect strength

- **Light 1 (Red)**: 
  - Position: far right (0.99, 0.15)
  - RGB Color: (255, 0, 0)
  - High intensity (2.0)

- **Light 2 (Green)**:
  - Position: left side (0.2, 0.3)
  - RGB Color: (0, 255, 0)
  - Medium intensity (0.7)

- **Light 3 (Blue)**:
  - Position: bottom center (0.3, 0.8)
  - RGB Color: (0, 0, 255)
  - Low intensity (0.2)

This setup creates vibrant color separation while maintaining the "Behind Subject" mode to emphasize the edges of the figure with dramatic rim lighting.

### Other Lighting Ideas to Try

Here are some additional lighting scenarios that showcase ReLight's versatility:

#### Split Lighting Portrait
![Brighten a subject and make it pop](https://github.com/user-attachments/assets/c1d81a70-0c10-460a-bd57-e6c78472a7e9)

Create dramatic portrait lighting with a strong contrast between light and shadow:
- Single light source at position (0.05, 0.5)
- Large outer radius (0.8)
- High contrast (25)
- Reduced saturation (-15)
- "In Front of Subject" light direction

#### Sunset Silhouette
![Spidey outlined by the sun](https://github.com/user-attachments/assets/8ed5851a-3607-46c1-a1ba-eb3d297733d4)

Create a beautiful sunset silhouette effect:
- Light positioned low and centered (0.5, 0.9)
- Warm colors (255, 180, 100) (enable "use colored lights")
- "Behind Subject" light direction
- High rim amplification (3.0)
- Moderate mask blur (60)

#### Atmospheric Fog Light
![Creature with foggy lighting](https://github.com/user-attachments/assets/db0d51da-6e9f-4359-9af0-2213954e010e)

Simulate light breaking through fog or mist:
- Light positioned high (0.5, 0.1)
- Cool blue-white color (200, 220, 255) (enable "use colored lights")
- "In Front of Subject" light direction
- High mask blur (100)
- Gradient mode enabled
- Medium intensity (1.5)

#### Moonlight Through Window
![Note hair illumination from moonlight in "After" image on right](https://github.com/user-attachments/assets/6e6e941c-9ea8-437e-bbcc-9fef3747dca4)

Simulate soft moonlight streaming through a window:
- Light positioned at upper corner (0.8, 0.2)
- Cool blue color (120, 150, 255) (enable "use colored lights")
- Gradient mode enabled
- Low brightness (-10)
- High blue cast (Temperature -30)


## 💡 Pro Tips

- **Layer Multiple Lights** - Use several ReLight nodes in sequence for complex lighting setups
- **Debug View** - Enable `show_debug_info` to visualize light positions and better understand the effect
- **Mask Quality Matters** - The better your foreground mask, the more realistic your lighting effects
- **Combine with ControlNet** - Use ReLight results as input for ControlNet for guided image generation
- **Perfect Rim Lighting** - For the best rim effects:
  1. Position light behind subject
  2. Use "Behind Subject" light direction
  3. Increase rim_amplification for stronger edges
  4. Keep mask_blur values low (20-40) for crisp edges

## 📝 Parameter Guide

### Core Parameters

| Parameter | Description |
|-----------|-------------|
| **image** | Input image to apply lighting effects. RGB or RGBA; alpha passes through untouched |
| **mask** | Foreground mask (white=subject, black=background). Resized to the image automatically |
| **preset** | Select from pre-configured lighting setups. Overrides the widgets it defines |
| **num_light_sources** | How many lights to use (1-3). Lights 2 and 3 have their own position, radius and color, but in color-correction mode they reuse Light 1's correction settings |
| **use_colored_lights** | Toggle between additive color and correction modes |
| **light_direction** | How light interacts with subject ("No Occlusion", "In Front", "Behind"). This is the control to use; `apply_3d_lighting` is just a master off-switch |
| **remove_background** | Composite the lit result back over the untouched original using the mask, so only the subject is relit. Despite the name it removes nothing. Off by default |
| **effect_strength** | Master intensity control for all lighting effects, gamma included. `0.0` leaves the image untouched. Scales a preset rather than being overridden by it. Does not scale `rim_amplification` or `mask_blur`, which have their own controls |
| **mask_blur** | Controls softness of light edges and transitions |
| **rim_amplification** | Specifically enhances rim light intensity |

### A note on gamma

`inner_gamma` and `outer_gamma` follow the convention used by Photoshop's Levels midtone slider, ImageMagick's `-gamma` and ffmpeg's `eq` filter: **values above 1.0 brighten midtones, values below 1.0 darken them.**

### Light Positioning

| Parameter | Description |
|-----------|-------------|
| **light_position_x/y** | Normalized (0-1) coordinates of light center |
| **inner_circle_radius** | Core area of strongest light effect |
| **outer_circle_radius** | Maximum extent of light falloff |

*For full parameter list, please refer to the detailed section below.*

## 🔍 Troubleshooting

| Problem | Solution |
|---------|----------|
| No visible effect | Increase effect_strength or light_intensity |
| Light too strong | Decrease effect_strength or specific intensity/brightness values |
| Occlusion not working | Set light_direction to "Behind Subject" or "In Front of Subject" and connect a mask |
| Only the subject changes, background untouched | `remove_background` is on — turn it off to light the whole frame |
| Editing a slider does nothing | A preset is active and overrides it. Set preset to "None" (`effect_strength` still works — it scales the preset) |
| Preset ignores its own light position/radius | `preserve_positioning` is on — turn it off to let the preset place its light |
| Black debug image | Check ComfyUI console for errors |
| Node fails to load | Ensure scipy is installed and ComfyUI is 0.3.48 or newer |
| Poor mask quality | Use RemBG from ComfyUI Essentials for better masks |
| Preset not working as expected | Try toggling use_colored_lights or apply_3d_lighting |
| Subject looks unlit / effect appears reversed | Your mask may be inverted — ReLight expects white=subject, black=background (invert it upstream with an InvertMask node) |
| Debug image not showing correctly | Enable show_debug_info and check console logs |

## 📚 Detailed Parameters Reference

### Core Inputs
- **image**: Input image to apply lighting effects
- **mask** (optional): Foreground mask (White=Subject, Black=Background). Required for occlusion modes and background compositing

### Global Behavior
- **preset**: Pre-configured starting points. Overrides the widgets it defines
- **num_light_sources**: Use 1, 2, or 3 lights
- **preserve_positioning**: Keep your own light positions and radii when a preset is selected, instead of letting the preset set them. Off by default so presets apply as designed
- **show_debug_info**: Output visualization showing base masks and light positions (first image of the batch). While it is off, the `debug_image` output is a placeholder frame telling you to turn it on — that output is never a rendered view unless this is enabled

### Lighting Mode & Occlusion
- **use_colored_lights**: Use additive colored light instead of color correction. The two are exclusive: with this on, every `inner_*` / `outer_*` correction value is ignored and only the light's RGB and `light_intensity` apply. "Warm Sunset Glow", "Cool Blue Moonlight" and "Rim Light (Behind)" switch it on, so those presets deliver a colored glow rather than a color grade — turn it off to get their grading half instead
- **use_gradient_mode**: Use directional gradient masks instead of radial
- **apply_3d_lighting**: Master switch for occlusion. Leave it on and drive the behaviour with `light_direction`
- **light_direction**: How light interacts with subject. "Behind"/"In Front" require a mask
- **remove_background**: Composite the lit result back over the untouched original using the mask. Ignored for "Behind Subject" and "In Front of Subject", which already light foreground and background separately

### Global Modifiers
- **effect_strength**: Overall intensity multiplier for lighting, gamma included. `0.0` is a true no-op, with or without a preset. Presets set a baseline that this widget scales, so `1.0` gives the preset as designed. It does not scale `rim_amplification` or `mask_blur` — each has its own control
- **mask_blur**: Blur radius for light mask edges
- **rim_amplification**: Boost specifically for rim light component

### Light Specific Settings (per light)
- **Position**: light_position_x/_y coordinates
- **Shape**: inner_circle_radius/outer_circle_radius
- **Color** (when using colored lights): light_color_r/_g/_b, light_intensity — available for all three lights
- **Corrections** (when using color correction): Brightness, Contrast, Saturation, Temperature, Tint, Gamma. These belong to Light 1; lights 2 and 3 reuse them at their own positions

In color-correction mode with `light_direction` on "No Occlusion", the `inner_*` settings apply inside `inner_circle_radius` and the `outer_*` settings apply in the ring out to `outer_circle_radius`. Outside that ring the image is untouched. The two occlusion modes ("Behind Subject" and "In Front of Subject") build a single subject-aware light mask instead, and apply the `inner_*` settings through it — the `outer_*` settings are not used there.

## 📜 License

MIT License - Feel free to use in personal and commercial projects

---

### 💪 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

The node can be tested without a ComfyUI install — `tests/stubs` provides a stand-in for `comfy_api`:

```bash
pip install -r requirements.txt
pip install torch pytest ruff
pytest -q
ruff check .
```

### 🔄 Updates

- **v3.1.2** - The debug output explains itself instead of going black
  - **Fixed: `debug_image` was a solid black frame whenever `show_debug_info` was off**, which is indistinguishable from a crashed node if you have that output wired to a preview. It now renders a legible placeholder naming the toggle that fills it. The same placeholder explains the other two empty cases: no light masks were generated, or the debug view failed to draw (with a console pointer)
  - Documented that `use_colored_lights` and the `inner_*`/`outer_*` correction values are mutually exclusive, and which presets ship with colored light on

- **v3.1.1** - Fixes a crash that broke every run on v3.0.0 and v3.1.0
  - **Fixed: `AttributeError: Cannot modify class attribute '_coord_cache' on locked class 'ReLightClone'`.** ComfyUI runs a v3 node on a *locked clone* of its class, which forbids writing class attributes. The coordinate cache added in v3.0.0 wrote to the class on the first mask it built, so the node crashed on every execution regardless of settings. The cache now lives at module level; behaviour and output are unchanged
  - The test suite now executes the node through a locked clone built exactly the way ComfyUI builds it, so class-attribute writes fail in CI instead of in a user's workflow

- **v3.1.0** - Audit follow-ups (one preset shifts by under one 8-bit step; see below)
  - **Fixed: `effect_strength` was dead under three presets.** "Spotlight", "Rim Light (Behind)" and "Negative Light (Darken)" set it themselves, so the master intensity — including the documented `0.0` no-op — did nothing for exactly the strongest presets. A preset now sets a *baseline* that the widget scales: `1.0` is the preset as designed, `0.0` is a true no-op, `2.0` is double
  - **Fixed: high `effect_strength` crushed dimmed zones to solid black.** Gamma was faded toward identity by linear interpolation, which ran a dimming gamma (say `0.77`) through zero and negative above strength ~4; the safety clamp then turned that into an exponent of 100. Gamma now scales in exponent space and is bounded by the widget's own `0.1`–`5.0` range, so the zone dims smoothly all the way to strength `5.0`
  - **Fixed: single-channel images silently became 3-channel** in colored-light mode, and produced a black debug view. Colored light now adds the light's luminance to a 1-channel image, and the debug visualization renders grayscale input properly. The debug output is now always RGB, whatever the input
  - Coordinate grids are cached as broadcast rows/columns instead of two full-size arrays — bit-identical masks, but a 4096×4096 workflow no longer pins ~268 MB for the life of the process, and each mask allocates one full-size temporary instead of three
  - Rim-mask gradient magnitude is computed once per frame instead of twice; gamma widget bounds and the internal clamp now share one constant. Output unchanged either way
  - Clarified that the two-zone `inner_*`/`outer_*` split applies to "No Occlusion" only; the occlusion modes use `inner_*` through a single subject-aware mask, and that `effect_strength` does not scale `rim_amplification` or `mask_blur`
  - Packaging: declares `requires-python >= 3.10`; dependency floors synced between `pyproject.toml` and `requirements.txt`; the publish workflow now requests only `contents: read`
  - Tests cover every fix above, plus schema self-consistency: widget defaults within their declared ranges, combo defaults among the offered options, preset values reachable by hand
  - **Output drift:** at default settings, existing workflows are unaffected. Across every preset × occlusion-mode combination only "Spotlight" changes at all, by at most 0.7/255 on a single pixel — below one 8-bit step. It is the only preset that pairs a non-`1.0` `effect_strength` with a dimming gamma

- **v3.0.0** - Correctness overhaul (some outputs change; see below)
  - **Fixed: crash when the mask resolution did not match the image.** Masks are now resized (and clamped to 0-1) automatically, so a mask made before an upscale no longer takes the node down
  - **Fixed: crash on RGBA input.** 4-channel images are handled, with the alpha channel passed through untouched
  - **Fixed: batched runs applied the first image's mask to every frame.** Occlusion and compositing are now computed per frame, which matters for any video or multi-image workflow
  - **Fixed: the `mask` output returned black** when both `apply_3d_lighting` and `remove_background` were off, and echoed the input's shape instead of a normalised `(batch, height, width)` mask
  - **Colour correction no longer round-trips through 8-bit.** Every correction previously quantised the image, so even all-neutral settings shifted pixels by 1/255 and the error compounded across light sources and chained ReLight nodes. Corrections now run in float32 on-device, and identity settings are a true no-op
  - Mask blurring likewise moved off the 8-bit path to a float convolution, removing banding in smooth falloffs
  - **`effect_strength` now scales gamma too**, so `0.0` genuinely leaves the image untouched (gamma was previously exempt)
  - **Changed default: `remove_background` is now off.** With it on, connecting a mask silently confined all lighting to the subject and left the background untouched — surprising, and not what the quick start described. Turn it on to get the old behaviour
  - **Changed default: `preserve_positioning` is now off**, so presets apply their own light positions and radii. Presets like "Spotlight" are defined by their tight radii and previously never applied them. Turn it on to keep your own positioning while a preset supplies everything else
  - **Preset gamma values corrected.** Gamma follows the Photoshop/ImageMagick/ffmpeg convention where values above 1.0 brighten — but presets paired dimmed outer zones with gamma above 1.0, partially undoing their own shading. The presets now darken and brighten as designed
  - Node inputs are grouped: secondary controls (`outer_*`, lights 2 and 3, debug) are marked advanced where ComfyUI supports it, cutting the visible surface from 48 widgets. Input order is unchanged, so saved workflows still load
  - Declares `requires-comfyui = ">=0.3.48"`, and a missing `comfy_api` now raises a readable message instead of a bare traceback
  - Example workflow moved to `example_workflows/relight_basic.json` (the conventional location)
  - Added a test suite and CI covering all of the above

- **v2.1.0** - Correctness & polish
  - **Fixed: outer-area color correction now actually applies.** The `outer_*` (brightness/contrast/saturation/temperature/tint/gamma) parameters were previously never used — standard-mode lighting now applies inner settings inside the inner radius and outer settings in the surrounding ring, as documented (this is what presets like "Soft Window Light" and "Spotlight" were designed around)
  - Colored lights in standard mode now have true radial falloff (full strength inside the inner radius, fading to zero at the outer radius) — `inner_circle_radius` previously had no effect
  - `mask` input is now optional (matching actual behavior); without a mask the node applies plain lighting and disables occlusion/compositing
  - Removed silent mask auto-inversion heuristic that could corrupt masks of large subjects; a console warning is logged instead if a mask looks inverted
  - Fixed v3 schema output ids (schema now passes ComfyUI validation)
  - Console spam removed — verbose diagnostics moved to debug-level logging
  - Packaging: removed `torch` from requirements (ComfyUI provides it)

- **v2.0.0** - ComfyUI v3 Migration
  - Fully migrated to ComfyUI v3 schema
  - Updated to use new io.ComfyNode base class
  - Converted INPUT_TYPES to define_schema with proper type objects
  - All methods converted to classmethods for v3 compatibility
  - Updated outputs to use proper display_name labels
  - Modernized extension registration with ComfyExtension and comfy_entrypoint

- **v1.0** - Initial Release
  - Added support for multiple light sources
  - Implemented rim lighting and 3D lighting simulation
  - Included 8 professional lighting presets
  - Added debug visualization
  - Improved mask handling and compatibility with ComfyUI Essentials
