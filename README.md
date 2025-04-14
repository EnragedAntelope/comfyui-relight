# ✨ ReLight Node for ComfyUI

![Platform](https://img.shields.io/badge/Platform-ComfyUI-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **Transform your images with cinematic lighting effects in a single click!**

ReLight is a powerful custom node for ComfyUI that adds professional-grade lighting capabilities to your images. Create dramatic shadows, natural window lighting, warm sunset glows, or striking rim effects with precise control over every aspect of your lighting setup.
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
  - "Rim Light" - Striking edge highlights
  - "Spotlight" - Focused dramatic lighting
  - "Negative Light" - Creative darkening effects

- **Visual Debugging** - See exactly where your lights are positioned and how they interact
- **Fine-Tuning Controls** - Perfect your lighting with precision adjustments for blur, strength, and rim amplification

## 🖼️ Before & After Examples

Here's an example of how ReLight transforms images with its lighting effects:

![Before/After Comparison](https://via.placeholder.com/900x300?text=Before+and+After+Comparison)

The image shows:
- **Left**: Original image of a subject
- **Right**: Same image with the "Warm Sunset Glow" lighting preset applied using "Behind Subject" light direction for a dramatic rim-lit effect

> *Replace with your own before/after examples when integrating this into your repository!*

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
git clone https://github.com/yourusername/comfyui_relight

# Install dependencies
pip install -r comfyui_relight/requirements.txt

# Restart ComfyUI
```

### Dependencies

ReLight works best with high-quality foreground masks. We recommend installing:

- **[ComfyUI Essentials](https://github.com/cubiq/ComfyUI_essentials)** - Provides enhanced mask generation and background removal tools
- **[Scipy](https://scipy.org/)** - Required for rim lighting effects (should be installed automatically)

## 🚀 Quick Start Guide

1. **Add the ReLight 💡 node** to your workflow (found under category "image/lighting")
2. **Connect your source image**
3. **Connect a foreground mask** (white = subject, black = background)
4. **Select a preset** like "Rim Light (Behind)" or design your own lighting
5. **Adjust settings** to taste
6. **Preview your results** in real-time

### Sample Workflow

![Sample Workflow](https://via.placeholder.com/800x400?text=Sample+Workflow+Screenshot)

The repository includes a sample workflow file (`Relight example workflow.json`) that demonstrates:

1. Loading an image
2. Removing the background using ComfyUI Essentials' RemBG nodes
3. Applying the ReLight node with "Warm Sunset Glow" preset in "Behind Subject" mode
4. Viewing the results through both standard preview and debug visualization

Simply load this workflow in ComfyUI to see ReLight in action!

## 📊 Lighting Workflow Examples

### Dramatic Portrait Lighting

![Portrait Workflow](https://via.placeholder.com/800x300?text=Portrait+Lighting+Workflow)

1. Connect portrait image and mask
2. Select "Dramatic Side Light" preset
3. Increase effect_strength to 1.2
4. Adjust light position slightly to match subject features

### Epic Fantasy Rim Light

![Fantasy Workflow](https://via.placeholder.com/800x300?text=Fantasy+Rim+Light+Workflow)

1. Connect character image and mask
2. Select "Rim Light (Behind)" preset
3. Set light color to blue-white (R:200, G:220, B:255)
4. Increase rim_amplification to 3.0 for maximum glow

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
| **image** | Input image to apply lighting effects |
| **mask** | Foreground mask (white=subject, black=background) |
| **preset** | Select from pre-configured lighting setups |
| **num_light_sources** | How many lights to use (1-3) |
| **use_colored_lights** | Toggle between additive color and correction modes |
| **apply_3d_lighting** | Enable simulated subject occlusion effects |
| **light_direction** | How light interacts with subject ("No Occlusion", "In Front", "Behind") |
| **effect_strength** | Master intensity control for all lighting effects |
| **mask_blur** | Controls softness of light edges and transitions |
| **rim_amplification** | Specifically enhances rim light intensity |

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
| Occlusion not working | Ensure apply_3d_lighting is True and mask is connected |
| Black debug image | Check ComfyUI console for errors |
| Node fails to load | Ensure scipy is installed |
| Poor mask quality | Use RemBG from ComfyUI Essentials for better masks |
| Preset not working as expected | Try toggling use_colored_lights or apply_3d_lighting |
| Debug image not showing correctly | Enable show_debug_info and check console logs |

## 📚 Detailed Parameters Reference

### Core Inputs
- **image**: Input image to apply lighting effects
- **mask**: Foreground mask (White=Subject, Black=Background)

### Global Behavior
- **preset**: Pre-configured starting points
- **num_light_sources**: Use 1, 2, or 3 lights
- **preserve_positioning**: Keep manual light positions when changing presets
- **show_debug_info**: Output visualization showing base masks and light positions

### Lighting Mode & Occlusion
- **use_colored_lights**: Use additive colored light instead of color correction
- **use_gradient_mode**: Use directional gradient masks instead of radial
- **apply_3d_lighting**: Simulate light occlusion by subject (requires mask)
- **light_direction**: How light interacts with subject
- **remove_background**: Composite result using mask

### Global Modifiers
- **effect_strength**: Overall intensity multiplier for lighting
- **mask_blur**: Blur radius for light mask edges
- **rim_amplification**: Boost specifically for rim light component

### Light Specific Settings (per light)
- **Position**: light_position_x/_y coordinates
- **Shape**: inner_circle_radius/outer_circle_radius
- **Color** (when using colored lights): light_color_r/_g/_b, light_intensity
- **Corrections** (when using color correction): Brightness, Contrast, Saturation, Temperature, Tint, Gamma

## 📜 License

MIT License - Feel free to use in personal and commercial projects

---

### 💪 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### 🔄 Updates

- **v1.0** - Initial Release
  - Added support for multiple light sources
  - Implemented rim lighting and 3D lighting simulation
  - Included 8 professional lighting presets
  - Added debug visualization
  - Improved mask handling and compatibility with ComfyUI Essentials
