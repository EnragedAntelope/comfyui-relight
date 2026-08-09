# AGENTS.md — comfyui-relight

A single, self-contained ComfyUI node that adds up to 3 positionable light sources to any image — colored additive light or precise color correction, with presets, directional gradients, rim lighting, and mask-aware 3D occlusion. Fast and deterministic: pure image processing (numpy + Pillow + scipy), no diffusion pass, no models to download. Built on ComfyUI v3 node schema.

## Current state

_Last verified: 2026-08-08_

- **Status:** released v3.0.0 (`pyproject.toml`). Published to the Comfy Registry via `.github/workflows/publish_action.yml`, which fires on a `pyproject.toml` version change on `main` — a functional change needs a version bump or it never ships.
- **Works:** up to three independent light sources; both per-source modes (colored additive light, and color correction with brightness/contrast/saturation/temperature/tint/gamma); circular-falloff and gradient mask shapes; mask-aware front / rim / standard subject interaction; the built-in preset set; the visual debug view. `.github/workflows/test.yml` runs pytest across Python 3.10–3.12 plus a separate, deliberately non-blocking ruff job.
- **In progress:** nothing — v3.0.0 closed out the known crash, batch-mask and 8-bit precision-loss bugs and added the test suite that had been missing.
- **Known gaps / next steps:** output quality depends heavily on the input mask, and there is no mask-quality warning; presets are plain dicts at the top of `relight.py` with no way for a user to add their own without editing the file; there is no example beyond the bundled workflow JSON.
- **Deep docs:** none — `README.md` is the user-facing reference and `relight.py` is the whole implementation.

## Architecture in 60 seconds

- **Single node.** `relight.py` contains the entire feature — one node class, all logic self-contained.
- **Up to 3 independent light sources.** Each with position, mode (colored light or color correction), mask shape (circular falloff or gradient), and fine-tuning controls.
- **Two lighting modes per source:** colored additive RGB light, or precise color correction (brightness, contrast, saturation, temperature, tint, gamma).
- **Mask shapes:** circular falloff (natural radial lighting with inner/outer radius) or gradient (directional lighting for sunset rays, window light effects).
- **3D subject interaction** (when used with mask input): front lighting, rim lighting (dramatic edge highlighting with background glow), or standard lighting.
- **Built-in presets.** Soft Window Light, Dramatic Side Light, Warm Sunset Glow, Cool Blue Moonlight, Studio Key Light, Rim Light, Spotlight, Negative Light.
- **Visual debugging.** Shows exactly where lights are positioned and how they interact.

## Layout

| File | Purpose |
|------|---------|
| `__init__.py` | ComfyUI custom-node entry point (registers the ReLight node) |
| `relight.py` | The entire node: lighting engine, presets, UI widgets, image processing |
| `requirements.txt` | numpy, Pillow, scipy |
| `tests/` | pytest suite (run in CI on Python 3.10-3.12) |
| `example_workflows/` | Example ComfyUI workflows demonstrating the node |

## Build / test / run

```bash
# Install via ComfyUI Manager (recommended)
# Search for "ReLight" in the Manager

# Manual install
cd ComfyUI/custom_nodes
git clone https://github.com/EnragedAntelope/comfyui-relight
pip install -r comfyui-relight/requirements.txt
# Restart ComfyUI

# Dependencies: numpy, Pillow, scipy (torch provided by ComfyUI)

# Run tests (CI runs this on Python 3.10-3.12)
pytest -q

# Lint (CI runs this as a separate job so a style nit can't hide a test result)
ruff check .

# Manual QA in ComfyUI after any visible change:
# node appears, presets work, lights position correctly, mask interaction works
```

## Conventions & gotchas

- Single-file node — `relight.py` is the entire feature. Keep it self-contained.
- No models to download — pure image processing, deterministic output.
- Dependencies are lightweight: numpy, Pillow, scipy. torch comes from ComfyUI itself.
- Works best with high-quality foreground masks (e.g. from ComfyUI Essentials).
- The node uses the ComfyUI v3 schema (`comfy_api.latest`).
- Presets are defined as dicts at the top of `relight.py` — easy to extend.

## Security

This file is **public-safe by default**. Never add local paths, credentials, API keys, personal data, infrastructure details, or subscription info.

Before pushing: `pwsh scripts/check-agents-md.ps1 AGENTS.md CLAUDE.md` — must exit 0.

## Maintenance

**Update rule:** When you change the architecture, build/test commands, or conventions, update this AGENTS.md in the same commit. Keep under 200 lines.

**CLAUDE.md:** One-line shim: `@AGENTS.md`.

**New-repo rule:** Create AGENTS.md in the first session a new repo is worked on.

**No-overlap rule:** Explanatory prose lives in one file. AGENTS.md = agent-facing summary; README.md = human/usage. Identical install commands may be restated verbatim. Explanatory prose must not be duplicated — link instead.
