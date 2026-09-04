# AGENTS.md — comfyui-relight

A single, self-contained ComfyUI node that adds up to 3 positionable light sources to any image — colored additive light, precise color correction, or both — with presets, directional gradients, rim lighting, and mask-aware subject occlusion including a cast shadow. Fast and deterministic: pure image processing (numpy + Pillow + scipy), no diffusion pass, no models to download. Built on the ComfyUI v3 node schema, with a small frontend in `web/`.

## Current state

_Last verified: 2026-09-04_

- **Status:** v4.0.0 is on branch `v4-overhaul`, not yet merged or released; `main` is still v3.1.2. Published to the Comfy Registry via `.github/workflows/publish_action.yml`, which fires on a `pyproject.toml` version change on **`main`** — so the bump is safe on a branch, and a functional change on main needs a bump or it never ships.
- **Works:** up to three independent light sources; three lighting modes (colored additive, two-zone grading, and both in that order); radial-falloff and directional-gradient mask shapes; subject interaction in front of / behind the subject, the latter with a rim highlight, a falloff background glow and a traced cast shadow; the built-in preset set; a debug view that follows the `debug_image` wiring; conditional widget visibility and a working node recreate. `.github/workflows/test.yml` runs pytest across Python 3.10–3.12, a `node --test` frontend job, and a separate ruff job.
- **In progress:** v4.0.0 on `v4-overhaul`, now **exercised in a live ComfyUI** (0.34.0 / frontend 1.51.9) via Playwright: node registration, conditional visibility and its round-trip, preset greying, the legacy-workflow migration against the real v3.1.2 save, single-run debug wiring, and "Fix node (recreate)" all behave, with no ReLight console warnings. Browser QA found two defects, both fixed here — see the v4.0.0 changelog. The three retuned presets have been rendered and signed off; "Rim Light (Behind)" carried a `(200, 255, 200)` light colour from v1.0 that put a +20/255 green cast on the rim, and is now neutral white at `light_intensity` 1.0.
- **Known gaps / next steps:** output quality depends heavily on the input mask, and the only mask-quality guard is a console-only warning when a mask is >90% white; presets are plain dicts at the top of `relight.py` with no way for a user to add their own without editing the file; there is no example beyond the bundled workflow JSON; the rim highlight still does per-frame CPU SciPy work (`fg_mask[b].cpu().numpy()` + Sobel per frame) because vectorising it risks numeric drift, though the cast shadow added in v4.0.0 is vectorised in torch.
- **Deep docs:** none — `README.md` is the user-facing reference and `relight.py` is the whole implementation.

## Architecture in 60 seconds

- **Single node.** `relight.py` contains the entire feature — one node class, all logic self-contained.
- **Up to 3 independent light sources.** Each with position, mode (colored light or color correction), mask shape (circular falloff or gradient), and fine-tuning controls.
- **Three lighting modes per source:** colored additive RGB light, precise color correction (brightness, contrast, saturation, temperature, tint, gamma), or both — colour first, then the grade applied to the lit result.
- **Mask shapes:** radial falloff (inner/outer radius) or directional gradient (sunset rays, window light).
- **3D subject interaction** (needs a mask): light in front of the subject, or behind it — rim highlight, a background glow with real falloff, and a cast shadow traced by `cast_shadow_mask` (a `grid_sample` ray march, run at `_SHADOW_TRACE_MAX` and upsampled; no SciPy).
- **Built-in presets.** Soft Window Light, Dramatic Side Light, Warm Sunset Glow, Cool Blue Moonlight, Studio Key Light, Rim Light (Behind), Spotlight, Negative Light (Darken).
- **Visual debugging with no toggle.** Wiring the `debug_image` output is the whole gesture; see the convention below for why that needs a hidden input.
- **A frontend that hides what is inert.** `web/` carries the legacy-workflow migration, the debug-output tracking, conditional widget visibility and a working "Fix node (recreate)"; `tests/frontend/` drives all four outside a browser.

## Layout

| File | Purpose |
|------|---------|
| `__init__.py` | ComfyUI custom-node entry point (registers the ReLight node) |
| `relight.py` | The entire node: lighting engine, presets, UI widgets, image processing |
| `web/relight_migrate.js` | Remaps a pre-v4 save's positional widget values by name on `onConfigure` |
| `web/relight_debug.js` | Keeps `debug_output_connected` in step with the `debug_image` wiring, and hides it |
| `web/relight_ui.js` | Conditional visibility: hides irrelevant blocks, greys what a preset overrides, fits the node |
| `web/relight_recreate.js` | A "Fix node (recreate)" that replaces instead of duplicating |
| `web/relight_presets.js` | GENERATED - which widgets each preset overrides, and what it sets them to |
| `scripts/dump_frontend_fixture.py` | Dumps the live schema to `tests/frontend/fixtures/schema.json` for the JS tests (`--check` gates staleness) |
| `requirements-dev.txt` | test-only deps (numpy, Pillow, scipy, pytest, pinned ruff); the pack declares no runtime deps |
| `tests/` | pytest suite (CI, Python 3.10-3.12) plus `tests/frontend/` (`node --test`) and `tests/fixtures/` (a verbatim v3.1.2 save) |
| `example_workflows/` | Example ComfyUI workflows demonstrating the node |
| `docs/images/` | README screenshots and renders; photographic panels are JPEG, UI captures PNG |

## Build / test / run

```bash
# Install via ComfyUI Manager (recommended)
# Search for "ReLight" in the Manager

# Manual install
cd ComfyUI/custom_nodes
git clone https://github.com/EnragedAntelope/comfyui-relight
# Restart ComfyUI - nothing to install

# No runtime dependencies: numpy, Pillow, scipy and torch all come with ComfyUI core

# Run tests (CI runs this on Python 3.10-3.12)
pytest -q

# Lint (CI runs this as a separate job so a style nit can't hide a test result)
ruff check .

# Frontend tests - drives the real web/*.js outside a browser (own CI job)
node --import ./tests/frontend/hooks.mjs --test "tests/frontend/*.test.mjs"

# Regenerate the fixture the frontend tests build fake nodes from
python scripts/dump_frontend_fixture.py

# Manual QA in ComfyUI after any visible change:
# node appears, presets work, lights position correctly, mask interaction works
```

## Conventions & gotchas

- Single-file node — `relight.py` is the entire feature. Keep it self-contained.
- No models to download — pure image processing, deterministic output.
- **Declare no runtime dependencies.** numpy, Pillow, scipy and torch are all in ComfyUI core's own `requirements.txt` at floors at or above anything here, so `pyproject.toml` keeps `dependencies = []`. Test-only deps live in `requirements-dev.txt`, which both CI jobs install.
- Works best with high-quality foreground masks (e.g. from ComfyUI Essentials).
- The node uses the ComfyUI v3 schema (`comfy_api`, `v0_0_2` with a `latest` fallback).
- Widget inputs are stored *positionally* in saved workflows, but the order is **no longer frozen**. What keeps pre-v4 files loading is `web/relight_migrate.js`, which remaps them by name. So: any schema change - add, remove, rename, reorder - must be paired with a check that the migration still maps correctly, and the legacy order pinned in `tests/test_relight.py` (`LEGACY_WIDGET_ORDER`) must keep matching the JS constant. Get the migration wrong and every saved workflow loads plausible garbage with no error, which is worse than a crash.
- A preset overrides whatever widgets it names — except `effect_strength`, which it scales (see `ReLight.STRENGTH_KEY`), and the `GEOMETRY_KEYS` when `preserve_positioning` is on.
- Presets are defined as dicts at the top of `relight.py` — easy to extend. Re-run `scripts/dump_frontend_fixture.py` after any change to them or to the schema; `web/relight_presets.js` and `tests/frontend/fixtures/schema.json` are generated from those and a stale one fails the suite.
- **Hiding a widget needs both halves** — swap `widget.type` *and* set `widget.hidden = true` — and showing it again means `delete widget.computeSize`, never reassigning a saved copy (most widgets have no own `computeSize`, so the saved value is `undefined` and the zero-size stub stays forever). Grey out (`disabled = true`) anything a preset overrides rather than hiding it, so the node does not reshuffle under the pointer. Never resize from `onDrawForeground`; defer it a frame.
- **A greyed widget cannot show its own value.** The frontend's `_displayValue` getter returns `""` for anything with `computedDisabled` set, so `disabled = true` alone paints an empty bar with a dim label and no number. `relight_ui.js` writes the *preset's* value into `widget.label` instead, which still paints — and it must be the preset's value, not the widget's, because the widget still holds whatever the user last set, which is exactly the number the preset is ignoring. `PRESET_VALUES` in the generated `web/relight_presets.js` is what supplies them. Restoring is self-healing: any label shaped `name  →  value` is treated as ReLight's to clear, so a label that arrives without matching bookkeeping cannot strand a preset value on a live control.
- **Anything drawn near a frame edge needs a fallback side.** The debug view's `L1`/`L2` marker labels are drawn to the right of the marker, and a light at `light_position_x` 0.9 — where "Warm Sunset Glow" puts one — pushed them off the frame, where PIL clips silently. They flip to the left when there is no room. Test this by measuring which *side* of the marker the label's backing box lands on, on a white frame: on a black one the box is invisible against the input and the measurement silently passes with the bug still in.
- `lighting_mode` is two independent switches underneath (`apply_colored`, `apply_correction`); `Both` runs the colour pass and then grades the result. Pre-v4 this was one boolean and the two were mutually exclusive, which left 12 values inert in three presets.
- **The debug view has no toggle.** Connecting the `debug_image` output is the whole gesture. `debug_output_connected` is a hidden boolean input that `web/relight_debug.js` writes; it exists only because ComfyUI's cache key is built from a node's *inputs*, so without it, wiring an *output* would replay the cached placeholder. Never surface it as a control.
- **Anything drawn onto a full-resolution frame must scale with it.** `_debug_font_size()` is the one rule (3.5% of frame height, floored at 13px, capped at 64). v3.1.2 drew 13px type on a 1344x768 canvas - legible on the 96x64 test fixture, a black rectangle in a preview - and every debug test passed because they all ran on that fixture. Test overlays at a realistic resolution, measuring ink coverage inside the border, not `max() > 0`.
- Never return an all-black image as an "empty" result. A wired preview makes it look like the node crashed; say why the frame is empty (`_blank_debug_image(image, reason)`).
- **Never store per-run state on the class.** ComfyUI does not call `execute` on `ReLight`; it calls it on a *locked clone* (`ReLightClone`) whose metaclass raises `AttributeError` on any class-attribute write, and whose instances reject `__setattr__` too. `cls._coord_cache = ...` shipped in v3.0.0 and crashed every single run until v3.1.1. Caches belong at module level (`_COORD_CACHE`).
- The `node` fixture in `tests/conftest.py` hands tests that same locked clone, mirroring `comfy_api.internal.lock_class`, so this class of bug fails in CI. Do not "simplify" it back to the bare class — the bare class is not what ComfyUI runs.

## Security

This file is **public-safe by default**. Never add local paths, credentials, API keys, personal data, infrastructure details, or subscription info.

Before pushing: run your denylist checker over this file and `CLAUDE.md` — it is not vendored here, it lives with your own agent tooling — then re-read to confirm nothing above crept in.

## Maintenance

**Update rule:** When you change the architecture, build/test commands, or conventions, update this AGENTS.md in the same commit. Keep under 200 lines.

**CLAUDE.md:** One-line shim: `@AGENTS.md`.

**New-repo rule:** Create AGENTS.md in the first session a new repo is worked on.

**No-overlap rule:** Explanatory prose lives in one file. AGENTS.md = agent-facing summary; README.md = human/usage. Identical install commands may be restated verbatim. Explanatory prose must not be duplicated — link instead.
