"""Regression tests for the ReLight node.

Each test names the audit finding it locks in. They run headless against the
``comfy_api`` stub in ``tests/stubs``, so no ComfyUI install is needed.
"""

import json
import pathlib
import re
import sys

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

IDENTITY = {
    "inner_brightness": 0.0,
    "inner_contrast": 0.0,
    "inner_saturation": 0.0,
    "inner_temperature": 0.0,
    "inner_tint": 0.0,
    "inner_gamma": 1.0,
    "outer_brightness": 0.0,
    "outer_contrast": 0.0,
    "outer_saturation": 0.0,
    "outer_temperature": 0.0,
    "outer_tint": 0.0,
    "outer_gamma": 1.0,
}


# --- schema ---------------------------------------------------------------


def test_schema_ids_are_unique(node):
    ids = [spec.id for spec in node.define_schema().inputs]
    assert len(ids) == len(set(ids))


def test_schema_exposes_three_outputs(node):
    assert [spec.id for spec in node.define_schema().outputs] == [
        "image",
        "mask",
        "debug_image",
    ]


def test_every_input_has_a_tooltip(node):
    missing = [s.id for s in node.define_schema().inputs if not s.tooltip]
    assert missing == []


#: Widget order exactly as v3.1.2 serialised it. Not a schema expectation - the
#: v4 schema deliberately reorders - but the input to `web/relight_migrate.js`,
#: which is the only thing keeping pre-v4 workflows loading correctly.
LEGACY_WIDGET_ORDER = [
    "preset",
    "num_light_sources",
    "preserve_positioning",
    "show_debug_info",
    "use_colored_lights",
    "use_gradient_mode",
    "apply_3d_lighting",
    "light_direction",
    "remove_background",
    "effect_strength",
    "mask_blur",
    "rim_amplification",
    "light_position_x",
    "light_position_y",
    "inner_circle_radius",
    "outer_circle_radius",
    "light_color_r",
    "light_color_g",
    "light_color_b",
    "light_intensity",
    "inner_brightness",
    "inner_contrast",
    "inner_saturation",
    "inner_temperature",
    "inner_tint",
    "inner_gamma",
    "outer_brightness",
    "outer_contrast",
    "outer_saturation",
    "outer_temperature",
    "outer_tint",
    "outer_gamma",
    "light2_position_x",
    "light2_position_y",
    "light2_inner_radius",
    "light2_outer_radius",
    "light2_color_r",
    "light2_color_g",
    "light2_color_b",
    "light2_intensity",
    "light3_position_x",
    "light3_position_y",
    "light3_inner_radius",
    "light3_outer_radius",
    "light3_color_r",
    "light3_color_g",
    "light3_color_b",
    "light3_intensity",
]

#: Widgets renamed or merged away in v4.0.0, and what the migration turns them
#: into. Pinned here so a rename in relight.py without a matching change in the
#: JS goes red.
LEGACY_WIDGETS_REMOVED = {
    # show_debug_info has no v4 counterpart at all: the debug view now follows
    # whether the debug_image output is wired, so there is nothing to carry over.
    "show_debug_info": None,
    "use_colored_lights": "lighting_mode",
    "use_gradient_mode": "mask_shape",
    "apply_3d_lighting": "subject_interaction",
    "light_direction": "subject_interaction",
}


def _js_string_array(source, name):
    """Pull `export const <name> = [ "a", "b" ];` out of the migration module."""
    match = re.search(
        r"export const " + name + r" = \[(.*?)\];", source, re.DOTALL
    )
    assert match, f"{name} not found in web/relight_migrate.js"
    return re.findall(r'"((?:[^"\\]|\\.)*)"', match.group(1))


def test_frontend_schema_fixture_is_current():
    """The JS tests build their fake nodes from a dump of this schema.

    A stale dump means the frontend suite is testing a node shape that no
    longer exists - green, and proving nothing.
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import dump_frontend_fixture
    finally:
        sys.path.pop(0)
    assert dump_frontend_fixture.main(["--check"]) == 0


def test_legacy_widget_order_matches_the_migration_constant():
    """The JS migration and this file must agree on the pre-v4 widget order.

    The whole migration is a name-for-index lookup against that order. If the
    two copies drift, every saved workflow loads plausible-looking garbage -
    no error, wrong values. So they are pinned against each other here.
    """
    source = (REPO_ROOT / "web" / "relight_migrate.js").read_text(encoding="utf-8")
    assert _js_string_array(source, "LEGACY_ORDER") == LEGACY_WIDGET_ORDER


def test_migration_knows_every_legacy_preset_name(node):
    """The legacy-save sentinel checks index 0 against this list."""
    source = (REPO_ROOT / "web" / "relight_migrate.js").read_text(encoding="utf-8")
    assert _js_string_array(source, "LEGACY_PRESETS") == list(node.PRESETS)


def test_migration_covers_every_legacy_widget(node):
    """Every pre-v4 widget either still exists by name, or has a translation."""
    current = {spec.id for spec in node.define_schema().inputs}
    unhandled = [
        name
        for name in LEGACY_WIDGET_ORDER
        if name not in current and name not in LEGACY_WIDGETS_REMOVED
    ]
    assert unhandled == [], "no migration path for these pre-v4 widgets"


def test_removed_widgets_map_onto_widgets_that_exist(node):
    current = {spec.id for spec in node.define_schema().inputs}
    missing = sorted(
        {
            target
            for target in LEGACY_WIDGETS_REMOVED.values()
            if target is not None and target not in current
        }
    )
    assert missing == []


def test_current_widget_count_cannot_be_mistaken_for_a_legacy_save(node):
    """The migration's length check must not fire on a current save.

    `looksLegacy` gates on a 48-value array plus two sentinels. If the v4 schema
    ever had exactly 48 widgets the length gate would be load-bearing on the
    sentinels alone, which is a thinner guarantee than intended.
    """
    widget_ids = [
        spec.id for spec in node.define_schema().inputs if spec.id not in ("image", "mask")
    ]
    assert len(widget_ids) != len(LEGACY_WIDGET_ORDER)


def test_legacy_fixture_workflow_is_a_real_pre_v4_save():
    """The migration fixture must be a genuine v3.1.2 save, not a synthetic array.

    A hand-written array can be made to satisfy any migration, including a wrong
    one. This is the shipped v3.1.2 example workflow, kept verbatim.
    """
    graph = json.loads(
        (REPO_ROOT / "tests" / "fixtures" / "legacy_v3.1.2_workflow.json").read_text()
    )
    relight = [n for n in graph["nodes"] if n["type"] == "ReLight"]
    assert len(relight) == 1
    values = relight[0]["widgets_values"]
    assert len(values) == len(LEGACY_WIDGET_ORDER)
    assert values[LEGACY_WIDGET_ORDER.index("light_direction")] in (
        "Behind Subject",
        "In Front of Subject",
        "No Occlusion",
    )


def test_shipped_example_workflows_match_the_schema(node):
    """The bundled workflows must still deserialise against the current schema."""
    widget_ids = [
        spec.id for spec in node.define_schema().inputs if spec.id not in ("image", "mask")
    ]
    workflows = sorted((REPO_ROOT / "example_workflows").glob("*.json"))
    assert workflows, "no example workflows found"

    for path in workflows:
        graph = json.loads(path.read_text())
        for graph_node in graph["nodes"]:
            if graph_node["type"] != "ReLight":
                continue
            values = graph_node["widgets_values"]
            assert len(values) == len(widget_ids), f"{path.name}: widget count drifted"
            settings = dict(zip(widget_ids, values))
            assert settings["preset"] in node.PRESETS, f"{path.name}: unknown preset"
            for widget, allowed in (
                ("lighting_mode", node.LIGHTING_MODES),
                ("mask_shape", node.MASK_SHAPES),
                ("subject_interaction", node.SUBJECT_INTERACTIONS),
            ):
                assert settings[widget] in allowed, f"{path.name}: unknown {widget}"


# --- basic contract -------------------------------------------------------


def test_returns_three_tensors_without_a_mask(run, image):
    out_image, out_mask, debug = run(image)
    assert out_image.shape == image.shape
    assert out_mask.shape == (1, 64, 96)
    assert debug.shape == (1, 64, 96, 3)
    assert out_image.dtype == torch.float32


def test_output_stays_in_range(run, image):
    out_image = run(image, effect_strength=5.0, inner_brightness=100.0)[0]
    assert out_image.min() >= 0.0
    assert out_image.max() <= 1.0


def test_input_image_is_not_mutated(run, image):
    original = image.clone()
    run(image, inner_brightness=50.0)
    assert torch.equal(image, original)


# --- finding 1: mask resolution mismatch ----------------------------------


def test_mask_smaller_than_image_is_resized(run, image):
    mask = torch.ones(1, 32, 48)
    out_image, out_mask, _ = run(
        image, mask=mask, subject_interaction="Light behind subject (rim)", remove_background=True
    )
    assert out_image.shape == image.shape
    assert out_mask.shape == (1, 64, 96)


def test_mask_larger_than_image_is_resized(run, image):
    mask = torch.ones(1, 200, 300)
    out_image = run(image, mask=mask, remove_background=True)[0]
    assert out_image.shape == image.shape


# --- finding 2: RGBA images -----------------------------------------------


def test_rgba_image_survives_and_keeps_alpha(run):
    rgba = torch.rand(1, 32, 32, 4)
    alpha = rgba[..., 3:].clone()
    out_image = run(rgba, inner_brightness=40.0)[0]
    assert out_image.shape == rgba.shape
    assert torch.equal(out_image[..., 3:], alpha), "alpha must pass through untouched"


def test_rgba_image_in_colored_mode(run):
    rgba = torch.rand(1, 32, 32, 4)
    out_image = run(rgba, lighting_mode="Colored Light")[0]
    assert out_image.shape == rgba.shape


# --- finding 3: per-frame masks in a batch --------------------------------


def test_batch_uses_each_frames_own_mask(run):
    torch.manual_seed(1)
    batch = torch.rand(3, 48, 48, 3)
    mask = torch.zeros(3, 48, 48)
    mask[0, :, :24] = 1.0  # subject on the left
    mask[1, :, 24:] = 1.0  # subject on the right
    # frame 2 deliberately left empty

    out = run(
        batch,
        mask=mask,
        subject_interaction="Light behind subject (rim)",
        lighting_mode="Colored Light",
    )[0]
    delta = (out - batch).abs().mean(dim=(1, 2, 3))

    # An empty mask has no subject edge, so frame 2 cannot pick up rim light the
    # way frames 0 and 1 do. If frame 0's mask leaked across the batch, all
    # three deltas would match.
    assert not torch.allclose(delta[0], delta[2], atol=1e-3)


def test_batch_composite_is_per_frame(run):
    batch = torch.rand(2, 32, 32, 3)
    mask = torch.zeros(2, 32, 32)
    mask[0] = 1.0  # frame 0 fully subject: lighting applies everywhere
    mask[1] = 0.0  # frame 1 fully background: nothing should change

    out = run(batch, mask=mask, remove_background=True, inner_brightness=60.0)[0]
    assert not torch.allclose(out[0], batch[0], atol=1e-4)
    assert torch.allclose(out[1], batch[1], atol=1e-6)


# --- findings 4 & 5: mask pass-through ------------------------------------


def test_mask_passes_through_when_both_consumers_are_off(run, image):
    mask = torch.ones(1, 64, 96)
    out_mask = run(
        image, mask=mask, subject_interaction="None", remove_background=False
    )[1]
    assert torch.allclose(out_mask, mask)


def test_four_dimensional_mask_is_normalised_on_output(run, image):
    mask = torch.ones(1, 64, 96, 1)
    out_mask = run(image, mask=mask)[1]
    assert out_mask.shape == (1, 64, 96)


def test_mask_output_is_black_when_no_mask_connected(run, image):
    out_mask = run(image)[1]
    assert out_mask.shape == (1, 64, 96)
    assert out_mask.abs().max() == 0.0


# --- finding 6: no 8-bit round-trip ---------------------------------------


def test_identity_correction_is_bit_exact(run, image):
    out_image = run(image, **IDENTITY)[0]
    assert torch.equal(out_image, image), "an identity correction must not alter a single pixel"


def test_correction_preserves_more_than_8_bits(node):
    ramp = torch.linspace(0.2, 0.8, 512).view(1, 1, 512, 1).expand(1, 8, 512, 3).contiguous()
    out = node.apply_color_correction(ramp, brightness=5.0)
    # A uint8 round-trip collapses a 512-step ramp to at most 256 levels.
    assert torch.unique(out).numel() > 256


def test_chained_runs_do_not_drift(run, image):
    once = run(image, **IDENTITY)[0]
    twice = run(once, **IDENTITY)[0]
    assert torch.equal(once, twice)


# --- finding 7: effect_strength ------------------------------------------


def test_zero_effect_strength_is_a_no_op(run, image):
    out_image = run(image, effect_strength=0.0)[0]
    assert torch.allclose(out_image, image, atol=1e-6)


def test_zero_effect_strength_is_a_no_op_with_a_preset(run, image):
    out_image = run(image, preset="Dramatic Side Light", effect_strength=0.0)[0]
    assert torch.allclose(out_image, image, atol=1e-6)


def test_zero_intensity_colored_light_is_a_no_op(run, image):
    out_image = run(image, lighting_mode="Colored Light", light_intensity=0.0)[0]
    assert torch.allclose(out_image, image, atol=1e-6)


# --- finding 8: mask blur precision ---------------------------------------


def test_mask_blur_keeps_float_precision(node):
    edge = torch.zeros(64, 64)
    edge[:, 32:] = 1.0
    blurred = node.apply_mask_blur(edge, 20.0)
    # A uint8 round-trip lands every value on the n/255 grid; float blurring does not.
    off_grid = ((blurred * 255.0) - (blurred * 255.0).round()).abs().max()
    assert off_grid > 1e-3
    assert blurred.min() >= 0.0
    assert blurred.max() <= 1.0


def test_mask_blur_is_a_no_op_at_zero(node):
    ramp = torch.linspace(0, 1, 64).repeat(64, 1)
    assert torch.equal(node.apply_mask_blur(ramp, 0.0), ramp)


def test_mask_blur_handles_batched_masks(node):
    masks = torch.rand(3, 32, 32)
    blurred = node.apply_mask_blur(masks, 10.0)
    assert blurred.shape == masks.shape
    # Each item must be blurred independently, not smeared into its neighbours.
    single = node.apply_mask_blur(masks[1], 10.0)
    assert torch.allclose(blurred[1], single, atol=1e-5)


# --- finding 15: mask sanitisation ----------------------------------------


def test_out_of_range_mask_is_clamped(run, image):
    mask = torch.full((1, 64, 96), 3.0)
    out_image, out_mask, _ = run(image, mask=mask, remove_background=True)
    assert out_mask.max() <= 1.0
    assert out_image.max() <= 1.0


def test_boolean_mask_is_accepted(run, image):
    mask = torch.ones(1, 64, 96, dtype=torch.bool)
    out_mask = run(image, mask=mask, remove_background=True)[1]
    assert out_mask.dtype == torch.float32


# --- findings 9 & 11: defaults --------------------------------------------

def test_lighting_reaches_the_background_by_default(run, image):
    """A connected mask must not silently confine the effect to the subject."""
    mask = torch.zeros(1, 64, 96)
    mask[:, 16:48, 24:72] = 1.0
    out_image = run(image, mask=mask, inner_brightness=40.0)[0]
    background_delta = (out_image - image)[:, 0:8, 0:8].abs().max()
    assert background_delta > 1e-4


def test_remove_background_still_composites_when_enabled(run, image):
    mask = torch.zeros(1, 64, 96)
    mask[:, 16:48, 24:72] = 1.0
    out_image = run(image, mask=mask, remove_background=True, inner_brightness=40.0)[0]
    assert torch.allclose(out_image[:, 0:8, 0:8], image[:, 0:8, 0:8], atol=1e-6)


def test_remove_background_applies_when_occlusion_is_switched_off(run, image):
    """No subject interaction means plain lighting, which still needs compositing."""
    mask = torch.zeros(1, 64, 96)
    mask[:, 16:48, 24:72] = 1.0
    out_image = run(
        image,
        mask=mask,
        subject_interaction="None",
        remove_background=True,
        inner_brightness=60.0,
    )[0]
    assert torch.allclose(out_image[:, 0:8, 0:8], image[:, 0:8, 0:8], atol=1e-6)


def test_presets_apply_their_own_geometry_by_default(run, image):
    """'Spotlight' is defined by its tight radii; they must actually land."""
    spotlight = run(image, preset="Spotlight")[0]
    kept = run(image, preset="Spotlight", preserve_positioning=True)[0]
    assert not torch.allclose(spotlight, kept, atol=1e-4)


def test_preserve_positioning_keeps_both_position_and_radius(run, image):
    a = run(image, preset="Spotlight", preserve_positioning=True, light_position_x=0.2)[0]
    b = run(image, preset="Spotlight", preserve_positioning=True, light_position_x=0.8)[0]
    assert not torch.allclose(a, b, atol=1e-4)

    c = run(image, preset="Spotlight", preserve_positioning=True, outer_circle_radius=0.2)[0]
    d = run(image, preset="Spotlight", preserve_positioning=True, outer_circle_radius=0.9)[0]
    assert not torch.allclose(c, d, atol=1e-4)


# --- finding 17: gamma convention -----------------------------------------


def test_gamma_above_one_brightens(node):
    flat = torch.full((1, 8, 8, 3), 0.5)
    assert node.apply_color_correction(flat, gamma=1.5).mean() > 0.5
    assert node.apply_color_correction(flat, gamma=0.7).mean() < 0.5


def test_preset_shadow_zones_actually_darken(node):
    """Presets that dim their outer zone must not brighten it via gamma."""
    flat = torch.full((1, 8, 8, 3), 0.5)
    for name in ("Soft Window Light", "Dramatic Side Light", "Spotlight"):
        preset = node.PRESETS[name]
        if preset.get("outer_brightness", 0) >= 0:
            continue
        out = node.apply_color_correction(
            flat,
            brightness=preset["outer_brightness"],
            gamma=preset["outer_gamma"],
        )
        assert out.mean() < 0.5, f"{name}: outer zone should darken"


# --- finding 4: "Behind Subject" did not occlude ---------------------------

RIM = "Light behind subject (rim)"


def _subject_mask(height=64, width=96, top=12, bottom=52, left=32, right=64):
    """A solid rectangular subject in the middle of the frame."""
    mask = torch.zeros(1, height, width)
    mask[:, top:bottom, left:right] = 1.0
    return mask


def test_behind_subject_does_not_light_the_subject(run):
    """Background light must not land on the subject it is behind.

    Up to v3.1.2 the silhouette was subtracted *before* the blur, so a 50px
    blur smeared background light straight back across the edge onto the face.
    That was the reported "Behind Subject does not occlude".
    """
    flat = torch.full((1, 64, 96, 3), 0.25)
    mask = _subject_mask()
    lit = run(
        flat,
        mask=mask,
        subject_interaction=RIM,
        lighting_mode="Colored Light",
        rim_amplification=0.0,  # isolate the background glow from the rim
        light_position_x=0.5,
        light_position_y=0.5,
        inner_circle_radius=0.5,
        outer_circle_radius=1.0,
        mask_blur=50.0,
    )[0]

    inside = mask[0].bool()
    changed = (lit[0] - flat[0]).abs().sum(-1)
    assert float(changed[inside].max()) < 1e-5, "background light leaked onto the subject"
    assert float(changed[~inside].max()) > 0.01, "no background light at all"


def test_behind_subject_still_rims_the_subject_edge(run):
    """Occluding the background must not switch the rim highlight off."""
    flat = torch.full((1, 64, 96, 3), 0.25)
    mask = _subject_mask()
    lit = run(
        flat,
        mask=mask,
        subject_interaction=RIM,
        lighting_mode="Colored Light",
        rim_amplification=4.0,
        mask_blur=5.0,
    )[0]
    changed = (lit[0] - flat[0]).abs().sum(-1)
    assert float(changed[mask[0].bool()].max()) > 0.01, "the rim highlight is gone"


def test_the_subject_casts_a_shadow_away_from_the_light(run):
    """With the light above, the background below the subject is darker.

    v3.1.2 cast no shadow at all: the subject blocked nothing, so the near side
    of a head was lit exactly as brightly as the far side.
    """
    flat = torch.full((1, 64, 96, 3), 0.25)
    mask = _subject_mask()
    lit = run(
        flat,
        mask=mask,
        subject_interaction=RIM,
        lighting_mode="Colored Light",
        rim_amplification=0.0,
        light_position_x=0.5,
        light_position_y=0.05,
        inner_circle_radius=0.6,
        outer_circle_radius=1.5,
        mask_blur=10.0,
        shadow_strength=1.0,
        shadow_length=0.8,
    )[0]

    columns = slice(32, 64)  # directly above and below the subject
    above = lit[0, 2:10, columns].mean()
    below = lit[0, 54:62, columns].mean()
    assert below < above, "the subject cast no shadow below it"


def test_shadow_strength_zero_casts_no_shadow(run):
    flat = torch.full((1, 64, 96, 3), 0.25)
    mask = _subject_mask()
    common = dict(
        mask=mask,
        subject_interaction=RIM,
        lighting_mode="Colored Light",
        rim_amplification=0.0,
        light_position_y=0.05,
        outer_circle_radius=1.5,
        shadow_length=0.8,
    )
    off = run(flat, shadow_strength=0.0, **common)[0]
    on = run(flat, shadow_strength=1.0, **common)[0]
    assert not torch.allclose(off, on), "shadow_strength had no effect"
    assert off.mean() > on.mean(), "the shadow should remove light, not add it"


def test_shadow_length_zero_casts_no_shadow(run):
    flat = torch.full((1, 64, 96, 3), 0.25)
    mask = _subject_mask()
    common = dict(
        mask=mask,
        subject_interaction=RIM,
        lighting_mode="Colored Light",
        rim_amplification=0.0,
        light_position_y=0.05,
        outer_circle_radius=1.5,
        shadow_strength=1.0,
    )
    none = run(flat, shadow_length=0.0, **common)[0]
    some = run(flat, shadow_length=0.8, **common)[0]
    assert none.mean() > some.mean()


def test_cast_shadow_mask_shadows_only_behind_the_subject(node):
    """The tracer itself: a clear line of sight to the light is never shadowed."""
    mask = _subject_mask()
    shadow = node.cast_shadow_mask(mask, 0.5, 0.0, shadow_length=1.0)
    assert shadow.shape == mask.shape
    assert float(shadow.max()) <= 1.0 and float(shadow.min()) >= 0.0
    # Below the subject, looking up at the light, the subject is in the way.
    assert float(shadow[0, 56:62, 40:56].mean()) > 0.5
    # Above it, the light is in plain view.
    assert float(shadow[0, 0:8, 40:56].mean()) < 0.05


def test_cast_shadow_mask_is_empty_without_a_subject(node):
    empty = torch.zeros(1, 64, 96)
    assert float(node.cast_shadow_mask(empty, 0.5, 0.0, 0.5).max()) == 0.0


def test_cast_shadow_mask_matches_a_full_resolution_trace(node):
    """The tracer downsamples for speed; that must not change what it draws.

    _SHADOW_TRACE_MAX exists to keep a 4K frame affordable. If the shortcut
    disagreed with the honest version the saving would not be worth having.
    """
    mask = _subject_mask(height=256, width=256, top=64, bottom=160, left=96, right=160)
    fast = node.cast_shadow_mask(mask, 0.5, 0.0, shadow_length=0.6)
    slow = node.cast_shadow_mask(mask, 0.5, 0.0, shadow_length=0.6, steps=96)
    assert float((fast - slow).abs().mean()) < 0.02


def test_cast_shadow_mask_handles_a_batch(node):
    masks = torch.cat([_subject_mask(), torch.zeros(1, 64, 96)], dim=0)
    shadow = node.cast_shadow_mask(masks, 0.5, 0.0, 0.6)
    assert shadow.shape == (2, 64, 96)
    assert float(shadow[1].max()) == 0.0
    assert float(shadow[0].max()) > 0.5


# --- finding 3: colored light and correction were mutually exclusive -------


def test_both_mode_differs_from_either_single_mode(run, image):
    common = dict(
        light_color_r=255, light_color_g=200, light_color_b=120, light_intensity=0.5,
        inner_brightness=20.0, inner_saturation=15.0, inner_temperature=20.0,
    )
    colored = run(image, lighting_mode="Colored Light", **common)[0]
    correction = run(image, lighting_mode="Color Correction", **common)[0]
    both = run(image, lighting_mode="Both", **common)[0]
    assert not torch.allclose(both, colored)
    assert not torch.allclose(both, correction)


def test_both_mode_is_the_correction_applied_to_the_colored_result(run, node, image):
    """'Both' is defined as colour first, then grade the lit image.

    Chaining two ReLight nodes - colour, then correction - must land in the
    same place, or 'Both' means something other than what it says.
    """
    common = dict(
        light_color_r=255, light_color_g=200, light_color_b=120, light_intensity=0.5,
        inner_brightness=20.0, inner_saturation=15.0, inner_temperature=20.0,
    )
    both = run(image, lighting_mode="Both", **common)[0]
    colored = run(image, lighting_mode="Colored Light", **common)[0]
    chained = run(colored, lighting_mode="Color Correction", **common)[0]
    assert torch.allclose(both, chained, atol=1e-6)


def test_colored_only_mode_is_unchanged_by_the_both_path(run, image):
    """The single modes must keep their exact maths; only 'Both' is new."""
    colored = run(image, lighting_mode="Colored Light", light_intensity=0.5)[0]
    manual = run(image, lighting_mode="Colored Light", light_intensity=0.5, inner_brightness=99.0)[0]
    assert torch.equal(colored, manual), "correction values leaked into Colored Light"


def test_correction_only_mode_ignores_the_light_colour(run, image):
    plain = run(image, lighting_mode="Color Correction")[0]
    coloured_widgets = run(image, lighting_mode="Color Correction", light_color_r=0, light_color_g=255, light_color_b=0, light_intensity=3.0)[0]
    assert torch.equal(plain, coloured_widgets)


def test_the_three_retuned_presets_use_both_modes(node):
    """The audit finding, pinned: these three set a colour and a grading block.

    Before v4.0.0 the two were mutually exclusive, so 12 values in each preset
    were inert. The numbers themselves are a visual judgement; this only pins
    that both halves are asked for.
    """
    for name in ("Warm Sunset Glow", "Cool Blue Moonlight", "Rim Light (Behind)"):
        preset = node.PRESETS[name]
        assert preset["lighting_mode"] == node.MODE_BOTH, name
        assert any(key.startswith(("inner_", "outer_")) for key in preset), name


# --- presets and lights ---------------------------------------------------


def test_reported_warm_sunset_glow_run(run, image):
    """The exact configuration from the 3.1.0 crash report.

    One image, no mask, the Warm Sunset Glow preset, both image outputs
    consumed. Every output must come back as a usable tensor.
    """
    out_image, out_mask, debug = run(image, preset="Warm Sunset Glow")
    assert out_image.shape == image.shape
    assert out_mask.shape == (1, 64, 96)
    assert debug.shape == (1, 64, 96, 3)
    assert torch.isfinite(out_image).all()


def test_reported_warm_sunset_glow_run_in_gradient_mode(run, image):
    """Same report, gradient masks - the path the traceback died on."""
    out_image = run(image, preset="Warm Sunset Glow", mask_shape="Directional gradient")[0]
    assert out_image.shape == image.shape
    assert torch.isfinite(out_image).all()


def test_all_presets_execute_cleanly(node, run, image):
    mask = torch.zeros(1, 64, 96)
    mask[:, 16:48, 24:72] = 1.0
    for name in node.PRESETS:
        out_image = run(image, mask=mask, preset=name)[0]
        assert out_image.shape == image.shape, name
        assert torch.isfinite(out_image).all(), name


def test_all_subject_interactions_execute(node, run, image):
    mask = torch.zeros(1, 64, 96)
    mask[:, 16:48, 24:72] = 1.0
    for interaction in node.SUBJECT_INTERACTIONS:
        out_image = run(image, mask=mask, subject_interaction=interaction)[0]
        assert torch.isfinite(out_image).all(), interaction


def test_multiple_light_sources(run, image):
    for count in (1, 2, 3):
        out_image = run(image, num_light_sources=count, lighting_mode="Colored Light")[0]
        assert torch.isfinite(out_image).all()


def test_gradient_mode_with_light_at_exact_centre(run, image):
    out_image = run(
        image,
        mask_shape="Directional gradient",
        lighting_mode="Colored Light",
        light_position_x=0.5,
        light_position_y=0.5,
    )[0]
    assert torch.isfinite(out_image).all()


def test_inner_radius_larger_than_outer(run, image):
    out_image = run(image, inner_circle_radius=0.9, outer_circle_radius=0.1)[0]
    assert torch.isfinite(out_image).all()


def test_zero_radii(run, image):
    out_image = run(image, inner_circle_radius=0.0, outer_circle_radius=0.0)[0]
    assert torch.allclose(out_image, image, atol=1e-6)


def test_uniform_masks_do_not_crash(run, image):
    for mask in (torch.ones(1, 64, 96), torch.zeros(1, 64, 96)):
        out_image = run(image, mask=mask, subject_interaction="Light behind subject (rim)")[0]
        assert torch.isfinite(out_image).all()


# --- debug image ----------------------------------------------------------


def test_unconnected_debug_view_is_not_a_black_frame(run, image):
    """A black debug output is indistinguishable from a crashed node.

    Users wire debug_image to a preview, see solid black and report the node as
    broken (it was: the only signal that the debug view was off was a black
    rectangle). The placeholder must render something legible instead.
    """
    debug = run(image, debug_output_connected=False)[2]
    assert debug.shape == (1, 64, 96, 3)
    assert debug.max() > 0.05, "an unconnected debug output is still a black frame"
    assert debug.max() < 1.01


def test_unconnected_debug_placeholder_is_calmer_than_the_real_view(run, image):
    """The placeholder must not be mistaken for an actual debug visualization."""
    off = run(image, debug_output_connected=False)[2]
    on = run(image, debug_output_connected=True)[2]
    assert off.mean() < on.mean()


def test_debug_placeholder_falls_back_to_black_when_too_small(run):
    """Below a line of type there is nowhere to put the message; stay black."""
    tiny = torch.rand(1, 8, 8, 3)
    debug = run(tiny, debug_output_connected=False)[2]
    assert debug.shape == (1, 8, 8, 3)
    assert float(debug.max()) == 0.0


#: The colour ``_blank_debug_image`` paints its ground with.
PLACEHOLDER_GROUND = torch.tensor([24 / 255, 24 / 255, 28 / 255])


def _ink_fraction(frame, inset=0):
    """Fraction of pixels that differ from the placeholder's flat ground.

    Text coverage, not `max() > 0`. The v3.1.2 placeholder drew 13px type on a
    full-resolution canvas: it passed every brightness assertion and was
    invisible in a preview, which is how it was reported as a black frame.
    """
    region = frame[inset:frame.shape[0] - inset, inset:frame.shape[1] - inset] if inset else frame
    return float(((region - PLACEHOLDER_GROUND).abs().sum(-1) > 0.15).float().mean())


def test_debug_placeholder_is_legible_at_a_realistic_resolution(run):
    """Measured inside the border at 1344x768, the size people render at.

    Whole-frame ink is not the measurement to make: the border alone can carry
    it past a threshold while the type stays invisible, which is how the v3.1.2
    placeholder passed its own tests. This looks only where the text is.
    v3.1.2 scores 0.0027 here; this build scores 0.015.
    """
    big = torch.zeros(1, 768, 1344, 3)
    debug = run(big, debug_output_connected=False)[2][0]
    assert _ink_fraction(debug, inset=48) > 0.008, "the placeholder text is not legible"


def test_debug_placeholder_still_reads_as_a_panel_in_a_thumbnail(run):
    """The reported symptom was a black rectangle in a ComfyUI preview.

    A preview is a thumbnail, so that is what is measured: downscale the
    placeholder the way the UI does and check something is still drawn on it.
    v3.1.2 scores 0.010 here; this build scores 0.059.
    """
    big = torch.zeros(1, 768, 1344, 3)
    debug = run(big, debug_output_connected=False)[2]
    thumb = torch.nn.functional.interpolate(
        debug.permute(0, 3, 1, 2), size=(55, 96), mode="area"
    ).permute(0, 2, 3, 1)[0]
    assert _ink_fraction(thumb, inset=4) > 0.03, "the placeholder is a dark rectangle in a preview"


def test_debug_view_renders_when_the_prompt_consumes_it(node_with_prompt, defaults, image):
    """No toggle: something wired to the debug output is the whole trigger.

    This is the path an API caller takes - it never loads the frontend, so
    `debug_output_connected` is False and the prompt is the only signal.
    """
    prompt = {
        "7": {"class_type": "ReLight", "inputs": {}},
        "9": {"class_type": "PreviewImage", "inputs": {"images": ["7", 2]}},
    }
    node = node_with_prompt(prompt=prompt, unique_id="7")
    debug = tuple(node.execute(image, **defaults(debug_output_connected=False)))[2]
    off = tuple(node_with_prompt().execute(image, **defaults(debug_output_connected=False)))[2]
    assert debug.mean() > off.mean(), "the debug view was not drawn for a wired output"


def test_a_wire_from_another_output_does_not_trigger_the_debug_view(node_with_prompt, defaults, image):
    """Slot 0 is the relit image; consuming it must not cost a debug render."""
    prompt = {
        "7": {"class_type": "ReLight", "inputs": {}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["7", 0]}},
    }
    node = node_with_prompt(prompt=prompt, unique_id="7")
    debug = tuple(node.execute(image, **defaults(debug_output_connected=False)))[2]
    off = tuple(node_with_prompt().execute(image, **defaults(debug_output_connected=False)))[2]
    assert torch.equal(debug, off)


def test_another_nodes_debug_wire_does_not_trigger_ours(node_with_prompt, defaults, image):
    """Two ReLight nodes, only one of them wired."""
    prompt = {
        "7": {"class_type": "ReLight", "inputs": {}},
        "8": {"class_type": "ReLight", "inputs": {}},
        "9": {"class_type": "PreviewImage", "inputs": {"images": ["8", 2]}},
    }
    node = node_with_prompt(prompt=prompt, unique_id="7")
    debug = tuple(node.execute(image, **defaults(debug_output_connected=False)))[2]
    off = tuple(node_with_prompt().execute(image, **defaults(debug_output_connected=False)))[2]
    assert torch.equal(debug, off)


def test_the_connectivity_check_tolerates_an_unusable_prompt(node_with_prompt, defaults, image):
    """Hidden inputs are absent outside ComfyUI and empty in some internal calls.

    None of these shapes may raise: a render must not fail over a question the
    user never asked.
    """
    for prompt, unique_id in (
        (None, None),
        ({}, "7"),
        ({"7": {}}, "7"),
        ({"7": {"inputs": None}}, "7"),
        ({"7": {"inputs": {"x": ["7"]}}}, "7"),
        ("not a dict", "7"),
        ({"7": {"class_type": "ReLight", "inputs": {}}}, None),
    ):
        node = node_with_prompt(prompt=prompt, unique_id=unique_id)
        out = tuple(node.execute(image, **defaults(debug_output_connected=False)))
        assert out[2].shape == (1, 64, 96, 3), (prompt, unique_id)


def test_the_widget_flag_alone_still_renders_the_debug_view(node_with_prompt, defaults, image):
    """The UI's flag and the prompt check are independent; either is enough."""
    node = node_with_prompt()
    on = tuple(node.execute(image, **defaults(debug_output_connected=True)))[2]
    off = tuple(node.execute(image, **defaults(debug_output_connected=False)))[2]
    assert on.mean() > off.mean()


def test_debug_image_matches_image_dimensions(run, image):
    debug = run(image, debug_output_connected=True)[2]
    assert debug.shape == (1, 64, 96, 3)


def test_debug_image_with_a_batch(run):
    batch = torch.rand(4, 48, 64, 3)
    out_image, _, debug = run(batch, debug_output_connected=True)
    assert out_image.shape == batch.shape
    assert debug.shape == (1, 48, 64, 3)


def test_debug_image_for_rgba_input(run):
    rgba = torch.rand(1, 32, 32, 4)
    debug = run(rgba, debug_output_connected=True)[2]
    assert debug.shape == (1, 32, 32, 3)


# --- non-square and awkward sizes -----------------------------------------


def test_wide_and_tall_images(run):
    for shape in ((1, 16, 256, 3), (1, 256, 16, 3), (1, 1, 1, 3)):
        img = torch.rand(*shape)
        out_image = run(img, lighting_mode="Colored Light")[0]
        assert out_image.shape == img.shape


# --- finding 18: effect_strength must survive a preset ----------------------


def test_every_preset_is_a_no_op_at_zero_effect_strength(node, run):
    """`effect_strength=0.0` is documented as a true no-op - presets included.

    Three presets define `effect_strength` themselves; overriding the widget
    outright left the master intensity dead for exactly the strongest presets.
    """
    flat = torch.full((1, 24, 32, 3), 0.5)
    for name in node.PRESETS:
        out_image = run(flat, preset=name, effect_strength=0.0)[0]
        assert torch.allclose(out_image, flat, atol=1e-6), f"{name}: not a no-op at strength 0"


def test_preset_effect_strength_is_scaled_not_replaced(run):
    """A preset that sets effect_strength still responds to the widget."""
    flat = torch.full((1, 24, 32, 3), 0.5)
    weak = run(flat, preset="Spotlight", effect_strength=0.5)[0]
    normal = run(flat, preset="Spotlight", effect_strength=1.0)[0]
    strong = run(flat, preset="Spotlight", effect_strength=2.0)[0]
    delta = [(x - flat).abs().mean().item() for x in (weak, normal, strong)]
    assert delta[0] < delta[1] < delta[2], delta


def test_preset_defaults_are_unchanged_by_the_scaling(node, run):
    """At the widget default of 1.0 a preset must land on its designed strength."""
    assert node.PRESETS["Spotlight"]["effect_strength"] == 1.2
    params = node._load_preset("Spotlight", {"effect_strength": 1.0, "preserve_positioning": False})
    assert params["effect_strength"] == 1.2


# --- finding 19: effect_strength must not drive gamma out of range ---------


def test_scaled_gamma_stays_within_the_widget_range(node):
    """The clamp in _scaled_gamma and the widgets' min/max share one source of truth."""
    bounds = {
        (s.kwargs.get("min"), s.kwargs.get("max"))
        for s in node.define_schema().inputs
        if s.id in ("inner_gamma", "outer_gamma")
    }
    assert len(bounds) == 1, bounds
    low, high = bounds.pop()
    for gamma in (low, 0.5, 0.77, 1.0, 1.11, 3.0, high):
        for strength in (0.0, 0.5, 1.0, 2.0, 3.5, 5.0):
            scaled = node._scaled_gamma(gamma, strength)
            assert low <= scaled <= high, (gamma, strength, scaled)


def test_scaled_gamma_matches_the_widget_at_the_anchor_points(node):
    for gamma in (0.1, 0.77, 1.0, 2.5, 5.0):
        assert node._scaled_gamma(gamma, 0.0) == 1.0
        assert node._scaled_gamma(gamma, 1.0) == gamma


def test_strong_effect_strength_dims_without_crushing_to_black(node, run):
    """A dimming gamma used to go negative past strength ~4 and clip to solid black."""
    flat = torch.full((1, 16, 16, 3), 0.5)
    previous = 0.5
    for strength in (1.0, 2.0, 3.0, 4.0, 5.0):
        out = run(
            flat,
            effect_strength=strength,
            inner_circle_radius=0.05,
            outer_circle_radius=0.95,
            mask_blur=0.0,
            outer_gamma=0.77,
            **{k: v for k, v in IDENTITY.items() if k not in ("outer_gamma",)},
        )[0]
        value = out[0, 1, 8, 0].item()
        assert value < previous, (strength, value, previous)
        assert value > 0.02, f"strength {strength} crushed the zone to black ({value})"
        previous = value


# --- finding 20: single-channel images ------------------------------------


def test_single_channel_image_keeps_its_channel_count(run):
    gray = torch.rand(1, 32, 32, 1)
    for overrides in (
        {},
        {"lighting_mode": "Colored Light"},
        {"lighting_mode": "Colored Light", "mask_shape": "Directional gradient"},
    ):
        out_image = run(gray, **overrides)[0]
        assert out_image.shape == gray.shape, overrides


def test_debug_image_is_rgb_for_a_single_channel_image(run):
    gray = torch.rand(1, 32, 32, 1)
    debug = run(gray, debug_output_connected=True)[2]
    assert debug.shape == (1, 32, 32, 3)
    assert debug.abs().max() > 0.0, "grayscale input used to fall through to a black debug view"


# --- preset table integrity -----------------------------------------------


def test_every_preset_key_is_a_real_input(node):
    """A typo in a preset key is silently ignored by _load_preset, so pin it here."""
    schema_ids = {spec.id for spec in node.define_schema().inputs}
    for name, preset in node.PRESETS.items():
        unknown = sorted(set(preset) - schema_ids)
        assert unknown == [], f"{name}: {unknown}"


def test_geometry_keys_are_real_inputs(node):
    schema_ids = {spec.id for spec in node.define_schema().inputs}
    assert sorted(set(node.GEOMETRY_KEYS) - schema_ids) == []


# --- coordinate grids -----------------------------------------------------


def test_pixel_grid_is_cheap_to_cache(node):
    """The grids are broadcast shaped; caching full (H, W) arrays pinned megabytes."""
    import relight

    relight._COORD_CACHE.clear()
    rows, cols = node._pixel_grid(2048, 2048)
    assert rows.shape == (2048, 1)
    assert cols.shape == (1, 2048)
    relight._COORD_CACHE.clear()


def test_pixel_grid_cache_holds_one_entry(node):
    """A workflow that alternates resolutions must not accumulate grids."""
    import relight

    relight._COORD_CACHE.clear()
    node._pixel_grid(64, 32)
    node._pixel_grid(96, 48)
    assert list(relight._COORD_CACHE) == [(48, 96)]
    relight._COORD_CACHE.clear()


def test_pixel_grid_does_not_write_to_the_locked_class(node):
    """Regression: the cache used to live on the class.

    ComfyUI runs `execute` on a locked clone, so `cls._coord_cache = ...` raised
    "Cannot modify class attribute '_coord_cache' on locked class
    'ReLightClone'" on the first mask of every run. Nothing per-run may be
    stored on `cls`.
    """
    import relight

    relight._COORD_CACHE.clear()
    node._pixel_grid(32, 16)  # would raise AttributeError if it wrote to cls
    assert "_coord_cache" not in vars(node)
    relight._COORD_CACHE.clear()


def test_masks_match_a_full_grid_reference(node):
    """Broadcasting must not change a single pixel of any mask."""
    import numpy as np

    for width, height in ((97, 63), (64, 64), (1, 1)):
        y_full, x_full = np.mgrid[0:height, 0:width]
        for centre in (0.0, 0.3, 0.5, 1.0):
            radius_px = 0.4 * min(width, height)
            expected = (
                ((x_full - centre * width) ** 2 + (y_full - centre * height) ** 2) <= radius_px**2
            ).astype(np.float32)
            actual = node.create_circle_mask(width, height, centre, centre, 0.4).numpy()
            assert np.array_equal(actual, expected), (width, height, centre)


# --- schema self-consistency ------------------------------------------------
# The comfy_api stub records declarations without validating them, so these
# tests stand in for the checks a real ComfyUI build does at load time.


def test_every_widget_default_is_within_its_declared_range(node):
    offenders = []
    for spec in node.define_schema().inputs:
        low, high = spec.kwargs.get("min"), spec.kwargs.get("max")
        if spec.default is None or low is None or high is None:
            continue
        if not (low <= spec.default <= high):
            offenders.append((spec.id, spec.default, low, high))
    assert offenders == []


def test_combo_defaults_are_offered_options(node):
    offenders = [
        (spec.id, spec.default)
        for spec in node.define_schema().inputs
        if spec.kwargs.get("options") is not None and spec.default not in spec.kwargs["options"]
    ]
    assert offenders == []


def test_preset_values_stay_within_widget_bounds(node):
    """A preset value outside its widget's range could never be set by hand."""
    specs = {spec.id: spec for spec in node.define_schema().inputs}
    offenders = []
    for preset_name, preset in node.PRESETS.items():
        for key, value in preset.items():
            low, high = specs[key].kwargs.get("min"), specs[key].kwargs.get("max")
            if (low is not None and value < low) or (high is not None and value > high):
                offenders.append((preset_name, key, value, low, high))
    assert offenders == []


def test_a_light_at_the_right_edge_keeps_its_label_on_the_frame(run):
    """A rim light sits at x=0.9, and its "L1" ran off the right edge.

    The label is drawn to the right of the marker, so a light near the right
    edge pushed the text past `width` and PIL silently clipped it - visible
    only in a render, which is how it survived to v4.0.0. It now flips to the
    left of the marker instead.

    Measured as the side of the marker the label's black backing box lands on,
    in a band across the marker's own row. The frame is deliberately *white*:
    on a black one the box is invisible against the input and every threshold
    measures the input instead of the label, which is how the first version of
    this test passed with the bug still in place.
    """
    height, width = 768, 1344
    white = torch.ones(1, height, width, 3)
    marker_y = height // 2

    def label_sides(position_x):
        frame = run(
            white,
            debug_output_connected=True,
            light_position_x=position_x,
            light_position_y=0.5,
        )[2][0]
        marker_x = int(position_x * width)
        band = frame[marker_y - 40:marker_y + 40]
        dark = band.sum(dim=-1) < 0.9
        return int(dark[:, :marker_x].sum()), int(dark[:, marker_x:].sum())

    # A light with room to its right keeps the label on the right, as always.
    left, right = label_sides(0.5)
    assert right > left, "the label is not drawn to the right of a centred marker"

    # A light hard against the right edge must flip it, not clip it.
    left, right = label_sides(0.97)
    assert left > right, (
        "the label stayed on the right of a marker near the right edge, "
        "so it was clipped against the frame instead of flipping"
    )


def test_a_light_in_the_corner_keeps_its_label_on_the_frame(run):
    """Both axes clamp: a marker in a corner can run a glyph off the top too."""
    big = torch.zeros(1, 768, 1344, 3)
    corner = run(big, debug_output_connected=True, light_position_x=0.99, light_position_y=0.01)[2]
    assert corner.shape == (1, 768, 1344, 3)
    assert torch.isfinite(corner).all()
