import { app } from "../../scripts/app.js";

/*
 * ReLight legacy-workflow migration.
 *
 * Why this file exists
 * --------------------
 * LiteGraph stores a node's widget values as a bare positional array. Up to
 * v3.1.2 ReLight had 48 widgets in one flat order; v4.0.0 regrouped them,
 * replaced four booleans with named combos, and added two shadow controls.
 * Loading a pre-v4 workflow against the v4 schema therefore lands every value
 * from the first changed index onward on the wrong widget - silently, with no
 * error, which is worse than a crash because the graph still runs.
 *
 * `onConfigure` runs after LiteGraph has assigned those values, so this module
 * detects a legacy array and rewrites the widgets by NAME, which is
 * order-independent by construction. Widgets that did not exist in v3.1.2 are
 * reset to their schema default (they are currently holding a shifted legacy
 * value).
 *
 * Ported from EA_LMStudio/web/ea_lmstudio.js (`migrateLegacyWidgetValues`,
 * `growToFitWidgets`) with ReLight's four boolean -> combo translations added.
 *
 * The option strings below are the saved-workflow format. They are mirrored in
 * relight.py (`ReLight.LIGHTING_MODES` and friends) and pinned in
 * tests/test_relight.py - change one and you must change all three.
 */

const NODE_CLASS = "ReLight";

/*
 * Widget order exactly as v3.1.2 serialised it, in LiteGraph order. IMAGE and
 * MASK are link inputs, not widgets, so they never appear here.
 *
 * Pinned as data in tests/test_relight.py (`LEGACY_WIDGET_ORDER`), which reads
 * this file and compares - the two cannot drift apart without going red.
 */
export const LEGACY_ORDER = [
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
];

/** v3.1.2 preset names. Index 0 of a legacy array is always one of these. */
export const LEGACY_PRESETS = [
    "None",
    "Soft Window Light",
    "Dramatic Side Light",
    "Warm Sunset Glow",
    "Cool Blue Moonlight",
    "Studio Key Light",
    "Rim Light (Behind)",
    "Spotlight",
    "Negative Light (Darken)",
];

/** v3.1.2 `light_direction` values. Index 7 of a legacy array is one of these. */
export const LEGACY_LIGHT_DIRECTIONS = [
    "Behind Subject",
    "In Front of Subject",
    "No Occlusion",
];

// v4.0.0 combo values. Kept as constants so the translations below read as a
// mapping rather than as string literals scattered through branches.
const MODE_CORRECTION = "Color Correction";
const MODE_COLORED = "Colored Light";
const SHAPE_RADIAL = "Radial falloff";
const SHAPE_GRADIENT = "Directional gradient";
const SUBJECT_NONE = "None";
const SUBJECT_FRONT = "Light in front of subject";
const SUBJECT_RIM = "Light behind subject (rim)";

function warn(message, error) {
    console.warn("[ReLight] " + message, error || "");
}

/**
 * Collect each input's declared default, so a widget that did not exist in
 * v3.1.2 can be reset after the remap instead of keeping whatever legacy value
 * happened to land at its index.
 */
export function collectDefaults(nodeData) {
    const defaults = {};
    for (const group of ["required", "optional"]) {
        const inputs = nodeData?.input?.[group] ?? {};
        for (const [name, spec] of Object.entries(inputs)) {
            const [type, options] = Array.isArray(spec) ? spec : [spec, undefined];
            if (options && Object.prototype.hasOwnProperty.call(options, "default")) {
                defaults[name] = options.default;
            } else if (Array.isArray(type)) {
                defaults[name] = type[0]; // combo: first entry
            }
        }
    }
    return defaults;
}

/**
 * Translate the four v3.1.2 booleans into the v4.0.0 combos.
 *
 * `legacy` is the legacy array keyed by name. Returns the values for the new
 * widgets only; every other widget keeps its legacy value under the same name.
 */
export function translateLegacyModes(legacy) {
    const translated = {};

    // use_colored_lights was the whole mode switch: on meant additive colour
    // and the entire inner_*/outer_* grading block was ignored. Migrating to
    // "Both" would therefore change the render of every colour workflow, so a
    // legacy save maps to the mode that matches what it actually produced.
    translated.lighting_mode = legacy.use_colored_lights ? MODE_COLORED : MODE_CORRECTION;

    translated.mask_shape = legacy.use_gradient_mode ? SHAPE_GRADIENT : SHAPE_RADIAL;

    // apply_3d_lighting was a master switch whose only effect was to force
    // "No Occlusion", so it collapses into the direction it was gating.
    if (!legacy.apply_3d_lighting) {
        translated.subject_interaction = SUBJECT_NONE;
    } else if (legacy.light_direction === "Behind Subject") {
        translated.subject_interaction = SUBJECT_RIM;
    } else if (legacy.light_direction === "In Front of Subject") {
        translated.subject_interaction = SUBJECT_FRONT;
    } else {
        translated.subject_interaction = SUBJECT_NONE;
    }

    // show_debug_info has no v4 counterpart and is deliberately dropped: the
    // debug view now follows the wiring of the debug_image output, which
    // relight_debug.js keeps `debug_output_connected` in step with. Whatever
    // the old toggle said, the connection is the truth.

    return translated;
}

/**
 * Is this widgets_values array a v3.1.2 (or earlier) save?
 *
 * Length alone is not enough - v4 has 49 widgets, but a v4 node with one widget
 * converted to an input socket would serialise 48 values. So the length test is
 * backed by two sentinels that a v4 array cannot satisfy: index 0 must be a
 * v3.1.2 preset name, and index 7 must be one of the three legacy
 * `light_direction` strings (in v4, index 7 is the numeric effect_strength).
 */
export function looksLegacy(widgetValues) {
    if (!Array.isArray(widgetValues)) return false;
    if (widgetValues.length !== LEGACY_ORDER.length) return false;
    if (!LEGACY_PRESETS.includes(widgetValues[LEGACY_ORDER.indexOf("preset")])) return false;
    if (!LEGACY_LIGHT_DIRECTIONS.includes(widgetValues[LEGACY_ORDER.indexOf("light_direction")])) {
        return false;
    }
    return true;
}

/**
 * Repair widget values loaded from a pre-v4 workflow.
 *
 * Returns true when a remap happened, false when the array was left alone.
 */
export function migrateLegacyWidgetValues(node, widgetValues, defaults) {
    if (!looksLegacy(widgetValues)) return false;

    const legacy = {};
    LEGACY_ORDER.forEach((name, index) => {
        legacy[name] = widgetValues[index];
    });
    const translated = translateLegacyModes(legacy);

    for (const widget of node.widgets ?? []) {
        const name = widget?.name;
        if (!name) continue;
        if (Object.prototype.hasOwnProperty.call(translated, name)) {
            widget.value = translated[name];
        } else if (Object.prototype.hasOwnProperty.call(legacy, name)) {
            widget.value = legacy[name];
        } else if (Object.prototype.hasOwnProperty.call(defaults, name)) {
            // New in v4.0.0 (shadow_strength, shadow_length) - right now it is
            // holding a shifted legacy value.
            widget.value = defaults[name];
        }
    }

    // No toast on purpose: the realignment is silent housekeeping, and a banner
    // on every legacy workflow load is noise users just click away.
    return true;
}

/**
 * Grow a node stored smaller than its widgets need.
 *
 * A workflow stores the node's size and LiteGraph restores it verbatim, without
 * re-checking that the widgets still fit. v4.0.0 has one more widget than
 * v3.1.2, so every upgraded workflow is a row short and the last widget draws
 * outside the frame. Only ever grows, so a deliberately widened node survives.
 */
export function growToFitWidgets(node) {
    try {
        const [minWidth, minHeight] = node.computeSize();
        if (node.size[0] < minWidth || node.size[1] < minHeight) {
            node.setSize([
                Math.max(node.size[0], minWidth),
                Math.max(node.size[1], minHeight),
            ]);
        }
    } catch (error) {
        warn("could not resize node to fit widgets:", error);
    }
}

app.registerExtension({
    name: "ReLight.migrate",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_CLASS) return;

        const defaults = collectDefaults(nodeData);

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            onConfigure?.apply(this, arguments);
            try {
                migrateLegacyWidgetValues(this, info?.widgets_values, defaults);
            } catch (error) {
                warn("legacy workflow migration failed:", error);
            }
            growToFitWidgets(this);
        };
    },
});
