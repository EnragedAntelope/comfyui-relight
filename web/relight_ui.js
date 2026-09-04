import { app } from "../../scripts/app.js";
import { GEOMETRY_KEYS, PRESET_KEYS, PRESET_VALUES, STRENGTH_KEY } from "./relight_presets.js";

/*
 * ReLight: show only the controls that are doing something.
 *
 * The node has 49 widgets and up to v3.1.2 painted every one of them, always.
 * Light 2 and Light 3 were on screen with `num_light_sources` at 1; the whole
 * grading block was on screen in colour mode, where it did nothing; a preset
 * silently overrode a dozen values that stayed fully draggable. All three are
 * the same defect - a control that looks live and is not.
 *
 * Two treatments, chosen deliberately:
 *
 *   HIDE what is irrelevant. Light 2/3 with one light source, colour controls
 *   in a grading-only mode, shadow controls when nothing casts a shadow. These
 *   are whole blocks, and every one of them sits BELOW the control that governs
 *   it, so hiding them never moves anything under the pointer that just clicked.
 *
 *   GREY OUT what a preset has taken over. `disabled = true` leaves the widget
 *   in place at its full height, so picking a preset does not reshuffle the
 *   node - and the preset's value stays readable, which matters, because seeing
 *   what "Spotlight" actually sets is how you learn to build your own.
 *
 * Three mechanics, all learned the hard way and all non-obvious:
 *
 *   Hiding needs `widget.type` swapped AND `widget.hidden = true`. Older
 *   LiteGraph skips widget types it does not know; newer frontends honour
 *   `hidden` and ignore the type. Set one and not the other and the widget
 *   stays painted on half the frontends out there.
 *
 *   Restore `computeSize` with `delete`, never by reassigning a saved copy.
 *   Most widgets define no `computeSize` of their own and inherit LiteGraph's,
 *   so the "saved" value is `undefined`; assigning it back leaves the zero-size
 *   stub in place and the widget never comes back. That is the single most
 *   likely cause of "hiding works but showing does not".
 *
 *   A greyed widget cannot show its own value. The frontend's `_displayValue`
 *   getter returns "" for anything with `computedDisabled` set, so `disabled`
 *   on its own leaves an empty bar with a dim label and no number - the
 *   opposite of "you can still read what the preset set". The label does still
 *   paint, so the value goes there instead. It is deliberately the *preset's*
 *   value and not the widget's: the widget still holds whatever the user last
 *   set, which is precisely the number the preset is ignoring.
 *
 * Changes are detected on both paths - the widget's own callback for an
 * immediate response, and a re-check on the draw loop, which is what catches a
 * workflow load, an undo, or an API edit that never fires a callback.
 */

const NODE_CLASS = "ReLight";

const MODE_CORRECTION = "Color Correction";
const MODE_COLORED = "Colored Light";
const SUBJECT_RIM = "Light behind subject (rim)";

const COLOUR_WIDGETS = ["light_color_r", "light_color_g", "light_color_b", "light_intensity"];
const GRADING_WIDGETS = [
    "inner_brightness", "inner_contrast", "inner_saturation",
    "inner_temperature", "inner_tint", "inner_gamma",
    "outer_brightness", "outer_contrast", "outer_saturation",
    "outer_temperature", "outer_tint", "outer_gamma",
];
const RIM_WIDGETS = ["rim_amplification", "shadow_strength", "shadow_length"];
const LIGHT_2 = [
    "light2_position_x", "light2_position_y", "light2_inner_radius", "light2_outer_radius",
    "light2_color_r", "light2_color_g", "light2_color_b", "light2_intensity",
];
const LIGHT_3 = [
    "light3_position_x", "light3_position_y", "light3_inner_radius", "light3_outer_radius",
    "light3_color_r", "light3_color_g", "light3_color_b", "light3_intensity",
];
const LIGHT_2_COLOUR = ["light2_color_r", "light2_color_g", "light2_color_b", "light2_intensity"];
const LIGHT_3_COLOUR = ["light3_color_r", "light3_color_g", "light3_color_b", "light3_intensity"];

const GEOMETRY = new Set(GEOMETRY_KEYS);

function warn(message, error) {
    console.warn("[ReLight] " + message, error || "");
}

function isReLight(node) {
    const type = node && (node.comfyClass || node.type);
    return type === NODE_CLASS;
}

function valuesOf(node) {
    const values = {};
    for (const widget of node.widgets ?? []) {
        if (widget?.name) values[widget.name] = widget.value;
    }
    return values;
}

function hideWidget(widget) {
    if (widget.hidden === true) return false;
    widget.hidden = true;
    widget.__relightType = widget.__relightType ?? widget.type;
    widget.type = "hidden-" + widget.__relightType;
    widget.computeSize = () => [0, -4];
    return true;
}

function showWidget(widget) {
    if (!widget.hidden) return false;
    widget.hidden = false;
    if (widget.__relightType !== undefined) widget.type = widget.__relightType;
    // delete, not reassign: most widgets have no own computeSize, so a saved
    // copy is `undefined` and assigning it back keeps the stub forever.
    delete widget.computeSize;
    return true;
}

/**
 * Which widgets should be hidden, given the node's current values.
 *
 * Pure and exported so it can be tested without a canvas.
 */
export function widgetsToHide(values) {
    const hidden = new Set();
    const add = (names) => names.forEach((name) => hidden.add(name));

    const mode = values.lighting_mode;
    if (mode === MODE_CORRECTION) add(COLOUR_WIDGETS);
    if (mode === MODE_COLORED) add(GRADING_WIDGETS);

    if (values.subject_interaction !== SUBJECT_RIM) add(RIM_WIDGETS);

    const lights = Number(values.num_light_sources) || 1;
    if (lights < 2) add(LIGHT_2);
    if (lights < 3) add(LIGHT_3);

    // Lights 2 and 3 have no grading of their own - they reuse Light 1's - so
    // in a grading-only mode their colour controls are as inert as Light 1's.
    if (mode === MODE_CORRECTION) {
        if (lights >= 2) add(LIGHT_2_COLOUR);
        if (lights >= 3) add(LIGHT_3_COLOUR);
    }

    // The flag relight_debug.js maintains is never a user control.
    hidden.add("debug_output_connected");
    return hidden;
}

/** Separates a widget's name from the preset value in a greyed-out label. */
const PRESET_LABEL_MARK = "  →  ";

/**
 * How a preset's value reads in a greyed widget's label.
 *
 * Numbers go through `String`, which drops JSON's trailing ".0" so an integer
 * preset value reads as `50` rather than `50.0`. Anything else (a combo's
 * string, a boolean) prints as itself.
 */
export function presetLabel(name, value) {
    return `${name}${PRESET_LABEL_MARK}${String(value)}`;
}

/** Is this label one ReLight wrote for a preset, rather than the user's own? */
function isPresetLabel(widget) {
    return typeof widget.label === "string" && widget.label.startsWith(widget.name + PRESET_LABEL_MARK);
}

/**
 * Grey a widget out, putting the preset's value where its own value would be.
 *
 * The original label is stashed rather than assumed absent: a future frontend
 * that ships display names would otherwise lose them on the first preset.
 */
function disableWidget(widget, presetValue) {
    const label = presetValue === undefined ? widget.name : presetLabel(widget.name, presetValue);
    if (widget.disabled === true && widget.label === label) return false;
    if (!("__relightLabel" in widget)) widget.__relightLabel = widget.label;
    widget.disabled = true;
    widget.label = label;
    return true;
}

function enableWidget(widget) {
    const stale = isPresetLabel(widget);
    if (!widget.disabled && !("__relightLabel" in widget) && !stale) return false;
    widget.disabled = false;
    if ("__relightLabel" in widget) {
        widget.label = widget.__relightLabel;
        delete widget.__relightLabel;
    } else if (stale) {
        // Self-heal: a preset label with no stash behind it means the widget
        // arrived carrying one - a workflow saved mid-edit, an undo that
        // restored the label but not our bookkeeping. Either way it is ours to
        // clear, and leaving it would show a preset's value on a live control.
        widget.label = undefined;
    }
    return true;
}

/**
 * Which widgets the selected preset has taken control of.
 *
 * `effect_strength` is excluded because a preset scales it rather than
 * replacing it, so it stays live - that is the whole point of the exception in
 * `_load_preset`. Geometry is excluded while `preserve_positioning` is on, for
 * the same reason: the preset is not setting it.
 */
export function widgetsToDisable(values) {
    const disabled = new Set();
    const keys = PRESET_KEYS[values.preset];
    if (!keys) return disabled;
    for (const key of keys) {
        if (key === STRENGTH_KEY) continue;
        if (values.preserve_positioning && GEOMETRY.has(key)) continue;
        disabled.add(key);
    }
    return disabled;
}

/**
 * Run `fn` after the current frame.
 *
 * `requestAnimationFrame` is a browser global and absent under `node --test`,
 * where a bare call throws and takes the caller with it.
 */
function deferToNextFrame(fn) {
    if (typeof requestAnimationFrame === "function") requestAnimationFrame(fn);
    else setTimeout(fn, 0);
}

/**
 * Resize the node to exactly the widgets it is currently showing.
 *
 * Height is set, not grown. Growing only would leave the node its full
 * 49-widget height with a large empty panel below the controls the moment
 * anything is hidden, which is most of the time - and the node getting shorter
 * when you drop to one light source is the visible payoff of the whole feature.
 * Width is still only ever grown, because a widened node usually is deliberate.
 */
function fitNode(node) {
    try {
        const [minWidth, minHeight] = node.computeSize();
        node.setSize([Math.max(node.size[0], minWidth), minHeight]);
    } catch (error) {
        warn("could not resize the node after a visibility change", error);
    }
}

/**
 * Apply both treatments to a node. Returns true if anything actually moved, so
 * the caller can skip the resize and the redraw when nothing did.
 *
 * `resize: false` is for the draw loop. Resizing a node from inside its own
 * draw leaves any DOM-backed widget positioned against geometry the in-progress
 * frame already committed; the caller defers the resize to the next frame
 * instead.
 */
export function applyVisibility(node, { resize = true } = {}) {
    const values = valuesOf(node);
    const hidden = widgetsToHide(values);
    const disabled = widgetsToDisable(values);
    let changed = false;

    const presetValues = PRESET_VALUES[values.preset] ?? {};

    for (const widget of node.widgets ?? []) {
        const name = widget?.name;
        if (!name) continue;
        changed = (hidden.has(name) ? hideWidget(widget) : showWidget(widget)) || changed;
        changed = (disabled.has(name)
            ? disableWidget(widget, presetValues[name])
            : enableWidget(widget)) || changed;
    }

    if (changed && resize) fitNode(node);
    return changed;
}

app.registerExtension({
    name: "ReLight.ui",

    async nodeCreated(node) {
        try {
            if (!isReLight(node)) return;

            // Immediate path: the widget's own callback. Not dependable on its
            // own - whether it fires, and whether `value` is current when it
            // does, varies by frontend version and widget type - so it is the
            // fast path, not the only one.
            for (const widget of node.widgets ?? []) {
                const inherited = widget.callback;
                widget.callback = function (...args) {
                    let result;
                    if (typeof inherited === "function") {
                        try {
                            result = inherited.apply(this, args);
                        } catch (error) {
                            warn("an upstream widget callback failed", error);
                        }
                    }
                    try {
                        applyVisibility(node);
                    } catch (error) {
                        warn("could not update widget visibility", error);
                    }
                    return result;
                };
            }

            // Safety net: re-check on the draw loop. This is what catches a
            // workflow load, an undo, and any edit made through the API - none
            // of which run a widget callback.
            const inheritedDraw = node.onDrawForeground;
            node.onDrawForeground = function (...args) {
                try {
                    // Never resize from inside a draw - hence resize: false and
                    // the deferral. In a steady state nothing changed, so this
                    // costs a map lookup per widget per frame and nothing else.
                    if (applyVisibility(this, { resize: false })) {
                        deferToNextFrame(() => fitNode(this));
                    }
                } catch (error) {
                    warn("could not update widget visibility while drawing", error);
                }
                if (typeof inheritedDraw === "function") return inheritedDraw.apply(this, args);
                return undefined;
            };

            applyVisibility(node);
        } catch (error) {
            warn("widget visibility setup failed", error);
        }
    },
});
