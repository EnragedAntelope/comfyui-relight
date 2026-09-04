/*
 * Conditional widget visibility.
 *
 * Every rule here is a control that looked live and was not: Light 2/3 painted
 * with one light source, the grading block painted in colour mode, a preset
 * silently overriding a dozen fully draggable values.
 */
import assert from "node:assert/strict";
import test from "node:test";

import {
    applyVisibility,
    widgetsToDisable,
    widgetsToHide,
} from "../../web/relight_ui.js";
import { PRESET_KEYS } from "../../web/relight_presets.js";
import { app, getExtension } from "./stubs/app.js";
import { makeNode, schema } from "./fake_node.mjs";

const DEFAULTS = Object.fromEntries(schema.widgets.map((w) => [w.name, w.default]));

function values(overrides) {
    return { ...DEFAULTS, ...overrides };
}

function widget(node, name) {
    return node.widgets.find((w) => w.name === name);
}

test("light 2 and 3 are hidden with a single light source", () => {
    const hidden = widgetsToHide(values({ num_light_sources: 1 }));
    assert.ok(hidden.has("light2_position_x"));
    assert.ok(hidden.has("light3_intensity"));
});

test("light 2 appears at two sources, light 3 only at three", () => {
    const two = widgetsToHide(values({ num_light_sources: 2, lighting_mode: "Colored Light" }));
    assert.equal(two.has("light2_position_x"), false);
    assert.ok(two.has("light3_position_x"));

    const three = widgetsToHide(values({ num_light_sources: 3, lighting_mode: "Colored Light" }));
    assert.equal(three.has("light3_position_x"), false);
});

test("colour controls are hidden in Color Correction", () => {
    const hidden = widgetsToHide(values({ lighting_mode: "Color Correction" }));
    for (const name of ["light_color_r", "light_color_g", "light_color_b", "light_intensity"]) {
        assert.ok(hidden.has(name), name);
    }
    assert.equal(hidden.has("inner_brightness"), false);
});

test("the grading block is hidden in Colored Light", () => {
    const hidden = widgetsToHide(values({ lighting_mode: "Colored Light" }));
    for (const name of ["inner_brightness", "inner_gamma", "outer_saturation", "outer_gamma"]) {
        assert.ok(hidden.has(name), name);
    }
    assert.equal(hidden.has("light_color_r"), false);
});

test("Both shows colour and grading together", () => {
    const hidden = widgetsToHide(values({ lighting_mode: "Both" }));
    assert.equal(hidden.has("light_color_r"), false);
    assert.equal(hidden.has("inner_brightness"), false);
});

test("rim and shadow controls only appear for a light behind the subject", () => {
    for (const interaction of ["None", "Light in front of subject"]) {
        const hidden = widgetsToHide(values({ subject_interaction: interaction }));
        for (const name of ["rim_amplification", "shadow_strength", "shadow_length"]) {
            assert.ok(hidden.has(name), `${interaction}: ${name}`);
        }
    }
    const rim = widgetsToHide(values({ subject_interaction: "Light behind subject (rim)" }));
    for (const name of ["rim_amplification", "shadow_strength", "shadow_length"]) {
        assert.equal(rim.has(name), false, name);
    }
});

test("the connectivity flag is never shown", () => {
    assert.ok(widgetsToHide(values({})).has("debug_output_connected"));
});

test("a preset greys the widgets it overrides, and only those", () => {
    const disabled = widgetsToDisable(values({ preset: "Spotlight" }));
    const expected = PRESET_KEYS["Spotlight"].filter((k) => k !== "effect_strength");
    assert.deepEqual([...disabled].sort(), [...expected].sort());
});

test("effect_strength stays live under a preset", () => {
    // A preset scales it rather than replacing it - that is the documented
    // exception in _load_preset, and greying it would contradict the node.
    for (const name of Object.keys(PRESET_KEYS)) {
        assert.equal(
            widgetsToDisable(values({ preset: name })).has("effect_strength"),
            false,
            name
        );
    }
});

test("no preset is selected, nothing is greyed", () => {
    assert.equal(widgetsToDisable(values({ preset: "None" })).size, 0);
});

test("preserve_positioning hands the geometry back", () => {
    const off = widgetsToDisable(values({ preset: "Spotlight", preserve_positioning: false }));
    const on = widgetsToDisable(values({ preset: "Spotlight", preserve_positioning: true }));
    assert.ok(off.has("light_position_x"));
    assert.equal(on.has("light_position_x"), false);
    // Non-geometry overrides are still the preset's.
    assert.ok(on.has("inner_brightness"));
});

test("hiding and showing a widget round-trips", () => {
    // The classic failure: computeSize is restored by reassigning a saved
    // `undefined`, which leaves the zero-size stub and the widget never returns.
    const node = makeNode({ num_light_sources: 1 });
    const originalType = widget(node, "light2_position_x").type;
    applyVisibility(node);
    const light2 = widget(node, "light2_position_x");
    assert.equal(light2.hidden, true);
    assert.ok(light2.type.startsWith("hidden-"));

    widget(node, "num_light_sources").value = 2;
    applyVisibility(node);
    assert.equal(light2.hidden, false);
    assert.equal(light2.type, originalType, "the original widget type was not restored");
    assert.equal(
        Object.prototype.hasOwnProperty.call(light2, "computeSize"),
        false,
        "computeSize stub was reassigned instead of deleted"
    );
});

test("the node shrinks when a block is hidden and grows when it comes back", () => {
    const node = makeNode({ num_light_sources: 3, lighting_mode: "Both" });
    applyVisibility(node);
    const tall = node.size[1];

    widget(node, "num_light_sources").value = 1;
    applyVisibility(node);
    assert.ok(node.size[1] < tall, "the node kept its full height with two blocks hidden");

    widget(node, "num_light_sources").value = 3;
    applyVisibility(node);
    assert.equal(node.size[1], tall, "the node did not grow back");
});

test("a deliberately widened node keeps its width", () => {
    const node = makeNode();
    node.size = [900, 100];
    applyVisibility(node);
    assert.equal(node.size[0], 900);
});

test("applying visibility twice reports no second change", () => {
    // The draw loop calls this every frame; a steady state must be a no-op or
    // the canvas is marked dirty forever.
    const node = makeNode();
    applyVisibility(node);
    assert.equal(applyVisibility(node), false);
});

test("nodeCreated wires both the callback and the draw loop", async () => {
    const extension = getExtension("ReLight.ui");
    assert.ok(extension, "the ui extension did not register");

    const node = makeNode({ num_light_sources: 1 });
    await extension.nodeCreated(node, app);

    assert.equal(widget(node, "light2_position_x").hidden, true);

    // The immediate path: changing a value through the widget's callback.
    const count = widget(node, "num_light_sources");
    count.value = 2;
    count.callback(2);
    assert.equal(widget(node, "light2_position_x").hidden, false);

    // The safety net: a value changed with no callback at all, as a workflow
    // load or an undo does it.
    count.value = 1;
    node.onDrawForeground();
    assert.equal(widget(node, "light2_position_x").hidden, true);
});

test("an upstream widget callback still runs", async () => {
    const extension = getExtension("ReLight.ui");
    const node = makeNode();
    let called = 0;
    widget(node, "num_light_sources").callback = () => {
        called += 1;
        return "upstream";
    };
    await extension.nodeCreated(node, app);
    assert.equal(widget(node, "num_light_sources").callback(2), "upstream");
    assert.equal(called, 1);
});

test("nodes from other packs are left alone", async () => {
    const extension = getExtension("ReLight.ui");
    const node = makeNode();
    node.comfyClass = "SomeoneElsesNode";
    node.type = "SomeoneElsesNode";
    const before = node.onDrawForeground;
    await extension.nodeCreated(node, app);
    assert.equal(node.onDrawForeground, before);
});
