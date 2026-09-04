/*
 * The legacy-workflow migration, driven against the real v3.1.2 save.
 *
 * This is the load-bearing piece of v4.0.0: if it is wrong, every workflow in
 * the wild loads plausible-looking garbage with no error. A synthetic 48-value
 * array could be made to satisfy a wrong migration, so the input here is the
 * workflow ReLight actually shipped at v3.1.2, kept verbatim in
 * tests/fixtures/.
 */
import assert from "node:assert/strict";
import test from "node:test";

import {
    LEGACY_ORDER,
    looksLegacy,
    migrateLegacyWidgetValues,
    collectDefaults,
    growToFitWidgets,
} from "../../web/relight_migrate.js";
import { getExtension } from "./stubs/app.js";
import {
    driveBeforeRegisterNodeDef,
    loadWorkflow,
    makeNode,
    makeNodeData,
    relightNodeOf,
    schema,
} from "./fake_node.mjs";

const legacyGraph = loadWorkflow("tests", "fixtures", "legacy_v3.1.2_workflow.json");
const legacyValues = relightNodeOf(legacyGraph).widgets_values;
const nodeData = makeNodeData();
const defaults = collectDefaults(nodeData);

/** The legacy array as a name -> value map, for asserting against. */
const legacy = Object.fromEntries(LEGACY_ORDER.map((name, i) => [name, legacyValues[i]]));

test("defaults are read out of the real /object_info shape", () => {
    // The server sends a combo as ["COMBO", {options, default}]. Reading the
    // default out of the wrong element would silently reset every new widget
    // to undefined during a migration.
    assert.equal(defaults.lighting_mode, "Color Correction");
    assert.equal(defaults.subject_interaction, "None");
    assert.equal(defaults.debug_output_connected, false);
    assert.equal(typeof defaults.shadow_strength, "number");
});

test("the shipped v3.1.2 workflow is recognised as legacy", () => {
    assert.equal(legacyValues.length, 48);
    assert.equal(looksLegacy(legacyValues), true);
});

test("a current save is not mistaken for a legacy one", () => {
    const current = schema.widgets.map((w) => w.default);
    assert.equal(looksLegacy(current), false);
});

test("a 48-long array that is not a legacy save is left alone", () => {
    // Same length, wrong sentinels: index 0 is not a preset name.
    const impostor = new Array(48).fill(0);
    assert.equal(looksLegacy(impostor), false);
});

test("every legacy value lands on the widget of the same name", () => {
    const node = makeNode();
    assert.equal(migrateLegacyWidgetValues(node, legacyValues, defaults), true);

    const renamed = new Set([
        "show_debug_info", // dropped outright
        "use_colored_lights",
        "use_gradient_mode",
        "apply_3d_lighting",
        "light_direction",
    ]);
    for (const name of LEGACY_ORDER) {
        if (renamed.has(name)) continue;
        assert.equal(
            node.widgetValue(name),
            legacy[name],
            `${name} did not survive the migration`
        );
    }
});

test("the four booleans become the right combo values", () => {
    // The shipped save has apply_3d_lighting on, "Behind Subject",
    // use_colored_lights off, use_gradient_mode off and show_debug_info on.
    assert.equal(legacy.apply_3d_lighting, true);
    assert.equal(legacy.light_direction, "Behind Subject");
    assert.equal(legacy.use_colored_lights, false);

    const node = makeNode();
    migrateLegacyWidgetValues(node, legacyValues, defaults);

    assert.equal(node.widgetValue("subject_interaction"), "Light behind subject (rim)");
    assert.equal(node.widgetValue("lighting_mode"), "Color Correction");
    assert.equal(node.widgetValue("mask_shape"), "Radial falloff");
});

test("show_debug_info is dropped, not carried into a new widget", () => {
    // v4 has no debug toggle: the debug view follows the wiring of the
    // debug_image output. A legacy true must not resurrect as a control.
    for (const wasOn of [true, false]) {
        const values = [...legacyValues];
        values[LEGACY_ORDER.indexOf("show_debug_info")] = wasOn;
        const node = makeNode();
        migrateLegacyWidgetValues(node, values, defaults);
        assert.equal(node.widgets.some((w) => w.name === "debug_view"), false);
        assert.equal(
            node.widgetValue("debug_output_connected"),
            defaults.debug_output_connected,
            "the connectivity flag is the UI's to set, not the migration's"
        );
    }
});

test("apply_3d_lighting off wins over whatever light_direction said", () => {
    const values = [...legacyValues];
    values[LEGACY_ORDER.indexOf("apply_3d_lighting")] = false;
    // light_direction stays "Behind Subject" - the master switch overrode it.
    const node = makeNode();
    migrateLegacyWidgetValues(node, values, defaults);
    assert.equal(node.widgetValue("subject_interaction"), "None");
});

test("each legacy light_direction maps to its own interaction", () => {
    const cases = [
        ["Behind Subject", "Light behind subject (rim)"],
        ["In Front of Subject", "Light in front of subject"],
        ["No Occlusion", "None"],
    ];
    for (const [direction, expected] of cases) {
        const values = [...legacyValues];
        values[LEGACY_ORDER.indexOf("light_direction")] = direction;
        const node = makeNode();
        migrateLegacyWidgetValues(node, values, defaults);
        assert.equal(node.widgetValue("subject_interaction"), expected, direction);
    }
});

test("colored lights migrate to Colored Light, not Both", () => {
    // "Both" would change the render of every colour workflow, because the
    // grading block was inert whenever use_colored_lights was on.
    const values = [...legacyValues];
    values[LEGACY_ORDER.indexOf("use_colored_lights")] = true;
    const node = makeNode();
    migrateLegacyWidgetValues(node, values, defaults);
    assert.equal(node.widgetValue("lighting_mode"), "Colored Light");
});

test("widgets new in v4 get their schema default, not a shifted legacy value", () => {
    const node = makeNode({ shadow_strength: 999, shadow_length: 999 });
    migrateLegacyWidgetValues(node, legacyValues, defaults);
    assert.equal(node.widgetValue("shadow_strength"), defaults.shadow_strength);
    assert.equal(node.widgetValue("shadow_length"), defaults.shadow_length);
    assert.notEqual(defaults.shadow_strength, undefined);
});

test("a node saved too short is grown to fit its widgets", () => {
    const node = makeNode();
    node.size = [317, 100];
    growToFitWidgets(node);
    assert.ok(node.size[1] >= node.computeSize()[1], "node was not grown");
});

test("a deliberately widened node is not shrunk", () => {
    const node = makeNode();
    node.size = [900, 4000];
    growToFitWidgets(node);
    assert.deepEqual(node.size, [900, 4000]);
});

test("onConfigure runs the migration when a workflow is loaded", async () => {
    const extension = getExtension("ReLight.migrate");
    assert.ok(extension, "the migration extension did not register");

    const NodeType = await driveBeforeRegisterNodeDef(extension, nodeData);
    const node = makeNode();
    NodeType.prototype.onConfigure.call(node, { widgets_values: legacyValues });

    assert.equal(node.widgetValue("preset"), legacy.preset);
    assert.equal(node.widgetValue("subject_interaction"), "Light behind subject (rim)");
    assert.equal(node.widgetValue("light_position_x"), legacy.light_position_x);
});

test("onConfigure leaves a current workflow untouched", async () => {
    const extension = getExtension("ReLight.migrate");
    const NodeType = await driveBeforeRegisterNodeDef(extension, nodeData);

    const currentGraph = loadWorkflow("example_workflows", "relight_basic.json");
    const currentValues = relightNodeOf(currentGraph).widgets_values;
    assert.equal(currentValues.length, schema.widgets.length);

    const node = makeNode();
    schema.widgets.forEach((widget, index) => {
        node.widgets[index].value = currentValues[index];
    });
    NodeType.prototype.onConfigure.call(node, { widgets_values: currentValues });

    schema.widgets.forEach((widget, index) => {
        assert.equal(node.widgetValue(widget.name), currentValues[index], widget.name);
    });
});
