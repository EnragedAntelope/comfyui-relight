/*
 * "Fix node (recreate)".
 *
 * ComfyUI-Manager's version passes a STRING node id to `connect()`, which only
 * resolves numbers, so it throws after adding the replacement and before
 * removing the original - leaving two nodes on the canvas. These pin the three
 * rules that make ours correct: the node object rather than its id, widget
 * values by name, links by slot name.
 */
import assert from "node:assert/strict";
import test from "node:test";

import { app, getExtension } from "./stubs/app.js";
import { makeNode, schema } from "./fake_node.mjs";

const MENU_LABEL = "Fix node (recreate)";

/** A minimal LiteGraph-shaped graph holding nodes and links. */
function makeGraph() {
    const graph = {
        nodes: [],
        links: {},
        nextLink: 1,
        add(node) {
            node.graph = graph;
            node.id = node.id ?? String(graph.nodes.length + 100);
            graph.nodes.push(node);
        },
        remove(node) {
            graph.nodes = graph.nodes.filter((n) => n !== node);
        },
        getNodeById(id) {
            return graph.nodes.find((n) => String(n.id) === String(id)) ?? null;
        },
        getLink(id) {
            return graph.links[id] ?? null;
        },
    };
    return graph;
}

/** A plain node with named input/output slots. */
function makePlainNode(id, inputs, outputs) {
    const node = {
        id,
        type: "Plain" + id,
        widgets: [],
        pos: [0, 0],
        size: [200, 60],
        inputs: inputs.map((name) => ({ name, link: null })),
        outputs: outputs.map((name) => ({ name, links: [] })),
        findInputSlot(name) {
            return this.inputs.findIndex((s) => s.name === name);
        },
        findOutputSlot(name) {
            return this.outputs.findIndex((s) => s.name === name);
        },
        connect(originSlot, target, targetSlot) {
            // The real LiteGraph resolves its second argument to a node ONLY
            // when it is a number. A string id sails past and blows up here -
            // which is the upstream bug. Model that faithfully.
            if (typeof target !== "object" || target === null) {
                throw new TypeError("target.findInputSlot is not a function");
            }
            const id = this.graph.nextLink++;
            this.graph.links[id] = {
                origin_id: this.id,
                origin_slot: originSlot,
                target_id: target.id,
                target_slot: targetSlot,
            };
            this.outputs[originSlot].links.push(id);
            target.inputs[targetSlot].link = id;
            return this.graph.links[id];
        },
    };
    return node;
}

/** A ReLight node with real slots, wired into a graph. */
function makeRelight(graph, overrides = {}) {
    const node = makeNode(overrides);
    node.pos = [40, 60];
    node.inputs = [
        { name: "image", link: null },
        { name: "mask", link: null },
    ];
    node.outputs = [
        { name: "image", links: [] },
        { name: "mask", links: [] },
        { name: "debug_image", links: [] },
    ];
    node.findInputSlot = makePlainNode("x", [], []).findInputSlot;
    node.findOutputSlot = makePlainNode("x", [], []).findOutputSlot;
    node.connect = makePlainNode("x", [], []).connect;
    graph.add(node);
    return node;
}

function withLiteGraph(graph, factory, fn) {
    const previous = globalThis.window;
    globalThis.window = {
        LiteGraph: {
            createNode(type) {
                assert.equal(type, schema.node_class);
                return factory();
            },
        },
    };
    const previousGraph = app.graph;
    app.graph = graph;
    try {
        return fn();
    } finally {
        globalThis.window = previous;
        app.graph = previousGraph;
    }
}

const { recreateNode, copyWidgetValues, replaceRecreateOption } = await import(
    "../../web/relight_recreate.js"
);

test("a recreate replaces the node instead of duplicating it", () => {
    const graph = makeGraph();
    const source = makePlainNode("1", [], ["IMAGE"]);
    graph.add(source);
    const relight = makeRelight(graph, { preset: "Spotlight" });
    source.connect(0, relight, 0);

    const before = graph.nodes.length;
    const ok = withLiteGraph(graph, () => makeRelight(makeGraph()), () => recreateNode(relight));

    assert.equal(ok, true);
    assert.equal(graph.nodes.length, before, "the original node was left on the canvas");
    assert.equal(graph.nodes.includes(relight), false, "the original node was not removed");
});

test("input links are restored, by slot name", () => {
    const graph = makeGraph();
    const source = makePlainNode("1", [], ["IMAGE", "MASK"]);
    graph.add(source);
    const relight = makeRelight(graph);
    source.connect(0, relight, 0);
    source.connect(1, relight, 1);

    let fresh;
    withLiteGraph(graph, () => (fresh = makeRelight(makeGraph())), () => recreateNode(relight));

    assert.ok(fresh.inputs[0].link, "image input was not reconnected");
    assert.ok(fresh.inputs[1].link, "mask input was not reconnected");
});

test("output links are restored, by slot name", () => {
    const graph = makeGraph();
    const relight = makeRelight(graph);
    const preview = makePlainNode("9", ["images"], []);
    graph.add(preview);
    relight.connect(2, preview, 0); // debug_image -> preview

    let fresh;
    withLiteGraph(graph, () => (fresh = makeRelight(makeGraph())), () => recreateNode(relight));

    assert.equal(fresh.outputs[2].links.length, 1, "the debug_image link was lost");
    assert.ok(preview.inputs[0].link);
});

test("connect is given the node object, never its id", () => {
    // The fake `connect` throws on a non-object exactly as LiteGraph does. If
    // this pack ever regressed to passing an id, this test is the crash.
    const graph = makeGraph();
    const source = makePlainNode("1", [], ["IMAGE"]);
    graph.add(source);
    const relight = makeRelight(graph);
    source.connect(0, relight, 0);

    assert.doesNotThrow(() => {
        withLiteGraph(graph, () => makeRelight(makeGraph()), () => recreateNode(relight));
    });
});

test("widget values are copied by name, not by index", () => {
    // The reason to recreate a node is that its schema moved, so an
    // index-based copy writes every value into the wrong widget.
    const from = makeNode({ preset: "Spotlight", light_position_x: 0.12, num_light_sources: 3 });
    const to = makeNode();
    to.widgets.reverse();

    const dropped = copyWidgetValues(from, to);

    assert.deepEqual(dropped, []);
    assert.equal(to.widgetValue("preset"), "Spotlight");
    assert.equal(to.widgetValue("light_position_x"), 0.12);
    assert.equal(to.widgetValue("num_light_sources"), 3);
});

test("a combo value the fresh node no longer offers is dropped, not forced", () => {
    const from = makeNode({ preset: "A Preset That Was Removed" });
    const to = makeNode();
    const dropped = copyWidgetValues(from, to);
    assert.deepEqual(dropped, ["preset"]);
    assert.equal(to.widgetValue("preset"), "None", "an invalid value was written onto the widget");
});

test("hidden widgets still carry their values across", () => {
    // relight_ui.js hides by swapping widget.type; those widgets hold real
    // values and skipping them would silently reset half the node.
    const from = makeNode({ light2_position_x: 0.77 });
    from.widgets.find((w) => w.name === "light2_position_x").type = "hidden-number";
    const to = makeNode();
    copyWidgetValues(from, to);
    assert.equal(to.widgetValue("light2_position_x"), 0.77);
});

test("the menu entry replaces another pack's broken one", () => {
    const node = makeNode();
    const options = [
        { content: "Title" },
        { content: MENU_LABEL, callback: () => "manager" },
        { content: "Remove" },
    ];
    replaceRecreateOption(node, options);

    const entries = options.filter((entry) => entry?.content === MENU_LABEL);
    assert.equal(entries.length, 1, "two recreate entries in the menu");
    assert.equal(entries[0].callback.toString().includes("manager"), false);
});

test("nodeCreated installs the entry on ReLight nodes only", async () => {
    const extension = getExtension("ReLight.recreate");
    assert.ok(extension, "the recreate extension did not register");

    const relight = makeNode();
    await extension.nodeCreated(relight, app);
    const options = [];
    relight.getExtraMenuOptions(app.canvas, options);
    assert.equal(options.filter((o) => o.content === MENU_LABEL).length, 1);

    const other = makeNode();
    other.comfyClass = "SomeoneElsesNode";
    other.type = "SomeoneElsesNode";
    await extension.nodeCreated(other, app);
    assert.equal(other.getExtraMenuOptions, undefined);
});

test("an upstream menu handler still contributes its entries", async () => {
    const extension = getExtension("ReLight.recreate");
    const node = makeNode();
    node.getExtraMenuOptions = function (canvas, options) {
        options.push({ content: "Someone else's entry" });
        return "upstream";
    };
    await extension.nodeCreated(node, app);

    const options = [];
    assert.equal(node.getExtraMenuOptions(app.canvas, options), "upstream");
    assert.ok(options.some((o) => o.content === "Someone else's entry"));
    assert.ok(options.some((o) => o.content === MENU_LABEL));
});
