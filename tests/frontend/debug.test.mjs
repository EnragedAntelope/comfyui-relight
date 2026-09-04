/*
 * The debug view has no toggle: connecting the output is the whole gesture.
 *
 * These pin the plumbing that makes that true - `debug_output_connected` must
 * track the wiring, must stay hidden, and must not dirty the graph when nothing
 * changed.
 */
import assert from "node:assert/strict";
import test from "node:test";

import { syncDebugFlag } from "../../web/relight_debug.js";
import { app, getExtension } from "./stubs/app.js";
import { makeNode, schema } from "./fake_node.mjs";

const FLAG = "debug_output_connected";

/** A node with the three real outputs, debug_image last. */
function nodeWithOutputs(debugLinks) {
    const node = makeNode();
    node.outputs = [
        { name: "image", links: [] },
        { name: "mask", links: [] },
        { name: "debug_image", links: debugLinks },
    ];
    return node;
}

function widget(node) {
    return node.widgets.find((w) => w.name === FLAG);
}

test("the schema still carries the flag the UI drives", () => {
    assert.ok(
        schema.widgets.some((w) => w.name === FLAG),
        `${FLAG} is gone from the schema; relight_debug.js has nothing to write`
    );
});

test("a wired debug output sets the flag", () => {
    const node = nodeWithOutputs([12]);
    assert.equal(syncDebugFlag(node), true, "no change reported");
    assert.equal(widget(node).value, true);
});

test("an unwired debug output clears the flag", () => {
    const node = nodeWithOutputs([]);
    widget(node).value = true;
    assert.equal(syncDebugFlag(node), true);
    assert.equal(widget(node).value, false);
});

test("syncing an unchanged node does not report a change", () => {
    // The draw loop and every connection event call this; an unconditional
    // write would mark the workflow modified continuously.
    const node = nodeWithOutputs([]);
    assert.equal(syncDebugFlag(node), false);
    assert.equal(syncDebugFlag(node), false);
});

test("a node with no outputs yet is not an error", () => {
    const node = makeNode();
    delete node.outputs;
    assert.doesNotThrow(() => syncDebugFlag(node));
});

test("nodeCreated hides the flag widget and tracks connections", async () => {
    const extension = getExtension("ReLight.debugView");
    assert.ok(extension, "the debug view extension did not register");

    const node = nodeWithOutputs([]);
    await extension.nodeCreated(node, app);

    const flag = widget(node);
    // Both halves: older LiteGraph skips unknown widget types, newer frontends
    // honour `hidden`. One without the other leaves it painted somewhere.
    assert.equal(flag.hidden, true, "flag widget is not hidden");
    assert.ok(flag.type.startsWith("hidden-"), "flag widget type was not swapped");
    assert.deepEqual(flag.computeSize(), [0, -4], "flag widget still reserves a row");

    node.outputs[2].links = [3];
    node.onConnectionsChange();
    assert.equal(flag.value, true, "connecting the output did not set the flag");

    node.outputs[2].links = [];
    node.onConnectionsChange();
    assert.equal(flag.value, false, "disconnecting the output did not clear the flag");
});

test("setting a node up reads the wiring once the graph has settled", async () => {
    // Connections are restored after nodeCreated when a workflow loads, so the
    // flag has to be read again on the next frame. requestAnimationFrame is a
    // browser global and absent here - a bare call throws and takes the rest of
    // nodeCreated with it, which is what this pins.
    const extension = getExtension("ReLight.debugView");
    const node = nodeWithOutputs([]);
    await extension.nodeCreated(node, app);
    node.outputs[2].links = [5];

    await new Promise((resolve) => setTimeout(resolve, 0));
    assert.equal(widget(node).value, true, "the deferred connection read did not run");
});

test("an upstream onConnectionsChange handler still runs", async () => {
    const extension = getExtension("ReLight.debugView");
    const node = nodeWithOutputs([]);
    let called = 0;
    node.onConnectionsChange = () => {
        called += 1;
        return "upstream";
    };
    await extension.nodeCreated(node, app);

    assert.equal(node.onConnectionsChange(), "upstream", "upstream return value lost");
    assert.equal(called, 1, "upstream handler was not called exactly once");
});

test("a throwing upstream handler does not stop the flag from syncing", async () => {
    const extension = getExtension("ReLight.debugView");
    const node = nodeWithOutputs([7]);
    node.onConnectionsChange = () => {
        throw new Error("some other pack");
    };
    await extension.nodeCreated(node, app);

    assert.doesNotThrow(() => node.onConnectionsChange());
    assert.equal(widget(node).value, true);
});

test("nodes from other packs are left alone", async () => {
    const extension = getExtension("ReLight.debugView");
    const node = nodeWithOutputs([1]);
    node.comfyClass = "SomeoneElsesNode";
    node.type = "SomeoneElsesNode";
    const before = node.onConnectionsChange;
    await extension.nodeCreated(node, app);
    assert.equal(node.onConnectionsChange, before);
    assert.equal(widget(node).value, false);
});
