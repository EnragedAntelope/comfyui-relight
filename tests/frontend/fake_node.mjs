/*
 * A LiteGraph-shaped fake node, built from the real schema fixture.
 *
 * Deliberately minimal and deliberately not helpful: it models only what the
 * pack's own frontend code touches, and does not synthesise conveniences the
 * real frontend does not provide (see the `serialize` note in
 * `comfyui-identity-forge`'s harness). Being more generous than the real thing
 * is how a harness hides the bug it was written to catch.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
export const REPO_ROOT = path.resolve(HERE, "..", "..");

export const schema = JSON.parse(
    fs.readFileSync(path.join(HERE, "fixtures", "schema.json"), "utf8")
);

/** `nodeData` in the shape `beforeRegisterNodeDef` receives it. */
export function makeNodeData() {
    const required = {};
    for (const widget of schema.widgets) {
        const options = { default: widget.default };
        required[widget.name] = widget.options
            ? [widget.options, options]
            : ["FLOAT", options];
    }
    return { name: schema.node_class, input: { required } };
}

/** A node instance carrying one widget per schema entry, at its default. */
export function makeNode(overrides = {}) {
    const widgets = schema.widgets.map((widget) => ({
        name: widget.name,
        value: Object.prototype.hasOwnProperty.call(overrides, widget.name)
            ? overrides[widget.name]
            : widget.default,
        options: widget.options ? { values: [...widget.options] } : {},
        callback: undefined,
    }));
    return {
        comfyClass: schema.node_class,
        type: schema.node_class,
        widgets,
        size: [317, 1250],
        // computeSize is what growToFitWidgets measures against. 24px a row is
        // close enough to LiteGraph's real spacing for a "did it grow" test.
        computeSize() {
            return [210, 30 + this.widgets.length * 24];
        },
        setSize(size) {
            this.size = size;
        },
        widgetValue(name) {
            return this.widgets.find((w) => w.name === name)?.value;
        },
    };
}

/**
 * Drive a `beforeRegisterNodeDef` extension the way ComfyUI does.
 *
 * The hook wraps methods on `nodeType.prototype`, so it needs a real class with
 * a real prototype - a plain object will not do.
 */
export async function driveBeforeRegisterNodeDef(extension, nodeData) {
    class FakeNodeType {}
    await extension.beforeRegisterNodeDef(FakeNodeType, nodeData);
    return FakeNodeType;
}

/** Load a workflow JSON from the repo. */
export function loadWorkflow(...segments) {
    return JSON.parse(fs.readFileSync(path.join(REPO_ROOT, ...segments), "utf8"));
}

/** The single ReLight node in a workflow graph. */
export function relightNodeOf(graph) {
    const nodes = graph.nodes.filter((node) => node.type === "ReLight");
    if (nodes.length !== 1) {
        throw new Error(`expected exactly one ReLight node, found ${nodes.length}`);
    }
    return nodes[0];
}
