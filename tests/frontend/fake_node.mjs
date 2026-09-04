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

/**
 * `nodeData` in the shape `beforeRegisterNodeDef` receives it.
 *
 * Verified against what this ComfyUI build actually serves at /object_info: a
 * combo arrives as `["COMBO", {options: [...], default: ...}]`, NOT as the
 * older `[[...options], {...}]`. A fake that emitted the old shape would let a
 * `collectDefaults` that only understood the old shape pass here and fail live.
 */
export function makeNodeData() {
    const required = {};
    for (const widget of schema.widgets) {
        required[widget.name] = widget.options
            ? ["COMBO", { default: widget.default, options: [...widget.options], multiselect: false }]
            : [typeof widget.default === "boolean" ? "BOOLEAN" : "FLOAT", { default: widget.default }];
    }
    return { name: schema.node_class, input: { required } };
}

/** LiteGraph's widget `type`, which the real frontend always sets. */
function widgetType(widget) {
    if (widget.options) return "combo";
    if (typeof widget.default === "boolean") return "toggle";
    return "number";
}

/** Row height LiteGraph reserves for one widget. */
const WIDGET_ROW = 24;

/** A node instance carrying one widget per schema entry, at its default. */
export function makeNode(overrides = {}) {
    const widgets = schema.widgets.map((widget) => ({
        name: widget.name,
        type: widgetType(widget),
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
        // LiteGraph sums each widget's own computeSize when it defines one and
        // reserves a fixed row otherwise, which is what makes a hidden widget's
        // zero-size stub shrink the node. A fake that just counted widgets
        // would report a node that never changes height.
        computeSize() {
            const rows = this.widgets.reduce((total, widget) => {
                if (typeof widget.computeSize === "function") return total + widget.computeSize()[1];
                return total + WIDGET_ROW;
            }, 0);
            return [210, 30 + Math.max(0, rows)];
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
