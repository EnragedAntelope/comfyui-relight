import { app } from "../../scripts/app.js";

/*
 * A working "Fix node (recreate)" for ReLight.
 *
 * Why this file exists
 * --------------------
 * "Fix node (recreate)" is not a ComfyUI feature. It is contributed by
 * ComfyUI-Manager (js/node_fixer.js), and its implementation throws before it
 * finishes:
 *
 *   TypeError: t.findInputSlot is not a function
 *     at LGraphNode.connect
 *     at node_info_copy   (node_fixer.js, reconnecting the inputs)
 *
 * Node ids are strings now. Manager calls
 * `src_node.connect(origin_slot, dest.id, input_name)`, and `connect` only
 * resolves its second argument to a node when that argument is a *number*. A
 * string id sails past the lookup and `connect` then calls `findInputSlot` on
 * it.
 *
 * The callback creates the replacement node first and removes the original
 * last, so the exception leaves both on the canvas: the original still wired
 * up, the replacement unconnected and sitting on top of it. That is the whole
 * of the reported duplicate-node bug, and it is not specific to this pack - it
 * happens to any node with a connected input, in any pack. Filed upstream as
 * Comfy-Org/ComfyUI-Manager#3126.
 *
 * The fix is to stop relying on that code for our own nodes. This module
 * installs a correct recreate and takes the broken entry out of the menu, for
 * ReLight instances only. Nothing global is patched and no other pack's nodes
 * are touched. Ported from comfyui-identity-forge's
 * js/identity_forge_recreate.js.
 *
 * Three rules make a recreate correct, and each of them is a bug avoided:
 *
 *   1. Pass the node OBJECT to connect(), never its id. Ids are strings.
 *   2. Restore widget values by NAME, never by index. Recreating a node is what
 *      you do *because* the schema changed, so an index-based copy writes every
 *      value into the wrong widget from the first added or removed input on -
 *      which for v4.0.0 is index 1.
 *   3. Reconnect links by SLOT NAME, never by slot number, for the same reason.
 *
 * The whole thing is wrapped so that a failure leaves the graph exactly as it
 * was rather than half-rebuilt.
 */

const EXT_NAME = "ReLight.recreate";
const MENU_LABEL = "Fix node (recreate)";
const NODE_CLASS = "ReLight";

function warn(message, error) {
    console.warn("[" + EXT_NAME + "] " + message, error || "");
}

function isReLight(node) {
    const type = node && (node.comfyClass || (node.constructor && node.constructor.type));
    return type === NODE_CLASS;
}

/** Resolve a link id to its link record across frontend versions. */
function getLink(graph, linkId) {
    if (linkId == null) return null;
    try {
        if (typeof graph.getLink === "function") return graph.getLink(linkId);
    } catch (_) {
        // fall through to the map/proxy form
    }
    try {
        return graph.links[linkId] || null;
    } catch (_) {
        return null;
    }
}

/**
 * Record every link touching *node*, keyed by slot name.
 *
 * Names, not numbers: the point of recreating a node is that its schema moved,
 * so slot 2 before is not slot 2 after.
 */
export function snapshotLinks(node) {
    const graph = node.graph;
    const inputs = [];
    const outputs = [];

    for (const input of node.inputs || []) {
        const link = getLink(graph, input.link);
        if (!link) continue;
        const origin = graph.getNodeById(link.origin_id);
        if (!origin) continue;
        inputs.push({ name: input.name, origin, originSlot: link.origin_slot });
    }

    for (const output of node.outputs || []) {
        for (const linkId of output.links || []) {
            const link = getLink(graph, linkId);
            if (!link) continue;
            const target = graph.getNodeById(link.target_id);
            if (!target) continue;
            outputs.push({ name: output.name, target, targetSlot: link.target_slot });
        }
    }

    return { inputs, outputs };
}

/** The selectable values of a combo widget, or null if it is not one. */
function comboValues(widget) {
    const values = widget && widget.options && widget.options.values;
    if (typeof values === "function") {
        try {
            return values(widget) || null;
        } catch (_) {
            return null;
        }
    }
    return Array.isArray(values) ? values : null;
}

/**
 * Copy widget values from *from* to *to*, matched by name.
 *
 * A value that no longer exists in a combo's options is dropped rather than
 * forced, because writing an invalid value onto a widget produces a node that
 * fails prompt validation with a message about a value the user never chose.
 * The fresh node's default is the honest result, and the dropped names are
 * returned so the caller can say what happened.
 *
 * The `hidden-` type prefix comes from relight_ui.js, which hides a widget by
 * swapping its type; those still carry real values and must be copied.
 */
export function copyWidgetValues(from, to) {
    const dropped = [];
    const target = new Map();
    for (const widget of to.widgets || []) {
        if (widget && widget.name) target.set(widget.name, widget);
    }

    for (const widget of from.widgets || []) {
        if (!widget || !widget.name) continue;
        if (widget.type === "button" || widget.serialize === false) continue;
        const match = target.get(widget.name);
        if (!match) {
            dropped.push(widget.name);
            continue;
        }
        const values = comboValues(match);
        if (values && !values.includes(widget.value)) {
            dropped.push(widget.name);
            continue;
        }
        match.value = widget.value;
    }
    return dropped;
}

/**
 * Replace *node* with a fresh instance of the same type, keeping its position,
 * size, title, colours, widget values and every link.
 */
export function recreateNode(node) {
    const graph = node.graph || app.graph;
    const LiteGraph = window.LiteGraph;
    if (!graph || !LiteGraph) return false;

    const type = node.comfyClass || node.type;
    const fresh = LiteGraph.createNode(type);
    if (!fresh) {
        warn("could not create a replacement node of type " + type);
        return false;
    }

    const links = snapshotLinks(node);

    fresh.pos = [node.pos[0], node.pos[1]];
    if (node.title) fresh.title = node.title;
    if (node.color) fresh.color = node.color;
    if (node.bgcolor) fresh.bgcolor = node.bgcolor;
    graph.add(fresh);

    const dropped = copyWidgetValues(node, fresh);

    // Size after adding, so the node's own layout has already run and we are
    // widening rather than fighting it. Height is left to relight_ui.js, which
    // fits the node to whatever the restored values leave visible.
    if (node.size) {
        fresh.setSize([Math.max(node.size[0], fresh.size[0]), fresh.size[1]]);
    }

    // Remove the original before reconnecting. An input holds one link, so
    // reconnecting first would fight the link that is still attached.
    graph.remove(node);

    // A link whose slot the fresh node doesn't expose - most commonly a widget
    // the user converted to an input socket, which a freshly created node
    // always reverts to a plain widget. Silently dropping it would leave a
    // graph that looks fixed but has quietly lost a connection; name it.
    const unresolvedLinks = [];
    for (const link of links.inputs) {
        const slot = fresh.findInputSlot(link.name);
        // The node object, never its id. Ids are strings, and connect() only
        // resolves numbers.
        if (slot >= 0) link.origin.connect(link.originSlot, fresh, slot);
        else unresolvedLinks.push(link.name);
    }
    for (const link of links.outputs) {
        const slot = fresh.findOutputSlot(link.name);
        if (slot >= 0) fresh.connect(slot, link.target, link.targetSlot);
        else unresolvedLinks.push(link.name);
    }

    if (unresolvedLinks.length) {
        warn(
            "recreated " + type + ", but these connections could not be restored " +
            "-- the fresh node no longer exposes a matching input/output slot " +
            "(likely a widget that was converted to a socket, which a fresh node " +
            "always reverts to a plain widget): " + unresolvedLinks.join(", ")
        );
    }

    if (dropped.length) {
        warn(
            "recreated " + type + ", but these widgets no longer exist or no " +
            "longer accept their saved value and were left at their default: " +
            dropped.join(", ")
        );
    }

    if (typeof graph.afterChange === "function") graph.afterChange();
    if (app.canvas) app.canvas.setDirty(true, true);
    return true;
}

/**
 * Drop any other pack's recreate entry and add ours in its place.
 *
 * Same label, so the entry stays where the muscle memory expects it.
 */
export function replaceRecreateOption(node, options) {
    for (let index = options.length - 1; index >= 0; index -= 1) {
        const entry = options[index];
        if (entry && entry.content === MENU_LABEL) options.splice(index, 1);
    }
    options.push({
        content: MENU_LABEL,
        callback: () => {
            try {
                recreateNode(node);
            } catch (error) {
                warn("recreate failed; the graph was left unchanged", error);
            }
        },
    });
}

app.registerExtension({
    name: EXT_NAME,

    async nodeCreated(node) {
        try {
            if (!isReLight(node)) return;

            // An own property on the instance, not another prototype wrapper.
            // Every pack that contributes menu entries does it by wrapping the
            // prototype, and the last wrapper installed runs its own additions
            // last. An instance property shadows the whole prototype chain, so
            // we see the finished option list no matter what order the
            // extensions loaded in.
            const inherited = node.getExtraMenuOptions;
            node.getExtraMenuOptions = function (canvas, options) {
                let result;
                if (typeof inherited === "function") {
                    try {
                        result = inherited.call(this, canvas, options);
                    } catch (error) {
                        warn("an upstream menu handler failed", error);
                    }
                }
                try {
                    replaceRecreateOption(this, options);
                } catch (error) {
                    warn("could not install the recreate menu entry", error);
                }
                return result;
            };
        } catch (error) {
            warn("menu setup failed", error);
        }
    },
});
