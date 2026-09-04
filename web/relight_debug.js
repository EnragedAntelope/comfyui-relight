import { app } from "../../scripts/app.js";

/*
 * ReLight debug view: no toggle, just the wire.
 *
 * The rule from the user's point of view is the whole feature: connect
 * `debug_image` to a preview and you get the debug visualization; don't, and
 * you don't. There is nothing to remember, nothing to switch on, and no way to
 * have the output wired and a placeholder on screen.
 *
 * Making that true needs one piece of plumbing, because of how ComfyUI decides
 * a node needs re-running. A node's cache key is built from its INPUTS. Wiring
 * something to an OUTPUT does not change any input, so the graph would happily
 * replay ReLight's cached result - the placeholder - into the preview you just
 * connected, and you would have to hit Run twice for no visible reason.
 *
 * So the node carries a `debug_output_connected` boolean input that this module
 * owns end to end: it hides the widget (nobody should see it), and it writes
 * true/false into it whenever the debug output's connections change. That write
 * is a widget-value change, which is exactly what the cache watches, so the
 * node re-runs on the same queue that added the wire.
 *
 * The Python side has a second, independent check on the submitted prompt, so
 * an API caller that never loads this file still gets the debug view when it
 * consumes the output. This module is what makes it feel immediate in the UI.
 */

const NODE_CLASS = "ReLight";
const FLAG_WIDGET = "debug_output_connected";
const DEBUG_OUTPUT = "debug_image";

function warn(message, error) {
    console.warn("[ReLight] " + message, error || "");
}

function isReLight(node) {
    const type = node && (node.comfyClass || node.type);
    return type === NODE_CLASS;
}

function flagWidget(node) {
    return (node.widgets ?? []).find((widget) => widget.name === FLAG_WIDGET);
}

/**
 * Hide a widget from the node body without removing it.
 *
 * Both halves are needed: older LiteGraph skips widgets whose `type` it does
 * not know, newer frontends honour `hidden`. Setting only one leaves the widget
 * painted on some versions. `computeSize` is stubbed to zero so the node does
 * not reserve a row for it.
 */
function hideWidget(node, widget) {
    widget.hidden = true;
    if (!widget.type?.startsWith?.("hidden-")) {
        widget.type = "hidden-" + (widget.type ?? "toggle");
    }
    widget.computeSize = () => [0, -4];
}

/**
 * Run `fn` once the current frame has settled.
 *
 * `requestAnimationFrame` is a browser global, so it is absent under
 * `node --test` - and a bare call there throws a ReferenceError that takes the
 * rest of `nodeCreated` with it. Fall back to a macrotask.
 */
function deferToNextFrame(fn) {
    if (typeof requestAnimationFrame === "function") requestAnimationFrame(fn);
    else setTimeout(fn, 0);
}

/** Does the debug output have at least one link? */
function debugOutputIsConnected(node) {
    const output = (node.outputs ?? []).find((slot) => slot.name === DEBUG_OUTPUT);
    return Boolean(output?.links?.length);
}

/**
 * Push the current wiring into the flag widget.
 *
 * Writes only on an actual change: an unconditional write on every draw would
 * dirty the graph continuously and mark the workflow modified for no reason.
 */
export function syncDebugFlag(node) {
    const widget = flagWidget(node);
    if (!widget) return false;
    const connected = debugOutputIsConnected(node);
    if (widget.value === connected) return false;
    widget.value = connected;
    return true;
}

app.registerExtension({
    name: "ReLight.debugView",

    async nodeCreated(node) {
        try {
            if (!isReLight(node)) return;

            const widget = flagWidget(node);
            if (widget) hideWidget(node, widget);

            // An own property on the instance rather than another prototype
            // wrapper, so we see the finished state no matter which extension
            // loaded last.
            const inherited = node.onConnectionsChange;
            node.onConnectionsChange = function (...args) {
                let result;
                if (typeof inherited === "function") {
                    try {
                        result = inherited.apply(this, args);
                    } catch (error) {
                        warn("an upstream onConnectionsChange handler failed", error);
                    }
                }
                try {
                    syncDebugFlag(this);
                } catch (error) {
                    warn("could not track the debug output connection", error);
                }
                return result;
            };

            // Connections are restored after nodeCreated when a workflow loads,
            // and onConnectionsChange is not always called for those, so take
            // one reading once the graph has settled.
            deferToNextFrame(() => {
                try {
                    syncDebugFlag(node);
                } catch (error) {
                    warn("could not read the debug output connection on load", error);
                }
            });
        } catch (error) {
            warn("debug view setup failed", error);
        }
    },
});
