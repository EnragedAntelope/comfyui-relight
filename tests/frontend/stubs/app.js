/*
 * Stand-in for ComfyUI's ../../scripts/app.js.
 *
 * A pack's frontend module imports `app` from a path that only resolves inside
 * a running ComfyUI. `hooks.mjs` redirects that specifier here so the real,
 * unmodified web/*.js can be imported by `node --test`.
 *
 * Only the surface ReLight's own modules touch is modelled; everything records
 * what it was given so a test can drive it.
 */

export const registeredExtensions = [];

export const app = {
    registerExtension(extension) {
        registeredExtensions.push(extension);
    },
    graph: {
        findNodesByType: () => [],
        setDirtyCanvas() {},
    },
    canvas: {
        setDirty() {},
    },
    extensionManager: undefined,
};

/** The extension a module registered, by its `name`. */
export function getExtension(name) {
    return registeredExtensions.find((ext) => ext.name === name);
}
