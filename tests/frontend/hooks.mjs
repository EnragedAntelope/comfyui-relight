/*
 * Redirect ComfyUI's frontend module specifiers to the local stubs.
 *
 * `web/relight_*.js` import "../../scripts/app.js", which resolves to nothing
 * outside a running ComfyUI. `module.registerHooks` (not the deprecated
 * `module.register`, which runs in a separate loader worker and warns) lets a
 * --import'd file rewrite the specifier before resolution, so `node --test` can
 * import the real, byte-for-byte unmodified pack files.
 *
 * Matched by SUFFIX, not by exact path, so it keeps working for files at
 * different relative depths.
 *
 *   node --import ./tests/frontend/hooks.mjs --test tests/frontend
 */
import module from "node:module";
import { pathToFileURL } from "node:url";

const STUBS = new URL("./stubs/", import.meta.url);

const REDIRECTS = [
    ["/scripts/app.js", new URL("app.js", STUBS).href],
];

module.registerHooks({
    resolve(specifier, context, nextResolve) {
        for (const [suffix, target] of REDIRECTS) {
            if (specifier.endsWith(suffix)) {
                return { url: target, shortCircuit: true };
            }
        }
        return nextResolve(specifier, context);
    },
});

export { pathToFileURL };
