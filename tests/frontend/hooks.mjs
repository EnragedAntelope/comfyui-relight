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
 *   node --import ./tests/frontend/hooks.mjs --test "tests/frontend/*.test.mjs"
 *
 * A glob, not the directory. On Node 24 `--test tests/frontend` puts the
 * directory itself through the ESM resolver, and the hook below hands it to
 * `nextResolve`, which throws ERR_UNSUPPORTED_DIR_IMPORT and fails the whole
 * run before a single test loads. Newer Node happens not to, so this passes
 * locally and fails in CI. The glob resolves to files and works on both.
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
