"""ComfyUI ReLight - v3 node pack entry point."""

if __package__:
    from .relight import comfy_entrypoint
else:
    # Imported as a top-level module rather than a package (pytest's collector
    # does this because the repo root is itself a package).
    from relight import comfy_entrypoint

#: Frontend JavaScript: the legacy-workflow migration, the conditional widget
#: visibility, and a working "Fix node (recreate)". Served by ComfyUI from this
#: directory; without it a pre-v4 workflow loads its widget values positionally
#: into a schema that has moved.
WEB_DIRECTORY = "./web"

__all__ = ["comfy_entrypoint", "WEB_DIRECTORY"]
