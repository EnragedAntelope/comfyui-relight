"""ComfyUI ReLight - v3 node pack entry point."""

if __package__:
    from .relight import comfy_entrypoint
else:
    # Imported as a top-level module rather than a package (pytest's collector
    # does this because the repo root is itself a package).
    from relight import comfy_entrypoint

__all__ = ["comfy_entrypoint"]
