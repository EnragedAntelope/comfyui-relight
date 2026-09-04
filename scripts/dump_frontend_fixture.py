"""Dump the live node schema to a fixture the frontend tests build fake nodes from.

The JS tests need the current widget names, order and defaults. Hand-listing
them would be a second copy that drifts silently, so they are derived from
``define_schema()`` and written here; ``--check`` fails if the committed file no
longer matches, and ``tests/test_relight.py`` runs that check.

    python scripts/dump_frontend_fixture.py           # rewrite the fixture
    python scripts/dump_frontend_fixture.py --check   # fail if it is stale
"""
import argparse
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "tests" / "frontend" / "fixtures" / "schema.json"

# The stub must win over any real ComfyUI on the path, exactly as in conftest.
sys.path.insert(0, str(REPO_ROOT / "tests" / "stubs"))
sys.path.insert(1, str(REPO_ROOT))

from relight import ReLight  # noqa: E402


def build():
    schema = ReLight.define_schema()
    widgets = []
    for spec in schema.inputs:
        if spec.id in ("image", "mask"):
            continue  # link inputs, never widgets
        entry = {"name": spec.id, "default": spec.default}
        options = spec.kwargs.get("options")
        if options is not None:
            entry["options"] = list(options)
        widgets.append(entry)
    return {"node_class": schema.node_id, "widgets": widgets}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the fixture is stale")
    args = parser.parse_args(argv)

    payload = json.dumps(build(), indent=2) + "\n"
    if args.check:
        current = FIXTURE.read_text(encoding="utf-8") if FIXTURE.exists() else ""
        if current != payload:
            print("tests/frontend/fixtures/schema.json is stale; re-run this script")
            return 1
        return 0
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(payload, encoding="utf-8")
    print(f"wrote {FIXTURE.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
