import os
import sys

import pytest
import torch

# The stub `comfy_api` must win over anything a real ComfyUI checkout might put
# on the path, so it goes first.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "stubs"))
sys.path.insert(1, os.path.dirname(os.path.dirname(__file__)))

from relight import ReLight  # noqa: E402


@pytest.fixture
def node():
    return ReLight


@pytest.fixture
def defaults():
    """Every widget input at its schema default, keyed by input id.

    Derived from ``define_schema()`` rather than hardcoded, so adding or
    renaming an input never silently desynchronises the tests.
    """

    def _defaults(**overrides):
        params = {
            spec.id: spec.default
            for spec in ReLight.define_schema().inputs
            if spec.id not in ("image", "mask")
        }
        params.update(overrides)
        return params

    return _defaults


@pytest.fixture
def run(defaults):
    """Execute the node with schema defaults plus overrides."""

    def _run(image, mask=None, **overrides):
        params = defaults(**overrides)
        if mask is not None:
            params["mask"] = mask
        return tuple(ReLight.execute(image, **params))

    return _run


@pytest.fixture
def image():
    torch.manual_seed(0)
    return torch.rand(1, 64, 96, 3)
