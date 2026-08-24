import os
import sys

import pytest
import torch

# The stub `comfy_api` must win over anything a real ComfyUI checkout might put
# on the path, so it goes first.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "stubs"))
sys.path.insert(1, os.path.dirname(os.path.dirname(__file__)))

from relight import ReLight  # noqa: E402


def _shallow_clone(cls):
    """Copy of ComfyUI's ``comfy_api.internal.shallow_clone_class``."""
    return type(f"{cls.__name__}Clone", (cls,) + cls.__bases__, dict(cls.__dict__))


def _lock(cls):
    """Copy of ComfyUI's ``comfy_api.internal.lock_class``.

    Kept byte-for-byte in behaviour with the real thing: any write to a class
    attribute, or to an attribute of an instance, raises AttributeError.
    """

    def locked_instance_setattr(self, name, value):
        raise AttributeError(
            f"Cannot set attribute '{name}' on immutable instance of {type(self).__name__}"
        )

    class LockedMeta(type(cls)):
        def __setattr__(cls_, name, value):
            raise AttributeError(
                f"Cannot modify class attribute '{name}' on locked class '{cls_.__name__}'"
            )

    locked_dict = dict(cls.__dict__)
    locked_dict["__setattr__"] = locked_instance_setattr
    return LockedMeta(cls.__name__, cls.__bases__, locked_dict)


@pytest.fixture
def node():
    """The node exactly as ComfyUI hands it to ``execute``: a locked clone.

    ComfyUI never calls the class you define - it calls a locked shallow clone
    whose metaclass rejects every class-attribute write. Testing against the
    bare class hid a crash on the first mask of every single run (3.0.0/3.1.0),
    because the tests could write class state the real runtime forbids.
    """
    return _lock(_shallow_clone(ReLight))


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
def run(node, defaults):
    """Execute the node with schema defaults plus overrides."""

    def _run(image, mask=None, **overrides):
        params = defaults(**overrides)
        if mask is not None:
            params["mask"] = mask
        return tuple(node.execute(image, **params))

    return _run


@pytest.fixture
def image():
    torch.manual_seed(0)
    return torch.rand(1, 64, 96, 3)
