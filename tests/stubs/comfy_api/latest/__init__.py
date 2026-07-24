"""Minimal stand-in for ComfyUI's ``comfy_api.latest``.

Lets the ReLight node be imported and executed outside a running ComfyUI, so the
schema and the image pipeline can be tested in CI. Only the surface ReLight
actually touches is modelled; everything records its arguments and gets out of
the way.
"""


class ComfyExtension:
    pass


class _Spec:
    """Records an input/output declaration so tests can assert on the schema."""

    def __init__(self, id=None, **kwargs):
        self.id = id
        self.kwargs = kwargs
        self.default = kwargs.get("default")
        self.optional = kwargs.get("optional", False)
        self.advanced = kwargs.get("advanced", False)
        self.tooltip = kwargs.get("tooltip")

    # ReLight reads `.id`; ComfyUI's real objects also expose `.display_name`.
    @property
    def display_name(self):
        return self.kwargs.get("display_name", self.id)


def _comfy_type(name):
    class _Type:
        Input = _Spec
        Output = _Spec

    _Type.__name__ = name
    return _Type


class io:
    Image = _comfy_type("Image")
    Mask = _comfy_type("Mask")
    Combo = _comfy_type("Combo")
    Int = _comfy_type("Int")
    Float = _comfy_type("Float")
    Boolean = _comfy_type("Boolean")

    class Schema:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.inputs = kwargs.get("inputs", [])
            self.outputs = kwargs.get("outputs", [])

    class ComfyNode:
        pass

    class NodeOutput:
        def __init__(self, *args):
            self.args = args

        def __iter__(self):
            return iter(self.args)

        def __getitem__(self, index):
            return self.args[index]
