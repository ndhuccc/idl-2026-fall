"""Layout registry. A layout is a function build(spec, lang=None) -> NFig, registered with @layout("name").

Add your own: copy `_template.py`, change the geometry, register it, add a line to ../../../LAYOUTS.md, and (if the layout is reusable)
an example under ../../../examples/. `build(name, spec)` is what a deck script calls; `freeform` lets a page draw an NFig by hand with
the parts in ../parts.py when no layout fits."""
import importlib
import os
import pkgutil

REGISTRY = {}


def layout(name, summary=""):
    def deco(fn):
        fn.layout_name, fn.summary = name, summary
        REGISTRY[name] = fn
        return fn
    return deco


def build(name, spec, lang=None):
    _load()
    if name not in REGISTRY:
        raise KeyError(f"unknown layout {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name](spec, lang)


def _load():
    pkg = os.path.dirname(os.path.abspath(__file__))
    for m in pkgutil.iter_modules([pkg]):
        if not m.name.startswith("_"):
            importlib.import_module(f"{__package__}.{m.name}")


def names():
    _load()
    return {k: v.summary for k, v in sorted(REGISTRY.items())}
