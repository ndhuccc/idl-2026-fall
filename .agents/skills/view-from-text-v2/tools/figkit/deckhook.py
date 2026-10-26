"""deckhook.py -- glue between a layout spec and a deck page (JianBauDau nodes) or a lecture PNG. One spec, two outputs.

    from figkit import deckhook
    bottom = deckhook.place(page, y, "columns_compare", spec)        # in a deck script: page is a kit.Page; native nodes, dark theme
    deckhook.png("columns_compare", spec, "assets/figures/x.png")    # lecture: light theme PNG from the same spec
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from figkit import layouts  # noqa: E402


def place(page, y, layout, spec, x=60, theme="dark"):
    return layouts.build(layout, spec).to_nodes(page, x, y, theme)


def png(layout, spec, path, lang=None, theme="light"):
    layouts.build(layout, spec, lang).to_png(path, theme)
