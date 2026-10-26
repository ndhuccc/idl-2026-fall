"""Copy this file to start a new layout. Rules (LAYOUTS.md): width 1160, height <= 480 (title and bridge sit above); every label >= 16 px;
dark text on bright fills (header_chip/chip(dark=True)), light text on tints; flat fills only (no shadows or gradients); all strings go
through T(); all numbers come from the spec (the spec is computed in code by the caller); give the NFig an alt text that lists every
number and the conclusion; keep one idea per layout."""
from ..parts import T, NFig, header_chip, panel, takeaway
from . import layout


@layout("template", "describe in one line what reader task this layout serves")
def build(spec, lang=None):
    f = NFig(spec["id"], 1160, 300, T(spec["alt"], lang))
    header_chip(f, 0, 0, 560, 34, "blue", T(spec["header"], lang))
    takeaway(f, 0, 250, 1160, 40, T(spec["takeaway"], lang))
    return f
