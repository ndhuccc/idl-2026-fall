from ..parts import T, NFig, header_chip, panel, chip, takeaway, tint
from . import layout


@layout("columns_text", "N alternatives compared on the same few attributes (e.g. approach / risk): header chip, one labelled block per attribute, a takeaway")
def build(spec, lang=None):
    """spec: id, alt, columns[{role, header, items[{label, lines[<=3]}], verdict?}], takeaway.  Height 300+."""
    f = NFig(spec["id"], 1160, 330, T(spec["alt"], lang))
    cols = spec["columns"]; n = len(cols); gap = 22; cw = (1160 - gap * (n - 1)) / n
    for k, c in enumerate(cols):
        x = k * (cw + gap)
        header_chip(f, x, 0, cw, 40, c["role"], T(c["header"], lang))
        panel(f, x, 46, cw, 216, c["role"])
        for j, it in enumerate(c["items"]):
            y = 58 + j * 100
            chip(f, x + 12, y, 90, 26, c["role"], T(it["label"], lang), bold=True, alpha=0.35)
            for i, ln in enumerate(it["lines"]):
                f.text(x + 14, y + 44 + i * 22, T(ln, lang), "TXT", 16, False, "left", w=cw - 28)
    takeaway(f, 0, 276, 1160, 40, T(spec["takeaway"], lang))
    return f
