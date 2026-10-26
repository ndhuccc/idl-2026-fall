from ..parts import T, tr_deep, NFig, grid, chip, takeaway, terms_strip, tint, solid, INK
from . import layout


@layout("grid_flow", "images, kernels and maps (grids of numbers) joined by operators in one or two rows, with a glossary strip and a takeaway")
def build(spec, lang=None):
    """spec: id, alt, rows[{items[...]}], terms[(name, definition)], takeaway.  Height 440.
    item kinds: {"kind":"grid", rows, cols, cell, cells, role, show, scale, window, caption}  |  {"kind":"op", text, sub?}  |  {"kind":"chip", text(math ok), role, w, h, caption?}
    Items of a row are centred together; every grid caption goes under its grid."""
    f = NFig(spec["id"], 1160, spec.get("height", 440), T(spec["alt"], lang))
    rows = spec["rows"]; top = 4
    for row in rows:
        items = row["items"]; widths = []
        for it in items:
            k = it["kind"]
            widths.append(it["cols"] * it["cell"] if k == "grid" else (it.get("w", 70) if k == "op" else it.get("w", 110)))
        total = sum(widths) + 30 * (len(items) - 1); x = (1160 - total) / 2
        gh = max([it["rows"] * it["cell"] for it in items if it["kind"] == "grid"] or [60])
        for it, w in zip(items, widths):
            k = it["kind"]
            if k == "grid":
                y = top + (gh - it["rows"] * it["cell"]) / 2
                grid(f, x, y, it["rows"], it["cols"], it["cell"], it.get("cells"), it.get("role", "orange"), it.get("show", False), it.get("scale"), it.get("window"))
                if it.get("caption"):
                    f.text(x + w / 2, top + gh + 16, T(it["caption"], lang), "TXT", 16, True, math=bool(it.get("math")))
            elif k == "op":
                f.text(x + w / 2, top + gh / 2 - (10 if it.get("sub") else 0), it["text"], "TXT", 26, True, w=w, math=bool(it.get("math")))
                if it.get("sub"):
                    f.text(x + w / 2, top + gh / 2 + 18, T(it["sub"], lang), "MUT", 16)
            else:
                h = it.get("h", 56)
                f.rect(x, top + (gh - h) / 2, w, h, fill=tint(it["role"], 0.30), radius=12, text=it["text"], tcolor="TXT", size=it.get("size", 24), bold=True, math=True)
                if it.get("caption"):
                    f.text(x + w / 2, top + gh + 16, T(it["caption"], lang), "TXT", 16, True)
            x += w + 30
        top += gh + 44
    ty = max(top, spec.get("terms_y", 0))
    if spec.get("terms"):
        terms_strip(f, ty, [(T(n, lang), T(d, lang)) for n, d in spec["terms"]])
        ty += 90
    takeaway(f, 0, ty, 1160, 38, T(spec["takeaway"], lang))
    return f
