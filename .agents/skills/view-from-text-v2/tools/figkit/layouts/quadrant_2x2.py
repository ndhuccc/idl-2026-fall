from ..parts import T, NFig, header_chip, panel, chip, takeaway, tint, solid, INK
from . import layout


@layout("quadrant_2x2", "four cases from two yes/no (or low/high) questions: axes with labels, four coloured cells with a title, a line and an optional number")
def build(spec, lang=None):
    """spec: id, alt, x_label, y_label, x_vals[2], y_vals[2], cells[4] (row-major: top-left, top-right, bottom-left, bottom-right){role, title, text, value?}, takeaway.  Height 400."""
    f = NFig(spec["id"], 1160, 400, T(spec["alt"], lang))
    x0, y0, cw, chh = 230, 56, 440, 120
    f.text(x0 + cw, 18, T(spec["x_label"], lang), "TXT", 18, True, w=600)
    for j in range(2):
        f.text(x0 + cw * j + cw / 2, 42, T(spec["x_vals"][j], lang), "MUT", 16, True, w=cw)
    f.text(70, y0 + chh, T(spec["y_label"], lang), "TXT", 18, True, w=120)
    for i in range(2):
        f.text(214, y0 + chh * i + chh / 2, T(spec["y_vals"][i], lang), "MUT", 16, True, "right", w=150)
    for k, c in enumerate(spec["cells"]):
        i, j = divmod(k, 2); x, y = x0 + cw * j + 4, y0 + chh * i + 4
        panel(f, x, y, cw - 8, chh - 8, c["role"])
        f.text(x + 18, y + 24, T(c["title"], lang), "TXT", 20, True, "left", w=cw - 60)
        f.text(x + 18, y + 62, T(c["text"], lang), "TXT", 16, False, "left", w=cw - 60, math=True)
        if c.get("value"):
            f.rect(x + cw - 150, y + chh - 54, 124, 36, fill=tint(c["role"], 0.35), radius=10, text=c["value"], tcolor="TXT", size=20, bold=True, math=True)
    takeaway(f, 0, 316, 1160, 44, T(spec["takeaway"], lang))
    return f
