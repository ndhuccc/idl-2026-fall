from ..parts import T, NFig, chip, takeaway, tint, solid
from . import layout


@layout("bars_compare", "compare a few values (accuracy, error, cost) as horizontal bars with the number at the end, optional ± whisker and a note per bar")
def build(spec, lang=None):
    """spec: id, alt, bars[{label, value, err?, role, note?}] (2..7), max, ticks[...]?, unit?, takeaway.  Height 70 + 54*n + 60."""
    bars = spec["bars"]; n = len(bars); H = 70 + 54 * n + 62
    f = NFig(spec["id"], 1160, H, T(spec["alt"], lang))
    x0, x1 = 300, 760; mx = spec["max"]; mn = spec.get("min", 0); sx = lambda v: x0 + (v - mn) / (mx - mn) * (x1 - x0)
    if spec.get("unit"):
        f.text(x0, 14, T(spec["unit"], lang), "MUT", 16, False, "left", w=500)
    f.line(x0, 34, x0, 34 + 54 * n + 8, "MUT", 2); f.line(x0, 34 + 54 * n + 8, x1, 34 + 54 * n + 8, "MUT", 2)
    for t in spec.get("ticks", []):
        f.line(sx(t), 34 + 54 * n + 8, sx(t), 34 + 54 * n + 16, "MUT", 2); f.text(sx(t), 34 + 54 * n + 30, f"{t:g}", "MUT", 16, w=80)
    for i, b in enumerate(bars):
        y = 44 + 54 * i
        f.text(x0 - 14, y + 20, T(b["label"], lang), "TXT", 18, True, "right", w=270)
        f.rect(x0, y, max(sx(b["value"]) - x0, 2), 40, fill=solid(b["role"]), radius=6)
        f.text(sx(b["value"] + b.get("err", 0)) + 12, y + 20, f"{b['value']:g}" + (f" ± {b['err']:g}" if b.get("err") else "") + T(b.get("suffix", ""), lang), "TXT", 18, True, "left", w=110)
        if b.get("err"):
            f.line(sx(b["value"] - b["err"]), y + 20, sx(b["value"] + b["err"]), y + 20, "#1c1712", 3)
        if b.get("note"):
            f.text(1150, y + 20, T(b["note"], lang), "MUT", 16, False, "right", w=260)
    takeaway(f, 0, H - 44, 1160, 40, T(spec["takeaway"], lang))
    return f
