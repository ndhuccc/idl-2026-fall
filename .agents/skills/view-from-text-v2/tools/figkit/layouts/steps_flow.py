from ..parts import T, NFig, header_chip, panel, chip, takeaway, tint, solid, INK
from . import layout


@layout("steps_flow", "an ordered procedure or causal chain: N boxes joined by arrows, each with a title, one example line and a result chip; then a takeaway")
def build(spec, lang=None):
    """spec: id, alt, steps[{role, title, lines[<=3 text or latex], result?}] (2..5 steps), takeaway. Height 300."""
    f = NFig(spec["id"], 1160, 300, T(spec["alt"], lang))
    st = spec["steps"]; n = len(st); gap = 44; w = (1160 - gap * (n - 1)) / n
    for i, s in enumerate(st):
        x = i * (w + gap)
        header_chip(f, x, 0, w, 36, s["role"], T(s["title"], lang), size=18)
        panel(f, x, 40, w, 190, s["role"])
        for j, ln in enumerate(s["lines"]):
            m = isinstance(ln, str) and ln.startswith("$")
            f.text(x + w / 2, 70 + 34 * j, T(ln, lang), "TXT", 18, w=w - 16, math=m)
        if s.get("result"):
            f.rect(x + 20, 176, w - 40, 44, fill=tint(s["role"], 0.30), radius=12, text=s["result"], tcolor="TXT", size=22, bold=True, math=True)
        if i < n - 1:
            f.text(x + w + gap / 2, 135, "→", "TXT", 30, True)
    takeaway(f, 0, 246, 1160, 44, T(spec["takeaway"], lang))
    return f
