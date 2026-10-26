from ..parts import T, NFig, chip, takeaway, tint, solid, INK
from . import layout


@layout("equation_parts", "take a formula apart: each piece in its own coloured box, a name and a plain meaning under it, and the same pieces with numbers substituted")
def build(spec, lang=None):
    """spec: id, alt, pieces[{tex, role?, name, meaning, value?}] (operators: role None and no name), takeaway.  Height 400."""
    f = NFig(spec["id"], 1160, 400, T(spec["alt"], lang))
    ps = spec["pieces"]; boxes = [p for p in ps if p.get("role")]; ops = len(ps) - len(boxes)
    ow = 54; bw = (1160 - ow * ops - 12 * (len(ps) - 1)) / max(len(boxes), 1); x = 0; xs = []
    for p in ps:
        w = bw if p.get("role") else ow; xs.append((x, w)); x += w + 12
    for p, (x, w) in zip(ps, xs):
        if not p.get("role"):
            f.text(x + w / 2, 80, p["tex"], "TXT", 32, True, w=w, math=True)
            if p.get("value"): f.text(x + w / 2, 190, p["value"], "TXT", 28, True, w=w, math=True)
            continue
        f.rect(x, 30, w, 100, fill=tint(p["role"], 0.25), stroke=solid(p["role"]), sw=2, radius=14, text=p["tex"], tcolor="TXT", size=30, bold=True, math=True)
        if p.get("value"):
            f.rect(x, 146, w, 70, fill=tint(p["role"], 0.12), radius=14, text=p["value"], tcolor="TXT", size=26, bold=True, math=True)
        f.text(x + w / 2, 240, T(p["name"], lang), "TXT", 18, True, w=w)
        f.text(x + w / 2, 268, T(p["meaning"], lang), "MUT", 16, w=w)
    f.text(580, 8, T(spec.get("caption", ""), lang), "MUT", 16, w=700)
    takeaway(f, 0, 330, 1160, 44, T(spec["takeaway"], lang))
    return f
