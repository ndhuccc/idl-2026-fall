from ..parts import T, NFig, chip, takeaway, tint, solid
from . import layout


@layout("network_layers", "a layered network: columns of units joined by lines (weights), a label and a one-line formula under each layer")
def build(spec, lang=None):
    """spec: id, alt, layers[{label, n (<=6), role, formula?, dots?}] (2..5), highlight[(layer, unit)...]?, takeaway.  Height 400."""
    f = NFig(spec["id"], 1160, 350, T(spec["alt"], lang))
    ls = spec["layers"]; m = len(ls); gx = 1160 / m; cx = lambda i: gx * (i + 0.5)
    ys = lambda L, k: 56 + (190 - 38 * (L["n"] - 1)) / 2 + 38 * k + 20
    hi = set(map(tuple, spec.get("highlight", [])))
    for i in range(m - 1):
        for a in range(ls[i]["n"]):
            for b in range(ls[i + 1]["n"]):
                on = (i, a) in hi and (i + 1, b) in hi
                f.line(cx(i), ys(ls[i], a), cx(i + 1), ys(ls[i + 1], b), solid("yellow") if on else "GAP", 3 if on else 1)
    for i, L in enumerate(ls):
        f.text(cx(i), 22, T(L["label"], lang), "TXT", 20, True, w=gx - 20)
        for k in range(L["n"]):
            f.ellipse(cx(i), ys(L, k), 30, solid(L["role"]))
        if L.get("dots"):
            f.text(cx(i), ys(L, L["n"] - 1) + 34, "⋮", "MUT", 24, True)
        if L.get("formula"):
            f.text(cx(i), 262, L["formula"], "TXT", 18, w=gx - 20, math=True)
    takeaway(f, 0, 300, 1160, 42, T(spec["takeaway"], lang))
    return f
