from ..parts import T, NFig, header_chip, panel, chip, takeaway, tint
from . import layout


@layout("columns_text", "N alternatives compared on the same few attributes (e.g. approach / risk): header chip, one labelled block per attribute, a takeaway")
def build(spec, lang=None):
    """spec: id, alt, columns[{role, header, items[{label, lines[<=3]}], verdict?}], takeaway.  Height 300+."""
    cols = spec["columns"]; n = len(cols); gap = 22; cw = (1160 - gap * (n - 1)) / n
    nit = max(len(c["items"]) for c in cols)      # item j gets the pitch the tallest column needs, so the rows stay aligned
    pitch = [max(100, max(44 + sum(28 if "$" in T(l, lang) else 22 for l in c["items"][j]["lines"]) + 14 for c in cols if j < len(c["items"]))) for j in range(nit)]
    ys = [58 + sum(pitch[:j]) for j in range(nit)]
    ph = 12 + sum(pitch) - 2                       # panel height
    f = NFig(spec["id"], 1160, 46 + ph + 14 + 40, T(spec["alt"], lang))
    for k, c in enumerate(cols):
        x = k * (cw + gap)
        header_chip(f, x, 0, cw, 40, c["role"], T(c["header"], lang))
        panel(f, x, 46, cw, ph, c["role"])
        for j, it in enumerate(c["items"]):
            y = ys[j]
            chip(f, x + 12, y, 90, 26, c["role"], T(it["label"], lang), bold=True, alpha=0.35)
            if it.get("name"):      # one node holding the whole definition, named term:<name> (the term audit looks for it)
                t_ = "\n".join(T(l, lang) for l in it["lines"])
                f.text(x + 14, y + 36 + len(it["lines"]) * 11, t_, "TXT", 16, False, "left", w=cw - 28, math="$" in t_, name=T(it["name"], lang))
            else:
                yy = y + 44
                for ln in it["lines"]:
                    t_ = T(ln, lang); f.text(x + 14, yy, t_, "TXT", 16, False, "left", w=cw - 28, math="$" in t_); yy += 28 if "$" in t_ else 22
    takeaway(f, 0, 46 + ph + 14, 1160, 40, T(spec["takeaway"], lang))
    return f
