from ..parts import T, tr_deep, NFig, header_chip, panel, chip, outline_box, takeaway, tint, solid, INK, SKETCHES
from . import layout


@layout("before_after", "two contrasted states (e.g. small vs large, with vs without), each with a sketch, headline numbers and a gap; plus info boxes and a takeaway")
def build(spec, lang=None):
    """spec: id, alt, panels[2]{role, header, sketch[(name, kwargs)...], caption, stats[2]{label, value, sub, role}, gap},
    boxes[{kind 'text'|'pair', x, w, title, lines, name?  |  left, right, arrow, caption}], takeaway, footnote. Height 478."""
    f = NFig(spec["id"], 1160, 478, T(spec["alt"], lang))
    for x0, p in zip((0, 600), spec["panels"]):
        role = p["role"]
        header_chip(f, x0, 0, 560, 34, role, T(p["header"], lang))
        panel(f, x0, 38, 560, 286, role)
        for name, kw in p.get("sketch", []):
            SKETCHES[name](f, **tr_deep(kw, lang))
        chip(f, x0 + 30, 192, 500, 26, "grey", T(p["caption"], lang), bold=True, alpha=0.10)
        for off, s in enumerate(p["stats"]):
            xx = x0 + 10 + off * 276
            f.rect(xx, 226, 266, 58, fill=tint(s["role"], 0.30), radius=12)
            f.text(xx + 133, 246, T(s["label"], lang) + T(s["value"], lang), "TXT", 26, True, w=262)
            f.text(xx + 133, 272, T(s["sub"], lang), "TXT", 16, w=262)
        chip(f, x0 + 10, 290, 542, 28, "grey", T(p["gap"], lang), size=18, bold=True, alpha=0.10)
    for b in spec["boxes"]:
        x, w = b["x"], b["w"]
        f.rect(x, 330, w, 86, fill=tint("blue", 0.12), stroke=tint("blue", 0.5), sw=1.5, radius=12)
        if b.get("kind", "text") == "text":
            chip(f, x + 10, 336, b.get("title_w", 190), 28, "blue", T(b["title"], lang), size=18, bold=True, dark=True)
            ls = [T(l, lang) for l in b["lines"]]
            f.text(x + 14, (398 if b.get("math") else 392) if len(ls) > 1 else 396, "\n".join(ls), "TXT", 16, False, "left", w=w - 20, math=bool(b.get("math")), name=T(b["name"], lang) if b.get("name") else None)
        else:
            chip(f, x + 12, 340, 100, 40, "green", T(b["left"], lang), size=18, bold=True)
            f.text(x + w / 2, 360, "→", "TXT", 26, True)
            chip(f, x + w - 112, 340, 100, 40, "red", T(b["right"], lang), size=18, bold=True)
            f.text(x + w / 2, 400, T(b["caption"], lang), "TXT", 16, w=w - 14)
    takeaway(f, 0, 422, 1160, 36, T(spec["takeaway"], lang))
    f.text(580, 470, T(spec["footnote"], lang), "MUT", 16, w=1100)
    return f
