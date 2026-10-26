from ..parts import T, NFig, header_chip, panel, chip, outline_box, icon_rows, takeaway, legend, tint, solid, INK
from . import layout


@layout("two_options", "one starting case, two candidate options drawn side by side with their parts, then result rows (e.g. right way vs naive way) with verdict chips")
def build(spec, lang=None):
    """spec: id, alt, parent{rows[(kind,n)], lines[text, text(math ok)]}, legend, options[2]{role, header, children[{rows, label}], comment},
    rows[{role, label[2 lines], formulas[2 latex], chip{text, role}}], takeaway. Height 466."""
    f = NFig(spec["id"], 1160, 466, T(spec["alt"], lang))
    outline_box(f, 300, 0, 560, 68, sw=2, radius=12)
    icon_rows(f, 318, 17, spec["parent"]["rows"])
    f.line(488, 10, 488, 58, "MUT", 1.5)
    f.text(506, 22, T(spec["parent"]["lines"][0], lang), "TXT", 18, True, "left")
    f.text(506, 48, T(spec["parent"]["lines"][1], lang), "TXT", 18, w=300, ha="left", math=True)
    legend(f, 920, 18, [(k, T(t, lang)) for k, t in spec["legend"]], vertical=True)
    for x0, opt in zip((0, 600), spec["options"]):
        role = opt["role"]
        header_chip(f, x0, 78, 560, 36, role, T(opt["header"], lang))
        panel(f, x0, 118, 560, 164, role)
        kids = opt["children"]
        ws = [max(max(n for _, n in k["rows"]) * 30 + 20, 60) for k in kids]
        gap = 60; bx = x0 + (560 - (sum(ws) + gap)) / 2
        for k, w in zip(kids, ws):
            outline_box(f, bx, 142, w, 80, "TXT")
            icon_rows(f, bx + (w - max(n for _, n in k["rows"]) * 30) / 2, 160, k["rows"])
            f.line(x0 + 280, 122, bx + w / 2, 142, "MUT", 2)
            f.text(bx + w / 2, 238, T(k["label"], lang), "TXT", 16, True, w=min(w + 40, 150))
            bx += w + gap
        chip(f, x0 + 60, 252, 440, 26, role, T(opt["comment"], lang))
    for i, r in enumerate(spec["rows"]):
        y0 = 290 + 66 * i
        f.rect(0, y0, 1160, 60, fill=tint(r["role"], 0.14), radius=12)
        f.text(14, y0 + 19, T(r["label"][0], lang), "TXT", 18, True, "left", w=176); f.text(14, y0 + 43, T(r["label"][1], lang), "TXT", 18, True, "left", w=176)
        f.text(398, y0 + 30, r["formulas"][0], "TXT", 18, w=420, math=True); f.text(832, y0 + 30, r["formulas"][1], "TXT", 18, w=450, math=True)
        chip(f, 1064, y0 + 12, 92, 36, r["chip"]["role"], T(r["chip"]["text"], lang), size=18, dark=True)
    takeaway(f, 0, 424, 1160, 42, T(spec["takeaway"], lang))
    return f
