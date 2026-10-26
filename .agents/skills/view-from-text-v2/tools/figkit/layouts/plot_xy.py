from ..parts import T, NFig, chip, takeaway, panel, tint, solid, ICON_COLOR
from . import layout


@layout("plot_xy", "a small x-y chart drawn natively: axes with ticks, lines with markers and value labels, scatter dots, reference lines, an optional side panel of cards")
def build(spec, lang=None):
    """spec: id, alt, plot{x[min,max], y[min,max], x_ticks, y_ticks, x_label, y_label, w, h, x_names?}, series[...], notes[{x, y, text, role?, ha?}],
    side[{title, text, role, name?}], takeaway.  Height 80 + plot h + 60.
    series kinds: {"kind":"line","points":[(x,y)..],"role","width","markers":bool,"labels":[str|None..],"label_dy":int,"name":legend text}
                  {"kind":"dots","points":[(x,y)..],"role"|"roles":[..],"shape":"circle|square","d":int,"name":legend text}
                  {"kind":"hline","y":v,"role","label":str} | {"kind":"vline","x":v,"role","label":str} | {"kind":"seg","a":(x,y),"b":(x,y),"role","width"}"""
    P = spec["plot"]; pw, ph = P.get("w", 700), P.get("h", 300); ox, oy = 90, 40
    H = oy + ph + 96 + (46 if spec.get("takeaway") else 0)
    f = NFig(spec["id"], 1160, H, T(spec["alt"], lang))
    (x0, x1), (y0, y1) = P["x"], P["y"]
    sx = lambda v: ox + (v - x0) / (x1 - x0) * pw; sy = lambda v: oy + ph - (v - y0) / (y1 - y0) * ph
    f.line(ox, oy, ox, oy + ph, "MUT", 2); f.line(ox, oy + ph, ox + pw, oy + ph, "MUT", 2)
    for t in P.get("x_ticks", []):
        f.line(sx(t), oy + ph, sx(t), oy + ph + 7, "MUT", 2)
        nm = (P.get("x_names") or {}).get(str(t))
        f.text(sx(t), oy + ph + 24, T(nm, lang) if nm else f"{t:g}", "MUT", 16)
    for t in P.get("y_ticks", []):
        f.line(ox - 7, sy(t), ox, sy(t), "MUT", 2); f.text(ox - 12, sy(t), f"{t:g}", "MUT", 16, False, "right", w=70)
    if P.get("x_label"): f.text(ox + pw / 2, oy + ph + 52, T(P["x_label"], lang), "TXT", 16, True, w=pw)
    if P.get("y_label"): f.text(ox - 18, oy - 28, T(P["y_label"], lang), "TXT", 16, True, "left", w=300)
    inside = lambda x, y: x0 <= x <= x1 and y0 <= y <= y1
    legend = []
    for s in spec["series"]:
        k = s["kind"]; role = s.get("role", "blue")
        if k == "line":
            pts = [(x, y) for x, y in s["points"]]
            for a, b in zip(pts, pts[1:]): f.line(sx(a[0]), sy(a[1]), sx(b[0]), sy(b[1]), solid(role), s.get("width", 3))
            if s.get("markers"):
                for i, (x, y) in enumerate(pts):
                    f.ellipse(sx(x), sy(y), 14, solid(role))
                    lb = (s.get("labels") or [None] * len(pts))[i]
                    if lb: f.text(sx(x), sy(y) + s.get("label_dy", -20), lb, "TXT", 16, True, w=90)
        elif k == "dots":
            roles = s.get("roles")
            for i, (x, y) in enumerate(s["points"]):
                if not inside(x, y): continue
                c = solid(roles[i] if roles else role)
                (f.ellipse(sx(x), sy(y), s.get("d", 8), c) if s.get("shape", "circle") == "circle" else f.square(sx(x), sy(y), s.get("d", 8), c))
        elif k == "hline":
            f.line(ox, sy(s["y"]), ox + pw, sy(s["y"]), solid(role), s.get("width", 2))
            if s.get("label"): f.text(ox + pw + 8, sy(s["y"]), s["label"], "TXT", 16, False, "left", w=70)
        elif k == "vline":
            f.line(sx(s["x"]), oy, sx(s["x"]), oy + ph, solid(role), s.get("width", 2))
            if s.get("label"): f.text(sx(s["x"]), oy - 10, s["label"], "TXT", 16, True, w=100)
        elif k == "seg":
            f.line(sx(s["a"][0]), sy(s["a"][1]), sx(s["b"][0]), sy(s["b"][1]), solid(role), s.get("width", 3))
        if s.get("name"): legend.append((k, role, T(s["name"], lang), s.get("shape", "circle")))
    for n in spec.get("notes", []):
        f.text(sx(n["x"]), sy(n["y"]), T(n["text"], lang), n.get("color", "TXT"), 16, True, n.get("ha", "center"), w=n.get("w"), math=bool(n.get("math")))
    for i, (k, role, nm, shp) in enumerate(legend):
        lx, ly = ox + pw - 190, oy + 8 + 26 * i
        f.rect(lx, ly + 6, 26, 6, fill=solid(role)) if k == "line" else f.ellipse(lx + 8, ly + 9, 12, solid(role))
        f.text(lx + 34, ly + 9, nm, "TXT", 16, False, "left", w=170)
    sd = spec.get("side", [])
    if sd:
        sx0 = ox + pw + 60; sw = 1160 - sx0; y = oy; hh = (ph + 40 - 12 * (len(sd) - 1)) / len(sd)
        for c in sd:
            panel(f, sx0, y, sw, hh, c["role"])
            f.text(sx0 + 14, y + 22, T(c["title"], lang), "TXT", 18, True, "left", w=sw - 24)
            txt = T(c["text"], lang); mth = "$" in txt; nl = txt.count("\n") + 1; lh = 29 if mth else 18.4
            f.text(sx0 + 14, y + 40 + nl * lh / 2, txt, "TXT", 16, False, "left", w=sw - 24, math=mth, name=T(c["name"], lang) if c.get("name") else None)
            y += hh + 12
    if spec.get("takeaway"):
        takeaway(f, 0, H - 44, 1160, 40, T(spec["takeaway"], lang))
    return f
