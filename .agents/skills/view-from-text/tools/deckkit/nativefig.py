"""nativefig.py -- draw a figure ONCE as a list of primitives, output it twice:

  * `to_nodes(page, x, y)` -> native JianBauDau nodes (rectangles, ellipses, triangles, thin rotated rectangles for lines, text).
    No embedded image: a figure is ~10-30 small nodes (a few KB) instead of a 50-80 KB base64 SVG, the text stays searchable and editable.
  * `to_png(path)` -> the same figure as a light-theme PNG for the Markdown lecture (matplotlib), so deck and lecture are the same data.

Use it for diagrams made of boxes, bars, dots, markers, straight lines and labels. Continuous curves, big scatter plots and anything
with hundreds of marks stay in matplotlib/SVG (see CLARITY.md C23).

Coordinates are figure pixels (origin top-left, y down), the same unit as a JianBauDau node. Rotation is in DEGREES (the player does
`rotate(<n>deg)`; the file-format spec text that says radians is wrong). Colours: "TXT" and "MUT" follow the theme, "GAP" is the
neutral grey of a remainder, anything else is a hex colour used as is. Labels use plain text (no Markdown); `*`, `_`, `$` are escaped.
"""
import math
import os

THEMES = {
    "dark": {"TXT": "#f2ede4", "MUT": "#b7b0a5", "GAP": "#6b645a", "TRACK": "#3b352c", "BG": "#1c1712", "WHITE": "#ffffff"},
    "light": {"TXT": "#1c1712", "MUT": "#6b645a", "GAP": "#c9c4b8", "TRACK": "#dedbd3", "BG": "#ffffff", "WHITE": "#ffffff"},
}


def _esc(s):
    return s.replace("\\", "\\\\").replace("*", "\\*").replace("_", "\\_").replace("$", "\\$").replace("#", "\\#")


def _text_w(s, size, bold):
    w = 0.0
    for ch in s:
        w += size * (1.0 if ord(ch) > 0x2E80 else (0.62 if bold else 0.56))
    return w


class NFig:
    def __init__(self, fid, w, h, alt=""):
        self.fid, self.w, self.h, self.alt = fid, w, h, alt
        self.items = []

    # ---- primitives
    def rect(self, x, y, w, h, fill=None, stroke=None, sw=0, dash=False, radius=0):
        self.items.append(("rect", dict(x=x, y=y, w=w, h=h, fill=fill, stroke=stroke, sw=sw, dash=dash, radius=radius)))

    def ellipse(self, cx, cy, d, fill):
        self.items.append(("ellipse", dict(cx=cx, cy=cy, d=d, fill=fill)))

    def square(self, cx, cy, d, fill):
        self.items.append(("rect", dict(x=cx - d / 2, y=cy - d / 2, w=d, h=d, fill=fill, stroke=None, sw=0, dash=False, radius=0)))

    def tri(self, cx, cy, d, fill):
        self.items.append(("tri", dict(cx=cx, cy=cy, d=d, fill=fill)))

    def line(self, x1, y1, x2, y2, color, lw=2):
        self.items.append(("line", dict(x1=x1, y1=y1, x2=x2, y2=y2, color=color, lw=lw)))

    def text(self, x, y, s, color="TXT", size=16, bold=False, ha="center"):
        self.items.append(("text", dict(x=x, y=y, s=s, size=size, color=color, bold=bold, ha=ha)))

    # ---- native JianBauDau nodes
    def to_nodes(self, page, x0, y0, theme="dark"):
        """Add the figure to a kit.Page at (x0, y0). Returns the bottom y."""
        col = lambda c: THEMES[theme].get(c, c) if c else "transparent"
        self_prefix = page.nid(f"nf-{self.fid}")
        n = [0]

        def add(typ, x, y, w, h, style, text="", kind=None, rot=0, z=2):
            n[0] += 1
            nid = f"{self_prefix}-{n[0]}"
            node = {"id": nid, "type": typ, "name": f"fig:{self.fid}#{n[0]}", "sceneId": page.u.sid,
                    "transform": {"x": page.ox + x0 + x, "y": y0 + y, "scaleX": 1, "scaleY": 1, "rotation": rot},
                    "width": max(w, 1), "height": max(h, 1), "zIndex": z, "locked": False, "style": style, "text": text}
            if kind: node["shapeKind"] = kind
            page._add(node)
            return nid

        base = lambda fill="transparent", stroke="transparent", **kw: dict({"fill": fill, "stroke": stroke, "textColor": "#ffffff", "fontSize": 16,
                                                                          "radius": 0, "borderWidth": 0}, **kw)
        # invisible region node: carries the figure id and its alt text in the name, so audits and editors can find the figure
        import kit as K
        rid = add("shape", 0, 0, self.w, self.h, base(), "", "rectangle", z=1)
        K.nodes[rid]["name"] = f"fig:{self.fid}|{self.alt}"
        for kind, d in self.items:
            if kind == "rect":
                st = base(col(d["fill"]) if d["fill"] else "transparent", col(d["stroke"]) if d["stroke"] else "transparent",
                          borderWidth=d["sw"], radius=d["radius"], borderStyle="dashed" if d["dash"] else "solid")
                add("shape" if d["stroke"] else "text", d["x"], d["y"], d["w"], d["h"], st, "", "rectangle")
            elif kind == "ellipse":
                add("shape", d["cx"] - d["d"] / 2, d["cy"] - d["d"] / 2, d["d"], d["d"], base(col(d["fill"])), "", "ellipse")
            elif kind == "tri":
                add("shape", d["cx"] - d["d"] / 2, d["cy"] - d["d"] / 2, d["d"], d["d"], base(col(d["fill"])), "", "triangle")
            elif kind == "line":
                L = math.hypot(d["x2"] - d["x1"], d["y2"] - d["y1"])
                cx, cy = (d["x1"] + d["x2"]) / 2, (d["y1"] + d["y2"]) / 2
                deg = math.degrees(math.atan2(d["y2"] - d["y1"], d["x2"] - d["x1"]))
                add("text", cx - L / 2, cy - d["lw"] / 2, L, d["lw"], base(col(d["color"])), "", "rectangle", rot=round(deg, 3))
            elif kind == "text":
                w = _text_w(d["s"], d["size"], d["bold"]) + 12
                h = round(d["size"] * 1.15)
                x = d["x"] - w / 2 if d["ha"] == "center" else (d["x"] if d["ha"] == "left" else d["x"] - w)
                st = base(textColor=col(d["color"]), fontSize=d["size"], fontWeight=700 if d["bold"] else 400, textAlign=d["ha"], lineHeight=1.15)
                add("text", x, d["y"] - h / 2, w, h, st, _esc(d["s"]), "rectangle", z=3)
        return y0 + self.h

    # ---- PNG for the lecture
    def to_png(self, path, theme="light", dpi=150):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, Ellipse, Polygon
        T = THEMES[theme]
        col = lambda c: T.get(c, c)
        fig = plt.figure(figsize=(self.w / 100, self.h / 100), dpi=dpi)
        fig.patch.set_facecolor(T["BG"])
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, self.w); ax.set_ylim(self.h, 0); ax.axis("off")
        for kind, d in self.items:
            if kind == "rect":
                ax.add_patch(Rectangle((d["x"], d["y"]), d["w"], d["h"], fc=col(d["fill"]) if d["fill"] else "none",
                                       ec=col(d["stroke"]) if d["stroke"] else "none", lw=d["sw"] * 0.72, ls=(0, (4, 3)) if d["dash"] else "-"))
            elif kind == "ellipse":
                ax.add_patch(Ellipse((d["cx"], d["cy"]), d["d"], d["d"], fc=col(d["fill"]), ec="none"))
            elif kind == "tri":
                r = d["d"] / 2
                ax.add_patch(Polygon([(d["cx"], d["cy"] - r), (d["cx"] - r, d["cy"] + r), (d["cx"] + r, d["cy"] + r)], fc=col(d["fill"]), ec="none"))
            elif kind == "line":
                ax.plot([d["x1"], d["x2"]], [d["y1"], d["y2"]], color=col(d["color"]), lw=d["lw"] * 0.72, solid_capstyle="butt")
            elif kind == "text":
                ax.text(d["x"], d["y"], d["s"].replace("$", r"\$"), color=col(d["color"]), fontsize=d["size"] * 0.72, ha=d["ha"], va="center",
                        fontweight="bold" if d["bold"] else "normal")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=dpi, facecolor=T["BG"])
        plt.close(fig)
