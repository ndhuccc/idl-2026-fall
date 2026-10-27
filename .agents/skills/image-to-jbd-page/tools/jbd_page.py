#!/usr/bin/env python3
"""jbd_page.py -- build ONE JianBauDau page out of native nodes, with every coordinate given in SOURCE-IMAGE pixels.

Typical use (see ../examples/example_page.py):
    from jbd_page import Page
    P = Page(src_w=1600, src_h=900)            # size of the picture you are reproducing; scale to 1280x720 is automatic
    P.rect(40, 125, 740, 590, fill="rgba(239,107,107,0.07)", stroke="#ef6b6b", sw=2, radius=16)
    P.text(70, 140, 500, 40, "BEFORE", size=28, color="#ef6b6b", bold=True)
    P.line([(100, 300), (160, 300), (160, 400)], "#6aa8ff", sw=4, arrow="end")
    P.dump("nodes.json")                       # then: insert_page.py deck.jbd.json --nodes nodes.json --after 3 ...

Everything you add becomes a real node (shape / text / polygon / polyline); nothing is an image. Open lines are `polyline`
(stroke only), closed outlines are `polygon`, arrow heads are small polygons. Colours: any CSS colour; use rgba() for fill opacity.
"""
import json
import math

PAGE_W, PAGE_H = 1280, 720
TXT = "#f2ede4"
MIN_FONT = 12          # smaller than this on the 1280x720 page is unreadable when projected


def text_width(s, size, bold=False):
    """Rough rendered width in page px (CJK/full-width = 1 em, Latin = 0.56 em, digits/punctuation 0.55 em)."""
    w = 0.0
    for ch in s:
        if ch == "\n": continue
        o = ord(ch)
        w += size * (1.0 if (o >= 0x2E80 or 0xFF00 <= o <= 0xFFEF) else (0.6 if bold else 0.56))
    return w


class Page:
    def __init__(self, src_w=PAGE_W, src_h=PAGE_H, prefix="pg"):
        if not (math.isfinite(src_w) and math.isfinite(src_h) and src_w > 0 and src_h > 0):
            raise ValueError("source dimensions must be positive finite numbers")
        self.k = min(PAGE_W / src_w, PAGE_H / src_h)
        self.offset_x = (PAGE_W - src_w * self.k) / 2
        self.offset_y = (PAGE_H - src_h * self.k) / 2
        self.src_w, self.src_h = src_w, src_h
        if self.offset_x > 1 or self.offset_y > 1: print(f"[info] source {src_w}x{src_h} fitted without cropping; centred with margins")
        self.prefix, self.nodes, self.warnings = prefix, [], []

    # ---------------------------------------------------------------- internals
    def _s(self, v): return v * self.k

    def _add(self, typ, x, y, w, h, style, text="", kind=None, z=2, name=None):
        n = {"id": f"{self.prefix}-{len(self.nodes) + 1}", "type": typ, "name": name or f"{typ}#{len(self.nodes) + 1}", "sceneId": "SCENE",
             "transform": {"x": round(self.offset_x + self._s(x), 2), "y": round(self.offset_y + self._s(y), 2), "scaleX": 1, "scaleY": 1, "rotation": 0},
             "width": max(round(self._s(w), 2), 1), "height": max(round(self._s(h), 2), 1), "zIndex": z, "locked": False, "style": style, "text": text}
        if kind: n["shapeKind"] = kind
        self.nodes.append(n)
        return n

    @staticmethod
    def _base(**k):
        return dict({"fill": "transparent", "stroke": "transparent", "textColor": TXT, "fontSize": 16, "radius": 0, "borderWidth": 0}, **k)

    def _pts(self, n, pts, bx, by, bw, bh):
        smooth = [bool(p[2]) if len(p) > 2 else False for p in pts]
        n["polygonPoints"] = [dict({"x": round((p[0] - bx) / bw * 100, 2), "y": round((p[1] - by) / bh * 100, 2)}, **({"smooth": True} if sm else {}))
                              for p, sm in zip(pts, smooth)]

    # ---------------------------------------------------------------- primitives (all arguments in source px)
    def rect(self, x, y, w, h, fill="transparent", stroke="transparent", sw=0, radius=0, dash=False, z=1, **style):
        """Rectangle / rounded rectangle. radius in source px. Extra style keys (shadowColor, gradientTo, fillOpacity ...) pass through."""
        st = self._base(fill=fill, stroke=stroke, radius=round(self._s(radius), 1), borderWidth=round(self._s(sw), 1), borderStyle="dashed" if dash else "solid", **style)
        return self._add("shape", x, y, w, h, st, "", "rectangle", z)

    def ellipse(self, cx, cy, rx, ry=None, fill="transparent", stroke="transparent", sw=0, z=2, **style):
        ry = rx if ry is None else ry
        st = self._base(fill=fill, stroke=stroke, borderWidth=round(self._s(sw), 1), **style)
        return self._add("shape", cx - rx, cy - ry, 2 * rx, 2 * ry, st, "", "ellipse", z)

    def text(self, x, y, w, h, s, size=20, color=TXT, bold=False, align="left", vcenter=False, line=1.25, z=4, math=False, **style):
        """Text box. size in source px. s may contain $...$ (KaTeX) and \\n. vcenter=True centres ONE line vertically inside h.
        Wrapping happens automatically inside w, so give the box the width of the original text block."""
        fs = round(self._s(size), 1)
        st = self._base(textColor=color, fontSize=fs, fontWeight=700 if bold else 400, textAlign=align, lineHeight=round(self._s(h) / fs, 3) if vcenter else line, **style)
        if fs < MIN_FONT: self.warnings.append(f"font {fs}px < {MIN_FONT}px: {s[:24]!r}")
        if not math and "$" not in s:
            lines = sum(max(1, math_ceil(text_width(ln, fs, bold) / max(self._s(w) - 4, 1))) for ln in s.split("\n"))
            need = lines * fs * (1.0 if vcenter else line)
            if need > self._s(h) * 1.15 + 2: self.warnings.append(f"text may overflow its box ({need:.0f}px needed, {self._s(h):.0f}px given): {s[:24]!r}")
        return self._add("text", x, y, w, h, st, s, "rectangle", z)

    def label(self, x, y, w, h, s, size=20, color=TXT, bold=True, **kw):
        """One centred line in a box (helper = text(..., align='center', vcenter=True))."""
        return self.text(x, y, w, h, s, size=size, color=color, bold=bold, align="center", vcenter=True, **kw)

    def poly(self, pts, fill="#ffffff", stroke="transparent", sw=0, z=3, **style):
        """Closed polygon, 3-200 points [(x, y) | (x, y, True)]; a point with True is a curve (Catmull-Rom) point."""
        assert 3 <= len(pts) <= 200
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        bx, by, bw, bh = min(xs), min(ys), max(max(xs) - min(xs), 1), max(max(ys) - min(ys), 1)
        n = self._add("shape", bx, by, bw, bh, self._base(fill=fill, stroke=stroke, borderWidth=round(self._s(sw), 1), **style), "", "polygon", z)
        self._pts(n, pts, bx, by, bw, bh)
        return n

    def line(self, pts, color="#ffffff", sw=3, dash=False, smooth=False, arrow=None, head=14, z=3, **style):
        """Open polyline through pts (2-200 points), stroke only. smooth=True turns every point into a curve point.
        arrow='end' | 'start' | 'both' adds a filled triangular head (head = length in source px) at that end."""
        pts = [(p[0], p[1], True if smooth else (bool(p[2]) if len(p) > 2 else False)) for p in pts]
        assert 2 <= len(pts) <= 200
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        bx, by, bw, bh = min(xs), min(ys), max(max(xs) - min(xs), 1), max(max(ys) - min(ys), 1)
        st = self._base(stroke=color, borderWidth=round(self._s(sw), 1), **({"borderStyle": "dashed"} if dash else {}), **style)
        n = self._add("shape", bx, by, bw, bh, st, "", "polyline", z)
        self._pts(n, pts, bx, by, bw, bh)
        if arrow in ("end", "both"): self._head(pts[-2], pts[-1], color, head, z)
        if arrow in ("start", "both"): self._head(pts[1], pts[0], color, head, z)
        return n

    def _head(self, a, b, color, L, z):
        dx, dy = b[0] - a[0], b[1] - a[1]; d = math.hypot(dx, dy) or 1; ux, uy = dx / d, dy / d
        px, py = -uy, ux; hw = L * 0.5
        bx_, by_ = b[0] - ux * L, b[1] - uy * L
        self.poly([(b[0], b[1]), (bx_ + px * hw, by_ + py * hw), (bx_ - px * hw, by_ - py * hw)], fill=color, z=z)

    def grid(self, x, y, cols, rows, cw, ch=None, fills=None, default="#f2ede4", stroke="rgba(30,30,30,0.7)", sw=1):
        """cols x rows of equal cells. fills = {(row, col): colour}. Returns nothing; cells are plain rectangles."""
        ch = cw if ch is None else ch
        for r in range(rows):
            for c in range(cols):
                self.rect(x + c * cw, y + r * ch, cw, ch, fill=(fills or {}).get((r, c), default), stroke=stroke, sw=sw, z=2)

    # ---------------------------------------------------------------- output
    def check(self):
        """Print geometry warnings: tiny fonts, text that probably overflows, nodes outside the 1280x720 page."""
        for n in self.nodes:
            t = n["transform"]
            if t["x"] < -1 or t["y"] < -1 or t["x"] + n["width"] > PAGE_W + 1 or t["y"] + n["height"] > PAGE_H + 1:
                self.warnings.append(f"outside the page: {n['name']} at ({t['x']:.0f},{t['y']:.0f}) {n['width']:.0f}x{n['height']:.0f}")
        for w in self.warnings: print("[warn]", w)
        return self.warnings

    def dump(self, path):
        self.check()
        json.dump({"nodes": self.nodes}, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"{len(self.nodes)} nodes -> {path}")


def math_ceil(v): return int(math.ceil(v - 1e-9))
