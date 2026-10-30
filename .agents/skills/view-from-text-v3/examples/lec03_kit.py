"""Helpers for the week-3 (lec03) decks, on top of lec01_kit. One Deck per language; pages are built with D.page / D.done."""
import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lec01_kit import *

GOLDL = "#fef3c7"

class D:
    def __init__(s, lang, kp, n):
        s.zh = lang == "zh"; s.kp = kp; s.n = n; s.pages = []; s.i = 0
    def T(s, a, b): return a if s.zh else b
    def page(s, tz, te):
        s.i += 1; return new_page(s.i, s.n, s.kp, tz if s.zh else te)
    def done(s, g, lz, le, notes):
        s.pages.append((g, lz if s.zh else le, {b: (v[0] if s.zh else v[1]) for b, v in notes.items()}))

def box(g, x, y, w, h, col, light, title=None, body=None, ts=24, bs=24, th=None, bold=True, tcol=None, bcol=INK, z=3):
    ix = g.mark(); g.rect(x, y, w, h, fill=light, stroke=col, sw=3, radius=16, z=z)
    th = th if th is not None else (round(ts * 2.0) if title else 0)
    if title: g.text(x + 6, y + 6, w - 12, th, title, size=ts, color=tcol or col, bold=True, align="center", z=8)
    if body: g.text(x + 8, y + th + 10, w - 16, h - th - 16, body, size=bs, color=bcol, bold=bold, align="center", z=8)
    return ix

def gold(g, x, y, w, h, text, size=24, col=INK, light=GOLDL, stroke=GOLD):
    g.rect(x, y, w, h, fill=light, stroke=stroke, sw=3, radius=16, z=3)
    g.text(x + 12, y + 8, w - 24, h - 14, text, size=size, color=col, bold=True, align="center", z=8)

def chain(g, x, y, w, h, items, gap=44, beat0=None, ts=22, bs=22, th=None, hooks=(4, 12), lcol=GRAY):
    """items: (col, light, title, body); equal widths; connectors between neighbours. beat0 -> beat of first item, +1 each if beat0 not None."""
    n = len(items); bw = (w - gap * (n - 1)) / n; ids = []
    for i, (col, light, t, b_) in enumerate(items):
        if beat0 is not None: g.beat = beat0 + i
        ids.append(box(g, x + i * (bw + gap), y, bw, h, col, light, t, b_, ts=ts, bs=bs, th=th))
        if i: g.link(ids[i - 1], ids[i], hooks[0], hooks[1], lcol, 4)
    return ids

def table(g, x, y, colw, header, rows, rh=62, fs=24, hfs=22, styles=None, beats=None, gap=10):
    """colw: column widths; header: list; rows: list of lists of strings; styles: per-row (fill, stroke) or None; beats: per-row beat."""
    cx = [x]; [cx.append(cx[-1] + w) for w in colw]
    for k, h in enumerate(header): g.text(cx[k], y, colw[k], 50, h, size=hfs, color=MUTED, bold=True, align="center", z=8)
    for i, r in enumerate(rows):
        if beats: g.beat = beats[i]
        yy = y + 54 + i * (rh + gap); fl, st = (styles[i] if styles and styles[i] else (CARD, GRAY))
        g.rect(x - 6, yy, sum(colw) + 12, rh, fill=fl, stroke=st, sw=3, radius=12, z=3)
        for k, t in enumerate(r): g.text(cx[k], yy + 6, colw[k], rh - 8, t, size=fs, color=INK, bold=True, align="center", z=8)
    return y + 54 + len(rows) * (rh + gap)

def axes(g, x0, y0, w, h, xr, yr, zero=True):
    ux = lambda a: x0 + (a - xr[0]) / (xr[1] - xr[0]) * w; uy = lambda v: y0 + h - (v - yr[0]) / (yr[1] - yr[0]) * h
    g.rect(x0, y0, w, h, fill=CARD, stroke=GRAY, sw=3, radius=12, z=1)
    if zero and yr[0] < 0 < yr[1]: g.line([(x0, uy(0)), (x0 + w, uy(0))], GRAY, sw=2, z=2)
    if zero and xr[0] < 0 < xr[1]: g.line([(ux(0), y0), (ux(0), y0 + h)], GRAY, sw=2, z=2)
    return ux, uy

def curve(g, ux, uy, fn, col, xr, yr, sw=5, n=80, dash=False):
    pts = []
    for i in range(n + 1):
        a = xr[0] + (xr[1] - xr[0]) * i / n; v = fn(a); pts.append((ux(a), uy(min(max(v, yr[0]), yr[1]))))
    g.line(pts, col, sw=sw, dash=dash, z=5)

def dot(g, x, y, r=12, col=ORG, z=7): g.ellipse(x, y, r, fill=col, stroke="#ffffff", sw=3, z=z)

def lab(g, x, y, w, s, size=20, col=INK, align="left", h=None):
    g.text(x, y, w, h or round(size * 2.1), s, size=size, color=col, bold=True, align=align, z=8)

def closing(d, items, q, nxt, kind="confused"):
    """last page: up to 3 myth cards + a question box + mascot. items: (col, light, zh_myth, en_myth, zh_fix, en_fix); q: (zh,en); nxt: (zh,en)"""
    T = d.T
    g = d.page(*(("最後:迷思與下一個問題", "Last: myths and the next question")))
    n = len(items); w = (1180 - 30 * (n - 1)) / n
    for i, (col, light, mz, me, fz, fe) in enumerate(items):
        g.beat = i + 1
        box(g, 50 + i * (w + 30), 130, w, 280, col, light, T(mz, me), T(fz, fe), ts=22 if n == 3 else 24, bs=21 if n == 3 else 23, th=110)
    g.beat = n + 1
    g.rect(50, 440, 1000, 170, fill=GOLDL, stroke=GOLD, sw=3, radius=16, z=3)
    g.text(70, 450, 960, 152, T(*q), size=24, color=INK, bold=True, align="center", z=8)
    mascot(g, 1100, 440, 0.5, kind)
    banner(g, T(*nxt))
    notes = {i + 1: ((items[i][2] + "。" + items[i][4]), (items[i][3] + ". " + items[i][5])) for i in range(n)}
    notes[n + 1] = (q[0], q[1]); notes[0] = ("最後看常見迷思，以及留給下一個單元的問題。", "Finally, the common myths and the question left for the next unit.")
    d.done(g, "迷思與下一個問題", "Myths and next question", notes)

def bowl_contours(g, ux, uy, levels=(1, 4, 16, 36, 64, 100), col="#bfdbfe", a=1.0, b=25.0):
    """level sets of f = (a x^2 + b y^2)/2: ellipses with semi-axes sqrt(2c/a), sqrt(2c/b)."""
    for c in levels:
        rx = math.sqrt(2 * c / a); ry = math.sqrt(2 * c / b)
        g.ellipse(ux(0), uy(0), abs(ux(rx) - ux(0)), abs(uy(ry) - uy(0)), stroke=col, sw=2, z=2)
