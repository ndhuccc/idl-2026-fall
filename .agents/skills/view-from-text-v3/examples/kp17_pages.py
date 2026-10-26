"""KP17 pages designed in HTML (view-from-text-v3, route A). Each pageN(lang) returns the body HTML of a 1160x470 figure."""
import os, sys, math
import numpy as np
SK = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, SK + "/view-from-text-v3/tools"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from htmlkit import *
import kp17_data as D
W, H = 1160, 470

import htmlkit as _hk
def _cw(ch, size, bold):
    o = ord(ch)
    if o > 0x2e80 or ch in "×⁻⁰¹²³⁴⁵⁶⁷⁸⁹": return size * (1.0 if o > 0x2e80 else 0.62)
    return size * (0.6 if bold else 0.55) * (0.6 if ch in " ,.:;'|!ilI()" else 1.0 if ch.islower() or ch.isdigit() else 1.2)
def _wrap(s, w, size, bold):
    if "<br>" in s: return s, s.count("<br>") + 1
    lines, cur, cw, last = [], "", 0.0, -1
    for ch in s:
        a = _cw(ch, size, bold)
        if ch == "〔" and cur and cw + 6 * size > w: lines.append(cur); cur = ""; cw = 0.0; last = -1
        if cw + a > w and cur and "〔" not in cur[-6:]:
            if last >= 0 and ord(ch) < 0x2e80: lines.append(cur[:last].rstrip()); cur = cur[last:].lstrip() + ch; cw = sum(_cw(c, size, bold) for c in cur)
            else: lines.append(cur); cur = ch; cw = a
            last = -1; continue
        cur += ch; cw += a
        if ch == " ": last = len(cur)
    lines.append(cur); return "<br>".join(lines), len(lines)
def txt(x, y, w, s, size=16, color=TXT, bold=False, ha="center", h=24):
    s2, n = _wrap(s, w - 8, size, bold); lh = round(size * 1.5)
    return _hk.txt(x, y, w, s2, size, color, bold, ha, n * lh if n > 1 else (h if h <= 30 else lh))
def named(x, y, w, s, name, size=16, color=_hk.MUT, h=24, ha="left"):
    s2, n = _wrap(s, w - 8, size, False); lh = round(size * 1.5)
    return f'<div data-name="{name}" style="{_hk._p(x, y, w, n * lh)}font-size:{size}px;color:{color};text-align:{ha};line-height:{lh}px">{s2}</div>'
formula = k
BL, RD, VI, GR, YE, OR, TE = (hexc(c) for c in ("blue", "red", "violet", "green", "yellow", "orange", "teal"))
def tr(lang): return lambda zh, en: zh if lang == "zh" else en
def take(t, zh, en, y=418, role="yellow"):
    return card(0, y, W, 50, role, .14) + txt(16, y + 12, W - 32, t(zh, en), 18, TXT, True, "center", 26)
def _sup(n): return str(n).translate(str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻"))
def fmt(v):
    if v >= 10: return f"{v:.1f}"
    if v >= 0.01: return f"{v:.3g}"
    if v >= 1e-3: return f"{v:.2g}"
    return sci(v)
def sci(v):
    if v >= 0.01: return f"{v:.3g}"
    if v >= 1e-3: return f"{v:.2g}"
    m, e = f"{v:.2e}".split("e"); sig = 3 if v >= 1e-4 else 2
    return f"{float(m):.{sig-1}f}".rstrip("0").rstrip(".") + "×10" + _sup(int(e))
class Axes:
    """SVG axes in a box (x,y,w,h) of the figure; data -> pixel; log y optional."""
    def __init__(s, x, y, w, h, xr, yr, log=False): s.x, s.y, s.w, s.h, s.xr, s.yr, s.log = x, y, w, h, xr, yr, log; s.o = []
    def X(s, v): return s.x + (v - s.xr[0]) / (s.xr[1] - s.xr[0]) * s.w
    def Y(s, v):
        a, b = s.yr; v = math.log10(v) if s.log else v; return s.y + s.h - (v - a) / (b - a) * s.h
    def frame(s, xt, yt, xlab=None, ylab=None, yfmt=None, xnames=None):
        s.o.append(f'<line x1="{s.x}" y1="{s.y}" x2="{s.x}" y2="{s.y+s.h}" stroke="#b7b0a5" stroke-width="2"/><line x1="{s.x}" y1="{s.y+s.h}" x2="{s.x+s.w}" y2="{s.y+s.h}" stroke="#b7b0a5" stroke-width="2"/>')
        for t in xt:
            px = s.X(t); s.o.append(f'<line x1="{px:.1f}" y1="{s.y+s.h}" x2="{px:.1f}" y2="{s.y+s.h+6}" stroke="#b7b0a5" stroke-width="2"/><text x="{px:.1f}" y="{s.y+s.h+24}" text-anchor="middle" font-size="16" fill="#b7b0a5">{(xnames or {}).get(t, f"{t:g}")}</text>')
        for t in yt:
            py = s.Y(t) if True else 0; lab = yfmt(t) if yfmt else f"{t:g}"
            s.o.append(f'<line x1="{s.x}" y1="{py:.1f}" x2="{s.x+s.w}" y2="{py:.1f}" stroke="#6b645a" stroke-width="1" opacity="0.45"/><line x1="{s.x-6}" y1="{py:.1f}" x2="{s.x}" y2="{py:.1f}" stroke="#b7b0a5" stroke-width="2"/><text x="{s.x-10}" y="{py+5:.1f}" text-anchor="end" font-size="16" fill="#b7b0a5">{lab}</text>')
        return s
    def line(s, pts, col, w=3, dash=None, op=1):
        for (a, b), (c, d) in zip(pts, pts[1:]):
            s.o.append(f'<line x1="{s.X(a):.1f}" y1="{s.Y(b):.1f}" x2="{s.X(c):.1f}" y2="{s.Y(d):.1f}" stroke="{col}" stroke-width="{w}" opacity="{op}"' + (f' stroke-dasharray="{dash}"' if dash else "") + '/>')
        return s
    def dot(s, x, y, col, r=7): s.o.append(f'<circle cx="{s.X(x):.1f}" cy="{s.Y(y):.1f}" r="{r}" fill="{col}"/>'); return s
    def text(s, x, y, t, col="#f2ede4", anchor="middle", size=16, bold=True, dx=0, dy=0): s.o.append(f'<text x="{s.X(x)+dx:.1f}" y="{s.Y(y)+dy:.1f}" text-anchor="{anchor}" font-size="{size}" font-weight="{700 if bold else 400}" fill="{col}">{t}</text>'); return s
    def rect(s, x0, x1, y0, y1, col, op=1): s.o.append(f'<rect x="{s.X(x0):.1f}" y="{s.Y(y1):.1f}" width="{s.X(x1)-s.X(x0):.1f}" height="{s.Y(y0)-s.Y(y1):.1f}" fill="{col}" opacity="{op}"/>'); return s
    def raw(s, t): s.o.append(t); return s
    def html(s): return svg(0, 0, W, H, "".join(s.o))
def legend(items, x, y, w=300):
    o = ""
    for i, (col, name) in enumerate(items): o += chip(x, y + i * 32, 22, "grey", "", h=4) .replace("rgba(183,176,165,0.28)", col) if False else f'<div style="position:absolute;left:{x}px;top:{y+i*30+9}px;width:26px;height:5px;background:{col};border-radius:3px"></div>' + txt(x + 34, y + i * 30, w, name, 16, TXT, False, "left", 24)
    return o

# ---------------------------------------------------------------- page 1: entry ticket
def page1(lang):
    t = tr(lang); o = ""
    cols = [("blue", t("多項式", "Polynomial"), t("Weierstrass 定理：任何連續函數", "Weierstrass: any continuous function"), t("誤差 ∼ r^(−2/l)", "error ∼ r^(−2/l)"), t("固定基底", "fixed basis"), r"\phi(x)=\sum_{i=0}^{r}c_i\,x^i"),
            ("red", t("一層 sigmoid 網路", "One-layer sigmoid network"), t("任何連續函數", "any continuous function"), t("誤差 ∼ 1/k", "error ∼ 1/k"), t("資料自適應基底", "data-adaptive basis"), r"\phi(x)=\sum_{j=1}^{k}w_j\,f(w_j^{T}x)"),
            ("violet", t("RBF 網路", "RBF network"), t("任何連續函數", "any continuous function"), t("同樣是通用近似器", "also a universal approximator"), t("核函數的加權和", "weighted sum of kernels"), r"\phi(x)=\sum_{i}w_i\,e^{-\|x-c_i\|^2/2\sigma^2}")]
    cw, gap = 366, 31
    for i, (role, name, what, speed, basis, f) in enumerate(cols):
        x = i * (cw + gap) + 0
        o += card(x, 8, cw, 288, role) + head(x, 8, cw, role, name, 38, 19)
        o += txt(x + 10, 56, cw - 20, f'{t("能逼近", "Can approximate")}', 16, MUT, False, "center", 24) + chip(x + 24, 84, cw - 48, role, "✓ " + what, h=34)
        o += formula(x + 8, 134, cw - 16, f, 21, 52)
        o += txt(x + 10, 198, cw - 20, t("誤差下降的速度", "How fast the error falls"), 16, MUT, False, "center", 24) + chip(x + 24, 226, cw - 48, role, speed, True, 34, 17)
        o += txt(x + 10, 266, cw - 20, basis, 16, MUT, False, "center", 24)
    o += named(0, 306, 560, t("通用近似定理：節點（階數）夠多，就能把緊緻集上的任何連續函數逼近到任意小的誤差。", "Universal approximation theorem: with enough nodes (or degree), any continuous function on a compact set can be approximated to any accuracy."), "term:" + t("通用近似定理", "Universal approximation theorem"), 17, TXT, 56, "left") if False else ""
    o += card(0, 306, 1160, 100, "grey", .1)
    o += named(20, 314, 1120, t("通用近似定理：節點（或階數）夠多，就能把緊緻集上的任何連續函數逼近到任意小的誤差。", "Universal approximation theorem: enough nodes (or degree) approximate any continuous function on a compact set to any accuracy."), "term:" + t("通用近似定理", "Universal approximation theorem"), 16, TXT, 28, "left")
    o += named(20, 346, 1120, t("緊緻集：有界又含邊界的範圍，例如區間 [0, 1]。", "Compact set: a bounded range that includes its boundary, for example the interval [0, 1]."), "term:" + t("緊緻集", "Compact set"), 16, TXT, 28, "left")
    o += txt(20, 376, 1120, t("三種都成立；真正的差別是誤差下降的速度與最佳化難不難。", "All three hold; what really differs is how fast the error falls and how hard the optimization is."), 17, MUT, False, "left", 26)
    o += take(t, "通用近似是入場券，不是競爭優勢；比的是速度與最佳化的難易。", "Universal approximation is the entry ticket, not an edge; the contest is speed and ease of optimization.")
    return o

# ---------------------------------------------------------------- page 2: polynomial guarantee
def page2(lang):
    t = tr(lang); e = D.poly_errs(); ax = Axes(104, 56, 640, 300, (0.5, 11.5), (-7.2, 0.45), log=True)
    ax.frame(range(1, 12), [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7], yfmt=lambda v: {1: "1", 0.1: "0.1"}.get(v, "10" + _sup(int(round(math.log10(v))))))
    pts = [(r, e[r]) for r in range(1, 12)]; ax.line(pts, BL, 3)
    for r, v in pts: ax.dot(r, v, BL, 7)
    for r in (1, 3, 5, 7, 9, 11): ax.text(r, e[r], sci(e[r]), TXT, "middle", 16, True, dy=-16 if r < 11 else -16, dx=(0 if r < 11 else -10))
    ax.raw(f'<text x="{ax.x+ax.w/2}" y="{ax.y+ax.h+52}" text-anchor="middle" font-size="16" font-weight="700" fill="#f2ede4">{t("多項式階數 r", "polynomial degree r")}</text><text x="{ax.x-8}" y="30" font-size="16" font-weight="700" fill="#f2ede4">{t("最大誤差（對數座標）", "max error (log scale)")}</text>')
    o = ax.html()
    x0 = 776
    o += card(x0, 8, 384, 398, "blue") + head(x0, 8, 384, "blue", t("Weierstrass 定理", "Weierstrass theorem"), 36, 18)
    o += txt(x0 + 12, 54, 360, t("緊緻集 S 上的連續函數 g，對任何 ε > 0，存在 r 階多項式 φ，使", "For continuous g on a compact set S and any ε > 0, some degree-r polynomial φ gives"), 16, TXT, False, "left", 48)
    o += formula(x0 + 8, 110, 368, r"|g(x)-\phi(x)|<\epsilon,\quad \forall x\in S", 21, 40)
    o += named(x0 + 12, 160, 360, t("最大誤差：目標與近似的差取絕對值，在整個區間取最大者，比平均誤差嚴格。", "Max error: the largest absolute gap between target and approximation over the whole interval, stricter than the average error."), "term:" + t("最大誤差", "Max error"), 16, TXT, 72)
    o += txt(x0 + 12, 242, 360, t("實驗：g = sin 2πx + 0.3x²，區間 [0, 1]，401 個等距點的最小二乘。", "Experiment: g = sin 2πx + 0.3x² on [0, 1], least squares on 401 equally spaced points."), 16, MUT, False, "left", 48)
    o += chip(x0 + 12, 304, 360, "yellow", t("r = 3 與 4、5 與 6 … 誤差相同：連成一個台階", "r = 3 and 4, 5 and 6 … equal error: a step"), False, 34, 16)
    o += txt(x0 + 12, 350, 360, t("每個台階約降 1～2 個數量級。〔程式計算〕", "Each step cuts the error by about one to two orders of magnitude. (computed)"), 16, TXT, False, "left", 24)
    o += take(t, "階數夠高，最大誤差可以任意小：r = 11 時只剩 2.5×10⁻⁷。", "With a high enough degree the max error becomes arbitrarily small: only 2.5×10⁻⁷ at r = 11.")
    return o

# ---------------------------------------------------------------- page 3: Runge
def page3(lang):
    t = tr(lang); xs, g, fi, fl = D.runge_curves(); tab = D.runge_table(); ax = Axes(70, 40, 560, 318, (-1, 1), (-4.2, 4.2))
    ax.frame([-1, -0.5, 0, 0.5, 1], [-4, -2, 0, 2, 4], yfmt=lambda v: f"{v:g}")
    ax.rect(-1, 1, -0.5, 0, "#6b645a", 0)
    ax.line([(x, min(max(y, -4.2), 4.2)) for x, y in zip(xs, fi)], RD, 3).line(list(zip(xs, fl)), BL, 3).line(list(zip(xs, g)), TXT, 3, "7 6")
    ax.raw(f'<text x="{ax.x+ax.w/2}" y="{ax.y+ax.h+50}" text-anchor="middle" font-size="16" font-weight="700" fill="#f2ede4">x</text>')
    ax.text(0.78, 3.6, t("等距內插 12 階", "equispaced interpolation, degree 12"), RD, "end", 16, True, dx=-4).text(0.78, 2.8, t("最小二乘 12 階", "least squares, degree 12"), BL, "end", 16, True, dx=-4).text(0.78, 2.0, t("目標 1/(1+25x²)", "target 1/(1+25x²)"), TXT, "end", 16, True, dx=-4)
    o = ax.html()
    x0 = 660
    o += card(x0, 8, 500, 400, "orange") + head(x0, 8, 500, "orange", t("最大誤差：階數越高越好嗎？", "Max error: is a higher degree always better?"), 36, 18)
    o += named(x0 + 14, 52, 472, t("龍格現象：用等距的點內插（逼多項式通過每個取樣點）高階多項式，端點附近劇烈擺動，誤差反而變大。", "Runge phenomenon: interpolating a high-degree polynomial through equally spaced points makes it swing wildly near the ends, so the error grows."), "term:" + t("龍格現象", "Runge phenomenon"), 16, TXT, 72)
    o += txt(x0 + 14, 132, 150, t("階數 r", "degree r"), 16, MUT, True, "center", 24) + txt(x0 + 164, 132, 150, t("等距內插", "interpolation"), 16, RD, True, "center", 24) + txt(x0 + 326, 132, 160, t("最小二乘", "least squares"), 16, BL, True, "center", 24)
    for i, (r, a, b) in enumerate(tab):
        y = 164 + i * 36; o += card(x0 + 14, y - 3, 472, 32, "grey", .08)
        o += txt(x0 + 14, y, 150, str(r), 17, TXT, True, "center", 26) + txt(x0 + 164, y, 150, fmt(a) if a < 10 else f"{a:.1f}", 17, TXT, True, "center", 26) + txt(x0 + 326, y, 160, fmt(b), 17, TXT, True, "center", 26)
    o += txt(x0 + 14, 354, 472, t("內插越高階越糟；最小二乘（400 個點）降得慢但穩。〔程式計算〕", "Interpolation gets worse with degree; least squares (400 points) falls slowly but steadily. (computed)"), 16, MUT, False, "left", 48)
    o += take(t, "迷思：階數越高越好。等距內插 r = 20 的最大誤差高達 59.8，最小二乘只有 0.014。", "Myth: a higher degree is better. Interpolation at r = 20 reaches max error 59.8; least squares only 0.014.")
    return o

# ---------------------------------------------------------------- page 4: sigmoid network
def page4(lang):
    t = tr(lang); xs, g, res = D.mlp_curves(); tab = D.mlp_table(); ax = Axes(66, 44, 480, 300, (0, 1), (-1.2, 1.4))
    ax.frame([0, 0.25, 0.5, 0.75, 1], [-1, 0, 1], yfmt=lambda v: f"{v:g}")
    ax.line(list(zip(xs, g)), TXT, 4, "7 6")
    cols = {1: OR, 2: TE, 4: VI}
    for kk in (1, 2, 4): ax.line(list(zip(xs, res[kk][1])), cols[kk], 4 if kk != 4 else 3)
    ax.line(list(zip(xs, g)), TXT, 2, "6 8")
    ax.raw(f'<text x="{ax.x+ax.w/2}" y="{ax.y+ax.h+50}" text-anchor="middle" font-size="16" font-weight="700" fill="#f2ede4">x</text>')
    o = ax.html() + legend([(TXT, t("目標 g = sin 2πx + 0.3x²", "target g = sin 2πx + 0.3x²")), (OR, f"k = 1  ({t('最大誤差', 'max error')} {fmt(tab[1])})"), (BL, f"k = 2  ({fmt(tab[2])})"), (RD, f"k = 4  ({fmt(tab[4])})")], 360, 372 - 2 if False else 52, 280) if False else ax.html()
    o += legend([(TXT, t("目標（虛線）", "target (dashed)")), (OR, "k = 1"), (TE, "k = 2"), (VI, "k = 4")], 372, 56, 140)
    x0 = 570
    o += card(x0, 8, 590, 400, "red") + head(x0, 8, 590, "red", t("一層 sigmoid 網路", "A one-layer sigmoid network"), 36, 18)
    o += formula(x0 + 8, 52, 574, r"\phi(x)=\sum_{j=1}^{k}w_j^{o}\,f\!\left(w_j^{hT}x\right)+w_o^{o}", 21, 46)
    o += named(x0 + 14, 104, 562, t("擠壓函數：把任何輸入壓進有限範圍的單調遞增函數，例如 sigmoid。活化函數是擠壓函數、k 夠大，就能逼近緊緻集上的連續函數。", "Squashing function: a monotone increasing function that maps any input into a finite range, such as the sigmoid."), "term:" + t("擠壓函數", "Squashing function"), 16, TXT, 96)
    o += txt(x0 + 14, 188, 562, t("一維：每個節點 2 個隱藏權重、1 個輸出權重，外加 1 個常數：參數 3k + 1 個。", "In 1-D each node has 2 hidden and 1 output weight, plus one constant: 3k + 1 parameters."), 16, MUT, False, "left", 48)
    o += txt(x0 + 14, 258, 90, "k", 16, MUT, True, "center", 24) + txt(x0 + 6, 292, 98, t("最大誤差", "max error"), 16, MUT, True, "center", 24)
    for i, kk in enumerate((1, 2, 3, 4, 6, 8)):
        x = x0 + 118 + i * 78
        o += chip(x, 256, 62, "red", str(kk), True, 28, 16) + txt(x - 5, 292, 72, fmt(tab[kk]), 16, TXT, True, "center", 24)
    o += txt(x0 + 14, 336, 562, t("最好的起點（8 個隨機起點取最好）。〔程式計算〕", "Best of 8 random starts. (computed)"), 16, MUT, False, "left", 24)
    o += take(t, "k = 2 就貼近目標：最大誤差 0.0142，k = 4 為 0.0058。", "Already close at k = 2: max error 0.0142, and 0.0058 at k = 4.")
    return o

# ---------------------------------------------------------------- page 5: equal parameters, 1-D vs 5-D
def barpanel(x, y, w, h, role, title, groups, lo, hi, ticks, names=("", "")):
    """Log-scale vertical bars. groups: [(label, [(value, colour, shown_text)...])]."""
    o = card(x, y, w, h, role) + head(x, y, w, role, title, 34, 17)
    ax = Axes(x + 64, y + 100, w - 84, h - 166, (0, 1), (math.log10(lo), math.log10(hi)), log=True)
    ax.frame([], ticks, yfmt=lambda v: {1: "1", 0.1: "0.1"}.get(v, "10" + _sup(int(round(math.log10(v)))))); n = len(groups); gw = ax.w / n; labels = []
    for gi, (lab, bars) in enumerate(groups):
        bw = min(64, (gw - 12) / len(bars)); x1 = ax.x + gi * gw + (gw - bw * len(bars)) / 2
        for bi, (v, col, shown) in enumerate(bars):
            top = ax.Y(max(v, lo)); bx = x1 + bi * bw; ax.raw(f'<rect x="{bx+3:.1f}" y="{top:.1f}" width="{bw-6:.1f}" height="{ax.y+ax.h-top:.1f}" rx="5" fill="{col}"/>')
            labels.append(f'<text x="{bx+bw/2-(16 if len(shown) > 7 else 0):.1f}" y="{top-8-22*bi:.1f}" text-anchor="middle" font-size="16" font-weight="700" fill="#f2ede4">{shown}</text>')
        ax.raw(f'<text x="{ax.x+gi*gw+gw/2:.1f}" y="{ax.y+ax.h+22}" text-anchor="middle" font-size="16" fill="#b7b0a5">{lab}</text>')
    ax.raw("".join(labels))
    return o + ax.html()
def page5(lang):
    t = tr(lang); ep = D.equal_params(); rg = D.ridge()
    g1 = [(f"{p} {t('參數', 'params')}", [(a, BL, sci(a)), (b, RD, sci(b))]) for p, a, b in ep]
    g2 = [(f"r={r}<tspan font-size='12'> </tspan>", [(v, BL, fmt(v))]) for r, n, v in rg] + [(f"k={kk}", [(1e-4, RD, "<10⁻⁴")]) for kk in (1, 2, 4)]
    g2 = [(f"r={r}", [(v, BL, fmt(v))]) for r, n, v in rg] + [(f"k={kk}", [(1e-4, RD, "<10⁻⁴")]) for kk in (1, 2, 4)]
    o = barpanel(0, 8, 520, 400, "teal", t("一維：相同參數量，多項式贏", "One dimension: same parameters, polynomial wins"), g1, 1e-14, 1, [1, 1e-3, 1e-6, 1e-9, 1e-12])
    o += barpanel(540, 8, 620, 400, "orange", t("5 維脊函數：網路贏", "A 5-D ridge function: the network wins"), g2, 1e-4, 1, [1, 1e-1, 1e-2, 1e-3, 1e-4])
    o += legend([(BL, t("多項式", "polynomial")), (RD, t("網路", "network"))], 386, 54, 120) if False else ""
    o += f'<div style="position:absolute;left:372px;top:62px;width:26px;height:5px;background:{BL};border-radius:3px"></div>' + txt(404, 54, 100, t("多項式", "polynomial"), 16, TXT, False, "left", 24) + f'<div style="position:absolute;left:372px;top:86px;width:26px;height:5px;background:{RD};border-radius:3px"></div>' + txt(404, 78, 100, t("網路", "network"), 16, TXT, False, "left", 24)
    o += txt(560, 48, 590, t("縱軸：新點誤差（對數座標）；橫軸標籤第二項為模型", "y: error on new points (log scale)"), 15, MUT, False, "left", 24) if False else ""
    o += named(556, 52, 596, t("脊函數：目標只依賴輸入的單一方向（這裡是 tanh(2 w·x)），w 是隨機單位向量。〔程式計算〕", "Ridge function: the target depends on the input only through one direction (here tanh(2 w·x)), with w a random unit vector. (computed)"), "term:" + t("脊函數", "Ridge function"), 16, TXT, 24)
    o += take(t, "維度一升，勝負反轉：1 維多項式勝，5 維脊函數網路勝（0.222 以上對 0.0001 以下）。", "Raise the dimension and the winner flips: polynomial in 1-D, network on a 5-D ridge (0.222 and up vs below 0.0001).")
    return o

# ---------------------------------------------------------------- page 6: rates
def page6(lang):
    t = tr(lang); hv = D.halving(); ls = (2, 5, 10, 20)
    o = card(0, 8, 460, 400, "violet") + head(0, 8, 460, "violet", t("速率（Barron 1993，最壞情況）", "Rates (Barron 1993, worst case)"), 36, 18)
    o += chip(16, 62, 428, "blue", t("多項式", "Polynomial"), True, 30, 17) + formula(16, 98, 428, r"O\!\left(r^{-2/l}\right)", 24, 50) + txt(16, 160, 428, t("輸入維度 l 越大，誤差降得越慢", "the larger the input dimension l, the slower the drop"), 16, MUT, False, "center", 24)
    o += chip(16, 206, 428, "red", t("一層 sigmoid 網路", "One-layer sigmoid network"), True, 30, 17) + formula(16, 244, 428, r"O\!\left(1/k\right)", 24, 50) + txt(16, 298, 428, t("維度 l 完全沒出現在式子裡", "the dimension l does not appear at all"), 16, MUT, False, "center", 24)
    o += txt(16, 326, 428, t("誤差減半要多少？多項式：r 乘 2^(l/2)；網路：k 加倍", "To halve the error: polynomial multiplies r by 2^(l/2); the network just doubles k"), 16, TXT, True, "center", 48)
    ax = Axes(560, 78, 540, 240, (0, 4), (0, 3.4)); o2 = card(480, 8, 680, 400, "orange") + head(480, 8, 680, "orange", t("誤差減半所需的增長倍數", "Growth needed to halve the error"), 36, 18)
    ax.raw(f'<line x1="{ax.x}" y1="{ax.y+ax.h}" x2="{ax.x+ax.w}" y2="{ax.y+ax.h}" stroke="#b7b0a5" stroke-width="2"/>')
    for t_ in (1, 10, 100, 1000): py = ax.y + ax.h - math.log10(t_) / 3.2 * ax.h; ax.raw(f'<line x1="{ax.x}" y1="{py:.1f}" x2="{ax.x+ax.w}" y2="{py:.1f}" stroke="#6b645a" stroke-width="1" opacity="0.45"/><text x="{ax.x-8}" y="{py+5:.1f}" text-anchor="end" font-size="16" fill="#b7b0a5">{t_}</text>')
    gw = ax.w / 4
    for i, l in enumerate(ls):
        for j, (v, col, nm) in enumerate(((hv[l], BL, f"{hv[l]:.1f}".rstrip("0").rstrip(".") if hv[l] < 10 else f"{hv[l]:g}"), (2, RD, "2"))):
            top = ax.y + ax.h - math.log10(v) / 3.2 * ax.h if v > 1 else ax.y + ax.h; bx = ax.x + i * gw + gw / 2 - 48 + j * 48
            ax.raw(f'<rect x="{bx+3:.1f}" y="{top:.1f}" width="42" height="{ax.y+ax.h-top:.1f}" rx="5" fill="{col}"/><text x="{bx+24:.1f}" y="{top-8:.1f}" text-anchor="middle" font-size="16" font-weight="700" fill="#f2ede4">{nm}</text>')
        ax.raw(f'<text x="{ax.x+i*gw+gw/2:.1f}" y="{ax.y+ax.h+26}" text-anchor="middle" font-size="16" fill="#b7b0a5">l = {l}</text>')
    ax.raw(f'<text x="{ax.x+ax.w/2}" y="{ax.y+ax.h+54}" text-anchor="middle" font-size="16" font-weight="700" fill="#f2ede4">{t("輸入維度 l（縱軸對數座標）", "input dimension l (log scale on the y axis)")}</text>')
    o += o2 + ax.html()
    o += f'<div style="position:absolute;left:590px;top:62px;width:26px;height:5px;background:{BL};border-radius:3px"></div>' + txt(622, 54, 150, t("多項式 r", "polynomial r"), 16, TXT, False, "left", 24) + f'<div style="position:absolute;left:590px;top:92px;width:26px;height:5px;background:{RD};border-radius:3px"></div>' + txt(622, 84, 150, t("網路 k", "network k"), 16, TXT, False, "left", 24)
    o += txt(16, 376, 428, t("最壞情況的漸近速率，不是實測。〔程式計算〕", "A worst-case asymptotic rate, not a measurement. (computed)"), 16, MUT, False, "center", 24)
    o += take(t, "20 維時，多項式的 r 要乘 1024 倍才能讓誤差減半，網路的 k 只要加倍。", "At l = 20 the polynomial's r must grow 1024 times to halve the error; the network's k only doubles.")
    return o

# ---------------------------------------------------------------- page 7: fixed vs adaptive basis
def page7(lang):
    t = tr(lang); o = ""
    for i, (role, name) in enumerate((("blue", t("固定基底：多項式", "Fixed basis: polynomials")), ("red", t("資料自適應基底：網路的隱藏節點", "Data-adaptive basis: the network's hidden nodes")))):
        x = i * 590; o += card(x, 8, 570, 330, role) + head(x, 8, 570, role, name, 36, 18)
    o += named(14, 54, 542, t("固定基底：不管資料長什麼樣都一樣的基底函數（例如下面的單項式）。", "Fixed basis: basis functions that are the same whatever the data look like (for example the monomials below)."), "term:" + t("固定基底", "Fixed basis"), 16, TXT, 48)
    for i, tex in enumerate((r"x_1", r"x_2", r"x_1^2", r"x_1x_2", r"x_2^2", r"x_1^2x_2")):
        xx = 40 + (i % 3) * 170; yy = 130 + (i // 3) * 74; o += chip(xx, yy, 150, "blue", "", False, 56, 16) + formula(xx, yy + 6, 150, tex, 24, 44)
    o += txt(14, 284, 542, t("資料畫成什麼樣，這六個函數都不變。", "These six functions never change, whatever the data."), 16, MUT, False, "center", 24)
    o += named(604, 54, 542, t("資料自適應基底：基底函數裡的權重向量 w 由資料最佳化出來；網路的隱藏節點就是。", "Data-adaptive basis: basis functions whose weight vector w is optimized from the data; the hidden nodes of a network are such functions."), "term:" + t("資料自適應基底", "Data-adaptive basis"), 16, TXT, 48)
    for i in range(3):
        xx = 640 + i * 168; o += chip(xx, 136, 150, "red", "", False, 64, 16) + formula(xx, 146, 150, r"f(w_%d^{T}x)" % (i + 1), 22, 44)
        o += chip(xx + 10, 214, 130, "grey", t("w 由資料學", "w learned"), False, 30, 16)
        o += svg(xx + 70, 190, 10, 24, '<line x1="5" y1="2" x2="5" y2="22" stroke="#b7b0a5" stroke-width="2"/><line x1="0" y1="14" x2="5" y2="22" stroke="#b7b0a5" stroke-width="2"/><line x1="10" y1="14" x2="5" y2="22" stroke="#b7b0a5" stroke-width="2"/>')
    o += txt(604, 284, 542, t("w 隨資料調整，所以形狀跟著目標走。", "w follows the data, so the shape follows the target."), 16, MUT, False, "center", 24)
    o += card(0, 350, 1160, 56, "violet", .12) + txt(16, 360, 1128, t("來源說：所有固定基底的展開，誤差下降都逃不過隨維度變慢的宿命；自適應基底可以免。", "The source says every fixed-basis expansion suffers an error decrease that slows with dimension; an adaptive basis can escape it."), 17, TXT, True, "center", 36)
    o += take(t, "網路勝在基底會跟著資料調整；代價是下一頁的最佳化。", "The network wins because its basis adapts to the data; the cost is the optimization on the next page.")
    return o

# ---------------------------------------------------------------- page 8: restarts
def page8(lang):
    t = tr(lang); r2 = D.restarts(2); r3 = D.restarts(3)
    ax = Axes(74, 84, 640, 246, (0, 8), (0, 1.05)); ax.frame([], [0, 0.25, 0.5, 0.75, 1], yfmt=lambda v: f"{v:g}")
    ax.raw(f'<line x1="{ax.x}" y1="{ax.Y(0.1):.1f}" x2="{ax.x+ax.w}" y2="{ax.Y(0.1):.1f}" stroke="#e0c85f" stroke-width="2" stroke-dasharray="8 6"/><text x="{ax.x+ax.w+8}" y="{ax.Y(0.1)+5:.1f}" font-size="16" font-weight="700" fill="#e0c85f">0.1</text>')
    gw = ax.w / 8
    for i in range(8):
        for j, (v, col) in enumerate(((r2[i], BL), (r3[i], VI))):
            top = ax.Y(min(v, 1.0)); bx = ax.x + i * gw + gw / 2 - 26 + j * 28
            ax.raw(f'<rect x="{bx:.1f}" y="{top:.1f}" width="24" height="{ax.y+ax.h-top:.1f}" rx="4" fill="{col}"/>')
        ax.raw(f'<text x="{ax.x+i*gw+gw/2:.1f}" y="{ax.y+ax.h+24}" text-anchor="middle" font-size="16" fill="#b7b0a5">{i}</text>')
        ax.raw(f'<text x="{ax.x+i*gw+gw/2:.1f}" y="62" text-anchor="middle" font-size="16" font-weight="700" fill="{BL}">{r2[i]:.2g}</text>')
    ax.raw(f'<text x="{ax.x+ax.w/2}" y="{ax.y+ax.h+54}" text-anchor="middle" font-size="16" font-weight="700" fill="#f2ede4">{t("隨機起點（種子 0～7）", "random start (seed 0–7)")}</text><text x="{ax.x-8}" y="78" font-size="16" font-weight="700" fill="#f2ede4" text-anchor="end">{t("最大誤差", "max error")}</text><text x="{ax.x-8}" y="62" font-size="16" font-weight="700" fill="{BL}" text-anchor="end">k=2</text>')
    o = ax.html() + legend([(BL, "k = 2"), (VI, "k = 3")], 110, 92, 100)
    o += f'<div style="position:absolute;left:660px;top:15px;width:0px;height:0px"></div>'
    x0 = 750; low = sum(1 for v in r2 if v < 0.1); stuck = sum(1 for v in r2 if v > 0.5)
    o += card(x0, 8, 410, 400, "orange") + head(x0, 8, 410, "orange", t("代價：最佳化變成非線性", "The cost: optimization becomes nonlinear"), 36, 18)
    o += txt(x0 + 14, 56, 382, t("固定基底的係數是線性最小二乘，一步解出；網路的權重要用迭代法找，結果看起點。", "Fixed-basis coefficients come from linear least squares in one step; network weights need an iterative search, and the result depends on the start."), 16, TXT, False, "left", 96)
    o += chip(x0 + 14, 196, 382, "blue", t(f"k = 2，8 個起點：{low} 個低於 0.1", f"k = 2, 8 starts: {low} below 0.1"), False, 34, 17) + chip(x0 + 14, 242, 382, "red", t(f"{stuck} 個卡在 0.82 附近", f"{stuck} stuck near 0.82"), True, 34, 17)
    o += txt(x0 + 14, 296, 382, t("同一份資料、同一個目標，只換隨機起點。〔程式計算〕", "Same data, same target; only the random start changes. (computed)"), 16, MUT, False, "left", 48)
    o += txt(x0 + 14, 350, 382, t("所以實務上要多試幾個起點，取最好的。", "So in practice you try several starts and keep the best."), 16, TXT, True, "left", 48)
    o += take(t, "k = 2 的 8 次嘗試：5 次低於 0.1，3 次卡在 0.82 附近。", "Eight tries at k = 2: five fall below 0.1, three get stuck near 0.82.")
    return o

# ---------------------------------------------------------------- page 9: depth
def page9(lang):
    t = tr(lang); o = ""
    for i, n in enumerate((1, 2, 3)):
        x = i * 300; x_, y_ = D.hat(n, 2001); o += card(x, 8, 280, 280, "teal")
        ax = Axes(x + 26, 78, 230, 150, (0, 1), (0, 1)); ax.raw(f'<line x1="{ax.x}" y1="{ax.y+ax.h}" x2="{ax.x+ax.w}" y2="{ax.y+ax.h}" stroke="#b7b0a5" stroke-width="2"/>')
        step = 2 ** (n + 1); pts = [(x_[j], y_[j]) for j in range(0, 2001, 1000 // 2 ** n)] if False else [(a / 2 ** n, (0 if a % 2 == 0 else 1)) for a in range(0, 2 ** n + 1)]
        ax.line(pts, TE, 3)
        for a, b in pts: ax.dot(a, b, TE, 4)
        o += head(x, 8, 280, "teal", t(f"疊 {n} 層", f"{n} layer" + ("s" if n > 1 else "")), 34, 17) + ax.html()
        o += chip(x + 40, 40, 200, "teal", t(f"{2 ** n} 段直線", f"{2 ** n} line segments"), True, 30, 17)
        o += txt(x + 10, 244, 260, t(f"深網路 {2 * n} 個節點", f"deep network: {2 * n} nodes"), 16, TXT, True, "center", 24)
    x0 = 910
    o += card(x0, 8, 250, 280, "violet") + head(x0, 8, 250, "violet", t("帽函數", "Hat function"), 34, 17) + formula(x0, 56, 250, r"h(x)=2\min(x,\,1-x)", 17, 40)
    o += txt(x0 + 10, 108, 230, t("2 個 ReLU 節點就做得出。<br>ReLU：max(0, x) 這種<br>折線型活化函數。", "Two ReLU nodes make it.<br>ReLU: the piecewise-linear<br>activation max(0, x)."), 16, TXT, False, "left", 72)
    o += card(0, 300, 1160, 108, "grey", .1)
    o += txt(14, 306, 300, t("層數 n", "layers n"), 16, MUT, True, "left", 26) + txt(14, 340, 300, t("深網路的節點數 2n", "deep network: 2n nodes"), 16, TXT, True, "left", 26) + txt(14, 372, 300, t("單隱藏層的節點數 2ⁿ − 1", "one hidden layer: 2ⁿ − 1 nodes"), 16, TXT, True, "left", 26)
    for i, n in enumerate((3, 6, 10)):
        xx = 380 + i * 250; a, b = D.hat_nodes(n); o += chip(xx, 308, 200, "teal", f"n = {n}", True, 26, 16) + chip(xx, 340, 200, "teal", str(a), False, 26, 17) + chip(xx, 372, 200, "red", f"{b:,}", False, 26, 17)
    o += take(t, "每多一層，段數加倍：n = 10 時，深網路 20 個節點，單隱藏層要 1023 個。", "Each extra layer doubles the segments: at n = 10 a deep network needs 20 nodes, one hidden layer needs 1023.")
    return o

# ---------------------------------------------------------------- page 10: exercises
def page10(lang):
    t = tr(lang); o = ""
    qs = [("blue", t("題 1", "Q1"), t("l = 8，多項式誤差 ∼ r^(−2/l)。誤差減半，r 要乘多少倍？", "l = 8 and the polynomial error ∼ r^(−2/l). By what factor must r grow to halve the error?"), r"r^{-2/8}\ \text{減半}\ \Rightarrow\ r\times 2^{8/2}=2^{4}=16" if lang == "zh" else r"r^{-2/8}\ \text{halves}\ \Rightarrow\ r\times 2^{8/2}=2^{4}=16", t("r 要乘 16 倍。", "r must grow 16 times."), t("注意：這是漸近式，不是精確的實測。", "Note: this is an asymptotic rate, not an exact measurement.")),
          ("red", t("題 2", "Q2"), t("5 層帽函數有幾段？深網路要幾個 ReLU 節點、單隱藏層要幾個？", "How many segments does a 5-layer hat function have? How many ReLU nodes for a deep network and for one hidden layer?"), r"2^5=32,\quad 2\times5=10,\quad 2^5-1=31", t("32 段；深網路 10 個、單隱藏層 31 個。", "32 segments; deep 10 nodes; one hidden layer 31."), t("核對：32 段要 31 個折點，單層 k 個節點最多 k 個折點，所以 k ≥ 31。", "Check: 32 segments need 31 kinks and k nodes make at most k kinks, so k ≥ 31."))]
    for i, (role, name, q, tex, ans, chk) in enumerate(qs):
        x = i * 590; o += card(x, 8, 570, 400, role) + head(x, 8, 570, role, name, 36, 18)
        o += txt(x + 16, 56, 538, q, 18, TXT, True, "left", 80)
        o += chip(x + 16, 134, 130, "grey", t("解題示範", "Worked answer"), False, 30, 16)
        o += card(x + 16, 176, 538, 92, "grey", .1) + formula(x + 16, 190, 538, tex, 26, 64)
        o += chip(x + 16, 288, 538, role, ans, True, 40, 19)
        o += txt(x + 16, 346, 538, chk, 16, MUT, False, "left", 48)
    o += take(t, "速率式與段數都是算出來的：16 倍、32 段。", "Both the rate and the segment count are computed: 16 times, 32 segments.")
    return o

PAGES_DESIGN = [("p1", page1), ("p2", page2), ("p3", page3), ("p4", page4), ("p5", page5), ("p6", page6), ("p7", page7), ("p8", page8), ("p9", page9), ("p10", page10)]
if __name__ == "__main__":
    out = sys.argv[1]; only = sys.argv[2:] 
    for fid, fn in PAGES_DESIGN:
        if only and fid not in only: continue
        print(fid); build("kp17_" + fid, fn, W, H, out)
