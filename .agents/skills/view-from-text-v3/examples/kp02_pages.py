"""KP02 pages designed in HTML (view-from-text-v3, route A). pageN(lang) -> body HTML of a 1160x480 figure."""
import os, sys, math
SK = "/home/user/pr-2026-fall/.agents/skills"
HERE = os.path.dirname(os.path.abspath(__file__)); HERE = HERE[:-5] if HERE.endswith(("/W_zh", "/W_en")) else HERE
sys.path.insert(0, SK + "/view-from-text-v3/examples"); sys.path.insert(0, SK + "/view-from-text-v3/tools"); sys.path.insert(0, HERE)
from kp17_pages import Axes, tr, take, legend, fmt, sci, txt, named, card, head, chip, formula, svg, BL, RD, VI, GR, YE, OR, TE, INK, TXT, MUT, hexc, build, native, _sup, rgba, rgb
from kp15_pages import box, cardH
import kp02_nums as N
W, H = 1160, 480
M = "−"   # minus sign
def sg(v): return (M if v < 0 else "") + f"{abs(v):g}"
def sg1(v): return ("+" if v > 0 else M) + f"{abs(v):g}"
AF, BF = "#47688f", "#94503a"       # class A / class B fills
# ---------------------------------------------------------------- geometry helpers
def clip(poly, a, b, c, keep_pos=True):
    """Sutherland-Hodgman: keep the part of poly where a*x+b*y-c >0 (keep_pos) or <=0."""
    f = (lambda p: a * p[0] + b * p[1] - c) if keep_pos else (lambda p: -(a * p[0] + b * p[1] - c))
    out = []
    for i in range(len(poly)):
        p, q = poly[i], poly[(i + 1) % len(poly)]; fp, fq = f(p), f(q)
        if fp >= 0: out.append(p)
        if (fp > 0 and fq < 0) or (fp < 0 and fq > 0):
            t = fp / (fp - fq); out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out
def box_poly(xr, yr): return [(xr[0], yr[0]), (xr[1], yr[0]), (xr[1], yr[1]), (xr[0], yr[1])]
def region(ax, cuts):
    poly = box_poly(ax.xr, ax.yr)
    for a, b, c, pos in cuts: poly = clip(poly, a, b, c, pos)
    return poly
def poly_svg(ax, poly, fill, op, stroke=None):
    if len(poly) < 3: return ""
    pts = " ".join(f"{ax.X(x):.1f},{ax.Y(y):.1f}" for x, y in poly)
    return f'<polygon points="{pts}" fill="{fill}" fill-opacity="{op}"' + (f' stroke="{stroke}" stroke-width="3"' if stroke else "") + '/>'
def seg_in_box(ax, a, b, c):
    """endpoints of line a*x+b*y=c inside the view box"""
    pts = []
    (x0, x1), (y0, y1) = ax.xr, ax.yr
    if b:
        for x in (x0, x1):
            y = (c - a * x) / b
            if y0 - 1e-9 <= y <= y1 + 1e-9: pts.append((x, y))
    if a:
        for y in (y0, y1):
            x = (c - b * y) / a
            if x0 - 1e-9 <= x <= x1 + 1e-9: pts.append((x, y))
    u = []
    for p in pts:
        if all(abs(p[0] - q[0]) + abs(p[1] - q[1]) > 1e-9 for q in u): u.append(p)
    return u[:2]
def line_svg(ax, a, b, c, col, w=4, dash=None):
    p = seg_in_box(ax, a, b, c)
    return f'<line x1="{ax.X(p[0][0]):.1f}" y1="{ax.Y(p[0][1]):.1f}" x2="{ax.X(p[1][0]):.1f}" y2="{ax.Y(p[1][1]):.1f}" stroke="{col}" stroke-width="{w}"' + (f' stroke-dasharray="{dash}"' if dash else "") + '/>'
def arrow(x1, y1, x2, y2, col="#b7b0a5", w=3, head_=12):
    d = math.hypot(x2 - x1, y2 - y1); ux, uy = (x2 - x1) / d, (y2 - y1) / d; bx, by = x2 - ux * head_, y2 - uy * head_
    p = f"{x2:.1f},{y2:.1f} {bx - uy * head_ * .55:.1f},{by + ux * head_ * .55:.1f} {bx + uy * head_ * .55:.1f},{by - ux * head_ * .55:.1f}"
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{bx:.1f}" y2="{by:.1f}" stroke="{col}" stroke-width="{w}"/><polygon points="{p}" fill="{col}"/>'
def axes_svg(ax, xt, yt, xlab, ylab, fs=16):
    s = f'<rect x="{ax.x}" y="{ax.y}" width="{ax.w}" height="{ax.h}" fill="none" stroke="#6b645a" stroke-width="2"/>'
    zx, zy = ax.X(0), ax.Y(0)
    s += f'<line x1="{ax.x}" y1="{zy:.1f}" x2="{ax.x+ax.w}" y2="{zy:.1f}" stroke="#8a8378" stroke-width="1.5"/><line x1="{zx:.1f}" y1="{ax.y}" x2="{zx:.1f}" y2="{ax.y+ax.h}" stroke="#8a8378" stroke-width="1.5"/>'
    for t in xt: s += f'<text x="{ax.X(t):.1f}" y="{ax.y+ax.h+22}" text-anchor="middle" font-size="{fs}" fill="#b7b0a5">{t:g}</text>'
    for t in yt: s += f'<text x="{ax.x-10}" y="{ax.Y(t)+6:.1f}" text-anchor="end" font-size="{fs}" fill="#b7b0a5">{t:g}</text>'
    s += f'<text x="{ax.x+ax.w-4}" y="{ax.y+ax.h-8}" text-anchor="end" font-size="{fs}" font-weight="700" fill="#f2ede4">{xlab}</text>'
    s += f'<text x="{ax.x+10}" y="{ax.y+22}" text-anchor="start" font-size="{fs}" font-weight="700" fill="#f2ede4">{ylab}</text>'
    return s
def mark(ax, x, y, cls, r=14, hollow=False):
    cx, cy = ax.X(x), ax.Y(y)
    if hollow: return f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{r}" ry="{r}" fill="none" stroke="#b7b0a5" stroke-width="3" stroke-dasharray="6 5"/>'
    if cls == "A": return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{AF}" stroke="#ffffff" stroke-width="2"/>'
    return f'<rect x="{cx-r+1:.1f}" y="{cy-r+1:.1f}" width="{2*r-2}" height="{2*r-2}" rx="4" fill="{BF}" stroke="#ffffff" stroke-width="2"/>'
def ring(ax, x, y, r=24, col=RD): return f'<ellipse cx="{ax.X(x):.1f}" cy="{ax.Y(y):.1f}" rx="{r}" ry="{r}" fill="none" stroke="{col}" stroke-width="4"/>'
XR = (-0.4, 1.4)
G1C, G2C = TE, VI
# ------------------------------------------------ 1: wiring
def page1(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 790, 480, "blue", t("兩層網路的接線：輸入 → 隱藏層 → 輸出層", "Wiring of the two-layer network: inputs, hidden layer, output layer"))
    s = ""
    for xi, (yy, name) in enumerate(((140, "x₁"), (330, "x₂"))): o += chip(20, yy, 100, "grey", name, True, 52, 22)
    hid = [(66, "teal", G1C, t("隱藏單元 1", "Hidden unit 1"), r"g_1=x_1+x_2-0.5", "y₁"), (266, "violet", G2C, t("隱藏單元 2", "Hidden unit 2"), r"g_2=x_1+x_2-1.5", "y₂")]
    for (y0, role, col, name, tex, yb) in hid:
        o += card(190, y0, 270, 150, role, .2) + head(190, y0, 270, role, name, 34, 17) + formula(190, y0 + 44, 270, tex, 24, 44)
        o += txt(190, y0 + 100, 270, t("分數 → 階梯規則 → 位元", "score → step rule → bit"), 16, TXT, False, "center", 26)
        o += chip(520, y0 + 50, 70, role, yb, True, 50, 22)
        s += arrow(460, y0 + 75, 516, y0 + 75, col, 4)
    for yy in (166, 356):
        for y0 in (141, 341): s += arrow(122, yy if False else (166 if yy == 166 else 356), 188, y0 if False else y0, "#b7b0a5", 2) if False else ""
    for ya, xa in ((166, 120), (356, 120)):
        for yb_ in (141, 341): s += arrow(xa + 2, ya, 188, yb_, "#b7b0a5", 2.5, 10)
    o += card(650, 120, 130, 240, "orange", .2) + head(650, 120, 130, "orange", t("輸出單元", "Output unit"), 34, 17)
    o += formula(646, 168, 138, r"g_{\text{out}}", 24, 40) + txt(650, 214, 130, t("= y₁ − y₂<br>− 0.5", "= y₁ − y₂<br>− 0.5"), 17, TXT, True, "center", 52)
    o += txt(650, 282, 130, t("判 A 或 B", "A or B"), 17, TXT, False, "center", 28)
    s += arrow(592, 141, 648, 200, G1C, 4) + arrow(592, 341, 648, 290, G2C, 4)
    o += svg(0, 0, W, H, s)
    o += txt(20, 420, 760, t("輸入只提供座標，不算一層；隱藏層 + 輸出層 = 兩層", "Inputs only supply coordinates, not a layer: hidden + output = two layers"), 17, TXT, True, "center", 28)
    o += txt(20, 452, 760, t("每個單元都先算一個分數，它的分界線就是它在平面上的圖形", "Each unit computes a score; its boundary is a line in the plane"), 16, MUT, False, "center", 26)
    o += cardH(806, 0, 354, 480, "orange", t("名詞", "Terms"))
    o += named(820, 52, 326, t("隱藏層：先問小問題的一排計算單元，每個單元的分界線是平面上的一條線。", "Hidden layer: the row of units that asks the small questions; each unit's boundary is a line in the plane."), "term:" + t("隱藏層", "Hidden layer"), 16, TXT, 120)
    o += named(820, 184, 326, t("輸出層：最後在編碼上做分類的單元。", "Output layer: the unit that makes the final classification from the codes."), "term:" + t("輸出層", "Output layer"), 16, TXT, 72)
    o += named(820, 270, 326, t("兩層網路：隱藏層加輸出層；輸入不算一層。", "Two-layer network: hidden layer plus output layer; the inputs do not count as a layer."), "term:" + t("兩層網路", "Two-layer network"), 16, TXT, 72)
    o += card(820, 360, 326, 104, "yellow", .14) + txt(828, 368, 310, t("上一個知識點的結論：兩條平行線夾出一條帶，帶內剛好是 XOR 的 A。這裡把它做成網路。", "Last KP's conclusion: two parallel lines enclose a strip that holds exactly the A points. Here it becomes a network."), 16, TXT, False, "left", 96)
    return o
# ------------------------------------------------ 2: positive sides + step rule
def page2(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 700, 480, "teal", t("兩條線各有一個正側", "Each line has a positive side"))
    ax = Axes(60, 56, 380, 380, XR, XR)
    s = poly_svg(ax, region(ax, [(1, 1, 0.5, True)]), G1C, .30) + poly_svg(ax, region(ax, [(1, 1, 1.5, True)]), G2C, .38) + axes_svg(ax, [0, 1], [0, 1], "x₁", "x₂")
    s += line_svg(ax, 1, 1, 0.5, G1C, 4) + line_svg(ax, 1, 1, 1.5, G2C, 4)
    s += arrow(ax.X(0.25), ax.Y(0.25), ax.X(0.25) + 40, ax.Y(0.25) - 40, G1C, 4, 14) + arrow(ax.X(0.75), ax.Y(0.75), ax.X(0.75) + 40, ax.Y(0.75) - 40, G2C, 4, 14)
    for (a, b) in N.PTS: s += mark(ax, a, b, N.label(a, b))
    o += svg(0, 0, W, H, s)
    o += txt(ax.X(0.62), ax.Y(-0.12), 90, "g₁ = 0", 16, "#8fe0d0", True, "left", 24) + txt(ax.X(1.08), ax.Y(0.3), 90, "g₂ = 0", 16, "#c4b0e6", True, "left", 24)
    o += chip(470, 70, 210, "teal", t("g₁ > 0 的一側", "g₁ > 0 side"), False, 34) + chip(470, 118, 210, "violet", t("g₂ > 0 的一側", "g₂ > 0 side"), False, 34)
    o += txt(470, 168, 214, t("箭頭指向正側；兩色重疊處：兩條線都在正側", "Arrows point to the positive side; where the colours overlap both lines are positive"), 16, TXT, False, "left", 96)
    o += svg(0, 0, W, H, mark(ax, 0, 0, "A", 0) .replace("", "")) if False else ""
    o += f'<div style="position:absolute;left:472px;top:290px;width:26px;height:26px;border-radius:13px;background:{AF};border:2px solid #fff"></div>' + txt(506, 290, 180, t("A：兩輸入不同", "A: inputs differ"), 16, TXT, False, "left", 26)
    o += f'<div style="position:absolute;left:472px;top:330px;width:26px;height:26px;border-radius:4px;background:{BF};border:2px solid #fff"></div>' + txt(506, 330, 180, t("B：兩輸入相同", "B: inputs equal"), 16, TXT, False, "left", 26)
    o += cardH(716, 0, 444, 480, "orange", t("階梯規則：正側記 1，否則記 0", "Step rule: positive side gives 1, otherwise 0"))
    o += card(730, 52, 416, 112, "orange", .16) + formula(730, 58, 416, r"y_j=1,\quad g_j(\mathbf{x})>0", 24, 44) + formula(730, 108, 416, r"y_j=0,\quad g_j(\mathbf{x})\leq 0", 24, 44)
    o += named(730, 178, 416, t("階梯規則：分數為正記 1，否則記 0（剛好等於 0 也記 0）。", "Step rule: a positive score gives 1, otherwise 0 (a score of exactly 0 also gives 0)."), "term:" + t("階梯規則", "Step rule"), 16, TXT, 72)
    o += named(730, 262, 416, t("位元：階梯規則給出的 0 或 1。", "Bit: the 0 or 1 that the step rule gives."), "term:" + t("位元", "Bit"), 16, TXT, 48)
    o += card(730, 330, 416, 134, "yellow", .14) + txt(738, 338, 400, t("每個點得到兩個位元 (y₁, y₂)：每條線一個。下一頁用 (0, 1) 走一遍。", "Every point gets two bits (y₁, y₂), one per line. The next page walks (0, 1) through."), 17, TXT, True, "center", 110)
    return o
# ------------------------------------------------ 3: walk (0,1)
def page3(lang):
    t = tr(lang); o = ""; a, b = 0, 1
    o += cardH(0, 0, 700, 480, "blue", t("以 (0, 1) 為例走一遍", "Walk the point (0, 1) through"))
    o += chip(200, 50, 300, "grey", "x = (x₁, x₂) = (0, 1)", True, 44, 20)
    s = arrow(300, 96, 190, 128, G1C, 3, 11) + arrow(400, 96, 510, 128, G2C, 3, 11)
    for i, (role, col, tex, g, y) in enumerate((("teal", G1C, r"g_1=0+1-0.5=0.5", N.g1(a, b), N.code(a, b)[0]), ("violet", G2C, r"g_2=0+1-1.5=-0.5", N.g2(a, b), N.code(a, b)[1]))):
        x0 = 20 + i * 350; o += card(x0, 134, 320, 150, role, .2) + head(x0, 134, 320, role, t("第一條線", "First line") if i == 0 else t("第二條線", "Second line"), 34, 17)
        o += formula(x0, 180, 320, tex, 24, 44)
        o += txt(x0, 232, 320, ("0.5 > 0 → " if g > 0 else M + "0.5 ≤ 0 → ") + t("正側", "positive side") if g > 0 else M + "0.5 ≤ 0 → " + t("不在正側", "not positive"), 17, TXT, True, "center", 28)
        s += arrow(x0 + 160, 286, x0 + 160, 318, col, 3, 11)
        o += chip(x0 + 60, 322, 200, role, t(f"階梯 → y{'₁' if i == 0 else '₂'} = {y}", f"step → y{'₁' if i == 0 else '₂'} = {y}"), True, 44, 20)
    s += arrow(180, 368, 280, 396, "#b7b0a5", 3, 11) + arrow(530, 368, 420, 396, "#b7b0a5", 3, 11)
    o += svg(0, 0, W, H, s)
    o += chip(150, 398, 400, "yellow", t("隱藏編碼 y = [y₁, y₂] = (1, 0)", "Hidden code y = [y₁, y₂] = (1, 0)"), True, 46, 21)
    o += txt(20, 450, 660, t("流程由上而下：輸入 → 兩個分數 → 階梯規則 → 隱藏編碼", "Top to bottom: input → two scores → step rule → hidden code"), 16, MUT, False, "center", 26)
    o += cardH(716, 0, 444, 480, "yellow", t("名詞與位置", "Term and position"))
    o += named(730, 52, 416, t("隱藏編碼：兩個位元連起來的 [y₁, y₂]，像替點貼上兩張位置標籤。", "Hidden code: the two bits written together as [y₁, y₂], like two position labels attached to a point."), "term:" + t("隱藏編碼", "Hidden code"), 16, TXT, 72)
    ax = Axes(800, 140, 250, 250, XR, XR)
    s = poly_svg(ax, region(ax, [(1, 1, 0.5, True), (1, 1, 1.5, False)]), BL, .28) + axes_svg(ax, [0, 1], [0, 1], "x₁", "x₂", 16) + line_svg(ax, 1, 1, 0.5, G1C, 3) + line_svg(ax, 1, 1, 1.5, G2C, 3)
    for (p, q) in N.PTS: s += mark(ax, p, q, N.label(p, q), 11)
    s += f'<polygon points="{ax.X(0)-2:.1f},{ax.Y(1)-18:.1f} {ax.X(0)+4:.1f},{ax.Y(1)-6:.1f} {ax.X(0)+18:.1f},{ax.Y(1)-6:.1f} {ax.X(0)+7:.1f},{ax.Y(1)+3:.1f} {ax.X(0)+11:.1f},{ax.Y(1)+18:.1f} {ax.X(0)-2:.1f},{ax.Y(1)+9:.1f} {ax.X(0)-15:.1f},{ax.Y(1)+18:.1f} {ax.X(0)-11:.1f},{ax.Y(1)+3:.1f} {ax.X(0)-22:.1f},{ax.Y(1)-6:.1f} {ax.X(0)-8:.1f},{ax.Y(1)-6:.1f}" fill="#e0c85f"/>' if False else ""
    s += ring(ax, 0, 1, 22, "#e0c85f")
    o += svg(0, 0, W, H, s)
    o += txt(730, 428, 416, t("黃圈：點 (0, 1) 落在兩條線之間，編碼 (1, 0)", "Yellow ring: (0, 1) is between the<br>two lines, code (1, 0)"), 16, TXT, False, "center", 48)
    return o
# ------------------------------------------------ 4: table + bands
def page4(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 570, 480, "blue", t("四個點的分數與編碼", "Scores and codes of the four points"))
    cols = [(14, 100, t("輸入", "input")), (118, 86, "g₁"), (208, 86, "g₂"), (298, 110, t("編碼", "code")), (412, 146, t("XOR 真值", "XOR class"))]
    for x0, w_, lab in cols: o += txt(x0, 48, w_, lab, 16, MUT, True, "center", 24)
    for i, r in enumerate(N.table()):
        y0 = 80 + i * 62; role = "blue" if r["truth"] == "A" else "orange"; o += card(10, y0, 550, 54, role, .18)
        a, b = r["x"]; c = r["code"]
        o += txt(14, y0 + 13, 100, f"({a}, {b})", 18, TXT, True, "center", 28) + txt(118, y0 + 13, 86, sg(r["g1"]), 18, TXT, False, "center", 28) + txt(208, y0 + 13, 86, sg(r["g2"]), 18, TXT, False, "center", 28)
        o += txt(298, y0 + 13, 110, f"({c[0]}, {c[1]})", 18, TXT, True, "center", 28) + chip(442, y0 + 11, 90, role, r["truth"], True, 32, 18)
    o += card(10, 338, 550, 126, "yellow", .14) + txt(18, 346, 470, t("(0, 1) 與 (1, 0) 得到同一個編碼 (1, 0)：兩個 A 合併成一個位置，三種編碼依序是 (0, 0)、(1, 0)、(1, 1)。", "(0, 1) and (1, 0) get the same code (1, 0): the two A points merge into one position, and the three codes are (0, 0), (1, 0), (1, 1)."), 17, TXT, True, "center", 108)
    o += cardH(586, 0, 574, 480, "violet", t("兩條線把平面切成三條帶", "Two lines cut the plane into three bands"))
    ax = Axes(620, 56, 380, 380, XR, XR)
    bands = [([(1, 1, 0.5, False)], OR, .30, "(0, 0)"), ([(1, 1, 0.5, True), (1, 1, 1.5, False)], BL, .36, "(1, 0)"), ([(1, 1, 1.5, True)], OR, .30, "(1, 1)")]
    s = ""
    for cuts, col, op, lab in bands: s += poly_svg(ax, region(ax, cuts), col, op)
    s += axes_svg(ax, [0, 1], [0, 1], "x₁", "x₂") + line_svg(ax, 1, 1, 0.5, G1C, 4) + line_svg(ax, 1, 1, 1.5, G2C, 4)
    for (p, q) in N.PTS: s += mark(ax, p, q, N.label(p, q))
    o += svg(0, 0, W, H, s)
    o += txt(ax.X(-0.37), ax.Y(-0.12), 120, "(0, 0)", 20, "#f0a98b", True, "left", 28) + txt(ax.X(0.3), ax.Y(0.62), 120, "(1, 0)", 20, "#9cc3ea", True, "left", 28) + txt(ax.X(1.0) - 4, ax.Y(1.24), 120, "(1, 1)", 20, "#f0a98b", True, "left", 28)
    o += f'<div style="position:absolute;left:1030px;top:70px;width:26px;height:26px;border-radius:13px;background:{AF};border:2px solid #fff"></div>' + txt(1064, 70, 90, "A", 17, TXT, True, "left", 26)
    o += f'<div style="position:absolute;left:1030px;top:110px;width:26px;height:26px;border-radius:4px;background:{BF};border:2px solid #fff"></div>' + txt(1064, 110, 90, "B", 17, TXT, True, "left", 26)
    o += txt(1020, 170, 132, t("兩個 A 都在中間那條帶 (1, 0)", "Both A points sit in the middle band (1, 0)"), 16, TXT, False, "left", 120)
    return o
# ------------------------------------------------ 5: hidden plane
def page5(lang):
    t = tr(lang); o = ""
    HX = (-0.5, 1.5)
    o += cardH(0, 0, 660, 480, "orange", t("隱藏空間：每個點換成它的編碼 (y₁, y₂)", "Hidden space: each point becomes its code (y₁, y₂)"))
    ax = Axes(100, 60, 380, 380, HX, HX)
    s = poly_svg(ax, region(ax, [(1, -1, 0.5, True)]), BL, .26) + axes_svg(ax, [0, 1], [0, 1], "y₁", "y₂") + line_svg(ax, 1, -1, 0.5, "#e0c85f", 4)
    s += mark(ax, 0, 0, "B") + mark(ax, 1, 0, "A", 17) + mark(ax, 1, 1, "B") + mark(ax, 0, 1, "A", 14, True)
    o += svg(0, 0, W, H, s)
    o += txt(ax.X(0) + 18, ax.Y(0) - 30, 140, "(0, 0)  B", 17, TXT, True, "left", 26) + txt(ax.X(1) + 30, ax.Y(0) - 36, 90, "(1, 0)  A", 17, TXT, True, "left", 26) + txt(ax.X(1) - 82, ax.Y(1) - 40, 140, "(1, 1)  B", 17, TXT, True, "left", 26)
    o += txt(ax.X(0) - 20, ax.Y(1) + 22, 190, t("(0, 1) 沒有輸入", "(0, 1): no input"), 16, MUT, False, "left", 26)
    o += txt(ax.X(0.05), ax.Y(0.65) + 50, 60, "", 16)
    o += txt(500, 120, 150, t("輸出線<br>y₂ = y₁ − 0.5", "output line<br>y₂ = y₁ − 0.5"), 17, "#e0c85f", True, "left", 52)
    o += txt(500, 200, 150, t("下方：判 A", "below: A"), 17, "#9cc3ea", True, "left", 26) + txt(500, 232, 150, t("上方：判 B", "above: B"), 17, "#f0a98b", True, "left", 26)
    o += txt(500, 372, 150, t("(1, 0) 是兩個輸入 (0, 1) 與 (1, 0) 合併後的位置", "(1, 0) is where the inputs (0, 1) and (1, 0) merged"), 16, MUT, False, "left", 120)
    o += cardH(676, 0, 484, 480, "yellow", t("輸出層在隱藏空間裡畫一條線", "The output layer draws one line in the hidden space"))
    o += named(690, 52, 456, t("隱藏空間：把每個點換成它的編碼 (y₁, y₂) 之後所在的新平面。", "Hidden space: the new plane in which each point sits at its code (y₁, y₂)."), "term:" + t("隱藏空間", "Hidden space"), 16, TXT, 72)
    o += card(690, 134, 456, 78, "yellow", .16) + formula(690, 144, 456, r"g_{\text{out}}=y_1-y_2-0.5", 26, 58)
    o += card(690, 226, 456, 86, "grey", .12) + formula(690, 234, 456, r"g_{\text{out}}=0\ \Rightarrow\ y_2=y_1-0.5", 24, 70)
    o += card(690, 326, 456, 86, "grey", .12) + formula(690, 334, 456, r"g_{\text{out}}>0\Leftrightarrow  y_2<y_1-0.5", 24, 70)
    o += txt(690, 420, 456, t("在線的下方判 A。兩個 A 合併不要緊，因為它們同類。", "Below the line means A. Merging the two A points is harmless: they are the same class."), 17, TXT, True, "center", 52)
    return o
# ------------------------------------------------ 6: output scores
def page6(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 600, 480, "orange", t("輸出分數 = y₁ − y₂ − 0.5", "Output score = y₁ − y₂ − 0.5"))
    rows = [((0, 0), r"0-0-0.5=-0.5"), ((1, 0), r"1-0-0.5=0.5"), ((1, 1), r"1-1-0.5=-0.5")]
    o += txt(14, 48, 110, t("編碼", "code"), 16, MUT, True, "center", 24) + txt(140, 48, 280, t("輸出分數", "output score"), 16, MUT, True, "center", 24) + txt(450, 48, 130, t("判決", "verdict"), 16, MUT, True, "center", 24)
    for i, (c, tex) in enumerate(rows):
        y0 = 80 + i * 100; v = N.gout(*c); pos = v > 0; role = "blue" if pos else "orange"
        o += card(10, y0, 580, 88, role, .18) + chip(20, y0 + 22, 100, "grey", f"({c[0]}, {c[1]})", False, 44, 20) + formula(130, y0 + 14, 300, tex, 24, 58)
        o += chip(450, y0 + 22, 120, role, t("判 A", "A") if pos else t("判 B", "B"), True, 44, 20)
    o += txt(14, 392, 572, t("只有 (1, 0) 的分數為正", "Only (1, 0) has a positive score"), 18, TXT, True, "center", 28)
    o += txt(14, 424, 572, t("分數為正判 A，為負判 B", "A positive score means A, a negative score means B"), 16, MUT, False, "center", 26)
    o += cardH(616, 0, 544, 480, "green", t("四個輸入逐點核對", "Check the four inputs one by one"))
    for xx, lab, w_ in ((628, t("輸入", "input"), 90), (722, t("編碼", "code"), 90), (816, t("分數", "score"), 80), (900, t("預測", "predicted"), 112), (1016, t("真值", "truth"), 120)): o += txt(xx, 48, w_, lab, 16, MUT, True, "center", 24)
    for i, r in enumerate(N.table()):
        y0 = 80 + i * 58; c = r["code"]; v = N.gout(*c); p = N.pred(*r["x"]); role = "blue" if p == "A" else "orange"; o += card(626, y0, 522, 50, role, .18)
        a, b = r["x"]
        o += txt(628, y0 + 11, 90, f"({a}, {b})", 17, TXT, True, "center", 28) + txt(722, y0 + 11, 90, f"({c[0]}, {c[1]})", 17, TXT, False, "center", 28) + txt(816, y0 + 11, 80, sg(v), 17, TXT, False, "center", 28)
        o += chip(918, y0 + 9, 76, role, p, True, 32, 18) + txt(1010, y0 + 11, 60, r["truth"], 17, TXT, True, "center", 28) + chip(1080, y0 + 9, 56, "green", "✓", True, 32, 18)
    vec = ", ".join(str(1 if N.pred(*p) == "A" else 0) for p in N.PTS)
    o += card(626, 322, 522, 66, "green", .2) + txt(634, 332, 506, t(f"預測 ({vec}) 剛好是 XOR，四點全對", f"Predictions ({vec}) are exactly XOR: all four right"), 19, TXT, True, "center", 46)
    o += card(626, 398, 522, 66, "yellow", .14) + txt(634, 406, 506, t("這只是這四點的驗算，不是新資料上的準確率。（A = 1，B = 0）", "This only checks these four points; it is not accuracy on new data. (A = 1, B = 0)"), 16, TXT, False, "center", 50)
    return o
# ------------------------------------------------ 7: strip in original plane
def page7(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 600, 480, "blue", t("回到原平面：整個網路選出一條帶", "Back in the original plane: the network selects a strip"))
    ax = Axes(80, 56, 380, 380, XR, XR)
    s = poly_svg(ax, region(ax, [(1, 1, 0.5, True), (1, 1, 1.5, False)]), BL, .36) + axes_svg(ax, [0, 1], [0, 1], "x₁", "x₂") + line_svg(ax, 1, 1, 0.5, G1C, 4) + line_svg(ax, 1, 1, 1.5, G2C, 4)
    for (p, q) in N.PTS: s += mark(ax, p, q, N.label(p, q))
    o += svg(0, 0, W, H, s)
    o += txt(ax.X(0.3), ax.Y(0.62), 160, t("帶內：判 A", "inside: A"), 18, "#9cc3ea", True, "left", 28) + txt(ax.X(-0.37), ax.Y(-0.1), 150, t("帶外：判 B", "outside: B"), 18, "#f0a98b", True, "left", 28) + txt(ax.X(0.95), ax.Y(1.28), 150, t("帶外：判 B", "outside: B"), 18, "#f0a98b", True, "left", 28)
    o += txt(476, 150, 118, "x₁ + x₂ = 0.5", 16, G1C, True, "left", 24) + txt(476, 190, 118, "x₁ + x₂ = 1.5", 16, "#c4b0e6", True, "left", 24)
    o += txt(476, 250, 118, t("兩個 A 在帶內，兩個 B 在帶外", "Two A inside, two B outside"), 16, TXT, False, "left", 96)
    o += cardH(616, 0, 544, 480, "violet", t("隱藏層做兩個判斷，輸出層說「兩個都成立」", "Hidden layer: two judgments; output layer: both hold"))
    o += card(630, 52, 516, 66, "teal", .2) + formula(630, 60, 516, r"y_1=1\Leftrightarrow  x_1+x_2>0.5", 24, 50)
    o += card(630, 128, 516, 66, "violet", .2) + formula(630, 136, 516, r"y_2=0\Leftrightarrow  x_1+x_2\leq 1.5", 24, 50)
    o += card(630, 204, 516, 66, "orange", .2) + formula(630, 212, 516, r"g_{\text{out}}>0\Leftrightarrow  (y_1,y_2)=(1,0)", 24, 50)
    o += card(630, 286, 516, 76, "yellow", .16) + formula(630, 296, 516, r"0.5<x_1+x_2\leq 1.5", 28, 56)
    o += txt(630, 366, 516, t("代入核對：x₁ + x₂ = 1 在帶內（兩個 A）；0 在帶下方、2 在帶上方（兩個 B）。〔程式計算〕", "Check: x₁ + x₂ = 1 is inside the strip (the two A points); 0 lies below and 2 above it (the two B points). (computed)"), 16, TXT, False, "center", 72)
    o += txt(630, 420, 516, t("這就是 KP01 的帶：兩條平行線夾出的區域", "This is the strip of KP01:<br>the region between two parallel lines"), 17, TXT, True, "center", 52)
    return o
# ------------------------------------------------ 8: merging
def page8(lang):
    t = tr(lang); o = ""
    def pane(x0, role, title, ins, code, verdict_role, verdict, ok):
        r = card(x0, 0, 570, 380, role) + head(x0, 0, 570, role, title, 36, 18); s = ""
        for i, (lab, cls) in enumerate(ins):
            yy = 100 + i * 100; r += chip(x0 + 20, yy, 150, "blue" if cls == "A" else "orange", lab, False, 48, 19); s += arrow(x0 + 172, yy + 24, x0 + 258, 188 if i == 0 else 210, "#b7b0a5", 3, 12)
        r += chip(x0 + 262, 160, 120, "grey", code, True, 72, 22) + txt(x0 + 262, 236, 120, t("同一個編碼", "same code"), 16, MUT, False, "center", 26)
        s += arrow(x0 + 384, 196, x0 + 430, 196, "#b7b0a5", 3, 12)
        r += chip(x0 + 436, 160, 114, verdict_role, verdict, True, 72, 21)
        r += txt(x0 + 14, 296, 542, ok, 17, TXT, True, "center", 76)
        return r + svg(0, 0, W, H, s)
    o += pane(0, "green", t("同類合併：沒關係（本例）", "Same-class merge: harmless (this example)"), [("(0, 1)  A", "A"), ("(1, 0)  A", "A")], "(1, 0)", "blue", t("判 A", "A"), t("兩個都是 A，輸出層把 (1, 0) 判成 A，兩點都對。", "Both are A; the output layer calls (1, 0) A, so both are right."))
    o += pane(590, "red", t("異類合併：就糟了（示意）", "Different-class merge: it fails (schematic)"), [("A", "A"), ("B", "B")], "(1, 1)", "red", t("必錯一個", "1 wrong"), t("同一個編碼只能得到一個判決，A 與 B 至少錯一個；後面怎麼畫線都救不回來。", "One code can get only one verdict, so A or B is wrong; no later line can repair it."))
    o += cardH(0, 396, 780, 84, "orange", "") if False else ""
    o += card(0, 394, 580, 78, "orange", .14) + named(12, 400, 556, t("編碼衝突：不同類的點得到同一個隱藏編碼，<br>輸出層怎麼畫線都分不開。", "Code conflict: points of different classes get the<br>same hidden code, and no output line can<br>separate them."), "term:" + t("編碼衝突", "Code conflict"), 16, TXT, 72)
    o += card(590, 394, 570, 78, "yellow", .14) + txt(598, 400, 554, t("y₁、y₂ 只是「在正側嗎」的位元，<br>不是類別的機率；先檢查隱藏編碼能否被輸出層分開。", "y₁, y₂ are only “positive side?” bits,<br>not class probabilities; first check that the<br>output layer can separate the codes."), 16, TXT, False, "center", 72)
    return o
# ------------------------------------------------ 9: exercise, bias +0.5 (Q1)
def page9(lang):
    t = tr(lang); o = ""
    pr, wrong = N.stats(0.5)
    o += card(0, 0, 640, 470, "blue") + head(0, 0, 640, "blue", t("小練習 1：把輸出分數的常數改成 +0.5", "Exercise 1: change the output constant to +0.5"), 36, 18)
    o += txt(14, 46, 612, t("題目：保留隱藏編碼，輸出分數改成 g(out) = y₁ − y₂ + 0.5。四個輸入的預測是什麼？", "Question: keep the hidden codes and set g(out) = y₁ − y₂ + 0.5. What are the four predictions?"), 17, TXT, True, "left", 78)
    o += chip(14, 130, 130, "grey", t("解題示範", "Solution"), False, 30, 16)
    o += txt(154, 130, 470, t("① 符號式　② 代入三個位置　③ 看正負", "① formula　② substitute　③ read signs"), 16, MUT, False, "left", 30)
    o += formula(14, 166, 612, r"g_{\text{out}}=y_1-y_2+0.5", 24, 40)
    for i, c in enumerate(((0, 0), (1, 0), (1, 1))):
        v = N.gout(*c, bias=0.5); x0 = 14 + i * 206
        o += card(x0, 214, 198, 80, "blue", .16) + txt(x0, 218, 198, f"({c[0]}, {c[1]})", 16, MUT, True, "center", 24) + txt(x0, 246, 198, f"{c[0]}−{c[1]}+0.5 = {v:g}".replace("−", "−"), 18, TXT, True, "center", 30)
    o += txt(14, 304, 612, t("三個分數都為正，四個輸入全判 A", "All three scores are positive, so all four inputs are called A"), 18, TXT, True, "center", 28)
    o += txt(14, 344, 100, t("核對", "Check"), 16, MUT, True, "left", 24)
    for i, r in enumerate(N.table()):
        x0 = 14 + i * 154; ok = pr[i] == r["truth"]; o += card(x0, 374, 146, 84, "green" if ok else "red", .22)
        a, b = r["x"]; o += txt(x0, 378, 146, f"({a}, {b})", 16, MUT, True, "center", 24) + txt(x0, 402, 146, f"{t('真值', 'truth')} {r['truth']}", 16, TXT, False, "center", 24) + txt(x0, 428, 146, f"{t('預測', 'pred.')} {pr[i]} " + ("✓" if ok else "✗"), 17, TXT, True, "center", 26)
    o += cardH(656, 0, 504, 470, "yellow", t("輸出線平移：y₂ = y₁ + 0.5", "The output line shifts: y₂ = y₁ + 0.5"))
    HX = (-0.5, 1.5); ax = Axes(716, 56, 330, 330, HX, HX)
    s = poly_svg(ax, region(ax, [(1, -1, -0.5, True)]), BL, .26) + axes_svg(ax, [0, 1], [0, 1], "y₁", "y₂") + line_svg(ax, 1, -1, -0.5, "#e0c85f", 4)
    s += mark(ax, 0, 0, "B") + mark(ax, 1, 0, "A", 17) + mark(ax, 1, 1, "B") + mark(ax, 0, 1, "A", 14, True)
    s += ring(ax, 0, 0, 24) + ring(ax, 1, 1, 24)
    o += svg(0, 0, W, H, s)
    o += txt(1086, 100, 70, t("線上方判 B", "above: B"), 16, "#f0a98b", True, "left", 52) + txt(1086, 250, 70, t("線下方判 A", "below: A"), 16, "#9cc3ea", True, "left", 52)
    o += txt(1086, 330, 70, t("紅圈：判錯", "red ring: wrong"), 16, "#f0a98b", True, "left", 52)
    o += txt(668, 418, 480, t(f"三個位置全在正側，兩個 B 被判成 A，錯 {wrong} 點。〔程式計算〕", f"All three positions are on the positive side; the two B points are called A: {wrong} wrong. (computed)"), 16, TXT, False, "center", 48)
    return o
# ------------------------------------------------ 10: exercise Q2 (merging under bias +0.5)
def page10(lang):
    t = tr(lang); o = ""
    o += card(0, 0, 560, 470, "red") + head(0, 0, 560, "red", t("小練習 2：誰被合併？為什麼沒有衝突？", "Exercise 2: who is merged, and why no conflict?"), 36, 18)
    o += txt(14, 46, 532, t("題目：哪些不同的輸入被合併？為什麼這次合併不造成衝突？（提示：看它們的標籤）", "Question: which different inputs are merged, and why does this merge cause no conflict? (Hint: look at their labels.)"), 17, TXT, True, "left", 78)
    o += chip(14, 130, 130, "grey", t("解題示範", "Solution"), False, 30, 16)
    steps = [(t("① 被合併的輸入", "① The merged inputs"), t("(0, 1) 與 (1, 0)，隱藏編碼都是 (1, 0)", "(0, 1) and (1, 0); both have hidden code (1, 0)")),
             (t("② 為什麼沒有衝突", "② Why no conflict"), t("兩點的真值都是 A；同類的點合併，輸出層把這個編碼判成 A 就對了", "Both are class A; merging same-class points is<br>harmless, the output layer just calls this code A")),
             (t("③ 核對", "③ Check"), t("錯的是 (0, 0) 與 (1, 1) 兩個 B 被判成 A；原因是偏置把線平移，不是合併", "The errors are the two B points (0, 0) and (1, 1)<br>called A; the cause is the shifted line, not the merge"))]
    for i, (a, b) in enumerate(steps):
        y0 = 172 + i * 96; o += card(14, y0, 532, 88, "grey", .1) + txt(24, y0 + 6, 512, a, 16, MUT, True, "left", 24) + txt(24, y0 + 32, 512, b, 16, TXT, False, "left", 50)
    o += cardH(576, 0, 584, 470, "orange", t("四個輸入 → 三個位置 → 判決（常數 +0.5）", "Four inputs → three positions → verdicts (constant +0.5)"))
    ins = [((0, 0), 62), ((0, 1), 142), ((1, 0), 222), ((1, 1), 302)]
    pos = {(0, 0): 100, (1, 0): 182, (1, 1): 264}
    s = ""
    for (c, yy) in ins:
        cc = N.code(*c); r = N.label(*c); o += chip(590, yy, 100, "blue" if r == "A" else "orange", f"({c[0]}, {c[1]}) {r}", False, 44, 17)
        s += arrow(692, yy + 22, 790, pos[cc] + 38 if True else 0, "#b7b0a5", 2.5, 10)
    for c, yy in pos.items():
        v = N.gout(*c, bias=0.5)
        o += chip(792, yy + 14, 100, "grey", f"({c[0]}, {c[1]})", True, 48, 19) + txt(900, yy + 22, 140, f"g(out) = {v:g}", 17, TXT, True, "left", 28) + chip(1042, yy + 14, 100, "blue", t("判 A", "A"), True, 48, 19)
    s += ""
    o += svg(0, 0, W, H, s)
    o += txt(590, 370, 556, t("(0, 0) 與 (1, 1) 的真值是 B，卻被判 A：兩處錯誤，不是合併造成的。", "(0, 0) and (1, 1) are class B but called A:<br>two errors, not caused by the merge."), 16, TXT, False, "center", 50)
    o += ring_html()
    o += txt(590, 430, 556, t("兩個 B 判錯，所以 XOR 錯 2 點。〔程式計算〕", "Two B points wrong, so XOR gets 2 points wrong. (computed)"), 16, MUT, False, "center", 26)
    return o
def ring_html(): return ""
# ------------------------------------------------ 11: no step
def page11(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 620, 470, "red", t("去掉階梯：線性式的組合還是線性式", "Remove the step: a combination of linear forms is still linear"))
    o += txt(14, 46, 592, t("沒有階梯時，y₁ 與 y₂ 直接是分數：", "Without the step, y₁ and y₂ are just the scores:"), 17, TXT, False, "left", 28)
    o += card(14, 82, 592, 66, "teal", .16) + formula(14, 90, 592, r"y_1=x_1+x_2-0.5,\quad y_2=x_1+x_2-1.5", 22, 50)
    o += card(14, 158, 592, 66, "orange", .16) + formula(14, 166, 592, r"g_{\text{out}}=y_1-y_2-0.5", 24, 50)
    o += card(14, 234, 592, 82, "yellow", .16) + formula(14, 244, 592, r"=(x_1+x_2-0.5)-(x_1+x_2-1.5)-0.5=0.5", 22, 62)
    o += chip(14, 332, 592, "red", t("0.5 與 x 無關：每個點都判 A", "0.5 does not depend on x: every point is A"), True, 50, 21)
    o += txt(14, 396, 592, t("線性式的組合還是線性式；非線性要靠階梯。", "A combination of linear forms is still linear;<br>the nonlinearity comes from the step."), 17, TXT, True, "center", 52)
    o += cardH(636, 0, 524, 470, "blue", t("原平面：整個平面預測 A", "Original plane: the whole plane predicts A"))
    ax = Axes(680, 56, 320, 320, XR, XR)
    s = f'<rect x="{ax.x}" y="{ax.y}" width="{ax.w}" height="{ax.h}" fill="{BL}" fill-opacity="0.30"/>' + axes_svg(ax, [0, 1], [0, 1], "x₁", "x₂")
    for (p, q) in N.PTS: s += mark(ax, p, q, N.label(p, q))
    s += ring(ax, 0, 0, 24) + ring(ax, 1, 1, 24)
    o += svg(0, 0, W, H, s)
    o += txt(1032, 90, 120, t("整個平面一種顏色：A", "one colour everywhere: A"), 16, "#9cc3ea", True, "left", 76)
    o += txt(1032, 200, 120, t("紅圈：兩個 B 判錯", "red rings: the two B points are wrong"), 16, "#f0a98b", True, "left", 76)
    o += txt(650, 412, 496, t("四個點的 g(out) 都是 0.5；XOR 錯 2 點。〔程式計算〕", "g(out) is 0.5 at all four points;<br>XOR gets 2 wrong. (computed)"), 16, TXT, False, "center", 26)
    return o
PAGES_DESIGN = [(f"p{i}", globals()[f"page{i}"]) for i in range(1, 12)]
if __name__ == "__main__":
    out = sys.argv[1]; only = sys.argv[2:]
    for fid, fn in PAGES_DESIGN:
        if only and fid not in only: continue
        print(fid); build("kp02_" + fid, fn, W, H, out)
