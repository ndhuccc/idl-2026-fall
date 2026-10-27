"""KP13 pages designed in HTML (view-from-text-v3, route A). pageN(lang) -> body HTML of a 1160x480 figure."""
import os, sys, math
import numpy as np
SK = "/home/user/pr-2026-fall/.agents/skills"
HERE = os.path.dirname(os.path.abspath(__file__)); HERE = HERE[:-5] if HERE.endswith(("/W_zh", "/W_en")) else HERE
sys.path.insert(0, SK + "/view-from-text-v3/examples"); sys.path.insert(0, SK + "/view-from-text-v3/tools"); sys.path.insert(0, HERE)
from kp17_pages import Axes, tr, take, legend, fmt, sci, txt, named, card, head, chip, formula, svg, BL, RD, VI, GR, YE, OR, TE, INK, TXT, MUT, hexc, build, native, _sup, rgba, rgb
from kp15_pages import box, cardH
import kp13_nums as K
W, H = 1160, 480
CA, CB = "#4f86bf", "#c4572e"
CAt, CBt = "#7fb0e6", "#e8896a"   # lighter versions for text on the dark canvas   # class A blue, class B orange

# ------------------------------------------------ 1: the sandwiched classes
def page1(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 760, 480, "blue", t("一維數線上的七個點與最好的單一門檻", "Seven points on a line and the best single threshold"))
    ax = Axes(30, 70, 700, 330, (-4, 4), (0, 1)); th = 1.5
    ax.rect(th, 4, 0, 1, "#6a9fd6", .12); ax.rect(-4, th, 0, 1, "#d97757", .10)
    ax.raw(f'<line x1="{ax.X(-4)}" y1="{ax.Y(.45)}" x2="{ax.X(4)}" y2="{ax.Y(.45)}" stroke="#b7b0a5" stroke-width="2"/>')
    for v in range(-3, 4): ax.raw(f'<line x1="{ax.X(v)}" y1="{ax.Y(.45)-6}" x2="{ax.X(v)}" y2="{ax.Y(.45)+6}" stroke="#b7b0a5" stroke-width="2"/><text x="{ax.X(v)}" y="{ax.Y(.45)+32}" text-anchor="middle" font-size="16" fill="#b7b0a5">{v}</text>')
    ax.raw(f'<line x1="{ax.X(th)}" y1="{ax.Y(.05)}" x2="{ax.X(th)}" y2="{ax.Y(.95)}" stroke="#e0c85f" stroke-width="3" stroke-dasharray="9 7"/>')
    ax.text(th, .95, t("門檻 x = 1.5", "threshold x = 1.5"), "#e0c85f", "middle", 17, True, 0, -10)
    ax.text(2.8, .8, t("判 A", "call A"), CAt, "middle", 18, True); ax.text(-1.2, .8, t("判 B", "call B"), CBt, "middle", 18, True)
    for v, ca in [(x, 1) for x in K.XA] + [(x, 0) for x in K.XB]:
        pred_a = v > th; wrong = (pred_a != bool(ca)); col = CA if ca else CB
        ax.dot(v, .45, col, 15)
        if wrong: ax.raw(f'<circle cx="{ax.X(v):.1f}" cy="{ax.Y(.45):.1f}" r="23" fill="none" stroke="#e0584f" stroke-width="4"/>')
        ax.text(v, .45, "A" if ca else "B", "#f2ede4", "middle", 16, True, 0, 6)
    o += ax.html()
    o += card(40, 408, 680, 52, "yellow", .14) + txt(48, 418, 664, t("紅圈是分錯的 2 個點：分對 5 筆，分錯 2 筆", "Red rings mark the 2 mistakes: 5 correct, 2 wrong"), 18, TXT, True, "center", 30)
    o += cardH(776, 0, 384, 480, "orange", t("為什麼分不開", "Why one threshold fails"))
    o += txt(792, 52, 352, t("類 A 在 x = −3、−2、2、3，類 B 在 x = −1、0、1，共 7 筆。", "Class A sits at x = −3, −2, 2, 3 and class B at x = −1, 0, 1: 7 points."), 17, TXT, False, "left", 76)
    o += txt(792, 140, 352, t("程式窮舉所有門檻（含方向），最多分對 5 筆。", "A program tries every threshold (with either direction): at most 5 are right."), 17, TXT, False, "left", 76)
    o += card(792, 232, 352, 70, "red", .22) + txt(800, 240, 336, t("A 在 B 的兩側", "A lies on both sides of B"), 19, TXT, True, "center", 30) + txt(800, 270, 336, t("只看 x 的大小，看不出「在兩邊」", "x alone cannot say “on both sides”"), 16, MUT, False, "center", 26)
    o += card(792, 318, 352, 130, "yellow", .14) + txt(800, 328, 336, t("接下來的問題", "The question"), 16, MUT, True, "center", 24) + txt(800, 358, 336, t("能不能換一個座標，讓兩類變成「一邊一個」？〔程式計算〕", "Can a new coordinate put one class on each side? (computed)"), 18, TXT, True, "center", 84)
    return o

# ------------------------------------------------ 2: lift to (x, x^2)
def page2(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 420, 480, "teal", t("逐點算 y = (x, x²)", "Point by point: y = (x, x²)"))
    for xx, lab, w_ in ((16, t("類別", "class"), 90), (110, "x", 90), (206, "y₂ = x²", 190)): o += txt(xx, 48, w_, lab, 17, MUT, True, "center", 26)
    pts = [(x, 1) for x in K.XA] + [(x, 0) for x in K.XB]
    for i, (x, ca) in enumerate(pts):
        y0 = 78 + i * 36; o += card(16, y0, 388, 30, "blue" if ca else "orange", .16)
        o += txt(16, y0 + 2, 90, "A" if ca else "B", 17, TXT, True, "center", 26) + txt(110, y0 + 2, 90, f"{x:g}", 17, TXT, False, "center", 26) + txt(206, y0 + 2, 190, f"{x*x:g}", 17, TXT, True, "center", 26)
    o += named(16, 340, 388, t("特徵映射：用一組函數 fᵢ 把 x 換成新的座標 y = (f₁(x), …, f_k(x))。", "Feature mapping: use functions fᵢ to replace x by new coordinates y = (f₁(x), …, f_k(x))."), "term:" + t("特徵映射", "Feature mapping"), 16, TXT, 100)
    o += cardH(436, 0, 724, 480, "blue", t("升到 (x, x²) 之後，一條水平線就分開", "After lifting to (x, x²), one horizontal line separates"))
    ax = Axes(500, 64, 640, 300, (-4, 4), (-0.8, 10.5)); ax.frame([-3, -2, -1, 0, 1, 2, 3], [0, 2.5, 5, 7.5, 10], yfmt=lambda v: f"{v:g}")
    ax.rect(-4, 4, 2.5, 10.5, "#6a9fd6", .12); ax.rect(-4, 4, -0.8, 2.5, "#d97757", .10)
    ax.line([(-4, 2.5), (4, 2.5)], "#e0c85f", 4)
    ax.text(3.6, 2.5, "y₂ = 2.5", "#e0c85f", "end", 17, True, 0, -10)
    ax.text(-3.6, 9.4, t("A 在上方", "A above"), CAt, "start", 17, True); ax.text(-3.6, 1.7, t("B 在下方", "B below"), CBt, "start", 17, True)
    for x, ca in pts:
        ax.dot(x, x * x, CA if ca else CB, 11); ax.text(x, x * x, "A" if ca else "B", "#f2ede4", "middle", 16, True, 0, 6)
    o += ax.html() + txt(500, 400, 640, t("原座標 x", "original coordinate x"), 16, MUT, False, "center", 24)
    o += card(450, 440, 696, 32, "yellow", .14) + txt(458, 444, 680, t("A 的 y₂ 都不小於 4，B 的都不大於 1：水平線 y₂ = 2.5 分開兩類", "A has y₂ ≥ 4, B has y₂ ≤ 1: the line y₂ = 2.5 separates them"), 16, TXT, True, "center", 24)
    return o

# ------------------------------------------------ 3: back to x it is curved
def page3(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 640, 480, "blue", t("回到 x 看，邊界是彎的：g(x) = −2.5 + x²", "Back in x the boundary bends: g(x) = −2.5 + x²"))
    ax = Axes(66, 66, 560, 330, (-3.8, 3.8), (-3.4, 7.2)); ax.frame([-3, -2, -1, 0, 1, 2, 3], [-2, 0, 2, 4, 6], yfmt=lambda v: f"{v:g}")
    r = math.sqrt(2.5)
    ax.rect(-3.8, -r, 0, 7.2, "#6a9fd6", .10); ax.rect(r, 3.8, 0, 7.2, "#6a9fd6", .10); ax.rect(-r, r, -3.4, 0, "#d97757", .10)
    xs = np.linspace(-3.15, 3.15, 90); ax.line([(float(x), float(-2.5 + x * x)) for x in xs], "#e0c85f", 4)
    ax.line([(-3.8, 0), (3.8, 0)], "#b7b0a5", 2, "6 5", .8)
    for x, ca in [(x, 1) for x in K.XA] + [(x, 0) for x in K.XB]:
        ax.dot(x, 0, CA if ca else CB, 11); ax.text(x, 0, "A" if ca else "B", "#f2ede4", "middle", 16, True, 0, 6)
    for sx in (-r, r): ax.dot(sx, 0, "#f2ede4", 6)
    ax.text(r, 0, f"x = {r:.2f}", "#f2ede4", "start", 16, True, 12, 28); ax.text(-r, 0, f"x = −{r:.2f}", "#f2ede4", "end", 16, True, -12, 28)
    ax.text(0, -2.1, t("g < 0：B", "g < 0: B"), CBt, "middle", 17, True); ax.text(3.0, 3.6, t("g > 0：A", "g > 0: A"), CAt, "middle", 17, True); ax.text(-3.0, 3.6, t("g > 0：A", "g > 0: A"), CAt, "middle", 17, True)
    o += ax.html()
    o += cardH(656, 0, 504, 480, "violet", t("廣義線性分類器", "Generalized linear classifier"))
    o += named(672, 48, 472, t("廣義線性分類器：先用固定的非線性函數 fᵢ 升維，再設計普通的線性分類器；「線性」是指對權重 wᵢ 線性。", "Generalized linear classifier: lift with fixed nonlinear functions fᵢ, then design an ordinary linear classifier; “linear” means linear in the weights wᵢ."), "term:" + t("廣義線性分類器", "Generalized linear classifier"), 16, TXT, 100)
    o += card(672, 158, 472, 66, "violet", .22) + formula(672, 164, 472, r"g(x)=w_0+\sum_{i=1}^{k}w_if_i(x)", 26, 52)
    steps = [t("f₁ = x、f₂ = x²，水平線對應 w = (−2.5, 0, 1)", "f₁ = x, f₂ = x²; the line gives w = (−2.5, 0, 1)"), t("g(x) = −2.5 + x²，g > 0 判 A，g < 0 判 B", "g(x) = −2.5 + x²: g > 0 is A, g < 0 is B"), t("g = 0 解得 x = ±√2.5 ≈ ±1.58", "g = 0 gives x = ±√2.5 ≈ ±1.58")]
    for i, s_ in enumerate(steps):
        y0 = 238 + i * 44; o += card(672, y0, 472, 38, "grey", .12) + txt(680, y0 + 5, 456, f"{i+1}. " + s_, 16, TXT, False, "left", 28)
    o += card(672, 378, 472, 76, "yellow", .14) + txt(680, 386, 456, t("新空間的一條直線，在原空間是兩個點：邊界在 x 軸上是「彎」的。", "One line in the new space is two points in the original: the boundary bends along x."), 17, TXT, True, "center", 60)
    return o

# ------------------------------------------------ 4: XOR again
def page4(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 380, 480, "red", t("XOR：原座標分不開", "XOR: not separable as is"))
    ax = Axes(60, 70, 290, 270, (-0.4, 1.4), (-0.4, 1.4)); ax.frame([0, 1], [0, 1], yfmt=lambda v: f"{v:g}")
    for (a, b), ca in (((0, 1), 1), ((1, 0), 1), ((0, 0), 0), ((1, 1), 0)):
        ax.dot(a, b, CA if ca else CB, 16); ax.text(a, b, "A" if ca else "B", "#f2ede4", "middle", 16, True, 0, 6)
    ax.text(1.38, -0.3, "x₁", "#b7b0a5", "end", 16, False, 0, -8); ax.text(-0.36, 1.36, "x₂", "#b7b0a5", "start", 16, False, 4, 4)
    o += ax.html()
    o += txt(14, 396, 352, t("兩個座標不同為一類：沒有任何一條直線分得開。", "A class is “the two coordinates differ”: no line separates it."), 16, TXT, False, "center", 60)
    o += cardH(396, 0, 420, 480, "blue", t("加一個座標 y₃ = x₁x₂", "Add one coordinate y₃ = x₁x₂"))
    ax = Axes(462, 70, 330, 270, (-0.3, 2.3), (-0.3, 1.3)); ax.frame([0, 1, 2], [0, 1], yfmt=lambda v: f"{v:g}")
    ax.rect(-0.3, 2.3, -0.3, -0.3, "#000000", 0)
    ax.line([(-0.3, -0.4), (2.3, 0.9)], "#e0c85f", 4)
    ax.raw(f'<polygon points="{ax.X(-0.3):.1f},{ax.Y(-0.4):.1f} {ax.X(2.3):.1f},{ax.Y(0.9):.1f} {ax.X(2.3):.1f},{ax.Y(-0.3):.1f} {ax.X(-0.3):.1f},{ax.Y(-0.3):.1f}" fill="#6a9fd6" opacity="0.12"/>')
    for (s, y3), ca in (((1, 0), 1), ((0, 0), 0), ((2, 1), 0)):
        ax.dot(s, y3, CA if ca else CB, 16)
    ax.text(1, 0, "A", "#f2ede4", "middle", 16, True, 0, 6); ax.text(0, 0, "B", "#f2ede4", "middle", 16, True, 0, 6); ax.text(2, 1, "B", "#f2ede4", "middle", 16, True, 0, 6)
    ax.text(1.0, 0.0, "×2", "#f2ede4", "middle", 16, True, 0, -22)
    o += ax.html() + txt(462, 376, 330, t("橫軸 x₁ + x₂，縱軸 y₃", "horizontal x₁ + x₂, vertical y₃"), 16, MUT, False, "center", 24)
    o += txt(410, 408, 392, t("側面看：(0,1) 與 (1,0) 重疊在 (1, 0)，一條線把 A 與 B 分開。", "Side view: (0,1) and (1,0) overlap at (1, 0); one line separates A from B."), 16, TXT, False, "center", 60)
    o += cardH(832, 0, 328, 480, "teal", t("取 g = x₁ + x₂ − 2y₃ − 0.5", "Take g = x₁ + x₂ − 2y₃ − 0.5"))
    for xx, lab, w_ in ((842, "(x₁, x₂)", 100), (946, "y₃", 60), (1010, "g", 140)): o += txt(xx, 48, w_, lab, 16, MUT, True, "center", 24)
    for i, (p, y3, g) in enumerate((("(0, 0)", 0, -0.5), ("(0, 1)", 0, 0.5), ("(1, 0)", 0, 0.5), ("(1, 1)", 1, -0.5))):
        y0 = 78 + i * 46; ca = g > 0; o += card(842, y0, 308, 40, "blue" if ca else "orange", .2)
        o += txt(842, y0 + 7, 100, p, 17, TXT, False, "center", 26) + txt(946, y0 + 7, 60, str(y3), 17, TXT, False, "center", 26) + txt(1010, y0 + 7, 130, f"{g:+.1f}", 18, TXT, True, "center", 26)
    o += txt(842, 276, 308, t("g > 0 為 A，g < 0 為 B：四點全對。〔程式計算〕", "g > 0 is A, g < 0 is B: all four right. (computed)"), 16, TXT, True, "center", 52)
    o += card(842, 340, 308, 112, "yellow", .14) + txt(850, 348, 292, t("KP02 用兩條線再合成；這裡是一個固定的映射加一個平面。", "KP02 combined two lines; here it is one fixed mapping plus one plane."), 16, TXT, False, "center", 96)
    return o

# ------------------------------------------------ 5: weights from the normal equation
def page5(lang):
    t = tr(lang); o = ""; w, g = K.lstsq_1d()
    o += cardH(0, 0, 600, 480, "violet", t("正規方程：讓平方誤差最小", "The normal equation: least squared error"))
    o += card(24, 52, 552, 78, "violet", .22) + formula(24, 60, 552, r"w=(\Phi^{T}\Phi)^{-1}\Phi^{T}t", 30, 62)
    o += txt(24, 142, 552, t("Φ 的每一列是一個樣本的 (1, x, x²)；t 中 A 為 +1、B 為 −1。", "Each row of Φ is (1, x, x²); t is +1 for A, −1 for B."), 16, TXT, False, "left", 52)
    for xx, lab, w_ in ((44, "1", 70), (124, "x", 70), (204, "x²", 70), (300, "t", 70)): o += txt(xx, 200, w_, lab, 17, MUT, True, "center", 26)
    for i, (x, tt) in enumerate(((-3, "+1"), (-2, "+1"), (None, "⋮"), (1, "−1"))):
        y0 = 228 + i * 38
        if x is None: o += txt(44, y0, 230, "⋮", 20, MUT, True, "center", 28) + txt(300, y0, 70, "⋮", 20, MUT, True, "center", 28)
        else: o += txt(44, y0, 70, "1", 17, TXT, False, "center", 28) + txt(124, y0, 70, f"{x}", 17, TXT, False, "center", 28) + txt(204, y0, 70, f"{x*x}", 17, TXT, False, "center", 28) + txt(300, y0, 70, tt, 17, TXT, True, "center", 28)
    o += card(24, 392, 552, 66, "yellow", .16) + formula(24, 398, 552, r"w=(-0.810,\ 0,\ 0.238)", 28, 54)
    o += cardH(616, 0, 544, 480, "teal", t("七個點的 g 值", "g at the seven points"))
    for xx, lab, w_ in ((630, "x", 60), (696, t("類別", "class"), 70), (770, "t", 60), (836, "g", 100), (944, t("正負號", "sign"), 190)): o += txt(xx, 48, w_, lab, 16, MUT, True, "center", 24)
    order = [(x, 1) for x in K.XA] + [(x, 0) for x in K.XB]
    for i, ((x, ca), gv) in enumerate(zip(order, g)):
        y0 = 76 + i * 36; ok = (gv > 0) == bool(ca); o += card(628, y0, 520, 31, "green" if ok else "red", .16)
        o += txt(630, y0 + 3, 60, f"{x:g}", 17, TXT, False, "center", 26) + txt(696, y0 + 3, 70, "A" if ca else "B", 17, TXT, True, "center", 26) + txt(770, y0 + 3, 60, "+1" if ca else "−1", 17, TXT, False, "center", 26)
        o += txt(836, y0 + 3, 100, f"{gv:+.3f}", 17, TXT, True, "center", 26) + txt(944, y0 + 3, 190, t("與目標同號 ✓", "same sign ✓"), 16, TXT, False, "center", 26)
    o += txt(628, 336, 520, t("七個 g 的正負都和目標相同，七筆全對。〔程式計算〕", "All seven signs match the targets: 7 of 7 right. (computed)"), 17, TXT, True, "center", 52)
    o += card(628, 396, 520, 66, "yellow", .14) + txt(636, 404, 504, t("與手設的 (−2.5, 0, 1) 不同；兩組都分對，最小二乘追求平方誤差最小，不是邊界離最遠。", "It differs from the hand-set (−2.5, 0, 1); both classify all right: least squares minimizes squared error, not the margin."), 16, TXT, False, "center", 52)
    return o

# ------------------------------------------------ 6: concentric rings
def page6(lang):
    t = tr(lang); o = ""; X, tt, rA, rB = K.rings(); fit, r, mA, mB = K.ring_fit()
    o += cardH(0, 0, 560, 480, "blue", t("同心環：兩類各 100 點與決策圓", "Concentric rings: 100 points per class and the decision circle"))
    ax = Axes(30, 56, 500, 400, (-3.1, 3.1), (-3.1, 3.1) if False else (-3.1, 3.1))
    ax.raw(f'<circle cx="{ax.X(0):.1f}" cy="{ax.Y(0):.1f}" r="{(ax.X(r) - ax.X(0)):.1f}" fill="none" stroke="#7dbb8c" stroke-width="5"/>')
    for (a, b), ca in zip(X, tt > 0): ax.raw(f'<circle cx="{ax.X(a):.1f}" cy="{ax.Y(b):.1f}" r="5" fill="{CA if ca else CB}" opacity="0.9"/>')
    ax.text(0, 0, f"r ≈ {r:.2f}", "#7dbb8c", "middle", 18, True, 0, 6)
    o += ax.html()
    o += cardH(576, 0, 584, 480, "orange", t("兩種特徵的正確率", "Accuracy of two feature sets"))
    for xx, lab, w_ in ((590, t("特徵", "features"), 250), (846, t("個數", "count"), 60), (912, t("答對／200", "correct"), 110), (1030, t("正確率", "accuracy"), 110)): o += txt(xx, 48, w_, lab, 16, MUT, True, "center", 24)
    rows = [("[1, x₁, x₂]", 3, fit["lin"][0]), ("[1, x₁, x₂, x₁², x₂², x₁x₂]", 6, fit["quad"][0])]
    for i, (nm, kk, c) in enumerate(rows):
        y0 = 74 + i * 46; hot = i == 1; o += card(588, y0, 560, 40, "green" if hot else "grey", .24 if hot else .1)
        o += txt(590, y0 + 7, 250, nm, 16, TXT, False, "center", 26) + txt(846, y0 + 7, 60, str(kk), 17, TXT, False, "center", 26) + txt(912, y0 + 7, 110, str(c), 17, TXT, False, "center", 26) + txt(1030, y0 + 7, 110, f"{c/2:.1f}%", 18, TXT, True, "center", 26)
    o += card(588, 168, 560, 70, "yellow", .14) + txt(596, 176, 544, t(f"二次特徵的 g = 0 在原空間是半徑約 {r:.2f} 的圓，落在兩圈（平均半徑約 {mA:.2f} 與 {mB:.2f}）之間。〔程式計算〕", f"With quadratic features g = 0 is a circle of radius about {r:.2f}, between the two rings (mean radii about {mA:.2f} and {mB:.2f}). (computed)"), 16, TXT, False, "center", 56)
    o += txt(588, 248, 560, t("二次特徵的個數 k = l(l + 3)/2", "Number of quadratic features k = l(l + 3)/2"), 17, TXT, True, "center", 28)
    o += txt(600, 296, 180, t("輸入維度 l", "input dimension l"), 16, MUT, True, "center", 24) + txt(600, 332, 180, t("特徵個數 k", "features k"), 16, MUT, True, "center", 24)
    for xx, l_ in ((800, 2), (920, 10), (1040, 100)):
        o += txt(xx - 8, 284, 116, str(l_), 17, TXT, True, "center", 26)
        o += card(xx - 8, 312, 116, 58, "teal", .2) + txt(xx - 8, 316, 116, f"{l_}×{l_+3}÷2", 16, MUT, False, "center", 24) + txt(xx - 8, 338, 116, f"= {l_*(l_+3)//2:,}", 18, TXT, True, "center", 28)
    o += txt(588, 380, 560, t("在同一份訓練資料上的正確率，只說明「能不能表示」，不是泛化能力。", "Accuracy on the same training data only shows what can be represented, not generalization."), 16, MUT, False, "center", 48)
    o += txt(588, 428, 560, t("常見迷思：升維後邊界變成直線。其實是新空間的平面；回到原空間是圓。", "Myth: after lifting the boundary is a line. It is a plane in the new space and a circle in the original."), 16, TXT, True, "center", 48)
    return o

# ------------------------------------------------ 7: multi-class
MC = ["#6a9fd6", "#e0c85f", "#d97757"]
def page7(lang):
    t = tr(lang); o = ""; x, cls, res = K.multi(); W3 = res["quad"][1]
    o += cardH(0, 0, 700, 480, "blue", t("三個權重向量，取 g 最大的類", "Three weight vectors: pick the class with the largest g"))
    ax = Axes(66, 60, 610, 300, (-3.5, 3.5), (-0.5, 1.45)); ax.frame([-3, -2, -1, 0, 1, 2, 3], [0, .5, 1], yfmt=lambda v: f"{v:g}")
    xs = np.linspace(-3.5, 3.5, 140); G = np.c_[np.ones(140), xs, xs ** 2] @ W3; am = G.argmax(1)
    for i in range(len(xs) - 1): ax.rect(float(xs[i]), float(xs[i + 1]), -0.4, -0.25, MC[am[i]], .55)
    for c in range(3): ax.line([(float(a), float(min(max(b, -0.5), 1.45))) for a, b in zip(xs, G[:, c])], MC[c], 4)
    for xv, c in zip(x, cls): ax.dot(float(xv), -0.14, MC[c], 8)
    o += ax.html() + txt(66, 440, 610, t("曲線：各類的 g；色帶：最高的類；圓點：九個樣本", "Curves: g per class. Strip: highest class. Dots: samples"), 16, MUT, False, "center", 24)
    o += legend([(MC[0], t("類 1", "class 1"))], 120, 404, 120) + legend([(MC[1], t("類 2", "class 2"))], 300, 404, 120) + legend([(MC[2], t("類 3", "class 3"))], 480, 404, 120)
    o += cardH(716, 0, 444, 480, "orange", t("做法與結果", "Method and result"))
    st = [t("類 1：x = −3、−2.5、−2；類 2：−0.5、0、0.5；類 3：2、2.5、3", "Class 1: x = −3, −2.5, −2; class 2: −0.5, 0, 0.5; class 3: 2, 2.5, 3"), t("目標用 one-hot：類 r 的目標向量只有第 r 個是 1", "One-hot targets: only the r-th entry of class r is 1"), t("特徵 (1, x, x²)，最小二乘解三組權重", "Features (1, x, x²); least squares gives three weight sets")]
    for i, s_ in enumerate(st):
        y0 = 52 + i * 72; o += card(732, y0, 412, 66, "grey", .12) + txt(740, y0 + 8, 396, f"{i+1}. " + s_, 16, TXT, False, "left", 54)
    o += card(732, 276, 412, 52, "green", .24) + txt(740, 286, 396, t(f"(1, x, x²)：九筆答對 {res['quad'][0]} 筆", f"(1, x, x²): {res['quad'][0]} of 9 right"), 19, TXT, True, "center", 32)
    o += card(732, 338, 412, 52, "red", .24) + txt(740, 348, 396, t(f"只用 (1, x)：只對 {res['lin'][0]} 筆", f"Only (1, x): {res['lin'][0]} of 9 right"), 19, TXT, True, "center", 32)
    o += txt(732, 398, 412, t("中間類只剩 x = 0 判對：該點三個 g 並列為 1/3，取中間類。〔程式計算〕", "The middle class keeps only x = 0: its three g tie at 1/3 and the middle class is taken. (computed)"), 16, MUT, False, "left", 72)
    return o

# ------------------------------------------------ 8: projection pursuit
def blk(x, y, w, h, c1, c2, label, lab2=None):
    o = box(x, y, w, h, c1, c2, 135, shadow=False, r=12) + txt(x, y + (14 if lab2 else h // 2 - 14), w, label, 17, TXT, True, "center", 28)
    if lab2: o += txt(x, y + 44, w, lab2, 16, TXT, False, "center", 26)
    return o
def arrow(x1, x2, y): return f'<line x1="{x1}" y1="{y}" x2="{x2-8}" y2="{y}" stroke="#b7b0a5" stroke-width="3"/><polygon points="{x2},{y} {x2-12},{y-7} {x2-12},{y+7}" fill="#b7b0a5"/>'
def page8(lang):
    t = tr(lang); o = ""; s2 = float(K.sig(2)); s1x2 = float(2 * K.sig(1)); a = ""
    o += cardH(0, 0, 760, 480, "violet", t("兩種結構", "Two structures"))
    o += txt(16, 48, 728, t("廣義線性：固定的映射 + 可學的線性層", "Generalized linear: fixed mapping + learnable linear layer"), 17, TXT, True, "left", 28)
    o += blk(20, 90, 120, 76, "#5a544b", "#3a352d", "x") + blk(200, 90, 200, 76, "#4f86bf", "#2f4f7a", t("fᵢ 固定", "fᵢ fixed"), t("事先決定", "chosen in advance")) + blk(460, 90, 140, 76, "#4f9a68", "#2c6140", t("w 可學", "w learned"), None) + blk(650, 90, 90, 76, "#5a544b", "#3a352d", "g")
    a += arrow(140, 200, 128) + arrow(400, 460, 128) + arrow(600, 650, 128)
    o += txt(16, 200, 728, t("投影追蹤：g(x) = Σ fᵢ(wᵢᵀx)", "Projection pursuit: g(x) = Σ fᵢ(wᵢᵀx)"), 17, TXT, True, "left", 28)
    o += blk(20, 242, 120, 76, "#5a544b", "#3a352d", "x") + blk(200, 242, 200, 76, "#4f9a68", "#2c6140", t("方向 wᵢ 可學", "direction wᵢ learned"), t("取投影 wᵢᵀx", "projection wᵢᵀx")) + blk(460, 242, 140, 76, "#d9822b", "#a8452b", t("fᵢ 要配", "fᵢ fitted"), None) + blk(650, 242, 90, 76, "#5a544b", "#3a352d", "g")
    a += arrow(140, 200, 280) + arrow(400, 460, 280) + arrow(600, 650, 280)
    o += svg(0, 0, W, H, a)
    o += card(20, 352, 720, 100, "yellow", .14) + txt(28, 360, 704, t("怎麼記：廣義線性 = 固定的映射 + 可學的線性層；多層感知器 = 可學的映射 + 可學的線性層。", "To remember: generalized linear = fixed mapping + learnable linear layer; multilayer perceptron = learnable mapping + learnable linear layer."), 17, TXT, True, "center", 84)
    o += cardH(776, 0, 384, 480, "orange", t("加倍測試：把 w 加倍", "Doubling test: double w"))
    o += card(792, 52, 352, 100, "green", .2) + txt(800, 60, 336, t("廣義線性", "Generalized linear"), 17, TXT, True, "center", 28) + txt(800, 92, 336, t("g 剛好加倍：g(2w) = 2g(w)", "g exactly doubles: g(2w) = 2g(w)"), 17, TXT, False, "center", 52)
    o += card(792, 164, 352, 140, "red", .22) + txt(800, 172, 336, t("投影追蹤（f 為 sigmoid、輸入 1、w = 1）", "Projection pursuit (f sigmoid, input 1, w = 1)"), 16, TXT, True, "center", 52) + txt(800, 228, 336, f"f(2) = {s2:.3f}", 20, TXT, True, "center", 30) + txt(800, 262, 336, t(f"不是 2f(1) = {s1x2:.3f}", f"not 2f(1) = {s1x2:.3f}"), 18, TXT, False, "center", 28)
    o += txt(792, 320, 352, t("所以它對 w 不是線性的，要用兩階段交替最佳化。若把 fᵢ 事先固定為 sigmoid，就是單隱藏層神經網路，實務上已被多層感知器取代。〔程式計算〕", "So it is not linear in w and needs two-stage alternating optimization. Fixing fᵢ as sigmoids gives a one-hidden-layer network, which multilayer perceptrons have replaced in practice. (computed)"), 16, MUT, False, "left", 140)
    return o

# ------------------------------------------------ 9: exercises
def page9(lang):
    t = tr(lang); o = ""
    qs = [("blue", t("題 1", "Q1"), t("特徵 y = (x₁², x₂²)，g = y₁ + y₂ − 4（g > 0 為 A）：判 (1,1)、(3,0)、(0,−2.5) 各屬哪一類？", "y = (x₁², x₂²), g = y₁ + y₂ − 4 (g > 0 is A):<br>classes of (1,1), (3,0), (0,−2.5)?"), [r"(1,1):\ 1+1-4=-2<0\ \Rightarrow\ \mathrm{B}", r"(3,0):\ 9+0-4=5>0\ \Rightarrow\ \mathrm{A}", r"(0,-2.5):\ 6.25-4=2.25>0\ \Rightarrow\ \mathrm{A}"], t("(1,1) 屬 B，(3,0) 與 (0,−2.5) 屬 A。", "(1,1) is B; (3,0) and (0,−2.5) are A."), t("核對：三個 g 的正負逐一代入。", "Check: substitute each point and read the sign.")),
          ("red", t("題 2", "Q2"), t("g = 0 在 x 平面上是什麼圖形？", "What shape is g = 0 in the x plane?"), [r"g=0\ \Rightarrow\ x_1^2+x_2^2=4", r"\text{radius}=\sqrt{4}=2"], t("半徑 2 的圓。", "A circle of radius 2."), t("核對：在 y 空間是一條直線，在 x 空間是彎的。", "Check: a straight line in y space, bent in x space."))]
    for i, (role, name, q, tex, ans, chk) in enumerate(qs):
        x = i * 590; o += card(x, 0, 570, 470, role) + head(x, 0, 570, role, name, 36, 18)
        o += txt(x + 16, 52, 538, q, 19, TXT, True, "left", 84)
        o += chip(x + 16, 150, 130, "grey", t("解題示範", "Solution"), False, 30, 16)
        o += card(x + 16, 194, 538, 150, "grey", .1)
        for j, tx in enumerate(tex): o += formula(x + 16, 200 + j * (150 // max(len(tex), 1)) - (0 if len(tex) == 3 else -16 if len(tex) == 2 else 0), 538, tx, 22 if len(tex) == 3 else 24, 50)
        o += chip(x + 16, 364, 538, role, ans, True, 42, 20)
        o += txt(x + 16, 420, 538, chk, 17, MUT, False, "left", 48)
    return o

PAGES_DESIGN = [(f"p{i}", globals()[f"page{i}"]) for i in range(1, 10)]
if __name__ == "__main__":
    out = sys.argv[1]; only = sys.argv[2:]
    for fid, fn in PAGES_DESIGN:
        if only and fid not in only: continue
        print(fid); build("kp13_" + fid, fn, W, H, out)
