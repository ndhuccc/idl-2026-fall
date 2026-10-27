"""KP14 pages designed in HTML (view-from-text-v3, route A). pageN(lang) -> body HTML of a 1160x480 figure."""
import os, sys, math, itertools
import numpy as np
SK = "/home/user/pr-2026-fall/.agents/skills"
HERE = os.path.dirname(os.path.abspath(__file__)); HERE = HERE[:-5] if HERE.endswith(("/W_zh", "/W_en")) else HERE
sys.path.insert(0, SK + "/view-from-text-v3/examples"); sys.path.insert(0, SK + "/view-from-text-v3/tools"); sys.path.insert(0, HERE)
from kp17_pages import Axes, tr, take, legend, fmt, sci, txt, named, card, head, chip, formula, svg, BL, RD, VI, GR, YE, OR, TE, INK, TXT, MUT, hexc, build, native, _sup, rgba, rgb
from kp15_pages import box, cardH
import kp14_nums as K
W, H = 1160, 480
_EXP = None
def exp():
    global _EXP
    if _EXP is None: _EXP = K.experiments()
    return _EXP
def dich():
    """All 16 labelings of the square corners A(top-left) C(top-right) B(bottom-left) D(bottom-right); True if linearly separable."""
    pts = np.array([[0, 1], [1, 1], [0, 0], [1, 0]], float); out = []
    for lab in itertools.product([-1., 1.], repeat=4):
        y = np.array(lab); out.append((lab, K.separable(pts, y)))
    return out

# ------------------------------------------------ 1: 16 labelings
def page1(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 760, 480, "blue", t("正方形四個角的 16 種標法", "The 16 labelings of the four corners of a square"))
    d = dich(); s = ""
    for i, (lab, ok) in enumerate(d):
        cx = 16 + (i % 4) * 184; cy = 48 + (i // 4) * 108; role = "green" if ok else "red"
        o += card(cx, cy, 172, 96, role, .2 if ok else .3)
        for (px, py), l in zip(((0, 0), (1, 0), (0, 1), (1, 1)), lab):
            x_ = cx + 30 + px * 50; y_ = cy + 24 + py * 48; s += f'<circle cx="{x_}" cy="{y_}" r="11" fill="{"#6a9fd6" if l > 0 else "#d97757"}"/>'
        if not ok: o += txt(cx + 100, cy + 34, 66, "XOR" if lab[0] == lab[3] and lab[0] != lab[1] else "XNOR", 16, TXT, True, "center", 24)
    o += svg(0, 0, W, H, s)
    o += cardH(776, 0, 384, 480, "orange", t("名詞", "Terms"))
    o += named(792, 56, 352, t("二分：把 N 個點分成兩類的一種標法；一條線（超平面）做得到的，叫線性二分。", "Dichotomy: one way of labeling N points with two classes; if a line (hyperplane) can do it, it is a linear dichotomy."), "term:" + t("二分", "Dichotomy"), 16, TXT, 100)
    o += card(792, 170, 352, 60, "green", .2) + txt(800, 184, 336, t("綠框可分：14 種", "Green, separable: 14"), 19, TXT, True, "center", 30)
    o += card(792, 240, 352, 60, "red", .3) + txt(800, 254, 336, t("紅框不可分：2 種", "Red, not separable: 2"), 19, TXT, True, "center", 30)
    o += txt(792, 316, 352, t("不可分的是「對角同類」：左上與右下同類（XOR），或把兩類名稱對調（XNOR）。", "The two that fail pair the diagonal corners: top-left with bottom-right (XOR), or the same with the class names swapped (XNOR)."), 16, MUT, False, "left", 96)
    o += card(792, 410, 352, 54, "yellow", .14) + txt(800, 418, 336, t("3 點：2³ = 8 種全可分；4 點：2⁴ = 16 種只剩 14 種", "3 points: all 2³ = 8; 4 points: only 14 of 2⁴ = 16"), 16, TXT, True, "center", 40)
    return o

# ------------------------------------------------ 2: Cover formula
def page2(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 600, 480, "violet", t("Cover 公式：一次算出可分的標法數", "Cover's formula: count the separable labelings in one step"))
    o += card(24, 52, 552, 96, "violet", .22) + formula(24, 62, 552, r"O(N,l)=2\sum_{i=0}^{l}\binom{N-1}{i}", 28, 76)
    o += named(24, 160, 552, t("一般位置：l 維空間的點，沒有任何 l + 1 個落在同一個 (l − 1) 維超平面上；二維就是任三點不共線。", "General position: no l + 1 points on one<br>(l − 1)-D hyperplane; in 2-D, no three are collinear."), "term:" + t("一般位置", "General position"), 16, TXT, 76)
    o += card(24, 246, 552, 76, "grey", .12) + formula(24, 252, 552, r"N=4,\ l=2:\ \ O=2(1+3+3)=14", 24, 64)
    o += card(24, 330, 552, 76, "grey", .12) + formula(24, 336, 552, r"N=3,\ l=2:\ \ O=2(1+2+1)=8=2^{3}", 24, 64)
    o += txt(24, 418, 552, t("一般地，N ≤ l + 1 時 O = 2ᴺ：全部標法都可分。", "In general, N ≤ l + 1 gives O = 2ᴺ: every labeling is separable."), 17, TXT, True, "center", 28)
    o += cardH(616, 0, 544, 480, "teal", t("l = 2、N = 1 到 8", "l = 2, N = 1 to 8"))
    for xx, lab, w_ in ((632, "N", 70), (712, "O(N, 2)", 190), (912, "2ᴺ", 220)): o += txt(xx, 48, w_, lab, 17, MUT, True, "center", 26)
    for i, n in enumerate(range(1, 9)):
        y0 = 80 + i * 40; hot = n == 4; o += card(628, y0, 520, 34, "yellow" if hot else "grey", .26 if hot else .1)
        o += txt(632, y0 + 3, 70, str(n), 18, TXT, True, "center", 28) + txt(712, y0 + 3, 190, str(K.O(n, 2)), 18, TXT, True, "center", 28) + txt(912, y0 + 3, 220, str(2 ** n), 18, MUT, False, "center", 28)
    o += txt(628, 412, 520, t("N = 4：16 種標法中 14 種可分，缺的就是 XOR 與 XNOR。〔程式計算〕", "N = 4: 14 of the 16 labelings are separable; the missing two are XOR and XNOR. (computed)"), 16, TXT, False, "center", 48)
    return o

# ------------------------------------------------ 3: probability, l = 2
def page3(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 700, 480, "blue", t("l = 2：可分機率對點數 N", "l = 2: separability probability against N"))
    ax = Axes(70, 70, 600, 330, (0.5, 10.5), (0, 1)); ax.frame(list(range(1, 11)), [0, .25, .5, .75, 1], xlab=None, yfmt=lambda v: f"{v:g}")
    pts = [(n, K.P(n, 2)) for n in range(1, 11)]
    ax.line([(0.5, .5), (10.5, .5)], "#e0c85f", 2, "8 6", .8)
    ax.line(pts, "#6a9fd6", 4)
    for n, p in pts: ax.dot(n, p, "#6a9fd6", 7)
    ax.dot(6, .5, "#e0c85f", 10); ax.text(6, .5, t("N = 6：0.5", "N = 6: 0.5"), "#e0c85f", "start", 17, True, 14, -14)
    ax.text(2, 1, t("N ≤ 3：機率 1", "N ≤ 3: P = 1"), "#f2ede4", "middle", 16, True, 0, -12)
    o += ax.html() + txt(70, 446, 600, t("點數 N", "Number of points N"), 16, MUT, False, "center", 24)
    o += cardH(716, 0, 444, 480, "orange", t("可分機率", "Separability probability"))
    o += card(732, 52, 412, 76, "orange", .2) + formula(732, 62, 412, r"P_N^l=O(N,l)/2^{N}", 26, 56)
    o += txt(732, 138, 412, t("隨機貼標籤時，2ᴺ 種標法機會均等；N ≤ l + 1 時 P = 1。", "With random labels all 2ᴺ labelings are equally likely; N ≤ l + 1 gives P = 1."), 16, MUT, False, "left", 72)
    for i, n in enumerate((4, 5, 6, 10)):
        y0 = 222 + i * 52; hot = n == 6; o += card(732, y0, 412, 44, "yellow" if hot else "grey", .26 if hot else .1)
        o += txt(740, y0 + 8, 130, f"N = {n}", 18, TXT, True, "left", 28) + txt(880, y0 + 8, 256, f"P = {K.P(n, 2):.4g}", 18, TXT, True, "right", 28)
    o += txt(732, 436, 412, t("N 一超過 l + 1 = 3，機率就開始掉。〔程式計算〕", "N above l + 1 = 3: the probability falls. (computed)"), 16, TXT, False, "center", 24)
    return o

# ------------------------------------------------ 4: phase transition
LCOL = {1: "#7dbb8c", 3: "#5fb8a8", 9: "#6a9fd6", 19: "#a78bd6", 49: "#e0c85f", 99: "#d97757"}
def page4(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 720, 480, "blue", t("P 對 r = N / (l + 1)：所有曲線都過 (2, ½)", "P against r = N / (l + 1): every curve passes (2, ½)"))
    ax = Axes(60, 64, 640, 262, (0, 3.5), (0, 1)); ax.frame([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5], [0, .5, 1], yfmt=lambda v: f"{v:g}")
    for l, col in LCOL.items():
        pts = [(n / (l + 1), K.P(n, l)) for n in range(1, int(3.5 * (l + 1)) + 1)]
        ax.line(pts, col, 3.5 if l in (1, 99) else 3)
    ax.dot(2, .5, "#f2ede4", 8)
    o += ax.html() + legend([(c, f"l = {l}") for l, c in list(LCOL.items())[:3]], 90, 382, 140) + legend([(c, f"l = {l}") for l, c in list(LCOL.items())[3:]], 250, 382, 140)
    o += txt(452, 390, 258, t("l 越大，曲線越陡，像一道台階，台階永遠在 r = 2。", "The larger l, the steeper: a step that always sits at r = 2."), 16, TXT, True, "left", 72)
    o += cardH(736, 0, 424, 480, "orange", t("相變與為什麼是一半", "Phase transition and why one half"))
    o += named(752, 46, 392, t("相變：機率從 1 掉到 0 的轉折；l 越大轉折越陡，像水結成冰。", "Phase transition: P drops from 1 to 0; the larger l, the sharper the drop."), "term:" + t("相變", "Phase transition"), 16, TXT, 70)
    steps = [t("N = 2l + 2，所以 N − 1 = 2l + 1", "N = 2l + 2, so N − 1 = 2l + 1"), t("C(2l+1, i) = C(2l+1, 2l+1−i)：前後對稱", "C(2l+1, i) = C(2l+1, 2l+1−i), symmetric"), t("i ≤ l 的和恰為 2²ˡ，全部 2²ˡ⁺¹ 的一半", "The sum over i ≤ l is 2²ˡ, half of 2²ˡ⁺¹"), t("O = 2·2²ˡ = 2²ˡ⁺¹，所以 P = 1/2", "O = 2·2²ˡ = 2²ˡ⁺¹, so P = 1/2")]
    for i, s_ in enumerate(steps):
        y0 = 122 + i * 46; o += card(752, y0, 392, 40, "grey", .12) + txt(760, y0 + 6, 380, f"{i+1}. " + s_, 16, TXT, False, "left", 30)
    o += txt(752, 312, 392, t("r = 1.5 與 2.5 的 P：", "P at r = 1.5 and 2.5:"), 16, MUT, True, "left", 24)
    for xx, lab, w_ in ((752, "l", 70), (830, "r = 1.5", 150), (990, "r = 2.5", 150)): o += txt(xx, 338, w_, lab, 16, MUT, True, "center", 22)
    for i, l in enumerate((1, 19, 99)):
        y0 = 362 + i * 34; o += card(752, y0, 392, 30, "grey", .1)
        o += txt(752, y0 + 3, 70, str(l), 16, TXT, True, "center", 24) + txt(830, y0 + 3, 150, f"{K.P(round(1.5*(l+1)), l):.2f}", 16, TXT, False, "center", 24) + txt(990, y0 + 3, 150, f"{K.P(round(2.5*(l+1)), l):.2g}" if K.P(round(2.5*(l+1)), l) < .001 else f"{K.P(round(2.5*(l+1)), l):.2f}", 16, TXT, False, "center", 24)
    return o

# ------------------------------------------------ bars helper (value 0..1)
def vbars(x, y, w, h, groups, title, role, notes=None):
    """groups: [(label, [(value, colour, name)])]; linear axis 0..1."""
    o = cardH(x, y, w, h, role, title); ax = Axes(x + 60, y + 60, w - 84, h - 176, (0, 1), (0, 1)); ax.frame([], [0, .25, .5, .75, 1], yfmt=lambda v: f"{v:g}"); n = len(groups); gw = ax.w / n; lab = ""
    for gi, (g, bars) in enumerate(groups):
        bw = min(90, (gw - 40) / len(bars)); x1 = ax.x + gi * gw + (gw - bw * len(bars)) / 2
        for bi, (v, c1, c2, shown) in enumerate(bars):
            top = ax.Y(max(v, .002)); bx = x1 + bi * bw; hh = ax.y + ax.h - top
            o += box(bx + 4, top, bw - 8, max(hh, 3), c1, c2, 180, shadow=False, r=8)
            o += txt(bx, top - 30, bw, shown, 18, TXT, True, "center", 28)
        o += txt(ax.x + gi * gw, ax.y + ax.h + 8, gw, g, 17, TXT, True, "center", 26)
    o += ax.html()
    return o
CF, CM = ("#6a9fd6", "#2f4f7a"), ("#d9822b", "#a8452b")

# ------------------------------------------------ 5: Monte Carlo
def page5(lang):
    t = tr(lang); e = exp(); o = ""
    g = [(f"N = {n}", [(K.P(n, 2), *CF, f"{K.P(n, 2):.3f}"), (e[n], *CM, f"{e[n]:.3f}")]) for n in (4, 6)]
    o += vbars(0, 0, 700, 480, g, t("公式與隨機實測", "Formula versus random trials"), "blue")
    o += legend([(CF[0], t("公式 P", "formula P")), (CM[0], t("實測（1500 次）", "measured (1500 trials)"))], 90, 410, 280)
    o += cardH(716, 0, 444, 480, "orange", t("實驗設定與差距", "Setup and gaps"))
    st = [t("抽 N 個二維標準常態點", "Draw N standard-normal points in 2-D"), t("各配隨機 ±1 標籤（機會均等）", "Give each a random ±1 label"), t("線性規劃判斷是否存在分開兩類的直線", "A linear program checks for a separating line"), t("每個 N 做 1500 次，亂數種子 1", "1500 trials per N, random seed 1")]
    for i, s_ in enumerate(st):
        y0 = 52 + i * 52; o += card(732, y0, 412, 44, "grey", .12) + txt(740, y0 + 7, 396, f"{i+1}. " + s_, 16, TXT, False, "left", 32)
    o += card(732, 270, 412, 80, "yellow", .16) + txt(740, 278, 396, t(f"差距：N = 4 為 {abs(e[4]-K.P(4,2)):.3f}，N = 6 為 {abs(e[6]-K.P(6,2)):.3f}", f"Gaps: {abs(e[4]-K.P(4,2)):.3f} at N = 4, {abs(e[6]-K.P(6,2)):.3f} at N = 6"), 18, TXT, True, "center", 32)
    o += txt(740, 318, 396, t("標準誤約 √(0.25/1500) ≈ 0.013", "Standard error ≈ √(0.25/1500) ≈ 0.013"), 16, MUT, False, "center", 24)
    o += txt(732, 366, 412, t("兩個差距都在約一個標準誤以內：公式與實測吻合。〔程式計算〕", "Both gaps are within about one standard error: formula and measurement agree. (computed)"), 16, TXT, True, "center", 72)
    return o

# ------------------------------------------------ 6: lift
def page6(lang):
    t = tr(lang); e = exp(); o = ""
    a, b = e["lift"]
    g = [(t("原座標 l = 2", "Original, l = 2"), [(K.P(6, 2), *CF, f"{K.P(6,2):.3g}"), (a, *CM, f"{a:.3f}")]), (t("加二次項 l = 5", "With squares, l = 5"), [(K.P(6, 5), *CF, f"{K.P(6,5):.3g}"), (b, *CM, f"{b:.3f}")])]
    o += vbars(0, 0, 640, 480, g, t("N = 6：升維前後的可分比例", "N = 6: separable fraction before and after lifting"), "blue")
    o += legend([(CF[0], t("公式 P", "formula P")), (CM[0], t("實測（1500 次）", "measured (1500 trials)"))], 70, 410, 250)
    o += cardH(656, 0, 504, 480, "teal", t("多加三個座標", "Three extra coordinates"))
    o += txt(668, 50, 480, t("原本 [x₁, x₂]，l = 2；加上三個二次座標，l = 5：", "From [x₁, x₂] with l = 2, add three quadratic coordinates to get l = 5:"), 16, TXT, False, "left", 48)
    labs = [("x_1", "#4f86bf", "#2f4f7a"), ("x_2", "#4f86bf", "#2f4f7a"), ("x_1^2", "#4f9a68", "#2c6140"), ("x_2^2", "#4f9a68", "#2c6140"), ("x_1x_2", "#d9822b", "#a8452b")]
    for i, (tex, c1, c2) in enumerate(labs):
        x0 = 672 + i * 96; o += box(x0, 112, 86, 70, c1, c2, 135, shadow=(i == 4)) + formula(x0, 120, 86, tex, 24, 52)
    o += txt(668, 190, 480, t("藍：原座標　綠、橘：新加的三個", "Blue: original; green, orange: new"), 16, MUT, False, "center", 24)
    o += card(672, 232, 472, 70, "yellow", .16) + formula(672, 238, 472, r"N=6\leq l+1=6\ \Rightarrow\ P=1", 26, 58)
    o += txt(672, 318, 472, t("同樣的點與標籤：升維前約一半可分，升維後 1500 次全部可分。", "Same points and labels: about half are separable before, all 1500 after. "), 17, TXT, True, "center", 52)
    o += txt(672, 394, 472, t("兩者之間只差在多加了三個二次座標。〔程式計算〕", "Only the three added quadratic coordinates differ. (computed)"), 16, MUT, False, "center", 48)
    return o

# ------------------------------------------------ 7: how many dimensions
def page7(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 720, 480, "blue", t("N = 100：可分機率對維度 l", "N = 100: separability probability against dimension l"))
    ax = Axes(70, 66, 620, 330, (30, 80), (0, 1)); ax.frame([30, 40, 49, 56, 60, 68, 80], [0, .5, 1], yfmt=lambda v: f"{v:g}")
    ax.line([(l, K.P(100, l)) for l in range(30, 81)], "#6a9fd6", 4); ax.line([(30, .5), (80, .5)], "#e0c85f", 2, "8 6", .8)
    for l in (40, 48, 49, 56, 60, 68): ax.dot(l, K.P(100, l), "#d9822b" if l == 49 else "#6a9fd6", 9 if l == 49 else 7)
    ax.text(49, .5, "l = 49", "#e0c85f", "end", 17, True, -14, -12)
    o += ax.html() + txt(70, 446, 620, t("維度 l", "Dimension l"), 16, MUT, False, "center", 24)
    o += cardH(736, 0, 424, 480, "orange", t("要幾維才夠", "How many dimensions are enough"))
    for xx, lab, w_ in ((752, "l", 110), (870, "P", 274)): o += txt(xx, 48, w_, lab, 17, MUT, True, "center", 26)
    for i, l in enumerate((40, 48, 49, 56, 60, 68)):
        y0 = 78 + i * 44; hot = l == 49; o += card(752, y0, 392, 38, "yellow" if hot else "grey", .26 if hot else .1)
        o += txt(752, y0 + 5, 110, str(l), 18, TXT, True, "center", 28) + txt(870, y0 + 5, 274, f"{K.P(100, l):.4g}", 18, TXT, True, "center", 28)
    o += txt(752, 352, 392, t("N > l + 1 就有分不開的標法；N > 2(l + 1) 時可分機率低於一半。", "N > l + 1 already leaves labelings that fail; N > 2(l + 1) puts the probability below one half."), 16, MUT, False, "left", 72)
    o += card(752, 424, 392, 44, "yellow", .14) + txt(760, 432, 376, t("維度要接近資料量的一半以上。〔程式計算〕", "Dimension must reach about half the data size. (computed)"), 16, TXT, True, "center", 30)
    return o

# ------------------------------------------------ 8: separable is not generalisable
def page8(lang):
    t = tr(lang); e = exp(); ok, trn, gen = e["gen"]; o = ""
    g = [("", [(trn, "#4f9a68", "#2c6140", f"{trn*100:.0f}%"), (gen, "#d9822b", "#a8452b", f"{gen*100:.1f}%")])]
    o += vbars(0, 0, 560, 480, g, t("同一個分類器：訓練與新資料", "One classifier: training and new data"), "red")
    o += legend([("#4f9a68", t("訓練（5 點）", "training (5 points)")), ("#d9822b", t("新資料（200 點）", "new data (200 points)"))], 70, 410, 300)
    o += cardH(576, 0, 584, 480, "orange", t("實驗設定與結果", "Setup and result"))
    st = [t("5 個二維標準常態點，各配隨機 ±1 標籤", "5 standard-normal 2-D points with random ±1 labels"), t("用二次特徵（5 維）以線性規劃找分開的超平面，共做 300 次", "Find a separating hyperplane in the 5-D quadratic features by linear programming, 300 times"), t("每次再抽 200 個新點，也配隨機標籤，用找到的超平面預測", "Each time draw 200 new points, also randomly labeled, and predict with that hyperplane")]
    for i, s_ in enumerate(st):
        y0 = 52 + i * 74; o += card(592, y0, 552, 66, "grey", .12) + txt(600, y0 + 8, 536, f"{i+1}. " + s_, 16, TXT, False, "left", 54)
    o += card(592, 282, 552, 70, "yellow", .16) + txt(600, 290, 536, t(f"{ok} 次全部找得到（N = 5 ≤ l + 1 = 6，與公式一致），訓練全對", f"All {ok} runs found one (N = 5 ≤ l + 1 = 6, as the formula says), training all correct"), 17, TXT, True, "center", 56)
    o += card(592, 362, 552, 58, "red", .22) + txt(600, 372, 536, t(f"新點平均只對 {gen*100:.1f}%，就是亂猜", f"New points: only {gen*100:.1f}% on average, pure guessing"), 18, TXT, True, "center", 40)
    o += txt(592, 430, 552, t("升維只保證訓練資料分得開；有規律的資料才學得到東西。〔程式計算〕", "Lifting only guarantees that the training data separate; only data with structure teach anything. (computed)"), 16, MUT, False, "center", 40)
    return o

# ------------------------------------------------ 9: exercises
def page9(lang):
    t = tr(lang); o = ""
    qs = [("blue", t("題 1", "Q1"), t("N = 5 個一般位置的點，l = 2：O(5, 2) 與 P 是多少？", "N = 5 points in general position, l = 2: what are O(5, 2) and P?"), [r"O(5,2)=2(1+4+6)=22", r"P=22/32=0.6875"], t("O = 22，P = 0.6875。", "O = 22, P = 0.6875."), t("核對：22 與第 2 頁的表相同。", "Check: 22 matches the table on page 2.")),
          ("red", t("題 2", "Q2"), t("升到 l = 3，P 是多少？", "Raise l to 3: what is P?"), [r"O(5,3)=2(1+4+6+4)=30", r"P=30/32=0.9375"], t("O = 30，P = 0.9375。", "O = 30, P = 0.9375."), t("核對：升一維，可分機率從 0.6875 升到 0.9375。", "Check: one more dimension lifts P from 0.6875 to 0.9375."))]
    for i, (role, name, q, tex, ans, chk) in enumerate(qs):
        x = i * 590; o += card(x, 0, 570, 470, role) + head(x, 0, 570, role, name, 36, 18)
        o += txt(x + 16, 52, 538, q, 19, TXT, True, "left", 84)
        o += chip(x + 16, 150, 130, "grey", t("解題示範", "Solution"), False, 30, 16)
        o += card(x + 16, 194, 538, 150, "grey", .1) + formula(x + 16, 200, 538, tex[0], 24, 66) + formula(x + 16, 274, 538, tex[1], 24, 66)
        o += chip(x + 16, 364, 538, role, ans, True, 42, 20)
        o += txt(x + 16, 420, 538, chk, 17, MUT, False, "left", 48)
    return o

PAGES_DESIGN = [(f"p{i}", globals()[f"page{i}"]) for i in range(1, 10)]
if __name__ == "__main__":
    out = sys.argv[1]; only = sys.argv[2:]
    for fid, fn in PAGES_DESIGN:
        if only and fid not in only: continue
        print(fid); build("kp14_" + fid, fn, W, H, out)
