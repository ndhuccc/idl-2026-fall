"""KP15 pages designed in HTML (view-from-text-v3, route A) with the new JianBauDau fill effects. pageN(lang) -> body HTML of a 1160x480 figure."""
import os, sys, math
import numpy as np
SK = os.environ.get("COURSE_SKILLS") or (os.path.join(os.path.abspath(os.environ["COURSE_REPO"]), ".agents", "skills") if os.environ.get("COURSE_REPO") else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
HERE = os.path.dirname(os.path.abspath(__file__)); HERE = HERE[:-5] if HERE.endswith(("/W_zh", "/W_en")) else HERE
sys.path.insert(0, SK + "/view-from-text-v3/examples"); sys.path.insert(0, SK + "/view-from-text-v3/tools"); sys.path.insert(0, HERE)
from kp17_pages import Axes, tr, take, legend, fmt, sci, txt, named, card, head, chip, formula, svg, BL, RD, VI, GR, YE, OR, TE, INK, TXT, MUT, hexc, build, native, _sup, rgba, rgb
import kp15_data as D
W, H = 1160, 480
def spts(a, n=70):
    idx = np.linspace(0, len(a) - 1, n).astype(int); return a[idx]
def box(x, y, w, h, c1, c2, ang=135, shadow=True, r=14):
    return f'<div style="position:absolute;left:{x}px;top:{y}px;width:{w}px;height:{h}px;border-radius:{r}px;background:linear-gradient({ang}deg,{c1},{c2});' + ('box-shadow:0 10px 18px 0 rgba(0,0,0,0.45);' if shadow else '') + '"></div>'
def cardH(x, y, w, h, role, title, size=17): return card(x, y, w, h, role) + head(x, y, w, role, title, 36, size)

# ------------------------------------------------ 1: second-order expansion
def page1(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 760, 480, "blue", t("兩個輸入、二階：把乘積項全寫出來", "Two inputs, second order: write every product term"))
    o += formula(10, 48, 740, r"g(x)=w_0+w_1x_1+w_2x_2+w_{12}x_1x_2+w_{11}x_1^2+w_{22}x_2^2", 22, 46)
    blocks = [("1", t("偏置", "bias"), "#5a544b", "#3a352d"), ("x_1", t("一次項", "linear"), "#4f86bf", "#2f4f7a"), ("x_2", t("一次項", "linear"), "#4f86bf", "#2f4f7a"),
              ("x_1x_2", t("交互項", "interaction"), "#d9822b", "#a8452b"), ("x_1^2", t("平方項", "square"), "#4f9a68", "#2c6140"), ("x_2^2", t("平方項", "square"), "#4f9a68", "#2c6140")]
    for i, (tex, lab, c1, c2) in enumerate(blocks):
        x = 22 + (i % 3) * 244; y = 112 + (i // 3) * 170; hot = i == 3
        o += box(x, y, 224, 150, c1, c2, 135, shadow=hot)
        o += formula(x, y + 22, 224, tex, 38 if hot else 34, 60) + txt(x, y + 100, 224, lab, 19, TXT, True, "center", 28)
    o += txt(22, 434, 716, t("新座標 y = [x₁, x₂, x₁x₂, x₁², x₂²]，加偏置共 6 項：g = wᵀy + w₀，對 6 個權重仍是線性。", "y = [x1, x2, x1x2, x1², x2²] + bias: 6 terms, g = wᵀy + w0 is linear in the weights."), 16, TXT, False, "left", 48)
    o += cardH(776, 0, 384, 480, "orange", t("名詞", "Terms"))
    o += named(792, 56, 352, t("多項式分類器：用最高 r 階的多項式當判別函數，也就是把次數總和不超過 r 的乘積項當新座標，再學一個線性分類器。", "Polynomial classifier: use a polynomial of degree at most r as the discriminant, that is, take every product term of total degree at most r as a new coordinate and learn a linear classifier on them."), "term:" + t("多項式分類器", "Polynomial classifier"), 16, TXT, 160)
    o += named(792, 236, 352, t("交互項：兩個以上不同輸入相乘的項，例如 x₁x₂；單獨的 x₁² 叫平方項。", "Interaction term: a term that multiplies two or more different inputs, for example x1x2; a lone x1² is a square term."), "term:" + t("交互項", "Interaction term"), 16, TXT, 120)
    o += card(792, 372, 352, 88, "yellow", .14) + txt(800, 380, 336, t("數一數：偏置 1 + 一次項 2 + 交互項 1 + 平方項 2 = 6 項", "Count: bias 1 + linear 2 + interaction 1 + square 2 = 6 terms"), 17, TXT, True, "center", 72)
    return o

# ------------------------------------------------ 2: XOR needs one interaction term; hyperbola
def page2(lang):
    t = tr(lang); o = ""
    g = lambda a, b: -0.25 + a + b - 2 * a * b
    o += cardH(0, 0, 500, 480, "orange", t("升維：只多一個交互項 x₁x₂", "Lift: just one more term, x1x2"))
    o += txt(12, 46, 476, t("新座標 y = (x₁, x₂, x₁x₂)", "new coordinates y = (x1, x2, x1x2)"), 17, TXT, True, "center", 28)
    o += txt(14, 84, 90, "(x₁, x₂)", 16, MUT, True, "center", 24) + txt(118, 84, 150, "y", 16, MUT, True, "center", 24) + txt(300, 84, 160, "g", 16, MUT, True, "center", 24)
    pts = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for i, (a, b) in enumerate(pts):
        gv = g(a, b); pos = gv > 0; y0 = 114 + i * 52; o += card(10, y0, 480, 44, "blue" if pos else "orange", .16)
        o += txt(14, y0 + 8, 90, f"({a}, {b})", 17, TXT, True, "center", 26) + txt(112, y0 + 8, 160, f"({a}, {b}, {a*b})", 17, TXT, False, "center", 26)
        o += chip(300, y0 + 6, 160, "blue" if pos else "orange", ("+" if pos else "−") + f"{abs(gv):.2f}", True, 32, 18)
    o += txt(14, 330, 472, t("g = −¼ + x₁ + x₂ − 2x₁x₂。取平面 y₁ + y₂ − 2y₃ − ¼ = 0；四點的 g 為 −0.25、+0.75、+0.75、−0.25，符號全對。", "Take the plane y1 + y2 − 2y3 − ¼ = 0; the four g values are −0.25, +0.75, +0.75, −0.25: every sign is right."), 16, TXT, False, "left", 72)
    o += card(10, 412, 480, 58, "yellow", .14) + txt(18, 418, 464, t("常數取 −¼ 或 −0.5 都行：只要 c 介於 0 與 1 之間，符號就全對", "A constant of −¼ or −0.5 both work: any c between 0 and 1 gives the right signs"), 16, TXT, False, "center", 48)
    # right: hyperbola regions (curved polygons)
    o += cardH(516, 0, 644, 480, "blue", t("回到二維：一對雙曲線", "Back in 2-D: a pair of hyperbola branches"))
    ax = Axes(560, 54, 400, 400, (-0.4, 1.4), (-0.4, 1.4))
    ur, ll = D.hyper_regions(0.9)
    s = f'<rect x="{ax.X(-0.4):.1f}" y="{ax.Y(1.4):.1f}" width="{ax.X(1.4)-ax.X(-0.4):.1f}" height="{ax.Y(-0.4)-ax.Y(1.4):.1f}" fill="{BL}" fill-opacity="0.28"/>'
    for reg in (ur, ll):
        pts_ = " ".join(f"{ax.X(x):.1f},{ax.Y(y):.1f}" for x, y, sm in reg)
        s += f'<polygon points="{pts_}" fill="{OR}" fill-opacity="0.55" stroke="#e0c85f" stroke-width="3" data-smooth="all"/>'
    s += f'<line x1="{ax.X(-0.4)}" y1="{ax.Y(0)}" x2="{ax.X(1.4)}" y2="{ax.Y(0)}" stroke="#6b645a" stroke-width="1.5"/><line x1="{ax.X(0)}" y1="{ax.Y(-0.4)}" x2="{ax.X(0)}" y2="{ax.Y(1.4)}" stroke="#6b645a" stroke-width="1.5"/>'
    for (a, b) in ((0, 1), (1, 0)): s += f'<circle cx="{ax.X(a):.1f}" cy="{ax.Y(b):.1f}" r="15" fill="#47688f" stroke="#ffffff" stroke-width="2"/>'
    for (a, b) in ((0, 0), (1, 1)): s += f'<rect x="{ax.X(a)-14:.1f}" y="{ax.Y(b)-14:.1f}" width="28" height="28" rx="4" fill="#94503a" stroke="#ffffff" stroke-width="2"/>'
    s += f'<circle cx="{ax.X(0.5):.1f}" cy="{ax.Y(0.5):.1f}" r="8" fill="#e0c85f"/>'
    o += svg(0, 0, W, H, s)
    o += txt(ax.X(0.5) + 12, ax.Y(0.5) - 30, 150, "(0.5, 0.5)", 16, "#e0c85f", True, "left", 24)
    o += txt(ax.X(0) - 14, ax.Y(1) + 14, 160, "(0, 1)", 16, TXT, False, "left", 24) if False else ""
    o += card(982, 54, 168, 190, "blue", .14) + txt(990, 62, 152, t("A（g > 0）<br>藍色帶狀區<br>含 (0,1)、(1,0)", "A (g > 0)<br>blue band<br>holds (0,1), (1,0)"), 16, TXT, True, "left", 96)
    o += card(982, 258, 168, 190, "orange", .14) + txt(990, 266, 152, t("B（g < 0）<br>橘色兩個角<br>含 (0,0)、(1,1)", "B (g < 0)<br>orange corners<br>hold (0,0), (1,1)"), 16, TXT, True, "left", 96)
    o += txt(982, 340, 168, t("例：(0.5, 0.5) 的 g = −0.25 + 1 − 0.5 = +0.25，在 A 的一側", "e.g. (0.5, 0.5) has g = −0.25 + 1 − 0.5 = +0.25, on the A side"), 16, MUT, False, "left", 96) if False else ""
    return o

# ------------------------------------------------ 3: how many terms
def page3(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 640, 480, "violet", t("共有幾項？", "How many terms?"))
    o += txt(14, 48, 612, t("l 個輸入、最高 r 階，新座標（含常數項）共", "With l inputs and degree up to r, the number of terms is"), 17, TXT, False, "center", 28)
    o += card(70, 86, 500, 130, "violet", .22) + formula(70, 100, 500, r"k=\frac{(l+r)!}{r!\,l!}", 32, 100)
    o += txt(14, 230, 612, t("驗算", "Check"), 16, MUT, True, "center", 24)
    for i, (tex, lab) in enumerate(((r"l=2,\ r=2:\ \ \frac{4!}{2!\,2!}=6", ""), (r"l=3,\ r=2:\ \frac{5!}{2!\,3!}=10", ""))):
        o += card(24, 262 + i * 76, 592, 66, "grey", .12) + formula(24, 270 + i * 76, 592, tex, 24, 52)
    o += txt(14, 420, 612, t("與 KP13 數過的一致：兩個輸入、二階是 6 項。", "Consistent with the count in KP13: two inputs, second order, 6 terms."), 16, MUT, False, "center", 24)
    o += cardH(656, 0, 504, 480, "teal", t("l = 2 時，階數 r 與項數 k", "With l = 2: degree r against the number of terms k"))
    ks = [D.k_terms(2, r) for r in range(1, 6)]; mx = 24
    for i, (r, v) in enumerate(zip(range(1, 6), ks)):
        bx = 690 + i * 90; hgt = v / mx * 300; top = 400 - hgt
        o += box(bx, top, 70, hgt, "#5fb8a8", "#2f7d70", 180, shadow=False, r=8)
        o += txt(bx - 10, top - 32, 90, str(v), 22, TXT, True, "center", 30) + txt(bx - 10, 410, 90, f"r = {r}", 16, MUT, True, "center", 24)
    o += txt(668, 442, 480, t("3、6、10、15、21：r 每加 1，多出的項越來越多", "3, 6, 10, 15, 21: each extra degree adds more and more terms"), 16, TXT, False, "center", 24)
    return o

# ------------------------------------------------ 4: explosion + memory
def page4(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 640, 480, "red", t("項數對階數：組合爆炸（對數座標）", "Terms against degree: combinatorial explosion (log scale)"))
    ax = Axes(86, 60, 520, 330, (0.5, 10.5), (0, 14), log=False)
    ax.raw(f'<line x1="{ax.x}" y1="{ax.y}" x2="{ax.x}" y2="{ax.y+ax.h}" stroke="#b7b0a5" stroke-width="2"/><line x1="{ax.x}" y1="{ax.y+ax.h}" x2="{ax.x+ax.w}" y2="{ax.y+ax.h}" stroke="#b7b0a5" stroke-width="2"/>')
    for e in (0, 2, 4, 6, 8, 10, 12, 14):
        py = ax.Y(e); ax.raw(f'<line x1="{ax.x}" y1="{py:.1f}" x2="{ax.x+ax.w}" y2="{py:.1f}" stroke="#6b645a" stroke-width="1" opacity="0.45"/><text x="{ax.x-8}" y="{py+5:.1f}" text-anchor="end" font-size="16" fill="#b7b0a5">10{_sup(e)}</text>')
    for r in range(1, 11): ax.raw(f'<text x="{ax.X(r):.1f}" y="{ax.y+ax.h+24}" text-anchor="middle" font-size="16" fill="#b7b0a5">{r}</text>')
    for l, col in ((2, TE), (10, OR), (100, RD)):
        pts = [(r, math.log10(D.k_terms(l, r))) for r in range(1, 11)]
        ax.line(pts, col, 4)
        for r, v in pts: ax.dot(r, v, col, 6)
        ax.text(10, pts[-1][1], f"l = {l}", col, "end", 16, True, dx=-14, dy=-14)
    ax.raw(f'<text x="{ax.x+ax.w/2}" y="{ax.y+ax.h+52}" text-anchor="middle" font-size="16" font-weight="700" fill="#f2ede4">{t("最高階數 r", "maximum degree r")}</text>')
    o += ax.html()
    o += card(86, 70, 270, 66, "red", .2) + txt(94, 76, 254, t("l = 10、r = 10：<br>184,756 項", "l = 10, r = 10:<br>184,756 terms"), 18, TXT, True, "center", 52)
    o += txt(14, 446, 612, t("l = 2 時 r = 10 也才 66 項；輸入多了才爆。〔程式計算〕", "With l = 2, r = 10 is only 66 terms; it explodes when inputs grow. (computed)"), 16, MUT, False, "center", 24)
    k, mphi, mpp = D.memory()
    o += cardH(656, 0, 504, 480, "orange", t("記憶體（1000 筆資料，l = 10、r = 10）", "Memory (1000 samples, l = 10, r = 10)"))
    o += txt(668, 46, 480, t("每個數 8 位元組；1 GB 以 10⁹ 位元組計；長條長度成正比", "8 bytes per number; 1 GB = 10⁹ bytes; bar lengths are to scale"), 16, MUT, False, "center", 24)
    o += txt(668, 96, 200, t("設計矩陣 Φ", "design matrix Φ"), 17, TXT, True, "left", 26) + txt(668, 122, 200, "1000 × 184,756", 16, MUT, False, "left", 24)
    o += box(668, 154, 4, 40, "#4f86bf", "#2f4f7a", 90, shadow=False, r=2) + txt(684, 158, 240, "≈ 1.48 GB", 22, TXT, True, "left", 30)
    o += txt(668, 232, 260, t("ΦᵀΦ（KP13 的正規方程）", "ΦᵀΦ (KP13)"), 17, TXT, True, "left", 26) + txt(668, 258, 220, "184,756 × 184,756", 16, MUT, False, "left", 24)
    o += box(668, 290, 470, 40, "#d97757", "#94503a", 90, shadow=True, r=8) + txt(676, 296, 440, "≈ 273 GB", 22, TXT, True, "left", 30)
    o += card(668, 356, 470, 92, "yellow", .14) + txt(676, 364, 454, t("差約 185 倍；一般電腦的記憶體裝不下。〔程式計算〕", "About 185 times apart; an ordinary computer cannot hold it. (computed)"), 18, TXT, True, "center", 72)
    return o

# ------------------------------------------------ 5: ability (two moons)
def page5(lang):
    t = tr(lang); Xtr, ytr, sw = D.sweep(); o = ""
    o += cardH(0, 0, 580, 480, "blue", t("兩個月牙：r = 3 的邊界", "Two moons: the r = 3 boundary"))
    ax = Axes(30, 56, 520, 340, (-1.6, 2.6), (-1.1, 1.6)); s = f'<rect x="{ax.x}" y="{ax.y}" width="{ax.w}" height="{ax.h}" fill="none" stroke="#6b645a" stroke-width="2"/>'
    o += svg(0, 0, W, H, s)
    for seg in D.boundary(3): ax.line([(float(x), float(y)) for x, y in spts(seg, 60)], "#e0c85f", 4)
    for seg in D.boundary(1): ax.line([(float(x), float(y)) for x, y in spts(seg, 12)], "#b7b0a5", 3, "9 7")
    for (x_, y_), yy in zip(Xtr, ytr): ax.dot(x_, y_, "#6a9fd6" if yy > 0 else "#d97757", 5)
    o += ax.html()
    o += legend([("#e0c85f", t("r = 3（曲線）", "r = 3 (curve)")), ("#b7b0a5", t("r = 1（直線）", "r = 1 (line)"))], 24, 404, 160)
    o += txt(236, 404, 336, t("訓練集每類 30 點、共 60 點；A 類（藍）取上半圓，B 類（橘）取平移的下半圓，各加 0.2 的雜訊。", "30 points per class: A (blue) on the upper half circle, B (orange) on the shifted lower one, noise 0.2."), 16, MUT, False, "left", 72)
    o += cardH(596, 0, 564, 480, "green", t("階數 1～5 的正確率", "Accuracy for degree 1 to 5"))
    xs = [596 + 14, 596 + 100, 596 + 220, 596 + 360]
    for xx, lab, w_ in ((610, "r", 70), (686, t("項數 k", "terms k"), 110), (808, t("訓練", "train"), 130), (950, t("新點", "new"), 130)): o += txt(xx, 48, w_, lab, 16, MUT, True, "center", 24)
    for i, r in enumerate(range(1, 6)):
        k, a, b, c = sw[r]; y0 = 78 + i * 56; hot = r == 3; o += card(606, y0, 544, 48, "green" if hot else "grey", .26 if hot else .1)
        o += txt(610, y0 + 10, 70, str(r), 18, TXT, True, "center", 28) + txt(686, y0 + 10, 110, str(k), 18, TXT, False, "center", 28)
        o += txt(808, y0 + 10, 130, f"{a*100:.1f}%", 18, TXT, False, "center", 28) + txt(950, y0 + 10, 130, f"{b*100:.1f}%", 18, TXT, True, "center", 28)
    o += card(606, 372, 544, 88, "yellow", .14) + txt(614, 380, 528, t("低階多項式就能彎出月牙形的邊界：r = 3 的新點正確率 93.2%，比直線（84.9%）多對約 8 個百分點。〔程式計算〕", "A low degree already bends around the moons: r = 3 gets 93.2% on new points vs 84.9% for the line. (computed)"), 16, TXT, True, "center", 72)
    return o

# ------------------------------------------------ 6: too high a degree
def page6(lang):
    t = tr(lang); Xtr, ytr, sw = D.sweep(); o = ""
    o += cardH(0, 0, 640, 480, "red", t("階數 1～10：訓練與新點正確率", "Degree 1 to 10: accuracy on training and new points"))
    ax = Axes(86, 62, 520, 300, (0.5, 10.5), (0.5, 1.02)); ax.frame([], [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], yfmt=lambda v: f"{int(round(v*100))}%")
    tr_ = [(r, sw[r][1]) for r in range(1, 11)]; te_ = [(r, sw[r][2]) for r in range(1, 11)]
    ax.line(tr_, BL, 4).line(te_, RD, 4)
    for r in range(1, 11):
        ax.dot(r, sw[r][1], BL, 6).dot(r, sw[r][2], RD, 6); ax.raw(f'<text x="{ax.X(r):.1f}" y="{ax.y+ax.h+24}" text-anchor="middle" font-size="16" fill="#b7b0a5">{r}</text>')
    ax.text(10, sw[10][2], "55.5%", RD, "end", 17, True, dx=-8, dy=24); ax.text(3, sw[3][2], "93.2%", RD, "middle", 17, True, dy=-16); ax.text(10, sw[10][1], "100%", BL, "end", 17, True, dx=-4, dy=-14)
    ax.raw(f'<text x="{ax.x+ax.w/2}" y="{ax.y+ax.h+52}" text-anchor="middle" font-size="16" font-weight="700" fill="#f2ede4">{t("最高階數 r", "maximum degree r")}</text>')
    o += ax.html() + legend([(BL, t("訓練", "training")), (RD, t("新點", "new points"))], 110, 280, 170)
    o += txt(14, 440, 612, t("新點在 r = 3 附近最高，之後一路下降。", "New-point accuracy peaks near r = 3 and falls afterwards."), 16, TXT, False, "center", 24)
    o += cardH(656, 0, 504, 480, "orange", t("條件數與過度配適", "Condition number and overfitting"))
    o += named(670, 48, 476, t("條件數：矩陣對微小誤差的放大倍率，越大解越不穩。", "Condition number: how much a matrix magnifies small errors; the larger, the less stable the solution."), "term:" + t("條件數", "Condition number"), 16, TXT, 72)
    for xx, lab, w_ in ((666, "r", 44), (716, "k", 56), (776, t("訓練", "train"), 80), (862, t("新點", "new"), 80), (950, t("Φ 條件數", "cond. of Φ"), 190)): o += txt(xx, 124, w_, lab, 16, MUT, True, "center", 24)
    for i, r in enumerate((6, 8, 9, 10)):
        k, a, b, c = sw[r]; y0 = 154 + i * 44; o += card(666, y0, 484, 38, "red" if r >= 9 else "grey", .22 if r >= 9 else .1)
        o += txt(666, y0 + 5, 44, str(r), 17, TXT, True, "center", 26) + txt(716, y0 + 5, 56, str(k), 17, TXT, False, "center", 26) + txt(776, y0 + 5, 80, f"{a*100:.1f}%", 17, TXT, False, "center", 26) + txt(862, y0 + 5, 80, f"{b*100:.1f}%", 17, TXT, True, "center", 26) + txt(950, y0 + 5, 190, sci(c) if c < 1e3 or c >= 1e4 else f"{c:.0f}", 17, TXT, False, "center", 26)
    o += named(670, 340, 476, t("過度配適：模型有餘裕把每個訓練點（含雜訊）背下來，訓練正確率到 100%，新點卻接近亂猜。r = 9 有 55 項、r = 10 有 66 項，接近或超過 60 個訓練點。", "Overfitting: the model has room to memorize every training point, noise included, so training accuracy reaches 100% while new points are close to guessing. r = 9 has 55 terms and r = 10 has 66, near or above the 60 training points."), "term:" + t("過度配適", "Overfitting"), 16, TXT, 120)
    return o

# ------------------------------------------------ 7: kernel preview
def page7(lang):
    t = tr(lang); v, vp, px, pz = D.kernel_check(); o = ""
    o += cardH(0, 0, 570, 400, "green", t("直接算：只做一次內積與一個次方", "Direct: one inner product and one power"))
    o += txt(12, 48, 546, t("l = 2、二階，x = (1, 2)，z = (3, −1)", "l = 2, second order, x = (1, 2), z = (3, −1)"), 17, TXT, True, "center", 28)
    o += card(30, 100, 510, 120, "green", .16) + formula(30, 134, 510, r"(1+x\cdot z)^2=(1+3-2)^2=2^2=4", 24, 60)
    o += txt(14, 252, 542, t("左邊只做一次內積、一個次方。", "The left side needs one inner product and one power."), 18, TXT, False, "center", 28)
    o += txt(14, 300, 542, t("l = 10、r = 10 時照樣只算這麼多。", "For l = 10, r = 10 it is still just this much."), 18, MUT, False, "center", 28)
    o += cardH(590, 0, 570, 400, "orange", t("造新座標：6 項再做內積", "Build the coordinates: a 6-term inner product"))
    o += formula(596, 44, 558, r"\varphi(x)=(1,\sqrt{2}x_1,\sqrt{2}x_2,x_1^2,x_2^2,\sqrt{2}x_1x_2)", 18, 40)
    o += txt(604, 100, 540, "φ(x) = (1, √2, 2√2, 1, 4, 2√2)", 17, TXT, False, "left", 26) + txt(604, 130, 540, "φ(z) = (1, 3√2, −√2, 9, 1, −3√2)", 17, TXT, False, "left", 26)
    terms = [1, 6, -4, 9, 4, -12]
    for i, tv in enumerate(terms): o += chip(604 + i * 90, 184, 80, "orange" if tv > 0 else "red", (("+" if i else "") + str(tv)) if tv > 0 else "−" + str(abs(tv)), False, 38, 20)
    o += card(604, 244, 540, 70, "yellow", .16) + formula(604, 254, 540, r"1+6-4+9+4-12=4", 30, 50)
    o += txt(604, 330, 540, t("逐項相乘相加，結果一樣是 4。", "Multiply term by term and add: again 4."), 17, TXT, False, "center", 28)
    o += card(0, 414, 1160, 66, "violet", .18) + txt(14, 424, 1132, t("l = 10、r = 10 時右邊是 184,756 項的內積。核方法（KP19）就是為了避開這個爆炸。", "For l = 10, r = 10 the right side is an inner product of 184,756 terms. Kernel methods (KP19) exist to avoid this explosion."), 18, TXT, True, "center", 48)
    return o

# ------------------------------------------------ 8: exercises
def page8(lang):
    t = tr(lang); o = ""
    qs = [("blue", t("題 1", "Q1"), t("l = 4 個輸入、最高 r = 3 階，共 k 項？", "With l = 4 inputs and degree up to 3, how many terms k?"), [r"k=\frac{(l+r)!}{r!\,l!}=\frac{7!}{3!\,4!}", r"=\frac{5040}{6\times24}=35"], t("k = 35 項。", "k = 35 terms."), t("核對：35 就是 C(7, 3)。", "Check: 35 is C(7, 3).")),
          ("red", t("題 2", "Q2"), t("g = −¼ + x₁ + x₂ − 2x₁x₂，點 (1.5, 1.5) 屬哪一類（g > 0 為 A）？", "g = −¼ + x1 + x2 − 2x1x2:<br>which class is (1.5, 1.5)? (g > 0 is A)"), [r"g(1.5,1.5)=-0.25+3-4.5", r"=-1.75<0"], t("g < 0，屬類 B。", "g < 0, class B."), t("核對：點在右上角的 B 區，和 (1,1) 同一區。", "Check: the point lies in the upper-right B region, together with (1,1)."))]
    for i, (role, name, q, tex, ans, chk) in enumerate(qs):
        x = i * 590; o += card(x, 0, 570, 470, role) + head(x, 0, 570, role, name, 36, 18)
        o += txt(x + 16, 52, 538, q, 19, TXT, True, "left", 84)
        o += chip(x + 16, 150, 130, "grey", t("解題示範", "Solution"), False, 30, 16)
        o += card(x + 16, 194, 538, 150, "grey", .1) + formula(x + 16, 200, 538, tex[0], 22, 66) + formula(x + 16, 274, 538, tex[1], 22, 66)
        o += chip(x + 16, 364, 538, role, ans, True, 42, 20)
        o += txt(x + 16, 420, 538, chk, 17, MUT, False, "left", 48)
    return o

PAGES_DESIGN = [("p1", page1), ("p2", page2), ("p3", page3), ("p4", page4), ("p5", page5), ("p6", page6), ("p7", page7), ("p8", page8)]
if __name__ == "__main__":
    out = sys.argv[1]; only = sys.argv[2:]
    for fid, fn in PAGES_DESIGN:
        if only and fid not in only: continue
        print(fid); build("kp15_" + fid, fn, W, H, out)
