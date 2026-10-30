import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lec01_kit import *

KP = "KP09"; N = 8
PTS = [(0, 0), (0, 1), (1, 0), (1, 1)]; YS = [0, 1, 1, 0]
CEN = [(0, 0), (1, 1)]
phi = lambda p, c: math.exp(-((p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2))
PHI = [[phi(p, c) for c in CEN] for p in PTS]
assert [round(v, 3) for v in PHI[0]] == [1.0, 0.135] and [round(v, 3) for v in PHI[1]] == [0.368, 0.368] and PHI[1] == PHI[2] and [round(v, 3) for v in PHI[3]] == [0.135, 1.0]
def lstsq(A, y):
    n = len(A[0]); M = [[sum(r[i] * r[j] for r in A) for j in range(n)] + [sum(r[i] * t for r, t in zip(A, y))] for i in range(n)]
    for i in range(n):
        piv = max(range(i, n), key=lambda r: abs(M[r][i])); M[i], M[piv] = M[piv], M[i]
        for r in range(n):
            if r != i:
                f = M[r][i] / M[i][i]; M[r] = [a - f * b for a, b in zip(M[r], M[i])]
    return [M[i][n] / M[i][i] for i in range(n)]
A = [r + [1.0] for r in PHI]; WLS = lstsq(A, YS)
assert [round(v, 3) for v in WLS] == [-2.503, -2.503, 2.841]
OUT = [sum(a * w for a, w in zip(r, WLS)) for r in A]; assert all(abs(o - t) < 1e-9 for o, t in zip(OUT, YS))
SUMS = [r[0] + r[1] for r in PHI]; assert round(SUMS[0], 3) == 1.135 and round(SUMS[1], 3) == 0.736
THR = (0.5 - WLS[2]) / WLS[0]; assert round(THR, 3) == 0.936
# SOM step
W0 = [(0.2, 0.2), (0.5, 0.5), (0.8, 0.8)]; X = (0.9, 0.7)
DIST = [math.hypot(X[0] - w[0], X[1] - w[1]) for w in W0]; assert [round(d, 3) for d in DIST] == [0.86, 0.447, 0.141]
WIN = min(range(3), key=lambda j: DIST[j]); assert WIN == 2
ETA = 0.5; W1 = list(W0)
for j in (1, 2): W1[j] = tuple(w + ETA * (x - w) for w, x in zip(W0[j], X))
assert [round(v, 2) for v in W1[2]] == [0.85, 0.75] and [round(v, 2) for v in W1[1]] == [0.7, 0.6] and W1[0] == W0[0]

def make(lang):
    zh = lang == "zh"
    T = lambda a, b_: a if zh else b_
    pages = []
    new = lambda n, title: new_page(n, N, KP, title)

    # ============ 1: fixed first layer ============
    g = new(1, T("KP09 第一層固定，只學第二層", "KP09 Fix the first layer, learn only the second"))
    g.beat = 1
    xs_ = []
    for k in range(2):
        y = 230 + k * 160; xs_.append(g.mark()); g.ellipse(120, y, 38, fill=BLUEL, stroke=BLUE, sw=3, z=3); g.text(82, y - 38, 76, 76, f"$x_{k + 1}$", size=22, color=BLUE, bold=True, align="center", vcenter=True, z=8)
    g.beat = 2
    hs = []
    for k in range(3):
        y = 160 + k * 150; hs.append(g.mark()); g.rect(440, y - 40, 160, 80, fill=PURL, stroke=PUR, sw=4, radius=12, z=3, dash=True); g.text(440, y - 40, 160, 80, rf"$\phi_{k + 1}$", size=28, color=PUR, bold=True, align="center", vcenter=True, z=8)
        for a in xs_: g.link(a, hs[-1], 4, 12, GRAY, 3)
    g.text(380, 590, 280, 40, T("固定:不訓練", "fixed: not trained"), size=22, color=PUR, bold=True, align="center", z=8)
    g.beat = 3
    out = g.mark(); g.ellipse(1030, 310, 52, fill=GRNL, stroke=GRN, sw=3, z=3); g.text(978, 258, 104, 104, r"$\hat y$", size=30, color=GRN, bold=True, align="center", vcenter=True, z=8)
    for h in hs: g.link(h, out, 4, 12, GRN, 5)
    g.text(780, 590, 300, 40, T("可訓練的權重 $w$", "trainable weights $w$"), size=22, color=GRN, bold=True, align="center", z=8)
    banner(g, T("RBF 網路(Broomhead 與 Lowe,1988):第一層固定，第二層用最小二乘", "RBF network (Broomhead and Lowe, 1988): fixed first layer, least squares for the second"))
    pages.append((g, T("RBF 的結構", "RBF structure"), {
        0: T("到現在，第一層的權重都是學出來的。如果第一層固定成一組局部的基函數，只學第二層，會怎樣？", "So far the first layer's weights were learned. What if the first layer is fixed as local basis functions and only the second is learned?"),
        1: T("輸入送進幾個固定的基函數 phi，第一層不訓練：先用 K-means 取群心當中心，再由各群的變異數決定寬度。", "The input goes through a few fixed basis functions phi; the first layer is not trained: K-means gives the centers and the cluster variances give the widths."),
        2: T("輸出是基函數的加權和，只有這一層的權重要學。這是 1988 年 Broomhead 與 Lowe 提出的 RBF 網路。", "The output is a weighted sum of the basis functions; only this layer's weights are learned. This is the RBF network of Broomhead and Lowe, 1988."),
        3: T("", "")}))

    # ============ 2: Gaussian local ============
    g = new(2, T("高斯基函數是「局部」的", "A Gaussian basis function is “local”"))
    x0, y0, w_, h_ = 60, 130, 740, 420
    ux = lambda r: x0 + w_ / 2 + r / 3.2 * (w_ / 2 - 30); uy = lambda v: y0 + h_ - 40 - v * (h_ - 90)
    g.beat = 1
    g.rect(x0, y0, w_, h_, fill=CARD, stroke=GRAY, sw=3, radius=12, z=1)
    g.line([(x0 + 20, uy(0)), (x0 + w_ - 20, uy(0))], GRAY, sw=2, z=2)
    g.line([(ux(i / 40 * 3.2 - 3.2), uy(math.exp(-(i / 40 * 3.2 - 3.2) ** 2 / 1))) for i in range(81)], PUR, sw=6, z=5) if False else None
    g.line([(ux(-3.0 + 6.0 * i / 80), uy(math.exp(-(-3.0 + 6.0 * i / 80) ** 2))) for i in range(81)], PUR, sw=6, z=5)
    g.text(x0, y0 + h_ + 4, w_, 36, T("橫軸:離中心的距離 $r$ 縱軸:$\\phi=e^{-r^2}$ ($2\\sigma^2=1$)", "x: distance $r$ from the center · y: $\\phi=e^{-r^2}$ ($2\\sigma^2=1$)"), size=18, color=MUTED, bold=True, align="center", z=8)
    g.beat = 2
    g.ellipse(ux(0), uy(1), 14, fill=GRN, stroke="#ffffff", sw=3, z=7); g.text(ux(0) + 20, uy(1) - 6, 240, 36, T("中心:值為 1", "center: value 1"), size=20, color=GRN, bold=True, z=8)
    g.beat = 3
    for r_, c_ in ((1.0, ORG), (2.0, RED)):
        g.ellipse(ux(r_), uy(math.exp(-r_ ** 2)), 12, fill=c_, stroke="#ffffff", sw=3, z=7); g.text(ux(r_) + 14, uy(math.exp(-r_ ** 2)) - 36, 180, 34, rf"$r={r_:g}\Rightarrow {math.exp(-r_ ** 2):.3f}$".replace(r"\Rightarrow", ",") if True else "", size=18, color=c_, bold=True, z=8)
    g.rect(840, 150, 390, 360, fill=PURL, stroke=PUR, sw=3, radius=16, z=3)
    g.text(850, 160, 370, 50, T("像一面旗子", "like a flag"), size=26, color=PUR, bold=True, align="center", z=8)
    g.text(850, 220, 370, 280, T("離旗子越遠，影響越接近 0\n只管附近的點\n一個點離哪面旗子近，就受哪面旗子影響", "the farther from the flag, the closer to 0\nit governs only nearby points\na point is influenced by the flags near it"), size=22, color=INK, bold=True, align="center", z=8)
    banner(g, T("第一層不訓練:群心當中心，群的變異數決定寬度", "the first layer is not trained: cluster centers give the centers, variances the widths"))
    pages.append((g, T("高斯基函數", "The Gaussian basis"), {
        0: T("常用的基函數是高斯：離中心越遠越小。", "The usual basis function is the Gaussian: smaller the farther from the center."),
        1: T("它在中心最大，值是 1。", "It is largest at the center, with value 1."),
        2: T("離中心越遠越接近 0，所以是局部的，像一面旗子只管附近。", "The farther from the center the closer to 0, so it is local, like a flag that governs only its neighborhood."),
        3: T("", "")}))

    # ============ 3: XOR in basis space ============
    g = new(3, T("XOR 搬到基函數空間", "XOR moved into basis-function space"))
    g.beat = 1
    cx = [60, 240, 480, 700, 900]; cw = [180, 200, 220, 200, 200]
    for k, h in enumerate((r"$(x_1,x_2)$", "$y$", r"$\phi_1$", r"$\phi_2$")):
        g.text([60, 220, 340, 520][k], 118, [150, 110, 160, 160][k], 40, h, size=22, color=MUTED, bold=True, align="center", vcenter=True, z=8)
    g.text(60, 160, 700, 34, T("兩個高斯:中心 $(0,0)$、$(1,1)$,$2\\sigma^2=1$", "two Gaussians: centers $(0,0)$, $(1,1)$, $2\\sigma^2=1$"), size=18, color=MUTED, bold=True, z=8) if False else None
    for i, (p, y, r) in enumerate(zip(PTS, YS, PHI)):
        g.beat = 1 if i == 0 else 2
        yy = 164 + i * 76
        mid = i in (1, 2)
        g.rect(50, yy, 680, 64, fill=(GRNL if y else CARD), stroke=(GRN if y else GRAY), sw=3, radius=12, z=3)
        for k, t in enumerate((f"$({p[0]},{p[1]})$", f"${y}$", f"${r[0]:.3f}$".replace(".000", "").replace("1.000", "1"), f"${r[1]:.3f}$".replace("1.000", "1"))):
            g.text([60, 220, 340, 520][k], yy, [150, 110, 160, 160][k], 64, t, size=24, color=INK, bold=True, align="center", vcenter=True, z=8)
    x0, y0, side = 790, 130, 440; sc = side / 1.3
    u = lambda v: x0 + 40 + v * sc * 1.0; vv = lambda v: y0 + side - 40 - v * sc
    g.beat = 2
    g.rect(x0, y0, side, side, fill=CARD, stroke=GRAY, sw=3, radius=14, z=1)
    g.text(x0, y0 + side + 4, side, 36, T("新座標 $(\\phi_1,\\phi_2)$", "new coordinates $(\\phi_1,\\phi_2)$"), size=20, color=INK, bold=True, align="center", z=8)
    seen = set()
    for p, y, r in zip(PTS, YS, PHI):
        key = (round(r[0], 3), round(r[1], 3))
        if key in seen: continue
        seen.add(key)
        g.ellipse(u(r[0]), vv(r[1]), 20, fill=(GRN if y else CARD), stroke=(GRN if y else BLUE), sw=4, z=6)
    g.text(u(0.368) - 100, vv(0.368) + 26, 220, 60, T("$(0,1)$ 與 $(1,0)$ 重疊", "$(0,1)$ and $(1,0)$ overlap"), size=17, color=MUTED, bold=True, z=8)
    g.beat = 3
    g.line([(u(0.0), vv(THR)), (u(THR), vv(0.0))], RED, sw=5, dash=True, z=5)
    banner(g, T("$(0,1)$ 與 $(1,0)$ 落在同一點 $(0.368,0.368)$，一條線就分開", "$(0,1)$ and $(1,0)$ land on one point $(0.368,0.368)$; one line separates"))
    pages.append((g, T("基函數空間", "Basis space"), {
        0: T("取兩個高斯，中心 0 0 與 1 1。點 0 0 的座標是 1 與 0.135。", "Take two Gaussians with centers 0 0 and 1 1. The point 0 0 has coordinates 1 and 0.135."),
        1: T("0 1 與 1 0 都是 0.368、0.368，落在同一點；1 1 是 0.135 與 1。畫出新座標，兩個類別 1 的點重疊在中間。", "0 1 and 1 0 are both 0.368, 0.368, the same point; 1 1 is 0.135 and 1. In the new coordinates the two class-1 points coincide in the middle."),
        2: T("0 0 與 1 1 在兩側，所以一條直線就能分開。", "0 0 and 1 1 sit on either side, so one straight line separates them."),
        3: T("", "")}))

    # ============ 4: least squares ============
    g = new(4, T("第二層一次解出:最小二乘", "The second layer in one shot: least squares"))
    g.beat = 1
    g.rect(190, 120, 900, 120, fill=BLUEL, stroke=BLUE, sw=3, radius=16, z=3)
    g.text(200, 120, 880, 120, r"$\mathbf{W}=(\Phi\Phi^\top)^{-1}\Phi\mathbf{Y}^\top$", size=42, color=INK, align="center", vcenter=True, z=8)
    g.beat = 2
    g.rect(50, 270, 560, 190, fill=GRNL, stroke=GRN, sw=3, radius=16, z=3)
    g.text(50, 278, 560, 44, T("XOR 的解", "the XOR solution"), size=24, color=GRN, bold=True, align="center", z=8)
    g.text(60, 330, 540, 120, T("$\\mathbf{W}=(-2.503,\\,-2.503)$\n偏置 $2.841$", "$\\mathbf{W}=(-2.503,\\,-2.503)$\nbias $2.841$"), size=28, color=INK, align="center", z=8)
    g.rect(650, 270, 580, 190, fill=ORGL, stroke=ORG, sw=3, radius=16, z=3)
    g.text(650, 278, 580, 44, T("四個輸出", "the four outputs"), size=24, color=ORG, bold=True, align="center", z=8)
    g.text(660, 330, 560, 110, "$0,\\ 1,\\ 1,\\ 0$", size=44, color=INK, bold=True, align="center", vcenter=True, z=8)
    g.beat = 3
    g.rect(50, 490, 1180, 120, fill="#fef3c7", stroke=GOLD, sw=3, radius=16, z=3)
    g.text(60, 496, 1160, 108, T("分界是輸出 $=0.5$,即 $\\phi_1+\\phi_2=0.936$\n類別 0 的和是 $1.135$,類別 1 的和是 $0.736$", "boundary: output $=0.5$, i.e. $\\phi_1+\\phi_2=0.936$\nclass 0 has sum $1.135$, class 1 has sum $0.736$"), size=26, color=INK, bold=True, align="center", z=8)
    banner(g, T("對 $\\mathbf{W}$ 是線性的，所以一次解出，不必反覆下降", "linear in $\\mathbf{W}$, so one shot, no repeated descent"))
    pages.append((g, T("最小二乘", "Least squares"), {
        0: T("因為第一層固定，剩下的問題對 W 是線性的，令導數為零，一次解出 W。", "Because the first layer is fixed, what remains is linear in W; set the derivative to zero and solve W in one shot."),
        1: T("XOR 加一個偏置項，解出 W 是負 2.503、負 2.503，偏置 2.841，四個輸出恰好是 0、1、1、0。", "For XOR with a bias term the solution is W equal minus 2.503, minus 2.503 with bias 2.841, and the four outputs are exactly 0, 1, 1, 0."),
        2: T("分界線是輸出等於 0.5，也就是 phi1 加 phi2 等於 0.936：類別 0 的和是 1.135，類別 1 的和是 0.736。", "The boundary is output equal 0.5, i.e. phi1 plus phi2 equals 0.936: class 0 has sum 1.135 and class 1 has sum 0.736."),
        3: T("", "")}))

    # ============ 5: compare with KP05 ============
    g = new(5, T("和 KP05 的隱藏層是同一個想法", "The same idea as KP05's hidden layer"))
    g.beat = 1
    a = g.mark(); g.rect(60, 150, 520, 200, fill=ORGL, stroke=ORG, sw=3, radius=16, z=3)
    g.text(60, 160, 520, 44, T("KP05 的隱藏層", "KP05's hidden layer"), size=26, color=ORG, bold=True, align="center", z=8); g.text(70, 214, 500, 120, T("先把點搬到新座標\n再用一條線分", "move the points to new coordinates\nthen separate with one line"), size=24, color=INK, bold=True, align="center", z=8)
    b2 = g.mark(); g.rect(700, 150, 520, 200, fill=PURL, stroke=PUR, sw=3, radius=16, z=3)
    g.text(700, 160, 520, 44, T("RBF 的第一層", "RBF's first layer"), size=26, color=PUR, bold=True, align="center", z=8); g.text(710, 214, 500, 120, T("先把點搬到新座標\n再用一條線分", "move the points to new coordinates\nthen separate with one line"), size=24, color=INK, bold=True, align="center", z=8)
    g.text(590, 220, 100, 60, "=", size=50, color=GOLD, bold=True, align="center", z=8)
    g.beat = 2
    g.rect(60, 400, 520, 150, fill=REDL, stroke=RED, sw=3, radius=16, z=3)
    g.text(70, 410, 500, 130, T("第一層的權重要找\n(第 3 章反向傳播)", "first-layer weights must be found\n(Chapter 3 backpropagation)"), size=24, color=RED, bold=True, align="center", z=8)
    g.rect(700, 400, 520, 150, fill=GRNL, stroke=GRN, sw=3, radius=16, z=3)
    g.text(710, 410, 500, 130, T("第一層事先固定\n(K-means)，只學第二層", "first layer fixed beforehand\n(K-means); learn only the second"), size=24, color=GRN, bold=True, align="center", z=8)
    banner(g, T("差別只在:第一層是要找的，還是事先固定的", "the only difference: is the first layer searched for, or fixed beforehand"))
    pages.append((g, T("與 KP05 對照", "Versus KP05"), {
        0: T("這和 KP05 的隱藏層是同一個想法：先把點搬到新座標，再用一條直線分。", "This is the same idea as KP05's hidden layer: move the points to new coordinates, then separate with a line."),
        1: T("差別是 KP05 的第一層權重要找，RBF 的第一層是事先固定的。", "The difference: KP05's first-layer weights must be found, while RBF's first layer is fixed beforehand."),
        2: T("", ""), 3: T("", "")}))

    # ============ 6: SOM idea ============
    g = new(6, T("SOM:不用標籤，神經元自己排成地圖", "SOM: no labels, neurons arrange themselves into a map"))
    g.beat = 1
    ns = []
    for j in range(3):
        x = 220 + j * 340; ns.append(g.mark()); g.ellipse(x, 280, 70, fill=ORGL, stroke=ORG, sw=4, z=3); g.text(x - 70, 210, 140, 140, f"$\\mathbf{{w}}_{j + 1}$", size=28, color=ORG, bold=True, align="center", vcenter=True, z=8)
        if j: g.link(ns[j - 1], ns[j], 4, 12, GRAY, 4)
    g.text(120, 380, 1040, 56, T("神經元排成預設的格子，每顆是一個簇的原型(權重向量)", "neurons sit on a preset grid; each is a prototype (weight vector) of a cluster"), size=22, color=MUTED, bold=True, z=8)
    steps = [T("1  初始化:小的隨機權重，鄰域設為格子的一半", "1  initialize: small random weights, neighborhood half the grid"), T("2  選一筆輸入，找勝者:距離最近的神經元", "2  pick an input, find the winner: the nearest neuron"), T("3  更新勝者鄰域內的神經元，鄰域外不動", "3  update the neurons in the winner's neighborhood; others stay"), T("4  學習率衰減，鄰域縮小;5 回到第 2 步", "4  decay the learning rate, shrink the neighborhood; 5 go to step 2")]
    for i, s_ in enumerate(steps):
        g.beat = 2 + i // 2
        y = 440 + (i % 2) * 0 + 0
    g.beat = 2
    g.rect(60, 440, 560, 160, fill=BLUEL, stroke=BLUE, sw=3, radius=14, z=3); g.text(70, 446, 540, 150, steps[0] + "\n" + steps[1], size=22, color=INK, bold=True, z=8)
    g.beat = 3
    g.rect(660, 440, 560, 160, fill=GRNL, stroke=GRN, sw=3, radius=14, z=3); g.text(670, 446, 540, 150, steps[2] + "\n" + steps[3], size=22, color=INK, bold=True, z=8)
    banner(g, T("Kohonen 1982:無監督的單層網路，測試時勝者就是所屬的簇", "Kohonen 1982: unsupervised single layer; at test time the winner is the cluster"))
    pages.append((g, T("SOM 的流程", "How SOM works"), {
        0: T("SOM 是自組織映射，沒有標籤。輸入接到幾顆神經元，每顆是一個簇的原型，也就是權重向量，排成預設的格子。", "SOM, the self-organizing map, uses no labels. The input connects to several neurons, each a cluster prototype, that is a weight vector, arranged on a preset grid."),
        1: T("訓練前兩步：用小的隨機權重初始化；選一筆輸入，找勝者，也就是距離最近的神經元。", "The first two training steps: initialize with small random weights; pick an input and find the winner, the nearest neuron."),
        2: T("後面：更新勝者鄰域內的神經元，鄰域外不動；學習率衰減、鄰域縮小，回到第二步重複。", "Then: update the neurons in the winner's neighborhood, leaving others; decay the learning rate, shrink the neighborhood, and repeat from step 2."),
        3: T("", "")}))

    # ============ 7: SOM by hand ============
    g = new(7, T("SOM 手算一步:鄰居也被拉過去", "One SOM step: the neighbors are pulled too"))
    x0, y0, side = 60, 130, 460; sc = side / 1.15
    u = lambda v: x0 + 40 + v * sc; vv = lambda v: y0 + side - 40 - v * sc
    g.beat = 1
    g.rect(x0, y0, side, side, fill=CARD, stroke=GRAY, sw=3, radius=14, z=1)
    for j, w in enumerate(W0): g.ellipse(u(w[0]), vv(w[1]), 20, fill=ORGL, stroke=ORG, sw=4, z=6); g.text(u(w[0]) - 60, vv(w[1]) + 22, 120, 34, f"$\\mathbf{{w}}_{j + 1}$", size=20, color=ORG, bold=True, align="center", z=8)
    g.ellipse(u(X[0]), vv(X[1]), 16, fill=GOLD, stroke=INK, sw=3, z=7); g.text(u(X[0]) + 16, vv(X[1]) - 38, 200, 34, T("輸入 $(0.9,0.7)$", "input $(0.9,0.7)$"), size=18, color=INK, bold=True, z=8)
    g.beat = 2
    for j, d in enumerate(DIST):
        yy = 150 + j * 70
        col = GRN if j == WIN else GRAY
        g.rect(580, yy, 650, 58, fill=(GRNL if j == WIN else CARD), stroke=col, sw=3, radius=12, z=3)
        g.text(590, yy, 200, 58, f"$\\mathbf{{w}}_{j + 1}$", size=24, color=INK, bold=True, align="center", vcenter=True, z=8); g.text(800, yy, 420, 58, T(f"距離 ${d:.3f}$", f"distance ${d:.3f}$") + (T("  ← 勝者", "  ← winner") if j == WIN else ""), size=24, color=(GRN if j == WIN else INK), bold=True, vcenter=True, z=8)
    g.beat = 3
    for j in (1, 2):
        g.line([(u(W0[j][0]), vv(W0[j][1])), (u(W1[j][0]), vv(W1[j][1]))], GRN, sw=5, arrow="end", head=16, z=5)
        g.ellipse(u(W1[j][0]), vv(W1[j][1]), 14, fill=GRNL, stroke=GRN, sw=3, z=6)
    g.rect(580, 380, 650, 200, fill="#fef3c7", stroke=GOLD, sw=3, radius=16, z=3)
    g.text(590, 384, 630, 72, T("$\\eta=0.5$,鄰域 = 勝者與它左邊一格", "$\\eta=0.5$, neighborhood = winner and its left neighbor"), size=20, color=INK, bold=True, align="center", z=8)
    g.text(590, 446, 630, 130, "$\\mathbf{w}_3\\to(0.85,\\,0.75)$\n$\\mathbf{w}_2\\to(0.7,\\,0.6)$\n" + T("$\\mathbf{w}_1$ 在鄰域外，不變", "$\\mathbf{w}_1$ is outside the neighborhood: unchanged"), size=22, color=INK, bold=True, align="center", z=8)
    banner(g, T("不只勝者，鄰居也被拉向輸入:相鄰神經元最後代表相近的資料", "not only the winner: neighbors are pulled too, so adjacent neurons end up representing similar data"))
    pages.append((g, T("SOM 一步", "One SOM step"), {
        0: T("三顆神經元排成一列，權重 0.2、0.2，0.5、0.5，0.8、0.8。輸入是 0.9、0.7。", "Three neurons in a row with weights 0.2, 0.2, then 0.5, 0.5, then 0.8, 0.8. The input is 0.9, 0.7."),
        1: T("歐氏距離是 0.860、0.447、0.141，勝者是神經元 3。", "The Euclidean distances are 0.860, 0.447, 0.141; the winner is neuron 3."),
        2: T("學習率 0.5，鄰域是勝者與它左邊一格。神經元 3 變成 0.85、0.75，神經元 2 變成 0.7、0.6，神經元 1 在鄰域外不變。這叫拓撲保序。", "Learning rate 0.5, the neighborhood is the winner and its left neighbor. Neuron 3 becomes 0.85, 0.75, neuron 2 becomes 0.7, 0.6, and neuron 1 is outside and unchanged. This is topology preservation."),
        3: T("", "")}))

    # ============ 8: myths + question ============
    g = new(8, T("兩個迷思，與一個問題", "Two myths and a question"))
    g.beat = 1
    g.rect(50, 130, 570, 220, fill=REDL, stroke=RED, sw=3, radius=16, z=3)
    g.text(60, 138, 550, 56, T("迷思", "myth"), size=24, color=RED, bold=True, align="center", z=8)
    g.text(60, 194, 550, 150, T("「RBF 的第一層也要用反向傳播訓練」\n不用。第一層由 K-means 事先決定，只有第二層用最小二乘解", "“RBF's first layer is also trained by backpropagation”\nno: K-means fixes it beforehand; only the second layer is solved by least squares"), size=22, color=INK, bold=True, align="center", z=8)
    g.beat = 2
    g.rect(660, 130, 570, 220, fill=ORGL, stroke=ORG, sw=3, radius=16, z=3)
    g.text(670, 138, 550, 56, T("迷思", "myth"), size=24, color=ORG, bold=True, align="center", z=8)
    g.text(670, 194, 550, 150, T("「SOM 的輸出是類別預測」\n不是。沒有標籤也沒有輸出層，勝者只告訴你屬於哪一簇", "“SOM outputs a class prediction”\nno: there are no labels and no output layer; the winner only tells which cluster"), size=22, color=INK, bold=True, align="center", z=8)
    g.beat = 3
    g.rect(50, 390, 1000, 220, fill="#fef3c7", stroke=GOLD, sw=3, radius=16, z=3)
    g.text(70, 400, 960, 200, T("看過 M-P、Perceptron、ADALINE、MLP、RBF、SOM，共通的是「疊起來的神經元」。只用一層隱藏層、夠多神經元，到底能表示多少種函數？", "We have seen M-P, Perceptron, ADALINE, MLP, RBF and SOM; what they share is “stacked neurons”. With a single hidden layer and enough neurons, how many functions can be represented?"), size=24, color=INK, bold=True, align="center", z=8)
    mascot(g, 1100, 400, 0.55, "confused")
    banner(g, T("下一個單元:萬有逼近定理", "next unit: universal approximation"))
    pages.append((g, T("迷思與問題", "Myths and question"), {
        0: T("第一個迷思：RBF 的第一層也要用反向傳播訓練。不用，第一層由 K-means 事先決定。", "First myth: RBF's first layer is also trained by backpropagation. No, K-means fixes it beforehand."),
        1: T("第二個迷思：SOM 的輸出是類別預測。不是，沒有標籤也沒有輸出層。", "Second myth: SOM outputs a class prediction. No, it has no labels and no output layer."),
        2: T("留給下一個單元：只用一層隱藏層、夠多神經元，到底能表示多少種函數？這就是萬有逼近定理。", "For the next unit: with one hidden layer and enough neurons, how many functions can be represented? That is the universal approximation theorem."),
        3: T("", "")}))
    return pages
