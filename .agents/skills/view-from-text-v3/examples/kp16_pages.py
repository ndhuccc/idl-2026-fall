"""KP16 pages designed in HTML (view-from-text-v3, route A). pageN(lang) returns the body HTML of a 1160x470 figure."""
import os, sys, math
import numpy as np
SK = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__)); HERE = HERE[:-5] if HERE.endswith(("/W_zh", "/W_en")) else HERE
sys.path.insert(0, SK + "/view-from-text-v3/examples"); sys.path.insert(0, SK + "/view-from-text-v3/tools"); sys.path.insert(0, HERE)
from kp17_pages import Axes, tr, take, legend, fmt, sci, txt, named, card, head, chip, formula, svg, W, H, BL, RD, VI, GR, YE, OR, TE, INK, TXT, MUT, hexc, build, native, _sup
import kp16_data as D

def spts(a, n=60):
    idx = np.linspace(0, len(a) - 1, n).astype(int); return a[idx]
def sq(ax, x, y, col, s=12): return f'<rect x="{ax.X(x)-s:.1f}" y="{ax.Y(y)-s:.1f}" width="{2*s}" height="{2*s}" rx="3" fill="{col}"/>'
def ring_mark(ax, x, y, r=12, col="#e0c85f"): return f'<circle cx="{ax.X(x):.1f}" cy="{ax.Y(y):.1f}" r="{r}" fill="none" stroke="{col}" stroke-width="3.5"/><circle cx="{ax.X(x):.1f}" cy="{ax.Y(y):.1f}" r="3" fill="{col}"/>'

# ---------------------------------------------------------------- 1: Gaussian nodes
def page1(lang):
    t = tr(lang); o = card(0, 8, 700, 400, "blue") + head(0, 8, 700, "blue", t("sigmoid 節點：一側全是高的；高斯節點：只在中心附近高", "A sigmoid node is high on one whole side; a Gaussian node only near its centre"), 34, 17)
    a1 = Axes(70, 84, 600, 108, (-4, 4), (0, 1.1)); a1.frame([-4, -2, 0, 2, 4], [0, 1], yfmt=lambda v: f"{v:g}")
    xs = np.linspace(-4, 4, 80); a1.line(list(zip(xs, 1 / (1 + np.exp(-2.2 * xs)))), RD, 4)
    o += txt(64, 52, 560, t("一個 sigmoid 節點（全域）", "one sigmoid node (global)"), 16, RD, True, "left", 24)
    a2 = Axes(70, 276, 600, 100, (-4, 4), (0, 1.1)); a2.frame([-4, -2, 0, 2, 4], [0, 1], yfmt=lambda v: f"{v:g}")
    for c, col in ((-2.4, BL), (0, GR), (2.4, OR)):
        a2.line(list(zip(xs, np.exp(-(xs - c) ** 2 / 1.0))), col, 4)
        a2.raw(f'<line x1="{a2.X(c):.1f}" y1="{a2.Y(0):.1f}" x2="{a2.X(c):.1f}" y2="{a2.Y(0)+10:.1f}" stroke="{col}" stroke-width="3"/>')
    o += txt(64, 236, 600, t("三個高斯節點（局部）：各在自己的中心附近", "three Gaussian nodes (local): each near its own centre"), 16, TXT, True, "left", 24)
    a2.raw(f'<text x="{a2.X(0):.1f}" y="{a2.Y(0)+44:.1f}" text-anchor="middle" font-size="16" fill="#b7b0a5">x</text>')
    o += a1.html() + a2.html()
    o += card(716, 8, 444, 400, "violet") + head(716, 8, 444, "violet", t("RBF 網路", "RBF network"), 34, 17)
    o += formula(724, 52, 428, r"g(x)=w_0+\sum_{i=1}^{k}w_i\exp\!\left(-\frac{\|x-c_i\|^2}{2\sigma_i^2}\right)", 20, 70)
    o += named(732, 132, 412, t("徑向基函數（RBF）：以「到中心的距離」為引數的函數，最常用高斯形。", "Radial basis function (RBF): a function of the distance to a centre; the Gaussian form is the most common."), "term:" + t("徑向基函數", "Radial basis function"), 16, TXT, 72)
    o += txt(732, 214, 412, t("一層高斯隱藏節點，加線性輸出。σ 越大，回應範圍越廣。", "One layer of Gaussian hidden nodes plus a linear output. The larger σ, the wider the response."), 16, TXT, False, "left", 72)
    o += card(732, 290, 412, 56, "yellow", .14) + txt(740, 296, 396, t("與 Parzen 估計很像，但 Parzen 是 k = N，這裡 k ≪ N", "Close to a Parzen estimate, but Parzen has k = N and here k ≪ N"), 16, TXT, True, "center", 48)
    o += txt(732, 346, 412, t("例：x = (0.5, 0.5)、c = (0, 0) 時 ‖x−c‖² = 0.25 + 0.25 = 0.5", "Example: x = (0.5, 0.5), c = (0, 0) gives ‖x−c‖² = 0.25 + 0.25 = 0.5"), 16, MUT, False, "left", 48)
    o += take(t, "把「在哪一側」換成「離中心多近」：每個節點只對自己附近的資料有反應。", "Replace “which side” by “how near a centre”: each node reacts only to data near itself.")
    return o

# ---------------------------------------------------------------- 2: local vs global
def page2(lang):
    t = tr(lang); o = ""
    o += card(0, 8, 372, 400, "red") + head(0, 8, 372, "red", t("sigmoid：等值線是直線", "sigmoid: iso-lines are straight"), 34, 17)
    ax = Axes(36, 56, 300, 220, (0, 1), (0, 1)); s = ""
    for c in (-0.5, -0.25, 0, 0.25, 0.5):
        x0, x1 = max(0, c), min(1, 1 + c); s += f'<line x1="{ax.X(x0):.1f}" y1="{ax.Y(x0-c):.1f}" x2="{ax.X(x1):.1f}" y2="{ax.Y(x1-c):.1f}" stroke="{RD}" stroke-width="3" opacity="{0.45+0.1*abs(c)*4}"/>'
    o += svg(0, 0, W, H, f'<rect x="{ax.x}" y="{ax.y}" width="{ax.w}" height="{ax.h}" fill="none" stroke="#6b645a" stroke-width="2"/>' + s)
    o += named(14, 290, 344, t("全域反應：沿整個超平面回應；二維的等值線是平行直線。", "Global response: a node responds along a whole hyperplane; in 2-D the iso-lines are parallel straight lines."), "term:" + t("全域反應", "Global response"), 16, TXT, 96)
    o += card(386, 8, 372, 400, "green") + head(386, 8, 372, "green", t("RBF：等值線是同心圓", "RBF: iso-lines are concentric circles"), 34, 17)
    ax2 = Axes(422, 56, 300, 220, (0, 1), (0, 1)); cx, cy = ax2.X(0.5), ax2.Y(0.5); s = ""
    for r in (22, 46, 70, 94):
        s += f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="none" stroke="{GR}" stroke-width="3" opacity="{1-r/160:.2f}"/>'
    s += f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="8" fill="#e0c85f"/>'
    o += svg(0, 0, W, H, f'<rect x="{ax2.x}" y="{ax2.y}" width="{ax2.w}" height="{ax2.h}" fill="none" stroke="#6b645a" stroke-width="2"/>' + s)
    o += named(400, 290, 344, t("局部反應：節點只在中心附近回應，離開就趨近 0；二維的等值線是以中心為圓心的圓。", "Local response: a node responds only near its centre and falls to 0 farther away; in 2-D the iso-lines are circles around the centre."), "term:" + t("局部反應", "Local response"), 16, TXT, 120)
    o += card(772, 8, 388, 400, "orange") + head(772, 8, 388, "orange", t("高斯回應 e^(−d²)（2σ² = 1）", "Gaussian response e^(−d²) (2σ² = 1)"), 34, 17)
    vals = [(d, math.exp(-d * d)) for d in (0, 1, 2, 3)]
    o += txt(790, 56, 80, t("距離 d", "d"), 16, MUT, True, "center", 24) + txt(890, 56, 250, t("回應", "response"), 16, MUT, True, "center", 24)
    for i, (d, v) in enumerate(vals):
        y0 = 96 + i * 62; o += txt(790, y0 + 6, 80, str(d), 20, TXT, True, "center", 28)
        wbar = max(2, 190 * v); o += f'<div style="position:absolute;left:890px;top:{y0+6}px;width:{wbar:.0f}px;height:28px;background:{OR};border-radius:6px"></div>' if v > 0.01 else ""
        o += txt(890 + (wbar + 8 if v > 0.01 else 0), y0 + 5, min(130, 1150 - (890 + (wbar + 8 if v > 0.01 else 0))), sci(v) if v < 0.5 else str(int(v)), 18, TXT, True, "left", 28)
    o += txt(786, 352, 360, t("離中心 3 個單位，回應只剩萬分之一。〔程式計算〕", "Three units from the centre the response is about one in ten thousand. (computed)"), 16, MUT, False, "left", 48)
    o += take(t, "sigmoid 的等值線是平行直線（全域），RBF 的是同心圓（局部）。", "The iso-lines are parallel lines for a sigmoid (global) and concentric circles for an RBF (local).")
    return o

# ---------------------------------------------------------------- 3: XOR with two centres
def page3(lang):
    t = tr(lang); X, Y, g = D.xor_table(); cur = spts(D.xor_curve(), 70); o = ""
    o += card(0, 8, 440, 316, "blue") + head(0, 8, 440, "blue", t("原空間：封閉曲線 g(x) = 0", "Original space: the closed curve g(x) = 0"), 34, 17)
    ax = Axes(34, 50, 380, 240, (-0.9, 1.9), (-0.7, 1.7)); ax.raw(f'<rect x="{ax.x}" y="{ax.y}" width="{ax.w}" height="{ax.h}" fill="none" stroke="#6b645a" stroke-width="2"/>')
    ax.line([(x, y) for x, y in cur] + [(cur[0][0], cur[0][1])], "#e0c85f", 3, "9 6")
    for (cx_, cy_) in ((1, 1), (0, 0)): ax.raw(ring_mark(ax, cx_, cy_, 20, VI))
    for (x_, y_) in ((0, 0), (1, 1)): ax.dot(x_, y_, "#47688f", 11)
    for (x_, y_) in ((1, 0), (0, 1)): ax.raw(sq(ax, x_, y_, "#94503a", 11))
    ax.raw(f'<text x="{ax.X(1.45):.1f}" y="{ax.Y(1.3):.1f}" font-size="16" font-weight="700" fill="{VI}">c₁</text>'.replace("c₁", "c1") + f'<text x="{ax.X(-0.45):.1f}" y="{ax.Y(-0.25):.1f}" font-size="16" font-weight="700" fill="{VI}">c2</text>')
    o += ax.html()
    o += txt(10, 290, 420, t("曲線內 g > 0，曲線外 g < 0", "inside the curve g > 0, outside g < 0"), 16, TXT, False, "center", 24)
    o += card(454, 8, 330, 316, "teal") + head(454, 8, 330, "teal", t("特徵空間：一條直線", "Feature space: one straight line"), 34, 17)
    bx = Axes(500, 56, 250, 220, (0, 1.15), (0, 1.15)); bx.raw(f'<rect x="{bx.x}" y="{bx.y}" width="{bx.w}" height="{bx.h}" fill="none" stroke="#6b645a" stroke-width="2"/>')
    bx.line([(0, 1), (1, 0)], "#e0c85f", 3, "9 6")
    bx.dot(Y[0][0], Y[0][1], "#47688f", 11).dot(Y[1][0], Y[1][1], "#47688f", 11)
    bx.raw(sq(bx, Y[2][0], Y[2][1], "#94503a", 12))
    bx.raw(f'<text x="{bx.x+bx.w+8}" y="{bx.y+bx.h+4}" font-size="16" fill="#b7b0a5">y1</text><text x="{bx.x-12}" y="{bx.y+14}" text-anchor="end" font-size="16" fill="#b7b0a5">y2</text>')
    o += bx.html()
    o += txt(466, 294, 306, t("(1,0)、(0,1) 摺成同一點 (0.368, 0.368)", "(1,0), (0,1) fold into (0.368, 0.368)"), 16, TXT, False, "center", 24)
    o += card(798, 8, 362, 316, "orange") + head(798, 8, 362, "orange", t("逐點計算（c₁=(1,1)、c₂=(0,0)）", "Point by point (c₁=(1,1), c₂=(0,0))"), 34, 17)
    o += txt(806, 52, 70, "x", 16, MUT, True, "center", 24) + txt(876, 52, 150, "y = (y1, y2)", 16, MUT, True, "center", 24) + txt(1030, 52, 120, "g = y1+y2−1", 16, MUT, True, "center", 24)
    names = ["(0,0)", "(1,1)", "(1,0)", "(0,1)"]
    for i in range(4):
        y0 = 84 + i * 56; pos = g[i] > 0; o += card(806, y0, 346, 48, "blue" if pos else "orange", .14)
        o += txt(810, y0 + 10, 64, names[i], 17, TXT, True, "center", 26) + txt(880, y0 + 10, 150, f"({Y[i][0]:.3f}, {Y[i][1]:.3f})".replace("(1.000", "(1").replace(", 1.000)", ", 1)"), 16, TXT, False, "center", 26)
        o += chip(1040, y0 + 8, 100, "blue" if pos else "orange", ("+" if pos else "−") + f"{abs(g[i]):.3f}", True, 32, 17)
    o += txt(806, 312 - 76, 346, t("", ""), 16, MUT, False, "center", 24) if False else ""
    o += card(0, 336, 784, 72, "yellow", .14) + txt(10, 342, 764, t("遠離所有中心時兩個指數項都趨近 0，g → −1：遠處一律歸曲線外那一類（局部反應）", "Far from every centre both exponentials go to 0, so g → −1:<br>far points fall in the outside class (local response)"), 16, TXT, False, "center", 48)
    o += chip(798, 336, 362, "violet", t("設定：k=2，2σ²=1", "Setup: k = 2, 2σ² = 1"), False, 72, 16)
    o += take(t, "直線 y1+y2=1 回到原空間，就是圍住兩個中心的封閉曲線：兩個中心就解了 XOR。", "The line y1+y2=1 is, in the original space, a closed curve around the two centres: two centres solve XOR.")
    return o

# ---------------------------------------------------------------- 4: fixed centres (rings)
def page4(lang):
    t = tr(lang); Xtr, ytr, C8, acc = D.rings_k(); o = ""
    o += card(0, 8, 500, 400, "blue") + head(0, 8, 500, "blue", t("同心環資料與 8 個隨機中心", "Concentric rings and 8 random centres"), 34, 17)
    ax = Axes(40, 54, 300, 300, (-2.8, 2.8), (-2.8, 2.8))
    ax.raw(f'<rect x="{ax.x}" y="{ax.y}" width="{ax.w}" height="{ax.h}" fill="none" stroke="#6b645a" stroke-width="2"/>')
    for (x_, y_), yy in zip(Xtr, ytr): ax.dot(x_, y_, "#6a9fd6" if yy > 0 else "#d97757", 4)
    for x_, y_ in C8: ax.raw(ring_mark(ax, x_, y_, 11))
    o += ax.html()
    o += legend([("#6a9fd6", t("A 類（內圈）", "class A (inner)")), ("#d97757", t("B 類（外圈）", "class B (outer)")), ("#e0c85f", t("中心", "centre"))], 350, 90, 140)
    o += named(14, 360, 474, t("固定中心：從訓練資料隨機挑 k 個點當中心；中心固定後只剩 w 要學，是最小二乘的線性問題。", "Fixed centres: pick k training points at random; then only w is learned, by least squares."), "term:" + t("固定中心", "Fixed centres"), 16, TXT, 48)
    o += card(516, 8, 644, 400, "green") + head(516, 8, 644, "green", t("新點正確率對中心數 k（同心環、隨機中心）", "New-point accuracy against the number of centres k (rings, random centres)"), 34, 17)
    bx = Axes(590, 66, 540, 236, (0, 6), (0.4, 1.05)); bx.frame([], [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], yfmt=lambda v: f"{int(round(v*100))}%")
    ks = [1, 2, 4, 8, 16, 32]
    pts = [(i + 0.5, acc[k]) for i, k in enumerate(ks)]; bx.line(pts, GR, 4)
    for i, (x_, y_) in enumerate(pts):
        bx.dot(x_, y_, GR, 9); bx.text(x_, y_, f"{y_*100:.1f}%", TXT, "middle", 16, True, dy=-16)
        bx.raw(f'<text x="{bx.X(x_):.1f}" y="{bx.y+bx.h+24}" text-anchor="middle" font-size="16" fill="#b7b0a5">{ks[i]}</text>')
    bx.raw(f'<text x="{bx.x+bx.w/2}" y="{bx.y+bx.h+52}" text-anchor="middle" font-size="16" font-weight="700" fill="#f2ede4">{t("中心數 k", "number of centres k")}</text>')
    o += bx.html()
    o += txt(530, 376, 620, t("200 個訓練點、1000 個新點，每個 k 重複 20 次取平均。〔程式計算〕", "200 training points, 1000 new points, 20 repeats per k, averaged. (computed)"), 16, MUT, False, "left", 24)
    o += take(t, "中心越多越能圍出內圈：k = 8 快速爬升，k = 16 已接近全對（99.5%）。", "More centres enclose the inner ring better: a quick rise at k = 8 and almost all correct at k = 16 (99.5%).")
    return o

# ---------------------------------------------------------------- 5: centres by clustering
def page5(lang):
    t = tr(lang); Xtr, ytr, C8, cmp = D.moons_cmp(); o = ""
    o += card(0, 8, 440, 400, "blue") + head(0, 8, 440, "blue", t("兩個月牙與 k-means 的 8 個中心", "Two moons and 8 k-means centres"), 34, 17)
    ax = Axes(30, 54, 380, 250, (-1.6, 2.6), (-1.1, 1.6)); ax.raw(f'<rect x="{ax.x}" y="{ax.y}" width="{ax.w}" height="{ax.h}" fill="none" stroke="#6b645a" stroke-width="2"/>')
    for (x_, y_), yy in zip(Xtr, ytr): ax.dot(x_, y_, "#6a9fd6" if yy > 0 else "#d97757", 4)
    for x_, y_ in C8: ax.raw(ring_mark(ax, x_, y_, 11))
    o += ax.html()
    o += named(14, 316, 412, t("分群：先把資料分成 k 群，每群的平均當中心，再用最小二乘學 w（非監督加監督）。", "Clustering: split the data into k groups,<br>use each group mean as a centre,<br>then learn w by least squares."), "term:" + t("分群", "Clustering"), 16, TXT, 96)
    o += card(456, 8, 704, 400, "green") + head(456, 8, 704, "green", t("隨機中心與 k-means 的新點正確率", "New-point accuracy: random centres vs k-means"), 34, 17)
    bx = Axes(540, 62, 560, 186, (0, 5), (0.7, 1.0)); bx.frame([], [0.7, 0.8, 0.9, 1.0], yfmt=lambda v: f"{int(round(v*100))}%")
    ks = [2, 4, 6, 8, 12]; r_ = [(i + 0.5, cmp[k][0]) for i, k in enumerate(ks)]; k_ = [(i + 0.5, cmp[k][1]) for i, k in enumerate(ks)]
    bx.line(r_, OR, 4).line(k_, GR, 4)
    for i, k in enumerate(ks):
        bx.dot(i + 0.5, cmp[k][0], OR, 8).dot(i + 0.5, cmp[k][1], GR, 8)
        up = cmp[k][1] >= cmp[k][0]
        bx.text(i + 0.5, cmp[k][1], f"{cmp[k][1]*100:.1f}", GR, "middle", 16, True, dy=-14 if up else 26)
        bx.text(i + 0.5, cmp[k][0], f"{cmp[k][0]*100:.1f}", OR, "middle", 16, True, dy=26 if up else -14)
        bx.raw(f'<text x="{bx.X(i+0.5):.1f}" y="{bx.y+bx.h+24}" text-anchor="middle" font-size="16" fill="#b7b0a5">{k}</text>')
    bx.raw(f'<text x="{bx.x+bx.w/2}" y="{bx.y+bx.h+52}" text-anchor="middle" font-size="16" font-weight="700" fill="#f2ede4">{t("中心數 k", "number of centres k")}</text>')
    o += bx.html() + legend([(OR, t("隨機中心", "random centres")), (GR, "k-means")], 1000, 270, 150) if False else bx.html()
    o += f'<div style="position:absolute;left:560px;top:326px;width:26px;height:5px;background:{OR};border-radius:3px"></div>' + txt(592, 317, 150, t("隨機中心", "random centres"), 16, TXT, False, "left", 24) + f'<div style="position:absolute;left:760px;top:326px;width:26px;height:5px;background:{GR};border-radius:3px"></div>' + txt(792, 317, 120, "k-means", 16, TXT, False, "left", 24)
    o += card(472, 346, 672, 54, "yellow", .14) + txt(480, 351, 656, t(f"迷思：分群一定比隨機準。k = 2、4 隨機反而略高；k ≥ 6 分群較高，主要是更穩（k = 8 標準差 {cmp[8][3]:.3f} 對 {cmp[8][2]:.3f}）", f"Myth: clustering always beats random. At k = 2, 4 random is slightly higher; from k ≥ 6 clustering wins, mainly by being steadier (k = 8 std {cmp[8][3]:.3f} vs {cmp[8][2]:.3f})"), 16, TXT, False, "center", 48)
    o += take(t, "k = 8 時，k-means 為 95.0%、隨機為 93.3%；分群的優勢是穩，不是每個 k 都準。", "At k = 8 k-means reaches 95.0% against 93.3% for random; clustering’s edge is stability, not accuracy at every k.")
    return o

# ---------------------------------------------------------------- 6: curse of dimensionality
def page6(lang):
    t = tr(lang); cov = D.coverage(); ls = (2, 5, 10, 20); o = card(0, 8, 700, 400, "red") + head(0, 8, 700, "red", t("100 個中心下，回應大於 0.1 的測試點比例", "Share of test points with response above 0.1 (100 centres)"), 34, 17)
    ax = Axes(80, 64, 590, 230, (0, 4), (0, 1.1)); ax.frame([], [0, 0.25, 0.5, 0.75, 1.0], yfmt=lambda v: f"{int(round(v*100))}%")
    gw = ax.w / 4
    for i, l in enumerate(ls):
        v = cov[l][1]; top = ax.Y(v); bx = ax.x + i * gw + gw / 2 - 40
        if v > 0: ax.raw(f'<rect x="{bx:.1f}" y="{top:.1f}" width="80" height="{ax.y+ax.h-top:.1f}" rx="6" fill="{RD}"/>')
        ax.raw(f'<text x="{bx+40:.1f}" y="{top-8:.1f}" text-anchor="middle" font-size="18" font-weight="700" fill="#f2ede4">{int(round(v*100))}%</text><text x="{bx+40:.1f}" y="{ax.y+ax.h+26}" text-anchor="middle" font-size="16" fill="#b7b0a5">l = {l}</text>')
    o += ax.html()
    o += txt(30, 328, 400, t("最近中心距離的中位數", "median distance to the nearest centre"), 16, MUT, True, "left", 24)
    for i, l in enumerate(ls): o += chip(80 + i * gw + gw / 2 - 40, 360, 80, "red", f"{cov[l][0]:.2f}", False, 30, 17)
    o += card(716, 8, 444, 400, "orange") + head(716, 8, 444, "orange", t("維度詛咒", "Curse of dimensionality"), 34, 17)
    o += named(732, 54, 412, t("維度詛咒：維度升高，要填滿空間所需的點或中心數呈指數成長，而資料與中心卻永遠不夠。", "Curse of dimensionality: as the dimension grows, the number of points or centres needed to fill the space grows exponentially, while data and centres are never enough."), "term:" + t("維度詛咒", "Curse of dimensionality"), 16, TXT, 120)
    o += txt(732, 176, 412, t("對照：每軸放 5 個中心的格子需要 5^l 個中心：", "For comparison, a grid with 5 centres per axis needs 5^l centres:"), 16, MUT, False, "left", 48)
    for i, (l, n) in enumerate(((2, 25), (5, 3125), (10, 9765625))):
        o += chip(732, 232 + i * 46, 110, "orange", f"l = {l}", False, 36, 17) + txt(850, 236 + i * 46, 290, f"{n:,}", 24, TXT, True, "left", 30)
    o += txt(732, 376, 412, t("高斯寬度 σ = 0.3，測試點 2000 個。〔程式計算〕", "Gaussian width σ = 0.3, 2000 test points. (computed)"), 16, MUT, False, "left", 24)
    o += take(t, "局部的代價是維度：l = 10 時 100 個中心只照到 40% 的點，l = 20 時一個也照不到。", "The price of locality is dimension: at l = 10 the 100 centres cover only 40% of the points and at l = 20 none.")
    return o

# ---------------------------------------------------------------- 7: MLP vs RBF
def page7(lang):
    t = tr(lang); rows = [(t("節點的引數", "argument of a node"), r"\sum_j w_j x_j", r"\|x-c_i\|"), (t("等值集合", "iso-set"), t("超平面", "hyperplane"), t("超球面", "hypersphere")), (t("反應", "response"), t("全域", "global"), t("局部", "local")),
                         (t("學習速度", "learning speed"), t("較慢", "slower"), t("較快", "faster")), (t("資料稀疏處的泛化", "generalization where data are sparse"), t("較好", "better"), t("較差", "worse")), (t("同樣效能所需節點數", "nodes needed for the same performance"), t("較少", "fewer"), t("多很多", "many more"))]
    o = card(0, 8, 1160, 400, "grey", .08)
    o += chip(0, 8, 400, "grey", "", False, 40, 16) .replace("rgba(183,176,165,0.28)", "rgba(0,0,0,0)") if False else ""
    o += head(420, 14, 340, "red", "MLP", 36, 18) + head(780, 14, 340, "green", t("RBF 網路", "RBF network"), 36, 18)
    for i, (nm, a, b) in enumerate(rows):
        y0 = 62 + i * 56; o += card(14, y0, 1132, 48, "grey", .07 + .04 * (i % 2))
        o += txt(26, y0 + 10, 380, nm, 17, TXT, True, "left", 26)
        for x0, v, col in ((420, a, "red"), (780, b, "green")):
            if i == 0: o += formula(x0, y0 + 2, 340, v, 20, 42)
            else: o += chip(x0 + 60, y0 + 7, 220, col, v, False, 34, 17)
    o += card(14, 394 - 6, 0, 0, "grey", 0) if False else ""
    o += take(t, "怎麼記：MLP 用「在超平面哪一側」思考，RBF 用「離中心多近」思考；局部學得快，但吃維度。", "MLP asks which side of a hyperplane, RBF how near a centre; local learns fast but eats dimensions.")
    return o

# ---------------------------------------------------------------- 8: other ways to choose centres
def page8(lang):
    t = tr(lang); items = [("grey", t("固定中心", "Fixed centres"), t("從訓練點隨機挑", "pick training points at random")), ("blue", t("訓練中心", "Trained centres"), t("w、c、σ 全用梯度下降，成本高", "w, c and σ all by gradient descent; costly")),
                           ("green", t("分群", "Clustering"), t("k-means，群平均當中心", "k-means; group means become centres")), ("violet", t("正交最小平方", "Orthogonal least squares"), t("先給一大堆候選，前向回歸挑出精簡中心，順便估出 k", "start from many candidates and pick a compact set by forward regression, which also estimates k")),
                           ("orange", t("支持向量", "Support vectors"), t("把 RBF 網路看成核映射，訓練是二次規劃、保證全域最佳，中心個數自動決定", "view the RBF network as a kernel map; training is a quadratic programme with a global optimum and the number of centres is chosen automatically")), ("teal", t("成長式", "Growing"), t("新樣本離所有中心都遠、誤差又大，就成為新中心", "a new sample far from all centres with a large error becomes a new centre"))]
    o = ""
    for i, (role, nm, how) in enumerate(items):
        x = (i % 3) * 392; y = 8 + (i // 3) * 200; o += card(x, y, 376, 188, role) + head(x, y, 376, role, nm, 38, 18) + txt(x + 14, y + 54, 348, how, 19, TXT, False, "left", 120)
    o += take(t, "六種選法，從最簡單的隨機挑，到保證全域最佳的支持向量。", "Six ways to choose centres, from the simplest random pick to support vectors with a global optimum.")
    return o

# ---------------------------------------------------------------- 9: exercises
def page9(lang):
    t = tr(lang); Y, g = D.practice(); o = ""
    qs = [("blue", t("題 1", "Q1"), t("把 (0.5, 0.5) 映射到 y 並判類。提示：到兩個中心的距離平方都是 0.5。", "Map (0.5, 0.5) to y and classify it. Hint: the squared distance to both centres is 0.5."), [r"y=(e^{-0.5},e^{-0.5})=(0.607,\,0.607)", r"g=0.607+0.607-1=+0.213"], t("g > 0，屬 ω₁。", "g > 0, class ω₁."), t("中間點在兩個中心之間，被圍進曲線內。", "The middle point lies between the centres, inside the curve.")),
          ("red", t("題 2", "Q2"), t("對遠點 (3, 3) 做同樣的事。", "Do the same for the far point (3, 3)."), [r"d^2=8,\ 18", r"y=(0.0003,\,1.5\times10^{-8}),\quad g\approx-1.0"], t("g < 0，屬 ω₂。", "g < 0, class ω₂."), t("遠點落到偏置 −1，歸預設那一類。", "A far point falls to the bias −1 and takes the default class."))]
    assert abs(g[0] - 0.213) < 1e-3 and abs(g[1] + 1) < 1e-3
    for i, (role, name, q, tex, ans, chk) in enumerate(qs):
        x = i * 590; o += card(x, 8, 570, 400, role) + head(x, 8, 570, role, name, 36, 18)
        o += txt(x + 16, 56, 538, q, 18, TXT, True, "left", 80)
        o += chip(x + 16, 134, 130, "grey", t("解題示範", "Worked answer"), False, 30, 16)
        o += card(x + 16, 176, 538, 96, "grey", .1) + formula(x + 16, 182, 538, tex[0], 20, 40) + formula(x + 16, 226, 538, tex[1], 20, 40)
        o += chip(x + 16, 288, 538, role, ans, True, 40, 19)
        o += txt(x + 16, 346, 538, chk, 16, MUT, False, "left", 48)
    o += take(t, "設定：c₁ = (1,1)、c₂ = (0,0)、y = exp(−‖x−c‖²)、g = y1 + y2 − 1（g > 0 為 ω₁）。", "Setup: c₁ = (1,1), c₂ = (0,0), y = exp(−‖x−c‖²), g = y1 + y2 − 1 (g > 0 means ω₁).")
    return o

PAGES_DESIGN = [("p1", page1), ("p2", page2), ("p3", page3), ("p4", page4), ("p5", page5), ("p6", page6), ("p7", page7), ("p8", page8), ("p9", page9)]
if __name__ == "__main__":
    out = sys.argv[1]; only = sys.argv[2:]
    for fid, fn in PAGES_DESIGN:
        if only and fid not in only: continue
        print(fid); build("kp16_" + fid, fn, W, H, out)
