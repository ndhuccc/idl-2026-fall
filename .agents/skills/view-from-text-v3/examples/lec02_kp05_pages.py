import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lec01_kit import *

KP = "KP05"; N = 8
PTS = [(0, 0), (0, 1), (1, 0), (1, 1)]
f = lambda s: 1 if s >= 0 else 0
def net(x1, x2, act=f):
    s1 = x1 + x2 - 1; h1 = act(s1); s2 = -x1 - x2 + 1.5; h2 = act(s2); s3 = h1 + h2 - 2; return s1, h1, s2, h2, s3, act(s3)
ROWS = [net(*p) for p in PTS]
assert [r[5] for r in ROWS] == [0, 1, 1, 0]
assert [(r[1], r[3]) for r in ROWS] == [(0, 1), (1, 1), (1, 1), (1, 0)]
LIN = [net(*p, act=lambda a: a) for p in PTS]; assert all(abs(r[4] - (-1.5)) < 1e-9 for r in LIN)
W2W1 = (1 * 1 + 1 * -1, 1 * 1 + 1 * -1); assert W2W1 == (0, 0)
fm = lambda v: ("%g" % v).replace("-", "-")

def make(lang):
    zh = lang == "zh"
    T = lambda a, b_: a if zh else b_
    pages = []
    new = lambda n, title: new_page(n, N, KP, title)

    def neuron(g, cx, cy, label, col, light, r=36):
        i = g.mark(); g.ellipse(cx, cy, r, fill=light, stroke=col, sw=3, z=3)
        g.text(cx - r, cy - r, 2 * r, 2 * r, label, size=22, color=col, bold=True, align="center", vcenter=True, z=8); return i

    # ============ 1: structure ============
    g = new(1, T("KP05 把神經元接成層，非線性才有力量", "KP05 Stack neurons into layers; nonlinearity gives power"))
    xs = [200, 600, 1000]; ys3 = [(250, 410), (190, 330, 470), (330,)]
    ins = [neuron(g, 200, y, f"$x_{k + 1}$", BLUE, BLUEL) for k, y in enumerate((260, 400))]
    g.beat = 1
    g.text(100, 530, 200, 40, T("輸入層", "input layer"), size=24, color=BLUE, bold=True, align="center", z=8)
    g.text(60, 570, 280, 56, T("接收原始資料", "receives the raw data"), size=18, color=MUTED, bold=True, align="center", z=8)
    g.beat = 2
    hid = [neuron(g, 620, y, f"$h_{k + 1}$", ORG, ORGL) for k, y in enumerate((190, 330, 470))]
    for a in ins:
        for b_ in hid: g.link(a, b_, 4, 12, GRAY, 3)
    g.text(520, 530, 200, 40, T("隱藏層", "hidden layer"), size=24, color=ORG, bold=True, align="center", z=8)
    g.text(480, 570, 280, 56, T("看不到答案，也看不到原始資料", "sees neither the answer nor the raw data"), size=18, color=MUTED, bold=True, align="center", z=8)
    g.beat = 3
    out = neuron(g, 1040, 330, "$o$", GRN, GRNL)
    for b_ in hid: g.link(b_, out, 4, 12, GRAY, 3)
    g.text(940, 530, 200, 40, T("輸出層", "output layer"), size=24, color=GRN, bold=True, align="center", z=8)
    g.text(900, 570, 280, 56, T("給出結果", "gives the result"), size=18, color=MUTED, bold=True, align="center", z=8)
    banner(g, T("全連接、只向前傳:前饋網路", "fully connected, information only forward: a feedforward network"))
    pages.append((g, T("層的結構", "Layers"), {
        0: T("把神經元接成層。輸入層接收原始資料。", "Connect neurons into layers. The input layer receives the raw data."),
        1: T("隱藏層是中間的層，看不到答案也看不到原始資料。前一層的每一顆都連到下一層的每一顆，叫全連接。", "The hidden layer is in the middle and sees neither the answer nor the raw data. Every neuron of one layer connects to every neuron of the next: fully connected."),
        2: T("輸出層給出結果。資訊只向前傳、不回頭，叫前饋。", "The output layer gives the result. Information only moves forward, never back: feedforward."),
        3: T("", "")}))

    # ============ 2: two steps per neuron ============
    g = new(2, T("每顆神經元做兩步:先線性，再非線性", "Every neuron takes two steps: linear, then nonlinear"))
    g.beat = 1
    a = g.mark(); g.rect(60, 150, 500, 200, fill=BLUEL, stroke=BLUE, sw=3, radius=16, z=3)
    g.text(60, 158, 500, 44, T("1  線性加總", "1  linear sum"), size=26, color=BLUE, bold=True, align="center", z=8); g.text(70, 210, 480, 120, r"$a_i=\sum_\ell w_{i\ell}\,z_\ell$", size=38, color=INK, align="center", vcenter=True, z=8)
    g.beat = 2
    b2 = g.mark(); g.rect(720, 150, 500, 200, fill=ORGL, stroke=ORG, sw=3, radius=16, z=3)
    g.text(720, 158, 500, 44, T("2  非線性活化", "2  nonlinear activation"), size=26, color=ORG, bold=True, align="center", z=8); g.text(730, 210, 480, 120, r"$z_i=\sigma(a_i)$", size=38, color=INK, align="center", vcenter=True, z=8)
    g.link(a, b2, 4, 12, GRAY, 5)
    g.beat = 3
    g.rect(190, 410, 900, 150, fill="#fef3c7", stroke=GOLD, sw=3, radius=16, z=3)
    g.text(210, 416, 860, 50, T("整層一起寫", "the whole layer at once"), size=22, color=MUTED, bold=True, align="center", z=8); g.text(210, 466, 860, 80, r"$\mathbf{z}^{(k)}=\sigma\!\left(\mathbf{W}_k^{\top}\mathbf{z}^{(k-1)}\right)$", size=38, color=INK, align="center", vcenter=True, z=8)
    banner(g, T("第 1 章的「模型是映射」，現在由層層映射組成", "Chapter 1's “model is a mapping” is now a composition of layer mappings"))
    pages.append((g, T("兩步", "Two steps"), {
        0: T("每顆神經元先把前一層的輸出加權相加，這是線性的一步。", "Each neuron first takes a weighted sum of the previous layer's outputs; this is the linear step."),
        1: T("再過活化函數 sigma，這是非線性的一步。", "Then it passes through the activation sigma; this is the nonlinear step."),
        2: T("整層一起寫成矩陣：z 的第 k 層等於 sigma 作用在 W 轉置乘上一層的 z。", "A whole layer is written with a matrix: layer k of z equals sigma applied to W transpose times the previous z."),
        3: T("", "")}))

    # ============ 3: building XOR ============
    g = new(3, T("用三個閘接成 XOR", "Build XOR from three gates"))
    g.beat = 1
    x1 = neuron(g, 110, 250, "$x_1$", BLUE, BLUEL); x2 = neuron(g, 110, 430, "$x_2$", BLUE, BLUEL)
    g.text(30, 480, 160, 36, T("輸入", "input"), size=20, color=BLUE, bold=True, align="center", z=8) if False else None
    h1 = neuron(g, 440, 250, "$h_1$", ORG, ORGL); h2 = neuron(g, 440, 430, "$h_2$", ORG, ORGL)
    for a in (x1, x2):
        for b_ in (h1, h2): g.link(a, b_, 4, 12, GRAY, 3)
    o = neuron(g, 760, 340, "$o$", GRN, GRNL)
    g.link(h1, o, 4, 12, GRAY, 3); g.link(h2, o, 4, 12, GRAY, 3)
    gates = [(ORG, ORGL, "h_1", "OR", T("至少一個為 1", "at least one is 1"), r"$s_1=x_1+x_2-1$"), (ORG, ORGL, "h_2", "NAND", T("不是兩個都為 1", "not both are 1"), r"$s_2=-x_1-x_2+1.5$"), (GRN, GRNL, "o", "AND", T("兩個條件都成立", "both conditions hold"), r"$s_3=h_1+h_2-2$")]
    for i, (col, light, nm, gate, desc, fml) in enumerate(gates):
        g.beat = 2 + i if i < 2 else 3
        y = 118 + i * 170
        g.rect(840, y, 400, 150, fill=light, stroke=col, sw=3, radius=14, z=3)
        g.text(850, y + 6, 380, 40, f"${nm}$:{gate}  ·  {desc}" if zh else f"${nm}$: {gate} · {desc}", size=20, color=col, bold=True, align="center", z=8)
        g.text(850, y + 56, 380, 80, fml, size=26, color=INK, align="center", vcenter=True, z=8)
    banner(g, T("XOR = 至少一個為 1(OR)而且不是兩個都為 1(NAND)", "XOR = at least one is 1 (OR) and not both are 1 (NAND)"))
    pages.append((g, T("組出 XOR", "Assembling XOR"), {
        0: T("2 2 1 的網路：兩個輸入，兩顆隱藏神經元，一顆輸出。權重用 KP01 的閘組出來。", "A 2-2-1 network: two inputs, two hidden neurons, one output. The weights come from KP01's gates."),
        1: T("第一顆隱藏神經元是 OR：s1 等於 x1 加 x2 減 1。", "The first hidden neuron is OR: s1 equals x1 plus x2 minus 1."),
        2: T("第二顆是 NAND：s2 等於負 x1 負 x2 加 1.5。", "The second is NAND: s2 equals minus x1 minus x2 plus 1.5."),
        3: T("輸出層是 AND：兩個條件都成立才輸出 1。這樣合起來就是 XOR。權重是手工湊的，怎麼讓機器學出來是第三章的事。", "The output layer is AND: it outputs 1 only if both conditions hold. Together that is XOR. The weights are hand-built; making a machine learn them is Chapter 3's job.")}))

    # ============ 4: table ============
    g = new(4, T("逐格手算:輸出 0、1、1、0", "Cell by cell: outputs 0, 1, 1, 0"))
    cx = [50, 190, 340, 450, 620, 730, 890, 1070]; cw = [140, 150, 110, 170, 110, 160, 180, 160]
    hd = [r"$(x_1,x_2)$", "$s_1$", "$h_1$", "$s_2$", "$h_2$", r"$s_3=h_1+h_2-2$", "$o$"]
    cx = [50, 210, 350, 500, 640, 790, 1040]; cw = [150, 130, 130, 130, 130, 240, 190]
    g.beat = 1
    for k, h in enumerate(hd): g.text(cx[k], 120, cw[k], 44, h, size=22, color=MUTED, bold=True, align="center", vcenter=True, z=8)
    for i, (p, r) in enumerate(zip(PTS, ROWS)):
        g.beat = 2 + i // 2
        y = 176 + i * 92
        ok = r[5] == 1
        g.rect(40, y, 1200, 78, fill=GRNL if ok else CARD, stroke=GRN if ok else GRAY, sw=3, radius=14, z=3)
        vals = [f"$({p[0]},{p[1]})$", f"${fm(r[0])}$", f"${r[1]}$", f"${fm(r[2])}$", f"${r[3]}$", f"${fm(r[4])}$", f"${r[5]}$"]
        for k, v in enumerate(vals): g.text(cx[k], y, cw[k], 78, v, size=26, color=(GRN if (k == 6 and ok) else INK), bold=True, align="center", vcenter=True, z=8)
    g.beat = 4
    banner(g, T("輸出 $0,1,1,0$ 正是 XOR", "outputs $0,1,1,0$ are exactly XOR"))
    pages.append((g, T("手算表", "The table"), {
        0: T("從左到右逐格算：先算 s1 與 h1，再算 s2 與 h2，最後算 s3 與輸出。", "Compute cell by cell from left to right: s1 and h1, then s2 and h2, finally s3 and the output."),
        1: T("輸入 0 0 與 0 1 的列。0 1 時輸出 1。", "The rows for inputs 0 0 and 0 1. At 0 1 the output is 1."),
        2: T("輸入 1 0 與 1 1 的列。1 0 時輸出 1，1 1 時輸出 0。", "The rows for 1 0 and 1 1. At 1 0 the output is 1, at 1 1 it is 0."),
        3: T("輸出 0、1、1、0，正是 XOR。", "The outputs 0, 1, 1, 0 are exactly XOR.")}))

    # ============ 5: changing coordinates ============
    g = new(5, T("第一層把四個點搬到新座標", "The first layer moves the points to new coordinates"))
    def sq(x0, y0, side, pts_xy, lines=None, labs=None):
        sc = side / 1.9; u = lambda v: x0 + sc * 0.35 + v * sc; vv = lambda v: y0 + side - sc * 0.35 - v * sc
        g.rect(x0, y0, side, side, fill=CARD, stroke=GRAY, sw=3, radius=14, z=1)
        return u, vv
    g.beat = 1
    side = 400; sc = side / 1.9
    u1 = lambda v: 60 + sc * 0.35 + v * sc; v1 = lambda v: 130 + side - sc * 0.35 - v * sc
    g.rect(60, 130, side, side, fill=CARD, stroke=GRAY, sw=3, radius=14, z=1)
    for p, r in zip(PTS, ROWS):
        y = r[5]; g.ellipse(u1(p[0]), v1(p[1]), 20, fill=(GRN if y else CARD), stroke=(GRN if y else BLUE), sw=4, z=6)
    g.text(60, 540, side, 62, T("輸入 $(x_1,x_2)$:對角線上", "input $(x_1,x_2)$: on a diagonal"), size=20, color=INK, bold=True, align="center", z=8)
    g.beat = 2
    u2 = lambda v: 820 + sc * 0.35 + v * sc; v2 = lambda v: 130 + side - sc * 0.35 - v * sc
    g.rect(820, 130, side, side, fill=CARD, stroke=GRAY, sw=3, radius=14, z=1)
    g.line([(u2(0.5), v2(1.0)), (u2(1.0), v2(0.5))], RED, sw=5, dash=True, z=5) if False else None
    labs = {}
    for p, r in zip(PTS, ROWS):
        h = (r[1], r[3]); labs.setdefault(h, []).append(p)
    for h, ps in labs.items():
        y = ROWS[PTS.index(ps[0])][5]
        g.ellipse(u2(h[0]), v2(h[1]), 22, fill=(GRN if y else CARD), stroke=(GRN if y else BLUE), sw=4, z=6)
        g.text(u2(h[0]) - 90 if h[0] == 1 else u2(h[0]) - 20, v2(h[1]) + (-60 if h[1] == 1 and h[0] == 1 else 30), 220, 44, "$" + "=".join(f"({q[0]},{q[1]})" for q in ps) + "$" if len(ps) == 1 else "$(0,1),(1,0)$", size=18, color=MUTED, bold=True, z=8)
    g.line([(u2(0.1), v2(1.4)), (u2(1.4), v2(0.1))], RED, sw=5, dash=True, z=5)
    g.text(820, 540, side, 62, T("新座標 $(h_1,h_2)$:一條線就切開", "new coordinates $(h_1,h_2)$: one line separates"), size=20, color=INK, bold=True, align="center", z=8)
    g.a_arrow = g.line([(500, 330), (780, 330)], GOLD, sw=8, arrow="end", head=26, z=5)
    g.text(480, 280, 320, 40, T("第一層", "first layer"), size=22, color=GOLD, bold=True, align="center", z=8)
    g.beat = 3
    banner(g, T("$(0,1)$ 與 $(1,0)$ 都變成 $(1,1)$;輸出為 1 的只剩一個點，一個 AND 就切開", "$(0,1)$ and $(1,0)$ both become $(1,1)$; only one point has output 1, so one AND cuts it"))
    pages.append((g, T("座標改變", "New coordinates"), {
        0: T("原來的四個點：同一類的兩個點在對角線上，一條線切不開。", "The original four points: the two points of a class lie on a diagonal; no single line cuts them."),
        1: T("第一層把每個點變成新座標 h1、h2：原本在對角線上的 0 1 與 1 0 都變成 1 1，0 0 變成 0 1，1 1 變成 1 0。新座標裡一條線就切開。", "The first layer turns each point into new coordinates h1, h2: 0 1 and 1 0 both become 1 1, 0 0 becomes 0 1, and 1 1 becomes 1 0. In the new coordinates a single line separates them."),
        2: T("輸出為 1 的點只剩一個，所以輸出層一個 AND 就能切開。", "Only one point has output 1, so one AND in the output layer cuts it off."),
        3: T("", "")}))

    # ============ 6: collapse numeric ============
    g = new(6, T("把階梯換成恆等:輸出變成常數", "Replace the step by identity: the output is constant"))
    g.beat = 1
    g.text(50, 112, 1180, 40, T("同一組權重，但活化函數換成 $\\sigma(a)=a$", "same weights, but the activation is $\\sigma(a)=a$"), size=24, color=MUTED, bold=True, align="center", z=8)
    cx = [60, 260, 520, 800, 1010]; cw = [180, 220, 250, 190, 190]
    for k, h in enumerate((r"$(x_1,x_2)$", "$h_1=x_1+x_2-1$", "$h_2=-x_1-x_2+1.5$", r"$y=h_1+h_2-2$", "XOR")):
        g.text(cx[k], 150, cw[k], 90, h, size=18, color=MUTED, bold=True, align="center", vcenter=True, z=8)
    g.beat = 2
    for i, (p, r) in enumerate(zip(PTS, LIN)):
        y = 240 + i * 70
        g.rect(50, y, 1180, 58, fill=REDL, stroke=RED, sw=3, radius=12, z=3)
        vals = [f"$({p[0]},{p[1]})$", f"${fm(r[1])}$", f"${fm(r[3])}$", f"${fm(r[4])}$", f"${[0, 1, 1, 0][i]}$"]
        for k, v in enumerate(vals): g.text(cx[k], y, cw[k], 58, v, size=26, color=(RED if k == 3 else INK), bold=True, align="center", vcenter=True, z=8)
    g.beat = 3
    g.rect(150, 524, 980, 104, fill="#fef3c7", stroke=GOLD, sw=3, radius=16, z=3)
    g.text(170, 524, 940, 104, r"$\mathbf{W}_2\mathbf{W}_1=(1,1)\begin{pmatrix}1&1\\-1&-1\end{pmatrix}=(0,0)$", size=30, color=INK, align="center", vcenter=True, z=8)
    banner(g, T("輸出恆為 $-1.5$:連 XOR 的一半都做不到", "the output is always $-1.5$: not even half of XOR"))
    pages.append((g, T("全線性會坍縮", "All-linear collapses"), {
        0: T("取同一組權重，但把階梯換成恆等活化，也就是什麼都不做。", "Take the same weights but replace the step by the identity activation, which does nothing."),
        1: T("不管輸入是什麼，輸出都是負 1.5。", "Whatever the input, the output is minus 1.5."),
        2: T("原因是兩層合起來的矩陣 W2 乘 W1 等於 0、0：整個網路被壓成一個常數。", "The reason: the two layers combine into the matrix W2 times W1, which is 0, 0: the whole network is squashed into a constant."),
        3: T("", "")}))

    # ============ 7: general collapse ============
    g = new(7, T("全線性:疊多少層都是一層", "All linear: any number of layers is one layer"))
    g.beat = 1
    ids = []
    for i, nm in enumerate((r"$\mathbf{V}_1^\top$", r"$\mathbf{V}_2^\top$", r"$\mathbf{V}_3^\top$", r"$\cdots$")):
        x = 60 + i * 240
        ids.append(g.mark()); g.rect(x, 140, 200, 110, fill=BLUEL, stroke=BLUE, sw=3, radius=14, z=3); g.text(x, 140, 200, 110, nm, size=36, color=INK, align="center", vcenter=True, z=8)
        if i: g.link(ids[i - 1], ids[i], 4, 12, GRAY, 4)
    g.text(60, 262, 900, 40, T("每一層都是一個矩陣 $\\mathbf{V}_i^\\top$(線性活化也算進去)", "each layer is a matrix $\\mathbf{V}_i^\\top$ (a linear activation is absorbed)"), size=20, color=MUTED, bold=True, z=8)
    g.beat = 2
    g.rect(190, 340, 900, 130, fill="#fef3c7", stroke=GOLD, sw=4, radius=16, z=3)
    g.text(200, 340, 880, 130, r"$\mathbf{y}=\mathbf{V}_3^\top\mathbf{V}_2^\top\mathbf{V}_1^\top\mathbf{x}=\mathbf{V}^\top\mathbf{x}$", size=44, color=INK, align="center", vcenter=True, z=8)
    g.beat = 3
    g.rect(190, 510, 900, 100, fill=REDL, stroke=RED, sw=3, radius=16, z=3)
    g.text(210, 510, 860, 100, T("疊 100 層也只是一次線性投影\n加上偏置也一樣:仿射函數連乘仍是仿射，邊界仍是直線", "100 layers are still one linear projection\nwith biases too: affine maps compose to an affine map, the boundary stays a line"), size=22, color=RED, bold=True, align="center", z=8)
    banner(g, T("力量不來自層數，來自層與層之間的非線性", "the power is not the number of layers but the nonlinearity between them"))
    pages.append((g, T("為什麼一定要非線性", "Why nonlinearity"), {
        0: T("若所有活化都是線性的，每一層仍是一個矩陣。", "If every activation is linear, each layer is still a matrix."),
        1: T("連乘之後還是一個矩陣，三層網路等於一層。", "Their product is still one matrix, so a three-layer network equals one layer."),
        2: T("疊一百層也只是一次線性投影；加上偏置也一樣，因為仿射函數連乘仍是仿射函數，邊界仍是直線。", "A hundred layers are still one linear projection; with biases too, since affine maps compose to an affine map and the boundary stays a straight line."),
        3: T("所以力量來自層與層之間的非線性。", "So the power comes from the nonlinearity between layers.")}))

    # ============ 8: myths + question ============
    g = new(8, T("兩個迷思，與一個問題", "Two myths and a question"))
    g.beat = 1
    g.rect(50, 130, 570, 220, fill=REDL, stroke=RED, sw=3, radius=16, z=3)
    g.text(60, 138, 550, 56, T("迷思", "myth"), size=24, color=RED, bold=True, align="center", z=8)
    g.text(60, 194, 550, 150, T("「層數越多，網路一定越強」\n全線性時 100 層與 1 層等價", "“more layers always means stronger”\nall-linear: 100 layers equal 1 layer"), size=24, color=INK, bold=True, align="center", z=8)
    g.beat = 2
    g.rect(660, 130, 570, 220, fill=ORGL, stroke=ORG, sw=3, radius=16, z=3)
    g.text(670, 138, 550, 56, T("迷思", "myth"), size=24, color=ORG, bold=True, align="center", z=8)
    g.text(670, 194, 550, 150, T("「有非線性就一定學得會」\n上面的權重是手工湊的；怎麼學是第 3 章的事", "“nonlinearity means it can always be learned”\nthe weights here were hand-built; learning them is Chapter 3"), size=24, color=INK, bold=True, align="center", z=8)
    g.beat = 3
    g.rect(50, 390, 1000, 210, fill="#fef3c7", stroke=GOLD, sw=3, radius=16, z=3)
    g.text(70, 400, 960, 190, T("階梯函數可以當非線性，但它在 0 處不可導、其他地方導數為 0。要讓機器用梯度學習，該選哪一種非線性？", "The step works as a nonlinearity, but it is not differentiable at 0 and has zero derivative elsewhere. To learn by gradients, which nonlinearity should we pick?"), size=24, color=INK, bold=True, align="center", z=8)
    mascot(g, 1100, 400, 0.55, "confused")
    banner(g, T("下一個單元:活化函數", "next unit: activation functions"))
    pages.append((g, T("迷思與問題", "Myths and question"), {
        0: T("第一個迷思：層數越多一定越強。全線性時一百層與一層完全等價。", "First myth: more layers always means stronger. When all-linear, 100 layers equal one."),
        1: T("第二個迷思：有非線性就一定學得會。上面的權重是手工湊的，怎麼讓機器學出來是第三章的事。", "Second myth: nonlinearity means it can always be learned. The weights above were hand-built; learning them is Chapter 3."),
        2: T("留給下一個單元：階梯函數在 0 不可導，其他地方導數為 0，要用梯度學習，該選哪一種非線性？", "For the next unit: the step is not differentiable at 0 and has zero derivative elsewhere; to learn by gradients, which nonlinearity?"),
        3: T("", "")}))
    return pages
