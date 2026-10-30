import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lec01_kit import *

KP = "KP02"; N = 8
DATA = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 1)]
def sim(eta=1):
    w = [0, 0, 0]; log = []; ends = []; rnd = 0
    while True:
        rnd += 1; changed = False
        for (x1, x2), y in DATA:
            s = w[0] * x1 + w[1] * x2 + w[2]; o = 1 if s >= 0 else 0
            if o != y:
                nw = [w[0] + eta * (y - o) * x1, w[1] + eta * (y - o) * x2, w[2] + eta * (y - o)]
                log.append((rnd, (x1, x2), y, tuple(w), s, o, tuple(nw))); w = nw; changed = True
        ends.append(tuple(w))
        if not changed: break
    return log, ends, rnd
LOG, ENDS, ROUNDS = sim()
assert len(LOG) == 5 and ROUNDS == 4 and ENDS == [(0, 1, 0), (1, 1, 0), (1, 1, -1), (1, 1, -1)]
assert LOG[1][3:7] == ((0, 0, -1), -1, 0, (0, 1, 0))
x_, s0, e_ = (0, 1, 1), -1, 1; assert s0 + 1 * e_ * sum(v * v for v in x_) == 1                       # s' = s + eta*(y-o)*|x|^2
assert sim(2)[1][-1] == (2, 2, -2)                                                             # eta=2 doubles the weights
wrong = lambda w: [p for p, y in DATA if (1 if w[0] * p[0] + w[1] * p[1] + w[2] >= 0 else 0) != y]
assert wrong((0, 1, 0)) == [(0, 0)] and wrong((1, 1, 0)) == [(0, 0)] and wrong((1, 1, -1)) == []

def panel(g, x0, y0, side, w, label, T, extra_points=None, line=True):
    """unit-square panel; returns nothing. points coloured by label y."""
    sc = side / 1.9; u = lambda v: x0 + sc * 0.35 + v * sc; vv = lambda v: y0 + side - sc * 0.35 - v * sc
    g.rect(x0, y0, side, side, fill=CARD, stroke=GRAY, sw=3, radius=14, z=1)
    if line and w is not None:
        a, b_, c = w; pts = []
        lo, hi = -0.35, 1.55
        if abs(b_) > 1e-9:
            for xx in (lo, hi):
                yy = -(a * xx + c) / b_
                if lo <= yy <= hi: pts.append((xx, yy))
        if abs(a) > 1e-9:
            for yy in (lo, hi):
                xx = -(b_ * yy + c) / a
                if lo <= xx <= hi and all(abs(xx - p[0]) + abs(yy - p[1]) > 1e-6 for p in pts): pts.append((xx, yy))
        if len(pts) >= 2: g.line([(u(pts[0][0]), vv(pts[0][1])), (u(pts[1][0]), vv(pts[1][1]))], RED, sw=5, dash=True, z=5)
    bad = wrong(w) if w is not None else []
    for p, y in DATA:
        g.ellipse(u(p[0]), vv(p[1]), 20, fill=(GRN if y else CARD), stroke=(GRN if y else BLUE), sw=4, z=6)
        if p in bad: g.ellipse(u(p[0]), vv(p[1]), 32, stroke=RED, sw=4, z=7)
    g.text(x0, y0 + side + 6, side, 36, label, size=20, color=INK, bold=True, align="center", z=8)

def make(lang):
    zh = lang == "zh"
    T = lambda a, b_: a if zh else b_
    pages = []
    new = lambda n, title: new_page(n, N, KP, title)

    # ============ 1: feedback loop ============
    g = new(1, T("KP02 猜錯才改權重，Perceptron 自己學", "KP02 Change weights only when wrong"))
    g.beat = 1
    ids = []
    spec = [(BLUE, BLUEL, T("看一筆資料", "take one sample"), r"$\mathbf{x}_i,\ y_i$"), (ORG, ORGL, T("算和與輸出", "sum and output"), r"$s,\ o_i=f(s)$"), (PUR, PURL, T("和標籤比", "compare with label"), r"$y_i-o_i$")]
    for i, (col, light, a, b_) in enumerate(spec):
        x = 50 + i * 290
        ids.append(g.mark()); g.rect(x, 170, 240, 130, fill=light, stroke=col, sw=3, radius=16, z=3)
        g.text(x, 180, 240, 50, a, size=22, color=col, bold=True, align="center", z=8); g.text(x, 238, 240, 50, b_, size=26, color=INK, align="center", vcenter=True, z=8)
        if i: g.link(ids[i - 1], ids[i], 4, 12, GRAY, 5)
    g.beat = 2
    dia = g.mark(); g.poly([(1110, 160), (1230, 235), (1110, 310), (990, 235)], fill=GOLD, stroke=GOLD, sw=3, z=3, fillOpacity=0.25)
    g.text(1010, 205, 200, 60, T("猜對了嗎?", "right?"), size=24, color=INK, bold=True, align="center", vcenter=True, z=8)
    g.link(ids[2], dia, 4, 12, GRAY, 5)
    g.beat = 3
    e = g.mark(); g.rect(880, 420, 340, 110, fill=GRNL, stroke=GRN, sw=3, radius=16, z=3)
    g.text(880, 430, 340, 40, T("猜對:什麼都不做", "right: do nothing"), size=24, color=GRN, bold=True, align="center", z=8)
    f = g.mark(); g.rect(440, 420, 400, 110, fill=REDL, stroke=RED, sw=3, radius=16, z=3)
    g.text(440, 426, 400, 56, T("猜錯:把權重推一步", "wrong: push the weights one step"), size=24, color=RED, bold=True, align="center", z=8)
    g.text(440, 482, 400, 44, r"$\mathbf{w}\leftarrow\mathbf{w}+\eta(y_i-o_i)\mathbf{x}_i$", size=24, color=INK, align="center", vcenter=True, z=8)
    g.link(dia, e, 8, 0, GRN, 4); g.link(dia, f, 8, 0, RED, 4); g.link(f, ids[0], 12, 8, RED, 4)
    g.text(40, 470, 340, 40, T("接著看下一筆", "then the next sample"), size=20, color=MUTED, bold=True, align="center", z=8) if False else g.text(40, 540, 340, 40, T("接著看下一筆", "then the next sample"), size=20, color=MUTED, bold=True, align="center", z=8)
    banner(g, T("監督式規則:由標籤與錯誤驅動", "a supervised rule: driven by labels and errors"))
    pages.append((g, T("回饋迴圈", "The feedback loop"), {
        0: T("KP01 的神經元，每個權重都要人手算。Rosenblatt 在 1958 年加了一個回饋。", "Every weight of the KP01 neuron had to be worked out by hand. In 1958 Rosenblatt added feedback."),
        1: T("每看一筆資料，先算加權和 s 與輸出，再和正確標籤比。", "For each sample, compute the sum s and the output, then compare with the correct label."),
        2: T("問：猜對了嗎？", "Ask: was it right?"),
        3: T("猜對，什麼都不做；猜錯，才把權重往正確方向推一步，然後看下一筆。這是監督式規則。", "If right, do nothing; only if wrong push the weights one step the right way, then take the next sample. This is a supervised rule.")}))

    # ============ 2: three errors ============
    g = new(2, T("三種誤差，三種動作", "Three errors, three actions"))
    g.beat = 1
    card(g, 190, 112, 900, 90, GRAY); g.text(200, 112, 880, 90, r"$\mathbf{w}\leftarrow\mathbf{w}+\eta\,(y_i-o_i)\,\mathbf{x}_i$", size=40, color=INK, align="center", vcenter=True, z=8)
    rows = [(GRN, GRNL, "$0$", T("猜對", "right"), T("不動", "no change"), T("不變", "unchanged")),
            (BLUE, BLUEL, "$+1$", T("該放電卻沒放", "should fire, did not"), r"$\mathbf{w}+\eta\mathbf{x}_i$", T("上升", "rises")),
            (RED, REDL, "$-1$", T("不該放電卻放了", "fired, should not"), r"$\mathbf{w}-\eta\mathbf{x}_i$", T("下降", "falls"))]
    cx = [60, 260, 600, 940]; cw = [180, 320, 320, 280]
    for k, h in enumerate((r"$y_i-o_i$", T("情形", "case"), T("對 $\\mathbf{w}$ 的動作", "action on $\\mathbf{w}$"), T("加權和 $s$", "sum $s$"))):
        g.text(cx[k], 226, cw[k], 40, h, size=22, color=MUTED, bold=True, align="center", vcenter=True, z=8)
    for i, (col, light, e_, cs, ac, sd) in enumerate(rows):
        g.beat = 2 + i // 2 if False else (2 if i < 2 else 3)
        y = 276 + i * 106
        g.rect(50, y, 1180, 90, fill=light, stroke=col, sw=3, radius=14, z=3)
        for k, t in enumerate((e_, cs, ac, sd)):
            g.text(cx[k], y, cw[k], 90, t, size=28 if k in (0, 2) else 24, color=col if k == 0 else INK, bold=True, align="center", vcenter=True, z=8)
    banner(g, T("輸入為 0 的那一路，權重完全不動", "an input that is 0 never changes its weight"))
    pages.append((g, T("更新規則", "The update rule"), {
        0: T("更新規則只有一條。誤差 y 減 o 只可能是 0、正 1、負 1。", "The update rule is a single line. The error y minus o can only be 0, plus 1 or minus 1."),
        1: T("誤差 0：猜對，不動。誤差正 1：該放電卻沒放，權重加上 eta 乘 x，s 上升。", "Error 0: right, no change. Error plus 1: should have fired but did not, add eta times x, s rises."),
        2: T("誤差負 1：不該放電卻放了，權重減掉 eta 乘 x，s 下降。輸入是 0 的那一路，更新量是 0，權重不動。", "Error minus 1: fired but should not have, subtract eta times x, s falls. An input that is 0 adds 0, so its weight stays."),
        3: T("可以把它想成調旋鈕：太小聲就轉大，太大聲就轉小，對了就不碰。", "Think of turning knobs: too quiet, turn up; too loud, turn down; right, leave it.")}))

    # ============ 3: direction always right ============
    g = new(3, T("每次更新，方向一定對", "Every update moves in the right direction"))
    g.beat = 1
    card(g, 90, 112, 1100, 100, BLUE); g.text(100, 112, 1080, 100, r"$s'=s+\eta\,(y_i-o_i)\,\|\mathbf{x}_i\|^2,\qquad \|\mathbf{x}_i\|^2\ge1$", size=36, color=INK, align="center", vcenter=True, z=8)
    g.text(90, 222, 1100, 40, T("因為 $x_{i0}=1$，所以 $\\|\\mathbf{x}_i\\|^2\\ge1$:誤差 $+1$ 時 $s$ 一定上升，$-1$ 時一定下降", "since $x_{i0}=1$, $\\|\\mathbf{x}_i\\|^2\\ge1$: $s$ always rises for error $+1$ and falls for $-1$"), size=20, color=MUTED, bold=True, align="center", z=8)
    g.beat = 2
    ex = [(BLUE, BLUEL, T("更新前", "before"), r"$\mathbf{w}=(0,0,-1)$", r"$s=-1,\ o=0,\ y=1$"), (ORG, ORGL, T("更新", "update"), r"$\mathbf{x}=(0,1,1)$", r"$\mathbf{w}+1\cdot(+1)\cdot\mathbf{x}=(0,1,0)$"), (GRN, GRNL, T("更新後", "after"), r"$\mathbf{w}'=(0,1,0)$", r"$s'=-1+1\cdot2=1$")]
    ii = []
    for i, (col, light, nm, a, b_) in enumerate(ex):
        x = 50 + i * 410
        ii.append(g.mark()); g.rect(x, 300, 360, 230, fill=light, stroke=col, sw=3, radius=16, z=3)
        g.text(x, 310, 360, 40, nm, size=24, color=col, bold=True, align="center", z=8); g.text(x + 8, 360, 344, 60, a, size=26, color=INK, align="center", vcenter=True, z=8); g.text(x + 8, 428, 344, 90, b_, size=22 if i != 1 else 20, color=INK, bold=True, align="center", z=8)
        if i: g.link(ii[i - 1], ii[i], 4, 12, GRAY, 5)
    g.beat = 3
    banner(g, T("例:猜錯 $(0,1)$，$s$ 從 $-1$ 升到 $1$，剛好增加 $\\|\\mathbf{x}\\|^2=2$", "e.g. wrong on $(0,1)$: $s$ goes from $-1$ to $1$, up by exactly $\\|\\mathbf{x}\\|^2=2$"))
    pages.append((g, T("方向一定對", "Always the right direction"), {
        0: T("更新之後，這一筆的加權和變成 s 加上 eta 乘誤差乘 x 的長度平方。x 補上 x0 等於 1，長度平方至少是 1。", "After the update, this sample's sum becomes s plus eta times the error times the squared length of x. With x0 equal to 1, the squared length is at least 1."),
        1: T("看一個實例：權重 0、0、負 1，遇到資料 0、1，s 等於負 1，輸出 0，但標籤是 1。", "A concrete case: weights 0, 0, minus 1 meet the sample 0, 1; s is minus 1, output 0, but the label is 1."),
        2: T("補上偏置項，x 是 0、1、1，更新後權重 0、1、0，新的 s 等於 1。增加量剛好是 x 的長度平方 2。", "Adding the bias input, x is 0, 1, 1; the new weights are 0, 1, 0 and the new s is 1. The increase is exactly the squared length of x, which is 2."),
        3: T("一次更新只保證這一筆的 s 往對的方向走。", "One update only guarantees that this sample's s moves the right way.")}))

    # ============ 4: trace table ============
    g = new(4, T("手算:從全 0 學 OR", "By hand: learning OR from all zeros"))
    g.beat = 1
    g.text(50, 108, 1180, 36, T("資料依序 $(0,0)\\to0$、$(0,1)\\to1$、$(1,0)\\to1$、$(1,1)\\to1$;從 $(0,0,0)$ 開始，$\\eta=1$", "data in order $(0,0)\\to0$, $(0,1)\\to1$, $(1,0)\\to1$, $(1,1)\\to1$; start at $(0,0,0)$, $\\eta=1$"), size=20, color=MUTED, bold=True, align="center", z=8)
    cx = [50, 170, 340, 460, 740, 860, 960]; cw = [110, 160, 110, 270, 110, 90, 270]
    hd = [T("輪", "pass"), T("猜錯的點", "wrong point"), "$y$", r"$(w_1,w_2,w_0)$", "$s$", "$o$", T("更新後", "after")]
    for k, h in enumerate(hd):
        g.text(cx[k], 150, cw[k], 40, h, size=21, color=MUTED, bold=True, align="center", vcenter=True, z=8)
    fmt = lambda w: "$(" + ",".join(str(v).replace("-", "-") for v in w) + ")$"
    for i, (r_, p, y, w, s_, o_, nw) in enumerate(LOG):
        g.beat = 2 if i < 2 else 3
        yy = 196 + i * 74
        g.rect(50, yy, 1180, 64, fill=(REDL if False else CARD), stroke=GRAY, sw=3, radius=12, z=3)
        for k, t in enumerate((str(r_), f"$({p[0]},{p[1]})$", str(y), fmt(w), str(s_).replace("-", "$-$") if s_ < 0 else str(s_), str(o_), fmt(nw))):
            g.text(cx[k], yy, cw[k], 64, t, size=24, color=(GRN if k == 6 else INK), bold=True, align="center", vcenter=True, z=8)
    g.beat = 4
    banner(g, T("共 5 次更新，第 4 輪四筆全對 → 停止", "5 updates in all; pass 4 is all right → stop"))
    pages.append((g, T("逐筆手算", "Step-by-step trace"), {
        0: T("資料依序是 0 0、0 1、1 0、1 1，從全 0 開始，學習率 1。", "The data come in the order 0 0, 0 1, 1 0, 1 1; start from all zeros with learning rate 1."),
        1: T("第 1 輪：點 0 0 猜錯，s 等於 0 輸出 1，但標籤 0，偏置減 1；接著點 0 1 猜錯，權重 w2 加 1，偏置加 1。", "Pass 1: the point 0 0 is wrong, s is 0 and output 1 but the label is 0, so the bias drops by 1; then 0 1 is wrong, w2 gains 1 and the bias gains 1."),
        2: T("第 2 輪：0 0 又猜錯，因為被 0 1 的更新打壞；1 0 猜錯。第 3 輪：0 0 再猜錯一次。", "Pass 2: 0 0 is wrong again, broken by the update for 0 1; 1 0 is wrong. Pass 3: 0 0 is wrong once more."),
        3: T("共五次更新，最後權重 1、1、負 1。第 4 輪四筆全對，整輪沒有更新，停止。", "Five updates in all, ending at weights 1, 1, minus 1. Pass 4 is all right with no updates, so it stops.")}))

    # ============ 5: boundary moving ============
    g = new(5, T("決策線被一次次推向猜錯的點", "The line is pushed toward the wrong points"))
    for i in range(3):
        g.beat = i + 1
        x0 = 40 + i * 415
        panel(g, x0, 150, 390, ENDS[i], T(f"第 {i + 1} 輪結束 $\\mathbf{{w}}=" + "(" + ",".join(map(str, ENDS[i])) + ")$", f"end of pass {i + 1}: $\\mathbf{{w}}=" + "(" + ",".join(map(str, ENDS[i])) + ")$"), T)
        g.text(x0, 588, 390, 40, (T("紅圈:還猜錯", "red ring: still wrong") if wrong(ENDS[i]) else T("四個點全對", "all four points right")), size=20, color=(RED if wrong(ENDS[i]) else GRN), bold=True, align="center", z=8)
    banner(g, T("實心:標籤 1 空心:標籤 0 虛線:$s=0$", "solid: label 1 · open: label 0 · dashed: $s=0$"))
    pages.append((g, T("決策線的移動", "The moving line"), {
        0: T("第 1 輪結束，線是水平的，點 0 0 還被劃進放電區，紅圈標出猜錯。", "At the end of pass 1 the line is horizontal and the point 0 0 is still in the firing region; the red ring marks the error."),
        1: T("第 2 輪結束，線轉成斜線，但仍穿過 0 0。", "At the end of pass 2 the line has turned diagonal but still passes through 0 0."),
        2: T("第 3 輪結束，線移到 x1 加 x2 等於 1，四個點全對。每次更新都把線推向猜錯的點。", "At the end of pass 3 the line sits at x1 plus x2 equals 1 and all four points are right. Each update pushes the line toward a wrong point."),
        3: T("", "")}))

    # ============ 6: stop, equals KP01 OR ============
    g = new(6, T("停下來的權重，就是 KP01 手算的 OR", "The final weights are the OR gate from KP01"))
    g.beat = 1
    a = g.mark(); g.rect(60, 150, 520, 200, fill=BLUEL, stroke=BLUE, sw=3, radius=16, z=3)
    g.text(60, 160, 520, 40, T("KP01:人手算", "KP01: by hand"), size=24, color=BLUE, bold=True, align="center", z=8)
    g.text(70, 214, 500, 60, r"$(w_1,w_2,w_0)=(1,1,-1)$", size=30, color=INK, align="center", vcenter=True, z=8); g.text(70, 284, 500, 50, r"$\theta=1$", size=26, color=INK, bold=True, align="center", z=8)
    g.beat = 2
    b2 = g.mark(); g.rect(700, 150, 520, 200, fill=GRNL, stroke=GRN, sw=3, radius=16, z=3)
    g.text(700, 160, 520, 40, T("KP02:機器自己找到", "KP02: found by the machine"), size=24, color=GRN, bold=True, align="center", z=8)
    g.text(710, 214, 500, 60, r"$(w_1,w_2,w_0)=(1,1,-1)$", size=30, color=INK, align="center", vcenter=True, z=8); g.text(710, 284, 500, 50, T("5 次更新，第 4 輪全對", "5 updates, pass 4 all right"), size=24, color=INK, bold=True, align="center", z=8)
    g.link(a, b2, 4, 12, GOLD, 6)
    g.text(590, 200, 100, 56, "=", size=44, color=GOLD, bold=True, align="center", z=8)
    g.beat = 3
    g.rect(190, 410, 900, 150, fill="#fef3c7", stroke=GOLD, sw=3, radius=16, z=3)
    g.text(210, 420, 860, 56, T("差別只有一個:這次權重不是人算的", "The only difference: the weights were not worked out by a person"), size=26, color=INK, bold=True, align="center", z=8)
    g.text(210, 484, 860, 60, T("只給資料與正確答案，它自己改出來", "given only data and right answers, it corrected itself"), size=22, color=MUTED, bold=True, align="center", z=8)
    mascot(g, 1100, 390, 0.5, "proud")
    banner(g, T("KP01 的問題有答案了:讓權重自己學", "KP01's question is answered: let the weights learn themselves"))
    pages.append((g, T("與 KP01 對照", "Back to KP01"), {
        0: T("KP01 手算出來的 OR，權重是 1、1、負 1，閾值 1。", "The OR gate worked out in KP01 has weights 1, 1, minus 1 and threshold 1."),
        1: T("這一頁學出來的權重，也是 1、1、負 1。", "The weights learned here are also 1, 1, minus 1."),
        2: T("完全一樣。差別是這次不是人算的，而是只給資料與答案，機器自己找到。", "Exactly the same. The difference is that this time a person did not work them out; the machine found them from data and answers alone."),
        3: T("", "")}))

    # ============ 7: when it stops ============
    g = new(7, T("什麼時候停，什麼時候永遠不停", "When it stops, and when it never does"))
    g.beat = 1
    panel(g, 60, 130, 360, (1, 1, -1), T("線性可分(OR)", "linearly separable (OR)"), T)
    g.rect(450, 190, 300, 250, fill=GRNL, stroke=GRN, sw=3, radius=16, z=3)
    g.text(460, 200, 280, 230, T("有限次更新後一定停下\n(感知器收斂定理，本講義不證明)", "stops after finitely many updates\n(perceptron convergence theorem, not proved here)"), size=22, color=GRN, bold=True, align="center", z=8)
    g.beat = 2
    xor = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]
    x0, y0, side = 800, 130, 360; sc = side / 1.9; u = lambda v: x0 + sc * 0.35 + v * sc; vv = lambda v: y0 + side - sc * 0.35 - v * sc
    g.rect(x0, y0, side, side, fill=CARD, stroke=GRAY, sw=3, radius=14, z=1)
    for p, y in xor:
        g.ellipse(u(p[0]), vv(p[1]), 20, fill=(GRN if y else CARD), stroke=(GRN if y else BLUE), sw=4, z=6)
    g.text(x0, y0 + side + 6, side, 36, T("線性不可分(XOR)", "not separable (XOR)"), size=20, color=INK, bold=True, align="center", z=8)
    g.rect(800, 540, 360, 56, fill=REDL, stroke=RED, sw=3, radius=14, z=3); g.text(800, 540, 360, 56, T("永遠停不下來(KP04)", "never stops (KP04)"), size=22, color=RED, bold=True, align="center", vcenter=True, z=8)
    g.beat = 3
    g.rect(60, 540, 690, 56, fill=BLUEL, stroke=BLUE, sw=3, radius=14, z=3)
    g.text(60, 540, 690, 56, T("從 0 開始時，$\\eta=2$ 只是把權重放大 2 倍:預測不變", "starting from 0, $\\eta=2$ just doubles the weights: same predictions"), size=20, color=BLUE, bold=True, align="center", vcenter=True, z=8)
    banner(g, T("整輪沒有更新，就停止", "stop when a whole pass makes no update"))
    pages.append((g, T("停止條件", "When it stops"), {
        0: T("資料線性可分時，Perceptron 一定在有限次更新後停下。這是感知器收斂定理，本講義不證明。", "On linearly separable data the Perceptron always stops after finitely many updates. This is the perceptron convergence theorem, not proved here."),
        1: T("資料線性不可分，就永遠停不下來。KP04 的 XOR 就是這種情形。", "On non-separable data it never stops. The XOR of KP04 is such a case."),
        2: T("實作是雙層迴圈，整輪沒有更新就停止。另外，從 0 開始時學習率 2 學出來的權重剛好是學習率 1 的兩倍，預測完全相同。", "In code it is a double loop and stops when a whole pass has no update. Also, starting from 0, learning rate 2 gives exactly twice the weights of rate 1, with identical predictions."),
        3: T("", "")}))

    # ============ 8: summary ============
    g = new(8, T("猜錯才改，但不保證一次就修好", "Fix only when wrong, but not fixed in one go"))
    g.beat = 1
    card(g, 50, 120, 1180, 90, BLUE); g.text(60, 120, 1160, 90, r"$\mathbf{w}\leftarrow\mathbf{w}+\eta\,(y_i-o_i)\,\mathbf{x}_i$", size=36, color=INK, align="center", vcenter=True, z=8)
    g.beat = 2
    g.rect(50, 240, 1180, 150, fill=REDL, stroke=RED, sw=3, radius=16, z=3)
    g.text(70, 248, 1140, 40, T("常見迷思:「每看一筆就把它改對，而且不會再錯」", "Myth: “each sample is fixed once and never wrong again”"), size=24, color=RED, bold=True, align="center", z=8)
    g.text(70, 296, 1140, 90, T("不一定:第 1 輪 $(0,0)$ 被修好，卻被 $(0,1)$ 的更新打壞，第 2 輪再錯。一次更新只保證這一筆的 $s$ 往對的方向走。", "Not so: in pass 1 the point $(0,0)$ was fixed, then broken by the update for $(0,1)$, and wrong again in pass 2. One update only moves this sample's $s$ the right way."), size=22, color=INK, bold=True, align="center", z=8)
    g.beat = 3
    g.rect(50, 420, 1000, 190, fill="#fef3c7", stroke=GOLD, sw=3, radius=16, z=3)
    g.text(70, 430, 960, 170, T("標籤用 0 和 1，誤差有三種情形。如果改用 $+1$ 和 $-1$，更新規則能不能寫得更短，而且看得出幾何意義？", "With labels 0 and 1 there are three error cases. If we use $+1$ and $-1$, can the rule be shorter and show its geometry?"), size=26, color=INK, bold=True, align="center", z=8)
    mascot(g, 1110, 420, 0.5, "confused")
    banner(g, T("下一個單元:標籤改成 ±1，規則縮成一行", "next unit: ±1 labels shrink the rule to one line"))
    pages.append((g, T("重點與迷思", "Summary and myth"), {
        0: T("本單元的重點：猜對就不動，猜錯就把權重往正確方向推一步。", "The point of this unit: do nothing when right, push the weights one step the right way when wrong."),
        1: T("常見迷思：以為每一筆改對之後就不會再錯。其實第 1 輪被修好的點，會被後面的更新打壞。", "A common myth: that a sample, once fixed, is never wrong again. In fact the point fixed in pass 1 is broken by a later update."),
        2: T("留給下一個單元的問題：標籤改成正負 1，更新規則能不能寫得更短？", "A question for the next unit: with labels plus and minus 1, can the update rule be shorter?"),
        3: T("", "")}))
    return pages
