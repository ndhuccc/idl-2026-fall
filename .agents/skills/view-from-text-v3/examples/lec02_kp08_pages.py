import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lec01_kit import *

KP = "KP08"; N = 8
OR = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 1)]
def lms(iters, eta=0.1):
    w = [0.0, 0.0, 0.0]; es = []; ws = []
    for _ in range(iters):
        yh = [w[0] * a + w[1] * b + w[2] for (a, b), _y in OR]; r = [y - h for (_, y), h in zip(OR, yh)]
        es.append(0.5 * sum(v * v for v in r))
        g = [-sum(rr * p[0] for rr, (p, _) in zip(r, OR)), -sum(rr * p[1] for rr, (p, _) in zip(r, OR)), -sum(r)]
        w = [w[i] - eta * g[i] for i in range(3)]; ws.append(tuple(w))
    return es, ws, w
ES, WS, WF = lms(2000)
assert [round(e, 3) for e in ES[:4]] == [1.5, 0.34, 0.183, 0.159] and [round(v, 3) for v in WS[0]] == [0.2, 0.2, 0.3] and [round(v, 3) for v in WF] == [0.5, 0.5, 0.25] and round(ES[-1], 3) == 0.125
OUT = [WF[0] * a + WF[1] * b + WF[2] for (a, b), _ in OR]; assert [round(v, 2) for v in OUT] == [0.25, 0.75, 0.75, 1.25]
XS = (0, 1, 2, 3); YS = (0, 1, 0, 1)
def newton(steps):
    b = [0.0, 0.0]; hist = []
    for _ in range(steps):
        ps = [1 / (1 + math.exp(-(b[0] + b[1] * x))) for x in XS]
        ll = sum(y * (b[0] + b[1] * x) - math.log(1 + math.exp(b[0] + b[1] * x)) for x, y in zip(XS, YS)); hist.append((tuple(b), ll))
        g = [sum(y - p for y, p in zip(YS, ps)), sum((y - p) * x for x, y, p in zip(XS, YS, ps))]
        a = sum(p * (1 - p) for p in ps); bb = sum(p * (1 - p) * x for p, x in zip(ps, XS)); c = sum(p * (1 - p) * x * x for p, x in zip(ps, XS)); det = a * c - bb * bb
        b = [b[0] + (c * g[0] - bb * g[1]) / det, b[1] + (-bb * g[0] + a * g[1]) / det]
    return b, hist
BF, HIST = newton(8)
assert [round(v, 3) for v in HIST[1][0]] == [-1.2, 0.8] and round(HIST[0][1], 4) == -2.7726 and round(HIST[1][1], 4) == -2.3526 and [round(v, 3) for v in BF] == [-1.362, 0.908]
PF = [1 / (1 + math.exp(-(BF[0] + BF[1] * x))) for x in XS]; assert [round(p, 3) for p in PF] == [0.204, 0.388, 0.612, 0.796] and round(-BF[0] / BF[1], 2) == 1.5

def make(lang):
    zh = lang == "zh"
    T = lambda a, b_: a if zh else b_
    pages = []
    new = lambda n, title: new_page(n, N, KP, title)

    # ============ 1: compare before the step ============
    g = new(1, T("KP08 拿連續的差去學，不只問對錯", "KP08 Learn from a continuous gap, not right or wrong"))
    for r_, (col, light, nm, desc, b_) in enumerate(((RED, REDL, "Perceptron", T("比階梯後的 0 / 1", "compares the 0/1 after the step"), 150), (GRN, GRNL, "ADALINE", T("比階梯前的連續和", "compares the continuous sum before the step"), 340))):
        g.beat = r_ + 1
        y = b_
        a = g.mark(); g.rect(60, y, 220, 120, fill=BLUEL, stroke=BLUE, sw=3, radius=14, z=3); g.text(60, y, 220, 120, r"$\mathbf{x}$", size=34, color=BLUE, bold=True, align="center", vcenter=True, z=8)
        b2 = g.mark(); g.rect(380, y, 240, 120, fill=ORGL, stroke=ORG, sw=3, radius=14, z=3); g.text(380, y, 240, 120, r"$\hat y=\sum w_jx_j+w_0$", size=24, color=INK, align="center", vcenter=True, z=8)
        c = g.mark(); g.rect(720, y, 200, 120, fill=PURL, stroke=PUR, sw=3, radius=14, z=3); g.text(720, y, 200, 120, T("階梯", "step"), size=26, color=PUR, bold=True, align="center", vcenter=True, z=8)
        d = g.mark(); g.rect(1020, y, 200, 120, fill=light, stroke=col, sw=4, radius=14, z=3); g.text(1020, y + 6, 200, 40, T("和標籤比", "vs label"), size=20, color=col, bold=True, align="center", z=8); g.text(1020, y + 50, 200, 60, "$o\\ \\mathrm{vs}\\ y$" if r_ == 0 else r"$\hat y\ \mathrm{vs}\ y$", size=26, color=INK, align="center", vcenter=True, z=8)
        g.link(a, b2, 4, 12, GRAY, 4); g.link(b2, c, 4, 12, GRAY, 4)
        if r_ == 0: g.link(c, d, 4, 12, RED, 5)
        else: g.link(b2, d, 8, 8, GRN, 5)
        g.text(60, y - 36, 900, 34, f"{nm}:" + desc, size=22, color=col, bold=True, z=8)
    g.beat = 3
    g.rect(60, 500, 1160, 110, fill=CARD, stroke=GRAY, sw=3, radius=14, z=3)
    g.text(70, 506, 1140, 40, T("MADALINE:把多顆 ADALINE 疊成層", "MADALINE: stack several ADALINEs into layers"), size=22, color=INK, bold=True, align="center", z=8)
    g.text(70, 550, 1140, 54, T("1962 MRI(只改輸入到隱藏層)· 1988 MRII(兩段都能改)· 1990 MRIII(更深)", "1962 MRI (input-to-hidden only) · 1988 MRII (both stages) · 1990 MRIII (deeper)"), size=20, color=MUTED, bold=True, align="center", z=8)
    banner(g, T("階梯只留到推論時才用;訓練時比連續的和", "the step is kept for inference only; training compares the continuous sum"))
    pages.append((g, T("比較的對象", "What is compared"), {
        0: T("Perceptron 訓練時，先用階梯切成 0 和 1，再和標籤比，只知道對或錯。", "The Perceptron first cuts the sum into 0 and 1 with the step, then compares with the label, knowing only right or wrong."),
        1: T("1960 年 Widrow 與 Hoff 的 ADALINE 做了小改變：直接拿還沒過階梯的加權和和標籤比，損失是 LMS 二次損失。階梯只留到推論時才用。", "In 1960 Widrow and Hoff's ADALINE made a small change: compare the weighted sum before the step with the label, using the LMS quadratic loss. The step is kept for inference only."),
        2: T("把多顆 ADALINE 疊成層就是 MADALINE：1962 年的 MRI 只能更新輸入到隱藏層，1988 年的 MRII 兩段都能更新，1990 年的 MRIII 支援更深的訓練。", "Stacking several ADALINEs gives MADALINE: MRI in 1962 updated only input-to-hidden weights, MRII in 1988 updated both stages, MRIII in 1990 supported deeper training."),
        3: T("", "")}))

    # ============ 2: LMS update ============
    g = new(2, T("LMS:梯度就是「（標籤 − 預測）× 輸入」", "LMS: the gradient is “(label − prediction) × input”"))
    g.beat = 1
    g.rect(50, 120, 1180, 150, fill=BLUEL, stroke=BLUE, sw=3, radius=16, z=3)
    g.text(60, 122, 1160, 72, r"$e=\frac12\sum_{i=1}^{b}(y_i-\hat y_i)^2$", size=34, color=INK, align="center", vcenter=True, z=8)
    g.text(60, 196, 1160, 70, r"$\frac{\partial e}{\partial w_j}=-\sum_i(y_i-\hat y_i)\,x_{ij},\qquad \frac{\partial e}{\partial w_0}=-\sum_i(y_i-\hat y_i)$", size=28, color=INK, align="center", vcenter=True, z=8)
    g.beat = 2
    g.text(50, 290, 1180, 40, T("OR 資料,從 $\\mathbf{w}=(0,0,0)$ 開始,$\\eta=0.1$", "OR data, start at $\\mathbf{w}=(0,0,0)$, $\\eta=0.1$"), size=22, color=MUTED, bold=True, align="center", z=8)
    st = [(BLUE, BLUEL, T("所有輸出", "all outputs"), r"$\hat y_i=0$"), (ORG, ORGL, T("殘差", "residuals"), r"$y_i-\hat y_i=(0,1,1,1)$"), (GRN, GRNL, T("更新一步", "one step"), r"$\mathbf{w}=(0.2,\,0.2,\,0.3)$")]
    ii = []
    for i, (col, light, nm, v) in enumerate(st):
        x = 50 + i * 410
        ii.append(g.mark()); g.rect(x, 350, 360, 170, fill=light, stroke=col, sw=3, radius=16, z=3)
        g.text(x, 358, 360, 44, nm, size=24, color=col, bold=True, align="center", z=8); g.text(x + 6, 410, 348, 100, v, size=26, color=INK, align="center", vcenter=True, z=8)
        if i: g.link(ii[i - 1], ii[i], 4, 12, GRAY, 5)
    g.beat = 3
    g.rect(190, 546, 900, 70, fill="#fef3c7", stroke=GOLD, sw=3, radius=16, z=3)
    g.text(190, 546, 900, 70, T("損失由 $e=\\frac12(0+1+1+1)=1.5$ 降到 $0.34$", "the loss falls from $e=\\frac12(0+1+1+1)=1.5$ to $0.34$"), size=26, color=INK, bold=True, align="center", vcenter=True, z=8)
    banner(g, T("和 KP07 的 MSE 是同一種損失，作用在一個神經元上", "the same loss as KP07's MSE, applied to one neuron"))
    pages.append((g, T("LMS 更新", "The LMS update"), {
        0: T("損失是殘差平方和的一半，對權重求梯度，得到負的標籤減預測乘輸入。更新就是梯度下降。", "The loss is half the sum of squared residuals; its gradient is minus the label minus prediction times the input. The update is gradient descent."),
        1: T("用 OR 資料手算：從全 0 開始，學習率 0.1，所有輸出是 0，殘差是 0、1、1、1，一步得到權重 0.2、0.2、0.3。", "By hand on OR data: from all zeros with learning rate 0.1, all outputs are 0 and the residuals are 0, 1, 1, 1; one step gives weights 0.2, 0.2, 0.3."),
        2: T("損失由 1.5 降到 0.34。", "The loss falls from 1.5 to 0.34."),
        3: T("", "")}))

    # ============ 3: loss curve ============
    g = new(3, T("損失一步步往下走", "The loss walks downhill"))
    g.beat = 1
    x0, y0, w_, h_ = 80, 130, 760, 440
    g.rect(x0, y0, w_, h_, fill=CARD, stroke=GRAY, sw=3, radius=12, z=1)
    NI = 24; ux = lambda i: x0 + 40 + i / (NI - 1) * (w_ - 80); uy = lambda e: y0 + h_ - 40 - e / 1.6 * (h_ - 80)
    g.line([(ux(i), uy(ES[i])) for i in range(NI)], GRN, sw=5, z=5)
    g.beat = 2
    for i in range(4):
        g.ellipse(ux(i), uy(ES[i]), 11, fill=ORG, stroke="#ffffff", sw=3, z=7)
        if i < 2: g.text(ux(i) + 14, uy(ES[i]) - 26, 130, 34, f"${ES[i]:.3g}$", size=18, color=ORG, bold=True, z=8)
    g.line([(x0 + 40, uy(0.125)), (x0 + w_ - 40, uy(0.125))], RED, sw=3, dash=True, z=4)
    g.text(x0 + w_ - 400, uy(0.125) - 40, 380, 34, T("最小值 $0.125$", "minimum $0.125$"), size=20, color=RED, bold=True, align="right", z=8)
    g.text(x0, y0 + h_ + 4, w_, 36, T("橫軸:第幾步 縱軸:損失 $e$", "x: step · y: loss $e$"), size=18, color=MUTED, bold=True, align="center", z=8)
    g.beat = 3
    g.rect(900, 150, 330, 380, fill=GRNL, stroke=GRN, sw=3, radius=16, z=3)
    g.text(910, 160, 310, 50, T("最後收斂到", "converges to"), size=24, color=GRN, bold=True, align="center", z=8)
    g.text(910, 214, 310, 120, r"$\mathbf{w}=(0.5,\,0.5,\,0.25)$", size=26, color=INK, align="center", vcenter=True, z=8)
    g.text(910, 350, 310, 150, T("最小平方解\n損失 $e=0.125$", "least-squares solution\nloss $e=0.125$"), size=24, color=INK, bold=True, align="center", z=8)
    banner(g, T("每一步都往損失更小的方向走:連續的損失才做得到", "every step goes toward a smaller loss: only a continuous loss allows it"))
    pages.append((g, T("損失的軌跡", "The loss path"), {
        0: T("繼續跑下去，損失一路往下。", "Keep running and the loss goes down steadily."),
        1: T("前四步的損失依序是 1.5、0.34、0.183、0.159。", "The losses of the first four steps are 1.5, 0.34, 0.183, 0.159."),
        2: T("最後收斂到最小平方解，權重 0.5、0.5、0.25，損失 0.125。Perceptron 的離散錯誤數，沒辦法這樣一步一步往下走。", "It converges to the least-squares solution with weights 0.5, 0.5, 0.25 and loss 0.125. The Perceptron's discrete error count cannot walk downhill like this."),
        3: T("", "")}))

    # ============ 4: boundary compare ============
    g = new(4, T("ADALINE 的邊界落在兩群中間", "ADALINE's boundary lies between the two groups"))
    side = 480; sc = side / 2.0; ux = lambda v: 60 + sc * 0.3 + v * sc; uy = lambda v: 130 + side - sc * 0.3 - v * sc
    g.beat = 1
    g.rect(60, 130, side, side, fill=CARD, stroke=GRAY, sw=3, radius=14, z=1)
    for (p, y) in OR: g.ellipse(ux(p[0]), uy(p[1]), 22, fill=(GRN if y else CARD), stroke=(GRN if y else BLUE), sw=4, z=6)
    g.line([(ux(-0.2), uy(1.7)), (ux(1.7), uy(-0.2))], RED, sw=5, dash=True, z=5); g.text(ux(1.0) + 6, uy(0.15) - 6, 200, 34, T("Perceptron $x_1+x_2=1$", "Perceptron $x_1+x_2=1$"), size=18, color=RED, bold=True, z=8) if False else None
    g.beat = 2
    g.line([(ux(-0.2), uy(0.7)), (ux(0.7), uy(-0.2))], GRN, sw=6, z=5)
    g.text(60, 614, side, 50, T("綠實線:ADALINE　紅虛線:KP02 的 Perceptron", "green line: ADALINE · red dashed: KP02's Perceptron"), size=18, color=MUTED, bold=True, align="center", z=8)
    g.beat = 3
    cx = [600, 780, 940, 1090]
    for k, h in enumerate((r"$(x_1,x_2)$", T("輸出", "output"), T("門檻 0.5", "threshold 0.5"))):
        g.text([600, 820, 1000][k], 150, [200, 160, 220][k], 40, h, size=20, color=MUTED, bold=True, align="center", vcenter=True, z=8)
    for i, ((p, y), o) in enumerate(zip(OR, OUT)):
        yy = 200 + i * 84; pred = 1 if o >= 0.5 else 0
        g.rect(590, yy, 640, 70, fill=GRNL if pred == y else REDL, stroke=GRN if pred == y else RED, sw=3, radius=12, z=3)
        g.text(600, yy, 200, 70, f"$({p[0]},{p[1]})$", size=26, color=INK, bold=True, align="center", vcenter=True, z=8); g.text(820, yy, 160, 70, f"${o:.2f}$", size=26, color=INK, bold=True, align="center", vcenter=True, z=8); g.text(1000, yy, 220, 70, f"${pred}$", size=28, color=GRN, bold=True, align="center", vcenter=True, z=8)
    g.text(590, 548, 640, 70, T("$(1,1)$ 輸出 $1.25$ 雖然分對，LMS 仍把它往 1 拉回來", "$(1,1)$ gives $1.25$: right, yet LMS still pulls it back toward 1"), size=20, color=MUTED, bold=True, align="center", z=8)
    banner(g, T("邊界 $x_1+x_2=0.5$，在 $(0,0)$ 與其餘三點的中間", "the boundary $x_1+x_2=0.5$ sits between $(0,0)$ and the other three points"))
    pages.append((g, T("邊界的位置", "Where the boundary sits"), {
        0: T("KP02 的 Perceptron 停在 x1 加 x2 等於 1，剛好壓在 0 1 與 1 0 上。", "KP02's Perceptron stopped at x1 plus x2 equal 1, right on top of the points 0 1 and 1 0."),
        1: T("ADALINE 的邊界是 x1 加 x2 等於 0.5，落在 0 0 與另外三點的中間。", "ADALINE's boundary is x1 plus x2 equal 0.5, between 0 0 and the other three points."),
        2: T("四個輸出是 0.25、0.75、0.75、1.25，以 0.5 為門檻就分成 0、1、1、1，是正確的 OR。所以 ADALINE 訓練完，推論時仍要用門檻。", "The four outputs are 0.25, 0.75, 0.75, 1.25; thresholded at 0.5 they become 0, 1, 1, 1, the correct OR. So after training, ADALINE still uses a threshold at inference."),
        3: T("", "")}))

    # ============ 5: logistic regression ============
    g = new(5, T("邏輯回歸:一顆神經元加 sigmoid", "Logistic regression: one neuron plus a sigmoid"))
    g.beat = 1
    a = g.mark(); g.rect(60, 150, 260, 120, fill=BLUEL, stroke=BLUE, sw=3, radius=14, z=3); g.text(60, 150, 260, 120, r"$\boldsymbol\beta^\top\mathbf{x}$", size=32, color=INK, align="center", vcenter=True, z=8)
    b2 = g.mark(); g.rect(430, 150, 260, 120, fill=ORGL, stroke=ORG, sw=3, radius=14, z=3); g.text(430, 150, 260, 120, "sigmoid", size=28, color=ORG, bold=True, align="center", vcenter=True, z=8)
    c = g.mark(); g.rect(800, 150, 420, 120, fill=GRNL, stroke=GRN, sw=3, radius=14, z=3); g.text(800, 150, 420, 120, T("機率 $p\\in(0,1)$", "probability $p\\in(0,1)$"), size=28, color=GRN, bold=True, align="center", vcenter=True, z=8)
    g.link(a, b2, 4, 12, GRAY, 5); g.link(b2, c, 4, 12, GRAY, 5)
    g.beat = 2
    g.rect(60, 320, 560, 210, fill=PURL, stroke=PUR, sw=3, radius=16, z=3)
    g.text(60, 328, 560, 44, T("損失:最大似然 = KP07 的二元交叉熵", "loss: maximum likelihood = KP07's binary cross-entropy"), size=20, color=PUR, bold=True, align="center", z=8)
    g.text(70, 380, 540, 140, r"$-\sum[y\ln p+(1-y)\ln(1-p)]$", size=28, color=INK, align="center", vcenter=True, z=8)
    g.beat = 3
    g.rect(660, 320, 560, 210, fill="#fef3c7", stroke=GOLD, sw=3, radius=16, z=3)
    g.text(660, 328, 560, 44, T("梯度:兩者長得一樣", "gradient: they look the same"), size=20, color=INK, bold=True, align="center", z=8)
    g.text(670, 376, 540, 70, r"$\sum(y_i-p_i)\,\mathbf{x}_i$" + T("　(邏輯回歸)", "  (logistic)"), size=24, color=INK, align="center", vcenter=True, z=8); g.text(670, 446, 540, 70, r"$\sum(y_i-\hat y_i)\,\mathbf{x}_i$" + T("　(ADALINE)", "  (ADALINE)"), size=24, color=INK, align="center", vcenter=True, z=8)
    banner(g, T("都是「(標籤 − 預測) × 輸入」，差別只在預測是線性輸出還是 sigmoid 輸出", "both are “(label − prediction) × input”; only the prediction differs: linear or sigmoid"))
    pages.append((g, T("邏輯回歸", "Logistic regression"), {
        0: T("邏輯回歸就是一顆神經元加 sigmoid：線性部分沒有上下界，用 sigmoid 壓到 0 與 1 之間，直接建模後驗機率。", "Logistic regression is one neuron plus a sigmoid: the linear part has no bounds, and the sigmoid squashes it into 0 to 1, modeling the posterior probability directly."),
        1: T("損失從最大似然來，取負號就是 KP07 的二元交叉熵。", "The loss comes from maximum likelihood; with a minus sign it is KP07's binary cross-entropy."),
        2: T("它的梯度和 ADALINE 並排看：兩者都是標籤減預測乘輸入，差別只在預測是線性輸出還是 sigmoid 輸出。", "Put its gradient next to ADALINE's: both are label minus prediction times input; they differ only in whether the prediction is linear or sigmoid."),
        3: T("", "")}))

    # ============ 6: Newton step ============
    g = new(6, T("牛頓法一步:$(0,0)\\to(-1.2,0.8)$", "One Newton step: $(0,0)\\to(-1.2,0.8)$"))
    g.beat = 1
    g.text(50, 112, 1180, 40, T("一維資料 $x=(0,1,2,3)$,$y=(0,1,0,1)$;從 $\\boldsymbol\\beta=(0,0)$ 開始，所有 $p_i=0.5$", "1-D data $x=(0,1,2,3)$, $y=(0,1,0,1)$; start at $\\boldsymbol\\beta=(0,0)$, all $p_i=0.5$"), size=22, color=MUTED, bold=True, align="center", z=8)
    ii = []
    cards = [(BLUE, BLUEL, T("梯度", "gradient"), r"$g=\sum(y_i-p_i)(1,x_i)=(0,\,1)$"), (ORG, ORGL, T("矩陣", "matrix"), r"$\mathbf{X}\mathbf{W}\mathbf{X}^\top=\begin{pmatrix}1&1.5\\1.5&3.5\end{pmatrix}$,  $\det=1.25$"), (GRN, GRNL, T("一步", "one step"), r"$\boldsymbol\beta=(0,0)+\tfrac{1}{1.25}\begin{pmatrix}3.5&-1.5\\-1.5&1\end{pmatrix}\begin{pmatrix}0\\1\end{pmatrix}=(-1.2,\,0.8)$")]
    for i, (col, light, nm, v) in enumerate(cards):
        g.beat = i + 1
        y = 170 + i * 130
        ii.append(g.mark()); g.rect(50, y, 1180, 116, fill=light, stroke=col, sw=3, radius=14, z=3)
        g.text(60, y, 190, 116, nm, size=26, color=col, bold=True, align="center", vcenter=True, z=8); g.text(260, y, 960, 116, v, size=26, color=INK, align="center", vcenter=True, z=8)
        if i: g.link(ii[i - 1], ii[i], 8, 0, GRAY, 4)
    g.beat = 4
    g.rect(190, 560, 900, 60, fill="#fef3c7", stroke=GOLD, sw=3, radius=14, z=3)
    g.text(190, 560, 900, 60, T("對數似然 $-2.7726\\ (=-4\\ln2)\\to-2.3526$", "log-likelihood $-2.7726\\ (=-4\\ln2)\\to-2.3526$"), size=24, color=INK, bold=True, align="center", vcenter=True, z=8)
    banner(g, T("要最大化對數似然:Hessian 為負，牛頓法的更新是加號", "maximizing the log-likelihood: the Hessian is negative, so the Newton update has a plus sign"))
    pages.append((g, T("牛頓法手算", "Newton by hand"), {
        0: T("四筆一維資料，不可分，所以最大似然解有限。從 beta 等於 0、0 開始，所有機率都是 0.5，梯度是 0、1。", "Four 1-D points that are not separable, so the maximum-likelihood solution is finite. Start at beta 0, 0, all probabilities 0.5, so the gradient is 0, 1."),
        1: T("矩陣 X W X 轉置等於 1、1.5、1.5、3.5，行列式 1.25。", "The matrix X W X transpose equals 1, 1.5, 1.5, 3.5 with determinant 1.25."),
        2: T("一步：beta 變成負 1.2、0.8。", "One step: beta becomes minus 1.2, 0.8."),
        3: T("對數似然從負 2.7726 升到負 2.3526。因為要最大化對數似然，Hessian 為負，牛頓法的更新是加號，而不是最小化時的減號。", "The log-likelihood rises from minus 2.7726 to minus 2.3526. Since we maximize, the Hessian is negative and the Newton update has a plus sign, not the minus used when minimizing.")}))

    # ============ 7: converged curve ============
    g = new(7, T("收斂後的機率曲線", "The converged probability curve"))
    x0, y0, w_, h_ = 60, 130, 760, 440
    ux = lambda v: x0 + 60 + v / 3.6 * (w_ - 120) + 0; uy = lambda p: y0 + h_ - 50 - p * (h_ - 100)
    g.beat = 1
    g.rect(x0, y0, w_, h_, fill=CARD, stroke=GRAY, sw=3, radius=12, z=1)
    g.line([(x0 + 20, uy(0.5)), (x0 + w_ - 20, uy(0.5))], GRAY, sw=2, dash=True, z=2)
    pts = [(ux(-0.2 + 3.6 * i / 80), uy(1 / (1 + math.exp(-(BF[0] + BF[1] * (-0.2 + 3.6 * i / 80)))))) for i in range(81)]
    g.line(pts, BLUE, sw=5, z=5)
    for x, y in zip(XS, YS): g.ellipse(ux(x), uy(y), 16, fill=(GRN if y else CARD), stroke=(GRN if y else RED), sw=4, z=7)
    g.text(x0, y0 + h_ + 4, w_, 36, T("橫軸:$x$ 縱軸:$p$;圓點是資料的標籤", "x: $x$ · y: $p$; the dots are the labels"), size=18, color=MUTED, bold=True, align="center", z=8)
    g.beat = 2
    g.line([(ux(1.5), uy(0) - 6), (ux(1.5), uy(1) + 6)], RED, sw=4, dash=True, z=4); g.text(ux(1.5) + 8, uy(1) - 8, 200, 34, T("邊界 $x=1.5$", "boundary $x=1.5$"), size=20, color=RED, bold=True, z=8)
    g.beat = 3
    g.rect(860, 130, 370, 440, fill=BLUEL, stroke=BLUE, sw=3, radius=16, z=3)
    g.text(870, 140, 350, 50, T("收斂後", "after convergence"), size=24, color=BLUE, bold=True, align="center", z=8)
    g.text(870, 196, 350, 70, r"$\boldsymbol\beta=(-1.362,\,0.908)$", size=24, color=INK, align="center", vcenter=True, z=8)
    g.text(870, 280, 350, 70, r"$p=0.204,\ 0.388,\ 0.612,\ 0.796$", size=22, color=INK, align="center", vcenter=True, z=8)
    g.text(870, 370, 350, 190, T("$x=1$ 與 $x=2$ 仍判錯:資料交錯，是資料的問題，不是演算法的問題", "$x=1$ and $x=2$ are still wrong: the data interleave; the data, not the algorithm, are at fault"), size=20, color=INK, bold=True, align="center", z=8)
    banner(g, T("第三步起幾乎不變:牛頓法收斂得快", "from the third step on almost nothing changes: Newton converges fast"))
    pages.append((g, T("收斂結果", "The converged result"), {
        0: T("牛頓法第三步起幾乎不變，收斂的機率曲線是一條 S 形。圓點是資料的標籤。", "From the third step Newton's method barely changes; the converged probability curve is an S. The dots are the labels."),
        1: T("邊界在 p 等於 0.5，也就是 x 等於 1.5。", "The boundary is at p equal 0.5, that is x equal 1.5."),
        2: T("beta 收斂在負 1.362、0.908，四個機率是 0.204、0.388、0.612、0.796。因為資料交錯，x 等於 1 與 2 這兩筆仍判錯，這是資料的問題，不是演算法的問題。", "Beta converges to minus 1.362, 0.908 and the four probabilities are 0.204, 0.388, 0.612, 0.796. Because the data interleave, x equal 1 and 2 are still wrong; that is the data's problem, not the algorithm's."),
        3: T("", "")}))

    # ============ 8: myths + question ============
    g = new(8, T("三個迷思，與一個問題", "Three myths and a question"))
    mm = [(RED, REDL, T("「ADALINE 就是 Perceptron 換名字」", "“ADALINE is the Perceptron renamed”"), T("差別在比較的對象:階梯後的 0/1，或階梯前的連續和", "the difference is what is compared: the 0/1 after the step or the continuous sum before it")),
          (ORG, ORGL, T("「邏輯回歸是回歸，輸出是連續數」", "“logistic regression outputs continuous numbers”"), T("它是分類模型;「回歸」指回歸出機率", "it is a classifier; “regression” means regressing a probability")),
          (PUR, PURL, T("「牛頓法更新前面是減號」", "“Newton's update always has a minus sign”"), T("最小化才是減號;最大化對數似然是加號", "minus is for minimizing; maximizing the log-likelihood gives plus"))]
    for i, (col, light, a, b_) in enumerate(mm):
        g.beat = i + 1 if i < 3 else 3
        x = 50 + i * 400
        g.rect(x, 130, 370, 290, fill=light, stroke=col, sw=3, radius=16, z=3)
        g.text(x + 8, 140, 354, 100, a, size=22, color=col, bold=True, align="center", z=8); g.text(x + 8, 250, 354, 160, b_, size=22, color=INK, bold=True, align="center", z=8)
    g.beat = 3
    g.rect(50, 450, 1000, 160, fill="#fef3c7", stroke=GOLD, sw=3, radius=16, z=3)
    g.text(70, 458, 960, 146, T("到目前為止，第一層都是學出來的權重。如果第一層固定成一組「局部」的基函數，只學第二層，會怎樣？另一個完全不用標籤的做法是什麼？", "So far the first layer's weights were always learned. What if the first layer is fixed as a set of “local” basis functions and only the second layer is learned? And what works with no labels at all?"), size=22, color=INK, bold=True, align="center", z=8)
    mascot(g, 1100, 440, 0.5, "confused")
    banner(g, T("下一個單元:RBF 與 SOM", "next unit: RBF and SOM"))
    pages.append((g, T("迷思與問題", "Myths and question"), {
        0: T("第一個迷思：ADALINE 就是 Perceptron 換名字。差別在訓練時比較的對象。", "First myth: ADALINE is the Perceptron renamed. The difference is what is compared in training."),
        1: T("第二個迷思：邏輯回歸輸出連續數。它是分類模型，回歸指的是回歸出機率。", "Second myth: logistic regression outputs continuous numbers. It is a classifier; regression means regressing a probability."),
        2: T("第三個迷思：牛頓法前面一定是減號。最小化才是減號，最大化對數似然是加號。留給下一個單元：第一層固定成局部基函數會怎樣？這就是 RBF；完全不用標籤的做法是 SOM。", "Third myth: Newton's update always has a minus sign. Minus is for minimizing; maximizing the log-likelihood gives plus. For the next unit: what if the first layer is fixed as local basis functions? That is RBF; the approach with no labels at all is SOM."),
        3: T("", "")}))
    return pages
