import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lec01_kit import *

KP = "KP06"; N = 8
sig = lambda a: 1 / (1 + math.exp(-a)); tanh = math.tanh
relu = lambda a: max(0.0, a); leaky = lambda a: a if a >= 0 else 0.01 * a
elu = lambda a: a if a >= 0 else math.exp(a) - 1; softplus = lambda a: math.log(1 + math.exp(a))
AV = (2, 1, 0); EX = [math.exp(v) for v in AV]; SM = [e / sum(EX) for e in EX]
assert [round(v, 3) for v in EX] == [7.389, 2.718, 1.0] and round(sum(EX), 3) == 11.107 and [round(v, 3) for v in SM] == [0.665, 0.245, 0.09] and abs(sum(SM) - 1) < 1e-12
assert sig(0) == 0.5 and sig(0) * (1 - sig(0)) == 0.25 and round(0.25 ** 5, 5) == 0.00098
TAB = {"ReLU": (relu(-2), relu(2), 0, 1), "Leaky ReLU": (leaky(-2), leaky(2), 0.01, 1), "ELU": (elu(-2), elu(2), math.exp(-2), 1), "Softplus": (softplus(-2), softplus(2), sig(-2), sig(2))}
assert [round(v, 3) for v in TAB["ELU"][:3]] == [-0.865, 2, 0.135] and [round(v, 3) for v in TAB["Softplus"]] == [0.127, 2.127, 0.119, 0.881] and TAB["Leaky ReLU"][0] == -0.02
fmtn = lambda v: ("%g" % round(v, 3)).replace("-", "-")

def make(lang):
    zh = lang == "zh"
    T = lambda a, b_: a if zh else b_
    pages = []
    new = lambda n, title: new_page(n, N, KP, title)

    def axes(g, x0, y0, w, h, xr, yr, tag=None):
        ux = lambda a: x0 + (a - xr[0]) / (xr[1] - xr[0]) * w; uy = lambda v: y0 + h - (v - yr[0]) / (yr[1] - yr[0]) * h
        g.rect(x0, y0, w, h, fill=CARD, stroke=GRAY, sw=3, radius=12, z=1)
        if yr[0] < 0 < yr[1]: g.line([(x0, uy(0)), (x0 + w, uy(0))], GRAY, sw=2, z=2)
        if xr[0] < 0 < xr[1]: g.line([(ux(0), y0), (ux(0), y0 + h)], GRAY, sw=2, z=2)
        return ux, uy
    def curve(g, ux, uy, fn, col, xr, yr, sw=5, n=60):
        pts = []
        for i in range(n + 1):
            a = xr[0] + (xr[1] - xr[0]) * i / n; v = min(max(fn(a), yr[0] - 0.05), yr[1] + 0.05); pts.append((ux(a), uy(v)))
        g.line(pts, col, sw=sw, smooth=False, z=5)

    # ============ 1: the gate ============
    g = new(1, T("KP06 活化函數:神經元門口的閘門", "KP06 Activation functions: the gate at each neuron"))
    g.beat = 1
    a = g.mark(); g.rect(60, 200, 260, 140, fill=BLUEL, stroke=BLUE, sw=3, radius=16, z=3); g.text(60, 200, 260, 140, T("加權和 $a$", "weighted sum $a$"), size=28, color=BLUE, bold=True, align="center", vcenter=True, z=8)
    b2 = g.mark(); g.rect(500, 170, 280, 200, fill=ORGL, stroke=ORG, sw=4, radius=20, z=3); g.text(500, 180, 280, 70, T("閘門", "gate"), size=28, color=ORG, bold=True, align="center", vcenter=True, z=8); g.text(500, 256, 280, 100, r"$\sigma(a)$", size=44, color=INK, align="center", vcenter=True, z=8)
    c = g.mark(); g.rect(960, 200, 260, 140, fill=GRNL, stroke=GRN, sw=3, radius=16, z=3); g.text(960, 200, 260, 140, T("輸出 $z$", "output $z$"), size=28, color=GRN, bold=True, align="center", vcenter=True, z=8)
    g.link(a, b2, 4, 12, GRAY, 6); g.link(b2, c, 4, 12, GRAY, 6)
    g.beat = 2
    g.rect(60, 420, 560, 170, fill=PURL, stroke=PUR, sw=3, radius=16, z=3); g.text(70, 430, 540, 50, T("決定 1:輸出的範圍", "decides 1: the output range"), size=24, color=PUR, bold=True, align="center", z=8); g.text(70, 486, 540, 90, T("能不能當機率?", "can it be read as a probability?"), size=26, color=INK, bold=True, align="center", z=8)
    g.beat = 3
    g.rect(660, 420, 560, 170, fill=REDL, stroke=RED, sw=3, radius=16, z=3); g.text(670, 430, 540, 50, T("決定 2:導數的大小", "decides 2: the derivative size"), size=24, color=RED, bold=True, align="center", z=8); g.text(670, 486, 540, 90, T("學習時梯度傳不傳得回去?", "can the gradient flow back during learning?"), size=26, color=INK, bold=True, align="center", z=8)
    banner(g, T("階梯是硬開關;換成別的閘門，網路才學得動", "the step is a hard switch; other gates make the network learnable"))
    pages.append((g, T("閘門", "The gate"), {
        0: T("每顆神經元門口有一個閘門，把加權和 a 轉成輸出 z，這就是活化函數。", "Each neuron has a gate that turns the weighted sum a into the output z; that is the activation function."),
        1: T("選哪一種，決定兩件事。第一，輸出的範圍：能不能當機率。", "The choice decides two things. First, the output range: can it be read as a probability."),
        2: T("第二，導數的大小：學習時梯度傳不傳得回去。", "Second, the size of the derivative: can the gradient flow back during learning."),
        3: T("", "")}))

    # ============ 2: four families ============
    g = new(2, T("十三種活化函數，分四族", "Thirteen activation functions in four families"))
    fam = [(BLUE, BLUEL, T("基礎", "basic"), ["Linear", "Binary step", "Sign", "Softmax"]), (ORG, ORGL, T("S 形", "S-shaped"), ["Sigmoid", "tanh", "Gaussian"]),
           (GRN, GRNL, T("ReLU 族", "ReLU family"), ["Softplus", "ReLU", "Leaky ReLU", "PReLU", "ELU"]), (PUR, PURL, T("跨層", "cross-unit"), ["Maxout"])]
    tot = 0
    for i, (col, light, nm, items) in enumerate(fam):
        g.beat = 1 + i if i < 3 else 3
        x = 50 + i * 300
        g.rect(x, 130, 270, 460, fill=light, stroke=col, sw=3, radius=16, z=3)
        g.text(x, 138, 270, 50, nm, size=28, color=col, bold=True, align="center", z=8)
        for k, it in enumerate(items):
            g.rect(x + 20, 200 + k * 76, 230, 60, fill=CARD, stroke=col, sw=3, radius=12, z=4); g.text(x + 20, 200 + k * 76, 230, 60, it, size=22, color=INK, bold=True, align="center", vcenter=True, z=8); tot += 1
    assert tot == 13
    banner(g, T("Softmax 與 Maxout 要整層一起算，其餘逐點作用", "softmax and maxout need the whole layer; the rest act pointwise"))
    pages.append((g, T("四族", "Four families"), {
        0: T("基礎族：線性、二元階梯、符號、Softmax。", "Basic family: linear, binary step, sign, softmax."),
        1: T("S 形：Sigmoid、tanh、高斯。", "S-shaped: sigmoid, tanh, Gaussian."),
        2: T("ReLU 族：Softplus、ReLU、Leaky ReLU、PReLU、ELU。", "ReLU family: softplus, ReLU, Leaky ReLU, PReLU, ELU."),
        3: T("跨層：Maxout。一共十三種。Softmax 與 Maxout 要看整層一起算，其餘逐點作用；線性放在隱藏層，就是 KP05 的坍縮。", "Cross-unit: maxout. Thirteen in all. Softmax and maxout need the whole layer; the rest act pointwise; linear in a hidden layer is KP05's collapse.")}))

    # ============ 3: curves ============
    g = new(3, T("S 形有飽和，ReLU 族不飽和", "S-shapes saturate; the ReLU family does not"))
    xr = (-4, 4)
    g.beat = 1
    yr1 = (-1.3, 1.3); ux, uy = axes(g, 60, 150, 560, 380, xr, yr1)
    curve(g, ux, uy, sig, BLUE, xr, yr1); curve(g, ux, uy, tanh, ORG, xr, yr1)
    g.text(60, 110, 560, 36, T("S 形", "S-shaped"), size=24, color=INK, bold=True, align="center", z=8)
    g.text(ux(1.6), uy(0.62) + 8, 120, 34, "sigmoid", size=18, color=BLUE, bold=True, z=8); g.text(ux(-3.9), uy(-0.9) - 8, 100, 34, "tanh", size=18, color=ORG, bold=True, z=8)
    g.beat = 2
    yr2 = (-1.5, 4.0); ux2, uy2 = axes(g, 660, 150, 560, 380, xr, yr2)
    for fn, col, nm, pos in ((softplus, ORG, "softplus", (-3.9, 0.45)), (elu, RED, "ELU", (-3.9, -1.25)), (leaky, PUR, "Leaky ReLU", (-3.9, -0.8)), (relu, GRN, "ReLU", (2.3, 2.2))):
        curve(g, ux2, uy2, fn, col, xr, yr2, sw=4 if nm != "ReLU" else 6)
        g.text(ux2(pos[0]), uy2(pos[1]) - 12, 160, 32, nm, size=17, color=col, bold=True, z=8)
    g.text(660, 110, 560, 36, T("ReLU 族", "ReLU family"), size=24, color=INK, bold=True, align="center", z=8)
    g.beat = 3
    g.text(60, 548, 560, 60, T("兩端變平:導數趨近 0", "both ends flatten: the derivative tends to 0"), size=22, color=BLUE, bold=True, align="center", z=8)
    g.text(660, 548, 560, 60, T("正側一直往上:導數不縮小", "keeps rising on the positive side: the derivative does not shrink"), size=22, color=GRN, bold=True, align="center", z=8)
    banner(g, T("曲線的形狀，決定訊號與學習的回饋能不能順利通過", "the curve's shape decides whether signal and learning feedback get through"))
    pages.append((g, T("曲線", "The curves"), {
        0: T("S 形的 sigmoid 與 tanh：兩端都變平，這叫飽和。", "The S-shaped sigmoid and tanh: both ends flatten, which is called saturation."),
        1: T("ReLU 族：ReLU、Leaky ReLU、ELU、softplus，正側一直往上。", "The ReLU family: ReLU, Leaky ReLU, ELU, softplus rise on the positive side."),
        2: T("S 形兩端導數趨近 0；ReLU 族正側導數不縮小。", "S-shapes have a derivative tending to 0 at both ends; the ReLU family keeps its derivative on the positive side."),
        3: T("", "")}))

    # ============ 4: softmax ============
    g = new(4, T("Softmax:把一組分數變成機率", "Softmax: turn a set of scores into probabilities"))
    g.beat = 1
    g.text(60, 112, 560, 40, T("加權和 $\\mathbf{a}=(2,1,0)$", "weighted sums $\\mathbf{a}=(2,1,0)$"), size=26, color=INK, bold=True, align="center", z=8)
    bw = 130; base = 560
    for i, (av, e) in enumerate(zip(AV, EX)):
        x = 100 + i * 170; hgt = e * 40
        g.rect(x, base - hgt, bw, hgt, fill=BLUEL, stroke=BLUE, sw=3, radius=8, z=3); g.text(x - 20, base - hgt - 40, bw + 40, 36, f"$e^{av}={e:.3f}$", size=20, color=BLUE, bold=True, align="center", z=8)
    g.line([(80, base), (590, base)], INK, sw=3, z=3)
    g.beat = 2
    g.text(640, 112, 590, 40, T("除以總和 $11.107$", "divide by the total $11.107$"), size=26, color=INK, bold=True, align="center", z=8)
    for i, p_ in enumerate(SM):
        x = 700 + i * 170; hgt = p_ * 360
        g.rect(x, base - hgt, bw, hgt, fill=GRNL, stroke=GRN, sw=3, radius=8, z=3); g.text(x - 20, base - hgt - 40, bw + 40, 36, f"${p_:.3f}$", size=22, color=GRN, bold=True, align="center", z=8)
    g.line([(680, base), (1210, base)], INK, sw=3, z=3)
    g.link(g.mark() - 1, g.mark() - 1, 4, 12, GRAY, 1) if False else None
    g.beat = 3
    g.rect(640, 150, 590, 40, fill="transparent", z=1) if False else None
    banner(g, T("$0.665+0.245+0.090=1$:輸出可直接讀成「屬於三類的機率」", "$0.665+0.245+0.090=1$: read the outputs as class probabilities"))
    pages.append((g, T("Softmax 算例", "Softmax example"), {
        0: T("輸出層三顆神經元，加權和 2、1、0。先取指數：7.389、2.718、1，總和 11.107。", "Three output neurons with sums 2, 1, 0. Take exponentials: 7.389, 2.718, 1, total 11.107."),
        1: T("除以總和，得 0.665、0.245、0.090。三個數都在 0 到 1 之間，加起來剛好是 1。", "Divide by the total: 0.665, 0.245, 0.090. All lie between 0 and 1 and add up to exactly 1."),
        2: T("所以輸出可以直接讀成屬於三類的機率。", "So the outputs can be read directly as probabilities of the three classes."),
        3: T("", "")}))

    # ============ 5: vanishing gradient ============
    g = new(5, T("S 形疊層:梯度越傳越小", "Stacked S-shapes: the gradient shrinks layer by layer"))
    g.beat = 1
    g.rect(60, 120, 560, 130, fill=BLUEL, stroke=BLUE, sw=3, radius=16, z=3)
    g.text(60, 126, 560, 44, T("Sigmoid 在 $a=0$", "sigmoid at $a=0$"), size=24, color=BLUE, bold=True, align="center", z=8); g.text(70, 176, 540, 64, r"$\sigma=0.5,\quad \sigma(1-\sigma)=0.25$", size=28, color=INK, align="center", vcenter=True, z=8)
    g.rect(660, 120, 560, 130, fill=ORGL, stroke=ORG, sw=3, radius=16, z=3)
    g.text(660, 126, 560, 44, T("tanh 在 $a=0$", "tanh at $a=0$"), size=24, color=ORG, bold=True, align="center", z=8); g.text(670, 176, 540, 64, r"$1-\sigma^2=1$", size=28, color=INK, align="center", vcenter=True, z=8)
    g.beat = 2
    ids = []
    vals = [0.25 ** k for k in range(1, 6)]
    for i, v in enumerate(vals):
        x = 50 + i * 240
        ids.append(g.mark()); g.rect(x, 330, 200, 150, fill=REDL, stroke=RED, sw=3, radius=14, z=3)
        g.text(x, 336, 200, 40, T(f"{i + 1} 層", f"{i + 1} layer" + ("s" if i else "")), size=20, color=RED, bold=True, align="center", z=8); g.text(x, 384, 200, 80, rf"$0.25^{i + 1}$" + "\n" + rf"$={v:.5g}$", size=22, color=INK, bold=True, align="center", z=8)
        if i: g.link(ids[i - 1], ids[i], 4, 12, GRAY, 4)
    g.beat = 3
    g.rect(190, 520, 900, 90, fill="#fef3c7", stroke=GOLD, sw=3, radius=16, z=3)
    g.text(210, 520, 860, 90, T("5 層 sigmoid，最好的情況只剩 $0.00098$:梯度消失", "five sigmoid layers keep at best $0.00098$: vanishing gradient"), size=24, color=INK, bold=True, align="center", vcenter=True, z=8)
    banner(g, T("梯度要一層一層連乘導數往回傳（第 3 章 KP06 完整推導）", "gradients multiply layer by layer on the way back (full derivation in Chapter 3 KP06)"))
    pages.append((g, T("梯度消失", "Vanishing gradient"), {
        0: T("Sigmoid 在 0 輸出 0.5，導數 0.25，這已是它導數的最大值；tanh 在 0 的導數是 1。", "Sigmoid outputs 0.5 at 0 with derivative 0.25, already its maximum; tanh has derivative 1 at 0."),
        1: T("學習時，梯度要一層一層連乘導數往回傳。一層最多乘 0.25，五層連乘越來越小。", "In learning the gradient multiplies derivatives layer by layer on the way back. Each layer multiplies by at most 0.25, so five layers shrink it fast."),
        2: T("五層 sigmoid，最好的情況也只剩 0.00098，這就是梯度消失。", "Five sigmoid layers keep at best 0.00098; this is the vanishing gradient."),
        3: T("", "")}))

    # ============ 6: ReLU family table ============
    g = new(6, T("ReLU 族:正側導數為 1，但要防 dying", "ReLU family: derivative 1, beware dying"))
    g.beat = 1
    cx = [60, 330, 520, 720, 930]; cw = [250, 170, 190, 190, 250]
    for k, h in enumerate((T("函數", "function"), r"$\sigma(-2)$", r"$\sigma(2)$", T("導數於 $a=-2$", "derivative at $a=-2$"), T("導數於 $a=2$", "derivative at $a=2$"))):
        g.text(cx[k], 112, cw[k], 72, h, size=19, color=MUTED, bold=True, align="center", z=8)
    for i, (nm, (v1, v2, d1, d2)) in enumerate(TAB.items()):
        g.beat = 1 if i < 2 else 2
        y = 184 + i * 82
        zero = abs(d1) < 1e-9
        g.rect(50, y, 1180, 68, fill=(REDL if zero else GRNL), stroke=(RED if zero else GRN), sw=3, radius=12, z=3)
        for k, v in enumerate((nm, fmtn(v1), fmtn(v2), fmtn(d1), fmtn(d2))):
            g.text(cx[k], y, cw[k], 68, (v if k == 0 else f"${v}$"), size=24, color=(RED if zero and k == 3 else INK), bold=True, align="center", vcenter=True, z=8)
    g.beat = 3
    g.rect(150, 530, 980, 90, fill="#fef3c7", stroke=GOLD, sw=3, radius=16, z=3)
    g.text(170, 540, 940, 80, T("若某顆神經元的輸入總是負的，ReLU 導數恆為 0，權重永遠不更新:dying ReLU", "if a neuron's inputs are always negative, ReLU's derivative is always 0 and its weights never update: dying ReLU"), size=22, color=INK, bold=True, align="center", z=8)
    banner(g, T("Leaky ReLU 保留 $0.01$ 小斜率;PReLU 讓斜率可學習", "Leaky ReLU keeps a $0.01$ slope; PReLU makes the slope learnable"))
    pages.append((g, T("ReLU 族數值", "ReLU family numbers"), {
        0: T("取 a 等於負 2 與 2。ReLU 在負 2 輸出 0，導數 0；在 2 輸出 2，導數 1。Leaky ReLU 負側保留 0.01 的小斜率。", "Take a equal minus 2 and 2. ReLU outputs 0 at minus 2 with derivative 0, and 2 at 2 with derivative 1. Leaky ReLU keeps a 0.01 slope on the negative side."),
        1: T("ELU 在負 2 導數 0.135；softplus 把轉折處磨平，在兩點導數都不為 0。", "ELU has derivative 0.135 at minus 2; softplus smooths the corner and has a nonzero derivative at both points."),
        2: T("但若某顆神經元的輸入總是負的，ReLU 導數恆為 0，權重永遠不更新，這叫 dying ReLU。", "But if a neuron's inputs are always negative, ReLU's derivative is always 0 and its weights never update: dying ReLU."),
        3: T("", "")}))

    # ============ 7: how to choose ============
    g = new(7, T("怎麼選", "How to choose"))
    g.beat = 1
    h = g.mark(); g.rect(60, 130, 360, 150, fill=ORGL, stroke=ORG, sw=3, radius=16, z=3); g.text(60, 140, 360, 50, T("隱藏層", "hidden layers"), size=26, color=ORG, bold=True, align="center", z=8); g.text(70, 196, 340, 70, T("先試 ReLU 族", "try the ReLU family first"), size=24, color=INK, bold=True, align="center", z=8)
    h2 = g.mark(); g.rect(560, 130, 380, 150, fill=PURL, stroke=PUR, sw=3, radius=16, z=3); g.text(560, 140, 380, 50, T("怕 dying ReLU", "afraid of dying ReLU"), size=24, color=PUR, bold=True, align="center", z=8); g.text(570, 196, 360, 70, T("換 Leaky ReLU 或 PReLU", "use Leaky ReLU or PReLU"), size=24, color=INK, bold=True, align="center", z=8)
    g.link(h, h2, 4, 12, GRAY, 5)
    g.beat = 2
    o = g.mark(); g.rect(60, 360, 280, 90, fill=GRNL, stroke=GRN, sw=3, radius=16, z=3); g.text(60, 360, 280, 90, T("輸出層看任務", "output layer: by task"), size=24, color=GRN, bold=True, align="center", vcenter=True, z=8)
    outs = [(BLUE, BLUEL, T("回歸", "regression"), "Linear"), (ORG, ORGL, T("二元分類", "binary classes"), "Sigmoid"), (PUR, PURL, T("多類互斥", "exclusive classes"), "Softmax"), (RED, REDL, T("多標籤", "multi-label"), T("各自 sigmoid", "one sigmoid each"))]
    for i, (col, light, a, b_) in enumerate(outs):
        x = 400 + i * 215
        r = g.mark(); g.rect(x, 340, 195, 130, fill=light, stroke=col, sw=3, radius=14, z=3); g.text(x, 346, 195, 50, a, size=20, color=col, bold=True, align="center", z=8); g.text(x, 402, 195, 60, b_, size=22, color=INK, bold=True, align="center", vcenter=True, z=8)
        g.link(o, r, 4, 12, col, 3) if i == 0 else g.link(o, r, 8 if i else 4, 0 if i else 12, col, 3)
    g.beat = 3
    g.rect(190, 520, 900, 90, fill="#fef3c7", stroke=GOLD, sw=3, radius=16, z=3)
    g.text(210, 536, 860, 70, T("ReLU、Leaky ReLU、PReLU 在 $a=0$ 不可導，實作上約定一個值(通常取 0 或 1)", "ReLU, Leaky ReLU and PReLU are not differentiable at $a=0$; implementations fix a value there (usually 0 or 1)"), size=22, color=INK, bold=True, align="center", z=8)
    banner(g, T("隱藏層優先 ReLU 家族;輸出層依任務選", "ReLU family for hidden layers; the output layer follows the task"))
    pages.append((g, T("怎麼選", "How to choose"), {
        0: T("隱藏層先試 ReLU 家族；怕 dying ReLU，就換 Leaky ReLU 或 PReLU。", "For hidden layers try the ReLU family first; if dying ReLU is a worry, use Leaky ReLU or PReLU."),
        1: T("輸出層看任務：回歸用 Linear，二元分類用 Sigmoid，多類別互斥分類用 Softmax；一個樣本可同時屬於多類，則每個輸出各自用 sigmoid。", "The output layer follows the task: Linear for regression, Sigmoid for binary classification, Softmax for exclusive multi-class; if a sample can belong to several classes, use one sigmoid per output."),
        2: T("表中的導數都是在可導處。ReLU 家族在 0 不可導，實作上約定一個值。", "The derivatives in the table are where they exist. The ReLU family is not differentiable at 0, so implementations fix a value."),
        3: T("", "")}))

    # ============ 8: myths + question ============
    g = new(8, T("兩個迷思，與一個問題", "Two myths and a question"))
    g.beat = 1
    g.rect(50, 130, 570, 220, fill=REDL, stroke=RED, sw=3, radius=16, z=3)
    g.text(60, 138, 550, 56, T("迷思", "myth"), size=24, color=RED, bold=True, align="center", z=8)
    g.text(60, 194, 550, 150, T("「ReLU 處處可導，所以最安全」\n它在 $a=0$ 不可導，負側導數為 0，可能讓神經元死掉", "“ReLU is differentiable everywhere, so safest”\nit is not at $a=0$, and its negative-side derivative is 0, so neurons can die"), size=22, color=INK, bold=True, align="center", z=8)
    g.beat = 2
    g.rect(660, 130, 570, 220, fill=ORGL, stroke=ORG, sw=3, radius=16, z=3)
    g.text(670, 138, 550, 56, T("迷思", "myth"), size=24, color=ORG, bold=True, align="center", z=8)
    g.text(670, 194, 550, 150, T("「sigmoid 與 softmax 是同一回事」\nsigmoid 對單一個數，輸出彼此獨立；softmax 對整層，輸出加起來為 1", "“sigmoid and softmax are the same”\nsigmoid acts on one number, outputs independent; softmax acts on the layer, outputs sum to 1"), size=22, color=INK, bold=True, align="center", z=8)
    g.beat = 3
    g.rect(50, 390, 1000, 210, fill="#fef3c7", stroke=GOLD, sw=3, radius=16, z=3)
    g.text(70, 400, 960, 190, T("網路輸出了一個結果，要和正確答案比，把「差多少」變成一個數，才知道往哪個方向調權重。這個數該怎麼定義？", "The network gives an output; we compare it with the right answer and turn “how far off” into one number, to know which way to adjust the weights. How should that number be defined?"), size=24, color=INK, bold=True, align="center", z=8)
    mascot(g, 1100, 400, 0.55, "confused")
    banner(g, T("下一個單元:損失函數", "next unit: loss functions"))
    pages.append((g, T("迷思與問題", "Myths and question"), {
        0: T("第一個迷思：ReLU 處處可導所以最安全。它在 0 不可導，負側導數為 0，可能讓神經元死掉。", "First myth: ReLU is differentiable everywhere and so safest. It is not at 0 and its negative-side derivative is 0, so neurons can die."),
        1: T("第二個迷思：sigmoid 與 softmax 是同一回事。sigmoid 對單一個數，輸出彼此獨立；softmax 對整層，輸出加起來為 1。", "Second myth: sigmoid and softmax are the same. Sigmoid acts on one number with independent outputs; softmax acts on the layer and its outputs sum to 1."),
        2: T("留給下一個單元：把差多少變成一個數，這個數該怎麼定義？", "For the next unit: how should “how far off” be turned into one number?"),
        3: T("", "")}))
    return pages
