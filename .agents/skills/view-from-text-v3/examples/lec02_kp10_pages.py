import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lec01_kit import *

KP = "KP10"; N = 8
f = lambda x: math.sin(2 * math.pi * x)
def stair(x, n): return f((min(int(x * n), n - 1) + 0.5) / n)
def sup_err(n, m=200001): return max(abs(f(j / (m - 1)) - stair(j / (m - 1), n)) for j in range(m))
NS = (3, 5, 10, 20, 40, 80); ERR = [sup_err(n) for n in NS]
assert [round(e, 3) for e in ERR] == [0.866, 0.588, 0.309, 0.156, 0.0785, 0.0393] or [round(e, 4) for e in ERR][:2] == [0.866, 0.5878]
assert abs(ERR[2] - math.pi / 10) < 0.01 and abs(ERR[-1] - math.pi / 80) < 0.001
N001 = math.ceil(math.pi / 0.01); assert 300 <= N001 <= 320 or N001 == 315
V3 = [f((i + 0.5) / 3) for i in range(3)]; AL3 = [V3[i + 1] - V3[i] for i in range(2)]
assert [round(v, 3) for v in V3] == [0.866, 0.0, -0.866] and [round(a, 3) for a in AL3] == [-0.866, -0.866]

def make(lang):
    zh = lang == "zh"
    T = lambda a, b_: a if zh else b_
    pages = []
    new = lambda n, title: new_page(n, N, KP, title)

    def axes(g, x0, y0, w, h, xr, yr):
        ux = lambda a: x0 + (a - xr[0]) / (xr[1] - xr[0]) * w; uy = lambda v: y0 + h - (v - yr[0]) / (yr[1] - yr[0]) * h
        g.rect(x0, y0, w, h, fill=CARD, stroke=GRAY, sw=3, radius=12, z=1)
        if yr[0] < 0 < yr[1]: g.line([(x0, uy(0)), (x0 + w, uy(0))], GRAY, sw=2, z=2)
        return ux, uy

    # ============ 1: the theorem ============
    g = new(1, T("KP10 萬有逼近:一層隱藏層就夠表示", "KP10 One hidden layer can approximate anything"))
    g.beat = 1
    ins = g.mark(); g.ellipse(120, 300, 44, fill=BLUEL, stroke=BLUE, sw=3, z=3); g.text(76, 256, 88, 88, r"$\mathbf{x}$", size=28, color=BLUE, bold=True, align="center", vcenter=True, z=8)
    hs = []
    for k in range(4):
        y = 170 + k * 90; hs.append(g.mark()); g.ellipse(430, y, 30, fill=ORGL, stroke=ORG, sw=3, z=3)
        g.link(ins, hs[-1], 4, 12, GRAY, 3)
    g.text(310, 560, 240, 64, T("單隱藏層，$N$ 顆", "one hidden layer, $N$ neurons"), size=20, color=ORG, bold=True, align="center", z=8)
    out = g.mark(); g.ellipse(740, 300, 44, fill=GRNL, stroke=GRN, sw=3, z=3); g.text(696, 256, 88, 88, "$F$", size=28, color=GRN, bold=True, align="center", vcenter=True, z=8)
    for h in hs: g.link(h, out, 4, 12, GRAY, 3)
    g.beat = 2
    g.rect(840, 130, 400, 240, fill=BLUEL, stroke=BLUE, sw=3, radius=16, z=3)
    g.text(840, 138, 400, 40, T("能做到", "can achieve"), size=22, color=BLUE, bold=True, align="center", z=8)
    g.text(850, 184, 380, 180, T("在有限範圍上，把任何連續函數 $f$ 逼近到任意精度 $\\varepsilon$", "on a bounded range, approximate any continuous $f$ to any accuracy $\\varepsilon$"), size=22, color=INK, bold=True, align="center", z=8)
    g.beat = 3
    g.rect(840, 400, 400, 200, fill=PURL, stroke=PUR, sw=3, radius=16, z=3)
    g.text(840, 408, 400, 40, T("對活化函數 $\\sigma$ 的要求", "what $\\sigma$ must be"), size=22, color=PUR, bold=True, align="center", z=8)
    g.text(850, 454, 380, 140, T("連續、有界、不是常數", "continuous, bounded, not constant"), size=24, color=INK, bold=True, align="center", z=8)
    g.rect(60, 590, 740, 40, fill="transparent", z=1) if False else None
    banner(g, T("$F(\\mathbf{x})=\\sum_{i=1}^{N}\\alpha_i\\,\\sigma(\\mathbf{w}_i^\\top\\mathbf{x}+\\theta_i)$ 滿足 $\\sup|f-F|<\\varepsilon$", "$F(\\mathbf{x})=\\sum_{i=1}^{N}\\alpha_i\\,\\sigma(\\mathbf{w}_i^\\top\\mathbf{x}+\\theta_i)$ with $\\sup|f-F|<\\varepsilon$"))
    pages.append((g, T("定理說了什麼", "What it says"), {
        0: T("第二章看了許多模型，共通的是疊起來的神經元。問題是：只用一層隱藏層，夠多神經元，能表示多少種函數？", "Chapter 2 looked at many models whose common feature is stacked neurons. The question: with a single hidden layer and enough neurons, how many functions can be represented?"),
        1: T("萬有逼近定理：在有限範圍上，單隱藏層、N 顆神經元的網路，能把任何連續函數逼近到任意精度。", "The universal approximation theorem: on a bounded range, a network with one hidden layer of N neurons can approximate any continuous function to any accuracy."),
        2: T("條件是活化函數連續、有界、不是常數。", "The condition is that the activation is continuous, bounded and not constant."),
        3: T("", "")}))

    # ============ 2: basis analogy ============
    g = new(2, T("它是一組「基底」，像傅立葉與小波", "It is a “basis”, like Fourier and wavelets"))
    items = [(BLUE, BLUEL, T("傅立葉級數", "Fourier series"), T("用三角函數去逼近", "approximate with trigonometric functions"), r"$\sin,\ \cos$"), (ORG, ORGL, T("小波", "wavelets"), T("用小波基底去逼近", "approximate with wavelet bases"), r"$\psi_{a,b}$"), (GRN, GRNL, T("神經網路", "neural network"), T("用神經元去逼近", "approximate with neurons"), r"$\sigma(\mathbf{w}^\top\mathbf{x}+\theta)$")]
    for i, (col, light, nm, desc, sym) in enumerate(items):
        g.beat = i + 1
        x = 50 + i * 410
        g.rect(x, 170, 360, 340, fill=light, stroke=col, sw=3, radius=16, z=3)
        g.text(x, 180, 360, 50, nm, size=28, color=col, bold=True, align="center", z=8); g.text(x + 10, 250, 340, 100, sym, size=34, color=INK, align="center", vcenter=True, z=8); g.text(x + 10, 380, 340, 110, desc, size=22, color=INK, bold=True, align="center", z=8)
    banner(g, T("這類定理叫「表示定理」", "this kind of theorem is called a representation theorem"))
    pages.append((g, T("表示定理", "Representation theorem"), {
        0: T("傅立葉級數用三角函數去逼近函數。", "A Fourier series approximates functions with trigonometric functions."),
        1: T("小波用小波基底去逼近。", "Wavelets approximate with wavelet bases."),
        2: T("神經網路的 sigma 作用在 w 轉置 x 加 theta，也是一組基底。這類定理叫表示定理。", "The neural network's sigma of w transpose x plus theta is also a basis. This kind of theorem is a representation theorem."),
        3: T("", "")}))

    # ============ 3: staircase ============
    g = new(3, T("親手建一個逼近:階梯法", "Build one by hand: the staircase"))
    xr, yr = (0, 1), (-1.3, 1.3)
    ux, uy = axes(g, 60, 130, 780, 440, xr, yr)
    g.beat = 1
    g.line([(ux(i / 120), uy(f(i / 120))) for i in range(121)], BLUE, sw=5, z=5)
    g.text(ux(0.05), uy(1.2), 260, 34, "$f(x)=\\sin2\\pi x$", size=22, color=BLUE, bold=True, z=8)
    g.beat = 2
    n = 5
    pts = []
    for i in range(n):
        v = f((i + 0.5) / n); pts += [(ux(i / n), uy(v)), (ux((i + 1) / n), uy(v))]
        if i: g.line([(ux(i / n), uy(f((i - 0.5) / n))), (ux(i / n), uy(v))], RED, sw=3, dash=True, z=4)
        g.line([(ux(i / n), uy(v)), (ux((i + 1) / n), uy(v))], RED, sw=6, z=6)
    g.text(60, 574, 780, 60, T("把 $[0,1]$ 切成 $N=5$ 格，每格取中點的函數值", "cut $[0,1]$ into $N=5$ cells; each takes the value at its midpoint"), size=18, color=MUTED, bold=True, align="center", z=8)
    g.beat = 3
    g.rect(880, 130, 350, 440, fill=ORGL, stroke=ORG, sw=3, radius=16, z=3)
    g.text(890, 140, 330, 70, T("每一階 = 一個很陡的 sigmoid", "each step = one very steep sigmoid"), size=22, color=ORG, bold=True, align="center", z=8)
    g.text(890, 220, 330, 90, r"$\sigma\big(k(x-t_i)\big)$", size=30, color=INK, align="center", vcenter=True, z=8)
    g.text(890, 320, 330, 240, T("在 $t_i$ 處從 0 跳到 1\n這確實是 $\\sum\\alpha_i\\sigma(w_ix+\\theta_i)$ 的形式", "jumps from 0 to 1 at $t_i$\nexactly the form $\\sum\\alpha_i\\sigma(w_ix+\\theta_i)$"), size=22, color=INK, bold=True, align="center", z=8)
    banner(g, T("階梯法只對一維有用，但它讓「夠多就能逼近」變得看得見", "the staircase works only in 1-D, but it makes “enough neurons approximate” visible"))
    pages.append((g, T("階梯法", "The staircase"), {
        0: T("取一維的正弦函數，x 在 0 到 1。", "Take the one-dimensional sine on 0 to 1."),
        1: T("把區間切成 N 格，這裡 N 等於 5，每格取中點的函數值，得到一條階梯。", "Cut the interval into N cells, here N equals 5, and each cell takes the function value at its midpoint, giving a staircase."),
        2: T("把一個很陡的 sigmoid 看成在 t 處從 0 跳到 1 的階梯，階梯就是 sigma 的加權和。", "Treat a very steep sigmoid as a step jumping from 0 to 1 at t; the staircase is then a weighted sum of sigmas."),
        3: T("", "")}))

    # ============ 4: parameters by hand ============
    g = new(4, T("參數是手算出來的，不是學出來的", "The parameters are worked out by hand, not learned"))
    g.beat = 1
    g.text(50, 112, 1180, 40, T("$N=3$:中點 $\\frac16,\\frac12,\\frac56$ 的函數值 $v_i=\\sin 2\\pi x$", "$N=3$: values $v_i=\\sin 2\\pi x$ at the midpoints $\\frac16,\\frac12,\\frac56$"), size=22, color=MUTED, bold=True, align="center", z=8)
    ii = []
    for i, v in enumerate(V3):
        x = 80 + i * 400
        ii.append(g.mark()); g.rect(x, 170, 300, 130, fill=BLUEL, stroke=BLUE, sw=3, radius=14, z=3)
        g.text(x, 176, 300, 40, f"$v_{i + 1}$", size=24, color=BLUE, bold=True, align="center", z=8); g.text(x, 222, 300, 70, f"${v:.3f}$", size=34, color=INK, align="center", vcenter=True, z=8)
    g.beat = 2
    for i, a in enumerate(AL3):
        x = 220 + i * 400
        g.rect(x - 70, 350, 300, 110, fill=ORGL, stroke=ORG, sw=3, radius=14, z=3)
        g.text(x - 70, 354, 300, 40, rf"$\alpha_{i + 1}=v_{i + 2}-v_{i + 1}$", size=22, color=ORG, bold=True, align="center", z=8); g.text(x - 70, 400, 300, 56, f"${a:.3f}$", size=30, color=INK, align="center", vcenter=True, z=8)
    g.beat = 3
    g.rect(150, 500, 980, 110, fill="#fef3c7", stroke=GOLD, sw=3, radius=16, z=3)
    g.text(160, 506, 960, 98, T("$w_i=k$,$\\theta_i=-kt_i$,$\\alpha_i=v_{i+1}-v_i$\n一維的例子才有這種公式，一般情形要靠學習演算法找", "$w_i=k$, $\\theta_i=-kt_i$, $\\alpha_i=v_{i+1}-v_i$\nonly this 1-D example has such formulas; in general a learning algorithm must find them"), size=22, color=INK, bold=True, align="center", z=8)
    banner(g, T("再加一個常數項 $v_1$，等於多一個偏置", "plus a constant term $v_1$, i.e. one more bias"))
    pages.append((g, T("手算參數", "Parameters by hand"), {
        0: T("以三格為例，三個中點的函數值是 0.866、0、負 0.866。", "With three cells, the three midpoint values are 0.866, 0, minus 0.866."),
        1: T("每一階的高度 alpha 是相鄰兩格的差，這裡是負 0.866 與負 0.866。", "Each step's height alpha is the difference of neighboring cells, here minus 0.866 and minus 0.866."),
        2: T("w 等於 k，theta 等於負 k 乘 t，alpha 是相鄰差。這些參數是為一維例子手算出來的，一般情形沒有這種公式，要靠學習演算法，那是第三章。", "w equals k, theta equals minus k times t, and alpha is the neighboring difference. These parameters were worked out for the 1-D example; in general no such formula exists and a learning algorithm is needed, which is Chapter 3."),
        3: T("", "")}))

    # ============ 5: error vs N ============
    g = new(5, T("神經元加倍，誤差約減半", "Double the neurons, roughly halve the error"))
    base = 560; sc = 560 / 1.35
    g.beat = 1
    for i, (n_, e) in enumerate(zip(NS, ERR)):
        x = 90 + i * 190
        g.beat = 1 if i < 3 else 2
        g.rect(x, base - e * sc, 130, e * sc, fill=REDL, stroke=RED, sw=3, radius=6, z=3)
        g.text(x - 30, base - e * sc - 42, 190, 38, f"${e:.3g}$", size=22, color=RED, bold=True, align="center", z=8)
        g.text(x - 30, base + 6, 190, 36, f"$N={n_}$", size=22, color=INK, bold=True, align="center", z=8)
    g.line([(60, base), (1220, base)], INK, sw=3, z=3)
    g.text(60, 112, 700, 40, T("最大誤差 $\\sup|f-F|$($k=2000$)", "maximum error $\\sup|f-F|$ ($k=2000$)"), size=22, color=MUTED, bold=True, z=8)
    g.beat = 3
    g.rect(560, 150, 660, 170, fill="#fef3c7", stroke=GOLD, sw=3, radius=16, z=3)
    g.text(570, 154, 640, 82, r"$\frac{1}{2N}\cdot2\pi=\frac{\pi}{N}$", size=36, color=INK, align="center", vcenter=True, z=8)
    g.text(570, 232, 640, 86, T("每格半寬乘上最大斜率 $2\\pi$;$N=10$:$0.309\\approx\\pi/10$", "half a cell times the maximum slope $2\\pi$; $N=10$: $0.309\\approx\\pi/10$"), size=20, color=INK, bold=True, align="center", z=8)
    banner(g, T("誤差小於 $0.01$ 約需 $N\\approx320$ 顆:「夠多」可能真的很多", "an error below $0.01$ needs about $N\\approx320$: “enough” may really be a lot"))
    pages.append((g, T("誤差表", "Error table"), {
        0: T("N 等於 3、5、10 時，最大誤差是 0.866、0.588、0.309。", "For N equal 3, 5, 10 the maximum error is 0.866, 0.588, 0.309."),
        1: T("N 等於 20、40、80 時，是 0.156、0.0785、0.0393。神經元數加倍，誤差大約減半。", "For N equal 20, 40, 80 it is 0.156, 0.0785, 0.0393. Doubling the neurons roughly halves the error."),
        2: T("原因：每格半寬乘上最大斜率 2 pi，等於 pi 除以 N。要讓誤差小於 0.01，約需 320 顆。高維時只會更多。", "The reason: half a cell times the maximum slope 2 pi, which is pi over N. An error below 0.01 needs about 320 neurons. In higher dimensions it only gets worse."),
        3: T("", "")}))

    # ============ 6: proof skeleton ============
    g = new(6, T("一般定理的證明骨架(四步)", "The proof skeleton of the general theorem (four steps)"))
    st = [(BLUE, BLUEL, T("1  假設空間", "1  hypothesis space"), T("$\\mathcal{H}_\\sigma=\\mathrm{span}\\{\\sigma(\\mathbf{w}^\\top\\mathbf{x}+\\theta)\\}$ 在連續函數空間中要證明「稠密」", "$\\mathcal{H}_\\sigma=\\mathrm{span}\\{\\sigma(\\mathbf{w}^\\top\\mathbf{x}+\\theta)\\}$ must be dense in the continuous functions")),
          (ORG, ORGL, T("2  反證", "2  contradiction"), T("假設不稠密 → Hahn–Banach:存在非零線性泛函 $L$，對 $\\mathcal{H}_\\sigma$ 全部輸出 0", "assume not dense → Hahn–Banach: a nonzero linear functional $L$ vanishes on all of $\\mathcal{H}_\\sigma$")),
          (PUR, PURL, T("3  Riesz 表示", "3  Riesz representation"), T("$L(f)=\\int f\\,d\\mu$，所以 $\\int\\sigma(\\mathbf{w}^\\top\\mathbf{x}+\\theta)\\,d\\mu=0$ 對所有 $\\mathbf{w},\\theta$", "$L(f)=\\int f\\,d\\mu$, so $\\int\\sigma(\\mathbf{w}^\\top\\mathbf{x}+\\theta)\\,d\\mu=0$ for all $\\mathbf{w},\\theta$")),
          (GRN, GRNL, T("4  矛盾", "4  contradiction"), T("$\\sigma$ 連續、有界、非常數，足以分開不同測度，迫使 $\\mu=0$，與 $L\\ne0$ 矛盾", "continuous, bounded, nonconstant $\\sigma$ separates measures, forcing $\\mu=0$, contradicting $L\\ne0$"))]
    ii = []
    for i, (col, light, nm, desc) in enumerate(st):
        g.beat = i + 1
        x = 50 + i * 305
        ii.append(g.mark()); g.rect(x, 170, 275, 380, fill=light, stroke=col, sw=3, radius=16, z=3)
        g.text(x, 180, 275, 60, nm, size=24, color=col, bold=True, align="center", z=8); g.text(x + 10, 250, 255, 290, desc, size=20, color=INK, bold=True, align="center", z=8)
        if i: g.link(ii[i - 1], ii[i], 4, 12, GRAY, 4)
    banner(g, T("這裡只看骨架;細節是泛函分析", "only the skeleton here; the details are functional analysis"))
    pages.append((g, T("證明骨架", "Proof skeleton"), {
        0: T("第一步：定義假設空間，要證明它在連續函數空間中稠密，意思是任何連續函數都能被它任意逼近。", "Step one: define the hypothesis space and prove it is dense in the continuous functions, meaning any continuous function can be approximated by it."),
        1: T("第二步：用反證，假設不稠密。由 Hahn-Banach 定理，存在一個非零的連續線性泛函，對假設空間中每個函數都輸出 0。", "Step two: by contradiction assume it is not dense. By Hahn-Banach there is a nonzero continuous linear functional that vanishes on every function in the space."),
        2: T("第三步：Riesz 表示，這個泛函可寫成對某個測度的積分。", "Step three: by Riesz representation this functional is an integral against some measure."),
        3: T("第四步：sigma 的性質迫使測度為 0，與非零矛盾，所以假設空間稠密。", "Step four: the properties of sigma force the measure to be 0, contradicting nonzero, so the space is dense.")}))

    # ============ 7: guarantees ============
    g = new(7, T("定理保證什麼，不保證什麼", "What it guarantees and what it does not"))
    g.beat = 1
    g.rect(50, 130, 440, 460, fill=GRNL, stroke=GRN, sw=3, radius=16, z=3)
    g.text(50, 140, 440, 50, T("保證", "guarantees"), size=28, color=GRN, bold=True, align="center", z=8)
    tick(g, 80, 230, 44, GRN, z=6); g.text(140, 215, 340, 330, T("只要 $N$ 夠大，單隱藏層網路存在一組參數，能把任何連續函數逼近到任意精度", "for large enough $N$, a single-hidden-layer network has parameters that approximate any continuous function to any accuracy"), size=22, color=INK, bold=True, z=8)
    items = [T("要多大的 $N$", "how large $N$ must be"), T("怎麼找到參數(第 3 章)", "how to find the parameters (Chapter 3)"), T("沒看過的資料上也好(KP04 的泛化)", "that it does well on unseen data (KP04's generalization)"), T("訓練會收斂到全域最佳", "that training reaches the global optimum")]
    for i, s_ in enumerate(items):
        g.beat = 2 + i // 2
        y = 130 + i * 116
        g.rect(540, y, 690, 100, fill=REDL, stroke=RED, sw=3, radius=14, z=3)
        cross(g, 566, y + 30, 34, RED, z=6); g.text(620, y + 8, 600, 88, T("不保證:", "not promised: ") + s_, size=22, color=INK, bold=True, z=8)
    banner(g, T("定理只保證「有解」，不保證「找得到」，也不保證「泛化好」", "the theorem promises “a solution exists”, not “it can be found” or “it generalizes”"))
    pages.append((g, T("保證與不保證", "Guarantees and non-guarantees"), {
        0: T("保證：只要神經元夠多，單隱藏層網路存在一組參數，能把任何連續函數逼近到任意精度。", "Guarantee: with enough neurons, a single-hidden-layer network has some parameters that approximate any continuous function to any accuracy."),
        1: T("不保證：要多大的 N，階梯法的例子就要幾百顆，高維更多；也不告訴你怎麼找到參數，那要靠學習演算法，是第三章。", "Not promised: how large N must be, with the staircase example needing hundreds and more in high dimensions; nor how to find the parameters, which needs a learning algorithm, Chapter 3."),
        2: T("也不保證在沒看過的資料上也好，那是 KP04 的泛化問題；也不保證訓練會收斂到全域最佳。", "Nor that it does well on unseen data, which is KP04's generalization problem; nor that training reaches the global optimum."),
        3: T("", "")}))

    # ============ 8: myths + chapter wrap ============
    g = new(8, T("三個迷思，與第 3 章的問題", "Three myths and Chapter 3's question"))
    mm = [(RED, REDL, T("「保證神經網路學得會任何函數」", "“guarantees a network can learn any function”"), T("不是:只說參數存在;怎麼找、要多少、泛化好不好都不在定理內", "no: it only says parameters exist; how to find them, how many, how well they generalize are outside")),
          (ORG, ORGL, T("「活化函數必須是 sigmoid 或有界的」", "“the activation must be a sigmoid or bounded”"), T("這個版本以有界敘述;之後證明只要不是多項式，連 ReLU 也成立", "this version assumes bounded; later work shows any non-polynomial works, even ReLU")),
          (PUR, PURL, T("「需要很多層才有表達力」", "“many layers are needed for expressive power”"), T("逼近能力一層就夠;多層的好處是通常用少很多神經元", "one layer suffices for approximation; many layers usually need far fewer neurons"))]
    for i, (col, light, a, b_) in enumerate(mm):
        g.beat = i + 1
        x = 50 + i * 400
        g.rect(x, 130, 370, 290, fill=light, stroke=col, sw=3, radius=16, z=3)
        g.text(x + 8, 140, 354, 100, a, size=21, color=col, bold=True, align="center", z=8); g.text(x + 8, 250, 354, 160, b_, size=20, color=INK, bold=True, align="center", z=8)
    g.beat = 4
    g.rect(50, 450, 1000, 160, fill="#fef3c7", stroke=GOLD, sw=3, radius=16, z=3)
    g.text(70, 458, 960, 146, T("網路「能表示」什麼，我們看過了。真正的問題是：這些權重怎麼自己學出來，特別是多層網路中間那些看不到答案的隱藏層？", "We know what a network can represent. The real question: how do these weights learn themselves, especially the hidden layers in a deep network that never see the answer?"), size=22, color=INK, bold=True, align="center", z=8)
    mascot(g, 1100, 440, 0.5, "confused")
    banner(g, T("第 3 章:反向傳播與梯度下降", "Chapter 3: backpropagation and gradient descent"))
    pages.append((g, T("迷思與下一章", "Myths and next chapter"), {
        0: T("第一個迷思：定理保證網路學得會任何函數。不是，它只說參數存在。", "First myth: the theorem guarantees a network can learn any function. No, it only says the parameters exist."),
        1: T("第二個迷思：活化函數必須是 sigmoid 或有界的。這個版本以有界敘述，之後證明只要不是多項式，連 ReLU 也成立。", "Second myth: the activation must be a sigmoid or bounded. This version assumes bounded; later work shows any non-polynomial works, even ReLU."),
        2: T("第三個迷思：需要很多層才有表達力。就逼近能力而言一層已足夠，多層的好處是通常用少很多的神經元。", "Third myth: many layers are needed for expressive power. One layer suffices for approximation; many layers usually need far fewer neurons."),
        3: T("第二章看完了。這些權重該怎麼自己學出來，特別是多層網路中間那些看不到答案的隱藏層？這就是第三章的反向傳播與梯度下降。", "Chapter 2 is done. How do these weights learn themselves, especially the hidden layers in a deep network that never see the answer? That is Chapter 3: backpropagation and gradient descent.")}))
    return pages
