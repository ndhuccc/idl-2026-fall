"""KP09 pages designed in HTML (view-from-text-v3, route A). pageN(lang) -> body HTML of a 1160x480 figure."""
import os, sys, math
import numpy as np
SK = os.environ.get("COURSE_SKILLS") or (os.path.join(os.path.abspath(os.environ["COURSE_REPO"]), ".agents", "skills") if os.environ.get("COURSE_REPO") else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
HERE = os.path.dirname(os.path.abspath(__file__)); HERE = HERE[:-5] if HERE.endswith(("/W_zh", "/W_en")) else HERE
sys.path.insert(0, SK + "/view-from-text-v3/examples"); sys.path.insert(0, SK + "/view-from-text-v3/tools"); sys.path.insert(0, HERE)
from kp17_pages import Axes, tr, take, legend, fmt, sci, txt, named, card, head, chip, formula, svg, BL, RD, VI, GR, YE, OR, TE, INK, TXT, MUT, hexc, build, native, _sup, rgba, rgb
from kp15_pages import box, cardH
import kp09_nums as K
W, H = 1160, 480
C1, C2 = "#7fb0e6", "#e8896a"      # blue / orange line colours on the dark canvas

# ------------------------------------------------ 1: two faults of the squared error
def page1(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 700, 480, "blue", t("目標為 1 時，兩種損失隨輸出 ŷ 的變化", "Target 1: two losses against the output ŷ"))
    ax = Axes(70, 66, 600, 330, (0, 1), (0, 7.5)); ax.frame([0, .25, .5, .75, 1], [0, 2, 4, 6], yfmt=lambda v: f"{v:g}")
    xs = np.linspace(0.001, 1, 400)
    ax.line([(float(x), float(0.5 * (1 - x) ** 2)) for x in xs], C1, 4); ax.line([(float(x), float(min(-math.log(x), 7.4))) for x in xs], C2, 4)
    ax.dot(0.001, 0.5 * (1 - .001) ** 2, "#f2ede4", 8); ax.text(0.04, 0.5, "0.499", "#f2ede4", "start", 16, True, 6, 24)
    ax.dot(0.001, -math.log(.001), "#f2ede4", 8); ax.text(0.045, 6.9, "6.908", "#f2ede4", "start", 16, True, 6, 5)
    o += ax.html() + txt(70, 426, 600, t("輸出 ŷ（目標 y = 1）", "output ŷ (target y = 1)"), 16, MUT, False, "center", 24)
    o += legend([(C1, t("平方誤差 ½(1 − ŷ)²", "squared error ½(1 − ŷ)²"))], 330, 80, 330) + legend([(C2, "−ln ŷ")], 330, 110, 330)
    o += cardH(716, 0, 444, 480, "orange", t("平方誤差的兩個毛病", "Two faults of the squared error"))
    o += card(732, 52, 412, 168, "red", .2) + txt(740, 58, 396, t("不公平", "Unfair"), 19, TXT, True, "center", 30)
    o += txt(740, 92, 396, t("誤差先平方再加總，大誤差主導學習：誤差 0.9 與 0.1 差 9 倍，平方後是 0.81 與 0.01，差 81 倍。", "Errors are squared before they are summed, so large errors dominate: 0.9 and 0.1 differ 9 times, but 0.81 and 0.01 differ 81 times."), 16, TXT, False, "left", 118)
    o += card(732, 232, 412, 168, "red", .2) + txt(740, 238, 396, t("會卡住", "Gets stuck"), 19, TXT, True, "center", 30)
    o += txt(740, 272, 396, t("輸出錯到極端（目標 1、輸出 0.001）時，平方誤差只有 0.499，不會再變大，梯度下降可能被困住；−ln ŷ 此時是 6.908。", "At the extreme (target 1, output 0.001) the squared error is only 0.499 and stops growing, so gradient descent can get stuck; −ln ŷ is 6.908 there."), 16, TXT, False, "left", 118)
    o += txt(732, 412, 412, t("接下來的問題：輸出錯到極端時，δ 到底有多小？〔程式計算〕", "Next: how small is δ at the extreme? (computed)"), 16, TXT, True, "center", 52)
    return o

# ------------------------------------------------ 2: plateau
def page2(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 700, 480, "blue", t("平方誤差的 δ 對 net（目標 y = 1）", "Squared-error δ against net (target y = 1)"))
    ax = Axes(70, 66, 600, 330, (-10, 10), (0, 0.16)); ax.frame([-10, -5, 0, 5, 10], [0, .05, .1, .15], yfmt=lambda v: f"{v:g}")
    ax.rect(-10, -5, 0, 0.16, "#d97757", .12); ax.rect(5, 10, 0, 0.16, "#d97757", .12)
    ns = np.linspace(-10, 10, 300); ax.line([(float(n), float((1 - K.sig(n)) * K.sig(n) * (1 - K.sig(n)))) for n in ns], C1, 4)
    ax.dot(-10, 0.000045, "#f2ede4", 8); ax.text(-9.6, 0.0, "net = −10, δ = 0.000045", "#f2ede4", "start", 16, True, 0, -14)
    ax.text(-7.5, 0.14, t("高原", "plateau"), C2, "middle", 18, True); ax.text(7.5, 0.14, t("高原", "plateau"), C2, "middle", 18, True)
    o += ax.html() + txt(70, 426, 600, "net", 16, MUT, False, "center", 24)
    o += cardH(716, 0, 444, 480, "orange", t("為什麼 δ 被壓小", "Why δ gets crushed"))
    o += named(732, 48, 412, t("高原：代價很大、梯度卻幾乎為 0 的平坦區，梯度下降在上面幾乎不動。", "Plateau: a flat region where the cost is large but the gradient is almost 0, so gradient descent hardly moves."), "term:" + t("高原", "Plateau"), 16, TXT, 100)
    o += card(732, 156, 412, 70, "violet", .22) + formula(732, 164, 412, r"\delta=(y-\hat y)\,\hat y(1-\hat y)", 24, 54)
    o += txt(732, 238, 412, t("ŷ 越接近 0 或 1，斜率 ŷ(1 − ŷ) 越小，把大誤差也壓成小訊號。", "The closer ŷ is to 0 or 1, the smaller the slope ŷ(1 − ŷ), which squeezes even a large error into a small signal."), 16, TXT, False, "left", 76)
    o += card(732, 322, 412, 130, "yellow", .14) + txt(740, 328, 396, t("y = 1、net = −10：", "y = 1, net = −10:"), 17, TXT, True, "center", 28) + txt(740, 358, 396, "ŷ = 0.000045", 19, TXT, True, "center", 30) + txt(740, 392, 396, t("誤差近 1，δ = 0.000045〔程式計算〕", "error near 1, δ = 0.000045 (computed)"), 17, TXT, False, "center", 52)
    return o

# ------------------------------------------------ 3: cross-entropy cancels the slope
def page3(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 560, 480, "violet", t("交叉熵：把斜率約掉", "Cross-entropy cancels the slope"))
    o += named(16, 46, 528, t("交叉熵：把輸出 ŷ 當成「目標為 1 的機率」，J = −(y ln ŷ + (1 − y) ln(1 − ŷ))；y = 1 時是 −ln ŷ。", "Cross-entropy: treat ŷ as the probability that the target is 1, J = −(y ln ŷ + (1 − y) ln(1 − ŷ)); for y = 1, −ln ŷ."), "term:" + t("交叉熵", "Cross-entropy"), 16, TXT, 100)
    steps = [r"\frac{\partial J}{\partial\hat y}=-\frac{y-\hat y}{\hat y(1-\hat y)}", r"\frac{\partial\hat y}{\partial net}=\hat y(1-\hat y)", r"\delta=-\frac{\partial J}{\partial net}=y-\hat y"]
    labs = [t("對輸出微分", "differentiate in ŷ"), t("sigmoid 的斜率", "the sigmoid slope"), t("連鎖律相乘：ŷ(1 − ŷ) 約掉", "chain rule: ŷ(1 − ŷ) cancels")]
    for i, (tx, lb) in enumerate(zip(steps, labs)):
        y0 = 160 + i * 98; o += card(16, y0, 528, 90, "violet" if i == 2 else "grey", .24 if i == 2 else .12) + txt(24, y0 + 4, 512, f"{i+1}. " + lb, 16, MUT, True, "left", 24) + formula(16, y0 + 26, 528, tx, 22, 58)
    o += cardH(576, 0, 584, 480, "teal", t("y = 1 時兩種 δ 的對照", "The two δ compared (y = 1)"))
    for xx, lab, w_ in ((590, "net", 70), (666, "ŷ", 110), (782, t("平方誤差 δ", "squared δ"), 170), (958, t("交叉熵 δ", "cross-entropy δ"), 180)): o += txt(xx, 48, w_, lab, 16, MUT, True, "center", 24)
    for i, net in enumerate((-10, -5, -2, 0, 2, 5)):
        y = float(K.sig(net)); hot = net == -10; y0 = 76 + i * 46; o += card(588, y0, 560, 40, "yellow" if hot else "grey", .26 if hot else .1)
        o += txt(590, y0 + 7, 70, f"{net}", 17, TXT, True, "center", 26) + txt(666, y0 + 7, 110, f"{y:.4g}" if y > .001 else f"{y:.5f}", 17, TXT, False, "center", 26)
        o += txt(782, y0 + 7, 170, f"{(1-y)*y*(1-y):.5f}", 17, TXT, False, "center", 26) + txt(958, y0 + 7, 180, f"{1-y:.4f}", 18, TXT, True, "center", 26)
    o += txt(588, 360, 560, t("交叉熵的 δ 不被壓小；平方誤差在 net = −10 時幾乎沒有訊號（0.00005 對 1.0000）。符號沿用 KP06：δ = −∂J/∂net。〔程式計算〕", "The cross-entropy δ is not squeezed; at net = −10 the squared error has almost no signal (0.00005 against 1.0000). The sign follows KP06: δ = −∂J/∂net. (computed)"), 16, TXT, True, "center", 96)
    return o

# ------------------------------------------------ 4: training comparison
def page4(lang):
    t = tr(lang); o = ""; ec, hc = K.train("ce", track=100); em, hm = K.train("mse", track=100)
    o += cardH(0, 0, 700, 480, "blue", t("x = 2 的輸出 ŷ 隨 epoch 變化", "The output ŷ at x = 2 against the epoch"))
    ax = Axes(70, 66, 600, 330, (0, 100), (0, 1)); ax.frame([0, 20, 40, 60, 80, 100], [0, .5, 1], yfmt=lambda v: f"{v:g}")
    ax.line([(0, float(K.sig(-3 * 2)))] + [(i + 1, h) for i, h in enumerate(hc)], C2, 4); ax.line([(0, float(K.sig(-3 * 2)))] + [(i + 1, h) for i, h in enumerate(hm)], C1, 4)
    ax.line([(0, .5), (100, .5)], "#e0c85f", 2, "8 6", .8)
    ax.dot(ec, hc[ec - 1], "#f2ede4", 9); ax.text(ec, hc[ec - 1], t(f"交叉熵 {ec} 個", f"cross-entropy {ec}"), "#f2ede4", "start", 16, True, 10, -12)
    ax.dot(em, hm[em - 1], "#f2ede4", 9); ax.text(em, hm[em - 1], t(f"平方誤差 {em} 個", f"squared {em}"), "#f2ede4", "end", 16, True, -10, -14)
    o += ax.html() + txt(70, 426, 600, "epoch", 16, MUT, False, "center", 24)
    o += legend([(C2, t("交叉熵", "cross-entropy"))], 420, 230, 250) + legend([(C1, t("平方誤差", "squared error"))], 420, 260, 250)
    o += cardH(716, 0, 444, 480, "orange", t("實驗設定與結果", "Setup and result"))
    st = [t("一個 sigmoid 神經元 ŷ = σ(b + wx)", "One sigmoid neuron ŷ = σ(b + wx)"), t("四點 x = −2, −1, 1, 2，目標 0, 0, 1, 1", "Four points x = −2, −1, 1, 2 with targets 0, 0, 1, 1"), t("起點 w = −3、b = 0，η = 0.5，批次更新", "Start w = −3, b = 0, η = 0.5, batch updates")]
    for i, s_ in enumerate(st):
        y0 = 50 + i * 54; o += card(732, y0, 412, 48, "grey", .12) + txt(740, y0 + 6, 396, f"{i+1}. " + s_, 16, TXT, False, "left", 40)
    o += card(732, 216, 412, 90, "yellow", .16) + txt(740, 224, 396, t(f"四點全對：交叉熵 {ec} 個 epoch，平方誤差 {em} 個〔程式計算〕", f"All four right: {ec} epochs for cross-entropy, {em} for squared error (computed)"), 18, TXT, True, "center", 72)
    o += card(732, 318, 412, 134, "red", .2) + txt(740, 326, 396, t("常見迷思", "Myth"), 16, MUT, True, "center", 24) + txt(740, 354, 396, t("換了交叉熵才分得對。其實兩種代價最後都能分對，差別是起點很錯時的速度。", "Only cross-entropy gets it right. Both costs end up right; the difference is speed when the start is badly wrong."), 16, TXT, False, "center", 96)
    return o

# ------------------------------------------------ 5: likelihood
def page5(lang):
    t = tr(lang); o = ""; y = np.array([.9, .8, .3, .1]); tg = np.array([1, 1, 0, 0]); p = np.where(tg == 1, y, 1 - y); ce = -np.log(p)
    o += cardH(0, 0, 700, 480, "blue", t("四筆資料：輸出 0.9, 0.8, 0.3, 0.1，目標 1, 1, 0, 0", "Four samples: outputs 0.9, 0.8, 0.3, 0.1 with targets 1, 1, 0, 0"))
    for xx, lab, w_ in ((16, t("輸出 ŷ", "output ŷ"), 110), (136, t("目標 y", "target y"), 110), (256, t("目標的機率", "prob. of the target"), 200), (466, t("−ln 機率（交叉熵）", "−ln prob. (cross-entropy)"), 220)): o += txt(xx, 48, w_, lab, 16, MUT, True, "center", 24)
    for i in range(4):
        y0 = 80 + i * 48; o += card(16, y0, 668, 42, "grey", .12)
        o += txt(16, y0 + 8, 110, f"{y[i]:g}", 17, TXT, False, "center", 26) + txt(136, y0 + 8, 110, str(tg[i]), 17, TXT, False, "center", 26)
        o += txt(256, y0 + 8, 200, (f"{p[i]:g}" if tg[i] == 1 else f"1 − {y[i]:g} = {p[i]:g}"), 17, TXT, True, "center", 26) + txt(466, y0 + 8, 220, f"{ce[i]:.3f}", 17, TXT, True, "center", 26)
    o += card(16, 284, 668, 68, "green", .2) + txt(24, 292, 652, t(f"相乘：0.9 × 0.8 × 0.7 × 0.9 = {p.prod():.4f}，−ln {p.prod():.4f} = {-math.log(p.prod()):.4f}", f"Product: 0.9 × 0.8 × 0.7 × 0.9 = {p.prod():.4f}, −ln {p.prod():.4f} = {-math.log(p.prod()):.4f}"), 19, TXT, True, "center", 52)
    o += card(16, 364, 668, 68, "green", .2) + txt(24, 372, 652, t(f"四筆交叉熵加總：0.105 + 0.223 + 0.357 + 0.105 ≈ {ce.sum():.4f}，相同", f"Sum of the four cross-entropies: 0.105 + 0.223 + 0.357 + 0.105 ≈ {ce.sum():.4f}, the same"), 19, TXT, True, "center", 52)
    o += cardH(716, 0, 444, 480, "orange", t("概似", "Likelihood"))
    o += named(732, 48, 412, t("概似：假設模型是對的，觀察到這批資料的機率；越大越好，獨立的各筆機率相乘。", "Likelihood: the probability of observing this data if the model is right; larger is better, and independent samples multiply."), "term:" + t("概似", "Likelihood"), 16, TXT, 100)
    o += card(732, 168, 412, 120, "yellow", .16) + txt(740, 176, 396, t("最小化交叉熵，就是最大化概似", "Minimizing the cross-entropy is maximizing the likelihood"), 20, TXT, True, "center", 100)
    o += txt(732, 304, 412, t("ln 把乘積變成加總，負號把「越大越好」翻成「越小越好」：兩邊的 0.7905 與 0.7905 是同一個數。〔程式計算〕", "ln turns the product into a sum and the minus sign turns larger-is-better into smaller-is-better: 0.7905 and 0.790 are the same number. (computed)"), 16, MUT, False, "left", 120)
    o += txt(732, 420, 412, t("接下來的問題：類別不只兩個時，輸出怎麼做？", "Next: how do we build the output with more than two classes?"), 16, TXT, True, "center", 52)
    return o

# ------------------------------------------------ 6: softmax
def page6(lang):
    t = tr(lang); o = ""; v = np.array([2, 1, .1]); e = np.exp(v); s = e / e.sum(); d = np.array([1, 0, 0]) - s
    o += cardH(0, 0, 720, 480, "blue", t("v = (2, 1, 0.1) 經 softmax 與目標 y = (1, 0, 0)", "v = (2, 1, 0.1) through softmax with the target y = (1, 0, 0)"))
    o += card(16, 52, 688, 76, "violet", .22) + formula(16, 60, 688, r"\hat y_k=\frac{e^{v_k}}{\sum_{k'}e^{v_{k'}}}", 24, 60)
    groups = [(t("加權和 v", "weighted sum v"), v, "#6a9fd6", "#2f4f7a"), (t("輸出 ŷ", "output ŷ"), s, "#4f9a68", "#2c6140"), (t("誤差訊號 δ = y − ŷ", "error signal δ = y − ŷ"), d, "#d9822b", "#a8452b")]
    for gi, (name, vals, c1, c2) in enumerate(groups):
        x0 = 24 + gi * 228; o += card(x0, 142, 216, 240, "grey", .1) + txt(x0, 148, 216, name, 16, TXT, True, "center", 24)
        scale = 96 / 2.0; base = 304
        for ki, vv in enumerate(vals):
            bx = x0 + 20 + ki * 62; hh = abs(vv) * scale; top = base - hh if vv >= 0 else base
            o += box(bx, top, 46, max(hh, 3), c1 if vv >= 0 else "#c4572e", c2 if vv >= 0 else "#8e2f17", 180, shadow=False, r=6)
            o += txt(bx - 7, (top - 28) if vv >= 0 else (top + hh + 4), 60, f"{vv:.3f}" if gi else f"{vv:g}", 16, TXT, True, "center", 24)
            o += txt(bx - 4, 362, 54, f"k={ki+1}", 16, MUT, False, "center", 24)
    o += txt(16, 394, 688, t("ŷ 的三個數都在 0 與 1 之間，加起來恰為 1；δ 目標那一類為正、其餘為負。", "The three ŷ lie between 0 and 1 and sum to exactly 1; δ is positive for the target class and negative for the rest."), 16, TXT, False, "center", 52)
    o += cardH(736, 0, 424, 480, "orange", t("一步一步算", "Step by step"))
    st = [t("e² = 7.389、e¹ = 2.718、e⁰·¹ = 1.105，總和 11.212", "e² = 7.389, e¹ = 2.718, e⁰·¹ = 1.105, total 11.212"), t("ŷ = (0.659, 0.242, 0.099)", "ŷ = (0.659, 0.242, 0.099)"), t(f"目標 y = (1, 0, 0)：J = −ln 0.659 = {-math.log(s[0]):.3f}", f"Target y = (1, 0, 0): J = −ln 0.659 = {-math.log(s[0]):.3f}"), t("δₖ = yₖ − ŷₖ = (0.341, −0.242, −0.099)，與數值差分相同", "δₖ = yₖ − ŷₖ = (0.341, −0.242, −0.099), the same as numerical differences")]
    for i, s_ in enumerate(st):
        y0 = 52 + i * 70; o += card(752, y0, 392, 62, "grey", .12) + txt(760, y0 + 8, 376, f"{i+1}. " + s_, 16, TXT, False, "left", 48)
    o += named(752, 340, 392, t("softmax：用於多類別，輸出都在 0 與 1 之間，加總恰為 1。〔程式計算〕", "softmax: for several classes; every output lies between 0 and 1 and they sum to exactly 1. (computed)"), "term:softmax", 16, TXT, 100)
    return o

# ------------------------------------------------ 7: discriminative learning
def page7(lang):
    t = tr(lang); o = ""; x, tt, a, b, cut = K.lsfit(); w, bb, cut2 = K.logit_fit()
    o += cardH(0, 0, 560, 480, "red", t("全部七點：離群點 x = 40 把直線拉平", "All seven points: the outlier x = 40 flattens the line"))
    ax = Axes(60, 66, 480, 300, (-6, 44), (-0.2, 1.3)); ax.frame([0, 10, 20, 30, 40], [0, .5, 1], yfmt=lambda v: f"{v:g}")
    ax.line([(-6, a * -6 + b), (44, a * 44 + b)], C1, 4)
    for xv, tv in zip(x, tt): ax.dot(float(xv), float(tv), CB_ := ("#c4572e" if tv == 0 else "#4f86bf"), 8)
    ax.text(40, 1.0, "x = 40", "#f2ede4", "end", 16, True, -12, -12)
    o += ax.html() + txt(60, 398, 480, t("x（類 0：−3, −2, −1；類 1：1, 2, 3, 40）", "x (class 0: −3, −2, −1; class 1: 1, 2, 3, 40)"), 16, MUT, False, "center", 24)
    o += legend([(C1, t(f"最小平方直線 y ≈ {a:.4f}x + {b:.3f}", f"least-squares line y ≈ {a:.4f}x + {b:.3f}"))], 80, 436, 440)
    o += cardH(576, 0, 584, 480, "blue", t("放大看 −4 到 4：兩條邊界", "Zoom on −4 to 4: the two boundaries"))
    ax = Axes(640, 66, 500, 300, (-4, 4), (-0.15, 1.2)); ax.frame([-3, -2, -1, 0, 1, 2, 3], [0, .5, 1], yfmt=lambda v: f"{v:g}")
    xs = np.linspace(-4, 4, 100); ax.line([(float(u), float(a * u + b)) for u in xs], C1, 4); ax.line([(float(u), float(K.sig(bb + w * u))) for u in xs], C2, 4)
    ax.line([(-4, .5), (4, .5)], "#e0c85f", 2, "8 6", .8)
    ax.raw(f'<line x1="{ax.X(cut):.1f}" y1="{ax.Y(-.1)}" x2="{ax.X(cut):.1f}" y2="{ax.Y(1.1)}" stroke="{C1}" stroke-width="3" stroke-dasharray="8 6"/>'); ax.raw(f'<line x1="{ax.X(0):.1f}" y1="{ax.Y(-.1)}" x2="{ax.X(0):.1f}" y2="{ax.Y(1.1)}" stroke="{C2}" stroke-width="3" stroke-dasharray="8 6"/>')
    for xv, tv in zip(x[:-1], tt[:-1]): ax.dot(float(xv), float(tv), "#c4572e" if tv == 0 else "#4f86bf", 8)
    ax.raw(f'<circle cx="{ax.X(1):.1f}" cy="{ax.Y(1):.1f}" r="16" fill="none" stroke="#e0584f" stroke-width="4"/>')
    ax.text(cut, 1.12, f"{cut:.2f}", C1, "start", 17, True, 8, 0); ax.text(0, 1.12, "0", C2, "end", 17, True, -8, 0)
    o += ax.html()
    o += legend([(C1, t("平方誤差：切點 1.40，x = 1 分錯", "squared error: cut 1.40, x = 1 wrong"))], 600, 400, 520) + legend([(C2, t("交叉熵：切點約 0，七點全對", "cross-entropy: cut near 0, all seven right"))], 600, 428, 520)
    o += txt(590, 452, 560, t("對照同時換了輸出與代價；起點 0、η = 0.5、20000 輪。〔程式計算〕", "Output and cost both change; η = 0.5, 20000 rounds. (computed)"), 16, MUT, False, "center", 24)
    return o

# ------------------------------------------------ 8: Bayesian training
def page8(lang):
    t = tr(lang); o = ""; ns = (0, 1, 5, 20); cols = ["#7dbb8c", "#5fb8a8", "#e0c85f", "#d97757"]
    o += cardH(0, 0, 700, 480, "blue", t("權重 w 的分布：先驗與後驗", "The distribution of the weight w: prior and posterior"))
    ax = Axes(70, 66, 600, 330, (-5, 5), (0, 1.9)); ax.frame([-4, -2, 0, 1.5, 2, 4], [0, 0.5, 1, 1.5], yfmt=lambda v: f"{v:g}", xnames={1.5: "1.5"})
    xs = np.linspace(-5, 5, 300)
    for n, c in zip(ns, cols):
        sd, mu = K.posterior(n); ax.line([(float(u), float(math.exp(-(u - mu) ** 2 / (2 * sd * sd)) / (sd * math.sqrt(2 * math.pi)))) for u in xs], c, 4)
    o += ax.html() + txt(70, 426, 600, "w", 16, MUT, False, "center", 24)
    o += legend([(c, f"n = {n}") for n, c in zip(ns[:2], cols[:2])], 100, 80, 150) + legend([(c, f"n = {n}") for n, c in zip(ns[2:], cols[2:])], 260, 80, 150)
    o += cardH(716, 0, 444, 480, "orange", t("貝氏訓練", "Bayesian training"))
    o += named(732, 48, 412, t("貝氏訓練：先設權重的先驗分布 p(w)，資料進場後用貝氏定理得後驗 p(w|Y)，分類時對所有 w 平均。", "Bayesian training: put a prior p(w) on the weights, get the posterior p(w|Y) by Bayes' theorem once data arrive, and average over all w when classifying."), "term:" + t("貝氏訓練", "Bayesian training"), 16, TXT, 124)
    o += txt(732, 180, 412, t("例：先驗 w ~ N(0, 2²)；每筆資料都說 w ≈ 1.5，雜訊標準差 1。", "Example: prior w ~ N(0, 2²); every sample says w ≈ 1.5 with noise standard deviation 1."), 16, MUT, False, "left", 76)
    for xx, lab, w_ in ((732, "n", 60), (796, t("後驗標準差", "posterior sd"), 170), (974, t("後驗平均", "posterior mean"), 170)): o += txt(xx, 258, w_, lab, 16, MUT, True, "center", 24)
    for i, n in enumerate(ns):
        sd, mu = K.posterior(n); y0 = 286 + i * 40; o += card(732, y0, 412, 34, "grey", .12) + txt(732, y0 + 3, 60, str(n), 17, TXT, True, "center", 26) + txt(796, y0 + 3, 170, f"{sd:.3g}", 17, TXT, False, "center", 26) + txt(974, y0 + 3, 170, f"{mu:.3g}", 17, TXT, False, "center", 26)
    o += txt(732, 440, 412, t("後驗精度 = ¼ + n；代價是高維積分很貴。〔程式計算〕", "Precision = ¼ + n; high-dim. integrals are costly. (computed)"), 16, TXT, True, "center", 28)
    return o

# ------------------------------------------------ 9: the worse the start, the bigger the gap
def page9(lang):
    t = tr(lang); o = ""; ws = (-1, -2, -3, -4, -6, -8, -10); res = {w: (K.train("mse", w)[0], K.train("ce", w)[0]) for w in ws}
    o += cardH(0, 0, 700, 480, "blue", t("所需 epoch 數（對數座標）：起點 w 越負越久", "Epochs needed (log scale): the more negative the start w"))
    ax = Axes(80, 66, 590, 320, (0, 6), (0, 5)); ax.frame([], [0, 1, 2, 3, 4, 5], yfmt=lambda v: "1" if v == 0 else "10" + _sup(int(v)))
    pm = [(i, math.log10(res[w][0])) for i, w in enumerate(ws)]; pc = [(i, math.log10(res[w][1])) for i, w in enumerate(ws)]
    ax.line(pm, C1, 4); ax.line(pc, C2, 4)
    for (i, ym), (_, yc), w in zip(pm, pc, ws):
        ax.dot(i, ym, C1, 8); ax.dot(i, yc, C2, 8); ax.text(i, 0, str(w), "#b7b0a5", "middle", 16, False, 0, 28)
        ax.text(i, ym, f"{res[w][0]:,}", "#f2ede4", "middle", 16, True, 0, -14); ax.text(i, yc, str(res[w][1]), "#f2ede4", "middle", 16, True, 0, 26)
    o += ax.html() + txt(80, 424, 590, t("起點 w（b = 0，η = 0.5，同第 4 頁的資料）", "start w (b = 0, η = 0.5, the data of page 4)"), 16, MUT, False, "center", 24)
    o += legend([(C1, t("平方誤差", "squared error"))], 110, 80, 200) + legend([(C2, t("交叉熵", "cross-entropy"))], 110, 110, 200)
    o += cardH(716, 0, 444, 480, "orange", t("為什麼差距越拉越大", "Why the gap keeps growing"))
    o += txt(732, 52, 412, t("平方誤差的 δ 隨起點變錯而指數縮小，所需 epoch 數大致指數增加。", "The squared-error δ shrinks exponentially as the start gets worse, so the epoch count grows roughly exponentially."), 17, TXT, False, "left", 100)
    o += txt(732, 168, 412, t("交叉熵的 δ 保持在 1 附近，所需 epoch 數幾乎平躺。", "The cross-entropy δ stays near 1, so its epoch count is almost flat."), 17, TXT, False, "left", 76)
    o += card(732, 262, 412, 76, "yellow", .16) + txt(740, 270, 396, t(f"w = −10：平方誤差 {res[-10][0]:,} 個，交叉熵 {res[-10][1]} 個", f"w = −10: {res[-10][0]:,} epochs against {res[-10][1]}"), 18, TXT, True, "center", 60)
    o += txt(732, 352, 412, t("兩種代價最後都能分對；差別是起點很錯時的速度。〔程式計算〕", "Both costs end up right; the difference is the speed from a badly wrong start. (computed)"), 16, MUT, False, "left", 76)
    return o

# ------------------------------------------------ 10: exercises
def page10(lang):
    t = tr(lang); o = ""; y = float(K.sig(-4)); d1 = (1 - y) * y * (1 - y); d2 = 1 - y
    qs = [("blue", t("題 1", "Q1"), t("單一 sigmoid 輸出，目標 1，net = −4：ŷ = σ(−4) 是多少？", "One sigmoid output, target 1, net = −4: what is ŷ = σ(−4)?"), [r"\hat y=\frac{1}{1+e^{4}}", rf"\hat y={y:.4f}"], t(f"ŷ = {y:.4f}。", f"ŷ = {y:.4f}."), t("核對：net = −4 時輸出很小，離目標 1 很遠。", "Check: at net = −4 the output is small, far from the target 1.")),
          ("red", t("題 2", "Q2"), t("平方誤差與交叉熵的 δ 各是多少？交叉熵的是平方誤差的幾倍？", "What are the squared-error and cross-entropy δ, and how many times larger is the cross-entropy one?"), [rf"\delta_{{mse}}=0.9820\times0.0180\times0.9820={d1:.5f}", rf"\delta_{{ce}}=0.9820,\ \ 0.9820/{d1:.5f}={d2/d1:.1f}"], t(f"{d1:.5f}、{d2:.4f}，約 {d2/d1:.1f} 倍。", f"{d1:.5f} and {d2:.4f}, about {d2/d1:.1f} times."), t("核對：倍數 = 1/[ŷ(1 − ŷ)]，越極端越大。", "Check: the factor is 1/[ŷ(1 − ŷ)], larger when more extreme."))]
    for i, (role, name, q, tex, ans, chk) in enumerate(qs):
        x = i * 590; o += card(x, 0, 570, 470, role) + head(x, 0, 570, role, name, 36, 18)
        o += txt(x + 16, 52, 538, q, 19, TXT, True, "left", 84)
        o += chip(x + 16, 150, 130, "grey", t("解題示範", "Solution"), False, 30, 16)
        o += card(x + 16, 194, 538, 150, "grey", .1) + formula(x + 16, 206, 538, tex[0], 22, 56) + formula(x + 16, 270, 538, tex[1], 22, 56)
        o += chip(x + 16, 364, 538, role, ans, True, 42, 20)
        o += txt(x + 16, 420, 538, chk, 17, MUT, False, "left", 48)
    return o

PAGES_DESIGN = [(f"p{i}", globals()[f"page{i}"]) for i in range(1, 11)]
if __name__ == "__main__":
    out = sys.argv[1]; only = sys.argv[2:]
    for fid, fn in PAGES_DESIGN:
        if only and fid not in only: continue
        print(fid); build("kp09_" + fid, fn, W, H, out)
