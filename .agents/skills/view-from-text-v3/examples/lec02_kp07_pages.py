import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lec01_kit import *

KP = "KP07"; N = 8
RES = (1, 1, 1, 10); DELTA = 1
mse = lambda e: e * e; mae = abs
hub = lambda e, d=DELTA: 0.5 * e * e if abs(e) <= d else d * (abs(e) - 0.5 * d)
L = {"MSE": [mse(e) for e in RES], "MAE": [mae(e) for e in RES], "Huber": [hub(e) for e in RES]}
assert L["MSE"] == [1, 1, 1, 100] and sum(L["MSE"]) == 103 and sum(L["MAE"]) == 13 and L["Huber"] == [0.5, 0.5, 0.5, 9.5] and sum(L["Huber"]) == 11
SH = {k: v[-1] / sum(v) for k, v in L.items()}
assert [round(SH[k] * 100) for k in ("MSE", "MAE", "Huber")] == [97, 77, 86]
_E = [math.exp(v) for v in (2, 1, 0)]; PX = [e / sum(_E) for e in _E]; P = PX; PR = (0.665, 0.245, 0.090); assert [round(v, 3) for v in PX] == list(PR); CE = [-math.log(p) for p in P]
assert [round(c, 3) for c in CE] == [0.408, 1.408, 2.408]
Y = (0.7, 0.2, 0.1); KL = sum(y * math.log(y / p) for y, p in zip(Y, P)); assert round(KL, 4) == 0.0058
KL1 = math.log(1 / PX[0]); assert round(KL1, 3) == 0.408
YF = (1.5, 0.3, -0.5); HG = [max(0, 1 - v) for v in YF]; assert HG == [0, 0.7, 1.5]
fm = lambda v: ("%g" % round(v, 4))

def make(lang):
    zh = lang == "zh"
    T = lambda a, b_: a if zh else b_
    pages = []
    new = lambda n, title: new_page(n, N, KP, title)

    def axes(g, x0, y0, w, h, xr, yr):
        ux = lambda a: x0 + (a - xr[0]) / (xr[1] - xr[0]) * w; uy = lambda v: y0 + h - (v - yr[0]) / (yr[1] - yr[0]) * h
        g.rect(x0, y0, w, h, fill=CARD, stroke=GRAY, sw=3, radius=12, z=1)
        if yr[0] < 0 < yr[1]: g.line([(x0, uy(0)), (x0 + w, uy(0))], GRAY, sw=2, z=2)
        if xr[0] < 0 < xr[1]: g.line([(ux(0), y0), (ux(0), y0 + h)], GRAY, sw=2, z=2)
        return ux, uy
    def curve(g, ux, uy, fn, col, xr, yr, sw=5, n=80):
        pts = []
        for i in range(n + 1):
            a = xr[0] + (xr[1] - xr[0]) * i / n; v = min(max(fn(a), yr[0]), yr[1]); pts.append((ux(a), uy(v)))
        g.line(pts, col, sw=sw, z=5)

    # ============ 1: loss = one number ============
    g = new(1, T("KP07 損失函數把「錯多少」變成一個數", "KP07 A loss turns “how wrong” into one number"))
    g.beat = 1
    a = g.mark(); g.rect(60, 190, 260, 130, fill=BLUEL, stroke=BLUE, sw=3, radius=16, z=3); g.text(60, 190, 260, 130, T("網路輸出", "network output"), size=26, color=BLUE, bold=True, align="center", vcenter=True, z=8)
    b2 = g.mark(); g.rect(60, 390, 260, 130, fill=GRNL, stroke=GRN, sw=3, radius=16, z=3); g.text(60, 390, 260, 130, T("正確答案", "right answer"), size=26, color=GRN, bold=True, align="center", vcenter=True, z=8)
    c = g.mark(); g.rect(480, 280, 300, 150, fill=ORGL, stroke=ORG, sw=4, radius=20, z=3); g.text(480, 290, 300, 60, T("損失函數", "loss function"), size=28, color=ORG, bold=True, align="center", z=8); g.text(480, 350, 300, 70, r"$\ell$", size=40, color=INK, align="center", vcenter=True, z=8)
    d = g.mark(); g.rect(940, 280, 280, 150, fill=REDL, stroke=RED, sw=3, radius=16, z=3); g.text(940, 280, 280, 150, T("一個數", "one number"), size=30, color=RED, bold=True, align="center", vcenter=True, z=8)
    g.link(a, c, 4, 12, GRAY, 5); g.link(b2, c, 4, 12, GRAY, 5); g.link(c, d, 4, 12, GRAY, 5)
    g.beat = 2
    g.text(60, 560, 1160, 50, T("有了這個數，才知道往哪個方向調權重", "with this number we know which way to adjust the weights"), size=26, color=INK, bold=True, align="center", z=8)
    banner(g, T("回歸量「距離」，分類量「預測機率」與答案差多少", "regression measures distance; classification measures probability against the answer"))
    pages.append((g, T("損失是一個數", "A loss is one number"), {
        0: T("網路輸出一個結果，和正確答案比，損失函數把差多少變成一個數。", "The network gives an output; compared with the right answer, the loss function turns how far off into one number."),
        1: T("有了這個數，才知道往哪個方向調權重。這個單元看回歸、分類與 hinge 三類損失。", "With this number we know which way to adjust the weights. This unit looks at regression, classification and hinge losses."),
        2: T("", ""), 3: T("", "")}))

    # ============ 2: regression loss curves ============
    g = new(2, T("回歸:MSE、MAE、Huber 的形狀", "Regression: the shapes of MSE, MAE and Huber"))
    xr, yr = (-2.2, 2.2), (0, 5)
    ux, uy = axes(g, 60, 130, 700, 440, xr, yr)
    g.beat = 1; curve(g, ux, uy, mse, RED, xr, yr); g.text(ux(1.5) + 10, uy(2.25) - 20, 90, 34, "MSE", size=20, color=RED, bold=True, z=8)
    g.beat = 2; curve(g, ux, uy, mae, BLUE, xr, yr); g.text(ux(1.9), uy(1.9) + 14, 90, 34, "MAE", size=20, color=BLUE, bold=True, z=8)
    g.beat = 3; curve(g, ux, uy, hub, GRN, xr, yr, sw=6); g.text(ux(-2.1), uy(1.5) + 6, 120, 34, "Huber", size=20, color=GRN, bold=True, z=8)
    g.text(60, 574, 700, 40, T("橫軸:殘差 $e$ 縱軸:損失", "x: residual $e$ · y: loss"), size=18, color=MUTED, bold=True, align="center", z=8)
    cards = [(RED, REDL, "MSE", r"$\sum e^2$", T("處處可導;平方放大離群值", "differentiable; squares magnify outliers")), (BLUE, BLUEL, "MAE", r"$\sum|e|$", T("對離群值穩健;零點不可導", "robust to outliers; not differentiable at 0")), (GRN, GRNL, "Huber", T("$|e|\\le\\delta$ 二次，否則線性", "quadratic for $|e|\\le\\delta$, else linear"), T("兩者折衷，要調 $\\delta$", "a compromise; tune $\\delta$"))]
    for i, (col, light, nm, fml, desc) in enumerate(cards):
        g.beat = i + 1
        y = 130 + i * 150
        g.rect(800, y, 430, 136, fill=light, stroke=col, sw=3, radius=14, z=3)
        g.text(800, y + 6, 430, 38, nm, size=24, color=col, bold=True, align="center", z=8); g.text(810, y + 46, 410, 40, fml, size=20, color=INK, bold=True, align="center", z=8); g.text(810, y + 90, 410, 40, desc, size=17, color=MUTED, bold=True, align="center", z=8)
    banner(g, T("殘差越大，MSE 罰得越兇;MAE 只按距離;Huber 近處像 MSE、遠處像 MAE", "large residuals: MSE punishes hardest, MAE by distance, Huber is MSE near and MAE far"))
    pages.append((g, T("回歸損失的形狀", "Shapes of regression losses"), {
        0: T("MSE 是殘差的平方，拋物線，處處可導，離得遠罰得很兇。", "MSE is the squared residual, a parabola, differentiable everywhere and very harsh for far points."),
        1: T("MAE 是殘差的絕對值，V 形，只按距離罰，對離群值穩健，但零點不可導。", "MAE is the absolute residual, a V, penalizing by distance only, robust to outliers but not differentiable at 0."),
        2: T("Huber 在近處是二次、遠處是線性，是兩者折衷，要調 delta。", "Huber is quadratic near zero and linear far away, a compromise; delta must be tuned."),
        3: T("", "")}))

    # ============ 3: outlier numbers ============
    g = new(3, T("殘差 $(1,1,1,10)$:離群點占多少", "Residuals $(1,1,1,10)$: how much is the outlier"))
    g.beat = 1
    g.text(50, 112, 1180, 40, T("四筆資料的殘差 $(1,1,1,10)$，最後一筆是離群值;Huber 取 $\\delta=1$", "residuals of four samples $(1,1,1,10)$; the last is an outlier; Huber uses $\\delta=1$"), size=22, color=MUTED, bold=True, align="center", z=8)
    base = 560; sc = 3.2
    for i, (nm, col, light) in enumerate((("MSE", RED, REDL), ("MAE", BLUE, BLUEL), ("Huber", GRN, GRNL))):
        g.beat = i + 1
        x = 120 + i * 380
        vals = L[nm]; tot = sum(vals); inl = tot - vals[-1]
        g.rect(x, base - inl * sc, 200, inl * sc, fill=light, stroke=col, sw=3, radius=6, z=3)
        g.rect(x, base - tot * sc, 200, vals[-1] * sc, fill=col, stroke=col, sw=3, radius=6, z=3, fillOpacity=0.35)
        g.text(x - 30, base - tot * sc - 44, 260, 38, T(f"總和 ${fm(tot)}$", f"total ${fm(tot)}$"), size=22, color=col, bold=True, align="center", z=8)
        g.text(x - 30, base + 6, 260, 38, nm, size=24, color=col, bold=True, align="center", z=8)
        g.text(x + 215, base - tot * sc + 4, 150, 80, T(f"離群點\n${round(SH[nm] * 100)}\\%$", f"outlier\n${round(SH[nm] * 100)}\\%$"), size=22, color=col, bold=True, z=8)
    g.line([(80, base), (1200, base)], INK, sw=3, z=3)
    g.beat = 4
    banner(g, T("MSE 幾乎只在乎那一個點;MAE 最不受影響;Huber 在中間", "MSE nearly cares about that one point; MAE least; Huber is in between"))
    pages.append((g, T("離群值的影響", "The outlier's effect"), {
        0: T("殘差一、一、一、十，最後一筆是離群值。", "Residuals 1, 1, 1, 10; the last is an outlier."),
        1: T("MSE 的值是 1、1、1、100，總和 103，離群點占 97%。", "MSE values 1, 1, 1, 100, total 103; the outlier is 97 percent."),
        2: T("MAE 的值是 1、1、1、10，總和 13，離群點占 77%。", "MAE values 1, 1, 1, 10, total 13; the outlier is 77 percent."),
        3: T("Huber 的最後一筆是 delta 乘上殘差減二分之一 delta，等於 9.5，總和 11，占 86%。這就是 MSE 對離群值敏感的具體意思。", "Huber's last value is delta times residual minus half delta, 9.5; total 11, 86 percent. This is what MSE being sensitive to outliers means concretely.")}))

    # ============ 4: cross-entropy ============
    g = new(4, T("分類:給正確類別的機率，取負對數", "Cross-entropy: minus log of the right probability"))
    xr, yr = (0, 1), (0, 3.2)
    ux, uy = axes(g, 60, 130, 640, 440, xr, yr)
    g.beat = 1
    pts = [(ux(0.04 + (1 - 0.04) * i / 80), uy(min(-math.log(0.04 + (1 - 0.04) * i / 80), 3.2))) for i in range(81)]
    g.line(pts, RED, sw=5, z=5)
    g.text(60, 574, 640, 40, T("橫軸:給正確類別的機率 $p$ 縱軸:$-\\ln p$", "x: probability $p$ of the right class · y: $-\\ln p$"), size=18, color=MUTED, bold=True, align="center", z=8)
    g.beat = 2
    cols = (GRN, ORG, RED)
    for i, (p, c_, col) in enumerate(zip(P, CE, cols)):
        g.ellipse(ux(p), uy(c_), 14, fill=col, stroke="#ffffff", sw=3, z=7)
        g.line([(ux(p), uy(c_)), (ux(p), uy(0))], col, sw=3, dash=True, z=4)
    g.beat = 3
    for i, (p, c_, col) in enumerate(zip(P, CE, cols)):
        y = 140 + i * 130
        g.rect(740, y, 490, 112, fill=CARD, stroke=col, sw=3, radius=14, z=3)
        g.text(750, y + 6, 470, 44, T(f"正確是第 {i + 1} 類", f"right class is {i + 1}"), size=22, color=col, bold=True, align="center", z=8)
        g.text(750, y + 54, 470, 50, rf"$p={p:.3f}$,  $-\ln p={c_:.3f}$", size=26, color=INK, bold=True, align="center", vcenter=True, z=8)
    banner(g, T("預測給正確答案的機率越小，損失越大;輸出層要接 softmax 或 sigmoid", "the smaller the probability given to the right answer, the larger the loss; use softmax or sigmoid"))
    pages.append((g, T("交叉熵", "Cross-entropy"), {
        0: T("交叉熵是給正確類別的機率取負對數：機率越小，損失越大，趨近 0 時損失暴增。", "Cross-entropy is minus the log of the probability given to the right class: the smaller the probability, the larger the loss, blowing up near 0."),
        1: T("沿用 KP06 的 softmax 輸出，0.665、0.245、0.090，標在曲線上。", "Reusing KP06's softmax output 0.665, 0.245, 0.090, mark them on the curve."),
        2: T("若正確是第一類，損失 0.408；第二類 1.408；第三類 2.408。交叉熵要求輸出像機率，所以輸出層要接 softmax 或 sigmoid。", "If the right class is the first, the loss is 0.408; the second 1.408; the third 2.408. Cross-entropy needs probability-like outputs, hence softmax or sigmoid in the output layer."),
        3: T("", "")}))

    # ============ 5: KL ============
    g = new(5, T("目標本身是機率:KL 散度", "When the target is itself a distribution: KL divergence"))
    g.beat = 1
    bw = 90; base = 430; sc = 330
    for i, (y, p) in enumerate(zip(Y, P)):
        x = 90 + i * 200
        g.rect(x, base - y * sc, bw, y * sc, fill=GRNL, stroke=GRN, sw=3, radius=6, z=3); g.text(x - 10, base - y * sc - 46, bw + 20, 44, f"${y}$", size=20, color=GRN, bold=True, align="center", z=8)
        g.rect(x + bw + 6, base - p * sc, bw, p * sc, fill=BLUEL, stroke=BLUE, sw=3, radius=6, z=3); g.text(x + bw - 4, base - p * sc - 46, bw + 20, 44, f"${PR[i]}$", size=20, color=BLUE, bold=True, align="center", z=8)
        g.text(x, base + 6, 2 * bw + 6, 34, T(f"第 {i + 1} 類", f"class {i + 1}"), size=18, color=MUTED, bold=True, align="center", z=8)
    g.line([(70, base), (700, base)], INK, sw=3, z=3)
    g.text(70, 120, 640, 36, T("綠:軟標籤目標 $\\mathbf{y}=(0.7,0.2,0.1)$　藍:預測 $\\mathbf{p}$", "green: soft target $\\mathbf{y}=(0.7,0.2,0.1)$ · blue: prediction $\\mathbf{p}$"), size=18, color=MUTED, bold=True, z=8)
    g.beat = 2
    g.rect(760, 150, 470, 190, fill=ORGL, stroke=ORG, sw=3, radius=16, z=3)
    g.text(770, 158, 450, 40, T("KL 散度", "KL divergence"), size=24, color=ORG, bold=True, align="center", z=8)
    g.text(770, 204, 450, 120, r"$\sum_l y_l\ln\frac{y_l}{p_l}=0.0058$", size=30, color=INK, align="center", vcenter=True, z=8)
    g.beat = 3
    g.rect(760, 370, 470, 180, fill=REDL, stroke=RED, sw=3, radius=16, z=3)
    g.text(770, 378, 450, 40, T("目標是 one-hot $(1,0,0)$", "one-hot target $(1,0,0)$"), size=22, color=RED, bold=True, align="center", z=8)
    g.text(770, 424, 450, 110, T("只剩 $\\ln\\frac{1}{0.665}=0.408$\n與交叉熵相同", "only $\\ln\\frac{1}{0.665}=0.408$\nthe same as cross-entropy"), size=24, color=INK, bold=True, align="center", z=8)
    banner(g, T("兩個分布很像，KL 就很接近 0;用在軟標籤(例如知識蒸餾)", "similar distributions give a KL near 0; used with soft labels (e.g. distillation)"))
    pages.append((g, T("KL 散度", "KL divergence"), {
        0: T("有時目標本身就是機率分布，叫軟標籤，例如知識蒸餾。目標 0.7、0.2、0.1，預測 0.665、0.245、0.090，兩個分布很像。", "Sometimes the target is itself a probability distribution, a soft label, as in distillation. Target 0.7, 0.2, 0.1 and prediction 0.665, 0.245, 0.090 are very similar."),
        1: T("KL 散度算出來是 0.0058，很接近 0。", "The KL divergence comes out at 0.0058, very close to 0."),
        2: T("若目標是 one-hot，KL 只剩負對數 0.665，等於 0.408，和交叉熵相同。", "If the target is one-hot, KL reduces to minus log 0.665, which is 0.408, the same as cross-entropy."),
        3: T("", "")}))

    # ============ 6: hinge ============
    g = new(6, T("Hinge:分對了但離邊界太近，也要罰", "Hinge: even “right but too close” is punished"))
    xr, yr = (-1.5, 2.5), (-0.3, 2.8)
    ux, uy = axes(g, 60, 130, 700, 440, xr, yr)
    g.beat = 1
    curve(g, ux, uy, lambda v: max(0, 1 - v), RED, xr, yr)
    g.line([(ux(1), uy(-0.3)), (ux(1), uy(2.8))], GRAY, sw=3, dash=True, z=3); g.text(ux(1) + 8, uy(2.7), 160, 34, T("邊距 $m=1$", "margin $m=1$"), size=18, color=MUTED, bold=True, z=8)
    g.text(60, 574, 700, 40, T("橫軸:$y\\,f_o$ 縱軸:損失 $\\max(0,\\,1-yf_o)$", "x: $y\\,f_o$ · y: loss $\\max(0,\\,1-yf_o)$"), size=18, color=MUTED, bold=True, align="center", z=8)
    g.beat = 2
    cols = (GRN, ORG, RED)
    for v, h_, col in zip(YF, HG, cols): g.ellipse(ux(v), uy(h_), 14, fill=col, stroke="#ffffff", sw=3, z=7)
    g.beat = 3
    notes = (T("分對，離邊界夠遠", "right and far enough"), T("分對了($0.3>0$)，卻離邊界不到 $m$", "right ($0.3>0$) but within $m$ of the boundary"), T("分錯", "wrong"))
    for i, (v, h_, col, nt) in enumerate(zip(YF, HG, cols, notes)):
        y = 140 + i * 140
        g.rect(800, y, 430, 124, fill=CARD, stroke=col, sw=3, radius=14, z=3)
        g.text(810, y + 6, 410, 56, nt, size=20, color=col, bold=True, align="center", z=8)
        g.text(810, y + 68, 410, 48, rf"$yf_o={v}$,  " + T("損失", "loss") + f" ${fm(h_)}$", size=24, color=INK, bold=True, align="center", z=8)
    banner(g, T("和 KP02 只問對錯的 0-1 損失不同:hinge 是連續的", "unlike KP02's right-or-wrong 0-1 loss, hinge is continuous"))
    pages.append((g, T("Hinge 損失", "Hinge loss"), {
        0: T("Hinge 用在二分類，標籤正負 1，損失是 0 與 m 減 y 乘 f 取大的那個，這裡 m 等於 1。", "Hinge is for binary classification with labels plus or minus 1; the loss is the larger of 0 and m minus y times f, here with m equal to 1."),
        1: T("三個點：y f 等於 1.5，損失 0；0.3，損失 0.7；負 0.5，損失 1.5。", "Three points: y f equal to 1.5 gives loss 0; 0.3 gives 0.7; minus 0.5 gives 1.5."),
        2: T("第二個點分對了，因為 0.3 大於 0，卻因離邊界不到 m，仍被罰 0.7。這和 KP02 只問對錯的 0-1 損失不同，是連續的。", "The second point is classified right since 0.3 is above 0, yet it is within m of the boundary and still punished with 0.7. Unlike KP02's right-or-wrong 0-1 loss, this is continuous."),
        3: T("", "")}))

    # ============ 7: different rulers ============
    g = new(7, T("不同損失，是不同的尺", "Different losses are different rulers"))
    g.beat = 1
    a = g.mark(); g.rect(60, 150, 520, 200, fill=REDL, stroke=RED, sw=3, radius=16, z=3)
    g.text(60, 158, 520, 44, "MSE", size=26, color=RED, bold=True, align="center", z=8); g.text(70, 210, 500, 130, "$103$", size=60, color=INK, align="center", vcenter=True, z=8)
    b2 = g.mark(); g.rect(700, 150, 520, 200, fill=BLUEL, stroke=BLUE, sw=3, radius=16, z=3)
    g.text(700, 158, 520, 44, "MAE", size=26, color=BLUE, bold=True, align="center", z=8); g.text(710, 210, 500, 130, "$13$", size=60, color=INK, align="center", vcenter=True, z=8)
    g.text(560, 210, 160, 80, "≠" if False else "?", size=50, color=GOLD, bold=True, align="center", vcenter=True, z=8)
    g.beat = 2
    g.rect(190, 390, 900, 200, fill="#fef3c7", stroke=GOLD, sw=3, radius=16, z=3)
    g.text(210, 400, 860, 60, T("同一組殘差，103 與 13 沒有誰比較大的意義", "for the same residuals, 103 versus 13 does not mean one is larger"), size=26, color=INK, bold=True, align="center", z=8)
    g.text(210, 480, 860, 100, T("它們是不同的尺。\n損失只在訓練資料上量到，太小可能只是把訓練資料背起來(KP04 的泛化問題)。", "They are different rulers.\nA loss is measured on training data only; a tiny loss may just mean memorizing it (KP04's generalization problem)."), size=22, color=MUTED, bold=True, align="center", z=8)
    banner(g, T("損失越小不一定越好;不同損失的數值不能直接比大小", "a smaller loss is not always better; values of different losses cannot be compared"))
    pages.append((g, T("兩個迷思", "Two myths"), {
        0: T("同一組殘差，MSE 是 103，MAE 是 13。", "For the same residuals, MSE is 103 and MAE is 13."),
        1: T("這兩個數沒有誰比較大的意義，它們是不同的尺。另外，損失越小不一定越好，因為損失只在訓練資料上量到，太小可能只是背起來。", "These two numbers have no meaning of one being larger; they are different rulers. Also, a smaller loss is not always better: it is measured on training data only and may just mean memorization."),
        2: T("", ""), 3: T("", "")}))

    # ============ 8: summary ============
    g = new(8, T("損失決定了「怎麼走」", "The loss decides which way to go"))
    g.beat = 1
    rows = [(RED, REDL, T("回歸", "regression"), "MSE · MAE · Huber"), (BLUE, BLUEL, T("分類", "classification"), T("交叉熵(輸出層配 softmax 或 sigmoid)· KL(軟標籤)", "cross-entropy (softmax or sigmoid output) · KL (soft labels)")), (GRN, GRNL, T("邊距分類", "margin classification"), "hinge")]
    for i, (col, light, nm, desc) in enumerate(rows):
        y = 130 + i * 110
        g.rect(50, y, 1180, 92, fill=light, stroke=col, sw=3, radius=14, z=3)
        g.text(60, y, 260, 92, nm, size=26, color=col, bold=True, align="center", vcenter=True, z=8); g.text(330, y, 880, 92, desc, size=24, color=INK, bold=True, vcenter=True, z=8)
    g.beat = 2
    g.rect(50, 470, 1000, 140, fill="#fef3c7", stroke=GOLD, sw=3, radius=16, z=3)
    g.text(70, 480, 960, 126, T("Perceptron 的損失只數「錯幾個」，是離散的，不能用梯度學。如果改成連續的平方誤差，就能一步一步往下走。這個想法就是 ADALINE。", "The Perceptron's loss only counts errors, so it is discrete and cannot be learned by gradients. With a continuous squared error we can walk downhill step by step. That idea is ADALINE."), size=19 if not zh else 22, color=INK, bold=True, align="center", z=8)
    mascot(g, 1100, 440, 0.5, "proud")
    banner(g, T("下一個單元:ADALINE 與邏輯回歸", "next unit: ADALINE and logistic regression"))
    pages.append((g, T("重點與預告", "Summary and preview"), {
        0: T("回歸用 MSE、MAE、Huber；分類用交叉熵，輸出層配 softmax 或 sigmoid，軟標籤用 KL；邊距分類用 hinge。", "Regression uses MSE, MAE, Huber; classification uses cross-entropy with a softmax or sigmoid output, KL for soft labels; margin classification uses hinge."),
        1: T("Perceptron 的損失只數錯幾個，是離散的；改成連續的平方誤差，就能一步一步往下走，這就是 ADALINE。", "The Perceptron's loss only counts errors and is discrete; a continuous squared error lets us walk downhill step by step. That is ADALINE."),
        2: T("", ""), 3: T("", "")}))
    return pages
