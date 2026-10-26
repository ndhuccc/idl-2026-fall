"""KP18 (probabilistic neural network) -- twelfth sample of view-from-text-v2. Plan: kp18_plan.json. Every number is recomputed here with kp18_nums.py
(same settings as the lecture; verified against it)."""
import os, sys
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools")); sys.path.insert(0, HERE)
from figkit.parts import T, NFig, header_chip, panel, chip, takeaway, tint, solid, INK
import kp18_nums as N
from kp19_specs import cbox, circ
from kp20_specs import card


def tr(zh, en): return {"zh": zh, "en": en}


def hu(x, nd):
    return str(Decimal(repr(round(float(x), 9))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def parzen():
    pts = [2.2, 3.0, 3.6, 7.4, 8.2]; h = 0.7; xs = np.linspace(-0.5, 10.7, 160); g = lambda x, c: np.exp(-(x - c) ** 2 / (2 * h * h)) / (h * np.sqrt(2 * np.pi)) / len(pts)
    tot = sum(g(xs, c) for c in pts)
    spec = {
        "id": "lec05-kp18-parzen",
        "alt": tr("Parzen 估計（一維、五個示意訓練點，核寬 h=0.7）：每個訓練點（圓點）上放一座小高斯（藍），把五座小高斯加起來（每座乘 1/5，即平均）得到估計的密度（紅色粗線），兩個訓練點密集處各有一個峰。",
                  "Parzen estimate (one dimension, five illustrative training points, kernel width h=0.7): a small Gaussian (blue) sits on every training point (dot); adding the five small Gaussians (each times 1/5, an average) gives the estimated density (thick red line), with one peak where each group of training points is dense."),
        "plot": {"x": [-0.5, 10.7], "y": [0, 0.3], "x_ticks": [], "y_ticks": [], "x_label": tr("x（示意）", "x (sketch)"), "y_label": tr("密度（示意）", "density (sketch)"), "w": 430, "h": 300, "legend_pos": "right"},
        "series": [{"kind": "line", "points": [(float(x), float(g(x, c))) for x in xs], "role": "blue", "width": 2, **({"name": tr("每點一座核", "kernel per point")} if k == 0 else {})} for k, c in enumerate(pts)]
                  + [{"kind": "line", "points": [(float(x), float(t)) for x, t in zip(xs, tot)], "role": "red", "width": 5, "name": tr("平均＝估計的密度", "average = estimate")},
                     {"kind": "dots", "points": [(c, 0.004) for c in pts], "role": "blue", "d": 14, "name": tr("訓練點", "training points")}],
        "side": [{"title": tr("Parzen 估計", "Parzen estimate"), "role": "blue", "name": tr("term:Parzen 估計", "term:Parzen estimate"), "h": 1.3, "text": tr("在每個訓練點放一個核函數（這裡是高斯），\n平均起來估計各類的機率密度 $\\hat{p}(x|\\omega)$。", "put a kernel (here a Gaussian) on every training point and\naverage them to estimate each class's density $\\hat{p}(x|\\omega)$.")},
                 {"title": tr("核寬 h", "Kernel width h"), "role": "orange", "name": tr("term:核寬", "term:Kernel width"), "h": 1.0, "text": tr("核函數的寬度：越大越平滑，\n越小越貼近個別訓練點。", "the width of the kernel: larger is smoother,\nsmaller hugs the individual training points.")},
                 {"title": tr("貝氏決策", "Bayes decision"), "role": "green", "h": 1.3, "text": tr("選 $P(\\omega_i)\\hat{p}(x|\\omega_i)$ 最大的類；\n核函數的加總能平行算，所以一點一個神經元。", "choose the class with the largest $P(\\omega_i)\\hat{p}(x|\\omega_i)$;\nthe kernel sums run in parallel: one neuron per point.")}],
        "takeaway": tr("Parzen 估計＝一堆核的平均：一個訓練點配一個神經元，幾乎不用訓練。", "A Parzen estimate is an average of kernels: one neuron per training point, almost no training."),
    }
    return "plot_xy", spec


def normal():
    f = NFig("lec05-kp18-circle", 1160, 450, T(tr("單位範數正規化：把每個特徵向量除以自己的長度，使長度變成 1。平方距離展開 ‖x−x_i‖²=‖x‖²+‖x_i‖²−2x_iᵀx，長度都是 1 時等於 2−2x_iᵀx；例 x=(0.6,0.8)、x_i=(1,0)：內積 0.6，距離平方 2−1.2=0.8，直接算 0.16+0.64=0.8。代入高斯並丟掉各類共同的常數後 g(ω_m)=P(ω_m)/N_m·Σexp((x_iᵀx−1)/h²)。驗算（h=0.8）：單位向量 (0.6,0.8)：距離平方 0.8，距離形式 0.5353、內積形式 0.5353；非單位向量 (1.2,1.6)：2.6，0.1312 對 1.3668，差很多，所以正規化是必要前提。",
                                              "Unit-norm normalization divides every feature vector by its own length so that the length becomes 1. The squared distance expands to ‖x−x_i‖²=‖x‖²+‖x_i‖²−2x_iᵀx, which is 2−2x_iᵀx when both lengths are 1; example x=(0.6,0.8), x_i=(1,0): the inner product is 0.6 and the squared distance 2−1.2=0.8, directly 0.16+0.64=0.8. Substituting into the Gaussian and dropping the constant common to all classes gives g(ω_m)=P(ω_m)/N_m·Σexp((x_iᵀx−1)/h²). Check (h=0.8): the unit vector (0.6,0.8): squared distance 0.8, distance form 0.5353, inner-product form 0.5353; the non-unit vector (1.2,1.6): 2.6, 0.1312 against 1.3668, very different, so normalization is a required precondition.")))
    header_chip(f, 0, 0, 360, 34, "blue", T(tr("單位圓上的兩個向量", "Two vectors on the unit circle")), size=17); panel(f, 0, 38, 360, 320, "blue")
    cx, cy, R = 150, 205, 110; circ(f, cx, cy, R, "MUT", 2)
    A = (cx + 0.6 * R, cy - 0.8 * R); B = (cx + R, cy)
    f.line(cx, cy, A[0], A[1], solid("orange"), 4); f.line(cx, cy, B[0], B[1], solid("blue"), 4); f.ellipse(cx, cy, 10, "TXT")
    for t in np.linspace(0, 1, 9)[:-1:2]: f.line(A[0] + (B[0] - A[0]) * t, A[1] + (B[1] - A[1]) * t, A[0] + (B[0] - A[0]) * (t + 0.125), A[1] + (B[1] - A[1]) * (t + 0.125), solid("red"), 3)
    f.ellipse(A[0], A[1], 14, solid("orange")); f.ellipse(B[0], B[1], 14, solid("blue"))
    f.text(A[0] - 18, A[1] - 16, "$x$", "TXT", 18, True, w=40, math=True); f.text(B[0] + 30, B[1] + 4, "$x_i$", "TXT", 18, True, w=50, math=True)
    f.text(B[0] + 20, (A[1] + B[1]) / 2 - 8, T(tr("弦", "chord")), "TXT", 16, True, "left", w=50)
    f.text(180, 338, T(tr("內積 xᵢᵀx = 0.6；弦長² = 2 − 2×0.6 = 0.8", "inner product xᵢᵀx = 0.6; chord² = 2 − 2×0.6 = 0.8")), "TXT", 16, w=340)
    header_chip(f, 380, 0, 400, 34, "orange", T(tr("從距離到判別式", "From distance to discriminant")), size=17)
    card(f, 380, 42, 400, 64, "yellow", T(tr("單位範數正規化：把每個特徵向量\n除以自己的長度，使長度變成 1。", "unit-norm normalization: divide every feature\nvector by its own length so it becomes 1.")), name=T(tr("term:單位範數正規化", "term:Unit-norm normalization")), math=False)
    card(f, 380, 112, 400, 82, "blue", T(tr("$\\|x-x_i\\|^2=\\|x\\|^2+\\|x_i\\|^2-2x_i^Tx$\n長度都是 1 時 $=2-2x_i^Tx$", "$\\|x-x_i\\|^2=\\|x\\|^2+\\|x_i\\|^2-2x_i^Tx$\nwith both lengths 1 it is $2-2x_i^Tx$")))
    card(f, 380, 202, 400, 56, "orange", T(tr("代入高斯，丟掉各類共同的常數：", "substitute into the Gaussian,\ndrop the common constant:")), math=False)
    f.rect(380, 266, 400, 92, fill=tint("green", 0.16), stroke=tint("green", 0.6), sw=2, radius=12); f.text(580, 312, "$g(\\omega_m)=(P(\\omega_m)/N_m)\\sum_i$\n$\\exp((x_i^Tx-1)/h^2)$", "TXT", 19, w=380, math=True)
    header_chip(f, 800, 0, 360, 34, "green", T(tr("驗算（$h=0.8$）", "Check ($h=0.8$)")).replace("$", ""), size=17); panel(f, 800, 38, 360, 220, "green")
    hd = [(tr("測試向量", "test vector"), 100), ("‖x−xᵢ‖²", 70), (tr("距離形式", "distance"), 80), (tr("內積形式", "inner"), 80)]
    x = 812
    for nm, w in hd: f.text(x + w / 2, 60, T(nm), "TXT", 16, True, w=w); x += w
    rows = [("(0.6, 0.8)", "0.8", "0.5353", "0.5353", "green"), ("(1.2, 1.6)", "2.6", "0.1312", "1.3668", "red")]
    for i, r in enumerate(rows):
        y = 80 + i * 62; f.rect(808, y, 344, 54, fill=tint(r[4], 0.2), radius=6); x = 812
        for (nm, w), v in zip(hd, r[:4]): f.text(x + w / 2, y + 27, v, "TXT", 16, True, w=w); x += w
    f.text(980, 232, T(tr("單位向量：兩式相同\n非單位向量：差很多", "unit vector: the same\nnon-unit: far apart")), "MUT", 16, w=340)
    chip(f, 800, 272, 360, 86, "red", "", size=16)
    f.text(980, 315, T(tr("不正規化也可以套這個式子？\n不行：正規化是必要前提。", "Can the formula be used without normalizing?\nNo: normalization is a required precondition.")), "TXT", 16, True, w=340)
    takeaway(f, 0, 374, 1160, 52, T(tr("長度都是 1 時，距離變內積：Parzen 的判別式正好是網路的加權和形式。", "With length 1 the distance becomes an inner product: the Parzen discriminant is exactly a network's weighted sum.")))
    return f


def pnn():
    th = np.deg2rad([20, 40, 60, 100, 120]); X = np.c_[np.cos(th), np.sin(th)]; xt = np.array([np.cos(np.deg2rad(70)), np.sin(np.deg2rad(70))]); dots = X @ xt; ys = np.exp((dots - 1) / 0.25)
    oA = 0.5 / 3 * ys[:3].sum(); oB = 0.5 / 2 * ys[3:].sum()
    f = NFig("lec05-kp18-arch", 1160, 470, T(tr(f"機率神經網路（PNN）三層：輸入層 l 個節點；隱藏層 N 個節點，第 k 個節點的權重就是第 k 個訓練向量，輸出 y_k=exp((x_kᵀx−1)/h²)；輸出層每類一個線性組合器，只連到該類的隱藏節點，output_m=P(ω_m)/N_m·Σy_k，取最大者為類別。幾乎不用訓練：沒有梯度下降、區域極小值或 epoch。算例（h=0.5、先驗各 0.5）：A 類在 20°、40°、60°，B 類在 100°、120°，測試向量 70°；x_kᵀx=cos(θ_k−70°)={'、'.join(hu(v,4) for v in dots)}；y={'、'.join(hu(v,4) for v in ys)}；A 類 0.5/3×{hu(ys[:3].sum(),4)}={hu(oA,4)}，B 類 0.5/2×{hu(ys[3:].sum(),4)}={hu(oB,4)}，A 勝。",
                                              f"The probabilistic neural network (PNN) has three layers: l input nodes; N hidden nodes, whose k-th weight is the k-th training vector and whose output is y_k=exp((x_kᵀx−1)/h²); one linear combiner per class in the output layer, connected only to that class's hidden nodes, output_m=P(ω_m)/N_m·Σy_k, the largest wins. Almost no training: no gradient descent, local minima or epochs. Worked example (h=0.5, priors 0.5): class A at 20°, 40°, 60°, class B at 100°, 120°, test vector 70°; x_kᵀx=cos(θ_k−70°)={', '.join(hu(v,4) for v in dots)}; y={', '.join(hu(v,4) for v in ys)}; class A 0.5/3×{hu(ys[:3].sum(),4)}={hu(oA,4)}, class B 0.5/2×{hu(ys[3:].sum(),4)}={hu(oB,4)}, A wins.")))
    card(f, 0, 0, 1160, 50, "yellow", T(tr("機率神經網路（PNN）：把 Parzen 判別式搬進網路；隱藏層的權重就是訓練向量，幾乎不用訓練。", "probabilistic neural network (PNN): the Parzen discriminant put into a network; the hidden weights are the training vectors, almost no training.")), name=T(tr("term:機率神經網路", "term:Probabilistic neural network")), math=False)
    for x0, w, role, hd in ((10, 220, "grey", tr("輸入層：$l$ 個節點", "input layer: $l$ nodes")), (330, 440, "blue", tr("隱藏層：$N$ 個節點（權重＝訓練向量）", "hidden layer: $N$ nodes (weights = training vectors)")), (850, 300, "orange", tr("輸出層：每類一個線性組合器", "output layer: one combiner per class"))):
        header_chip(f, x0, 62, w, 32, role, T(hd).replace("$", ""), size=16)
    xin, xh, xo = 120, 430, 940; yin = [190, 290]; yh = [130 + k * 62 for k in range(5)]; yo = [190, 292]
    for a in yin:
        for b in yh: f.line(xin + 20, a, xh - 22, b, "MUT", 1)
    for k, b in enumerate(yh): f.line(xh + 22, b, xh + 36, b, solid("blue" if k < 3 else "orange"), 3); f.line(800, b, xo - 70, yo[0] if k < 3 else yo[1], solid("blue" if k < 3 else "orange"), 3)
    for lab, v, a in (("$x_1$", np.cos(np.deg2rad(70)), yin[0]), ("$x_2$", np.sin(np.deg2rad(70)), yin[1])):
        f.rect(xin - 22, a - 22, 44, 44, fill=tint("grey", 0.4), stroke=solid("grey"), sw=3, radius=22, text=lab, tcolor="TXT", size=16, bold=True, math=True); f.text(xin - 72, a, hu(v, 3), "TXT", 16, True, "right", w=60)
    f.text(xin, 345, T(tr("測試向量 70°\n$(\\cos70°,\\ \\sin70°)$", "test vector 70°\n$(\\cos70°,\\ \\sin70°)$")), "TXT", 16, w=200, math=True)
    angs = [20, 40, 60, 100, 120]
    for k, b in enumerate(yh):
        role = "blue" if k < 3 else "orange"; f.rect(xh - 22, b - 22, 44, 44, fill=tint(role, 0.4), stroke=solid(role), sw=3, radius=22, text=f"$y_{k+1}$", tcolor="TXT", size=16, bold=True, math=True)
        f.text(xh - 40, b, f"{angs[k]}°", "TXT", 16, True, "right", w=50); f.text(xh + 36, b, f"$x_k^Tx={hu(dots[k],4)}$，$y_{k+1}={hu(ys[k],4)}$", "TXT", 16, False, "left", w=340, math=True)
    for lab, o, a, role, nm in (("A", oA, yo[0], "blue", tr("A 類：$\\frac{0.5}{3}\\times1.7658$", "class A: $\\frac{0.5}{3}\\times1.7658$")), ("B", o2 if False else oB, yo[1], "orange", tr("B 類：$\\frac{0.5}{2}\\times0.8247$", "class B: $\\frac{0.5}{2}\\times0.8247$"))):
        f.rect(xo - 70, a - 38, 280, 76, fill=tint(role, 0.25), stroke=solid(role), sw=2, radius=12); f.text(xo + 70, a - 14, T(nm), "TXT", 16, w=260, math=True); f.text(xo + 70, a + 16, f"= {hu(o,4)}", "TXT", 20, True, w=260)
    chip(f, 850, 340, 300, 44, "green", T(tr("A 勝（70° 離 60°、40° 較近）", "A wins (70° is near 60° and 40°)")), size=16, bold=True)
    f.rect(0, 400, 1160, 60, fill=tint("violet", 0.14), radius=12)
    f.text(580, 430, T(tr("隱藏 $y_k=\\exp((x_k^Tx-1)/h^2)$（$h=0.5$）；輸出 $(P(\\omega_m)/N_m)\\sum y_k$，先驗各 0.5，取最大", "hidden $y_k=\\exp((x_k^Tx-1)/h^2)$ ($h=0.5$); output $(P(\\omega_m)/N_m)\\sum y_k$, priors 0.5, take the largest")), "TXT", 17, w=1120, math=True)
    return f


def hcurve():
    hs = (0.02, 0.05, 0.1, 0.3, 1, 5); Xtr, ytr, a = N.dirdata(50, 0, 0); Xte, yte, ae = N.dirdata(500, 1, 0)
    tr_acc = [float(hu(np.mean(N.pnn(Xtr, ytr, Xtr, h) == ytr) * 100, 0)) for h in hs]; te_acc = [float(hu(np.mean(N.pnn(Xtr, ytr, Xte, h) == yte) * 100, 1)) for h in hs]
    base = float(hu(np.mean((np.degrees(np.arctan2(Xte[:, 1], Xte[:, 0])) > 75) == (yte == 1)) * 100, 1))
    spec = {
        "id": "lec05-kp18-hcurve",
        "alt": tr("核寬 h 對準確率的影響（方向資料：A 類方向 N(40°,25°²)、B 類 N(110°,25°²)，長度均勻 [0.5,3]；每類 50 個訓練點種子 0、每類 500 個新點種子 1）：h=0.02、0.05、0.1、0.3、1、5 的訓練正確率 " + "、".join(f"{v:g}%" for v in tr_acc) + "；新點 " + "、".join(f"{v:g}%" for v in te_acc) + f"；基準：只在 75° 切一刀的新點準確率 {base}%。兩類本來就重疊，訓練不會到 100%；h=0.02 核太窄，貼近個別訓練點。",
                  "Effect of the kernel width h on accuracy (direction data: class A direction N(40°,25°²), class B N(110°,25°²), length uniform in [0.5,3]; 50 training points per class with seed 0, 500 new points per class with seed 1): for h=0.02, 0.05, 0.1, 0.3, 1, 5 the training accuracy is " + ", ".join(f"{v:g}%" for v in tr_acc) + "; new points " + ", ".join(f"{v:g}%" for v in te_acc) + f"; baseline: a single cut at 75° gives {base}% on new points. The classes overlap, so training cannot reach 100%; h=0.02 is too narrow and hugs individual training points."),
        "plot": {"x": [0.5, 6.5], "y": [86, 98], "x_ticks": [1, 2, 3, 4, 5, 6], "x_names": {str(i + 1): f"{h:g}" for i, h in enumerate(hs)}, "y_ticks": [88, 92, 96], "x_label": tr("核寬 h（對數刻度）", "kernel width h (log scale)"), "y_label": tr("準確率 %", "accuracy %"), "w": 430, "h": 300, "legend_pos": "left"},
        "series": [{"kind": "hline", "y": base, "role": "grey", "width": 3, "label": f"{base:g}"},
                   {"kind": "line", "points": [(i + 1, v) for i, v in enumerate(tr_acc)], "role": "blue", "width": 4, "markers": True, "name": tr("訓練", "training")},
                   {"kind": "line", "points": [(i + 1, v) for i, v in enumerate(te_acc)], "role": "red", "width": 4, "markers": True, "labels": [f"{v:g}" for v in te_acc], "label_dy": 22, "name": tr("新點", "new points")}],
        "side": [{"title": tr("基準：只切一刀", "Baseline: a single cut"), "role": "grey", "h": 1.0, "text": tr(f"在 75° 切一刀，新點準確率 {base:g}%（灰線）。", f"one cut at 75° gives {base:g}% on new points (grey line).")},
                 {"title": tr("太窄才傷人", "Only too narrow hurts"), "role": "red", "h": 1.2, "text": tr(f"$h=0.02$：訓練 {tr_acc[0]:g}%、新點 {te_acc[0]:g}%：\n核太窄，貼近個別訓練點。", f"$h=0.02$: training {tr_acc[0]:g}%, new {te_acc[0]:g}%:\nthe kernel is too narrow and hugs single points.")},
                 {"title": tr("其餘差不多", "The rest are alike"), "role": "green", "h": 1.2, "text": tr("$h=0.1$ 到 1，新點約 91.6%，\n與切一刀接近：這個簡單例子對 $h$ 不敏感。", "$h=0.1$ to 1: about 91.6% on new points,\nclose to the cut: this simple example is insensitive to $h$.")},
                 {"title": tr("訓練不到 100%", "Training below 100%"), "role": "orange", "h": 0.9, "text": tr("兩類本來就重疊。", "the two classes overlap.")}],
        "takeaway": tr("核寬只有太小才傷人：h=0.1 到 1 的新點準確率約 91.6%，接近只切一刀。", "Only a too-small kernel width hurts: h=0.1 to 1 gives about 91.6%, close to a single cut."),
    }
    return "plot_xy", spec


def nsub():
    Xte, yte, ae = N.dirdata(500, 1, 0); ns = (2, 5, 10, 25, 50, 200); R = []
    for n in ns:
        acc = [np.mean(N.pnn(*N.dirdata(n, s, 0)[:2], Xte, 0.3) == yte) for s in range(100, 120)]; R.append((float(hu(np.mean(acc) * 100, 1)), float(np.std(acc)) * 100, float(hu(np.std(acc), 3))))
    spec = {
        "id": "lec05-kp18-nsub",
        "alt": tr("每類訓練點數 n 與新點準確率（同樣的資料生成方式，h=0.3，每個 n 用 20 組不同訓練集，種子 100 到 119，新點仍是 1000 個）：n=2、5、10、25、50、200 的平均準確率 " + "、".join(f"{r[0]:g}%" for r in R) + "；標準差 " + "、".join(f"{r[2]:.3f}" for r in R) + "。n≥10 後準確率幾乎不再升高，主要變得更穩。",
                  "Number of training points per class n against accuracy on new points (same data generation, h=0.3, 20 different training sets per n with seeds 100 to 119, the same 1000 new points): for n=2, 5, 10, 25, 50, 200 the mean accuracy is " + ", ".join(f"{r[0]:g}%" for r in R) + "; standard deviation " + ", ".join(f"{r[2]:.3f}" for r in R) + ". Beyond n=10 accuracy hardly rises, it mainly becomes steadier."),
        "plot": {"x": [0.5, 6.5], "y": [78, 100], "x_ticks": [1, 2, 3, 4, 5, 6], "x_names": {str(i + 1): str(n) for i, n in enumerate(ns)}, "y_ticks": [80, 90, 100], "x_label": tr("每類訓練點數 n", "training points per class n"), "y_label": tr("新點平均準確率 %", "mean accuracy on new points %"), "w": 430, "h": 300},
        "series": [{"kind": "seg", "a": (i + 1, r[0] - r[1]), "b": (i + 1, min(r[0] + r[1], 99.5)), "role": "grey", "width": 6} for i, r in enumerate(R)]
                  + [{"kind": "line", "points": [(i + 1, r[0]) for i, r in enumerate(R)], "role": "red", "width": 4, "markers": True, "labels": [f"{r[0]:g}" for r in R], "label_dy": 22}],
        "side": [{"title": tr("怎麼讀", "How to read"), "role": "blue", "h": 1.1, "text": tr("紅點是 20 組訓練集的平均，\n灰色棒是正負一個標準差。", "red dots are the mean of 20 training sets,\ngrey bars are plus and minus one standard deviation.")},
                 {"title": tr("標準差", "Standard deviation"), "role": "orange", "h": 1.3, "text": tr("$n=2$：" + f"{R[0][2]:.3f}" + "；$n=5$：" + f"{R[1][2]:.3f}" + "；$n=10$：" + f"{R[2][2]:.3f}" + "\n$n=25$：" + f"{R[3][2]:.3f}" + "；$n=50$：" + f"{R[4][2]:.3f}" + "；$n=200$：" + f"{R[5][2]:.3f}", "$n=2$: " + f"{R[0][2]:.3f}" + "; $n=5$: " + f"{R[1][2]:.3f}" + "; $n=10$: " + f"{R[2][2]:.3f}" + "\n$n=25$: " + f"{R[3][2]:.3f}" + "; $n=50$: " + f"{R[4][2]:.3f}" + "; $n=200$: " + f"{R[5][2]:.3f}")},
                 {"title": tr("不必全部保留", "No need to keep all"), "role": "green", "h": 1.2, "text": tr("這個簡單的資料：$n\\geq10$ 後準確率\n幾乎不再升高，主要是變得更穩。", "this simple data: beyond $n=10$ accuracy hardly rises,\nit mainly becomes steadier.")}],
        "takeaway": tr("留得少，網路就小：n≥10 後準確率幾乎不再升高，只是更穩（標準差 0.011→0.003）。", "Keep fewer and the network is smaller: beyond n=10 accuracy hardly rises, it just steadies (std 0.011→0.003)."),
    }
    return "plot_xy", spec


def cost():
    rows = [(100, 2), (10 ** 4, 10), (10 ** 6, 100)]; memo = lambda b: f"{b/1e9:g} GB" if b >= 1e8 else (f"{b/1e6:g} MB" if b >= 1e5 else f"{b/1e3:g} KB")
    f = NFig("lec05-kp18-cost", 1160, 400, T(tr("PNN 的成本：每筆測試要 N·l 次乘加，要存 N·l 個數（每個 8 位元組）。N=100、l=2：200 次、" + memo(100 * 2 * 8) + "；N=10⁴、l=10：10⁵ 次、" + memo(10 ** 4 * 10 * 8) + "；N=10⁶、l=100：10⁸ 次、" + memo(10 ** 6 * 100 * 8) + "。怎麼記：PNN＝Parzen 估計的平行硬體版；訓練＝抄資料，推論＝全部算一遍，把學習成本轉嫁到推論成本；與稀疏化（KP21）是光譜兩端：全部保留對只留支撐向量。",
                                              "The cost of a PNN: every test point needs N·l multiply-adds and N·l stored numbers (8 bytes each). N=100, l=2: 200 operations, " + memo(100 * 2 * 8) + "; N=10⁴, l=10: 10⁵, " + memo(10 ** 4 * 10 * 8) + "; N=10⁶, l=100: 10⁸, " + memo(10 ** 6 * 100 * 8) + ". To remember: a PNN is the parallel-hardware version of a Parzen estimate; training = copying the data, inference = computing everything, moving the learning cost into the inference cost; with sparsification (KP21) it sits at the opposite end of a spectrum: keep everything against keeping only the support vectors.")))
    header_chip(f, 0, 0, 760, 34, "blue", T(tr("每筆測試的成本：$N\\cdot l$ 次乘加，$N\\cdot l$ 個數（每個 8 位元組）", "Cost per test point: $N\\cdot l$ multiply-adds, $N\\cdot l$ numbers (8 bytes each)")).replace("$", "").replace("\\cdot", "·"), size=17); panel(f, 0, 38, 760, 226, "blue")
    cols = [(20, 210, tr("訓練點數 N、維度 l", "training points N, dimension l")), (240, 250, tr("每筆測試的乘加次數", "multiply-adds per test")), (500, 240, tr("記憶體", "memory"))]
    for x, w, h in cols: f.text(x + w / 2, 66, T(h), "TXT", 16, True, w=w)
    for i, (n, l) in enumerate(rows):
        y = 84 + i * 56; role = ["green", "orange", "red"][i]; f.rect(10, y, 740, 48, fill=tint(role, 0.2), radius=8)
        f.text(125, y + 24, f"$N={'10^{%d}' % round(np.log10(n)) if n > 100 else n}$，$l={l}$", "TXT", 17, True, w=210, math=True); f.text(365, y + 24, f"{n * l:,}" if n * l < 10 ** 5 else f"$10^{{{round(np.log10(n * l))}}}$", "TXT", 18, True, w=250, math=True); f.text(620, y + 24, memo(n * l * 8), "TXT", 18, True, w=240)
    f.text(380, 255, T(tr("隨 $N$ 與 $l$ 線性增加；$N=10^6$、$l=100$ 時每筆測試就要 $10^8$ 次乘加", "grows linearly with $N$ and $l$; at $N=10^6$, $l=100$ each test point needs $10^8$ multiply-adds")), "MUT", 16, w=740, math=True)
    card(f, 780, 0, 380, 126, "violet", T(tr("PNN＝Parzen 估計的平行硬體版：\n訓練＝抄資料；\n推論＝全部算一遍。", "a PNN is the parallel-hardware Parzen estimate:\ntraining = copying the data;\ninference = computing everything.")), math=False)
    card(f, 780, 138, 380, 126, "yellow", T(tr("把學習成本轉嫁到推論成本。", "the learning cost is moved into the inference cost.")), math=False)
    header_chip(f, 0, 280, 1160, 34, "grey", T(tr("與稀疏化（KP21）是光譜的兩端", "Opposite ends of a spectrum with sparsification (KP21)")), size=17)
    f.line(120, 340, 1040, 340, "MUT", 3); f.ellipse(120, 340, 16, solid("red")); f.ellipse(1040, 340, 16, solid("green"))
    f.text(120, 366, T(tr("全部保留（PNN）", "keep everything (PNN)")), "TXT", 16, True, w=240); f.text(1040, 366, T(tr("只留支撐向量（KP21）", "keep only support vectors (KP21)")), "TXT", 16, True, w=280)
    return f


def rings():
    rng = np.random.default_rng(0); th = rng.uniform(0, 2 * np.pi, 400); r = np.r_[rng.normal(1.0, 0.15, 200), rng.normal(2.2, 0.2, 200)]; X = np.c_[r * np.cos(th), r * np.sin(th)]; y = np.r_[np.zeros(200), np.ones(200)]; idx = rng.permutation(400)
    acc = float(np.mean(N.pnn(X[idx[:200]], y[idx[:200]], X[idx[200:]], 0.3) == y[idx[200:]]))
    f = NFig("lec05-kp18-rings", 1160, 430, T(tr(f"正規化的代價：同心環，兩類只差在半徑。角度均勻；半徑 A 類 N(1.0,0.15²)、B 類 N(2.2,0.2²)；每類 200 點共 400 點（default_rng(0)），隨機分成 200 訓練、200 測試；h=0.3。左：原本的兩個同心環（藍內圈、橘外圈）；右：正規化後兩類都落在單位圓上、角度分布相同，PNN 新點只對 {hu(acc*100,1)}%，等於亂猜。正規化丟掉了長度；用 PNN 前要確認類別資訊在方向，不在長度。",
                                              f"The price of normalization: concentric rings whose two classes differ only in radius. Angles uniform; radius N(1.0,0.15²) for class A and N(2.2,0.2²) for class B; 200 points per class, 400 in all (default_rng(0)), randomly split into 200 training and 200 test points; h=0.3. Left: the two original rings (blue inner, orange outer); right: after normalization both classes lie on the unit circle with the same angle distribution and the PNN gets only {hu(acc*100,1)}% on new points, a coin flip. Normalization throws away the length; before using a PNN make sure the class information is in the direction, not the length.")))
    header_chip(f, 0, 0, 360, 34, "blue", T(tr("正規化前：兩個同心環", "Before: two concentric rings")), size=17); panel(f, 0, 38, 360, 300, "blue")
    cx, cy, sc = 180, 188, 52
    for p, s in zip(X, y): (f.ellipse(cx + p[0] * sc, cy - p[1] * sc, 7, solid("blue")) if s < 1 else f.square(cx + p[0] * sc, cy - p[1] * sc, 7, solid("orange")))
    header_chip(f, 380, 0, 360, 34, "red", T(tr("正規化後：全落在單位圓上", "After: all on the unit circle")), size=17); panel(f, 380, 38, 360, 300, "red")
    cx2, R = 560, 105; circ(f, cx2, cy, R, "MUT", 1.5)
    for p, s in zip(X, y):
        u = p / np.linalg.norm(p); (f.ellipse(cx2 + u[0] * R, cy - u[1] * R, 7, solid("blue")) if s < 1 else f.square(cx2 + u[0] * R, cy - u[1] * R, 7, solid("orange")))
    f.text(560, 322, T(tr("兩類角度分布相同，無法區分", "same angle distribution, no way to tell")), "TXT", 16, True, w=340)
    card(f, 760, 0, 400, 128, "grey", T(tr("設定：角度均勻；半徑 A 類 $N(1.0,0.15^2)$、\nB 類 $N(2.2,0.2^2)$；每類 200 點共 400 點，\n隨機分成 200 訓練、200 測試；$h=0.3$。", "setup: uniform angle; radius $N(1.0,0.15^2)$ (A),\n$N(2.2,0.2^2)$ (B); 200 per class, split\n200 train / 200 test; $h=0.3$.")))
    card(f, 760, 140, 400, 90, "red", T(tr(f"結果：PNN 新點只對 {hu(acc*100,1)}%，\n等於亂猜。", f"result: the PNN gets only {hu(acc*100,1)}% on new points,\na coin flip.")))
    card(f, 760, 242, 400, 96, "yellow", T(tr("正規化丟掉了長度：用 PNN 前要確認\n類別資訊在方向，不在長度。", "normalization throws away the length: before a PNN\nmake sure the class information is in the direction.")), math=False)
    takeaway(f, 0, 356, 1160, 52, T(tr(f"兩類只差在半徑時，正規化後 PNN 只對 {hu(acc*100,1)}%：正規化的代價是丟掉長度。", f"When the classes differ only in radius, the PNN gets only {hu(acc*100,1)}% after normalization: the price is the lost length.")))
    return f


def solution():
    spec = {
        "id": "lec05-kp18-solution",
        "alt": tr("解答：x_A=(1,0)、x_B=(0,1)，x=(0.8,0.6)，先驗各 0.5。h=1：x_Aᵀx=0.8、x_Bᵀx=0.6；y_A=e^{−0.2}=0.819、y_B=e^{−0.4}=0.670；輸出 0.5×0.819=0.409 對 0.5×0.670=0.335，屬 A。h=0.2：y_A=e^{−5}=0.0067、y_B=e^{−10}=0.000045；輸出 0.0034 對 0.00002，仍屬 A；比值從 0.409/0.335≈1.22 變成約 148。核對：h 越小，近的訓練點越被放大，遠的越被壓小。",
                  "Solution: x_A=(1,0), x_B=(0,1), x=(0.8,0.6), priors 0.5. h=1: x_Aᵀx=0.8, x_Bᵀx=0.6; y_A=e^{−0.2}=0.819, y_B=e^{−0.4}=0.670; outputs 0.5×0.819=0.409 against 0.5×0.670=0.335, class A. h=0.2: y_A=e^{−5}=0.0067, y_B=e^{−10}=0.000045; outputs 0.0034 against 0.00002, still A; the ratio goes from 0.409/0.335≈1.22 to about 148. Check: the smaller h, the more the near training points are amplified and the far ones suppressed."),
        "steps": [{"role": "blue", "title": tr("① 題 1：h=1", "① Q1: h=1"), "lines": ["$x_A^Tx=0.8,\\ x_B^Tx=0.6$", "$y_A=e^{-0.2}=0.819$", "$y_B=e^{-0.4}=0.670$"], "result": tr("0.409 對 0.335：A", "0.409 vs 0.335: A")},
                  {"role": "green", "title": tr("② 題 2：h=0.2", "② Q2: h=0.2"), "lines": ["$h^2=0.04$", "$y_A=e^{-5}=0.0067$", "$y_B=e^{-10}=0.000045$"], "result": tr("0.0034 對 0.00002：仍 A", "0.0034 vs 0.00002: still A")},
                  {"role": "orange", "title": tr("③ 核對", "③ Check"), "lines": [tr("$h=1$：比值約 1.22", "$h=1$: ratio about 1.22"), tr("$h=0.2$：比值約 148", "$h=0.2$: ratio about 148"), tr("近的被放大，遠的被壓小", "near amplified, far suppressed")], "result": "1.22 → 148"}],
        "takeaway": tr("h 越小，近的訓練點越被放大：兩類輸出的比值從約 1.22 變成約 148。", "The smaller h, the more near training points are amplified: the output ratio goes from about 1.22 to about 148."),
    }
    return "steps_flow", spec
