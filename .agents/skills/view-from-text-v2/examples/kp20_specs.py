"""KP20 (beyond SVM: the kernel trick and kernel perceptron) -- tenth sample of view-from-text-v2. Plan: kp20_plan.json. Every number is recomputed here with kp20_nums.py
(same settings as the lecture; verified against it)."""
import os, sys
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools")); sys.path.insert(0, HERE)
from figkit.parts import T, NFig, header_chip, panel, chip, outline_box, takeaway, tint, solid, INK
import kp20_nums as N
from kp21_nums import rings, gk


def tr(zh, en): return {"zh": zh, "en": en}


def hu(x, nd):   # half-up rounding to nd decimals
    return str(Decimal(repr(round(float(x), 9))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


G5 = lambda A, B: gk(A, B, 0.5)


def card(f, x, y, w, h, role, text, name=None, size=16, math=True):
    f.rect(x, y, w, h, fill=tint(role, 0.12), stroke=tint(role, 0.55), sw=1.5, radius=12)
    f.text(x + 16, y + h / 2, text, "TXT", size, False, "left", w=w - 32, math=math, name=name)


def recipe():
    f = NFig("lec05-kp20-recipe", 1160, 372, T(tr("核技巧的三步配方：①把演算法改寫成只含內積；②把每個內積換成核 K；③把解寫成訓練點的線性組合。權重向量永遠不必算出來；K 要滿足 Mercer 條件。核化＝把只用內積的演算法的內積換成核函數，得到原空間裡的非線性版本。",
                                              "The three-step recipe of the kernel trick: (1) rewrite the algorithm so that it uses only inner products; (2) replace every inner product by a kernel K; (3) write the solution as a linear combination of the training points. The weight vector never has to be computed; K must satisfy the Mercer condition. Kernelization = replacing the inner products of an algorithm that uses only inner products by a kernel function, giving a non-linear version in the original space.")))
    card(f, 0, 0, 1160, 66, "blue", T(tr("核化：把只用內積的演算法，內積換成核函數，變成原空間裡的非線性版本。", "kernelization: replace the inner products of an algorithm that uses only inner products by a kernel function, giving a non-linear version in the original space.")), name=T(tr("term:核化", "term:Kernelization")), math=False)
    steps = [("blue", tr("① 只含內積", "① Only inner products"), tr("把演算法改寫成\n只用內積 $x^Tz$", "rewrite the algorithm so\nit uses only inner products $x^Tz$")),
             ("orange", tr("② 內積換成核", "② Inner product to kernel"), tr("每個 $x^Tz$ 換成 $K(x,z)$，\n$K=\\phi(x)^T\\phi(z)$", "replace every $x^Tz$ by $K(x,z)$,\n$K=\\phi(x)^T\\phi(z)$")),
             ("green", tr("③ 解是線性組合", "③ A linear combination"), tr("把解寫成訓練點的\n線性組合 $\\sum_ia_iK(x,x_i)$", "write the solution as a linear\ncombination $\\sum_ia_iK(x,x_i)$"))]
    w = (1160 - 2 * 36) / 3
    for k, (role, hd, tx) in enumerate(steps):
        x = k * (w + 36); header_chip(f, x, 84, w, 36, role, T(hd), size=18); panel(f, x, 124, w, 110, role); f.text(x + w / 2, 179, T(tx), "TXT", 18, w=w - 24, math=True)
        if k < 2: f.text(x + w + 18, 179, "→", "TXT", 26, True, w=36)
    chip(f, 0, 252, 560, 44, "yellow", T(tr("權重向量 $w$ 永遠不必算出來", "the weight vector $w$ never has to be computed")), size=18, bold=True, dark=True, math=True)
    chip(f, 600, 252, 560, 44, "violet", T(tr("核 $K$ 要滿足 Mercer 條件（KP19）", "the kernel $K$ must satisfy the Mercer condition (KP19)")), size=18, bold=True, dark=True, math=True)
    takeaway(f, 0, 314, 1160, 50, T(tr("核技巧不是 SVM 的專利：任何只用內積的線性演算法，都能核化成非線性版本。", "The kernel trick is not an SVM patent: any algorithm using only inner products can be kernelized.")))
    return f


def kmean():
    rng = np.random.default_rng(5); P1 = rng.normal([-1.5, 0.6], 0.45, (7, 2)); P2 = rng.normal([1.5, -0.4], 0.45, (7, 2)); m1, m2 = P1.mean(0), P2.mean(0); xn = np.array([0.35, 0.55])
    f = NFig("lec05-kp20-kmean", 1160, 462, T(tr("歐氏分類器（最近平均）：新點離哪一類的平均比較近，就判給哪一類。左：核空間（示意）裡兩類的點、各自的平均 μ₁、μ₂ 與新點 φ(x) 到兩個平均的距離。右：展開平方、兩距離相減時 ‖φ(x)‖² 互相消掉；類 2 較近等於 ⟨φ(x),μ₂−μ₁⟩>θ，θ=(‖μ₂‖²−‖μ₁‖²)/2；換成核後 ⟨φ(x),μ_k⟩ 是各類核值 K(x,x_i) 的平均，‖μ_k‖² 是類內核值的平均。判類規則：類 2 的核值平均減類 1 的核值平均大於 θ 就判類 2。與 Parzen 估計的差別在還要扣掉門檻 θ。",
                                              "The Euclidean (nearest-mean) classifier: a new point goes to the class whose mean it is closer to. Left: a sketch of the kernel space with the two classes, their means μ₁, μ₂ and the distances from the new point φ(x) to both means. Right: when the squares are expanded and the two distances subtracted, ‖φ(x)‖² cancels; class 2 is closer exactly when ⟨φ(x),μ₂−μ₁⟩>θ with θ=(‖μ₂‖²−‖μ₁‖²)/2; with kernels ⟨φ(x),μ_k⟩ is the average of the kernel values K(x,x_i) of class k and ‖μ_k‖² the average of the kernel values inside the class. The rule: choose class 2 when the average kernel value of class 2 minus that of class 1 exceeds θ. The difference from a Parzen estimate is the extra threshold θ.")))
    header_chip(f, 0, 0, 500, 34, "blue", T(tr("核空間裡的兩類平均（示意）", "The two class means in kernel space (sketch)")), size=17); panel(f, 0, 38, 500, 330, "blue")
    sx = lambda v: 250 + v * 100; sy = lambda v: 215 - v * 100
    for p in P1: f.ellipse(sx(p[0]), sy(p[1]), 10, solid("blue"))
    for p in P2: f.square(sx(p[0]), sy(p[1]), 10, solid("orange"))
    f.ellipse(sx(m1[0]), sy(m1[1]), 22, "#1c1712"); f.ellipse(sx(m1[0]), sy(m1[1]), 16, solid("blue")); f.square(sx(m2[0]), sy(m2[1]), 22, "#1c1712"); f.square(sx(m2[0]), sy(m2[1]), 16, solid("orange"))
    f.ellipse(sx(xn[0]), sy(xn[1]), 18, solid("green"))
    for m in (m1, m2):
        for t in np.linspace(0, 1, 11)[:-1:2]: f.line(sx(xn[0] + (m[0] - xn[0]) * t), sy(xn[1] + (m[1] - xn[1]) * t), sx(xn[0] + (m[0] - xn[0]) * (t + 0.1)), sy(xn[1] + (m[1] - xn[1]) * (t + 0.1)), "MUT", 2)
    f.text(sx(m1[0]), sy(m1[1]) + 34, "$\\mu_1$", "TXT", 18, True, w=60, math=True); f.text(sx(m2[0]) - 34, sy(m2[1]) + 2, "$\\mu_2$", "TXT", 18, True, w=60, math=True); f.text(sx(xn[0]) + 4, sy(xn[1]) - 26, "$\\phi(x)$", "TXT", 18, True, w=80, math=True)
    f.text(250, 354, T(tr("新點離 $\\mu_2$ 較近 → 判類 2", "the new point is closer to $\\mu_2$ → class 2")), "TXT", 16, True, w=470, math=True)
    f.rect(0, 380, 500, 62, fill=tint("orange", 0.12), stroke=tint("orange", 0.55), sw=1.5, radius=12)
    f.text(16, 411, T(tr("歐氏分類器（最近平均）：新點離哪一類的\n平均比較近，就判給哪一類。", "Euclidean (nearest-mean) classifier: a new point goes\nto the class whose mean it is closer to.")), "TXT", 16, False, "left", w=470, name=T(tr("term:歐氏分類器", "term:Euclidean classifier")))
    header_chip(f, 520, 0, 640, 34, "green", T(tr("推導的關鍵三步", "The three key steps")), size=17)
    steps = [("blue", tr("① 展開平方，兩個距離相減：\n$\\|\\phi(x)\\|^2$ 兩邊都有，互相消掉。", "① expand the squares and subtract the distances:\n$\\|\\phi(x)\\|^2$ appears in both and cancels.")),
             ("orange", tr("② 類 2 較近 $\\Leftrightarrow\\langle\\phi(x),\\mu_2-\\mu_1\\rangle>\\theta$，\n$\\theta=\\frac{\\|\\mu_2\\|^2-\\|\\mu_1\\|^2}{2}$。", "② class 2 is closer $\\Leftrightarrow\\langle\\phi(x),\\mu_2-\\mu_1\\rangle>\\theta$,\n$\\theta=\\frac{\\|\\mu_2\\|^2-\\|\\mu_1\\|^2}{2}$.")),
             ("violet", tr("③ 換成核：$\\langle\\phi(x),\\mu_k\\rangle$ 是 $K(x,x_i)$ 的類平均，\n$\\|\\mu_k\\|^2$ 是類內核值的平均。", "③ with kernels: $\\langle\\phi(x),\\mu_k\\rangle$ is the class average of $K(x,x_i)$,\n$\\|\\mu_k\\|^2$ the average of the kernel values in the class."))]
    for k, (role, tx) in enumerate(steps):
        y = 44 + k * 82; card(f, 520, y, 640, 74, role, T(tx))
    f.rect(520, 292, 640, 100, fill=tint("green", 0.14), stroke=tint("green", 0.6), sw=2, radius=12)
    f.text(840, 322, T(tr("判類規則（類 2 的條件）：", "the rule (the condition for class 2):")), "TXT", 16, True, w=620); f.text(840, 362, "$\\frac{1}{N_2}\\sum_{i:y_i=-1}K(x,x_i)-\\frac{1}{N_1}\\sum_{i:y_i=+1}K(x,x_i)>\\theta$", "TXT", 20, w=620, math=True)
    chip(f, 520, 404, 640, 52, "yellow", T(tr("與 Parzen 估計的差別：核化的歐氏分類器還要扣掉門檻 $\\theta$", "unlike a Parzen estimate, it also subtracts the threshold $\\theta$")), size=17, bold=True, dark=True, math=True)
    return f


def _cmp():
    X, y = rings(60, 1); Xt, yt = rings(500, 2); R = []
    for nm, K, th in (("lin", N.lin, True), ("quad", N.quad, True), ("g", G5, True), ("g0", G5, False)):
        p, m1, m2 = N.nearest_mean(X, y, Xt, K, th); R.append((int((p == yt).sum()), m1, m2))
    return R


def kmeanres():
    R = _cmp(); m1, m2 = R[2][1], R[2][2]
    spec = {
        "id": "lec05-kp20-kmeanres",
        "alt": tr(f"同心環（標準差 0.25，訓練每類 60 點種子 1，新資料每類 500 點共 1000 點種子 2，高斯核 γ=0.5，二次核 (xᵀz+1)²）上核化歐氏分類器的新點判對數：線性核 {R[0][0]}（{hu(R[0][0]/10,1)}%）、二次核 {R[1][0]}（{hu(R[1][0]/10,1)}%）、高斯核 {R[2][0]}（{hu(R[2][0]/10,1)}%）、高斯核拿掉 θ {R[3][0]}（{hu(R[3][0]/10,1)}%）。高斯核的類內核值平均：類 1（外環）{hu(m1,3)}、類 2（內環）{hu(m2,3)}，2θ≈{hu(m2-m1,3)}。",
                  f"Number of the 1000 new points classified correctly by the kernelized Euclidean classifier on concentric rings (standard deviation 0.25, 60 training points per class with seed 1, 500 new points per class with seed 2, Gaussian kernel γ=0.5, quadratic kernel (xᵀz+1)²): linear kernel {R[0][0]} ({hu(R[0][0]/10,1)}%), quadratic {R[1][0]} ({hu(R[1][0]/10,1)}%), Gaussian {R[2][0]} ({hu(R[2][0]/10,1)}%), Gaussian without θ {R[3][0]} ({hu(R[3][0]/10,1)}%). The in-class kernel averages of the Gaussian kernel: class 1 (outer ring) {hu(m1,3)}, class 2 (inner ring) {hu(m2,3)}, so 2θ≈{hu(m2-m1,3)}."),
        "unit": tr("新點準確率（%）；同心環 σ=0.25、每類 60 點訓練、γ=0.5", "accuracy on new points (%); rings σ=0.25, 60 training points per class, γ=0.5"),
        "bars": [{"label": tr("線性核", "linear kernel"), "value": float(hu(R[0][0] / 10, 1)), "role": "grey", "note": T(tr(f"判對 {R[0][0]}／1000", f"{R[0][0]} of 1000"))},
                 {"label": tr("二次核", "quadratic kernel"), "value": float(hu(R[1][0] / 10, 1)), "role": "blue", "note": T(tr(f"判對 {R[1][0]}／1000", f"{R[1][0]} of 1000"))},
                 {"label": tr("高斯核", "Gaussian kernel"), "value": float(hu(R[2][0] / 10, 1)), "role": "green", "note": T(tr(f"判對 {R[2][0]}／1000", f"{R[2][0]} of 1000"))},
                 {"label": tr("高斯核，拿掉 θ", "Gaussian, without θ"), "value": float(hu(R[3][0] / 10, 1)), "role": "red", "note": T(tr(f"判對 {R[3][0]}／1000", f"{R[3][0]} of 1000"))}],
        "max": 100, "ticks": [0, 25, 50, 75, 100],
        "takeaway": tr(f"兩類內部核值平均不同（{hu(m1,3)} 對 {hu(m2,3)}），θ 在補這個差；拿掉就剩 {hu(R[3][0]/10,1)}%。", f"The in-class kernel averages differ ({hu(m1,3)} against {hu(m2,3)}) and θ makes up for it; without it only {hu(R[3][0]/10,1)}% is left."),
    }
    return "bars_compare", spec


def kperc():
    f = NFig("lec05-kp20-kperc", 1160, 450, T(tr("核感知器：線性感知器每次分錯（y_t g(x_t)≤0）就更新 w←w+y_t φ(x_t)，w₀←w₀+y_t；所以最後 w 是訓練點的線性組合，令 α_i 為第 i 個點被分錯的次數：w=Σα_i y_i φ(x_i)，w₀=Σα_i y_i；判別式 g(x)=Σα_i y_i K(x_i,x)+Σα_i y_i，權重向量不必算出來。演算法：α 全為 0；一輪輪掃過訓練點，用目前的 α 算 g(x_i)；若 y_i g(x_i)≤0 就 α_i←α_i+1；整輪都沒有分錯就停。收斂：映射到高維後依 Cover 定理有很高的機率線性可分。",
                                              "The kernel perceptron: the linear perceptron updates w←w+y_t φ(x_t), w₀←w₀+y_t each time a point is misclassified (y_t g(x_t)≤0); the final w is therefore a linear combination of training points, and with α_i the number of times point i was misclassified, w=Σα_i y_i φ(x_i), w₀=Σα_i y_i; the discriminant g(x)=Σα_i y_i K(x_i,x)+Σα_i y_i never needs the weight vector. Algorithm: all α are 0; sweep the training points round by round, computing g(x_i) from the current α; if y_i g(x_i)≤0 then α_i←α_i+1; stop when a whole round has no mistakes. Convergence: after the mapping to a high dimension, by Cover's theorem the data are linearly separable with high probability.")))
    cols = [("blue", tr("線性感知器", "Linear perceptron"), tr("分錯就更新：\n$w\\leftarrow w+y_t\\phi(x_t)$，\n$w_0\\leftarrow w_0+y_t$", "update on each mistake:\n$w\\leftarrow w+y_t\\phi(x_t)$,\n$w_0\\leftarrow w_0+y_t$"), None),
            ("orange", tr("所以 w 是訓練點的組合", "So w is a combination"), tr("$\\alpha_i$：第 $i$ 點被分錯的次數\n$w=\\sum_i\\alpha_iy_i\\phi(x_i)$\n$w_0=\\sum_i\\alpha_iy_i$", "$\\alpha_i$: how often point $i$ was misclassified\n$w=\\sum_i\\alpha_iy_i\\phi(x_i)$\n$w_0=\\sum_i\\alpha_iy_i$"), None),
            ("green", tr("核感知器", "Kernel perceptron"), tr("核感知器：用誤分類次數 $\\alpha_i$ 取代權重向量。\n$g(x)=\\sum_i\\alpha_iy_iK(x_i,x)+\\sum_i\\alpha_iy_i$", "kernel perceptron (counters $\\alpha_i$):\n$g(x)=\\sum_i\\alpha_iy_iK(x_i,x)$\n$\\ \\ +\\sum_i\\alpha_iy_i$"), T(tr("term:核感知器", "term:Kernel perceptron")))]
    w = (1160 - 2 * 36) / 3
    for k, (role, hd, tx, nm) in enumerate(cols):
        x = k * (w + 36); header_chip(f, x, 0, w, 36, role, T(hd), size=18); panel(f, x, 40, w, 124, role); f.text(x + w / 2, 102, T(tx), "TXT", 16, w=w - 6, math=True, name=nm)
        if k < 2: f.text(x + w + 18, 102, "→", "TXT", 26, True, w=36)
    steps = [tr("① 全部 $\\alpha_i=0$", "① all $\\alpha_i=0$"), tr("② 掃過訓練點，用目前的 $\\alpha$ 算 $g(x_i)$", "② sweep the points, compute $g(x_i)$ from $\\alpha$"), tr("③ $y_ig(x_i)\\leq0$（分錯）就 $\\alpha_i\\leftarrow\\alpha_i+1$", "③ mistake ($y_ig(x_i)\\leq0$): $\\alpha_i\\leftarrow\\alpha_i+1$"), tr("④ 整輪都沒有分錯就停", "④ stop when a round has no mistake")]
    for k, tx in enumerate(steps): chip(f, (k % 2) * 590, 184 + (k // 2) * 66, 570, 56, ["blue", "blue", "orange", "green"][k], T(tx), size=16, bold=True, math=True)
    chip(f, 0, 324, 1160, 52, "yellow", T(tr("收斂：映射到高維後，依 Cover 定理（KP14）資料很可能線性可分，感知器就會停。", "Convergence: after lifting, the data are very likely separable (Cover, KP14), so it stops.")), size=17, bold=True, dark=True)
    takeaway(f, 0, 392, 1160, 52, T(tr("權重向量從不需要被算出來：只留每個訓練點的誤分類次數。", "The weight vector is never computed: only each training point's mistake count is kept.")))
    return f


def _trace():
    P = np.array([[1, 0], [-1, 0], [0, 1], [0, -1.]]); y = np.array([1, 1, -1, -1.]); G = N.quad(P, P); a = np.zeros(4); rows = []
    for ep in range(1, 4):
        for i in range(4):
            g = (a * y) @ G[:, i] + (a * y).sum(); bad = y[i] * g <= 0
            if bad: a[i] += 1
            rows.append((ep, i + 1, int(round(g)), int(round(y[i] * g)), bad, tuple(int(v) for v in a)))
    return G, rows


def trace():
    G, rows = _trace()
    f = NFig("lec05-kp20-trace", 1160, 456, T(tr("手算追蹤：四個點 x₁=(1,0)、x₂=(−1,0) 為 +1，x₃=(0,1)、x₄=(0,−1) 為 −1，核 K=(xᵀz+1)²，XOR 型資料沒有直線能分開。核矩陣 G：第一列 4、0、1、1；第二列 0、4、1、1；第三列 1、1、4、0；第四列 1、1、0、4。第 1 輪：點 1 錯、點 2 對、點 3 錯、點 4 錯；第 2 輪：點 1 對、點 2 錯、點 3、4 對；第 3 輪全對，α=(1,1,1,1)，停。最後 g(x)=2(x₁²−x₂²)：(2,0) 得 +8，(0,2) 得 −8。",
                                              "Hand trace: the four points x₁=(1,0), x₂=(−1,0) are +1 and x₃=(0,1), x₄=(0,−1) are −1, with kernel K=(xᵀz+1)²; this XOR-type data cannot be separated by a line. The kernel matrix G has rows 4, 0, 1, 1; 0, 4, 1, 1; 1, 1, 4, 0; 1, 1, 0, 4. Round 1: point 1 wrong, 2 right, 3 wrong, 4 wrong; round 2: point 1 right, 2 wrong, 3 and 4 right; round 3 all right, α=(1,1,1,1), stop. The final g(x)=2(x₁²−x₂²): (2,0) gives +8 and (0,2) gives −8.")))
    header_chip(f, 0, 0, 320, 34, "blue", T(tr("XOR 型的四個點", "Four XOR-type points")), size=17); panel(f, 0, 38, 320, 230, "blue")
    cx, cy, sc = 160, 150, 60; f.line(cx - 110, cy, cx + 110, cy, "MUT", 2); f.line(cx, cy - 90, cx, cy + 90, "MUT", 2)
    for (px, py), s in zip(((1, 0), (-1, 0), (0, 1), (0, -1)), (1, 1, -1, -1)):
        (f.ellipse(cx + px * sc, cy - py * sc, 24, solid("blue")) if s > 0 else f.square(cx + px * sc, cy - py * sc, 24, solid("orange")))
    for k, (px, py) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))): f.text(cx + px * sc + (22 if px else 22), cy - py * sc - (22 if py else 16), f"$x_{k+1}$", "TXT", 16, True, w=40, math=True)
    f.text(160, 245, T(tr("藍圓：$+1$　橘方：$-1$；沒有直線能分開", "blue: $+1$, orange: $-1$\nno line separates them")), "TXT", 16, w=300, math=True)
    header_chip(f, 0, 282, 320, 34, "orange", T(tr("核矩陣 G（4×4）", "Kernel matrix G (4×4)")), size=16); panel(f, 0, 320, 320, 130, "orange")
    for i in range(4):
        for j in range(4): f.text(52 + j * 72, 344 + i * 28, str(int(G[i][j])), "TXT", 18, True, w=60)
    f.text(250, 438, "", "TXT", 16)
    X0 = 340; cw = [50, 50, 70, 70, 70, 190]; hd = [tr("輪", "round"), tr("點", "point"), "$g(x_i)$", "$y_ig$", tr("結果", "result"), tr("更新後 $\\alpha$", "$\\alpha$ after")]
    header_chip(f, X0, 0, 820, 34, "green", T(tr("逐輪追蹤", "Round-by-round trace")), size=17); panel(f, X0, 38, 820, 330, "green")
    x = X0 + 14
    for w, h in zip(cw, hd): f.text(x + w / 2, 56, T(h), "TXT", 16, True, w=w, math=True); x += w
    for r, (ep, pt, g, yg, bad, al) in enumerate(rows):
        y0 = 74 + r * 24; f.rect(X0 + 10, y0 - 2, 800, 22, fill=tint("red" if bad else "green", 0.16 if bad else 0.08), radius=4)
        vals = [str(ep), str(pt), str(g), str(yg), T(tr("錯", "wrong") if bad else tr("對", "right")), "(" + ",".join(map(str, al)) + ")" + ("　停" if (ep == 3 and pt == 4) else "")]
        x = X0 + 14
        for w, v in zip(cw, vals): f.text(x + w / 2, y0 + 9, v, "TXT", 16, bad and w == 70, w=w); x += w
    chip(f, X0, 384, 820, 56, "yellow", T(tr("最後 $g(x)=2(x_1^2-x_2^2)$：$(2,0)\\to+8$，$(0,2)\\to-8$", "final $g(x)=2(x_1^2-x_2^2)$: $(2,0)\\to+8$, $(0,2)\\to-8$")), size=19, bold=True, dark=True, math=True)
    return f


def _eps():
    out = {}
    for sd in (0.15, 0.25):
        X, y = rings(60, 1, sd); Xt, yt = rings(500, 2, sd)
        a, e = N.kperc(X, y, G5); p = N.kpred(X, y, a, Xt, G5); out[sd] = (e, int((a > 0).sum()), int((p == yt).sum()))
    X, y = rings(60, 1, 0.25); Xt, yt = rings(500, 2, 0.25); a, e = N.kperc(X, y, N.lin, 500); p = N.kpred(X, y, a, Xt, N.lin); out["lin"] = (e, 0, int((p == yt).sum()))
    return out


def epochs():
    O = _eps(); lg = lambda i: float(np.log10(i + 1))
    sd1, sd2, ln = O[0.15], O[0.25], O["lin"]
    spec = {
        "id": "lec05-kp20-epochs",
        "alt": tr(f"同心環上感知器每一輪的錯誤數（橫軸為對數刻度的輪數；高斯核 γ=0.5，訓練每類 60 點種子 1，新資料每類 500 點種子 2）。標準差 0.15：核感知器 {len(sd1[0])} 輪停、{sd1[1]} 個 α>0、新點 {sd1[2]}/1000（{hu(sd1[2]/10,1)}%）；標準差 0.25：{len(sd2[0])} 輪停、{sd2[1]} 個、{sd2[2]}/1000（{hu(sd2[2]/10,1)}%）；線性感知器最多跑 500 輪仍有錯（最後一輪 {ln[0][-1]} 個錯），新點 {ln[2]}/1000（{hu(ln[2]/10,1)}%）。標準差 0.25 換 γ=0.1、2、8 要 191、571、175 輪才停。",
                  f"Mistakes per round of the perceptron on concentric rings (log-scaled rounds; Gaussian kernel γ=0.5, 60 training points per class with seed 1, 500 new points per class with seed 2). Standard deviation 0.15: the kernel perceptron stops after {len(sd1[0])} rounds with {sd1[1]} points having α>0 and {sd1[2]}/1000 ({hu(sd1[2]/10,1)}%) on new points; standard deviation 0.25: {len(sd2[0])} rounds, {sd2[1]} points, {sd2[2]}/1000 ({hu(sd2[2]/10,1)}%); the linear perceptron still makes mistakes after at most 500 rounds ({ln[0][-1]} in the last round) and gets {ln[2]}/1000 ({hu(ln[2]/10,1)}%). With standard deviation 0.25, γ=0.1, 2, 8 need 191, 571, 175 rounds to stop."),
        "plot": {"x": [-0.1, 2.9], "y": [0, 6.4], "x_ticks": [0, 1, 2, float(np.log10(501))], "x_names": {"0": "1", "1": "10", "2": "100", str(float(np.log10(501))): "500"}, "y_ticks": [0, 2, 4, 6], "x_label": tr("輪數（對數刻度）", "round (log scale)"), "y_label": tr("該輪的錯誤數", "mistakes in the round"), "w": 430, "h": 300, "legend_pos": "right"},
        "series": [{"kind": "line", "points": [(lg(i), v) for i, v in enumerate(sd2[0])], "role": "orange", "width": 3, "name": tr("核，σ=0.25", "kernel, σ=0.25")},
                   {"kind": "line", "points": [(lg(i), v) for i, v in enumerate(sd1[0])], "role": "green", "width": 4, "markers": True, "name": tr("核，σ=0.15", "kernel, σ=0.15")}],
        "side": [{"title": tr("σ=0.15（兩環分得開）", "σ=0.15 (rings apart)"), "role": "green", "h": 1.25, "text": tr(f"核感知器 {len(sd1[0])} 輪停，{sd1[1]} 個 $\\alpha>0$，\n新點 {sd1[2]}/1000（{hu(sd1[2]/10,1)}%）。", f"the kernel perceptron stops after {len(sd1[0])} rounds,\n{sd1[1]} points have $\\alpha>0$, {sd1[2]}/1000 ({hu(sd1[2]/10,1)}%) on new points.")},
                 {"title": tr("σ=0.25（邊緣相碰）", "σ=0.25 (rings touch)"), "role": "orange", "h": 1.25, "text": tr(f"{len(sd2[0])} 輪停，{sd2[1]} 個 $\\alpha>0$，\n新點 {sd2[2]}/1000（{hu(sd2[2]/10,1)}%）。", f"stops after {len(sd2[0])} rounds, {sd2[1]} points have $\\alpha>0$,\n{sd2[2]}/1000 ({hu(sd2[2]/10,1)}%) on new points.")},
                 {"title": tr("線性感知器", "Linear perceptron"), "role": "grey", "h": 1.1, "text": tr(f"500 輪仍有錯（最後一輪 {ln[0][-1]} 個），新點 {ln[2]}/1000（{hu(ln[2]/10,1)}%）。", f"still wrong after 500 rounds ({ln[0][-1]} in the last), {ln[2]}/1000 ({hu(ln[2]/10,1)}%).")},
                 {"title": tr("輪數也看 γ", "Rounds depend on γ"), "role": "red", "h": 1.25, "text": tr("σ=0.25 換 $\\gamma=0.1,2,8$：\n191、571、175 輪才停。", "σ=0.25 with $\\gamma=0.1,2,8$:\n191, 571, 175 rounds to stop.")}],
        "takeaway": tr(f"可分時有限步停（{len(sd1[0])} 輪），兩類靠近時輪數大增（{len(sd2[0])} 輪）；線性感知器 500 輪仍有錯。", f"Separable: stops in {len(sd1[0])} rounds; close classes: {len(sd2[0])}; the linear perceptron still errs after 500."),
    }
    return "plot_xy", spec


def losses():
    zs = np.linspace(-3, 3, 121); hinge = lambda z: max(0, 1 - z); ex = lambda z: np.exp(-z); lg = lambda z: float(np.log1p(np.exp(-z)))
    clip = lambda fn: [(float(z), float(fn(z))) for z in zs if fn(z) <= 8.3]
    spec = {
        "id": "lec05-kp20-losses",
        "alt": tr("四種損失對間隔值 z=y·g(x)（z>0 分對、z≤0 分錯）：hinge max(0,1−z)、指數 e^{−z}、logistic ln(1+e^{−z})、0/1。z=−2、0、2：hinge 3、1、0；指數 7.389、1、0.135；logistic 2.127、0.693、0.127；0/1 1、1、0。指數在負側最陡，z=−2 時指數的罰分是 hinge 的 7.389/3=2.46 倍。",
                  "The four losses against the margin value z=y·g(x) (z>0 right, z≤0 wrong): hinge max(0,1−z), exponential e^{−z}, logistic ln(1+e^{−z}), 0/1. At z=−2, 0, 2: hinge 3, 1, 0; exponential 7.389, 1, 0.135; logistic 2.127, 0.693, 0.127; 0/1 1, 1, 0. The exponential is steepest on the negative side: at z=−2 its penalty is 7.389/3=2.46 times that of hinge."),
        "plot": {"x": [-3.1, 3.1], "y": [0, 8.6], "x_ticks": [-2, 0, 2], "y_ticks": [0, 2, 4, 6, 8], "x_label": tr("間隔值 z", "margin value z"), "y_label": tr("損失", "loss"), "w": 430, "h": 300, "legend_pos": "right"},
        "series": [{"kind": "line", "points": clip(ex), "role": "red", "width": 4, "name": tr("指數", "exponential")},
                   {"kind": "line", "points": clip(hinge), "role": "blue", "width": 4, "name": "hinge"},
                   {"kind": "line", "points": clip(lg), "role": "green", "width": 4, "name": "logistic"},
                   {"kind": "seg", "a": (-3.1, 1), "b": (0, 1), "role": "grey", "width": 4, "name": "0/1"}, {"kind": "seg", "a": (0.02, 0), "b": (3.1, 0), "role": "grey", "width": 4},
                   {"kind": "vline", "x": 0, "role": "grey", "width": 1}],
        "side": [{"title": tr("間隔值", "Margin value"), "role": "blue", "name": tr("term:間隔值", "term:Margin value"), "h": 0.85, "text": tr("z=y·g(x)：z>0 分對，z≤0 分錯。", "z=y·g(x): z>0 is right, z≤0 is wrong.")},
                 {"title": tr("損失函數", "Loss function"), "role": "violet", "name": tr("term:損失函數", "term:Loss function"), "h": 0.85, "text": tr("給每個 z 一個罰分，演算法的個性由它決定。", "a penalty for every z; it sets the algorithm's personality.")},
                 {"title": tr("三個點的值", "Values at three points"), "role": "orange", "h": 1.6, "text": tr("z=−2：hinge 3、指數 7.389、logistic 2.127、0/1 為 1\nz=0：1、1、0.693、1\nz=2：0、0.135、0.127、0", "z=−2: hinge 3, exp 7.389, logistic 2.127, 0/1 is 1\nz=0: 1, 1, 0.693, 1\nz=2: 0, 0.135, 0.127, 0")},
                 {"title": tr("指數最敏感", "Exponential is most sensitive"), "role": "red", "h": 1.2, "text": tr("z=−2 時指數是 hinge 的 7.389/3=2.46 倍：\n錯得離譜的點被罰得特別重。", "at z=−2 the exponential is 7.389/3=2.46 times hinge:\na wildly wrong point is punished far harder.")}],
        "takeaway": tr("指數損失在負側最陡：AdaBoost 把力氣集中在難點的動力，也是它對離群點敏感的地方。", "Steepest at negative z: the exponential loss drives AdaBoost toward hard points and outliers."),
    }
    return "plot_xy", spec


def logup():
    zs = np.linspace(-3, 3, 121); lg = lambda z: float(np.log1p(np.exp(-z))); L2 = np.log(2)
    spec = {
        "id": "lec05-kp20-logup",
        "alt": tr("0/1 損失的階梯（z≤0 為 1、否則 0）、logistic ln(1+e^{−z}) 與除以 ln2 的 logistic。logistic 在 z=0 只有 ln2=0.693<1，低於階梯，不是上界；除以 ln2 後 z=0 剛好是 1、z≤0 都≥1，是 0/1 損失的凸上界（來源稱 normalized logistic）。驗算：z=−2：2.127/0.693=3.069；z=−1：1.313/0.693=1.895。",
                  "The step of the 0/1 loss (1 for z≤0, else 0), the logistic ln(1+e^{−z}) and the logistic divided by ln2. At z=0 the logistic is only ln2=0.693<1, below the step, so it is not an upper bound; after dividing by ln2 it is exactly 1 at z=0 and ≥1 for z≤0, a convex upper bound of the 0/1 loss (the source calls it the normalized logistic). Check: z=−2: 2.127/0.693=3.069; z=−1: 1.313/0.693=1.895."),
        "plot": {"x": [-3.1, 3.1], "y": [0, 4.4], "x_ticks": [-2, 0, 2], "y_ticks": [0, 1, 2, 3, 4], "x_label": tr("間隔值 z", "margin value z"), "y_label": tr("損失", "loss"), "w": 430, "h": 300, "legend_pos": "right"},
        "series": [{"kind": "seg", "a": (-3.1, 1), "b": (0, 1), "role": "grey", "width": 5, "name": tr("0/1 損失", "0/1 loss")}, {"kind": "seg", "a": (0.02, 0), "b": (3.1, 0), "role": "grey", "width": 5},
                   {"kind": "line", "points": [(float(z), lg(z)) for z in zs if lg(z) <= 4.4], "role": "blue", "width": 4, "name": "logistic"},
                   {"kind": "line", "points": [(float(z), lg(z) / L2) for z in zs if lg(z) / L2 <= 4.4], "role": "green", "width": 4, "name": tr("除以 ln2", "divided by ln2")},
                   {"kind": "dots", "points": [(0, L2), (0, 1)], "roles": ["blue", "green"], "d": 14}],
        "side": [{"title": tr("logistic 不是上界", "Logistic is not an upper bound"), "role": "blue", "h": 1.2, "text": tr("hinge 與指數都不小於 0/1 損失；\nlogistic 在 $z=0$ 只有 $\\ln2=0.693<1$。", "hinge and exponential are never below the 0/1 loss;\nlogistic is only $\\ln2=0.693<1$ at $z=0$.")},
                 {"title": tr("正規化的 logistic 損失", "Normalized logistic loss"), "role": "green", "name": tr("term:正規化的 logistic 損失", "term:Normalized logistic loss"), "h": 1.5, "text": tr("正規化的 logistic 損失：除以 $\\ln2$ 的 logistic；\n$z=0$ 剛好是 1，$z\\leq0$ 都 $\\geq1$，\n是 0/1 損失的凸上界。", "normalized logistic loss: the logistic divided by $\\ln2$;\nit is exactly 1 at $z=0$ and $\\geq1$ for $z\\leq0$,\na convex upper bound of the 0/1 loss.")},
                 {"title": tr("驗算", "Check"), "role": "orange", "h": 1.0, "text": tr("$z=-2$：$2.127/0.693=3.069$；\n$z=-1$：$1.313/0.693=1.895$，都大於 1。", "$z=-2$: $2.127/0.693=3.069$;\n$z=-1$: $1.313/0.693=1.895$, both above 1.")}],
        "takeaway": tr("logistic 要除以 ln2，才貼在 0/1 損失階梯的上方。", "The logistic must be divided by ln2 to sit above the 0/1 loss step."),
    }
    return "plot_xy", spec


def solution():
    spec = {
        "id": "lec05-kp20-solution",
        "alt": tr("解答：x=(1,2) 的四個核值 K(x,(1,0))=4、K(x,(−1,0))=0、K(x,(0,1))=9、K(x,(0,−1))=1；Σα_i y_i=0；g=4+0−9−1+0=−6，屬負類。四個平方相加得 g(x)=2(x₁²+1)−2(x₂²+1)=2(x₁²−x₂²)，x=(1,2) 得 2(1−4)=−6，與題 1 相同。g>0 就是 |x₁|>|x₂|，邊界是兩條斜線。",
                  "Solution: for x=(1,2) the four kernel values are K(x,(1,0))=4, K(x,(−1,0))=0, K(x,(0,1))=9, K(x,(0,−1))=1; Σα_i y_i=0; g=4+0−9−1+0=−6, the negative class. Adding the four squares gives g(x)=2(x₁²+1)−2(x₂²+1)=2(x₁²−x₂²), and x=(1,2) gives 2(1−4)=−6, the same as exercise 1. g>0 means |x₁|>|x₂|, so the boundary is two diagonal lines."),
        "steps": [{"role": "blue", "title": tr("① 題 1：四個核", "① Q1: four kernels"), "lines": [tr("$K=4,\\ 0,\\ 9,\\ 1$（依序四點）", "$K=4,0,9,1$ at the four points"), tr("$\\sum_i\\alpha_iy_i=0$", "$\\sum_i\\alpha_iy_i=0$"), "$g=4+0-9-1+0$"], "result": "g = −6"},
                  {"role": "green", "title": tr("② 題 2：化簡", "② Q2: simplify"), "lines": ["$g=2(x_1^2+1)-2(x_2^2+1)$", "$=2(x_1^2-x_2^2)$", tr("$x=(1,2)$：$2(1-4)$", "$x=(1,2)$: $2(1-4)$")], "result": "−6"},
                  {"role": "orange", "title": tr("③ 核對", "③ Check"), "lines": [tr("兩種算法都是 $-6$，屬負類", "both ways give $-6$: the negative class"), tr("$g>0$ 即 $|x_1|>|x_2|$", "$g>0$ means $|x_1|>|x_2|$"), tr("邊界是兩條斜線", "the boundary is two diagonal lines")], "result": tr("直線分不了的四點被分開", "all four are separated")}],
        "takeaway": tr("兩種算法都得 −6；g>0 就是 |x₁|>|x₂|，邊界是兩條斜線。", "Both ways give −6; g>0 means |x₁|>|x₂|, and the boundary is two diagonal lines."),
    }
    return "steps_flow", spec


def ringcmp():
    def circ(f, cx, cy, r, color, sw=2): f.rect(cx - r, cy - r, 2 * r, 2 * r, stroke=color, sw=sw, radius=r)
    R = _cmp(); m1, m2 = R[2][1], R[2][2]; a = [r[0] for r in R]; d1, d2 = hu(m1, 3), hu(m2, 3); th2 = hu(float(d2) - float(d1), 3); th = hu((float(d2) - float(d1)) / 2, 4)
    f = NFig("lec05-kp20-ringcmp", 1160, 452, T(tr(f"同心環上比較（標準差 0.25，訓練每類 60 點種子 1，新資料每類 500 點共 1000 點種子 2）：線性核判對 {a[0]}／1000（{hu(a[0]/10,1)}%）、二次核 {a[1]}（{hu(a[1]/10,1)}%）、高斯核保留 θ {a[2]}（{hu(a[2]/10,1)}%）、高斯核拿掉 θ {a[3]}（{hu(a[3]/10,1)}%）；拿掉 θ 少判對 {a[2]-a[3]} 點。類內核值平均：外環（類 1）{d1}、內環（類 2）{d2}，2θ≈{th2}。",
                                                                       f"Comparison on concentric rings (standard deviation 0.25, 60 training points per class with seed 1, 500 new points per class, 1000 in all, with seed 2): the linear kernel gets {a[0]} of 1000 right ({hu(a[0]/10,1)}%), the quadratic kernel {a[1]} ({hu(a[1]/10,1)}%), the Gaussian kernel with θ {a[2]} ({hu(a[2]/10,1)}%), the Gaussian kernel without θ {a[3]} ({hu(a[3]/10,1)}%); dropping θ costs {a[2]-a[3]} points. In-class kernel averages: outer ring (class 1) {d1}, inner ring (class 2) {d2}, so 2θ≈{th2}.")))
    header_chip(f, 0, 0, 560, 32, "blue", T(tr("① 同一批資料，比四種方法", "① One data set, four methods")), size=17); panel(f, 0, 36, 560, 222, "blue")
    cx, cy, sc = 135, 136, 35
    circ(f, cx, cy, 2 * sc - 12, tint("blue", 0.5), 2); circ(f, cx, cy, 2 * sc + 12, tint("blue", 0.5), 2); circ(f, cx, cy, 2 * sc, solid("blue"), 2)
    circ(f, cx, cy, sc - 10, tint("red", 0.6), 2); circ(f, cx, cy, sc + 10, tint("red", 0.6), 2); circ(f, cx, cy, sc, solid("red"), 2)
    f.line(cx - 78, cy - 58, cx + 70, cy + 52, "MUT", 3); f.ellipse(cx, cy, 10, "TXT")
    f.text(10, 62, T(tr("外環：+1\n平均半徑 2", "outer: +1\nmean radius 2")), "TXT", 16, True, "left", w=140); f.text(174, 62, T(tr("內環：-1\n平均半徑 1", "inner: -1\nmean radius 1")), "TXT", 16, True, "left", w=124)
    f.text(10, 236, T(tr("一條直線，難以分開內外環\n幾何示意，非實驗散點或精確邊界", "one line cannot separate inner from outer\nsketch only, not experiment points")), "MUT", 16, False, "left", w=285)
    f.rect(300, 46, 250, 204, fill=tint("grey", 0.25), radius=10); f.text(314, 64, T(tr("實驗設定", "Experiment setup")), "TXT", 16, True, "left", w=232)
    f.text(314, 156, T(tr("角度均勻隨機\n半徑常態分布，標準差 0.25\n訓練：每類 60 點（種子 1）\n新資料：每類 500 點（種子 2）\n兩類原空間平均都近圓心", "angles uniformly random\nradii normal, standard deviation 0.25\ntraining: 60 per class (seed 1)\nnew data: 500 per class (seed 2)\nboth class means near the origin")), "TXT", 16, False, "left", w=232)
    header_chip(f, 580, 0, 580, 32, "green", T(tr("② 新資料的分類結果", "② Classification of new data")), size=17); panel(f, 580, 36, 580, 222, "green")
    f.rect(596, 44, 552, 26, fill=tint("blue", 0.3), radius=4)
    for x, w, h in ((600, 260, tr("歐氏分類器使用方式", "Euclidean classifier variant")), (870, 130, tr("判對／1000", "right / 1000")), (1010, 130, tr("準確率", "accuracy"))): f.text(x + w / 2, 57, T(h), "TXT", 16, True, w=w)
    rows = [(tr("線性核", "linear kernel"), a[0], "grey"), (tr("二次核", "quadratic kernel"), a[1], "grey"), (tr("高斯核（保留 θ）", "Gaussian (with θ)"), a[2], "green"), (tr("高斯核（拿掉 θ）", "Gaussian (without θ)"), a[3], "orange")]
    for i, (nm, v, role) in enumerate(rows):
        y = 74 + i * 28; f.rect(596, y, 552, 26, fill=tint(role, 0.28 if role != "grey" else 0.12), radius=4)
        f.text(730, y + 13, T(nm), "TXT", 16, True, w=260); f.text(935, y + 13, str(v), "TXT", 16, True, w=130); f.text(1075, y + 13, hu(v / 10, 1) + "%", "TXT", 16, True, w=130)
    f.rect(596, 188, 552, 46, fill=tint("yellow", 0.2), radius=10)
    f.text(872, 200, T(tr(f"拿掉 θ：少判對 {a[2]-a[3]} 點，下降 {hu((a[2]-a[3])/10,1)} 個百分點", f"without θ: {a[2]-a[3]} fewer right, down {hu((a[2]-a[3])/10,1)} points")), "TXT", 16, True, w=540)
    f.text(872, 220, f"{hu(a[2]/10,1)}% → {hu(a[3]/10,1)}%", "TXT", 20, True, w=540)
    f.text(872, 246, "二次核 $K=(x^Tz+1)^2$；高斯核 $K=\\exp(-0.5\\|x-z\\|^2)$" if True else "", "TXT", 16, w=552, math=True)
    header_chip(f, 0, 272, 560, 32, "orange", T(tr("為什麼核化有幫助？", "Why does kernelization help?")), size=17); panel(f, 0, 308, 560, 112, "orange")
    f.text(14, 364, T(tr("原空間：兩類平均都接近圓心，只比到平均的距離，很難辨別內外。\n核空間：核函數給出映射後的相似度，不必真的算出高維座標；\n二次核與高斯核都大幅改善新資料準確率。", "original space: both class means sit near the origin, so\ndistances to the means can hardly tell inside from outside.\nkernel space: similarities after the mapping, no high-dimensional\ncoordinates needed; the quadratic and Gaussian kernels both help a lot.")), "TXT", 16, False, "left", w=535)
    header_chip(f, 580, 272, 580, 32, "violet", T(tr("③ θ 補上兩類自身的差異", "③ θ makes up for the classes' own difference")), size=17); panel(f, 580, 308, 580, 112, "violet")
    f.rect(596, 316, 300, 52, fill=tint("violet", 0.2), radius=6)
    f.text(606, 329, T(tr(f"外環（類 1）：核值平均 ≈ {d1}", f"outer: kernel avg ≈ {d1}")), "TXT", 16, False, "left", w=286); f.text(606, 355, T(tr(f"內環（類 2）：核值平均 ≈ {d2}", f"inner: kernel avg ≈ {d2}")), "TXT", 16, False, "left", w=286)
    f.text(1030, 344, f"$2\\theta\\approx{d2}-{d1}={th2}$\n$\\theta\\approx{th}$", "TXT", 16, w=260, math=True)
    f.text(596, 396, T(tr("類內核值平均：同一類訓練點彼此核相似度的平均；\n兩類不同，θ 不能直接刪掉。", "in-class kernel average: mean similarity among one class's\ntraining points; the classes differ, so θ cannot be dropped.")), "TXT", 16, False, "left", w=555)
    f.text(580, 438, T(tr("本講義設計的單組資料實驗，未重複抽樣；不代表所有資料皆有相同排名。", "One experiment designed for this lecture, drawn once; it does not mean every data set ranks the same.")), "MUT", 16, w=1140)
    return f
