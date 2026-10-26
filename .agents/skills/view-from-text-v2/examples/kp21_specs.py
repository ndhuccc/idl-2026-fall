"""KP21 (representer theorem and sparsity) -- ninth sample of view-from-text-v2. Plan: kp21_plan.json. Every number is recomputed here with kp21_nums.py
(same settings as the lecture; verified against it)."""
import os, sys
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools")); sys.path.insert(0, HERE)
from figkit.parts import T, NFig, header_chip, panel, chip, outline_box, takeaway, tint, solid, INK
import kp21_nums as N
from sklearn.svm import SVC


def tr(zh, en): return {"zh": zh, "en": en}


def hu(x, nd):   # half-up rounding to nd decimals
    return str(Decimal(repr(round(float(x), 9))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


X, Y = N.rings(60, 1); XT, YT = N.rings(500, 2)
M = SVC(C=10, kernel="precomputed").fit(N.gk(X, X, 0.5), Y); SV = set(int(i) for i in M.support_)
ACC = float(np.mean(M.predict(N.gk(XT, X, 0.5)) == YT))


def circ(f, cx, cy, r, color, sw=1.5): f.rect(cx - r, cy - r, 2 * r, 2 * r, stroke=color, sw=sw, radius=r)


def rings_panel(f, x0, y0, w, h, role, show):
    """show(i) -> 'ring' | 'dot' | 'dict'; draws the 120 training points."""
    panel(f, x0, y0, w, h, role); cx, cy, sc = x0 + w / 2, y0 + h / 2, 56
    for i in range(len(X)):
        px, py = cx + X[i, 0] * sc, cy - X[i, 1] * sc; k = show(i); col = solid("blue") if Y[i] < 0 else solid("orange")
        if k == "dot": f.ellipse(px, py, 6, "#8c857b")
        else:
            f.ellipse(px, py, 8, col)
            if k == "ring": circ(f, px, py, 9, "#7a8f86")
            if k == "dict": circ(f, px, py, 10, solid("green"), 3)


def sparse():
    f = NFig("lec05-kp21-sparse", 1160, 456, T(tr(f"同心環 120 個訓練點（標準差 0.25、每類 60 點、種子 1）。左：Parzen 展開每個點都是一個核（外面有圈）。右：SVM（高斯核 γ=0.5、C=10）只有 {len(SV)} 個支撐向量有圈，其餘 {120-len(SV)} 個灰點係數恰為 0；每類 500 點的新資料正確率 {hu(ACC*100,1)}%。稀疏解：展開式中大多數係數是 0。",
                                              f"120 training points on two concentric rings (standard deviation 0.25, 60 per class, seed 1). Left: in the Parzen expansion every point is a kernel (circled). Right: the SVM (Gaussian kernel γ=0.5, C=10) has only {len(SV)} support vectors circled and the other {120-len(SV)} grey points have coefficients exactly 0; accuracy on 500 new points per class is {hu(ACC*100,1)}%. Sparse solution: most coefficients in the expansion are 0.")))
    header_chip(f, 0, 0, 350, 34, "blue", T(tr("Parzen 展開：120 個核", "Parzen expansion: 120 kernels")), size=17); header_chip(f, 360, 0, 350, 34, "green", T(tr(f"SVM：{len(SV)} 個核", f"SVM: {len(SV)} kernels")), size=17)
    rings_panel(f, 0, 38, 350, 340, "blue", lambda i: "ring"); rings_panel(f, 360, 38, 350, 340, "green", lambda i: "ring" if i in SV else "dot")
    f.text(175, 392, T(tr("藍：內環　橘：外環　每個點都用到", "blue: inner ring, orange: outer ring; every point is used")), "TXT", 16, True, w=330); f.text(535, 392, T(tr(f"{120-len(SV)} 個灰點的係數恰為 0", f"the {120-len(SV)} grey points have coefficient exactly 0")), "TXT", 16, True, w=340)
    f.rect(730, 0, 430, 96, fill=tint("blue", 0.12), stroke=tint("blue", 0.55), sw=1.5, radius=12)
    f.text(746, 48, T(tr("稀疏解：展開式中大多數係數是 0，\n只有少數項真正用到。", "sparse solution: most coefficients in the expansion\nare 0 and only a few terms are really used.")), "TXT", 16, False, "left", w=400, name=T(tr("term:稀疏解", "term:Sparse solution")))
    f.rect(730, 108, 430, 160, fill=tint("green", 0.12), stroke=tint("green", 0.55), sw=1.5, radius=12)
    f.text(746, 188, T(tr("支撐向量：權重 $\\lambda_i>0$ 的訓練點。\nSVM 的解 $g(x)=\\sum_{i=1}^{N_s}a_iK(x,x_i)+w_0$，\n$a_i=\\lambda_iy_i$，$N_s$ 是支撐向量數。", "support vector: a training point with weight $\\lambda_i>0$.\nThe SVM solution is $g(x)=\\sum_{i=1}^{N_s}a_iK(x,x_i)+w_0$,\n$a_i=\\lambda_iy_i$, with $N_s$ the number of support vectors.")), "TXT", 16, False, "left", w=400, math=True, name=T(tr("term:支撐向量", "term:Support vector")))
    f.rect(730, 280, 430, 98, fill=tint("orange", 0.12), stroke=tint("orange", 0.55), sw=1.5, radius=12)
    f.text(746, 329, T(tr("與 Parzen 的兩個差別：\n1. $g$ 不是密度（積分不為 1）。\n2. 項數：Parzen 用全部 $N$ 個點，SVM 只用 $N_s$ 個。", "two differences from Parzen:\n1. $g$ is not a density (it does not integrate to 1).\n2. terms: Parzen uses all $N$ points, the SVM only $N_s$.")), "TXT", 16, False, "left", w=400, math=True)
    takeaway(f, 0, 410, 1160, 40, T(tr(f"同樣 120 個點：Parzen 用 120 個核，SVM 只用 {len(SV)} 個，新點正確率仍有 {hu(ACC*100,1)}%。", f"The same 120 points: Parzen uses 120 kernels, the SVM only {len(SV)}, and accuracy on new points is still {hu(ACC*100,1)}%.")))
    return f


def representer():
    f = NFig("lec05-kp21-rep", 1160, 452, T(tr("Representer 定理：對任意非負損失 L 與嚴格遞增的 Ω，正則化代價 Σ L(g(x_i),y_i)+Ω(‖g‖) 在 RKHS（核對應的函數空間，可能無窮維）裡的最小解，形如 g(x)=Σ_{j=1}^{N} a_j K(x,x_j)。右圖是示意：三個訓練點上的三個核（細線）加權相加成解（粗線）。搜尋範圍可能無窮維，答案卻只剩 N 個係數。",
                                              "Representer theorem: for any non-negative loss L and strictly increasing Ω, the regularised cost Σ L(g(x_i),y_i)+Ω(‖g‖) has a minimiser in the RKHS (the function space of the kernel, possibly infinite-dimensional) of the form g(x)=Σ_{j=1}^{N} a_j K(x,x_j). The right panel is a sketch: three kernels centred on three training points (thin lines) add up with weights to the solution (thick line). The search space may be infinite-dimensional, yet the answer has only N coefficients.")))
    header_chip(f, 0, 0, 520, 34, "blue", T(tr("要最小化的正則化代價", "The regularised cost to minimise")), size=17); panel(f, 0, 38, 520, 138, "blue")
    f.text(260, 88, "$\\sum_{i=1}^{N}\\mathcal{L}(g(x_i),y_i)+\\Omega(\\|g\\|)$", "TXT", 22, w=480, math=True)
    f.text(260, 142, T(tr("$\\mathcal{L}$：任意非負損失；$\\Omega$：嚴格遞增的函數", "$\\mathcal{L}$: any non-negative loss; $\\Omega$: a strictly increasing function")), "TXT", 16, w=490, math=True)
    f.rect(0, 190, 520, 100, fill=tint("orange", 0.12), stroke=tint("orange", 0.55), sw=1.5, radius=12)
    f.text(16, 240, T(tr("正則化項 $\\Omega(\\|g\\|)$：懲罰解的範數，\n讓解小而簡單。", "regularisation term $\\Omega(\\|g\\|)$: penalises the norm\nof the solution so that it stays small and simple.")), "TXT", 16, False, "left", w=490, math=True, name=T(tr("term:正則化項", "term:Regularisation term")))
    f.rect(0, 302, 520, 100, fill=tint("violet", 0.12), stroke=tint("violet", 0.55), sw=1.5, radius=12)
    f.text(16, 352, T(tr("RKHS：核函數對應的函數空間，\n搜尋範圍可能是無窮維。", "RKHS: the function space that the kernel defines;\nthe search range may be infinite-dimensional.")), "TXT", 16, False, "left", w=490, name=T(tr("term:RKHS", "term:RKHS")))
    chip(f, 530, 156, 100, 52, "yellow", T(tr("定理", "theorem")), size=18, bold=True, dark=True)
    f.text(580, 236, "→", "TXT", 30, True, w=60)
    header_chip(f, 640, 0, 520, 34, "green", T(tr("最小解的形狀（示意）", "The shape of the minimiser (sketch)")), size=17); panel(f, 640, 38, 520, 364, "green")
    f.text(900, 88, T(tr("Representer 定理：最小解形如\n$g(x)=\\sum_{j=1}^{N}a_jK(x,x_j)$", "Representer theorem: the minimiser has the form\n$g(x)=\\sum_{j=1}^{N}a_jK(x,x_j)$")), "TXT", 18, w=490, math=True, name=T(tr("term:Representer 定理", "term:Representer theorem")))
    ox, base, sx, sy = 670, 285, 80, 80; cs = [(1.2, 1.0, "blue"), (3.0, -0.6, "orange"), (4.8, 0.8, "violet")]
    gk = lambda x, c: np.exp(-(x - c) ** 2 / (2 * 0.7 ** 2)); xs = np.linspace(0, 5.9, 90)
    f.line(ox, base, ox + 5.9 * sx, base, "MUT", 2)
    for c, a, role in cs:
        for u, v in zip(xs, xs[1:]): f.line(ox + u * sx, base - a * gk(u, c) * sy, ox + v * sx, base - a * gk(v, c) * sy, solid(role), 2)
        f.ellipse(ox + c * sx, base, 12, solid(role))
    tot = lambda x: sum(a * gk(x, c) for c, a, _ in cs)
    for u, v in zip(xs, xs[1:]): f.line(ox + u * sx, base - tot(u) * sy, ox + v * sx, base - tot(v) * sy, solid("green"), 5)
    for k, (c, a, role) in enumerate(cs): f.text(ox + c * sx, base + 24, T(tr(f"$x_{k+1}$", f"$x_{k+1}$")), "TXT", 16, True, w=60, math=True)
    f.text(900, 340, T(tr("細線：$a_jK(x,x_j)$；粗線：相加得到 $g$", "thin: $a_jK(x,x_j)$; thick: their sum $g$")), "TXT", 16, w=480, math=True)
    chip(f, 660, 356, 480, 40, "yellow", T(tr("搜尋範圍可能無窮維，答案卻只剩 $N$ 個係數", "infinite-dimensional search, only $N$ coefficients")), size=16, bold=True, math=True, dark=True)
    takeaway(f, 0, 410, 1160, 38, T(tr("最佳解永遠只是放在 N 個訓練點上的 N 個核的線性組合。", "The best solution is always a linear combination of N kernels placed on the N training points.")))
    return f


def check6d():
    f = NFig("lec05-kp21-repcheck", 1160, 428, T(tr("數值驗證：二次核對應 6 維特徵 φ(x)=[1, √2·x₁, √2·x₂, x₁², √2·x₁x₂, x₂²]；30 個二維標準常態點（種子 3）、目標也取標準常態，C=0.7。路線 A 在 6 維解 w=(ΦᵀΦ+CI)⁻¹Φᵀy；路線 B 用核解 a=(𝒦+CI)⁻¹y。兩者算出的 w 與 Φᵀa 最大差、5 個新點預測的最大差都約 10⁻¹⁴，是捨入誤差；w=Σ a_j φ(x_j) 落在訓練點特徵向量張成的空間裡。",
                                              "Numerical check: the quadratic kernel corresponds to the 6-dimensional features φ(x)=[1, √2·x₁, √2·x₂, x₁², √2·x₁x₂, x₂²]; 30 two-dimensional standard-normal points (seed 3), standard-normal targets, C=0.7. Route A solves in 6 dimensions, w=(ΦᵀΦ+CI)⁻¹Φᵀy; route B solves with the kernel, a=(𝒦+CI)⁻¹y. The largest difference between w and Φᵀa, and between the predictions at 5 new points, are both about 10⁻¹⁴, rounding error; w=Σ a_j φ(x_j) lies in the space spanned by the training points' feature vectors.")))
    chip(f, 0, 0, 1160, 44, "grey", T(tr("設定：$K(x,z)=(x^Tz+1)^2$，$\\phi(x)=[1,\\sqrt{2}x_1,\\sqrt{2}x_2,x_1^2,\\sqrt{2}x_1x_2,x_2^2]$；30 個二維標準常態點，$C=0.7$", "setup: $K(x,z)=(x^Tz+1)^2$, $\\phi(x)=[1,\\sqrt{2}x_1,\\sqrt{2}x_2,x_1^2,\\sqrt{2}x_1x_2,x_2^2]$; 30 two-dimensional standard-normal points, $C=0.7$")), size=16, math=True)
    header_chip(f, 0, 58, 500, 34, "blue", T(tr("路線 A：在 6 維特徵空間解 w", "Route A: solve for w in the 6-dimensional space")), size=17); panel(f, 0, 96, 500, 120, "blue")
    f.text(250, 134, "$w=(\\Phi^T\\Phi+CI)^{-1}\\Phi^Ty$", "TXT", 22, w=470, math=True); f.text(250, 184, T(tr("要解 $6\\times6$ 的方程", "solves a $6\\times6$ system")), "TXT", 16, w=470, math=True)
    header_chip(f, 660, 58, 500, 34, "green", T(tr("路線 B：用核解 N 個係數 a", "Route B: solve for the N coefficients a with the kernel")), size=17); panel(f, 660, 96, 500, 120, "green")
    f.text(910, 134, "$a=(\\mathcal{K}+CI)^{-1}y$", "TXT", 22, w=470, math=True); f.text(910, 184, T(tr("要解 $30\\times30$ 的方程", "solves a $30\\times30$ system")), "TXT", 16, w=470, math=True)
    f.text(580, 156, "=", "TXT", 40, True, w=60); f.text(580, 196, "$w=\\Phi^Ta$", "TXT", 16, w=150, math=True)
    chip(f, 0, 230, 560, 52, "yellow", T(tr("$w$ 與 $\\Phi^Ta$ 最大差約 $10^{-14}$（捨入誤差）", "$w$ against $\\Phi^Ta$: largest difference about $10^{-14}$")), size=17, bold=True, math=True, dark=True)
    chip(f, 600, 230, 560, 52, "yellow", T(tr("5 個新點的預測最大差約 $10^{-14}$", "5 new points: largest prediction difference about $10^{-14}$")), size=17, bold=True, math=True, dark=True)
    f.rect(0, 294, 1160, 62, fill=tint("orange", 0.12), stroke=tint("orange", 0.55), sw=1.5, radius=12)
    f.text(16, 325, T(tr("Gram 矩陣 $\\mathcal{K}$：把 $K(x_i,x_j)$ 排成的 $N\\times N$ 方陣；$\\mathcal{K}=\\Phi\\Phi^T$。", "Gram matrix $\\mathcal{K}$: the $N\\times N$ matrix of the values $K(x_i,x_j)$; $\\mathcal{K}=\\Phi\\Phi^T$.")), "TXT", 16, False, "left", w=1120, math=True, name=T(tr("term:Gram 矩陣", "term:Gram matrix")))
    takeaway(f, 0, 372, 1160, 48, T(tr("兩條路答案相同：w 落在訓練點特徵向量張成的空間裡，這是 Representer 定理的一個特例。", "Same answer: w lies in the span of the training points' feature vectors, a Representer special case.")))
    return f


def hand():
    xs = np.linspace(-0.2, 1.3, 2)
    spec = {
        "id": "lec05-kp21-hand",
        "alt": tr("兩個訓練點 x=0、1，目標 +1、−1，核 K=xz+1。𝒦 的四個元素 1、1、1、2；C=0：a=(3,−2)，g(x)=3K(x,0)−2K(x,1)=1−2x，直線剛好穿過兩個點，g(0.5)=0；C=1：a=(0.8,−0.6)，g=0.2−0.6x，被壓向 0。",
                  "Two training points x=0, 1 with targets +1, −1 and kernel K=xz+1. The four entries of 𝒦 are 1, 1, 1, 2; C=0: a=(3,−2), g(x)=3K(x,0)−2K(x,1)=1−2x, the line passes exactly through both points and g(0.5)=0; C=1: a=(0.8,−0.6), g=0.2−0.6x, pulled toward 0."),
        "plot": {"x": [-0.3, 1.4], "y": [-1.4, 1.6], "x_ticks": [0, 0.5, 1], "y_ticks": [-1, 0, 1], "x_label": "x", "y_label": "g(x)", "w": 430, "h": 300},
        "series": [{"kind": "hline", "y": 0, "role": "grey", "width": 1}, {"kind": "seg", "a": (-0.2, 1.4), "b": (1.3, -1.6 + 0.0), "role": "blue", "width": 4}, {"kind": "seg", "a": (-0.2, 0.32), "b": (1.3, -0.58), "role": "red", "width": 4},
                   {"kind": "dots", "points": [(0, 1), (1, -1)], "role": "grey", "d": 16}, {"kind": "dots", "points": [(0.5, 0)], "role": "green", "d": 16}],
        "notes": [{"x": 0.2, "y": 1.45, "text": tr("C=0：g=1−2x", "C=0: g=1−2x"), "ha": "left", "w": 150}, {"x": 0.55, "y": 0.55, "text": tr("C=1：g=0.2−0.6x", "C=1: g=0.2−0.6x"), "ha": "left", "w": 190}, {"x": 0.08, "y": -0.35, "text": "g(0.5)=0", "ha": "left", "w": 110}],
        "side": [{"title": tr("核最小平方", "Kernel least squares"), "role": "blue", "name": tr("term:核最小平方", "term:Kernel least squares"), "h": 2.6, "text": tr("把 $g=\\sum_ja_jK(x,x_j)$ 代入平方誤差再求最小：\n1. $J(a)=(y-\\mathcal{K}a)^T(y-\\mathcal{K}a)$\n2. 梯度為零：$a=\\mathcal{K}^{-1}y$\n3. $g(x)=y^T\\mathcal{K}^{-1}p$，\n   $p=[K(x,x_1),\\dots,K(x,x_N)]^T$", "plug $g=\\sum_ja_jK(x,x_j)$ into the squared error and minimise:\n1. $J(a)=(y-\\mathcal{K}a)^T(y-\\mathcal{K}a)$\n2. zero gradient: $a=\\mathcal{K}^{-1}y$\n3. $g(x)=y^T\\mathcal{K}^{-1}p$,\n   $p=[K(x,x_1),\\dots,K(x,x_N)]^T$")},
                 {"title": tr("兩個點的例子", "A two-point example"), "role": "orange", "h": 1.1, "text": tr("$\\mathcal{K}$ 的元素 1、1、1、2，解得 $a=(3,-2)$：\n$g(x)=3K(x,0)-2K(x,1)=1-2x$。", "the entries of $\\mathcal{K}$ are 1, 1, 1, 2, giving $a=(3,-2)$:\n$g(x)=3K(x,0)-2K(x,1)=1-2x$.")},
                 {"title": tr("常見迷思", "Common myth"), "role": "red", "h": 1.0, "text": tr("定理保證解稀疏？不：$a$ 通常全非零。", "the theorem guarantees a sparse solution? No: $a$\nis usually non-zero everywhere.")}],
        "takeaway": tr("C=0 時 g 剛好穿過兩個點，C=1 時被壓向 0；兩個係數都非零。", "At C=0 g passes exactly through both points, at C=1 it is pulled toward 0; both coefficients are non-zero."),
    }
    return "plot_xy", spec


def _sinfit():
    x, y = N.sin9(); k = lambda A, B: np.exp(-(A[:, None] - B[None, :]) ** 2 / (2 * 0.8 ** 2)); K = k(x, x); xg = np.linspace(0, 2 * np.pi, 200)
    return x, y, K, xg, k(xg, x)


def sinfit():
    x, y, K, xg, Kg = _sinfit(); fit = lambda C: Kg @ np.linalg.solve(K + C * np.eye(9), y); st = {C: N.part_sin(C) for C in (0, 0.1, 10)}
    spec = {
        "id": "lec05-kp21-sin",
        "alt": tr(f"9 個點取自 y=sin x 加雜訊（標準差 0.2，種子 0），高斯核 σ=0.8，Gram 矩陣的條件數 {hu(st[0][4],0)}。C=0（核最小平方）曲線穿過每個點：訓練誤差 0、9 個 a_j 全非零（絕對值最大 {hu(st[0][3],2)}），對真正 sin x 的均方誤差 {hu(st[0][1],3)}；C=0.1 較平順，誤差 {hu(st[0.1][1],3)}；C=10 被壓平，誤差 {hu(st[10][1],3)}。",
                  f"9 points from y=sin x plus noise (standard deviation 0.2, seed 0), Gaussian kernel σ=0.8, Gram matrix condition number {hu(st[0][4],0)}. With C=0 (kernel least squares) the curve passes through every point: training error 0, all 9 a_j non-zero (largest absolute value {hu(st[0][3],2)}), mean squared error against the true sin x {hu(st[0][1],3)}; C=0.1 is smoother, error {hu(st[0.1][1],3)}; C=10 is pressed flat, error {hu(st[10][1],3)}."),
        "plot": {"x": [-0.2, 6.5], "y": [-1.7, 1.7], "x_ticks": [0, 1, 2, 3, 4, 5, 6], "y_ticks": [-1, 0, 1], "x_label": "x", "y_label": "y", "w": 430, "h": 300},
        "series": [{"kind": "line", "points": [(float(a), float(np.sin(a))) for a in xg], "role": "grey", "width": 5, "name": tr("真正的 sin x", "true sin x")},
                   {"kind": "line", "points": [(float(a), float(b)) for a, b in zip(xg, fit(0))], "role": "red", "width": 3, "name": "C=0"},
                   {"kind": "line", "points": [(float(a), float(b)) for a, b in zip(xg, fit(0.1))], "role": "green", "width": 3, "name": "C=0.1"},
                   {"kind": "line", "points": [(float(a), float(b)) for a, b in zip(xg, fit(10))], "role": "violet", "width": 3, "name": "C=10"},
                   {"kind": "dots", "points": [(float(a), float(b)) for a, b in zip(x, y)], "role": "blue", "d": 12}],
        "side": [{"title": tr("設定", "Setup"), "role": "orange", "name": tr("term:條件數", "term:Condition number"), "h": 1.3, "text": tr(f"9 個點、雜訊 0.2，高斯核 $\\sigma=0.8$。\n條件數：最大與最小特徵值的比，越大越不穩；\n$\\mathcal{{K}}$ 的條件數 {hu(st[0][4],0)}。", f"9 points, noise 0.2, Gaussian kernel $\\sigma=0.8$.\ncondition number: ratio of largest to smallest eigenvalue,\nlarger = less stable; that of $\\mathcal{{K}}$ is {hu(st[0][4],0)}.")},
                 {"title": tr("C=0：穿過每個點", "C=0: through every point"), "role": "red", "h": 1.3, "text": tr(f"訓練誤差 0；9 個 $a_j$ 全非零（最大 {hu(st[0][3],2)}）；\n對真正 $\\sin x$ 的均方誤差 {hu(st[0][1],3)}。", f"training error 0; all 9 $a_j$ non-zero (largest {hu(st[0][3],2)});\nmean squared error against the true $\\sin x$: {hu(st[0][1],3)}.")},
                 {"title": tr("C 加大", "Larger C"), "role": "green", "h": 1.0, "text": tr(f"$C=0.1$ 較平順：{hu(st[0.1][1],3)}；\n$C=10$ 被壓平：{hu(st[10][1],3)}。", f"$C=0.1$ is smoother: {hu(st[0.1][1],3)};\n$C=10$ is pressed flat: {hu(st[10][1],3)}.")}],
        "takeaway": tr(f"C=0 穿過每個雜訊點（誤差 {hu(st[0][1],3)}）；C=0.1 較貼近 sin x（{hu(st[0.1][1],3)}）。", f"C=0 passes through every noisy point (error {hu(st[0][1],3)}); C=0.1 follows sin x more closely ({hu(st[0.1][1],3)})."),
    }
    return "plot_xy", spec


def csweep():
    cs = (0, 0.01, 0.1, 1, 10, 100); st = [N.part_sin(C) for C in cs]; tr_e = [s[0] for s in st]; te = [s[1] for s in st]
    def lab(v): return hu(v, 4) if v < 0.01 else (hu(v, 3) if v < 0.1 else hu(v, 3))
    spec = {
        "id": "lec05-kp21-csweep",
        "alt": tr("同第 5 節的 9 個點，掃 C=0、0.01、0.1、1、10、100：訓練均方誤差 " + "、".join(lab(v) for v in tr_e) + "；對真正 sin x 的均方誤差 " + "、".join(lab(v) for v in te) + "。C 太小穿過雜訊、太大曲線被壓平，C=0.1 對 sin x 的誤差最小。每個 C 下 a_j 非零個數都是 9；同心環分類（核最小平方，C=0、0.001、0.1、1）也都是 120 個全非零。",
                  "The same 9 points as before, sweeping C=0, 0.01, 0.1, 1, 10, 100: training mean squared error " + ", ".join(lab(v) for v in tr_e) + "; mean squared error against the true sin x " + ", ".join(lab(v) for v in te) + ". Too small a C passes through the noise, too large flattens the curve, and C=0.1 has the smallest error against sin x. The number of non-zero a_j is 9 for every C; on the concentric rings (kernel least squares, C=0, 0.001, 0.1, 1) all 120 coefficients are also non-zero."),
        "plot": {"x": [0.5, 6.5], "y": [-0.12, 0.55], "x_ticks": [1, 2, 3, 4, 5, 6], "x_names": {str(i + 1): str(c) for i, c in enumerate(cs)}, "y_ticks": [0, 0.1, 0.2, 0.3, 0.4, 0.5], "x_label": "C", "y_label": tr("均方誤差", "mean squared error"), "w": 430, "h": 300, "legend_pos": "left"},
        "series": [{"kind": "line", "points": [(i + 1, v) for i, v in enumerate(tr_e)], "role": "blue", "width": 4, "markers": True, "labels": [lab(v) for v in tr_e[:3]] + [None] * 3, "label_dy": 24, "name": tr("訓練誤差", "training error")},
                   {"kind": "line", "points": [(i + 1, v) for i, v in enumerate(te)], "role": "red", "width": 4, "markers": True, "labels": [lab(v) for v in te], "label_dy": -22, "name": tr("對 sin x 的誤差", "error against sin x")}],
        "side": [{"title": tr("加上 C·I", "Adding C·I"), "role": "blue", "h": 1.25, "text": tr("$a=(\\mathcal{K}+CI)^{-1}y$：$C=0$ 就是核最小平方；\n$C$ 越大解的範數越被壓小，\n$CI$ 也讓矩陣更好求逆。", "$a=(\\mathcal{K}+CI)^{-1}y$: $C=0$ is kernel least squares;\nthe larger $C$ the smaller the norm of the solution,\nand $CI$ also makes the matrix easier to invert.")},
                 {"title": tr("C 的取捨", "Choosing C"), "role": "green", "h": 1.5, "text": tr("太小穿過雜訊，太大曲線被壓平；$C=0.1$ 最小。\n$C\\geq1$ 兩條線幾乎重合：訓練 0.078、0.349、\n0.465，對 $\\sin x$ 0.079、0.355、0.479。", "too small passes through the noise, too large flattens; $C=0.1$ is lowest.\nFor $C\\geq1$ the lines nearly coincide: training 0.078, 0.349,\n0.465 against $\\sin x$ 0.079, 0.355, 0.479.")},
                 {"title": tr("仍不稀疏", "Still not sparse"), "role": "red", "h": 1.2, "text": tr("每個 $C$ 下 9 個 $a_j$ 都非零；\n同心環（$C=0,0.001,0.1,1$）\n也都是 120 個全非零。", "all 9 $a_j$ are non-zero for every $C$;\non the concentric rings ($C=0,0.001,0.1,1$)\nall 120 are non-zero too.")}],
        "takeaway": tr("C=0.1 對 sin x 的誤差最小（0.033）；加 CI 不產生稀疏。", "C=0.1 has the smallest error against sin x (0.033); adding CI does not create sparsity."),
    }
    return "plot_xy", spec


def svcount():
    Cs = (0.01, 1, 10, 1000); gs = (0.01, 0.5, 50); R = {g: [N.svc(X, Y, XT, YT, C, g) for C in Cs] for g in gs}
    spec = {
        "id": "lec05-kp21-svcount",
        "alt": tr("同心環 120 個訓練點，高斯核，支撐向量數對 C（0.01、1、10、1000）：γ=0.01：" + "、".join(str(r[0]) for r in R[0.01]) + "；γ=0.5：" + "、".join(str(r[0]) for r in R[0.5]) + "；γ=50：" + "、".join(str(r[0]) for r in R[50]) + "。γ=0.5 隨 C 增大一路降到 13，γ=50 始終貼著 116，γ=0.01 要 C 大到 1000 才稀疏。新點正確率：γ=0.5、C=10 為 " + hu(R[0.5][2][1] * 100, 1) + "%，γ=0.01、C=1000 為 " + hu(R[0.01][3][1] * 100, 1) + "%。",
                  "120 concentric-ring training points, Gaussian kernel, number of support vectors against C (0.01, 1, 10, 1000): γ=0.01: " + ", ".join(str(r[0]) for r in R[0.01]) + "; γ=0.5: " + ", ".join(str(r[0]) for r in R[0.5]) + "; γ=50: " + ", ".join(str(r[0]) for r in R[50]) + ". For γ=0.5 it drops steadily to 13 as C grows, for γ=50 it stays at about 116, and γ=0.01 needs C as large as 1000 to become sparse. Accuracy on new points: " + hu(R[0.5][2][1] * 100, 1) + "% for γ=0.5, C=10 and " + hu(R[0.01][3][1] * 100, 1) + "% for γ=0.01, C=1000."),
        "plot": {"x": [0.5, 4.5], "y": [0, 175], "x_ticks": [1, 2, 3, 4], "x_names": {str(i + 1): str(c) for i, c in enumerate(Cs)}, "y_ticks": [0, 40, 80, 120, 160], "x_label": tr("SVM 的 C（違規代價）", "SVM's C (violation cost)"), "y_label": tr("支撐向量數", "number of support vectors"), "w": 430, "h": 300},
        "series": [{"kind": "line", "points": [(i + 1, R[0.01][i][0]) for i in range(4)], "role": "blue", "width": 4, "markers": True, "labels": [None, None, None, str(R[0.01][3][0])], "label_dy": -20, "name": "γ=0.01"},
                   {"kind": "line", "points": [(i + 1, R[50][i][0]) for i in range(4)], "role": "red", "width": 4, "markers": True, "labels": [None, None, None, str(R[50][3][0])], "label_dy": 22, "name": "γ=50"},
                   {"kind": "line", "points": [(i + 1, R[0.5][i][0]) for i in range(4)], "role": "green", "width": 4, "markers": True, "labels": [None] + [str(R[0.5][i][0]) for i in (1, 2, 3)], "label_dy": 22, "name": "γ=0.5"}],
        "side": [{"title": tr("稀疏從哪裡來", "Where sparsity comes from"), "role": "blue", "h": 1.8, "text": tr("SVM 的代價 $J=\\frac{1}{2}\\|w\\|^2+C\\sum_iI(\\xi_i)$ 中，\n$\\|w\\|^2$ 是正則化項，隱含地\n把大多數 $\\lambda_i$ 逼成 0。", "in the SVM cost $J=\\frac{1}{2}\\|w\\|^2+C\\sum_iI(\\xi_i)$,\n$\\|w\\|^2$ is the regularisation term and implicitly\npushes most $\\lambda_i$ to 0.")},
                 {"title": tr("C 的方向相反", "C points the other way"), "role": "orange", "h": 1.1, "text": tr("這裡的 $C$ 是違規代價，越大越不正則；\n與上一頁加在 $\\mathcal{K}$ 對角線上的 $C$ 相反。", "here $C$ is the violation cost, larger = less regularised;\nopposite to the $C$ added to the diagonal of $\\mathcal{K}$.")},
                 {"title": tr("不是自動的", "Not automatic"), "role": "red", "h": 1.2, "text": tr("$\\gamma=50$：$C$ 再大也有 116 個；\n$\\gamma=0.01$：要 $C=1000$ 才稀疏（19 個）。", "$\\gamma=50$: even a huge $C$ leaves 116;\n$\\gamma=0.01$: needs $C=1000$ to become sparse (19).")}],
        "takeaway": tr("稀疏來自正則化項，但不是自動的：取決於 C 與 γ。", "Sparsity comes from the regularisation term but is not automatic: it depends on C and γ."),
    }
    return "plot_xy", spec


def dictm():
    ths = (0.99, 0.9, 0.7, 0.5); order = np.random.default_rng(0).permutation(120); D = []
    def build(th):
        d = []
        for i in order:
            if not d or N.gk(X[i:i + 1], X[d], 0.5).max() <= th: d.append(int(i))
        return d
    sizes = [len(build(t)) for t in ths]; D = set(build(0.7))
    f = NFig("lec05-kp21-dict", 1160, 412, T(tr(f"字典法：同心環 120 點、γ=0.5；點依隨機順序（種子 0）逐一進來，新點與字典中最近者的核值高於門檻就不加入。門檻 0.7 留下 {len(D)} 個點（綠圈）。門檻 0.99、0.9、0.7、0.5 的字典大小 " + "、".join(map(str, sizes)) + "。四條稀疏化路線：RVM、線上自適應、投影法、字典法。",
                                              f"Dictionary method: 120 concentric-ring points, γ=0.5; the points arrive one by one in random order (seed 0) and a new point is not added when its kernel value to the nearest dictionary member is above the threshold. Threshold 0.7 keeps {len(D)} points (green circles). For thresholds 0.99, 0.9, 0.7, 0.5 the dictionary size is " + ", ".join(map(str, sizes)) + ". Four sparsification routes: RVM, online adaptive, projection, dictionary.")))
    header_chip(f, 0, 0, 330, 34, "green", T(tr(f"門檻 0.7：字典留下 {len(D)} 個點", f"Threshold 0.7: {len(D)} points kept")), size=16); rings_panel(f, 0, 38, 330, 300, "green", lambda i: "dict" if i in D else "dot")
    header_chip(f, 350, 0, 360, 34, "blue", T(tr("門檻越低，字典越小", "Lower threshold, smaller dictionary")), size=16); panel(f, 350, 38, 360, 300, "blue")
    for k, (t, n) in enumerate(zip(ths, sizes)):
        y = 70 + 62 * k; f.text(400, y + 20, f"{t:g}", "TXT", 18, True, "right", w=60); f.rect(420, y, max(n / 90 * 230, 4), 40, fill=solid("blue"), radius=6); f.text(420 + n / 90 * 230 + 10, y + 20, str(n), "TXT", 18, True, "left", w=60)
    f.text(530, 322, T(tr("橫軸：字典大小（點數）；左側為門檻", "bar: dictionary size (points); left: threshold")), "MUT", 16, w=330)
    routes = [("blue", tr("RVM", "RVM"), tr("貝氏先驗，比 SVM 更稀疏", "Bayesian prior, sparser than the SVM")), ("orange", tr("線上自適應", "Online adaptive"), tr("核化 LMS，範數約束", "kernel LMS with a norm constraint")), ("violet", tr("投影法", "Projection"), tr("超球約束，等價遺忘因子", "hypersphere constraint, equal to a forgetting factor")), ("green", tr("字典法", "Dictionary method"), tr("新點與字典的相似度低於門檻才加入", "a new point joins only if its similarity to the dictionary is below the threshold"))]
    for k, (role, nm, tx) in enumerate(routes):
        y = 0 + 86 * k; f.rect(730, y, 430, 78, fill=tint(role, 0.12), stroke=tint(role, 0.55), sw=1.5, radius=12)
        f.text(746, y + 22, T(nm), "TXT", 17, True, "left", w=400); f.text(746, y + 52, T(tx), "TXT", 16, False, "left", w=400, name=(T(tr("term:字典法", "term:Dictionary method")) if k == 3 else None))
    takeaway(f, 0, 356, 1160, 48, T(tr("怎麼記：Representer 定理把無窮維搜尋化成 N 個係數；SVM 靠正則化項讓多數為零，字典法靠門檻。", "Representer gives N coefficients; the SVM zeros most via regularisation, the dictionary via a threshold.")))
    return f


def extra():
    cells = [(0.01, 1000), (0.5, 1000), (50, 1000), (0.5, 1)]; rr = [N.svc(X, Y, XT, YT, C, g) for g, C in cells]; roles = ["blue", "green", "red", "orange"]
    spec = {
        "id": "lec05-kp21-extra",
        "alt": tr("同心環、高斯核：C=1000 時 γ=0.01、0.5、50 的支撐向量數 " + "、".join(str(r[0]) for r in rr[:3]) + "，新點正確率 " + "、".join(hu(r[1] * 100, 1) + "%" for r in rr[:3]) + "；γ=0.5、C=1 有 " + str(rr[3][0]) + " 個、" + hu(rr[3][1] * 100, 1) + "%。支撐向量太多時泛化退化，但少不保證好：39 個的結果比 13 個好。單次實驗，差距可能含雜訊。",
                  "Concentric rings, Gaussian kernel: for C=1000 the number of support vectors at γ=0.01, 0.5, 50 is " + ", ".join(str(r[0]) for r in rr[:3]) + " and the accuracy on new points " + ", ".join(hu(r[1] * 100, 1) + "%" for r in rr[:3]) + "; γ=0.5, C=1 has " + str(rr[3][0]) + " and " + hu(rr[3][1] * 100, 1) + "%. Too many support vectors hurt generalisation, but fewer is not always better: 39 does better than 13. One experiment, the gaps may contain noise."),
        "unit": tr("新點正確率（%）", "accuracy on new points (%)"),
        "bars": [{"label": tr(f"γ={g:g}、C={C:g}", f"γ={g:g}, C={C:g}"), "value": float(hu(r[1] * 100, 1)), "role": role, "note": T(tr(f"支撐向量 {r[0]}", f"support vectors {r[0]}"))} for (g, C), r, role in zip(cells, rr, roles)],
        "min": 80, "max": 100, "ticks": [80, 85, 90, 95, 100],
        "takeaway": tr("支撐向量太多（116）泛化退化，但少不保證好：39 個（97.3%）勝過 13 個（95.4%）。", "Too many support vectors (116) hurt generalisation, but fewer is not always better: 39 (97.3%) beats 13 (95.4%)."),
    }
    return "bars_compare", spec


def solution():
    spec = {
        "id": "lec05-kp21-solution",
        "alt": tr("解答：𝒦+I 的四個元素 2、1、1、3，行列式 2×3−1×1=5，反矩陣是 1/5 乘（第一列 3、−1，第二列 −1、2）。a=1/5·(3·1+(−1)(−1), −1·1+2·(−1))=1/5·(4,−3)=(0.8,−0.6)。g(0)=0.8·1−0.6·1=0.2；g(1)=0.8·1−0.6·2=−0.4。C=0 時是 +1、−1，C=1 縮成 0.2、−0.4。",
                  "Solution: the entries of 𝒦+I are 2, 1, 1, 3, the determinant is 2×3−1×1=5 and the inverse is 1/5 times (first row 3, −1; second row −1, 2). a=1/5·(3·1+(−1)(−1), −1·1+2·(−1))=1/5·(4,−3)=(0.8,−0.6). g(0)=0.8·1−0.6·1=0.2; g(1)=0.8·1−0.6·2=−0.4. At C=0 they are +1, −1, at C=1 they shrink to 0.2, −0.4."),
        "steps": [{"role": "blue", "title": tr("① 反矩陣", "① The inverse"), "lines": [tr("$\\mathcal{K}+I$ 的元素：2、1、1、3", "entries of $\\mathcal{K}+I$: 2, 1, 1, 3"), tr("行列式 $2\\times3-1\\times1=5$", "determinant $2\\times3-1\\times1=5$"), tr("反矩陣 $\\frac{1}{5}\\times$（3、-1；-1、2）", "inverse $\\frac{1}{5}\\times$ (3, -1; -1, 2)")], "result": tr("行列式 5", "determinant 5")},
                  {"role": "green", "title": tr("② 解 a", "② Solve for a"), "lines": ["$a=\\frac{1}{5}(3\\cdot1+(-1)(-1),$", "$\\ -1\\cdot1+2\\cdot(-1))$", "$=\\frac{1}{5}(4,-3)$"], "result": "(0.8, −0.6)"},
                  {"role": "orange", "title": tr("③ 算 g", "③ Evaluate g"), "lines": ["$g(0)=0.8\\cdot1-0.6\\cdot1$", "$g(1)=0.8\\cdot1-0.6\\cdot2$", tr("$C=0$ 時是 $+1$、$-1$", "at $C=0$ they were $+1$, $-1$")], "result": "0.2；−0.4"}],
        "takeaway": tr("C=0 時是 +1、-1，C=1 縮成 0.2、-0.4：C 把解壓向 0，兩個係數仍都非零。", "C pulls the solution toward 0 (+1, -1 become 0.2, -0.4); both coefficients stay non-zero."),
    }
    return "steps_flow", spec
