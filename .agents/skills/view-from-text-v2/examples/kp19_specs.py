"""KP19 (nonlinear SVM and the Mercer kernel trick) -- eleventh sample of view-from-text-v2. Plan: kp19_plan.json. Every number is recomputed here with kp19_nums.py
(same settings as the lecture; verified against it)."""
import os, sys
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
from math import comb
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools")); sys.path.insert(0, HERE)
from figkit.parts import T, NFig, header_chip, panel, chip, takeaway, tint, solid, INK
import kp19_nums as N
from kp21_nums import rings, gk
from kp20_nums import lin, quad
from kp20_specs import card


def cbox(f, x, y, w, h, role, text, size=16, bold=False, math=False, dark=False):
    """A rounded box whose text is a separate node (multi-line safe)."""
    if dark: return chip(f, x, y, w, h, role, text, size=size, bold=bold, math=math, dark=True)
    f.rect(x, y, w, h, fill=tint(role, 0.28), radius=10)
    f.text(x + w / 2, y + h / 2, text, "TXT", size, bold, w=w - 16, math=math)


def tr(zh, en): return {"zh": zh, "en": en}


def hu(x, nd):
    return str(Decimal(repr(round(float(x), 9))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def circ(f, cx, cy, r, color, sw=1.5): f.rect(cx - r, cy - r, 2 * r, 2 * r, stroke=color, sw=sw, radius=r)


def inner():
    f = NFig("lec05-kp19-inner", 1160, 424, T(tr("SVM 的分類規則只用內積：g(x)=Σ_{i=1}^{N_s} λ_i y_i x_iᵀx + w₀，g(x)>0 判為一類，否則判為另一類。新點 x 的處理順序：先對每個支撐向量算內積 x_iᵀx，再乘權重 λ_i y_i、加總、加 w₀，最後看 g 的正負；資料只在「內積」那一步進來。換到 k 維映射空間，只要把 x_i 換成 φ(x_i)，內積換成 φ(x_i)ᵀφ(x)；天真的擔心是 k 遠大於 l 時複雜度會爆炸。",
                                              "The SVM decision rule uses only inner products: g(x)=Σ_{i=1}^{N_s} λ_i y_i x_iᵀx + w₀, class one if g(x)>0 and the other class otherwise. A new point x is processed in this order: take the inner product x_iᵀx with every support vector, multiply by the weight λ_i y_i, sum, add w₀ and look at the sign of g; the data enter only at the inner-product step. In a k-dimensional mapped space just replace x_i by φ(x_i) and the inner product by φ(x_i)ᵀφ(x); the naive worry is that complexity explodes when k is much larger than l.")))
    card(f, 0, 0, 560, 70, "blue", T(tr("內積：兩個向量對應位置相乘再相加，\n$x^Tz=x_1z_1+x_2z_2$；大小反映兩者「有多像」。", "inner product: multiply the matching entries and add,\n$x^Tz=x_1z_1+x_2z_2$; its size says how alike the two are.")), name=T(tr("term:內積", "term:Inner product")))
    card(f, 600, 0, 560, 70, "green", T(tr("支撐向量：訓練後權重 $\\lambda_i>0$ 的訓練點；\n分類規則只靠它們，數量通常遠少於全部訓練點。", "support vector: a training point with weight $\\lambda_i>0$ after training;\nthe rule depends only on them, usually far fewer than all points.")), name=T(tr("term:支撐向量", "term:Support vector")))
    steps = [("blue", tr("新點 $x$", "new point $x$")), ("yellow", tr("內積 $x_i^Tx$", "inner product $x_i^Tx$")), ("orange", tr("乘權重 $\\lambda_iy_i$", "times weight $\\lambda_iy_i$")), ("orange", tr("加總再加 $w_0$", "sum, add $w_0$")), ("green", tr("看 $g(x)$ 的正負", "sign of $g(x)$"))]
    w = (1160 - 4 * 36) / 5
    for k, (role, tx) in enumerate(steps):
        x = k * (w + 36); cbox(f, x, 92, w, 56, role, T(tx), size=17, bold=True, math=True, dark=(role == "yellow"))
        if k < 4: f.text(x + w + 18, 120, "→", "TXT", 26, True, w=36)
    f.text(1 * (w + 36) + w / 2, 168, T(tr("只有這一步碰到資料", "the only step that touches the data")), "TXT", 16, True, w=w + 30)
    f.rect(0, 190, 1160, 70, fill=tint("blue", 0.14), radius=12)
    f.text(580, 225, "$g(x)=\\sum_{i=1}^{N_s}\\lambda_iy_i\\,x_i^Tx+w_0$", "TXT", 22, w=700, math=True)
    cbox(f, 0, 276, 560, 52, "violet", T(tr("映射到 $k$ 維：$x_i\\to\\phi(x_i)$，內積變 $\\phi(x_i)^T\\phi(x)$", "in $k$ dimensions: $x_i\\to\\phi(x_i)$, inner product $\\phi(x_i)^T\\phi(x)$")), size=17, bold=True, math=True)
    cbox(f, 600, 276, 560, 52, "red", T(tr("$k\\gg l$：複雜度會爆炸嗎？下一頁回答", "$k\\gg l$: does complexity explode? Next page")), size=17, bold=True, math=True)
    takeaway(f, 0, 346, 1160, 52, T(tr("SVM 的分類規則裡，資料只透過內積出現：把內積換掉，規則就換了空間。", "The data enter the SVM rule only through inner products: swap them and the space changes.")))
    return f


def quadmap():
    f = NFig("lec05-kp19-identity", 1160, 440, T(tr("二次映射：二維的 x 映射成三維 φ(x)=[x₁², √2·x₁x₂, x₂²]ᵀ，則 φ(x)ᵀφ(z)=(xᵀz)²。驗算 x=(1,2)、z=(2,1)：路線 A 先映射，φ(x)=(1, 2.828, 4)、φ(z)=(4, 2.828, 1)，內積 1×4+2.828×2.828+4×1=4+8+4=16；路線 B 用核，xᵀz=1·2+2·1=4，4²=16；兩者相同。核函數 K(x,z)：直接給出映射後內積的函數。",
                                              "The quadratic map: the two-dimensional x is mapped to the three-dimensional φ(x)=[x₁², √2·x₁x₂, x₂²]ᵀ, so φ(x)ᵀφ(z)=(xᵀz)². Check with x=(1,2), z=(2,1): route A maps first, φ(x)=(1, 2.828, 4), φ(z)=(4, 2.828, 1), inner product 1×4+2.828×2.828+4×1=4+8+4=16; route B uses the kernel, xᵀz=1·2+2·1=4, 4²=16; both agree. The kernel function K(x,z) gives the inner product after the mapping directly.")))
    card(f, 0, 0, 1160, 60, "orange", T(tr("核函數 $K(x,z)$：不必先算映射，就直接給出映射後的內積 $\\phi(x)^T\\phi(z)$ 的函數。", "kernel function $K(x,z)$: the function that gives the inner product $\\phi(x)^T\\phi(z)$ after the mapping directly, without computing the mapping first.")), name=T(tr("term:核函數", "term:Kernel function")))
    cbox(f, 0, 74, 1160, 46, "grey", T(tr("例子：$x=(1,2)$、$z=(2,1)$；$\\phi(x)=[x_1^2,\\ \\sqrt{2}x_1x_2,\\ x_2^2]^T$", "example: $x=(1,2)$, $z=(2,1)$; $\\phi(x)=[x_1^2,\\ \\sqrt{2}x_1x_2,\\ x_2^2]^T$")), size=17, math=True)
    header_chip(f, 0, 134, 540, 34, "blue", T(tr("路線 A：先映射，再算內積", "Route A: map first, then the inner product")), size=17); panel(f, 0, 172, 540, 130, "blue")
    f.text(270, 206, "$\\phi(x)=(1,\\ 2.828,\\ 4)$，$\\phi(z)=(4,\\ 2.828,\\ 1)$", "TXT", 18, w=510, math=True); f.text(270, 250, "$1\\times4+2.828\\times2.828+4\\times1$", "TXT", 18, w=510, math=True); f.text(270, 284, "$=4+8+4$", "TXT", 18, w=510, math=True)
    header_chip(f, 620, 134, 540, 34, "green", T(tr("路線 B：用核，直接算", "Route B: use the kernel directly")), size=17); panel(f, 620, 172, 540, 130, "green")
    f.text(890, 210, "$x^Tz=1\\cdot2+2\\cdot1=4$", "TXT", 18, w=510, math=True); f.text(890, 256, "$K(x,z)=(x^Tz)^2=4^2$", "TXT", 18, w=510, math=True)
    f.text(580, 232, "=", "TXT", 40, True, w=60)
    cbox(f, 0, 318, 540, 44, "yellow", "16", size=24, bold=True, dark=True); cbox(f, 620, 318, 540, 44, "yellow", "16", size=24, bold=True, dark=True)
    takeaway(f, 0, 378, 1160, 48, T(tr("高維空間的內積，用原空間的內積再平方就算得出來：不必真的造出三維向量。", "A high-dimensional inner product is the original inner product squared: no 3-D vectors needed.")))
    return f


def lift():
    Xt, yt = rings(500, 2); idx = np.r_[0:100, 500:600]; P = Xt[idx]; Y = yt[idx]; y1, y3 = P[:, 0] ** 2, P[:, 1] ** 2; pred = np.where(y1 + y3 > 2.5, 1, -1); ok = int((pred == Y).sum())
    f = NFig("lec05-kp19-lift", 1160, 430, T(tr(f"同心環升維（內環標籤 −1 平均半徑 1、外環 +1 平均半徑 2，標準差 0.25；訓練每類 60 點種子 1，新資料每類 500 點種子 2）。映射把半徑變成座標：y₁=x₁²、y₃=x₂²，y₁+y₃=‖x‖²，內環約 1、外環約 4；手動取中間的 2.5 畫線 y₁+y₃=2.5，這條線沒有經過訓練。新資料每類前 100 點共 200 點，這條線判對 {ok} 點，{hu(ok/2,1)}%；錯的是兩環邊緣重疊處。三維裡的平面回到原空間是一個圓；右圖沒有畫中間的座標 √2·x₁x₂。",
                                              f"Concentric-ring lift (inner ring label −1 with mean radius 1, outer ring +1 with mean radius 2, standard deviation 0.25; 60 training points per class with seed 1, 500 new points per class with seed 2). The map turns the radius into coordinates: y₁=x₁², y₃=x₂², y₁+y₃=‖x‖², about 1 for the inner ring and about 4 for the outer; the line y₁+y₃=2.5 at the midpoint was chosen by hand and never trained. On the first 100 new points of each class, 200 in all, it gets {ok} right, {hu(ok/2,1)}%; the mistakes are where the rings overlap at their edges. A plane in three dimensions is a circle back in the original space; the right panel omits the middle coordinate √2·x₁x₂.")))
    header_chip(f, 0, 0, 330, 34, "blue", T(tr("原空間：沒有直線能分開", "Original space: no line separates")), size=16); panel(f, 0, 38, 330, 330, "blue")
    cx, cy, sc = 165, 203, 50
    for p, s in zip(P, Y):
        (f.ellipse(cx + p[0] * sc, cy - p[1] * sc, 8, solid("blue")) if s < 0 else f.square(cx + p[0] * sc, cy - p[1] * sc, 8, solid("orange")))
    header_chip(f, 350, 0, 330, 34, "green", T(tr("映射後：一條線就分開", "After the map: one line separates")), size=16); panel(f, 350, 38, 330, 330, "green")
    ox, oy, s2 = 372, 346, 38
    f.line(ox, oy, ox + 300, oy, "MUT", 2); f.line(ox, oy, ox, oy - 295, "MUT", 2)
    f.text(ox + 290, oy + 16, "$y_1$", "TXT", 16, w=40, math=True); f.text(ox + 16, oy - 292, "$y_3$", "TXT", 16, w=40, math=True)
    for a, b, s, g in zip(y1, y3, Y, pred):
        if a > 7.6 or b > 7.6: continue
        px, py = ox + a * s2, oy - b * s2
        (f.ellipse(px, py, 8, solid("blue")) if s < 0 else f.square(px, py, 8, solid("orange")))
        if g != s: circ(f, px, py, 9, solid("red"), 2)
    f.line(ox, oy - 2.5 * s2, ox + 2.5 * s2, oy, solid("red"), 4); f.text(ox + 2.6 * s2 + 10, oy - 1.4 * s2, "$y_1+y_3=2.5$", "TXT", 16, True, "left", w=130, math=True)
    card(f, 700, 0, 460, 124, "orange", T(tr("映射把半徑變成座標：令 $y_1=x_1^2$、$y_3=x_2^2$，\n則 $y_1+y_3=\\|x\\|^2$。內環約 1、外環約 4，\n手動取中間的 2.5：這條線沒有經過訓練。", "the map turns the radius into a coordinate:\n$y_1=x_1^2$, $y_3=x_2^2$, so $y_1+y_3=\\|x\\|^2$.\nInner ring about 1, outer about 4; 2.5 was\npicked by hand and never trained.")))
    card(f, 700, 134, 460, 92, "yellow", T(tr(f"結果：新資料每類前 100 點，共 200 點，\n這條線判對 {ok} 點，即 {hu(ok/2,1)}%；\n紅圈是錯的，都在兩環邊緣重疊處。", f"first 100 new points per class, 200 in all:\n{ok} right, {hu(ok/2,1)}%; red circles are the\nmistakes, at the overlap of the rings.")))
    card(f, 700, 236, 460, 132, "grey", T(tr("設定：內環 $-1$ 平均半徑 1、外環 $+1$ 平均半徑 2，\n標準差 0.25；訓練每類 60 點（種子 1），\n新資料每類 500 點（種子 2）。\n三維的平面回到原空間是圓；右圖省略 $\\sqrt{2}x_1x_2$。", "setup: inner ring $-1$, mean radius 1; outer $+1$, 2;\nstandard deviation 0.25; 60 training points per class\n(seed 1), 500 new per class (seed 2). A 3-D plane is\na circle back here; $\\sqrt{2}x_1x_2$ is omitted.")))
    takeaway(f, 0, 380, 1160, 44, T(tr(f"原空間分不開的同心環，映射後一條線就切開：{ok}／200（{hu(ok/2,1)}%）。", f"Rings no line could separate in the original space are cut by one line after the map: {ok}/200 ({hu(ok/2,1)}%).")))
    return f


def mercer():
    f = NFig("lec05-kp19-mercer", 1160, 440, T(tr("Mercer 定理：什麼樣的函數可以當核？Mercer 條件：K 對稱 K(x,z)=K(z,x)、連續，而且對任何平方可積的函數 g，∬K(x,z)g(x)g(z)dxdz≥0。定理：滿足條件的 K，必有向量空間 H（RKHS）和映射 φ，使 K(x,z)=⟨φ(x),φ(z)⟩；反過來內積形式的核也滿足條件。定理沒告訴我們 φ 長什麼樣、H 有幾維（高斯核對應無窮維）。有限點檢查：任意有限個點的 Gram 矩陣 G_ij=K(x_i,x_j) 必須半正定（特徵值都≥0）：二次核 K=(xᵀz+1)² 的兩點 e₁=(1,0)、e₂=(0,1) 得 G=[[4,1],[1,4]]，特徵值 3、5；30 個隨機點（種子 0）高斯核最小特徵值 8.8×10⁻⁸，二次核 −8×10⁻¹⁵（捨入誤差，視為 0）。找到負特徵值就證明不合法；沒找到只代表這組點沒問題。",
                                              "The Mercer theorem: which functions can serve as kernels? Mercer condition: K is symmetric, K(x,z)=K(z,x), continuous, and for every square-integrable g, ∬K(x,z)g(x)g(z)dxdz≥0. Theorem: a K satisfying it has a vector space H (an RKHS) and a map φ with K(x,z)=⟨φ(x),φ(z)⟩; conversely every inner-product kernel satisfies the condition. The theorem does not tell us what φ looks like or how many dimensions H has (the Gaussian kernel's is infinite). Finite-point check: the Gram matrix G_ij=K(x_i,x_j) of any finite set of points must be positive semi-definite (eigenvalues ≥0): the quadratic kernel K=(xᵀz+1)² on e₁=(1,0), e₂=(0,1) gives G=[[4,1],[1,4]] with eigenvalues 3, 5; for 30 random points (seed 0) the smallest eigenvalue is 8.8×10⁻⁸ for the Gaussian kernel and −8×10⁻¹⁵ for the quadratic (rounding error, treated as 0). A negative eigenvalue proves the kernel illegal; finding none only says this set of points is fine.")))
    cw = 360; xs = [0, 400, 800]
    heads = [("blue", tr("① Mercer 條件", "① The Mercer condition")), ("orange", tr("② 定理與它沒說的", "② The theorem and what it omits")), ("green", tr("③ 有限點的檢查", "③ A finite-point check"))]
    for x, (role, hd) in zip(xs, heads): header_chip(f, x, 0, cw, 34, role, T(hd), size=17); panel(f, x, 38, cw, 292, role)
    f.text(180, 98, T(tr("核 $K$ 要同時滿足：\n• 對稱：$K(x,z)=K(z,x)$\n• 連續\n• 對任何平方可積的 $g$：", "$K$ must satisfy all of:\n• symmetric: $K(x,z)=K(z,x)$\n• continuous\n• for every square-integrable $g$:")), "TXT", 16, w=335, math=True)
    f.text(180, 178, "$\\int\\int K(x,z)g(x)g(z)\\,dx\\,dz\\geq0$", "TXT", 18, w=335, math=True)
    f.text(180, 262, T(tr("Mercer 條件：以上三條；最後一條是\n「半正定」在連續情形的寫法。", "Mercer condition: the three above; the last one is\n“positive semi-definite” for the continuous case.")), "TXT", 16, w=335, name=T(tr("term:Mercer 條件", "term:Mercer condition")))
    f.text(580, 100, T(tr("滿足條件的 $K$，必有空間 $H$ 與映射 $\\phi$，\n使 $K(x,z)=\\langle\\phi(x),\\phi(z)\\rangle$；反之亦然。", "such a $K$ has a space $H$ and a map $\\phi$ with\n$K(x,z)=\\langle\\phi(x),\\phi(z)\\rangle$, and conversely.")), "TXT", 16, w=335, math=True)
    f.text(580, 190, T(tr("RKHS：核對應的函數空間\n（本課不需要細節）。", "RKHS: the function space of the kernel\n(details not needed here).")), "TXT", 16, w=335, name=T(tr("term:RKHS", "term:RKHS")))
    cbox(f, 420, 226, 320, 92, "red", T(tr("定理沒告訴我們：\n$\\phi$ 長什麼樣、$H$ 有幾維\n（高斯核是無窮維）", "the theorem does not tell us:\nwhat $\\phi$ looks like, how many\ndimensions $H$ has (Gaussian: infinite)")), size=16, math=True)
    f.text(980, 96, T(tr("Gram 矩陣 $G_{ij}=K(x_i,x_j)$：\n把核值排成的方陣。", "Gram matrix $G_{ij}=K(x_i,x_j)$:\nthe square matrix of kernel values.")), "TXT", 16, w=335, math=True, name=T(tr("term:Gram 矩陣", "term:Gram matrix")))
    f.text(980, 150, T(tr("半正定：特徵值都不是負的。任意有限個點的\nGram 矩陣都必須半正定。", "positive semi-definite: no negative eigenvalue. The Gram matrix of\nany finite set of points must be.")), "TXT", 16, w=335, name=T(tr("term:半正定", "term:Positive semi-definite")))
    cbox(f, 820, 196, 320, 44, "green", T(tr("二次核兩點：特徵值 3、5", "quadratic, 2 points: eigenvalues 3, 5")), size=16)
    cbox(f, 820, 246, 320, 80, "green", T(tr("30 個隨機點（種子 0）：高斯核\n最小特徵值 $8.8\\times10^{-8}$；二次核\n$-8\\times10^{-15}$（捨入誤差，視為 0）", "30 random points (seed 0): Gaussian\nsmallest eigenvalue $8.8\\times10^{-8}$; quadratic\n$-8\\times10^{-15}$ (rounding, treated as 0)")), size=16, math=True)
    cbox(f, 0, 346, 1160, 36, "yellow", T(tr("找到負特徵值，就證明這個核不合法；沒找到，只代表這一組點沒問題，不能證明對所有資料都合法。", "A negative eigenvalue proves the kernel illegal; finding none only says this set of points is fine, not that the kernel is legal for all data.")), size=16, bold=True, dark=True)
    takeaway(f, 0, 392, 1160, 40, T(tr("合法的核＝某個映射的內積；定理保證存在，但不告訴你它長什麼樣。", "A legal kernel is the inner product of some map; the theorem guarantees it exists but not what it looks like.")))
    return f


def dualsvm():
    f = NFig("lec05-kp19-kernels", 1160, 450, T(tr("核化 SVM：常用的核有多項式核 (xᵀz+1)^q、高斯核（RBF）exp(−γ‖x−z‖²)（兩點越近值越接近 1，越遠越接近 0，γ 越大「近」的範圍越小）、雙曲正切核 tanh(βxᵀz+γ)（不一定合法）。對偶問題：max Σλ_i − ½ΣΣλ_iλ_j y_i y_j K(x_i,x_j)，限制 0≤λ_i≤C、Σλ_i y_i=0；C 在這裡變成 λ_i 的上限。訓練好之後分類規則 g(x)=Σ_{i=1}^{N_s} λ_i y_i K(x_i,x)+w₀，g>0 判給 ω₁；由於核的非線性，這在原空間是非線性分類器。驗算：把解出的 λ 代入，與函式庫決策值最大相差約 10⁻¹⁴。",
                                              "Kernelized SVM: common kernels are the polynomial kernel (xᵀz+1)^q, the Gaussian (RBF) kernel exp(−γ‖x−z‖²) (close to 1 for near points, close to 0 for far ones, and the larger γ the smaller the 'near' range) and the hyperbolic-tangent kernel tanh(βxᵀz+γ) (not always legal). The dual problem: max Σλ_i − ½ΣΣλ_iλ_j y_i y_j K(x_i,x_j) subject to 0≤λ_i≤C, Σλ_i y_i=0; here C becomes the upper limit of λ_i. After training the rule is g(x)=Σ_{i=1}^{N_s} λ_i y_i K(x_i,x)+w₀, ω₁ if g>0; thanks to the kernel's non-linearity this is a non-linear classifier in the original space. Check: substituting the solved λ matches the library's decision value to about 10⁻¹⁴.")))
    header_chip(f, 0, 0, 440, 34, "blue", T(tr("常用的核", "Common kernels")), size=17)
    ks = [("orange", tr("多項式核", "Polynomial kernel"), tr("$(x^Tz+1)^q$，$q$ 是次數", "$(x^Tz+1)^q$, $q$ is the degree"), None),
          ("green", tr("高斯核（RBF）", "Gaussian (RBF) kernel"), tr("$\\exp(-\\gamma\\|x-z\\|^2)$；兩點越近越接近 1，\n$\\gamma$ 越大「近」的範圍越小", "$\\exp(-\\gamma\\|x-z\\|^2)$; near points give values near 1,\nthe larger $\\gamma$ the smaller the 'near' range"), None),
          ("red", tr("雙曲正切核", "Hyperbolic-tangent kernel"), tr("$\\tanh(\\beta x^Tz+\\gamma)$；不一定合法（見延伸）", "$\\tanh(\\beta x^Tz+\\gamma)$; not always legal (see extension)"), None)]
    for k, (role, nm, tx, name) in enumerate(ks):
        y = 42 + k * 120; f.rect(0, y, 440, 112, fill=tint(role, 0.12), stroke=tint(role, 0.55), sw=1.5, radius=12); f.text(14, y + 22, T(nm), "TXT", 18, True, "left", w=410); f.text(14, y + 70, T(tx), "TXT", 16, False, "left", w=412, math=True, name=name)
    header_chip(f, 460, 0, 700, 34, "violet", T(tr("對偶問題：選 N 個權重 λ", "The dual problem: choose N weights λ")), size=17)
    f.rect(460, 42, 700, 80, fill=tint("violet", 0.14), radius=12); f.text(810, 82, "$\\max_\\lambda\\ \\sum_i\\lambda_i-\\frac{1}{2}\\sum_{i,j}\\lambda_i\\lambda_jy_iy_jK(x_i,x_j)$", "TXT", 20, w=680, math=True)
    cbox(f, 460, 132, 340, 44, "grey", T(tr("限制 $0\\leq\\lambda_i\\leq C$：每個點的影響不超過 $C$", "$0\\leq\\lambda_i\\leq C$: no point counts above $C$")), size=16, math=True)
    cbox(f, 820, 132, 340, 44, "grey", T(tr("限制 $\\sum_i\\lambda_iy_i=0$：兩類權重互相抵消", "$\\sum_i\\lambda_iy_i=0$: class weights cancel")), size=16, math=True)
    cbox(f, 460, 184, 340, 58, "blue", T(tr("第一項 $\\sum\\lambda_i$ 鼓勵權重大", "the first term $\\sum\\lambda_i$ rewards large weights")), size=16, math=True)
    cbox(f, 820, 184, 340, 58, "orange", T(tr("第二項扣掉「相似且同類的點\n同時權重大」的重複貢獻", "the second term subtracts double counting\nof similar same-class heavy points")), size=16)
    header_chip(f, 460, 252, 700, 34, "green", T(tr("訓練好之後的分類規則", "The rule after training")), size=17)
    f.rect(460, 290, 700, 72, fill=tint("green", 0.14), radius=12); f.text(810, 326, "$g(x)=\\sum_{i=1}^{N_s}\\lambda_iy_iK(x_i,x)+w_0$", "TXT", 20, w=680, math=True)
    cbox(f, 460, 372, 700, 44, "yellow", T(tr("$g>0$ 判給 $\\omega_1$；核是非線性的，所以這在原空間是非線性分類器", "$g>0$ gives $\\omega_1$; the kernel makes it non-linear in the original space")), size=16, bold=True, math=True, dark=True)
    takeaway(f, 0, 404, 440, 40, T(tr("驗算：與函式庫決策值差約 10⁻¹⁴", "Check: matches the library to about 10⁻¹⁴")))
    return f


def _fit_all():
    X, y = rings(60, 1); Xt, yt = rings(500, 2); g = lambda A, B: gk(A, B, 0.5); out = []
    for nm, K in (("lin", lin), ("quad", quad), ("gauss", g)):
        tr_, te, ns, m = N.svc_fit(X, y, [Xt], [yt], K, 10); out.append((nm, K, tr_, te[0], ns, m))
    return X, y, out


def regions():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    X, y, out = _fit_all(); gx = np.linspace(-3.2, 3.2, 90); GX, GY = np.meshgrid(gx, gx); G = np.c_[GX.ravel(), GY.ravel()]
    names = [tr("線性核 xᵀz", "linear kernel xᵀz"), tr("二次核 (xᵀz+1)²", "quadratic kernel (xᵀz+1)²"), tr("高斯核 γ=0.5", "Gaussian kernel γ=0.5")]; roles = ["grey", "blue", "green"]
    f = NFig("lec05-kp19-regions", 1160, 452, T(tr("同心環實驗：只換核（標準差 0.25，訓練每類 60 點種子 1，新資料每類 500 點共 1000 點種子 2，C=10，SVC）。線性核：訓練判對 66/120、新點 562/1000（56.2%）、支撐向量 118，決策邊界是一條直線；二次核：117/120、976/1000（97.6%）、10 個支撐向量；高斯核（γ=0.5）：119/120、968/1000（96.8%）、19 個支撐向量。圓圈標出支撐向量。高斯核只留 19 個點；若用 PNN（KP18）要留全部 120 個訓練點。",
                                              "Ring experiment, only the kernel changes (standard deviation 0.25, 60 training points per class with seed 1, 500 new points per class, 1000 in all, with seed 2, C=10, SVC). Linear kernel: 66/120 training points right, 562/1000 new points (56.2%), 118 support vectors, the decision boundary is a straight line; quadratic kernel: 117/120, 976/1000 (97.6%), 10 support vectors; Gaussian kernel (γ=0.5): 119/120, 968/1000 (96.8%), 19 support vectors. Circles mark support vectors. The Gaussian kernel keeps only 19 points; a PNN (KP18) keeps all 120 training points.")))
    for k, ((nm, K, trn, te, ns, m), role) in enumerate(zip(out, roles)):
        x0 = k * 400; header_chip(f, x0, 0, 360, 34, role, T(names[k]), size=17); panel(f, x0, 38, 360, 330, role)
        cx, cy, sc = x0 + 180, 203, 52
        Z = m.decision_function(K(G, X)).reshape(GX.shape); fig, ax = plt.subplots(); cs = ax.contour(GX, GY, Z, levels=[0]); segs = cs.allsegs[0]; plt.close(fig)
        for s in segs:
            for a, b in zip(s[:-1], s[1:]):
                p1 = (cx + a[0] * sc, cy - a[1] * sc); p2 = (cx + b[0] * sc, cy - b[1] * sc)
                if all(x0 + 4 < p[0] < x0 + 356 and 42 < p[1] < 364 for p in (p1, p2)): f.line(p1[0], p1[1], p2[0], p2[1], solid("red"), 4)
        sv = set(int(i) for i in m.support_)
        for i in range(len(X)):
            px, py = cx + X[i, 0] * sc, cy - X[i, 1] * sc
            (f.ellipse(px, py, 7, solid("blue")) if y[i] < 0 else f.square(px, py, 7, solid("orange")))
            if i in sv: circ(f, px, py, 9, "#3a342c", 1.5)
        f.text(x0 + 180, 392, T(tr(f"訓練 {trn}/120；新點 {te}/1000＝{hu(te/10,1)}%\n支撐向量 {ns}", f"training {trn}/120; new points {te}/1000 = {hu(te/10,1)}%\nsupport vectors {ns}")), "TXT", 16, True, w=350)
    f.text(580, 438, T(tr("藍圓：內環 −1；橘方：外環 +1；紅線：決策邊界；圓圈：支撐向量", "blue circle: inner −1; orange square: outer +1; red line: decision boundary; ring: support vector")), "MUT", 16, w=1100)
    return f


def _chart(f, x0, y0, w, h, labels, series, ylo, yhi, yticks, xlab, best=None):
    f.line(x0 + 50, y0, x0 + 50, y0 + h, "MUT", 2); f.line(x0 + 50, y0 + h, x0 + w, y0 + h, "MUT", 2)
    sy = lambda v: y0 + h - (v - ylo) / (yhi - ylo) * h; n = len(labels); sx = lambda i: x0 + 50 + (i + 0.5) * (w - 50) / n
    for t in yticks: f.line(x0 + 44, sy(t), x0 + 50, sy(t), "MUT", 2); f.text(x0 + 38, sy(t), f"{t:g}", "MUT", 16, False, "right", w=40)
    for i, lb in enumerate(labels): f.text(sx(i), y0 + h + 20, lb, "TXT", 16, w=70)
    f.text(x0 + 50 + (w - 50) / 2, y0 + h + 46, xlab, "TXT", 16, True, w=w)
    for role, vals, lab in series:
        for i in range(n - 1): f.line(sx(i), sy(vals[i]), sx(i + 1), sy(vals[i + 1]), solid(role), 4)
        for i, v in enumerate(vals):
            f.ellipse(sx(i), sy(v), 11, solid(role))
            if lab and (lab is True or i in lab): f.text(sx(i), sy(v) - 18, f"{v:g}", "TXT", 16, True, w=64)
    if best is not None: circ(f, sx(best[0]), sy(best[1]), 14, "#1c1712", 2.5)


def hyper():
    X, y = rings(60, 3, 0.45); Xv, yv = rings(500, 5, 0.45); Xt, yt = rings(500, 4, 0.45); gs = (0.01, 0.1, 0.5, 2, 10, 50); Cs = (0.01, 0.1, 1, 10, 1000)
    RG = [N.svc_fit(X, y, [Xv, Xt], [yv, yt], lambda A, B, g=g: gk(A, B, g), 10)[:3] for g in gs]; RC = [N.svc_fit(X, y, [Xv, Xt], [yv, yt], lambda A, B: gk(A, B, 0.5), C)[:3] for C in Cs]
    pct = lambda v, d: float(hu(v / d * 100, 1))
    f = NFig("lec05-kp19-hyper", 1160, 470, T(tr("超參數：標準差 0.45 的同心環（訓練每類 60 點種子 3；驗證每類 500 點種子 5，專門挑超參數；測試每類 500 點種子 4，最後只報告一次）。固定 C=10 掃 γ=0.01、0.1、0.5、2、10、50：訓練判對 " + "、".join(str(r[0]) for r in RG) + "（/120），驗證 " + "、".join(str(r[1][0]) for r in RG) + "，測試 " + "、".join(str(r[1][1]) for r in RG) + "（/1000）；固定 γ=0.5 掃 C=0.01、0.1、1、10、1000：訓練 " + "、".join(str(r[0]) for r in RC) + "，驗證 " + "、".join(str(r[1][0]) for r in RC) + "，測試 " + "、".join(str(r[1][1]) for r in RC) + "。挑驗證最高的 γ=0.5、C=10（866/1000），測試 861/1000＝86.1%；偷看測試欄挑 γ=0.1 可得 87.0%，那是被挑過的數字，偏樂觀。",
                                              "Hyperparameters: concentric rings with standard deviation 0.45 (60 training points per class with seed 3; 500 validation points per class with seed 5, used only to pick hyperparameters; 500 test points per class with seed 4, reported once at the end). With C=10 fixed, sweeping γ=0.01, 0.1, 0.5, 2, 10, 50: training right " + ", ".join(str(r[0]) for r in RG) + " (of 120), validation " + ", ".join(str(r[1][0]) for r in RG) + ", test " + ", ".join(str(r[1][1]) for r in RG) + " (of 1000); with γ=0.5 fixed, sweeping C=0.01, 0.1, 1, 10, 1000: training " + ", ".join(str(r[0]) for r in RC) + ", validation " + ", ".join(str(r[1][0]) for r in RC) + ", test " + ", ".join(str(r[1][1]) for r in RC) + ". Pick the highest validation, γ=0.5 and C=10 (866/1000), which gives 861/1000 = 86.1% on the test data; peeking at the test column to choose γ=0.1 would give 87.0%, a picked and therefore optimistic number.")))
    for k, (hd, lab, R, xl) in enumerate(((tr("掃 γ（C=10）", "Sweeping γ (C=10)"), [f"{g:g}" for g in gs], RG, "γ"), (tr("掃 C（γ=0.5）", "Sweeping C (γ=0.5)"), [f"{c:g}" for c in Cs], RC, "C"))):
        x0 = k * 590; header_chip(f, x0, 0, 570, 32, "blue" if k == 0 else "orange", T(hd), size=17); panel(f, x0, 36, 570, 280, "blue" if k == 0 else "orange")
        ser = [("blue", [pct(r[0], 120) for r in R], False), ("green", [pct(r[1][0], 1000) for r in R], True), ("red", [pct(r[1][1], 1000) for r in R], False)]
        bi = max(range(len(R)), key=lambda i: R[i][1][0]); _chart(f, x0 + 20, 100, 530, 160, lab, ser, 60, 100, [60, 80, 100], xl, best=(bi, pct(R[bi][1][0], 1000)))
        for j, (role, nm) in enumerate((("blue", tr("訓練", "training")), ("green", tr("驗證", "validation")), ("red", tr("測試", "test")))):
            f.ellipse(x0 + 90 + j * 140, 68, 12, solid(role)); f.text(x0 + 104 + j * 140, 68, T(nm), "TXT", 16, False, "left", w=110)
    card(f, 0, 330, 380, 84, "violet", T(tr("超參數：訓練前要自己決定、\n不是由資料解出來的參數（$\\gamma$、$C$）。", "hyperparameter: a parameter you must choose before\ntraining, not solved from the data ($\\gamma$, $C$).")), name=T(tr("term:超參數", "term:Hyperparameter")))
    card(f, 400, 330, 380, 84, "green", T(tr("驗證資料：專門用來挑超參數的資料；\n測試資料最後只用來報告一次。", "validation data: data used only to pick hyperparameters;\nthe test data are used once at the end.")), name=T(tr("term:驗證資料", "term:Validation data")))
    card(f, 800, 330, 360, 84, "yellow", T(tr("挑驗證最高（黑圈）：$\\gamma=0.5$、$C=10$\n→ 測試 861/1000＝86.1%", "highest validation (black ring):\n$\\gamma=0.5$, $C=10$ → test 861/1000 = 86.1%")))
    takeaway(f, 0, 424, 1160, 40, T(tr("訓練越貼越高，驗證與測試先升後降；偷看測試挑 γ=0.1（87.0%）是偏樂觀的數字。", "Training keeps rising while validation and test rise then fall; peeking at the test to pick γ=0.1 (87.0%) is optimistic.")))
    return f


def dims():
    rows = [((2, 2), "grey"), ((10, 3), "blue"), ((100, 5), "orange"), ((784, 5), "red")]
    f = NFig("lec05-kp19-cost", 1160, 376, T(tr("維度無關，但不是沒有代價：維度 l、次數 q 的多項式核，顯式特徵數 C(l+q, q)：(2,2) 為 6、(10,3) 為 286、(100,5) 為 96,560,646、(784,5) 為 2,515,874,132,157（約 2.5×10¹²）。算一次核：一個內積再取次方，約 l 次乘法，與核空間的維度無關；高斯核的空間是無窮維，一次也只算一個距離。限制：避免的是把高維特徵造出來，不是所有計算；訓練有 N 個點就要處理 N×N 個核值（Gram 矩陣），樣本一多就是瓶頸；沒人知道怎麼選最好的核，靠驗證資料判斷。",
                                              "Dimension-free, but not free of cost: for a polynomial kernel of dimension l and degree q the explicit features number C(l+q, q): (2,2) gives 6, (10,3) gives 286, (100,5) gives 96,560,646 and (784,5) gives 2,515,874,132,157 (about 2.5×10¹²). One kernel evaluation is an inner product and a power, about l multiplications, independent of the kernel space's dimension; the Gaussian kernel's space is infinite-dimensional and one evaluation still computes one distance. Limits: what is avoided is building the high-dimensional features, not all computation; with N training points N×N kernel values (the Gram matrix) must be handled, a bottleneck for many samples; nobody knows how to choose the best kernel, validation data decide.")))
    header_chip(f, 0, 0, 640, 34, "blue", T(tr("多項式核的顯式特徵數（對數刻度）", "Explicit features of a polynomial kernel (log scale)")), size=17); panel(f, 0, 38, 640, 270, "blue")
    for k, ((l, q), role) in enumerate(rows):
        v = comb(l + q, q); y = 58 + k * 56; wbar = max(np.log10(v) / 13 * 250, 8)
        f.text(160, y + 20, f"$(l,q)=({l},{q})$", "TXT", 17, True, "right", w=150, math=True); f.rect(174, y, wbar, 40, fill=solid(role), radius=5); f.text(174 + wbar + 10, y + 20, f"{v:,}", "TXT", 17, True, "left", w=190)
    f.text(320, 290, T(tr("橫條長度＝特徵數的位數（對數刻度）", "bar length = number of digits of the feature count (log scale)")), "MUT", 16, w=620)
    card(f, 660, 0, 500, 100, "green", T(tr("算一次核：一個內積再取次方，約 $l$ 次乘法，\n與核空間的維度無關；高斯核的空間是無窮維，\n一次也只算一個距離。", "one kernel evaluation: an inner product and a power, about $l$\nmultiplications, independent of the kernel space's dimension;\nthe Gaussian kernel's space is infinite and still needs one distance.")))
    card(f, 660, 112, 500, 100, "red", T(tr("限制：避免的是「把高維特徵造出來」，不是所有計算；\n訓練有 $N$ 個點，就要處理 $N\\times N$ 個核值\n（Gram 矩陣），樣本一多就是瓶頸。", "limit: what is avoided is building the high-dimensional features,\nnot all computation; $N$ training points mean $N\\times N$ kernel\nvalues (the Gram matrix), a bottleneck for many samples.")))
    card(f, 660, 224, 500, 84, "orange", T(tr("沒人知道怎麼選最好的核；\n核的好壞要靠驗證資料判斷。", "nobody knows how to pick the best kernel;\nvalidation data decide how good a kernel is.")))
    takeaway(f, 0, 324, 1160, 48, T(tr("怎麼記：只用內積的演算法，內積換成核就能升維；代價是不知道在哪個空間，且要付 Gram 矩陣的成本。", "Swap inner products for kernels to lift an algorithm; the cost: unknown space, Gram matrix.")))
    return f


def tanhk():
    P3 = np.array([[-1, 0], [1.6, 0.9], [0.1, -0.8]]); ev = np.linalg.eigvalsh(np.tanh(2 * P3 @ P3.T + 1)); rng = np.random.default_rng(5); nt = ng = 0
    for _ in range(200):
        P = rng.standard_normal((10, 2)); nt += np.linalg.eigvalsh(np.tanh(2 * P @ P.T + 1)).min() < -1e-12; ng += np.linalg.eigvalsh(gk(P, P, 0.5)).min() < -1e-12
    f = NFig("lec05-kp19-tanh", 1160, 420, T(tr(f"雙曲正切核 K=tanh(2xᵀz+1)（β=2、γ=1）不一定合法：三個點 (−1,0)、(1.6,0.9)、(0.1,−0.8) 的 Gram 矩陣特徵值是 {hu(ev[0],3)}、{hu(ev[1],3)}、{hu(ev[2],3)}，有負的，違反 Mercer 條件。統計：每次取 10 個隨機點（種子 5）、抽 200 次：雙曲正切核 {nt} 次都有負特徵值，高斯核 {ng} 次。理論推論（非本實驗量測）：對偶問題不保證是凸的；用之前先檢查，或改用高斯核。",
                                              f"The hyperbolic-tangent kernel K=tanh(2xᵀz+1) (β=2, γ=1) is not always legal: the Gram matrix of the three points (−1,0), (1.6,0.9), (0.1,−0.8) has eigenvalues {hu(ev[0],3)}, {hu(ev[1],3)}, {hu(ev[2],3)}, one negative, violating the Mercer condition. Statistics: 10 random points each time (seed 5), 200 draws: the tanh kernel has a negative eigenvalue {nt} times, the Gaussian kernel {ng} times. A theoretical inference (not measured here): the dual problem is not guaranteed to be convex; check before use, or switch to the Gaussian kernel.")))
    header_chip(f, 0, 0, 560, 34, "red", T(tr("三個點的 Gram 矩陣特徵值", "Eigenvalues of the three points' Gram matrix")), size=17); panel(f, 0, 38, 560, 250, "red")
    base, sc = 190, 50; f.line(60, base, 520, base, "MUT", 2)
    for k, v in enumerate(ev):
        x = 110 + k * 140; h = v * sc; role = "red" if v < 0 else "green"
        f.rect(x, base - max(h, 0), 90, abs(h), fill=solid(role), radius=4); f.text(x + 45, base - h - 16 if v >= 0 else base - h + 18, hu(v, 3), "TXT", 18, True, w=90)
    f.text(280, 270, T(tr("三個點 $(-1,0)$、$(1.6,0.9)$、$(0.1,-0.8)$；有負值就違反 Mercer 條件", "the three points; a negative value breaks Mercer")), "MUT", 16, w=540, math=True)
    header_chip(f, 600, 0, 560, 34, "blue", T(tr("每次 10 個隨機點，抽 200 次", "10 random points each time, 200 draws")), size=17); panel(f, 600, 38, 560, 250, "blue")
    for k, (nm, v, role) in enumerate(((tr("雙曲正切核", "tanh kernel"), nt, "red"), (tr("高斯核", "Gaussian kernel"), ng, "green"))):
        y = 78 + k * 80; f.text(750, y + 22, T(nm), "TXT", 18, True, "right", w=140); f.rect(770, y, max(v / 200 * 230, 4), 44, fill=solid(role), radius=5); f.text(770 + max(v / 200 * 230, 4) + 10, y + 22, f"{v}／200", "TXT", 18, True, "left", w=120)
    f.text(880, 254, T(tr("橫條：出現負特徵值的次數", "bars: draws with a negative eigenvalue")), "MUT", 16, w=520)
    card(f, 0, 302, 1160, 56, "orange", T(tr("理論推論（非本實驗量測）：雙曲正切核不合法時，對偶問題不保證是凸的；\n用之前先檢查，或改用高斯核。", "theoretical inference (not measured here): when the tanh kernel is illegal the dual problem need not be convex;\ncheck before use, or switch to the Gaussian kernel.")))
    takeaway(f, 0, 370, 1160, 44, T(tr("看起來像核的函數不一定合法：用有限點的 Gram 矩陣，一個負特徵值就能抓到。", "A function that looks like a kernel need not be legal: one negative eigenvalue of a finite Gram matrix catches it.")))
    return f


def solution():
    spec = {
        "id": "lec05-kp19-solution",
        "alt": tr("解答：K(x,z)=(xᵀz+1)²，x=(1,2)、z=(3,1)。題 1：xᵀz=1·3+2·1=5，K=(5+1)²=36，只用 2 次乘法。題 2：φ(x)=(1,1.414,2.828,1,2.828,4)、φ(z)=(1,4.243,1.414,9,4.243,1)，內積 1+6+4+9+12+4=36，與題 1 相同。核對：同一個 36；核只在 2 維算，映射要先造 6 個特徵，維度越高差距越大。",
                  "Solution: K(x,z)=(xᵀz+1)², x=(1,2), z=(3,1). Exercise 1: xᵀz=1·3+2·1=5, K=(5+1)²=36, only 2 multiplications. Exercise 2: φ(x)=(1,1.414,2.828,1,2.828,4), φ(z)=(1,4.243,1.414,9,4.243,1), inner product 1+6+4+9+12+4=36, the same as exercise 1. Check: the same 36; the kernel works in 2 dimensions while the map must first build 6 features, and the gap widens with the dimension."),
        "steps": [{"role": "green", "title": tr("① 題 1：用核", "① Q1: the kernel"), "lines": ["$x^Tz=1\\cdot3+2\\cdot1=5$", "$K=(5+1)^2$", tr("只用 2 次乘法", "only 2 multiplications")], "result": "K = 36"},
                  {"role": "blue", "title": tr("② 題 2：用映射", "② Q2: the map"), "lines": ["$\\phi(x)=(1,1.414,2.828,1,2.828,4)$", "$\\phi(z)=(1,4.243,1.414,9,4.243,1)$", "$1+6+4+9+12+4$"], "result": "36"},
                  {"role": "orange", "title": tr("③ 核對", "③ Check"), "lines": [tr("同一個 $36$", "the same $36$"), tr("核在 2 維算，映射先造 6 個特徵", "kernel: 2-D; map: 6 features"), tr("維度越高，差距越大", "higher dimension, wider gap")], "result": tr("相同", "equal")}],
        "takeaway": tr("核只在 2 維算出 36，映射要先造 6 個特徵；維度越高，差距越大。", "Kernel: 36 in 2-D; map: 6 features first; higher dimension, wider gap."),
    }
    return "steps_flow", spec
