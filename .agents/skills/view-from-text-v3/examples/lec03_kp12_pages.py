import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lec03_kit import *

KP = "KP12"; N = 7
L = lambda w: min(10 * (w + 1) ** 2, 0.3 * (w - 1.5) ** 2)
MAXN = lambda w, r=0.3: max(L(w + e / 100 * r) for e in range(-100, 101))
R_SHARP, R_FLAT = 0.5 * 20 * 0.3 ** 2, 0.5 * 0.6 * 0.3 ** 2; assert round(R_SHARP, 3) == 0.9 and round(R_FLAT, 3) == 0.027 and round(R_SHARP / R_FLAT, 1) == 33.3
WS = [-3 + i * 0.01 for i in range(601)]; WB = min(WS, key=MAXN); assert abs(WB - 1.5) < 0.02 and round(MAXN(-1), 3) == 0.9 and round(MAXN(1.5), 3) == 0.027
lam = lambda w1, w2: w1 * w1 + w2 * w2   # det H = 0 at the minimum, so lambda_max = trace
LA, LB = lam(1, 1), lam(4, 0.25); assert LA == 2 and abs(LB - 16.0625) < 1e-9 and round(LB / LA, 2) == 8.03
Lw = lambda a, b: 0.5 * (a * b - 1) ** 2
def circ(w1, w2, r=0.3): return max(Lw(w1 + r * math.cos(t / 360 * 2 * math.pi), w2 + r * math.sin(t / 360 * 2 * math.pi)) for t in range(360))
CA, CB = circ(1, 1), circ(4, 0.25); assert round(CA, 3) == 0.11 and round(CB, 3) == 0.733
QA, QB = 0.5 * LA * 0.09, 0.5 * LB * 0.09; assert round(QA, 3) == 0.09 and round(QB, 3) == 0.723
assert all(abs(1 * 1 * x - 4 * 0.25 * x) < 1e-12 for x in (-2, 0.5, 3))

def make(lang):
    d = D(lang, KP, N); T = d.T
    g = d.page("KP12 損失一樣低，命運不一樣", "KP12 Equal loss, different fates")
    xr, yr = (-2.5, 3.5), (0, 3.2)
    ux, uy = axes(g, 50, 120, 760, 470, xr, yr)
    g.beat = 1
    curve(g, ux, uy, L, BLUE, xr, yr, n=300)
    lab(g, ux(-1) - 100, uy(0) - 80, 200, T("尖銳谷\n$k=20$", "sharp valley\n$k=20$"), 20, RED, "center", 70); lab(g, ux(1.5) - 100, uy(0) - 80, 200, T("平坦谷\n$k=0.6$", "flat valley\n$k=0.6$"), 20, GRN, "center", 70)
    g.beat = 2
    for w0, col in ((-1, RED), (1.5, GRN)):
        dot(g, ux(w0 + 0.3), uy(L(w0 + 0.3)), 11, col)
    g.line([(ux(-1.3), uy(0.9)), (ux(-0.7), uy(0.9))], RED, sw=3, dash=True, z=4)
    g.text(50, 594, 760, 34, T("$L(w)=\\min\\{\\frac12\\cdot20(w+1)^2,\\ \\frac12\\cdot0.6(w-1.5)^2\\}$:兩個谷底訓練損失都是 $0$", "$L(w)=\\min\\{\\frac12\\cdot20(w+1)^2,\\ \\frac12\\cdot0.6(w-1.5)^2\\}$: both valley floors have training loss $0$"), size=17, color=MUTED, bold=True, align="center", z=8)
    g.beat = 2
    box(g, 850, 126, 380, 170, RED, REDL, T("尖銳谷,$\\Delta w=0.3$", "sharp valley, $\\Delta w=0.3$"), r"$\Delta L=\frac12\cdot20\cdot0.3^2=0.9$", ts=20, bs=26)
    g.beat = 3
    box(g, 850, 316, 380, 150, GRN, GRNL, T("平坦谷,$\\Delta w=0.3$", "flat valley, $\\Delta w=0.3$"), r"$\Delta L=\frac12\cdot0.6\cdot0.3^2=0.027$", ts=20, bs=24)
    g.beat = 4
    gold(g, 850, 486, 380, 130, T("差 $0.9/0.027=33.3$ 倍,正好是 $20/0.6$", "a factor $0.9/0.027=33.3$, exactly $20/0.6$"), size=22)
    banner(g, T("測試資料和訓練資料總有差別:粗略當作權重被推了一小段 $\\Delta w$(近似)", "test data differ from training data: roughly a small push $\\Delta w$ (an approximation)"))
    d.done(g, "損失一樣低", "Equal loss", {0: ("訓練損失低就夠了嗎？不夠。現代網路的參數遠多於樣本，損失曲面上有很多訓練損失接近零的極小點。同一條曲線上有尖銳谷與平坦谷。", "Is a low training loss enough? No. Modern networks have far more parameters than samples, so many minima have near-zero training loss. One curve has a sharp valley and a flat one."), 1: ("兩個谷底的訓練損失都是零。", "Both valley floors have training loss zero."), 2: ("測試資料和訓練資料有差別，粗略當作權重被推了一小段 0.3。尖銳谷的損失升到 0.9。", "Test data differ from training data, roughly as the weights being pushed by 0.3. The sharp valley's loss rises to 0.9."), 3: ("平坦谷只升到 0.027。", "The flat valley rises only to 0.027."), 4: ("差 33.3 倍，正好是曲率的比 20 除以 0.6。這只是近似，一般情況是啟發式，不是定理。", "A factor 33.3, exactly the curvature ratio 20 over 0.6. This is only an approximation, a heuristic rather than a theorem.")})

    g = d.page("用 Hessian 算鄰域的損失上升", "Computing the neighborhood rise with the Hessian")
    g.beat = 1
    g.rect(120, 118, 1040, 110, fill=CARD, stroke=GRAY, sw=3, radius=16, z=3); g.text(130, 118, 1020, 110, r"$\Delta L\approx\tfrac12\,\Delta w^{\top}H\,\Delta w\ \le\ \tfrac12\,\lambda_{\max}\,\|\Delta w\|^{2}$", size=38, color=INK, align="center", vcenter=True, z=8)
    g.beat = 2
    chain(g, 50, 260, 1180, 150, [(BLUE, BLUEL, T("極小點梯度為 $0$", "zero gradient at the minimum"), T("展開到二階,丟掉一次項", "expand to second order; the linear term drops")), (ORG, ORGL, T("分解 $\\Delta w$", "decompose $\\Delta w$"), r"$\sum_i\lambda_ic_i^2\le\lambda_{\max}\|\Delta w\|^2$"), (RED, REDL, T("最壞方向", "the worst direction"), T("沿最大特徵向量\n半徑 $\\rho$ 內約 $\\frac12\\lambda_{\\max}\\rho^2$", "along the top eigenvector\nwithin radius $\\rho$ about $\\frac12\\lambda_{\\max}\\rho^2$"))], ts=20, bs=22, gap=50, beat0=2)
    g.beat = 5
    gold(g, 150, 440, 980, 90, T("一維:$H=k$,$\\Delta L=\\frac12k\\Delta w^2$,正是上一頁的 $0.9$ 與 $0.027$", "one dimension: $H=k$, $\\Delta L=\\frac12k\\Delta w^2$, exactly the $0.9$ and $0.027$ of the last page"), size=22)
    g.beat = 6
    gold(g, 150, 550, 980, 70, T("三個條件:極小點、$\\Delta w$ 夠小、只是局部結論", "three conditions: a minimum, a small $\\Delta w$, only a local conclusion"), size=21, light=REDL, stroke=RED)
    banner(g, T("$\\lambda_{\\max}$ 控制「鄰域內最壞的方向」", "$\\lambda_{\\max}$ controls the worst direction in the neighborhood"))
    d.done(g, "Hessian", "The Hessian", {0: ("能不能由極小點的曲率直接算出來？可以，只要位移夠小。在極小點梯度是零，展開到二階，損失上升約等於二分之一位移乘海森再乘位移，上界是二分之一最大特徵值乘位移長度平方。", "Can we compute it from the curvature at the minimum? Yes, if the displacement is small. The gradient is zero there; expanding to second order the rise is about one half displacement times Hessian times displacement, bounded by one half the largest eigenvalue times the squared displacement."), 1: ("極小點的梯度是零，所以展開到二階，丟掉一次項。", "The gradient is zero at the minimum, so expanding to second order drops the linear term."), 2: ("把位移沿海森的特徵向量分解，每個特徵值都不超過最大特徵值。", "Decompose the displacement along the Hessian's eigenvectors; every eigenvalue is at most the largest."), 3: ("等號在位移恰好沿最大特徵向量時成立，所以最大特徵值控制鄰域內最壞的方向，半徑 rho 內最壞的上升約二分之一最大特徵值乘 rho 平方。", "Equality holds along the top eigenvector, so the largest eigenvalue controls the worst direction; within radius rho the worst rise is about one half lambda max times rho squared."), 4: ("", ""), 5: ("一維時 H 就是曲率 k，正是上一頁的 0.9 與 0.027。", "In one dimension H is just the curvature k, exactly the previous page's 0.9 and 0.027."), 6: ("條件要說清楚：要是極小點，位移要夠小，而且這只是局部結論，谷與谷之間發生什麼事它沒有說。", "State the conditions: a minimum, a small displacement, and only a local conclusion; it says nothing about what happens between valleys.")})

    g = d.page("平坦度不是泛化的證明", "Flatness is not a proof of generalization")
    g.beat = 1
    g.text(50, 108, 1180, 36, T("兩層線性網路 $y=w_2w_1x$,一筆樣本 $(1,1)$,$L=\\frac12(w_2w_1-1)^2$:所有 $w_1w_2=1$ 都是極小點", "two-layer linear net $y=w_2w_1x$, one sample $(1,1)$, $L=\\frac12(w_2w_1-1)^2$: every $w_1w_2=1$ is a minimum"), size=19, color=MUTED, bold=True, align="center", z=8)
    box(g, 50, 160, 570, 190, BLUE, BLUEL, T("點 $A=(1,1)$", "point $A=(1,1)$"), r"$\lambda_{\max}=2$", ts=22, bs=40)
    box(g, 660, 160, 570, 190, ORG, ORGL, T("點 $B=(4,\\frac14)$", "point $B=(4,\\frac14)$"), r"$\lambda_{\max}=16.0625$", ts=22, bs=40)
    g.beat = 2
    gold(g, 190, 375, 900, 90, T("乘積都是 $1$:對任何 $x$ 預測都一樣($x=-2,0.5,3$ 都是 $-2,0.5,3$),是同一個函數", "both products are $1$: the prediction is the same for every $x$ ($-2,0.5,3$ for $x=-2,0.5,3$): the same function"), size=21)
    g.beat = 3
    box(g, 190, 490, 900, 120, RED, REDL, T("但 $\\lambda_{\\max}$ 差 $8.03$ 倍", "yet $\\lambda_{\\max}$ differs by $8.03\\times$"), T("曲率依賴參數的寫法,函數本身不依賴", "curvature depends on how the parameters are written; the function does not"), ts=22, bs=22)
    banner(g, T("曲率是診斷,不是證明;不同參數化之間不能直接比大小", "curvature is a diagnostic, not a proof; different parameterizations cannot be compared directly"))
    d.done(g, "同一個函數", "The same function", {0: ("兩層線性網路，一筆樣本，所有 w1 乘 w2 等於 1 的點都是極小點，損失都是零。點 1、1 的最大特徵值是 2；點 4、四分之一是 16.0625。", "A two-layer linear network with one sample: every point with w1 times w2 equal 1 is a minimum with loss zero. The point 1, 1 has largest eigenvalue 2; the point 4, one quarter has 16.0625."), 1: ("乘積都是 1，所以對任何輸入，預測完全一樣。兩個網路是同一個函數。", "Both products are 1, so the prediction is identical for every input: the two networks are the same function."), 2: ("但最大特徵值差 8.03 倍。曲率依賴參數的寫法，而函數本身不依賴。所以曲率是診斷，不是證明，不同參數化之間不能直接比大小。", "Yet the largest eigenvalues differ by 8.03 times. Curvature depends on how the parameters are written and the function does not. So curvature is a diagnostic, not a proof, and cannot be compared across parameterizations."), 3: ("", "")})

    g = d.page("重新縮放可以偽造尖銳與平坦", "Rescaling can fake sharp and flat")
    g.beat = 1
    chain(g, 50, 120, 1180, 150, [(BLUE, BLUEL, T("改寫參數", "rewrite the parameters"), r"$(w_1,w_2)\to(\alpha w_1,\,w_2/\alpha)$"), (GRN, GRNL, T("函數不變", "the function is unchanged"), T("預測不變", "same predictions")), (RED, REDL, T("但", "but"), r"$\lambda_{\max}=\alpha^2+\alpha^{-2}$" + T("\n可任意大", "\narbitrarily large"))], ts=20, bs=24, gap=50, beat0=1)
    g.beat = 4
    rows = [["$A$", f"{QA:.3f}", f"{CA:.3f}"], ["$B$", f"{QB:.3f}", f"{CB:.3f}"]]
    table(g, 190, 310, [140, 380, 380], ["", T("二次近似 $\\frac12\\lambda_{\\max}\\cdot0.3^2$", "quadratic estimate $\\frac12\\lambda_{\\max}\\cdot0.3^2$"), T("半徑 $0.3$ 圓內真正的最大損失", "true maximum loss in the radius-$0.3$ disk")], rows, rh=64, fs=28, hfs=18, styles=[(BLUEL, BLUE), (ORGL, ORG)], gap=8)
    g.beat = 5
    gold(g, 150, 530, 980, 90, T("近似的量級對,略低一些:忽略了三次以上的項。Dinh 等人(2017)把這個現象推到 ReLU 網路", "the order of magnitude is right but a bit low: higher-order terms are ignored. Dinh et al. (2017) extend this to ReLU networks"), size=20)
    banner(g, T("尖銳與平坦可以靠重新縮放權重來偽造", "sharp and flat can be faked by rescaling the weights"))
    d.done(g, "重新縮放", "Rescaling", {0: ("把參數改成 alpha 乘 w1 與 w2 除以 alpha，", "Rewrite the parameters as alpha times w1 and w2 over alpha,"), 1: ("函數不變，", "the function is unchanged,"), 2: ("但最大特徵值是 alpha 平方加 alpha 負二次方，可以任意大。", "but the largest eigenvalue is alpha squared plus alpha to the minus 2, arbitrarily large."), 3: ("", ""), 4: ("半徑 0.3 的圓內，二次近似給出的最壞上升是 0.090 與 0.723，真正的損失最大值是 0.110 與 0.733。", "Within a disk of radius 0.3 the quadratic estimate of the worst rise is 0.090 and 0.723 while the true maxima are 0.110 and 0.733."), 5: ("量級對，略低，因為忽略了三次以上的項。Dinh 等人把這個現象推到有 ReLU 的網路，所以尖銳與平坦可以靠重新縮放權重偽造。", "The order is right, slightly low because higher-order terms are ignored. Dinh and colleagues extended this to ReLU networks, so sharp and flat can be faked by rescaling the weights.")})

    g = d.page("優化器偏好平坦解?只是有條件的經驗傾向", "Do optimizers prefer flat minima?")
    g.beat = 1
    box(g, 50, 126, 570, 230, ORG, ORGL, T("來源的說法", "the source's claim"), T("小批量 SGD 的噪聲把參數「踢出」尖銳谷\n大批量與牛頓法會陷入尖銳谷", "small-batch SGD noise “kicks” parameters out of sharp valleys\nlarge batches and Newton's method fall into them"), ts=22, bs=21)
    g.beat = 2
    box(g, 660, 126, 570, 230, BLUE, BLUEL, T("文獻實際上", "what the literature shows"), T("Keskar 等人(2017):大批量的泛化差距與較尖銳的解有關\n後續:調整學習率與訓練步數可縮小差距(Hoffer 等人;Goyal 等人)", "Keskar et al. (2017): the large-batch generalization gap relates to sharper solutions\nlater: tuning the learning rate and steps narrows the gap (Hoffer et al.; Goyal et al.)"), ts=22, bs=18)
    g.beat = 3
    gold(g, 150, 400, 980, 120, T("只是有條件的經驗傾向,沒有定理;本講義不把它當成證明", "only a conditional empirical tendency, no theorem; this note does not treat it as a proof"), size=24)
    banner(g, T("「大批量一定收斂到尖銳解」說得太絕對", "“large batches always land in sharp minima” is too absolute"))
    d.done(g, "經驗傾向", "A tendency", {0: ("優化器偏好平坦解的說法：來源說小批量 SGD 的噪聲把參數踢出尖銳谷，大批量與牛頓法會陷入尖銳谷。", "The claim that optimizers prefer flat solutions: the source says small-batch SGD noise kicks parameters out of sharp valleys while large batches and Newton's method fall in."), 1: ("文獻上：Keskar 等人觀察到大批量訓練的泛化差距與較尖銳的解有關，但後續研究指出調整學習率與訓練步數可以縮小差距。", "In the literature: Keskar and colleagues saw that the large-batch generalization gap relates to sharper solutions, but later work shows tuning the learning rate and steps narrows the gap."), 2: ("所以這只是有條件的經驗傾向，沒有定理，本講義不把它當成證明。", "So it is only a conditional empirical tendency, not a theorem, and this note does not treat it as a proof."), 3: ("", "")})

    g = d.page("SAM 的動機:把鄰域最壞損失放進目標", "SAM: put the worst neighborhood loss in the objective")
    xr, yr = (-2.5, 3.5), (0, 3.2)
    ux, uy = axes(g, 50, 120, 760, 400, xr, yr)
    g.beat = 1
    curve(g, ux, uy, L, BLUE, xr, yr, n=300, sw=4) if False else curve(g, ux, uy, L, BLUE, xr, yr, sw=4, n=300)
    g.beat = 2
    curve(g, ux, uy, MAXN, RED, xr, yr, sw=5, n=300)
    dot(g, ux(1.5), uy(MAXN(1.5)), 12, GRN)
    g.text(50, 524, 760, 60, T("藍:$L(w)$　紅:鄰域最壞損失 $\\max_{\\|\\epsilon\\|\\le0.3}L(w+\\epsilon)$", "blue: $L(w)$ · red: the worst neighborhood loss $\\max_{|\\epsilon|\\le0.3}L(w+\\epsilon)$"), size=17, color=MUTED, bold=True, align="center", z=8)
    g.beat = 1
    g.rect(850, 126, 380, 130, fill=CARD, stroke=GRAY, sw=3, radius=16, z=3); g.text(860, 126, 360, 130, r"$\min_w\ \max_{\|\epsilon\|\le\rho}L(w+\epsilon)$", size=26, color=INK, align="center", vcenter=True, z=8)
    g.beat = 2
    box(g, 850, 276, 380, 140, PUR, PURL, T("銳度項", "the sharpness term"), T("鄰域最壞損失比 $L(w)$ 高多少\n大致就是 $\\lambda_{\\max}$ 的大小", "how far the worst neighborhood loss exceeds $L(w)$\nroughly the size of $\\lambda_{\\max}$"), ts=19, bs=17)
    g.beat = 3
    box(g, 850, 436, 380, 120, GRN, GRNL, T("$\\rho=0.3$", "$\\rho=0.3$"), T("尖銳谷底 $0.9$,平坦谷底 $0.027$\n最小點是 $w=1.5$ 的平坦谷", "sharp floor $0.9$, flat floor $0.027$\nthe minimum is the flat valley $w=1.5$"), ts=19, bs=17)
    g.beat = 4
    gold(g, 50, 592, 1180, 40, T("限制:歐氏球不是縮放不變的(ASAM 改用隨縮放調整的鄰域);最小化鄰域最壞損失是動機,不是泛化的保證", "limits: the Euclidean ball is not scale-invariant (ASAM adapts the neighborhood); minimizing the worst loss is a motivation, not a guarantee"), size=15)
    banner(g, T("內層的最大化每步都要解,太貴了:怎麼用一次梯度近似它?(KP13)", "the inner maximization is too costly each step: how to approximate it with one gradient? (KP13)"))
    d.done(g, "SAM 的動機", "The motivation for SAM", {0: ("既然鄰域的損失上升和泛化有關，能不能不期待優化器碰巧走到平坦解，而是直接把它寫進目標？這就是 SAM 的動機：不只要求 L 低，還要求鄰域內最壞的損失也低。", "Since the neighborhood rise relates to generalization, why not write it into the objective rather than hope the optimizer stumbles on a flat solution? That is SAM's motivation: not only a low L but a low worst loss in the neighborhood."), 1: ("紅線是鄰域最壞損失。方括號是銳度項：鄰域最壞損失比 L 高多少，大致就是最大特徵值的大小。", "The red line is the worst neighborhood loss. The sharpness term is how far it exceeds L, roughly the size of the largest eigenvalue."), 2: ("半徑 0.3：尖銳谷底的鄰域最壞損失是 0.9，平坦谷底是 0.027。原來的 L 在兩個谷底都是零，打平；加上銳度項之後，最小點是平坦谷。", "Radius 0.3: the sharp floor's worst neighborhood loss is 0.9, the flat floor's 0.027. The original L ties at zero; with the sharpness term the minimum is the flat valley."), 3: ("", ""), 4: ("兩個限制：歐氏球不是縮放不變的，所以 SAM 不是萬靈丹；最小化鄰域最壞損失是動機，不是泛化更好的保證。內層最大化每步都要解，太貴了，怎麼用一次梯度近似？這是 KP13。", "Two limits: the Euclidean ball is not scale-invariant, so SAM is no cure-all; minimizing the worst neighborhood loss is a motivation, not a guarantee. The inner maximization is too costly each step; how to approximate it with one gradient is KP13.")})
    closing(d, [(RED, REDL, "「平坦解的 $\\Delta L\\approx0$」", "“a flat solution has $\\Delta L\\approx0$”", "是較小,不是 $0$:$0.027$", "smaller, not zero: $0.027$"), (ORG, ORGL, "「平坦的解一定泛化更好」", "“a flat solution always generalizes better”", "同一個函數可以有 $8.03$ 倍的 $\\lambda_{\\max}$:是診斷,不是證明", "the same function can have an $8.03\\times$ $\\lambda_{\\max}$: a diagnostic, not a proof"), (PUR, PURL, "「SAM 保證找到平坦解」", "“SAM guarantees a flat solution”", "它最小化鄰域最壞損失;是動機,不是保證", "it minimizes the worst neighborhood loss: a motivation, not a guarantee")],
            ("內層最大化每一步都要解,太貴了。怎麼用一次梯度近似它?", "The inner maximization is too costly every step. How can one gradient approximate it?"), ("下一個單元:Vanilla SAM", "next unit: Vanilla SAM"))
    return d.pages
