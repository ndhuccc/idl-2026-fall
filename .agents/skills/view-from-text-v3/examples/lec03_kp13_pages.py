import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lec03_kit import *

KP = "KP13"; N = 7
f = lambda x, y: 0.5 * (x * x + 25 * y * y)
RHO = 0.5; ETA = 0.02
def worst(w, rho=RHO, n=36000): return max(f(w[0] + rho * math.cos(i / n * 2 * math.pi), w[1] + rho * math.sin(i / n * 2 * math.pi)) for i in range(n))
W0 = (10.0, 2.0); G0 = (W0[0], 25 * W0[1]); NG = math.hypot(*G0); EPS = (RHO * G0[0] / NG, RHO * G0[1] / NG); WA = (W0[0] + EPS[0], W0[1] + EPS[1])
assert round(NG, 2) == 50.99 and [round(v, 3) for v in EPS] == [0.098, 0.49] and round(f(*WA), 2) == 128.50 and round(worst(W0), 2) == 128.53 and round(RHO * NG, 3) == 25.495
assert round(f(*W0) + RHO * NG, 3) == 125.495 and round(worst(W0) - (f(*W0) + RHO * NG), 2) == 3.03
def step(w, sam):
    g = (w[0], 25 * w[1])
    if sam: n = math.hypot(*g); e = (RHO * g[0] / n, RHO * g[1] / n); g = (w[0] + e[0], 25 * (w[1] + e[1]))
    return (w[0] - ETA * g[0], w[1] - ETA * g[1]), g
def traj(sam):
    w = W0; out = {0: (f(*W0), worst(W0))}; g1 = None
    for k in range(1, 101):
        w, g = step(w, sam)
        if k == 1: g1 = (g, w)
        if k in (1, 10, 30, 60, 100): out[k] = (f(*w), worst(w))
    return out, g1
GD, GD1 = traj(False); SM, SM1 = traj(True)
assert [round(v, 3) for v in GD1[1]] == [9.8, 1.0] and [round(v, 3) for v in SM1[1]] == [9.798, 0.755] and [round(v, 3) for v in SM1[0]] == [10.098, 62.257] and round(math.hypot(*SM1[0]), 2) == 63.07
assert (round(GD[1][0], 2), round(GD[1][1], 2), round(SM[1][0], 2), round(SM[1][1], 2)) == (60.52, 76.79, 55.12, 68.45) and (round(GD[100][0], 3), round(GD[100][1], 2), round(SM[100][0], 3), round(SM[100][1], 2)) == (0.879, 4.04, 0.813, 5.98)

def make(lang):
    d = D(lang, KP, N); T = d.T
    g = d.page("KP13 把「平坦」寫成目標", "KP13 From flatness to an objective")
    g.beat = 1
    g.rect(120, 118, 1040, 100, fill=CARD, stroke=GRAY, sw=3, radius=16, z=3); g.text(130, 118, 1020, 100, r"$\min_{w}\ \max_{\|\epsilon\|_p\le\rho}\ L(w+\epsilon)$", size=42, color=INK, align="center", vcenter=True, z=8)
    g.beat = 2
    chain(g, 50, 250, 1180, 150, [(RED, REDL, T("內層 $\\max$", "inner $\\max$"), T("在半徑 $\\rho$ 的球裡找最壞的鄰居", "find the worst neighbor in the radius-$\\rho$ ball")), (GRN, GRNL, T("外層 $\\min$", "outer $\\min$"), T("調整 $w$ 壓低最壞鄰居的損失", "adjust $w$ to lower its loss"))], ts=22, bs=21, th=60, gap=100, beat0=2)
    g.beat = 4
    g.rect(120, 430, 1040, 100, fill=CARD, stroke=GRAY, sw=3, radius=16, z=3); g.text(130, 430, 1020, 100, r"$L_{\mathrm{SAM}}(w)=L(w)+\max_{\|\epsilon\|\le\rho}\big(L(w+\epsilon)-L(w)\big)$", size=30, color=INK, align="center", vcenter=True, z=8)
    g.beat = 5
    gold(g, 150, 540, 980, 84, T("只是移項,沒有近似:銳度 = 鄰域內損失最多還能再升多少;$\\rho\\to0$ 退回 $L(w)$", "a rearrangement, no approximation: sharpness = how much the loss can still rise; $\\rho\\to0$ gives $L(w)$"), size=20)
    banner(g, T("SAM:sharpness-aware minimization(銳度感知最小化)", "SAM: sharpness-aware minimization"))
    d.done(g, "min-max 目標", "The min-max objective", {0: ("一般訓練只最小化 L，L 只回報這一個點的損失，看不到周圍的地形。SAM 把問法改成：在 w 附近走一小段，最壞會變成多糟。", "Ordinary training minimizes L, which reports only this point's loss and not the terrain around it. SAM asks: if I walk a little around w, how bad can it get."), 1: ("內層的最大化在半徑 rho 的球裡找讓損失最大的擾動，也就是最壞的鄰居。", "The inner maximization finds, in a ball of radius rho, the perturbation that maximizes the loss: the worst neighbor."), 2: ("外層的最小化調整 w，把最壞鄰居的損失壓低。", "The outer minimization adjusts w to lower the worst neighbor's loss."), 3: ("", ""), 4: ("把內層最大值減掉 L，剩下的就是銳度。這個等式只是移項，沒有近似。", "Subtracting L from the inner maximum leaves the sharpness. This identity is just a rearrangement, no approximation."), 5: ("SAM 同時要求訓練損失低，而且鄰域內損失升得不多。rho 趨近零時回到一般的 L。", "SAM asks for a low training loss and a loss that does not rise much in the neighborhood. As rho goes to zero it returns to plain L.")})

    g = d.page("內層最大化的閉式解", "The closed-form worst perturbation")
    g.beat = 1
    chain(g, 50, 120, 1180, 170, [(BLUE, BLUEL, T("一階泰勒", "first-order Taylor"), r"$L(w+\epsilon)\approx L(w)+\epsilon^{\top}g$"), (ORG, ORGL, T("內層問題", "the inner problem"), r"$\max_{\|\epsilon\|_2\le\rho}\epsilon^{\top}g$"), (PUR, PURL, T("Cauchy–Schwarz", "Cauchy–Schwarz"), r"$\epsilon^{\top}g\le\rho\|g\|_2$")], ts=20, bs=26, gap=50, beat0=1)
    g.beat = 4
    g.rect(190, 330, 900, 120, fill="#fef3c7", stroke=GOLD, sw=4, radius=16, z=3); g.text(200, 330, 880, 120, r"$\epsilon^{*}=\rho\,\dfrac{g}{\|g\|_2}$".replace(r"\dfrac", r"\frac"), size=48, color=INK, align="center", vcenter=True, z=8)
    g.beat = 5
    gold(g, 150, 480, 980, 130, T("最壞擾動 = 沿梯度方向走 $\\rho$ 那麼遠:長度永遠是 $\\rho$,與梯度大小無關\n一階只決定方向", "the worst perturbation = walk $\\rho$ along the gradient: its length is always $\\rho$, whatever the gradient size\nfirst order fixes only the direction"), size=22)
    banner(g, T("一般 $p$:用 Hölder 不等式;$p=\\infty$ 時 $\\epsilon^{*}=\\rho\\,\\mathrm{sign}(g)$", "general $p$: Hölder's inequality; for $p=\\infty$, $\\epsilon^{*}=\\rho\\,\\mathrm{sign}(g)$"))
    d.done(g, "閉式解", "Closed form", {0: ("直接在深度網路的損失上找最大值很難。但 rho 很小時，損失在鄰域內近似一個平面，做一階泰勒展開。", "Finding the maximum directly on a deep network's loss is hard. But for small rho the loss is nearly flat in the neighborhood, so expand to first order."), 1: ("內層問題變成找讓擾動乘梯度最大的擾動。", "The inner problem becomes maximizing the perturbation times the gradient."), 2: ("由 Cauchy–Schwarz 不等式，它不超過 rho 乘梯度長度。", "By Cauchy–Schwarz it is at most rho times the gradient length."), 3: ("", ""), 4: ("等號成立的條件是擾動與梯度同方向，長度剛好是 rho，所以最壞擾動是 rho 乘梯度除以梯度長度。", "Equality holds when the perturbation points along the gradient with length exactly rho, so the worst perturbation is rho times the gradient over its length."), 5: ("白話：一階看起來往哪裡走損失升得最快就是梯度方向，rho 決定走多遠。擾動長度與梯度大小無關。", "In plain words: the direction of fastest rise to first order is the gradient, and rho decides how far. The length does not depend on the gradient size.")})

    g = d.page("在碗上算:最壞的鄰居", "On the bowl: the worst neighbor")
    x0, y0, w_, h_ = 50, 120, 700, 470
    xr, yr = (9.3, 10.9), (1.6, 3.0)
    ux = lambda v: x0 + (v - xr[0]) / (xr[1] - xr[0]) * w_; uy = lambda v: y0 + h_ - (v - yr[0]) / (yr[1] - yr[0]) * h_
    g.beat = 1
    g.rect(x0, y0, w_, h_, fill=CARD, stroke=GRAY, sw=3, radius=12, z=1)
    g.ellipse(ux(10), uy(2), abs(ux(10.5) - ux(10)), abs(uy(2.5) - uy(2)), stroke=BLUE, sw=4, z=3)
    dot(g, ux(10), uy(2), 12, BLUE); lab(g, ux(10) - 160, uy(2) + 14, 150, "$w=(10,2)$", 18, BLUE, "right")
    g.beat = 2
    g.line([(ux(10), uy(2)), (ux(WA[0]), uy(WA[1]))], RED, sw=6, arrow="end", head=18, z=6); dot(g, ux(WA[0]), uy(WA[1]), 12, RED); lab(g, ux(WA[0]) + 16, uy(WA[1]) - 30, 240, "$w+\\epsilon^{*}$", 20, RED)
    g.text(50, 594, 700, 34, T("放大看:半徑 $\\rho=0.5$ 的圓(座標範圍 $9.3$–$10.9$,$1.6$–$3.0$)", "zoomed in: the circle of radius $\\rho=0.5$ (ranges $9.3$–$10.9$, $1.6$–$3.0$)"), size=17, color=MUTED, bold=True, align="center", z=8)
    g.beat = 1
    box(g, 790, 126, 440, 150, BLUE, BLUEL, T("梯度", "gradient"), r"$g=(10,\,50),\ \|g\|_2=50.99$" + "\n" + r"$\epsilon^{*}=(0.098,\,0.490)$", ts=20, bs=24)
    g.beat = 2
    box(g, 790, 296, 440, 130, RED, REDL, T("最壞鄰居的損失", "loss at the worst neighbor"), r"$f(10.098,\,2.490)=128.50$", ts=20, bs=26)
    g.beat = 3
    box(g, 790, 446, 440, 160, GRN, GRNL, T("真正的最大值", "the true maximum"), T("$128.53$(只差 $0.02$,方向差 $2.1$ 度)\n一階只估方向:$L+\\rho\\|g\\|=125.495$,低估 $3.03$", "$128.53$ (only $0.02$ more; direction off by $2.1°$)\nfirst order gives a direction only: $L+\\rho\\|g\\|=125.495$ is $3.03$ too low"), ts=19, bs=17)
    banner(g, T("在圓周上取 20 萬個隨機方向:最大的 $\\epsilon^{\\top}g=25.495=\\rho\\|g\\|_2$,沒有任何一個超過", "200,000 random directions on the circle: the largest $\\epsilon^{\\top}g=25.495=\\rho\\|g\\|_2$, none exceeds it"))
    d.done(g, "最壞的鄰居", "The worst neighbor", {0: ("碗上，起點 10、2，梯度 10、50，長度 50.99。以 rho 0.5 為半徑畫圓。", "On the bowl: start 10, 2, gradient 10, 50, length 50.99. Draw a circle of radius 0.5."), 1: ("最壞擾動是 0.098、0.490，走到 10.098、2.490，那裡的損失是 128.50。", "The worst perturbation is 0.098, 0.490, reaching 10.098, 2.490 where the loss is 128.50."), 2: ("真正的最大值是 128.53，方向只差 2.1 度，兩者只差 0.02：一階近似挑方向挑得很準。但一階只估方向，不是值：用損失加 rho 乘梯度長度會低估 3.03，被忽略的是二階項。", "The true maximum is 128.53 with the direction off by only 2.1 degrees; they differ by 0.02, so first order picks the direction well. But first order estimates the direction, not the value: loss plus rho times the gradient length is 3.03 too low, the missing part being the second-order term.")})

    g = d.page("兩步更新", "The two-step update")
    g.beat = 0
    ids = chain(g, 50, 130, 1180, 190, [(BLUE, BLUEL, T("1 算梯度", "1 gradient"), r"$g=\nabla L(w)$"), (ORG, ORGL, T("2 走到最壞鄰居", "2 worst neighbor"), r"$w+\epsilon^{*}$"), (PUR, PURL, T("3 在那裡取梯度", "3 gradient there"), r"$g_{adv}=\nabla L(w+\epsilon^{*})$"), (GRN, GRNL, T("4 回到 $w$ 下降", "4 descend at $w$"), r"$w\leftarrow w-\eta\,g_{adv}$")], ts=20, bs=24, gap=40, beat0=1)
    g.beat = 5
    gold(g, 150, 360, 980, 90, T("關鍵是第 4 步:梯度在 $w+\\epsilon^{*}$ 算出來,更新卻套用在 $w$ 上", "the key is step 4: the gradient is taken at $w+\\epsilon^{*}$ but applied at $w$"), size=24)
    g.beat = 6
    box(g, 50, 470, 570, 140, RED, REDL, T("忽略一項", "one term is dropped"), T("$\\nabla\\Phi=(I+d\\epsilon^{*}/dw)^{\\top}\\nabla L(w+\\epsilon^{*})$\n$d\\epsilon^{*}/dw$ 含 Hessian:直接忽略", "$\\nabla\\Phi=(I+d\\epsilon^{*}/dw)^{\\top}\\nabla L(w+\\epsilon^{*})$\n$d\\epsilon^{*}/dw$ contains the Hessian: ignored"), ts=20, bs=18)
    box(g, 660, 470, 570, 140, GRN, GRNL, T("實作", "in practice"), T("小批次的兩次傳遞用同一個小批次\n$\\epsilon^{*}$ 當常數(不對它反向傳播)", "both passes use the same minibatch\n$\\epsilon^{*}$ is a constant (no backprop through it)"), ts=20, bs=18)
    banner(g, T("每步兩次前向加反向", "two forward-and-backward passes per step"))
    d.done(g, "兩步更新", "The two-step update", {0: ("有了最壞擾動，外層要對 w 最小化擾動後的損失，但最壞擾動本身依賴 w。SAM 的完整一步是四個動作。", "With the worst perturbation, the outer step minimizes the perturbed loss, yet the perturbation depends on w. SAM's full step is four actions."), 1: ("第一，在 w 做一次前向與反向，得到梯度。", "First, one forward and backward pass at w gives the gradient."), 2: ("第二，算最壞擾動，走到最壞的鄰居。", "Second, compute the worst perturbation and go to the worst neighbor."), 3: ("第三，在最壞鄰居再做一次前向與反向，得到對抗梯度。", "Third, another pass at the worst neighbor gives the adversarial gradient."), 4: ("第四，回到原來的 w，用對抗梯度下降。", "Fourth, return to w and descend with the adversarial gradient."), 5: ("關鍵是第四步：梯度在最壞鄰居算出來，更新卻套用在 w 上。", "The key is step four: the gradient is taken at the worst neighbor but applied at w."), 6: ("用連鎖律，擾動對 w 的導數含海森，要多算二階資訊，SAM 直接忽略這一項。小批次訓練時兩次傳遞用同一個小批次，擾動當常數。", "By the chain rule the perturbation's derivative contains the Hessian, extra second-order work, and SAM simply ignores it. With minibatches both passes use the same one and the perturbation is a constant.")})

    g = d.page("碗上的第 1 步", "Step 1 on the bowl")
    rows = [[T("用的梯度", "gradient used"), r"$(10,\ 50)$", r"$(10.098,\ 62.257)$"], [T("新的 $w$", "new $w$"), r"$(9.800,\ 1.000)$", r"$(9.798,\ 0.755)$"], [T("新的 $L(w)$", "new $L(w)$"), "60.52", "55.12"], [T("新位置的鄰域最壞損失", "worst loss near the new point"), "76.79", "68.45"]]
    g.beat = 1
    table(g, 50, 120, [420, 340, 400], ["", T("一般梯度下降", "plain gradient descent"), "SAM"], rows, rh=76, fs=26, hfs=22, styles=[(CARD, GRAY), (CARD, GRAY), (GRNL, GRN), (GRNL, GRN)], beats=[1, 2, 3, 4], gap=10)
    g.beat = 5
    gold(g, 150, 530, 980, 90, T("SAM 的梯度範數 $63.07$ 是原梯度 $50.99$ 的 $1.24$ 倍:$w+\\epsilon^{*}$ 在更陡的位置。這是碗的性質,不是 SAM 的一般規律", "SAM's gradient norm $63.07$ is $1.24\\times$ the original $50.99$: $w+\\epsilon^{*}$ sits on a steeper spot. A property of this bowl, not a general rule"), size=20)
    banner(g, T("$\\eta=0.02$,$\\rho=0.5$:SAM 的一步走得多、損失降得多", "$\\eta=0.02$, $\\rho=0.5$: SAM's step moves further and lowers the loss more"))
    d.done(g, "第 1 步比較", "Step 1 compared", {0: ("碗上的第 1 步，學習率 0.02。一般梯度下降用的梯度是 10、50，SAM 用在最壞鄰居取的梯度 10.098、62.257。", "Step 1 on the bowl with learning rate 0.02. Plain gradient descent uses the gradient 10, 50; SAM uses the gradient taken at the worst neighbor, 10.098, 62.257."), 1: ("新的 w：一般是 9.800、1.000，SAM 是 9.798、0.755。", "The new w: 9.800, 1.000 for plain, 9.798, 0.755 for SAM."), 2: ("新的損失：60.52 對 55.12。", "The new loss: 60.52 versus 55.12."), 3: ("新位置的鄰域最壞損失：76.79 對 68.45。", "The worst neighborhood loss at the new point: 76.79 versus 68.45."), 4: ("", ""), 5: ("SAM 的梯度範數是原來的 1.24 倍，因為最壞鄰居在更陡的位置，所以 y 方向走得比較多。這是碗的性質，不是 SAM 的一般規律。", "SAM's gradient norm is 1.24 times the original because the worst neighbor sits on a steeper spot, so it moves more in y. That is this bowl's property, not a general rule.")})

    g = d.page("在碗上跑一百步", "A hundred steps on the bowl")
    ks = (0, 1, 10, 30, 60, 100)
    rows = [[str(k), f"{GD[k][0]:.4g} / {GD[k][1]:.4g}", f"{SM[k][0]:.4g} / {SM[k][1]:.4g}"] for k in ks]
    g.beat = 1
    table(g, 80, 120, [160, 460, 460], [T("步 $k$", "step $k$"), T("一般梯度下降:$L(w)$ / 鄰域最壞損失", "plain GD: $L(w)$ / worst nearby loss"), T("SAM:$L(w)$ / 鄰域最壞損失", "SAM: $L(w)$ / worst nearby loss")], rows, rh=54, fs=22, hfs=17, styles=[None, (GOLDL, GOLD), None, None, None, (REDL, RED)], beats=[1, 1, 2, 2, 3, 4], gap=8)
    g.beat = 5
    gold(g, 150, 548, 980, 76, T("前 60 步 SAM 的兩個數都較低;第 100 步 SAM 的 $L$ 略低,但鄰域最壞損失 $5.98$ 高於 $4.04$:固定步長在碗底來回跳", "SAM is lower in both columns up to step 60; at step 100 its $L$ is slightly lower but its worst nearby loss $5.98$ exceeds $4.04$: a fixed step bounces at the bottom"), size=19)
    banner(g, T("鄰域最壞損失用二次函數的精確解算出,不是一階估計", "the worst nearby loss uses the exact quadratic solution, not the first-order estimate"))
    d.done(g, "一百步", "A hundred steps", {0: ("同樣的起點與學習率，兩種方法各跑一百步，每一步算出鄰域最壞損失。起點兩者都是 100 與 128.53。", "Same start and learning rate, a hundred steps each, with the worst nearby loss at every step. Both start at 100 and 128.53."), 1: ("第 1 步與第 10 步：SAM 的兩個數都比較低。", "At steps 1 and 10 both of SAM's numbers are lower."), 2: ("第 30 步 SAM 是 13.71 與 17.40，一般是 14.88 與 18.62。", "At step 30 SAM has 13.71 and 17.40 against plain's 14.88 and 18.62."), 3: ("第 60 步 SAM 是 3.46 與 6.72，一般是 4.43 與 7.74。", "At step 60 SAM has 3.46 and 6.72 against 4.43 and 7.74."), 4: ("第 100 步 SAM 的損失 0.813 略低於 0.879，但鄰域最壞損失 5.98 高於 4.04：固定步長在碗底來回跳動。", "At step 100 SAM's loss 0.813 is slightly below 0.879 but its worst nearby loss 5.98 exceeds 4.04: a fixed step bounces at the bottom."), 5: ("", "")})
    closing(d, [(RED, REDL, "「SAM 保證找到平坦解」", "“SAM guarantees a flat solution”", "只是懲罰鄰域最壞損失;沒有保證,碗上固定步長還會彈跳", "it only penalizes the worst nearby loss: no guarantee; a fixed step even bounces on the bowl"), (ORG, ORGL, "「$\\epsilon^{*}$ 是內層問題的精確解」", "“$\\epsilon^{*}$ is the exact inner solution”", "它是一階近似的解;$\\rho$ 越大差越多", "it solves the first-order approximation; the larger $\\rho$, the larger the gap"), (PUR, PURL, "「$\\rho$ 越大越安全」", "“a larger $\\rho$ is safer”", "太大會跨出一個谷,一階近似也會失準", "too large crosses out of a valley and first order fails")],
            ("每步兩次前向加反向,成本高。能不能用更省的做法,或讓鄰域隨參數的尺度調整?", "Two forward-and-backward passes per step cost a lot. Can we be cheaper, or let the neighborhood adapt to the parameters' scale?"), ("下一個單元:高效 SAM 與自適應 SAM", "next unit: efficient and adaptive SAM"))
    return d.pages
