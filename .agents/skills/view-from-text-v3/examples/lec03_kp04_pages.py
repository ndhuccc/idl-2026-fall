import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lec03_kit import *

KP = "KP04"; N = 7
f5 = lambda w: 5 * w * w - 4 * w + 1
G0 = 16; C = 0.1
TR = []
for eta in (1, 0.5, 0.25, 0.125):
    w = 2 - 16 * eta; fv = f5(w); th = 13 - C * eta * G0 ** 2; TR.append((eta, w, fv, th, fv <= th))
assert [t[4] for t in TR] == [False, False, False, True] and [t[2] for t in TR] == [1037, 205, 29, 1] and [round(t[3], 1) for t in TR] == [-12.6, 0.2, 6.6, 9.8]
fb = lambda x, y: 0.5 * (x * x + 25 * y * y)
PATH = [(10.0, 2.0)]; ETAS = []
for _ in range(8):
    x, y = PATH[-1]; g = (x, 25 * y); eta = (g[0] ** 2 + g[1] ** 2) / (g[0] ** 2 + 25 * g[1] ** 2); ETAS.append(eta); PATH.append((x - eta * g[0], y - eta * g[1]))
assert round(ETAS[0], 5) == 0.04153 and round(fb(*PATH[8]), 4) == 0.2007 and 2600 / 62600 == ETAS[0]
G1 = (PATH[1][0], 25 * PATH[1][1]); assert abs(10 * G1[0] + 50 * G1[1]) < 1e-9

def make(lang):
    d = D(lang, KP, N); T = d.T
    g = d.page("KP04 不預先猜步長，每步實際試一試", "KP04 Do not guess the step: try it at every step")
    g.beat = 1
    chain(g, 50, 130, 1180, 230, [(RED, REDL, T("固定 $\\eta$", "fixed $\\eta$"), T("深度網路的 $L$ 幾乎算不出有用的值\n設大發散、設小太慢", "$L$ of a deep network is hard to estimate\ntoo big diverges, too small crawls")), (GRN, GRNL, T("線搜索", "line search"), T("每一步實際試一個步長\n損失沒降就縮小再試", "try a step size at every step\nshrink it if the loss does not fall"))], ts=26, bs=22, gap=100, beat0=1)
    g.beat = 3
    gold(g, 190, 400, 900, 110, T("不在出發前估計全域的常數,而是沿著下坡方向「試一試」", "do not estimate a global constant beforehand: try the downhill direction and look"), size=24)
    banner(g, T("看損失有沒有確實下降", "check that the loss really falls"))
    d.done(g, "為什麼需要線搜索", "Why line search", {0: ("KP02 說只要學習率小於 2 除以 L，損失就不會上升。但深度網路的 L 幾乎算不出有用的值，只能憑經驗設固定學習率，設大發散，設小太慢。", "KP02 said the loss never rises if the learning rate is below 2 over L. But L of a deep network is hard to compute usefully, so we set a fixed rate by experience: too big diverges, too small is slow."), 1: ("線搜索換一個想法：每一步實際試一試。", "Line search changes the idea: try it at every step."), 2: ("沿著下坡方向試一個步長，看損失有沒有確實下降；沒有，就縮小步長再試。", "Try a step size along the downhill direction and see whether the loss really falls; if not, shrink and try again."), 3: ("", "")})

    g = d.page("回溯:不降就減半", "Backtracking: halve until it decreases")
    g.beat = 1
    g.rect(120, 118, 1040, 100, fill=CARD, stroke=GRAY, sw=3, radius=16, z=3); g.text(130, 118, 1020, 100, r"$f\big(w^{(k)}-\eta g_k\big)\le f\big(w^{(k)}\big)-c\,\eta\,\|g_k\|_2^2$", size=36, color=INK, align="center", vcenter=True, z=8)
    g.text(120, 222, 1040, 40, T("充分下降條件(Armijo):至少要降這麼多", "sufficient decrease (Armijo): it must fall at least this much"), size=20, color=MUTED, bold=True, align="center", z=8)
    g.beat = 2
    ids = chain(g, 50, 290, 1180, 190, [(BLUE, BLUEL, T("1 試 $\\eta=1$", "1 try $\\eta=1$"), T("大膽試一個大步長", "boldly try a big step")), (ORG, ORGL, T("2 檢查條件", "2 check"), T("充分下降成立嗎?", "does sufficient decrease hold?")), (RED, REDL, T("3 不成立", "3 fails"), r"$\eta\leftarrow\eta/2$"), (GRN, GRNL, T("4 成立,更新", "4 holds: update"), r"$w^{(k+1)}=w^{(k)}-\eta g_k$")], ts=22, bs=22)
    g.link(ids[2], ids[1], 0, 0, RED, 4)
    g.beat = 3
    gold(g, 150, 510, 980, 100, T("為什麼從 $\\eta=1$ 開始?牛頓法對二次函數的自然步長就是 1,給模型最大膽的機會;$c$ 實務取 $10^{-4}$,本單元取 $0.1$", "why start at $\\eta=1$? Newton's natural step on a quadratic is 1, the boldest chance; in practice $c=10^{-4}$, here $c=0.1$"), size=21)
    banner(g, T("$L$-smooth 時一定會停:步長遲早小於 $2(1-c)/L$", "with $L$-smoothness it must stop: the step eventually falls below $2(1-c)/L$"))
    d.done(g, "回溯線搜索", "Backtracking", {0: ("只要求損失下降有漏洞：每次只降一點點，可能在離最小點很遠的地方停滯。所以採用充分下降條件，至少要降這麼多：步長越大、梯度越陡，要求降得越多。", "Merely requiring a decrease has a loophole: a tiny decrease each time may stall far from the minimum. So we use sufficient decrease: it must fall at least this much, more for bigger steps and steeper gradients."), 1: ("回溯四步：先試步長 1；檢查條件；不成立就減半；成立就用這個步長更新。", "Backtracking in four steps: try step 1; check the condition; halve if it fails; update with the step once it holds."), 2: ("從 1 開始是慣例：牛頓法對二次函數的自然步長就是 1。實務上 c 取很小，這裡取 0.1 讓數字清楚。", "Starting from 1 is a convention: Newton's natural step on a quadratic is 1. In practice c is tiny; here 0.1 keeps the numbers clear."), 3: ("", "")})

    g = d.page("完整算一遍", "One complete run")
    g.beat = 1
    g.text(50, 108, 1180, 36, T("$f(w)=5w^2-4w+1$,$w_0=2$:$g=16$,$f=13$,門檻 $13-25.6\\,\\eta$", "$f(w)=5w^2-4w+1$, $w_0=2$: $g=16$, $f=13$, threshold $13-25.6\\,\\eta$"), size=21, color=MUTED, bold=True, align="center", z=8)
    rows = [[("1" if e == 1 else f"1/{round(1 / e)}"), f"${w:g}$".replace("-", "-"), f"${fv:g}$", f"${th:.1f}$".replace(".0$", "$"), T("接受", "accepted") if ok else T("拒絕", "rejected")] for e, w, fv, th, ok in TR]
    table(g, 60, 150, [180, 220, 220, 260, 240], [T("試的 $\\eta$", "trial $\\eta$"), r"$w=2-16\eta$", "$f(w)$", T("門檻", "threshold"), T("判定", "verdict")], rows, rh=76, fs=28, styles=[(REDL, RED)] * 3 + [(GRNL, GRN)], beats=[1, 2, 3, 4], gap=10)
    g.beat = 5
    gold(g, 190, 556, 900, 60, T("接受 $\\eta=1/8$:一步從損失 $13$ 降到 $1$", "accept $\\eta=1/8$: one step from loss $13$ down to $1$"), size=26)
    banner(g, T("損失 $1037,205,29$ 都比門檻高:被拒絕", "losses $1037,205,29$ all exceed the threshold: rejected"))
    d.done(g, "逐列驗算", "Row by row", {0: ("主例子從 w0 等於 2 出發，梯度 16，損失 13，充分下降的門檻是 13 減 25.6 倍步長。", "The main example starts at w0 equal 2 with gradient 16 and loss 13; the threshold is 13 minus 25.6 times the step."), 1: ("步長 1：w 等於負 14，損失 1037，門檻負 12.6，拒絕。", "Step 1: w is minus 14, loss 1037, threshold minus 12.6: rejected."), 2: ("步長二分之一：損失 205，門檻 0.2，拒絕。", "Step one half: loss 205, threshold 0.2: rejected."), 3: ("步長四分之一：損失 29，門檻 6.6，拒絕。", "Step one quarter: loss 29, threshold 6.6: rejected."), 4: ("步長八分之一：w 等於 0，損失 1，小於門檻 9.8，接受。", "Step one eighth: w is 0, loss 1, below the threshold 9.8: accepted."), 5: ("一步從損失 13 降到 1。", "One step takes the loss from 13 to 1.")})

    g = d.page("最陡下降:每步精算最佳步長", "Steepest descent: the best step at every iteration")
    xr, yr = (-12, 12), (-4, 4)
    ux, uy = axes(g, 50, 120, 760, 480, xr, yr)
    g.beat = 1
    bowl_contours(g, ux, uy, levels=(4, 16, 36, 64))
    g.line([(ux(x), uy(y)) for x, y in PATH], ORG, sw=4, z=6)
    for x, y in PATH: dot(g, ux(x), uy(y), 6, ORG)
    g.text(50, 604, 760, 34, T("精確線搜索 8 步:仍是之字形", "exact line search, 8 steps: still a zigzag"), size=18, color=MUTED, bold=True, align="center", z=8)
    g.beat = 2
    box(g, 850, 126, 380, 170, BLUE, BLUEL, T("二次函數的閉式解", "closed form for quadratics"), r"$\eta_k=\dfrac{g_k^\top g_k}{g_k^\top Hg_k}$".replace(r"\dfrac", r"\frac"), ts=21, bs=30)
    g.beat = 3
    box(g, 850, 320, 380, 150, PUR, PURL, T("玩具碗第一步", "toy bowl, step one"), r"$\eta_0=\frac{2600}{62600}\approx0.04153$", ts=21, bs=26)
    g.beat = 4
    gold(g, 850, 495, 380, 110, T("8 步後損失 $0.2007$", "loss $0.2007$ after 8 steps"), size=26)
    banner(g, T("每步在下坡射線上把一維問題解到底:$\\eta_k=\\arg\\min\\phi(\\eta)$", "solve the 1-D problem to the end each step: $\\eta_k=\\arg\\min\\phi(\\eta)$"))
    d.done(g, "最陡下降", "Steepest descent", {0: ("回溯只要夠好。最陡下降更徹底：每一步把一維問題解到底，選出讓損失最小的步長。在玩具碗上走八步，路徑仍是之字形。", "Backtracking only needs good enough. Steepest descent goes all the way: each step solves the 1-D problem exactly, choosing the step that minimizes the loss. On the toy bowl, eight steps still zigzag."), 1: ("對二次函數有閉式解：步長等於梯度內積除以梯度乘海森再乘梯度。", "For quadratics there is a closed form: the step equals the gradient's inner product divided by the gradient times the Hessian times the gradient."), 2: ("玩具碗第一步：2600 除以 62600，約 0.04153。", "Toy bowl, first step: 2600 over 62600, about 0.04153."), 3: ("八步後損失是 0.2007。", "After eight steps the loss is 0.2007."), 4: ("", "")})

    g = d.page("相鄰兩步的梯度互相垂直", "Adjacent gradients are perpendicular")
    g.beat = 1
    ids = chain(g, 50, 130, 1180, 190, [(BLUE, BLUEL, r"$g_0$", r"$(10,\,50)$"), (ORG, ORGL, r"$g_1$", r"$(9.5847,\,-1.9169)$")], ts=30, bs=30, gap=120, beat0=1)
    g.beat = 3
    box(g, 190, 360, 900, 130, PUR, PURL, T("內積", "inner product"), r"$g_0^\top g_1=10\cdot9.5847+50\cdot(-1.9169)=0$", ts=22, bs=26)
    g.beat = 4
    gold(g, 190, 510, 900, 100, T("原因:$\\eta_k$ 使 $\\phi'(\\eta_k)=0$,所以 $\\nabla f(w^{(k+1)})^\\top g_k=0$。垂直轉彎,路徑仍是之字形", "because $\\phi'(\\eta_k)=0$, $\\nabla f(w^{(k+1)})^\\top g_k=0$: right-angle turns, so still a zigzag"), size=21)
    banner(g, T("最陡下降每步效果好,但路徑並沒有變直", "each step is good, but the path does not straighten"))
    d.done(g, "垂直", "Perpendicular", {0: ("第一步的梯度是 10、50。", "The first gradient is 10, 50."), 1: ("第二步的梯度是 9.5847、負 1.9169。", "The second gradient is 9.5847, minus 1.9169."), 2: ("內積為零，兩個梯度互相垂直。", "Their inner product is zero: the gradients are perpendicular."), 3: ("原因是最佳步長使一維函數的導數為零，所以新位置的梯度和舊梯度的內積是零。每一步都垂直轉彎，路徑仍是之字形。", "The best step makes the 1-D derivative zero, so the new gradient is orthogonal to the old one. Every step turns at a right angle, so the path remains a zigzag.")})

    g = d.page("為什麼深度學習很少做完整線搜索", "Why deep learning seldom runs a full line search")
    chain(g, 50, 130, 1180, 330, [(RED, REDL, T("前向傳播的成本", "forward-pass cost"), T("每次試探都要重新算一次損失,也就是重新跑一次網路的前向傳播", "every trial recomputes the loss: another forward pass of the network")), (ORG, ORGL, T("小批次的雜訊", "minibatch noise"), T("在當前這一批找到的最佳步長,不代表全部資料上損失也下降(KP07、KP08)", "the best step on this batch need not lower the loss on all the data (KP07, KP08)")), (PUR, PURL, T("動態分支", "dynamic branching"), T("「不成立就減半」是依條件執行的迴圈,降低大規模平行運算的效率", "“halve if it fails” is a conditional loop that hurts large-scale parallel execution"))], ts=24, bs=22, beat0=1, th=70)
    g.beat = 4
    gold(g, 190, 500, 900, 110, T("深度學習多半改用固定的、依排程遞減的或自適應的學習率(KP10、KP11);整批問題與 L-BFGS 仍常用線搜索", "deep learning mostly uses fixed, scheduled or adaptive rates (KP10, KP11); full-batch problems and L-BFGS still use line search"), size=21)
    banner(g, T("線搜索把「步長要看實際的局部證據」講清楚了", "line search makes clear that the step must follow real local evidence"))
    d.done(g, "三個原因", "Three reasons", {0: ("既然每步有效，為什麼深度學習不用？第一是成本：每次試探都要重跑一次前向傳播。", "If every step is good, why does deep learning not use it? First, cost: each trial reruns a forward pass."), 1: ("第二是小批次的雜訊：在當前這一批找到的最佳步長，不代表全部資料上損失也下降。", "Second, minibatch noise: the best step on this batch does not mean the loss falls on all the data."), 2: ("第三是動態分支：不成立就減半是依條件執行的迴圈，會降低大規模平行運算的效率。", "Third, dynamic branching: halve if it fails is a conditional loop that hurts large-scale parallel execution."), 3: ("所以深度學習多半改用固定的、依排程遞減的或自適應的學習率。線搜索仍是重要概念，在整批問題或 L-BFGS 中仍常用。", "So deep learning mostly uses fixed, scheduled or adaptive learning rates. Line search remains an important idea and is still common in full-batch problems and L-BFGS."), 4: ("", "")})
    closing(d, [(RED, REDL, "「步子越大進展越快」", "“bigger steps always make faster progress”", "$\\eta=1$ 常被拒絕,損失可能暴增", "$\\eta=1$ is often rejected and the loss can blow up"), (ORG, ORGL, "「損失降一點就夠了」", "“any small decrease is enough”", "可能原地假死,所以要用充分下降條件", "it may stall in place, hence sufficient decrease"), (PUR, PURL, "「精確線搜索一定最好」", "“exact line search is always best”", "每步好但成本高,路徑仍是之字形", "good per step but costly, and the path still zigzags")],
            ("之字形(相鄰梯度互相垂直)能不能抑制?", "Can the zigzag (adjacent gradients perpendicular) be suppressed?"), ("下一個單元:動量", "next unit: momentum"))
    return d.pages
