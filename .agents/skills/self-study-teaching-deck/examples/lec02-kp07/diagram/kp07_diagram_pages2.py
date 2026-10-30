"""第 16–29 頁(Hinge、選讀、整理、迷思、銜接、練習、自我檢核)。由 kp07_diagram_pages.make 呼叫。"""
import math


def build(d, pg, fin, fml, ticks, G):
    globals().update(G)
    T = d.T
    HG = lambda v: max(0.0, 1 - v)

    # 16 ──────────────────────────────────────────── Hinge 曲線
    g = pg(d, 16); g.beat = 0
    xr, yr = (-2, 3), (0, 3); ux, uy = axes(g, 60, 140, 640, 430, xr, yr); ticks(g, ux, uy, (-2, -1, 0, 1, 2, 3), 576)
    lab(g, 300, 600, 400, T("橫軸:$y\\,f_o$(標籤乘分數)", "x-axis: $y\\,f_o$ (label times score)"), size=18, col=MUTED)
    g.beat = 1
    g.line([(ux(-2), uy(1)), (ux(0), uy(1))], GRAY, sw=4, dash=True, z=4); g.line([(ux(0), uy(0)), (ux(3), uy(0))], GRAY, sw=4, dash=True, z=4)
    lab(g, ux(-1.9), uy(1) - 36, 230, T("0-1 損失", "0-1 loss"), size=18, col=MUTED)
    g.rect(ux(0), 142, ux(1) - ux(0), 426, fill=GOLDL, fillOpacity=0.7, z=2)
    g.beat = 2
    g.line([(ux(-2), uy(3)), (ux(1), uy(0)), (ux(3), uy(0))], ORG, sw=6, z=5); lab(g, ux(-1.9), uy(2.8), 200, "hinge", size=20, col=ORG)
    for v, col in zip((1.5, 0.3, -0.5), (GRN, ORG, RED)):
        dot(g, ux(v), uy(HG(v)), 12, col); lab(g, ux(v) - 60, uy(HG(v)) - 40 if v != -0.5 else uy(HG(v)) - 40, 120, f"${HG(v):g}$", size=20, col=col, align="center")
    fml(g, 740, 150, 490, 80, r"$\ell=\max(0,\,m-y f_o),\ m=1$", size=26)
    fml(g, 740, 250, 490, 70, r"$y\in\{-1,+1\}$", size=26, fill=BLUEL, stroke=BLUE)
    fml(g, 740, 340, 490, 70, r"$y f_o\ge m\Rightarrow\ell=0$", size=26, fill=GRNL, stroke=GRN)
    gold(g, 740, 430, 490, 120, T("黃色帶:分對了\n但還不到 $m$,也要罰", "Yellow band: right side\nbut still within $m$, punished"), size=22)
    fin(d, g, 16, T("Hinge 用在二分類,標籤 $y$ 是 -1 或 +1,損失是 $\\max(0,\\,m-y f_o)$,這裡 $m=1$。灰色虛線是只問對錯的 0-1 損失:分錯才是 1。橘色 hinge 在 $y f_o<1$ 時就開始罰,所以黃色帶裡「分對了但太靠近邊界」也有損失。三個標出的點損失是 0、0.7、1.5。", "Hinge is for binary classification with label $y$ equal to -1 or +1; the loss is $\\max(0,\\,m-y f_o)$ with $m=1$. The grey dashed line is the 0-1 loss that asks only right or wrong: 1 only when wrong. The orange hinge starts punishing already when $y f_o<1$, so a point in the yellow band, right but too close to the boundary, still has loss. The three marked points have loss 0, 0.7, 1.5."))

    # 17 ──────────────────────────────────────────── Hinge 算例
    g = pg(d, 17); g.beat = 0
    fml(g, 330, 130, 620, 60, r"$y=+1,\quad m=1$", size=28, fill=GOLDL, stroke=GOLD)
    rows = [[f"${v:g}$", f"${v:g}$", f"$1-{v:g}={1-v:g}$" if v >= 0 else f"$1-({v:g})={1-v:g}$", f"${HG(v):g}$", t] for v, t in ((1.5, T("安全:分對而且超過邊界", "safe: right and beyond the margin")), (0.3, T("太近:分對了,仍被罰", "too close: right, still punished")), (-0.5, T("分錯", "wrong")))]
    table(g, 60, 205, (130, 150, 280, 130, 470), ["$f_o$", "$y f_o$", r"$m-y f_o$", r"$\ell$", T("意思", "meaning")], rows, rh=70, fs=22, styles=[(GRNL, GRN), (GOLDL, GOLD), (REDL, RED)], beats=[1, 2, 2])
    g.beat = 1; tick(g, 1160, 288, 36, GRN)
    g.beat = 2; g.rect(1160, 358, 40, 36, fill=GOLD, radius=6, z=6); lab(g, 1160, 356, 40, "!", size=26, col="#ffffff", align="center"); cross(g, 1162, 428, 34, RED)
    gold(g, 150, 540, 980, 74, T("第二個點分對了($0.3>0$),卻因離邊界不到 $m$ 被罰 $0.7$", "The second point is right ($0.3>0$) yet punished by $0.7$: it is within $m$"), size=24)
    fin(d, g, 17, T("取 $y=+1$、$m=1$,代入三個分數。$f_o=1.5$:標籤乘分數是 1.5,$1-1.5=-0.5$,取 $\\max(0,\\cdot)$ 得 0,安全。$f_o=0.3$:$1-0.3=0.7$,分對了但太近,罰 0.7。$f_o=-0.5$:$1-(-0.5)=1.5$,分錯,罰 1.5。注意要先乘標籤,不能只看絕對值。", "Take $y=+1$, $m=1$ and plug in three scores. $f_o=1.5$: $y f_o=1.5$, $1-1.5=-0.5$, and $\\max(0,\\cdot)$ gives 0: safe. $f_o=0.3$: $1-0.3=0.7$; right but too close, punished 0.7. $f_o=-0.5$: $1-(-0.5)=1.5$; wrong, punished 1.5. Multiply by the label first; do not look at the absolute value only."))
    assert [HG(v) for v in (1.5, 0.3, -0.5)] == [0, 0.7, 1.5]

    # 18 ──────────────────────────────────────────── 分數餘裕不是距離
    g = pg(d, 18); g.beat = 0
    u = lambda x: 480 + x * 500
    for k, (w_, s_, hv, beat) in enumerate(((1, 0.3, 0.7, 1), (2, 0.6, 0.4, 2))):
        y0 = 140 + k * 200; g.beat = beat
        g.rect(60, y0, 1170, 180, fill=CARD, stroke=BLUE if k == 0 else ORG, sw=3, radius=16, z=1)
        lab(g, 90, y0 + 20, 340, f"$(w,b)=({w_},0)$", size=28, col=BLUE if k == 0 else ORG)
        lab(g, 90, y0 + 90, 340, T(f"換尺:$(2w,2b)$" if k else "原來的尺", "rescaled: $(2w,2b)$" if k else "original scale"), size=20, col=MUTED)
        yy = y0 + 95; g.line([(u(-0.15), yy), (u(0.8), yy)], GRAY, sw=3, z=3); g.line([(u(0), y0 + 30), (u(0), y0 + 160)], INK, sw=4, dash=True, z=3); lab(g, u(0) - 70, y0 + 4, 140, T("邊界", "boundary"), size=18, col=INK, align="center")
        dot(g, u(0.3), yy, 12, GRN); lab(g, u(0.3) - 50, yy - 52, 100, "$x=0.3$", size=20, col=GRN, align="center")
        g.line([(u(0), yy + 30), (u(0.3), yy + 30)], RED, sw=4, z=4); lab(g, u(0) + 20, yy + 34, 160, "$d=0.3$", size=20, col=RED)
        lab(g, 850, y0 + 28, 360, f"$s={s_:g},\\ d=0.3$", size=26, col=INK); lab(g, 850, y0 + 100, 360, f"$\\ell=\\max(0,1-{s_:g})={hv:g}$", size=26, col=ORG)
    g.beat = 0; fml(g, 330, 540, 620, 92, r"$s=\mathbf w^\top\mathbf x+b,\quad d=\frac{s}{\|\mathbf w\|_2}$", size=28, fill=GOLDL, stroke=GOLD)
    fin(d, g, 18, T("分數 $s=\\mathbf w^\\top\\mathbf x+b$ 不是幾何距離,幾何距離要再除以 $\\|\\mathbf w\\|_2$。同一個點 $x=0.3$、同一條邊界,把 $(w,b)$ 同時放大 2 倍,分數從 0.3 變成 0.6,hinge 損失從 0.7 變成 0.4,可是點到邊界的距離還是 0.3。所以 Hinge 的「餘裕 $m$」是分數的門檻,不是公尺。此頁是補充說明。", "The score $s=\\mathbf w^\\top\\mathbf x+b$ is not a geometric distance; the distance divides by $\\|\\mathbf w\\|_2$. For the same point $x=0.3$ and the same boundary, doubling $(w,b)$ turns the score from 0.3 into 0.6 and the hinge loss from 0.7 into 0.4, while the distance to the boundary stays 0.3. So hinge's margin $m$ is a threshold on the score, not a length. This page is an added explanation."))

    # 19 ──────────────────────────────────────────── Huber 接點
    g = pg(d, 19); g.beat = 0
    xr, yr = (-3, 3), (0, 5); ux, uy = axes(g, 60, 140, 600, 420, xr, yr); ticks(g, ux, uy, (-3, -1, 0, 1, 3), 566)
    g.beat = 1; curve(g, ux, uy, hub, GRN, xr, yr, sw=6)
    g.beat = 2
    curve(g, ux, uy, lambda e: 0.5 * e * e, ORG, (0.2, 2.2), yr, sw=3, n=40, dash=True); curve(g, ux, uy, lambda e: e - 0.5, BLUE, (0.0, 1.0), yr, sw=3, n=10, dash=True)
    dot(g, ux(1), uy(0.5), 13, GOLD, z=8); dot(g, ux(-1), uy(0.5), 13, GOLD, z=8)
    lab(g, ux(1) + 16, uy(0.5) + 6, 150, "$(\\delta,\\tfrac12\\delta^2)$", size=18, col=INK)
    fml(g, 700, 150, 530, 80, r"$h_\delta(\pm\delta)=\tfrac12\delta^2$", size=28, fill=GRNL, stroke=GRN)
    fml(g, 700, 250, 530, 70, r"$h_\delta'(+\delta)=+\delta$", size=28, fill=ORGL, stroke=ORG); fml(g, 700, 340, 530, 70, r"$h_\delta'(-\delta)=-\delta$", size=28, fill=BLUEL, stroke=BLUE)
    gold(g, 700, 440, 530, 110, T("值接上了,一次導數也接上;\n二次導數在接點仍會切換", "values match and first derivatives match;\nthe second derivative still switches"), size=22)
    fin(d, g, 19, T("(選讀)Huber 的二次段和線性段在 $|e|=\\delta$ 相接。二次段在這裡的值是 $\\tfrac12\\delta^2$、斜率 $\\delta$;線性段 $\\delta(\\delta-\\tfrac12\\delta)=\\tfrac12\\delta^2$,斜率也是 $\\delta$。值和斜率都相同,所以整條曲線平滑沒有折角。二次導數在接點則從 1 切到 0。另外,二元交叉熵的負值就是 KP08 要用的對數似然。", "(Optional) Huber's quadratic and linear pieces meet at $|e|=\\delta$. The quadratic piece has value $\\tfrac12\\delta^2$ and slope $\\delta$ there; the linear piece gives $\\delta(\\delta-\\tfrac12\\delta)=\\tfrac12\\delta^2$ and slope $\\delta$ too. Values and slopes agree, so the curve is smooth with no corner. The second derivative switches from 1 to 0 at the joint. Also, the negative of binary cross-entropy is the log-likelihood used in KP08."))

    # 20 ──────────────────────────────────────────── 選擇表
    g = pg(d, 20); g.beat = 0
    rows = [[T("回歸(實數目標)", "regression (real target)"), "MSE / MAE / Huber", T("MSE 處處可導但放大離群值;MAE 穩健但零點不可導;Huber 折衷、要調 $\\delta$", "MSE smooth but amplifies outliers; MAE robust but not differentiable at 0; Huber in between, tune $\\delta$")],
            [T("分類(互斥多類)", "classification (exclusive classes)"), T("交叉熵 + softmax", "cross-entropy + softmax"), T("輸出要像機率:落在 $[0,1]$ 且和為 1", "outputs must look like probabilities: in $[0,1]$, sum 1")],
            [T("分類(二元)", "classification (binary)"), T("交叉熵 + sigmoid", "cross-entropy + sigmoid"), T("兩種結果都寫進式子", "both outcomes in the formula")],
            [T("目標是機率分布(軟標籤)", "target is a distribution (soft label)"), T("KL 散度", "KL divergence"), T("例如知識蒸餾;固定目標時和交叉熵差一個常數", "e.g. distillation; differs from cross-entropy by a constant for a fixed target")],
            [T("二分類、要留餘裕", "binary, keep a margin"), "Hinge", T("連續;分對但離邊界太近也罰", "continuous; punishes right-but-too-close")]]
    table(g, 60, 135, (310, 280, 590), [T("任務", "task"), T("損失", "loss"), T("特點", "key point")], rows, rh=64, fs=20, hfs=20, styles=[(REDL, RED), (BLUEL, BLUE), (BLUEL, BLUE), (PURL, PUR), (GRNL, GRN)], beats=[1, 2, 2, 2, 2], gap=8)
    g.beat = 3
    gold(g, 100, 566, 1080, 66, T("像練射箭:損失是箭離靶心的距離;分類量的是「預測機率」和答案差多少", "Like archery: loss is the arrow's distance from the bullseye;\nclassification measures probability against the answer"), size=20)
    fin(d, g, 20, T("六種損失整理成一張表。先看任務是什麼:目標是實數就選回歸的三把尺;目標是類別就用交叉熵(輸出層配 softmax 或 sigmoid);目標本身是分布用 KL;二分類想留餘裕用 Hinge。白話比喻:像練射箭,損失是箭離靶心的距離;MSE 把距離平方,離很遠的箭被重罰。", "The six losses in one table. Start from the task: a real-valued target uses the three regression rulers; a class target uses cross-entropy (softmax or sigmoid in the output layer); a distribution target uses KL; a binary margin uses hinge. Archery analogy: loss is the distance of the arrow from the bullseye; MSE squares it, so a far arrow is heavily punished."))

    # 21 ──────────────────────────────────────────── 迷思
    g = pg(d, 21); g.beat = 0
    items = [(T("「損失越小,網路一定越好」", "“The smaller the loss, the better the network.”"), T("不對:損失只在訓練資料上量到,太小可能只是把訓練資料背起來(KP04 的泛化問題)", "No: the loss is measured on the training data only; a very small loss may just mean the data was memorized (KP04's generalization problem)")),
             (T("「不同損失的數值可以直接比大小」", "“Values of different losses can be compared directly.”"), T("不能:MSE 的 $103$ 與 MAE 的 $13$ 沒有誰比較大的意義,它們是不同的尺", "No: MSE's $103$ and MAE's $13$ say nothing about which is larger; they are different rulers"))]
    for k, (m_, f_) in enumerate(items):
        x = 60 + k * 600
        g.beat = 1; box(g, x, 140, 570, 170, RED, REDL, T("迷思", "myth"), m_, ts=24, bs=28, th=50); cross(g, x + 20, 150, 28, RED)
        g.beat = 2; box(g, x, 335, 570, 250, GRN, GRNL, T("正解", "the fix"), f_, ts=24, bs=28, th=50, bold=False); tick(g, x + 20, 345, 30, GRN)
    fin(d, g, 21, T("兩個常見迷思。第一,損失小不代表網路一定好,因為損失只在訓練資料上量,太小可能只是把訓練題背起來。第二,不同損失的數值不能直接比大小,103 和 13 是兩把不同的尺量出來的。", "Two common misconceptions. First, a small loss does not guarantee a good network: it is measured on the training data only, and a tiny loss may mean memorizing. Second, values of different losses cannot be compared directly: 103 and 13 come from two different rulers."))

    # 22 ──────────────────────────────────────────── 銜接 KP08
    g = pg(d, 22); g.beat = 0
    xr, yr = (-2, 2), (0, 4)
    ux, uy = axes(g, 60, 140, 520, 330, (-2, 2), (0, 1.5)); lab(g, 70, 144, 300, T("0-1 損失(只數錯幾個)", "0-1 loss (counts mistakes)"), size=20, col=MUTED)
    g.beat = 1; g.line([(ux(-2), uy(1)), (ux(0), uy(1))], GRAY, sw=6, z=4); g.line([(ux(0), uy(0)), (ux(2), uy(0))], GRAY, sw=6, z=4); lab(g, 60, 476, 520, T("離散:一階一階跳,沒有斜率可走", "discrete: it jumps, with no slope to follow"), size=20, col=MUTED, align="center")
    ux2, uy2 = axes(g, 700, 140, 520, 330, (-2, 2), (0, 4)); g.beat = 2; lab(g, 710, 144, 300, T("平方誤差 $\\ell=(f_o-y)^2$", "squared error $\\ell=(f_o-y)^2$"), size=20, col=RED)
    curve(g, ux2, uy2, lambda e: e * e, RED, (-2, 2), (0, 4), sw=6); dot(g, ux2(-1.6), uy2(2.56), 13, BLUE)
    g.line([(ux2(-1.5), uy2(2.4)), (ux2(-0.5), uy2(0.35))], BLUE, sw=5, z=6); g.poly([(ux2(-0.4), uy2(0.12)), (ux2(-0.62), uy2(0.45)), (ux2(-0.9), uy2(0.2))], fill=BLUE, z=6)
    lab(g, 700, 476, 520, T("連續:有斜率,可以一步一步往下走", "continuous: a slope to walk down, step by step"), size=20, col=RED, align="center")
    gold(g, 150, 520, 980, 96, T("Perceptron 只數「錯幾個」,不能用梯度學;\n改成連續的平方誤差,就能往下走——這就是 ADALINE(KP08)", "The Perceptron only counts mistakes and cannot learn by gradient;\nswitch to the continuous squared error and we can walk downhill: ADALINE (KP08)"), size=22)
    fin(d, g, 22, T("留給下一個知識點的問題:Perceptron 的損失只數「錯幾個」,是離散的,不能用梯度學。如果改成連續的平方誤差,就能一步一步往下走。這個想法就是 ADALINE,下一個單元(KP08)的主題。", "The question left for the next knowledge point: the Perceptron's loss only counts how many are wrong, which is discrete and cannot be learned by gradient. With a continuous squared error we can go down step by step. That idea is ADALINE, the topic of the next unit (KP08)."))

    # 23 ──────────────────────────────────────────── 練習 1
    EX = (2, -1, 0.5, 6); MS = sum(e * e for e in EX); MA = sum(map(abs, EX)); HU = [hub(e) for e in EX]
    assert MS == 41.25 and MA == 9.5 and HU == [1.5, 0.5, 0.125, 5.5] and sum(HU) == 7.625 and round(36 / MS * 100) == 87 and round(5.5 / sum(HU) * 100) == 72
    g = pg(d, 23); g.beat = 0
    fml(g, 60, 135, 1170, 70, T(r"殘差 $\mathbf e=(2,-1,0.5,6),\ \delta=1$:算 MSE、MAE、Huber 的總和", r"residuals $\mathbf e=(2,-1,0.5,6),\ \delta=1$: sums of MSE, MAE, Huber"), size=26, fill=GOLDL, stroke=GOLD)
    g.beat = 1; lab(g, 60, 215, 1170, T("提示:Huber 要先判斷每一筆落在二次段還是線性段;先自己算,再看解答", "Hint: for Huber first decide, per point, quadratic or linear piece. Try first, then see the solution."), size=20, col=MUTED)
    g.beat = 2
    fml(g, 60, 262, 1170, 62, r"$\mathrm{MSE}:\ 2^2+1^2+0.5^2+6^2=4+1+0.25+36=41.25$", size=24, fill=REDL, stroke=RED)
    fml(g, 60, 334, 1170, 62, r"$\mathrm{MAE}:\ 2+1+0.5+6=9.5$", size=24, fill=BLUEL, stroke=BLUE)
    fml(g, 60, 406, 1170, 62, r"$\mathrm{Huber}:\ 1.5+0.5+0.125+5.5=7.625$", size=24, fill=GRNL, stroke=GRN)
    lab(g, 70, 474, 1160, T("$|2|>1$ 線性 $1\\times(2-0.5)$;$|-1|\\le1$ 二次 $\\tfrac12$;$0.5$ 二次 $\\tfrac12\\cdot0.25$;$6$ 線性 $5.5$", "$|2|>1$ linear $1\\times(2-0.5)$; $|-1|\\le1$ quadratic $\\tfrac12$; $0.5$ quadratic $\\tfrac12\\cdot0.25$; $6$ linear $5.5$"), size=18, col=MUTED)
    fml(g, 60, 524, 1170, 70, T(r"檢查:離群值 6 占 MSE $36/41.25\approx87\%$,占 Huber $5.5/7.625\approx72\%$", r"check: the outlier 6 is $36/41.25\approx87\%$ of MSE but $5.5/7.625\approx72\%$ of Huber"), size=22, fill=GOLDL, stroke=GOLD)
    fin(d, g, 23, T("練習一:殘差 $(2,-1,0.5,6)$。MSE:$4+1+0.25+36=41.25$。MAE:$2+1+0.5+6=9.5$。Huber($\\delta=1$)先判斷每筆落在哪一段:2 在線性段得 1.5,$-1$ 和 0.5 在二次段得 0.5 與 0.125,6 在線性段得 5.5,合計 7.625。檢查:離群值 6 在 MSE 占 87%,在 Huber 只占 72%,和第 1 節的規律一致。", "Exercise 1: residuals $(2,-1,0.5,6)$. MSE: $4+1+0.25+36=41.25$. MAE: $2+1+0.5+6=9.5$. Huber ($\\delta=1$): first decide the piece for each point: 2 is linear giving 1.5; $-1$ and 0.5 are quadratic giving 0.5 and 0.125; 6 is linear giving 5.5; total 7.625. Check: the outlier 6 is 87% of MSE but 72% of Huber, consistent with section 1."))

    # 24 ──────────────────────────────────────────── 練習 2
    g = pg(d, 24); g.beat = 0
    fml(g, 60, 135, 1170, 70, T(r"3 類 softmax 的輸入 $\mathbf a=(0,0,0)$,正解任選:$\ell=-\ln p_{\rm true}=?$", r"3-class softmax input $\mathbf a=(0,0,0)$, any right class: $\ell=-\ln p_{\rm true}=?$"), size=24, fill=GOLDL, stroke=GOLD)
    g.beat = 1; lab(g, 60, 215, 1170, T("提示:三個機率各是多少?", "Hint: what is each of the three probabilities?"), size=20, col=MUTED)
    g.beat = 2
    for k in range(3):
        x = 250 + k * 280; g.rect(x, 480 - 330 / 3, 150, 330 / 3, fill=BLUE, radius=6, z=3); lab(g, x - 20, 480 - 110 - 40, 190, r"$p=\tfrac13$", size=24, col=BLUE, align="center"); lab(g, x - 20, 488, 190, f"{T('第','class ')}{k+1}{T(' 類','')}", size=20, col=MUTED, align="center")
    g.line([(200, 480), (1050, 480)], INK, sw=3, z=3)
    fml(g, 200, 535, 880, 80, r"$-\ln\tfrac13=\ln3\approx1.099$", size=32, fill=GRNL, stroke=GRN)
    fin(d, g, 24, T("練習二:輸入全是 0,指數都是 1,所以 softmax 三個機率各是 1/3。不管正解是哪一類,損失都是 $-\\ln\\tfrac13=\\ln3\\approx1.099$。(此參考答案是補充,講義只給提示。)", "Exercise 2: all inputs 0 give exponentials 1, so each softmax probability is 1/3. Whatever the right class, the loss is $-\\ln\\tfrac13=\\ln3\\approx1.099$. (This reference answer is added; the lecture gives only the hint.)"))
    assert round(math.log(3), 3) == 1.099

    # 25 ──────────────────────────────────────────── 練習 3
    g = pg(d, 25); g.beat = 0
    fml(g, 60, 135, 1170, 70, T(r"Hinge($m=1$),$y=-1$、$f_o=0.8$:損失是多少?它分對了嗎?", r"Hinge ($m=1$), $y=-1$, $f_o=0.8$: what is the loss, and is it right?"), size=24, fill=GOLDL, stroke=GOLD)
    g.beat = 1; lab(g, 60, 215, 1170, T("提示:先算 $y f_o$(別漏掉標籤的負號)", "Hint: compute $y f_o$ first (do not drop the label's minus sign)"), size=20, col=MUTED)
    g.beat = 2
    chain(g, 60, 290, 1170, 200, [(BLUE, BLUEL, T("① 先乘標籤", "① multiply by the label"), r"$y f_o=(-1)(0.8)=-0.8$"), (ORG, ORGL, T("② 代入 Hinge", "② plug into hinge"), r"$\max(0,\,1-(-0.8))=1.8$"), (RED, REDL, T("③ 判斷", "③ decide"), T(r"$y f_o<0$:分錯", r"$y f_o<0$: wrong"))], gap=60, ts=22, bs=24, th=60)
    lab(g, 60, 510, 1170, T("正分數 0.8 遇到負標籤:分數的方向和標籤相反", "A positive score 0.8 meets a negative label: the score points the opposite way"), size=22, col=MUTED, align="center")
    fin(d, g, 25, T("練習三:先乘標籤:$y f_o=(-1)(0.8)=-0.8$。代入 hinge:$\\max(0,1-(-0.8))=1.8$。因為 $y f_o<0$,這個點分錯。標籤是負的、分數卻是正的,方向相反,所以損失大。(此參考答案是補充,講義只給提示。)", "Exercise 3: multiply by the label first: $y f_o=(-1)(0.8)=-0.8$. Plug into hinge: $\\max(0,1-(-0.8))=1.8$. Since $y f_o<0$, the point is misclassified: a negative label with a positive score points the opposite way, hence the large loss. (This reference answer is added; the lecture gives only the hint.)"))
    assert HG(-0.8) == 1.8

    # 26–29 ───────────────────────────────────────── 自我檢核
    def quiz(i, q, opts, hint, ans, ex):
        g = pg(d, i); g.beat = 0
        fml(g, 60, 135, 1170, 80, q, size=24, fill=GOLDL, stroke=GOLD)
        pos = []
        for k, o in enumerate(opts):
            x = 60 + (k % 2) * 600; y = 240 + (k // 2) * 125; pos.append((x, y))
            g.rect(x, y, 570, 105, fill=CARD, stroke=BLUE, sw=3, radius=14, z=3); g.rect(x + 12, y + 28, 48, 48, fill=BLUE, radius=24, z=4); g.text(x + 12, y + 30, 48, 44, "ABCD"[k], size=26, color="#ffffff", bold=True, align="center", z=8)
            g.text(x + 72, y + 24, 430, 74, o, size=22, color=INK, bold=True, z=8)
        g.beat = 1; lab(g, 60, 500, 1170, T("提示:", "Hint: ") + hint, size=20, col=MUTED)
        g.beat = 2; x, y = pos[ans]
        g.rect(x - 4, y - 4, 578, 113, fill="transparent", stroke=GRN, sw=6, radius=16, z=9); tick(g, x + 524, y + 8, 30, GRN, z=9)
        lab(g, 60, 545, 1170, T("參考答案:", "Reference answer: ") + "ABCD"[ans] + T("(依講義內容推得;講義原文不附答案)", " (derived from the lecture; the source gives no answer)"), size=20, col=GRN)
        fin(d, g, i, ex)

    E1 = (T("想想那筆離群值 10:平方後變成 100,占了總和 103 的 97%。所以 MSE 對離群值敏感(B)。MSE 處處可導,也能用於回歸。", "Think of the outlier 10: squared it becomes 100, 97% of the sum 103. So MSE is sensitive to outliers (B). MSE is differentiable everywhere and is used for regression."))
    quiz(26, T("自我檢核一:MSE 的主要缺點是什麼?", "Self-check 1: what is the main drawback of MSE?"),
         [T("零點不可導", "not differentiable at zero"), T("平方把離群值放大,對離群值敏感", "squaring amplifies outliers; sensitive to them"), T("無法用於回歸", "cannot be used for regression"), T("輸出不落在 $[0,1]$", "output is not in $[0,1]$")], T("看第 1 節的表。", "the table of section 1."), 1, E1)
    quiz(27, T("自我檢核二:Huber 的超參數 $\\delta$ 作用是什麼?", "Self-check 2: what does Huber's hyperparameter $\\delta$ do?"),
         [T("決定學習率", "sets the learning rate"), T("決定二次與線性區的切換門檻", "sets the threshold between quadratic and linear regions"), T("決定類別數", "sets the number of classes"), T("決定 margin", "sets the margin")], T("看公式。", "see the formula."), 1,
         T("公式在 $|e|\\le\\delta$ 用二次、超過就用線性,所以 $\\delta$ 是兩段切換的門檻(B)。學習率、類別數、margin 都不在 Huber 公式裡。", "The formula is quadratic for $|e|\\le\\delta$ and linear beyond, so $\\delta$ is the switching threshold (B). The learning rate, the number of classes and the margin are not in Huber's formula."))
    quiz(28, T("自我檢核三:哪種損失在目標是軟標籤時特別有用?", "Self-check 3: which loss suits soft targets?"),
         ["Hinge", "MSE", "KL divergence", "MAE"], T("看第 2 節。", "section 2."), 2,
         T("軟標籤的目標本身是一份機率分布(如知識蒸餾),KL 散度正是比較兩份分布的差異(C)。", "A soft target is itself a probability distribution (as in distillation), and KL divergence compares two distributions (C)."))
    quiz(29, T("自我檢核四:使用交叉熵時,輸出層必須具備什麼條件?", "Self-check 4: what must the output layer satisfy for cross-entropy?"),
         [T("線性活化即可", "a linear activation is enough"), T("接 softmax 或 sigmoid,使預測落在 $[0,1]$ 且和為 1", "softmax or sigmoid, so predictions lie in $[0,1]$ and sum to 1"), T("用 maxout", "use maxout"), T("必須有 bias", "must have a bias")], T("對數裡的值要像機率。", "the value inside the log must look like a probability."), 1,
         T("交叉熵取的是機率的對數,所以輸出要像機率:多類接 softmax、二元接 sigmoid(B)。二元時一個 sigmoid 輸出和它的補數即可,不必讓單一輸出和為 1。", "Cross-entropy takes the log of a probability, so outputs must look like probabilities: softmax for multi-class, sigmoid for binary (B). For two classes one sigmoid output and its complement suffice; a single output need not sum to 1."))
