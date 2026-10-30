import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lec03_kit import *

KP = "KP15"; N = 6
XV = (1, 2); YV = 3; WT = (1, 1)
assert XV[0] * WT[0] + XV[1] * WT[1] == YV
def limit(w0):
    r = (YV - (XV[0] * w0[0] + XV[1] * w0[1])) / (XV[0] ** 2 + XV[1] ** 2); return (w0[0] + XV[0] * r, w0[1] + XV[1] * r)
WA, WB = limit((0, 0)), limit((2, 0))
test = lambda w: (w[0] - WT[0]) ** 2 + (w[1] - WT[1]) ** 2
assert [round(v, 3) for v in WA] == [0.6, 1.2] and [round(v, 3) for v in WB] == [2.2, 0.4] and round(test(WA), 3) == 0.2 and round(test(WB), 3) == 1.8 and round(test(WB) / test(WA)) == 9
assert abs(XV[0] * WB[0] + XV[1] * WB[1] - 3) < 1e-12 and abs(XV[0] * WA[0] + XV[1] * WA[1] - 3) < 1e-12

def make(lang):
    d = D(lang, KP, N); T = d.T
    g = d.page("KP15 過參數化:解不唯一", "KP15 Over-parameterization: the solution is not unique")
    g.beat = 1
    chain(g, 50, 120, 1180, 170, [(BLUE, BLUEL, T("過參數化", "over-parameterized"), r"$p>n$" + T("\n參數比資料多", "\nmore parameters than data")), (ORG, ORGL, T("最小平方", "least squares"), r"$L(w)=\tfrac12\|Xw-y\|^2$"), (GRN, GRNL, T("方程式 $Xw=y$", "the equation $Xw=y$"), T("$p$ 個未知數、$n$ 條限制", "$p$ unknowns, $n$ constraints"))], ts=22, bs=24, gap=50, beat0=1)
    g.beat = 4
    gold(g, 190, 330, 900, 110, T("零損失的解有 $p-n$ 個自由度:答案不是一個點,而是一大片", "the zero-loss solutions have $p-n$ degrees of freedom: not a point but a whole family"), size=26)
    g.beat = 5
    gold(g, 150, 470, 980, 110, T("梯度法只要走得進這一片,訓練損失就是 $0$;但這一片裡每個點的測試表現不一樣", "gradient descent reaches zero training loss once it enters this family; but the points in it differ on test data"), size=24, light=REDL, stroke=RED)
    banner(g, T("定理說的是「在什麼條件下走得進去、要走幾步」", "the theorems say under which conditions it gets in, and in how many steps"))
    d.done(g, "過參數化", "Over-parameterization", {0: ("過參數化是參數比資料多。最簡單的例子是線性模型的最小平方。", "Over-parameterization means more parameters than data. The simplest example is least squares for a linear model."), 1: ("損失是二分之一乘預測與標籤差的平方範數。", "The loss is one half the squared norm of prediction minus label."), 2: ("方程式有 p 個未知數，卻只有 n 條限制，", "The equation has p unknowns but only n constraints,"), 3: ("", ""), 4: ("所以零損失的解有 p 減 n 個自由度：答案不是一個點，而是一大片。", "so the zero-loss solutions have p minus n degrees of freedom: not a point but a whole family."), 5: ("梯度法只要走得進這一片，訓練損失就是零，但這一片裡每個點的測試表現不一樣。", "Gradient descent gets zero training loss once it enters the family, yet the points in it differ on test data.")})

    g = d.page("GD 落在哪一個,由初始點決定", "Where GD lands is decided by the start")
    xr, yr = (-0.4, 3.0), (-0.4, 2.4)
    x0, y0, w_, h_ = 50, 120, 700, 470
    ux = lambda v: x0 + (v - xr[0]) / (xr[1] - xr[0]) * w_; uy = lambda v: y0 + h_ - (v - yr[0]) / (yr[1] - yr[0]) * h_
    g.beat = 1
    g.rect(x0, y0, w_, h_, fill=CARD, stroke=GRAY, sw=3, radius=12, z=1)
    g.line([(ux(-0.3), uy((3 + 0.3) / 2)), (ux(2.9), uy((3 - 2.9) / 2))], BLUE, sw=5, z=4)
    g.text(ux(1.8), uy(0.7) - 50, 300, 60, T("零損失的解:$w_1+2w_2=3$", "zero-loss solutions: $w_1+2w_2=3$"), size=18, color=BLUE, bold=True, z=8)
    dot(g, ux(1), uy(1), 12, GOLD); lab(g, ux(1) + 14, uy(1) - 40, 200, T("真實權重 $(1,1)$", "true weights $(1,1)$"), 18, INK)
    g.beat = 2
    for w0, wl, col, nm in (((0, 0), WA, GRN, "A"), ((2, 0), WB, RED, "B")):
        g.line([(ux(w0[0]), uy(w0[1])), (ux(wl[0]), uy(wl[1]))], col, sw=5, arrow="end", head=16, z=6); dot(g, ux(w0[0]), uy(w0[1]), 9, GRAY); dot(g, ux(wl[0]), uy(wl[1]), 12, col)
        lab(g, ux(wl[0]) + 14, uy(wl[1]) - 30, 180, f"{nm}: $({wl[0]:.1f},\\,{wl[1]:.1f})$", 18, col)
    g.text(50, 594, 700, 34, T("一筆樣本 $x=(1,2)$,$y=3$;灰點:初始點,圓點:GD 收斂處", "one sample $x=(1,2)$, $y=3$; grey: starts, dots: where GD converges"), size=17, color=MUTED, bold=True, align="center", z=8)
    g.beat = 1
    box(g, 790, 126, 440, 130, BLUE, BLUEL, T("GD 的極限", "the GD limit"), r"$w_{\infty}=w_0+X^{+}(y-Xw_0)$", ts=20, bs=24)
    g.beat = 2
    box(g, 790, 276, 440, 130, GRN, GRNL, T("A:$w_0=(0,0)$", "A: $w_0=(0,0)$"), r"$w_\infty=(0.6,\,1.2)$" + T("\n測試損失 $0.2$", "\ntest loss $0.2$"), ts=20, bs=22)
    box(g, 790, 426, 440, 130, RED, REDL, T("B:$w_0=(2,0)$", "B: $w_0=(2,0)$"), r"$w_\infty=(2.2,\,0.4)$" + T("\n測試損失 $1.8$", "\ntest loss $1.8$"), ts=20, bs=22)
    g.beat = 3
    gold(g, 790, 570, 440, 62, T("訓練損失同為 $0$,測試差 $9$ 倍", "equal training loss $0$, test loss differs $9\\times$"), size=20)
    banner(g, T("從 $w_0=0$ 出發得到最小範數解 $X^{+}y$", "starting from $w_0=0$ gives the minimum-norm solution $X^{+}y$"))
    d.done(g, "初始點決定", "The start decides", {0: ("最小平方中，GD 收斂到初始點加上把殘差補起來所需的最小修正。一筆樣本，零損失的解是一條線。", "In least squares GD converges to the starting point plus the smallest correction that fills the residual. With one sample the zero-loss solutions form a line."), 1: ("從原點出發，收斂到 0.6、1.2，測試損失 0.2。從 2、0 出發，收斂到 2.2、0.4，測試損失 1.8。", "From the origin it converges to 0.6, 1.2 with test loss 0.2. From 2, 0 it converges to 2.2, 0.4 with test loss 1.8."), 2: ("", ""), 3: ("兩個解訓練損失同為零，測試損失差 9 倍。從零出發得到最小範數解。", "Both solutions have training loss zero while the test loss differs 9 times. Starting from zero gives the minimum-norm solution.")})

    g = d.page("非凸時:淺層二次啟用網路", "Non-convex: a shallow quadratic network")
    g.beat = 1
    g.rect(190, 118, 900, 100, fill=CARD, stroke=GRAY, sw=3, radius=16, z=3); g.text(200, 118, 880, 100, r"$f(x)=\sum_{j=1}^{k}v_j\,(w_j^{\top}x)^2=x^{\top}Mx$", size=38, color=INK, align="center", vcenter=True, z=8)
    g.beat = 2
    box(g, 50, 240, 570, 140, BLUE, BLUEL, T("來源所述的條件", "conditions stated by the source"), T("寬度 $k\\ge2d$;輸出權重 $v$ 固定\n至少 $d$ 個正、$d$ 個負", "width $k\\ge2d$; output weights $v$ fixed\nat least $d$ positive and $d$ negative"), ts=20, bs=21)
    g.beat = 3
    box(g, 660, 240, 570, 190, GRN, GRNL, T("三點結論", "three conclusions"), T("所有局部極小都是全域極小\n每個鞍點都有負曲率方向\n$d\\le n\\le c\\,d^2$ 時全域最小損失為 $0$", "every local minimum is global\nevery saddle has a negative-curvature direction\nglobal minimum loss $0$ when $d\\le n\\le c\\,d^2$"), ts=20, bs=20)
    g.beat = 4
    rows = [["1", "0%"], ["2", "83%"], ["4", "99%"], ["8", "96%"]]
    g.text(50, 400, 570, 34, T("講義的 100 次實驗($d=2$,$n=3$):到零損失的比例", "the note's 100 runs ($d=2$, $n=3$): fraction reaching zero loss"), size=16, color=MUTED, bold=True, align="center", z=8)
    table(g, 80, 440, [250, 280], [T("寬度 $k$", "width $k$"), T("到零損失", "reaches zero")], rows, rh=30, fs=19, hfs=15, styles=[(REDL, RED), (GOLDL, GOLD), (GRNL, GRN), (GRNL, GRN)], gap=2)
    g.beat = 5
    gold(g, 660, 450, 570, 160, T("$k=1$ 根本不存在零損失解:條件不成立時,定理什麼都沒說", "for $k=1$ no zero-loss solution exists at all: when the conditions fail the theorem says nothing"), size=20, light=REDL, stroke=RED)
    banner(g, T("最小的非凸網路:沒有「假的」局部極小(在條件下)", "the smallest non-convex network: no “fake” local minima (under the conditions)"))
    d.done(g, "淺層二次網路", "Shallow quadratic network", {0: ("最小平方是凸的，保證很容易。神經網路是非凸的，擔憂是 GD 會卡在梯度為零但損失很高的點。最小的非凸網路：單一隱藏層、二次啟用。", "Least squares is convex, so guarantees are easy. Neural networks are non-convex and the worry is that GD sticks where the gradient is zero but the loss is high. The smallest non-convex network has one hidden layer and quadratic activation."), 1: ("來源所述的條件：寬度至少兩倍輸入維度，輸出權重固定且至少有輸入維度個正數與負數。", "The source's conditions: width at least twice the input dimension, with fixed output weights having at least that many positive and negative ones."), 2: ("結論三點：所有局部極小都是全域極小；每個鞍點都有負曲率方向；樣本數在範圍內時，全域最小損失是零。", "Three conclusions: every local minimum is global; every saddle has a negative-curvature direction; the global minimum loss is zero when the sample count is in range."), 3: ("講義做了小實驗：寬度 1 是 0%，2 是 83%，4 是 99%，8 是 96%。", "The note ran a small experiment: width 1 gives 0 percent, 2 gives 83, 4 gives 99, 8 gives 96."), 4: ("", ""), 5: ("寬度 1 時零損失解根本不存在。條件不成立時，定理什麼都沒說。", "At width 1 no zero-loss solution exists. When the conditions fail the theorem says nothing.")})

    g = d.page("深層 ReLU 網路:多項式步數內收斂", "Deep ReLU: polynomially many steps")
    g.beat = 1
    chain(g, 50, 120, 1180, 150, [(BLUE, BLUEL, T("資料", "data"), T("輸入範數 1\n不同樣本距離至少 $\\delta$", "input norm 1\ndistinct samples at least $\\delta$ apart")), (ORG, ORGL, T("初始化", "initialization"), T("高斯隨機", "Gaussian random")), (GRN, GRNL, T("寬度與步長", "width and step"), T("寬度夠大、步長夠小", "wide enough, small enough step"))], ts=21, bs=19, gap=50, beat0=1)
    g.beat = 4
    g.rect(190, 300, 900, 100, fill="#fef3c7", stroke=GOLD, sw=3, radius=16, z=3); g.text(200, 300, 880, 100, r"$T=\mathrm{poly}(n,l)\cdot\dfrac{1}{\delta^{2}}\cdot\log\dfrac{1}{\varepsilon}$".replace(r"\dfrac", r"\frac"), size=34, color=INK, align="center", vcenter=True, z=8)
    g.beat = 5
    box(g, 50, 430, 370, 190, GRN, GRNL, T("多項式,不是指數", "polynomial, not exponential"), T("非凸不代表要花指數時間", "non-convex does not mean exponential time"), ts=18, bs=19)
    box(g, 455, 430, 370, 190, BLUE, BLUEL, T("$\\varepsilon$ 在 $\\log$ 裡", "$\\varepsilon$ sits in a $\\log$"), T("每把目標縮小十倍,只多固定步數(像線性收斂)", "each tenfold smaller target costs a fixed number of extra steps (linear-like)"), ts=18, bs=18)
    box(g, 860, 430, 370, 190, RED, REDL, T("$\\delta$ 小,步數大", "small $\\delta$, many steps"), T("步數約與 $1/\\delta^2$ 成正比:資料靠得近就慢", "steps grow like $1/\\delta^2$: close data slow it down"), ts=18, bs=18)
    banner(g, T("「很快」是誤讀:常數裡有 $\\mathrm{poly}(n,l)$ 與 $1/\\delta^2$", "“fast” is a misreading: the constant hides $\\mathrm{poly}(n,l)$ and $1/\\delta^2$"))
    d.done(g, "深層 ReLU", "Deep ReLU", {0: ("第二個結果處理 l 層、每層寬度 m 的全連接 ReLU 網路，損失是均方誤差。條件有三類。第一，資料：每個輸入範數為 1，任兩個不同樣本的距離至少是 delta。", "The second result covers fully connected ReLU networks of l layers of width m with mean squared loss. Three kinds of conditions. First, data: each input has norm 1 and any two distinct samples are at least delta apart."), 1: ("第二，初始化是高斯隨機。", "Second, initialization is Gaussian random."), 2: ("第三，寬度夠大、步長夠小。", "Third, the width is large enough and the step small enough."), 3: ("", ""), 4: ("結論是所需步數大約是 n 與 l 的多項式，乘上一除以 delta 平方，乘上一除以 epsilon 的對數。", "The conclusion: the steps needed are about a polynomial in n and l, times one over delta squared, times the log of one over epsilon."), 5: ("三個讀法：步數是多項式不是指數；epsilon 只出現在對數裡，每縮小十倍只多固定步數；delta 小時步數大，資料靠得近就慢。", "Three readings: the steps are polynomial, not exponential; epsilon sits in a log so each tenfold shrink costs a fixed extra number; small delta means many steps, close data slow it down.")})

    g = d.page("條件式的結論，與它沒說的事", "A conditional conclusion, and what it does not say")
    g.beat = 1
    chain(g, 50, 120, 1180, 170, [(BLUE, BLUEL, T("若", "if"), T("寬度夠、初始化隨機、資料可分開、步長夠小", "wide enough, random init, separable data, small step")), (GRN, GRNL, T("則", "then"), T("訓練損失可降到任意小", "the training loss can be made arbitrarily small"))], ts=26, bs=22, gap=120, beat0=1)
    g.beat = 3
    box(g, 50, 320, 570, 150, RED, REDL, T("條件不成立時", "when the conditions fail"), T("寬度太窄、資料重複卻標籤不同、步長過大:它什麼都沒說", "too narrow, duplicate data with different labels, a step too large: it says nothing"), ts=21, bs=21)
    g.beat = 4
    box(g, 660, 320, 570, 150, PUR, PURL, T("保證的是訓練損失", "what is guaranteed is training loss"), T("前面看到:訓練損失同為 $0$ 的解,測試損失差 $9$ 倍", "as seen: solutions with training loss $0$ can differ $9\\times$ on test"), ts=21, bs=21)
    g.beat = 5
    gold(g, 150, 510, 980, 100, T("「最佳化會成功」不能推出「模型會預測得好」:這個缺口留給 KP16", "“optimization will succeed” does not imply “the model will predict well”: that gap is KP16's"), size=24)
    banner(g, T("最佳化與泛化是兩個問題", "optimization and generalization are two separate problems"))
    d.done(g, "條件式結論", "A conditional conclusion", {0: ("把兩個結果並排看：它們都是若寬度夠、初始化隨機、資料可以分開、步長夠小，", "Put the two results side by side: both say that if the width is enough, the initialization random, the data separable and the step small,"), 1: ("則訓練損失可以降到任意小。", "then the training loss can be made arbitrarily small."), 2: ("", ""), 3: ("這是有條件的結論，條件不成立時它什麼都沒有說。", "This is a conditional conclusion: when the conditions fail it says nothing."), 4: ("更重要的是，這兩個結果保證的是訓練損失。訓練損失同樣是零的解，測試損失可以差很多。", "More importantly, both results guarantee the training loss. Solutions with equal zero training loss can differ a lot on test loss."), 5: ("所以最佳化會成功，不能推出模型會預測得好，這個缺口就是 KP16 要處理的。", "So optimization succeeding does not imply the model predicts well; that gap is what KP16 handles.")})
    closing(d, [(RED, REDL, "「過參數化的網路一定會過擬合」", "“an over-parameterized network must overfit”", "最佳化與泛化是兩個問題,泛化要另外檢驗", "optimization and generalization are separate; test generalization separately"), (ORG, ORGL, "「收斂保證就是泛化保證」", "“a convergence guarantee is a generalization guarantee”", "保證的是訓練損失;測試表現不在定理裡", "it guarantees training loss; test performance is not in the theorem"), (PUR, PURL, "「步數是 $\\log(1/\\varepsilon)$,所以一定很快」", "“the steps are $\\log(1/\\varepsilon)$ so it is surely fast”", "常數有 $\\mathrm{poly}(n,l)$ 與 $1/\\delta^2$,資料靠近時很大", "the constant has $\\mathrm{poly}(n,l)$ and $1/\\delta^2$, large for close data")],
            ("訓練損失同樣是零的解,測試表現可以差很多。到底什麼決定泛化?", "Solutions with the same zero training loss can differ a lot on test data. What decides generalization?"), ("下一個單元:全章總結", "next unit: chapter synthesis"))
    return d.pages
