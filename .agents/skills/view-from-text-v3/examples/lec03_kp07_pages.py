import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lec03_kit import *

KP = "KP07"; N = 7
A = (2, 4, 6, 8); AB = sum(A) / 4
G0 = [0 - a for a in A]; assert G0 == [-2, -4, -6, -8] and sum(G0) / 4 == 0 - AB == -5
GS = [5 - a for a in A]; assert GS == [3, 1, -1, -3] and sum(GS) == 0
VAR = sum(g * g for g in GS) / 4; SIG = math.sqrt(VAR); assert VAR == 5 and round(SIG, 3) == 2.236
W1 = 5 - 0.2 * 3; assert abs(W1 - 4.4) < 1e-12
STD = lambda eta: SIG * math.sqrt(eta / (2 - eta)); assert round(STD(0.2), 3) == 0.745

def make(lang):
    d = D(lang, KP, N); T = d.T
    g = d.page("KP07 完整梯度每一步都要掃過全部資料", "KP07 The full gradient scans all the data at every step")
    g.beat = 1
    g.rect(120, 118, 1040, 100, fill=CARD, stroke=GRAY, sw=3, radius=16, z=3); g.text(130, 118, 1020, 100, r"$F(w)=\frac1n\sum_{i=1}^{n}f_i(w),\qquad \nabla F(w)=\frac1n\sum_{i=1}^{n}\nabla f_i(w)$", size=32, color=INK, align="center", vcenter=True, z=8)
    g.beat = 2
    chain(g, 50, 260, 1180, 170, [(BLUE, BLUEL, T("每更新一步", "every update"), T("全部 $n$ 筆各做一次\n前向與反向", "all $n$ samples: one\nforward and backward each")), (ORG, ORGL, T("$n=10^6$", "$n=10^6$"), T("一步 = 掃過一百萬筆", "one step = a million samples")), (RED, ORGL if False else REDL, T("$n$ 是數十億", "$n$ in the billions"), T("一步要花好幾個小時\n訓練走不動", "one step takes hours\ntraining cannot move"))], ts=22, bs=22, gap=50, beat0=2)
    g.beat = 5
    gold(g, 190, 470, 900, 130, T("與其花很大的力氣算一個精確的梯度,不如用便宜的替身走很多步", "rather than one costly exact gradient, take many steps with a cheap stand-in"), size=26)
    banner(g, T("機器學習的目標:讓 $n$ 筆樣本損失的平均最小", "the goal: minimize the average loss over $n$ samples"))
    d.done(g, "完整梯度的代價", "The cost of the full gradient", {0: ("目標是讓 n 筆樣本損失的平均最小。微分是線性的，所以完整梯度是每一筆梯度的平均。", "The goal is to minimize the average loss over n samples. Differentiation is linear, so the full gradient is the average of per-sample gradients."), 1: ("代價是每更新一步，都要把全部 n 筆資料各做一次前向與反向傳播。", "The price is that every update needs a forward and backward pass over all n samples."), 2: ("n 是一百萬時，一步就等於掃過一百萬筆。", "With n equal a million, one step scans a million samples."), 3: ("n 是數十億時，一步要花好幾個小時，訓練根本走不動。", "With n in the billions one step takes hours and training cannot move."), 4: ("", ""), 5: ("與其算一個精確的梯度，不如用便宜的替身走很多步。", "Rather than one exact gradient, take many steps with a cheap stand-in.")})

    g = d.page("SGD:每步只抽一筆", "SGD: draw one sample per step")
    g.beat = 1
    g.rect(120, 118, 1040, 100, fill=CARD, stroke=GRAY, sw=3, radius=16, z=3); g.text(130, 118, 1020, 100, r"$w^{(k+1)}=w^{(k)}-\eta_k\,\nabla f_{i_k}\big(w^{(k)}\big)$", size=38, color=INK, align="center", vcenter=True, z=8)
    g.beat = 2
    box(g, 50, 250, 570, 150, GRN, GRNL, T("成本", "cost"), T("每步只做一筆的前向與反向:$O(1)$,和 $n$ 無關", "one forward and backward per step: $O(1)$, independent of $n$"), ts=22, bs=22)
    g.beat = 3
    box(g, 660, 250, 570, 150, BLUE, BLUEL, T("無偏", "unbiased"), r"$\mathbb{E}_i[\nabla f_i(w)]=\nabla F(w)$", ts=22, bs=28)
    g.beat = 4
    gold(g, 190, 440, 900, 130, T("均勻抽樣時,單筆梯度的期望方向就是真正的完整梯度\n算一次完整梯度的時間,單筆更新可以走上好幾千甚至上百萬步", "with uniform sampling the expected direction is the true full gradient\nthe time of one full gradient lets single-sample updates take thousands to millions of steps"), size=22)
    banner(g, T("用一筆梯度當完整梯度的替身", "one sample's gradient stands in for the full gradient"))
    d.done(g, "SGD 的想法", "The idea of SGD", {0: ("想法很樸素：每步只隨機抽一筆，用它的梯度當完整梯度的替身。", "The idea is plain: each step draws one sample at random and uses its gradient as a stand-in for the full gradient."), 1: ("每步只對一筆資料做前向與反向，成本是 O(1)，和 n 無關。", "Each step runs a forward and backward pass on one sample only, cost O(1) independent of n."), 2: ("這個替身可以接受，因為它是無偏的：單筆梯度的期望等於完整梯度。", "The stand-in is acceptable because it is unbiased: the expected single-sample gradient equals the full gradient."), 3: ("", ""), 4: ("也就是說，期望方向就是真正的完整梯度。算一次完整梯度的時間，單筆更新可以走上好幾千步。", "That is, the expected direction is the true full gradient. In the time of one full gradient, single-sample updates can take thousands of steps.")})

    g = d.page("驗算:四筆樣本的單筆梯度", "A check: the single-sample gradients of four samples")
    g.beat = 1
    g.text(50, 108, 1180, 36, T("$f_i(w)=\\frac12(w-a_i)^2$,$a=(2,4,6,8)$,所以 $\\nabla f_i=w-a_i$,$\\nabla F(w)=w-5$", "$f_i(w)=\\frac12(w-a_i)^2$, $a=(2,4,6,8)$, so $\\nabla f_i=w-a_i$ and $\\nabla F(w)=w-5$"), size=21, color=MUTED, bold=True, align="center", z=8)
    base = 380; sc = 34
    g.line([(80, base), (700, base)], INK, sw=3, z=3)
    for i, gv in enumerate(G0):
        x = 110 + i * 150
        g.rect(x, base, 100, -gv * sc * -1 if False else abs(gv) * sc, fill=REDL, stroke=RED, sw=3, radius=6, z=3)
        g.text(x - 20, base + abs(gv) * sc + 6, 140, 36, f"${gv}$", size=22, color=RED, bold=True, align="center", z=8)
        g.text(x - 20, base - 40, 140, 34, f"$a_{i + 1}={A[i]}$", size=18, color=MUTED, bold=True, align="center", z=8)
    g.text(80, 590, 620, 36, T("在 $w=0$:四個單筆梯度", "at $w=0$: the four single-sample gradients"), size=18, color=MUTED, bold=True, align="center", z=8)
    g.beat = 2
    box(g, 740, 150, 490, 180, BLUE, BLUEL, T("平均", "their average"), r"$\frac{-2-4-6-8}{4}=-5$", ts=22, bs=34)
    g.beat = 3
    box(g, 740, 360, 490, 170, GRN, GRNL, T("完整梯度", "the full gradient"), r"$\nabla F(0)=0-5=-5$", ts=22, bs=32)
    g.beat = 4
    gold(g, 740, 550, 490, 76, T("平均 = 完整梯度:無偏", "average = full gradient: unbiased"), size=24)
    banner(g, T("單筆梯度各不相同,但平均剛好是完整梯度", "the single-sample gradients differ but average to the full gradient"))
    d.done(g, "驗算無偏", "Checking unbiasedness", {0: ("四筆樣本，每筆的損失是二分之一乘 w 減 a 的平方，a 是 2、4、6、8。在 w 等於 0，四個單筆梯度是負 2、負 4、負 6、負 8。", "Four samples, each loss one half of w minus a squared, with a equal 2, 4, 6, 8. At w equal 0 the four gradients are minus 2, minus 4, minus 6, minus 8."), 1: ("它們的平均是負 5。", "Their average is minus 5."), 2: ("完整梯度是 w 減 5，在 0 是負 5。", "The full gradient is w minus 5, which at 0 is minus 5."), 3: ("平均等於完整梯度，這就是無偏。", "The average equals the full gradient: that is unbiasedness."), 4: ("", "")})

    g = d.page("無偏不代表每一步都往下", "Unbiased does not mean every step goes down")
    g.beat = 1
    g.text(50, 108, 1180, 36, T("在最小點 $w^*=5$:$\\nabla F(5)=0$,但單筆梯度 $5-a_i$ 都不是 $0$", "at the minimum $w^*=5$: $\\nabla F(5)=0$ but every $5-a_i$ is nonzero"), size=22, color=MUTED, bold=True, align="center", z=8)
    for i, gv in enumerate(GS):
        x = 80 + i * 280
        box(g, x, 170, 240, 130, ORG if gv > 0 else BLUE, ORGL if gv > 0 else BLUEL, T(f"抽到第 {i + 1} 筆", f"draw sample {i + 1}"), f"${gv:+d}$".replace("+", "+"), ts=20, bs=40)
    g.beat = 2
    box(g, 50, 330, 570, 150, PUR, PURL, T("變異數", "variance"), r"$\sigma^2=\frac{9+1+1+9}{4}=5,\ \ \sigma=2.236$", ts=22, bs=26)
    g.beat = 3
    box(g, 660, 330, 570, 170, RED, REDL, T("明明已在最小點", "already at the minimum"), r"$w=5-0.2\cdot3=4.4$" + T("\n抽到第 1 筆、$\\eta=0.2$,一步就被推離", "\ndraw sample 1 with $\\eta=0.2$: pushed away in one step"), ts=22, bs=24)
    g.beat = 4
    gold(g, 150, 506, 980, 112, T("$\\nabla f_i=\\nabla F+\\xi_i$,$\\mathbb{E}[\\xi_i]=0$:平均往對的方向,但每步帶一份隨機偏差", "$\\nabla f_i=\\nabla F+\\xi_i$, $\\mathbb{E}[\\xi_i]=0$: right on average, with a random error every step"), size=22)
    banner(g, T("期望方向對,單步可以走錯", "the expected direction is right; a single step can be wrong"))
    d.done(g, "雜訊", "Noise", {0: ("看最小點，w 等於 5。完整梯度是零，但四個單筆梯度是 3、1、負 1、負 3，都不是零。", "At the minimum, w equal 5, the full gradient is zero but the four single-sample gradients are 3, 1, minus 1, minus 3, none zero."), 1: ("單筆梯度的變異數是 5，標準差 2.236。", "The single-sample variance is 5, the standard deviation 2.236."), 2: ("若恰好抽到第一筆，用學習率 0.2 更新，w 變成 4.4。明明已在最小點，一步就被推離。", "If sample one is drawn with learning rate 0.2, w becomes 4.4: already at the minimum yet pushed away in one step."), 3: ("單筆梯度等於真梯度加零均值的雜訊：平均而言往對的方向，但每步都有一份隨機偏差。", "A single-sample gradient is the true gradient plus zero-mean noise: right on average, but each step carries a random error.")})

    g = d.page("固定步長:在最小點附近擺動", "A fixed step: swinging around the minimum")
    g.beat = 1
    g.rect(190, 108, 900, 140, fill=CARD, stroke=GRAY, sw=3, radius=16, z=3); g.text(200, 108, 880, 140, r"$\mathrm{std}(e)=\sigma\sqrt{\dfrac{\eta}{2-\eta}}$".replace(r"\dfrac", r"\frac"), size=44, color=INK, align="center", vcenter=True, z=8)
    rows = [[f"{e:g}", f"{STD(e):.3f}"] for e in (0.05, 0.2, 0.5)]
    g.beat = 2
    table(g, 190, 260, [330, 570], [r"$\eta$", T("擺動的標準差(此例)", "std of the swing (this example)")], rows, rh=70, fs=30, styles=[(GRNL, GRN), (GOLDL, GOLD), (REDL, RED)], gap=8)
    g.beat = 3
    gold(g, 190, 560, 900, 60, T("學習率越小擺動越小,但收斂也越慢(KP02 的「太小」)", "a smaller rate swings less but converges slower (KP02's “too small”)"), size=22)
    banner(g, T("$\\eta=0.2$:$2.236\\cdot\\sqrt{0.2/1.8}=0.745$", "$\\eta=0.2$: $2.236\\cdot\\sqrt{0.2/1.8}=0.745$"))
    d.done(g, "擺動的大小", "The size of the swing", {0: ("令誤差是 w 減 5，更新是 1 減 eta 乘誤差，加 eta 乘零均值的雜訊。穩態標準差是 sigma 乘根號 eta 除以 2 減 eta。", "Let the error be w minus 5; the update is 1 minus eta times the error plus eta times zero-mean noise. The steady-state standard deviation is sigma times the square root of eta over 2 minus eta."), 1: ("學習率 0.05 擺動約 0.34，0.2 約 0.745，0.5 約 1.29。", "Learning rate 0.05 swings about 0.34, 0.2 about 0.745, 0.5 about 1.29."), 2: ("學習率越小擺動越小，但收斂也越慢。", "A smaller rate swings less but converges slower."), 3: ("", "")})

    g = d.page("遞減步長與取捨:便宜但吵", "A decaying step and the trade-off: cheap but noisy")
    g.beat = 1
    box(g, 50, 126, 1180, 130, BLUE, BLUEL, T("讓 SGD 真的停在最小點:步長遞減 $\\eta_k\\to0$", "to make SGD truly settle: decay the step, $\\eta_k\\to0$"), T("例:$\\eta_k=1/(k+1)$ 使 $w^{(k+1)}$ 正好是到目前為止抽到的所有 $a$ 的樣本平均", "e.g. $\\eta_k=1/(k+1)$ makes $w^{(k+1)}$ exactly the sample average of all $a$ drawn so far"), ts=22, bs=22)
    rows = [[T("每步成本", "cost per step"), r"$O(n)$", r"$O(1)$"], [T("梯度估計", "gradient estimate"), T("精確", "exact"), T("無偏、有雜訊", "unbiased, noisy")], [T("每步一定下降嗎", "does every step descend"), T("步長適當時是", "yes, with a suitable step"), T("不一定", "not necessarily")], [T("最小點附近", "near the minimum"), T("梯度為零、停得住", "zero gradient, can stop"), T("持續擺動,除非步長遞減", "keeps swinging unless the step decays")]]
    g.beat = 2
    table(g, 60, 280, [330, 380, 440], ["", T("完整梯度下降", "full gradient descent"), T("SGD(單筆)", "SGD (single sample)")], rows, rh=58, fs=20, hfs=20, beats=[2, 2, 3, 3], gap=6)
    g.beat = 4
    gold(g, 190, 590, 900, 40, T("雜訊是否幫助逃離鞍點、是否泛化更好:不保證,留到 KP12", "whether noise helps escape saddles or generalize better: no guarantee, left to KP12"), size=18)
    banner(g, T("單筆梯度的雜訊還是太大,實務上折衷成小批次(KP08)", "single-sample noise is still too large, so we compromise with minibatches (KP08)"))
    d.done(g, "取捨", "The trade-offs", {0: ("怎麼讓 SGD 真的停在最小點？讓步長遞減，雜訊的影響就跟著縮小。", "How to make SGD truly settle? Decay the step so the influence of noise shrinks."), 1: ("取學習率 1 除以 k 加 1，每次更新的結果正好是到目前為止抽到的所有 a 的樣本平均。", "With learning rate 1 over k plus 1, each update gives exactly the sample average of all a drawn so far."), 2: ("取捨可以整理成一句話：便宜但吵。每步成本從 n 降到 1；梯度無偏但有雜訊。", "The trade-off in one phrase: cheap but noisy. Cost per step falls from n to 1; the gradient is unbiased but noisy."), 3: ("每步不一定下降；在最小點附近持續擺動，除非步長遞減。", "Not every step descends; it keeps swinging near the minimum unless the step decays."), 4: ("來源說雜訊是逃離鞍點的跳板、泛化更好，這在某些條件下可能成立，但不是保證，留到 KP12 討論。", "The source says noise helps escape saddles and generalizes better; that may hold under some conditions but is not guaranteed, left to KP12.")})
    closing(d, [(RED, REDL, "「無偏就是每步都下降」", "“unbiased means every step descends”", "期望方向對,但單步可以走錯", "the expected direction is right but one step can go wrong"), (ORG, ORGL, "「SGD 一定比完整梯度下降快」", "“SGD is always faster than full gradient descent”", "每步便宜但需要更多步", "cheap steps, but more of them"), (PUR, PURL, "「固定步長的 SGD 會收斂到最小點」", "“fixed-step SGD converges to the minimum”", "只會在附近擺動", "it only swings around it")],
            ("單筆梯度的雜訊還是太大。能不能折衷,每步用一小批資料?", "Single-sample noise is still too large. Can we compromise with a small batch per step?"), ("下一個單元:小批次與 epoch", "next unit: minibatches and epochs"))
    return d.pages
