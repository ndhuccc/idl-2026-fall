import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lec03_kit import *

KP = "KP09"; N = 7
S2 = 5.0
RES = lambda t: 12.5 * 0.64 ** t; ERR = lambda t: 5 * 0.8 ** t
assert [round(RES(t), 4) for t in (0, 1, 5, 10, 20)] == [12.5, 8.0, 1.3422, 0.1441, 0.0017] and [round(ERR(t), 4) for t in (0, 1, 5, 10, 20)] == [5.0, 4.0, 1.6384, 0.5369, 0.0576]
T01 = next(t for t in range(100) if RES(t) <= 0.01); assert T01 == 16 and 62.5 / 6250 == 0.01
BAND = lambda e: math.sqrt(S2 * e / (2 - e)); PLAT = lambda e: e * S2 / (2 * (2 - e)); STEPS = lambda e: math.log(10) / -math.log(1 - e)
assert round(BAND(0.2), 3) == 0.745 and round(PLAT(0.2), 3) == 0.278 and round(PLAT(0.1), 3) == 0.132 and round(PLAT(0.05), 3) == 0.064
def sgd(etas, n):
    m = 5.0; v = 0.0; out = {}
    for k in range(n): e = etas(k); m *= (1 - e); v = (1 - e) ** 2 * v + e * e * S2; out[k + 1] = 0.5 * (m * m + v)
    return out
R1 = sgd(lambda k: 1 / (k + 1), 3000); R2 = sgd(lambda k: 1 / math.sqrt(k + 1), 100000); R3 = sgd(lambda k: 0.5 / (k + 1) ** 2, 20000); R0 = sgd(lambda k: 0.2, 300)
assert abs(R1[1000] * 1000 - 2.5) < 0.01 and abs(R2[100000] * math.sqrt(100000) - 1.25) < 0.01 and round(R3[20000], 2) == 1.96 and round(R0[300], 3) == 0.278
assert abs(sum(1 / m for m in range(1, 1000001)) - 14.39) < 0.01 and round(math.pi ** 2 / 6, 3) == 1.645

def make(lang):
    d = D(lang, KP, N); T = d.T
    g = d.page("KP09 沒有雜訊時，「收斂」是什麼意思", "KP09 What “convergence” means without noise")
    g.beat = 1
    chain(g, 50, 120, 1180, 150, [(BLUE, BLUEL, T("殘差", "residual"), r"$F(w_t)-F^{*}$"), (ORG, ORGL, T("收斂率", "convergence rate"), T("殘差隨步數 $t$ 縮小得多快", "how fast the residual shrinks with $t$"))], ts=24, bs=24, gap=100, beat0=1)
    g.beat = 3
    g.text(50, 290, 1180, 40, T("四筆樣本:$w^{*}=5$,$F^{*}=2.5$,殘差 $=\\frac12(w-5)^2$;$\\eta=0.2$,$w_0=0$", "four samples: $w^{*}=5$, $F^{*}=2.5$, residual $=\\frac12(w-5)^2$; $\\eta=0.2$, $w_0=0$"), size=20, color=MUTED, bold=True, align="center", z=8)
    rows = [[T("誤差 $|w_t-5|$", "error $|w_t-5|$")] + [f"{ERR(t):.4g}" for t in (0, 1, 5, 10, 20)], [T("殘差 $12.5\\cdot0.64^t$", "residual $12.5\\cdot0.64^t$")] + [f"{RES(t):.4g}" for t in (0, 1, 5, 10, 20)], [T("上界 $62.5/t$", "bound $62.5/t$")] + ["—"] + [f"{62.5 / t:.4g}" for t in (1, 5, 10, 20)][:0] + [f"{62.5 / t:.4g}" for t in (1, 5, 10, 20)]]
    rows[2] = [rows[2][0], "—", "62.5", "12.5", "6.25", "3.125"]
    table(g, 60, 330, [330, 150, 150, 150, 150, 150], ["$t$", "0", "1", "5", "10", "20"], rows, rh=52, fs=22, hfs=20, styles=[(CARD, GRAY), (GRNL, GRN), (REDL, RED)], gap=6, beats=[3, 4, 5])
    g.beat = 6
    gold(g, 150, 590, 980, 40, T("實際 16 步殘差就 $\\le0.01$;上界要 $t\\ge6250$ 才保證", "the residual is $\\le0.01$ after 16 steps; the bound needs $t\\ge6250$"), size=21)
    banner(g, T("殘差每步乘 $0.64$:線性收斂,比 $O(1/t)$ 快得多", "the residual shrinks by $0.64$ per step: linear convergence, far faster than $O(1/t)$"))
    d.done(g, "沒有雜訊的尺", "A yardstick without noise", {0: ("先有一把沒有雜訊時的尺，叫殘差：目前的損失比最低損失高出多少。", "First a noiseless yardstick, the residual: how much higher the current loss is than the lowest loss."), 1: ("收斂率說的是殘差隨步數縮小得多快。", "The convergence rate says how fast the residual shrinks with the number of steps."), 2: ("", ""), 3: ("四筆樣本的例子，學習率 0.2：誤差每步乘 0.8。", "In the four-sample example with learning rate 0.2 the error is multiplied by 0.8 each step."), 4: ("殘差每步乘 0.64，這叫線性收斂。", "The residual is multiplied by 0.64 per step: linear convergence."), 5: ("上界是 62.5 除以 t，只是最壞情況的保證，不是實際速度。", "The bound 62.5 over t is a worst-case guarantee, not the actual speed."), 6: ("要殘差降到 0.01 以下，實際只要 16 步，但上界保證要 6250 步才夠。強凸正是這個差別的原因。", "Reaching a residual below 0.01 actually takes 16 steps, while the bound only guarantees it after 6250. Strong convexity explains the gap.")})

    g = d.page("固定步長的 SGD:停在一條帶子裡", "Fixed-step SGD stays in a band")
    g.beat = 1
    g.rect(120, 112, 1040, 100, fill=CARD, stroke=GRAY, sw=3, radius=16, z=3); g.text(130, 112, 1020, 100, r"$e_{k+1}=(1-\eta)\,e_k+\eta\,\xi_k,\quad \mathbb{E}\xi_k=0,\ \mathrm{Var}\,\xi_k=\sigma^2=5$", size=30, color=INK, align="center", vcenter=True, z=8)
    g.beat = 2
    box(g, 50, 236, 570, 130, BLUE, BLUEL, T("縮小 + 注入", "shrink + inject"), T("舊誤差被縮小,每步又注入新雜訊\n兩股力量在 $v_\\infty$ 平衡", "old error shrinks, new noise enters every step\nthe two balance at $v_\\infty$"), ts=21, bs=20)
    box(g, 660, 236, 570, 130, GRN, GRNL, T("長期", "long run"), r"$\mathrm{std}(e)=\sigma\sqrt{\frac{\eta}{2-\eta}},\ \ \mathbb{E}[\text{residual}]=\frac{\eta\sigma^2}{2(2-\eta)}$", ts=21, bs=20)
    rows = [[f"{e:g}", f"{BAND(e):.3f}", f"{PLAT(e):.3f}", f"{STEPS(e):.1f}"] for e in (0.4, 0.2, 0.1, 0.05)]
    g.beat = 3
    table(g, 60, 386, [170, 340, 340, 330], [r"$\eta$", T("帶寬(標準差)", "band (std)"), T("殘差平台", "residual plateau"), T("誤差縮到 1/10 的步數", "steps to shrink error to 1/10")], rows, rh=40, fs=20, hfs=18, styles=[None, (GOLDL, GOLD), None, None], gap=4)
    banner(g, T("殘差平台 $\\propto\\eta$(即 $O(\\eta)$);帶寬 $\\propto\\sqrt\\eta$", "residual plateau $\\propto\\eta$ (that is $O(\\eta)$); band width $\\propto\\sqrt\\eta$"))
    d.done(g, "固定步長的帶子", "The band for a fixed step", {0: ("學習率仍是 0.2，但每步只用一筆樣本。誤差更新是 1 減 eta 乘舊誤差，加 eta 乘零均值的雜訊，方差 5。", "Keep learning rate 0.2 but use one sample per step. The error update is 1 minus eta times the old error plus eta times zero-mean noise with variance 5."), 1: ("第一項把舊誤差縮小，第二項每步又注入新雜訊，兩股力量在某個變異數平衡。", "The first term shrinks the old error and the second injects new noise each step; the two balance at a variance."), 2: ("長期誤差標準差是 sigma 乘根號 eta 除以 2 減 eta，殘差期望是 eta sigma 平方除以 2 乘 2 減 eta。學習率 0.2 時標準差 0.745，殘差平台 0.278。", "The long-run error standard deviation is sigma times the square root of eta over 2 minus eta and the expected residual is eta sigma squared over 2 times 2 minus eta. At 0.2 the standard deviation is 0.745 and the residual plateau 0.278."), 3: ("學習率減半，平台約減半，所以殘差平台約與學習率成正比；帶寬則與學習率的平方根成正比。", "Halving the learning rate roughly halves the plateau: the residual plateau is about proportional to the rate, while the band width is proportional to its square root.")})

    g = d.page("雜訊為什麼不會自己熄", "Why the noise does not die out by itself")
    g.beat = 1
    for i, gv in enumerate((3, 1, -1, -3)):
        box(g, 60 + i * 290, 130, 250, 130, ORG if gv > 0 else BLUE, ORGL if gv > 0 else BLUEL, T("在 $w^{*}$ 抽到的單筆梯度", "single-sample gradient at $w^{*}$") if i == 0 else "", f"${gv:+d}$", ts=17 if i == 0 else 4, bs=44, th=60 if i == 0 else 0) if False else box(g, 60 + i * 290, 130, 250, 130, ORG if gv > 0 else BLUE, ORGL if gv > 0 else BLUEL, None, f"${gv:+d}$", bs=44)
    g.text(60, 270, 1160, 40, T("在 $w^{*}$ 抽到一筆時的單筆梯度:真梯度 $\\nabla F(w^{*})=0$,但這四個都不是 $0$", "single-sample gradients at $w^{*}$: the true gradient $\\nabla F(w^{*})=0$, yet none of these four is $0$"), size=21, color=MUTED, bold=True, align="center", z=8)
    g.beat = 2
    box(g, 50, 330, 570, 160, RED, REDL, T("每步仍被推一下", "pushed a little every step"), T("推的幅度是 $\\eta\\times$ 這個數\n固定 $\\eta$ 讓推力永遠不縮小", "the push is $\\eta\\times$ that number\na fixed $\\eta$ never shrinks it"), ts=22, bs=22)
    g.beat = 3
    box(g, 660, 330, 570, 160, GRN, GRNL, T("結果", "result"), T("固定步長的 SGD 不停在 $w^{*}$\n它收斂到一個分布(一條帶子)", "fixed-step SGD does not settle at $w^{*}$\nit converges to a distribution (a band)"), ts=22, bs=22)
    g.beat = 4
    gold(g, 150, 530, 980, 90, T("多迭代只是在帶內繼續晃:帶寬由 $\\eta$ 決定", "more iterations only keep it wandering inside the band: $\\eta$ sets the width"), size=24)
    banner(g, T("要停下來,推力得越來越小", "to settle, the push must get smaller and smaller"))
    d.done(g, "雜訊不會自己熄", "Noise does not die", {0: ("在最小點，真正的梯度是零，但單筆梯度是 3、1、負 1、負 3，都不是零。", "At the minimum the true gradient is zero, but the single-sample gradients are 3, 1, minus 1, minus 3, none zero."), 1: ("每步仍被推一下，推的幅度是學習率乘這個數，固定的學習率讓推力永遠不縮小。", "Each step still gets a push of the learning rate times that number, and a fixed rate never shrinks it."), 2: ("所以固定步長的 SGD 不會停在最小點，它會收斂到一個分布，也就是一條帶子，而不是一個點。", "So fixed-step SGD does not settle at the minimum; it converges to a distribution, a band, not a point."), 3: ("多迭代只是在帶內繼續晃，帶寬由學習率決定。", "More iterations only keep it wandering inside the band, whose width the learning rate sets."), 4: ("", "")})

    g = d.page("遞減步長與 Robbins–Monro 條件", "Decreasing steps and the Robbins–Monro conditions")
    g.beat = 1
    box(g, 50, 126, 570, 190, BLUE, BLUEL, T("走得到", "it gets there"), r"$\sum_{k}\eta_k=\infty$" + T("\n總路程無限,起點再遠也走得到", "\ninfinite total distance: it reaches even a far start"), ts=22, bs=28)
    g.beat = 2
    box(g, 660, 126, 570, 170, ORG, ORGL, T("雜訊退場", "the noise retires"), r"$\sum_{k}\eta_k^2<\infty$" + T("\n注入的雜訊方差 $\\eta_k^2\\sigma^2$ 加總有限", "\nthe injected variances $\\eta_k^2\\sigma^2$ sum to a finite number"), ts=22, bs=26)
    rows = [[T("固定 0.2", "fixed 0.2"), r"$\infty$", r"$\infty$", T("停在 0.278", "stalls at 0.278")], [r"$1/(k+1)$", r"$\infty$", r"$\pi^2/6=1.645$", T("殘差 $2.5/k$", "residual $2.5/k$")], [r"$1/\sqrt{k+1}$", r"$\infty$", r"$\infty$", T("較慢,約 $1.25/\\sqrt k$", "slower, about $1.25/\\sqrt k$")], [r"$0.5/(k+1)^2$", T("$0.82$(有限)", "$0.82$ (finite)"), r"$0.27$", T("停在 1.96", "stalls at 1.96")]]
    g.beat = 3
    table(g, 60, 320, [300, 220, 260, 400], [T("步長 $\\eta_k$", "step $\\eta_k$"), r"$\sum\eta_k$", r"$\sum\eta_k^2$", T("結果", "result")], rows, rh=52, fs=22, hfs=20, styles=[(REDL, RED), (GRNL, GRN), (GOLDL, GOLD), (REDL, RED)], gap=6)
    banner(g, T("縮得太快(第 4 列)走到半路就停;$1/\\sqrt{k+1}$ 不滿足第二條件但仍收斂,只是慢", "shrinking too fast (row 4) stops halfway; $1/\\sqrt{k+1}$ misses condition two yet converges, slowly"))
    d.done(g, "Robbins–Monro", "Robbins–Monro", {0: ("帶子的厚度跟學習率有關，很自然的做法是讓步長縮小。Robbins 與 Monro 在 1951 年給出兩個條件。第一個管走得到：步長加總要無限。", "The band's thickness depends on the rate, so shrink the step. Robbins and Monro gave two conditions in 1951. The first is about getting there: the steps must sum to infinity."), 1: ("第二個管雜訊退場：步長平方加總要有限。", "The second is about the noise retiring: the squared steps must sum to a finite number."), 2: ("", ""), 3: ("用四種步長檢查：固定 0.2 停在 0.278；1 除以 k 加 1 殘差是 2.5 除以 k；1 除以根號較慢；0.5 除以 k 加 1 的平方縮太快，停在 1.96。Robbins–Monro 是充分條件，不是必要條件。", "Checking four step sizes: fixed 0.2 stalls at 0.278; 1 over k plus 1 gives a residual of 2.5 over k; 1 over the square root is slower; 0.5 over k plus 1 squared shrinks too fast and stalls at 1.96. Robbins–Monro is sufficient, not necessary.")})

    g = d.page("$\\eta_k=1/(k+1)$:正好是樣本平均", "$\\eta_k=1/(k+1)$: exactly the sample average")
    g.beat = 1
    g.rect(120, 112, 1040, 130, fill=CARD, stroke=GRAY, sw=3, radius=16, z=3); g.text(130, 112, 1020, 130, r"$w_{k+1}=\tfrac{k}{k+1}\,w_k+\tfrac{1}{k+1}\,a_{i_k}$", size=40, color=INK, align="center", vcenter=True, z=8)
    g.beat = 2
    chain(g, 50, 260, 1180, 150, [(BLUE, BLUEL, T("$\\eta_0=1$", "$\\eta_0=1$"), T("$w_1$ 直接設成抽到的樣本", "$w_1$ is set to the drawn sample")), (ORG, ORGL, T("之後每步", "every later step"), T("$w$ 是已抽樣本的平均", "$w$ is the average of the samples so far")), (GRN, GRNL, T("平均 $k$ 筆獨立樣本", "average of $k$ independent samples"), r"$\mathrm{Var}=\sigma^2/k=5/k$")], ts=20, bs=22, gap=50, beat0=2)
    g.beat = 5
    rows = [[f"{k}", f"{R1[k]:.4g}", f"{2.5 / k:.4g}"] for k in (10, 100, 1000)]
    table(g, 190, 450, [160, 380, 360], ["$k$", T("殘差(精確遞推)", "residual (exact recursion)"), r"$2.5/k$"], rows, rh=40, fs=20, hfs=18, styles=[None, None, None], gap=4)
    banner(g, T("殘差 $=\\frac12\\cdot5/k=2.5/k$,強凸時一般結果是 $O(1/t)$", "residual $=\\frac12\\cdot5/k=2.5/k$; the general strongly convex rate is $O(1/t)$"))
    d.done(g, "樣本平均", "The sample average", {0: ("學習率取 1 除以 k 加 1 有一個漂亮的解釋。", "Taking the rate 1 over k plus 1 has a lovely interpretation."), 1: ("第一步學習率是 1，直接把 w1 設成抽到的樣本。", "The first step has rate 1, setting w1 to the drawn sample."), 2: ("之後每步的 w 正是已抽樣本的平均。", "After that w is exactly the average of the samples drawn so far."), 3: ("平均 k 筆獨立樣本，誤差方差是 5 除以 k。", "Averaging k independent samples gives an error variance of 5 over k."), 4: ("", ""), 5: ("所以殘差是 2.5 除以 k，和精確遞推吻合；強凸時一般結果是 1 除以 t 的量級。一般凸且梯度有界時，用 1 除以根號 k 的步長，得 1 除以根號 t 量級。", "So the residual is 2.5 over k, matching the exact recursion; the general strongly convex rate is of order 1 over t. For general convex with bounded gradients a 1 over square root of k step gives order 1 over square root of t.")})

    g = d.page("取捨:為什麼深度學習接受 $O(\\eta)$ 的殘差", "Why deep learning accepts an $O(\\eta)$ residual")
    g.beat = 1
    chain(g, 50, 120, 1180, 220, [(BLUE, BLUEL, T("全梯度下降", "full gradient descent"), T("殘差 $\\le0.01$:16 步\n每步掃 $n$ 筆 → 共 $16n$ 筆樣本梯度", "residual $\\le0.01$: 16 steps\n$n$ samples each → $16n$ sample gradients")), (ORG, ORGL, T("SGD,$\\eta_k=1/(k+1)$", "SGD, $\\eta_k=1/(k+1)$"), T("$2.5/k\\le0.01$:$k=250$\n共 250 筆樣本梯度", "$2.5/k\\le0.01$: $k=250$\n250 sample gradients in all"))], ts=22, bs=22, gap=100, beat0=1)
    g.beat = 3
    gold(g, 150, 370, 980, 90, T("$n=10^6$:$16\\times10^6$ 對 $250$,差約 $6.4\\times10^4$ 倍(依賴強凸、玩具題的條件)", "$n=10^6$: $16\\times10^6$ versus $250$, about $6.4\\times10^4$ times (relies on strong convexity and this toy problem)"), size=22)
    g.beat = 4
    box(g, 50, 490, 570, 120, GRN, GRNL, T("兩階段(學習率排程)", "two phases (a schedule)"), T("前期大 $\\eta$ 走得快,後期降 $\\eta$ 壓窄帶子", "large $\\eta$ early to move fast, then smaller $\\eta$ to narrow the band"), ts=20, bs=20)
    box(g, 660, 490, 570, 120, PUR, PURL, T("判斷停止", "judging when to stop"), T("拿不到便宜的全梯度:改看驗證損失進平台或損失的移動平均", "no cheap full gradient: watch the validation loss plateau or a moving average of the loss"), ts=20, bs=20)
    banner(g, T("一個全域的 $\\eta$ 顧得了所有維度嗎?(KP10)", "can one global $\\eta$ serve every dimension? (KP10)"))
    d.done(g, "取捨", "The trade-off", {0: ("遞減步長可以把殘差壓到零，為什麼實務上常還是用不小的學習率？因為每步的價格。", "A decaying step can drive the residual to zero, so why use a sizeable rate in practice? Because of the price of each step."), 1: ("玩具題要殘差小於 0.01：全梯度要 16 步，每步掃全部 n 筆。", "In the toy problem a residual below 0.01 takes 16 steps of full gradient, each scanning all n samples."), 2: ("遞減步長的 SGD 要 250 筆樣本梯度。n 是一百萬時差約六萬四千倍，但這依賴強凸與玩具題的條件。", "SGD with a decaying step needs 250 sample gradients. With n a million that is about 64 thousand times fewer, though it relies on strong convexity and the toy setup."), 3: ("前期用較大學習率走得快，後期降下來壓窄帶子，是常見的兩階段做法。SGD 拿不到便宜的全梯度，改看驗證損失進入平台。", "A common two-phase practice uses a large rate early and lowers it later to narrow the band. Without a cheap full gradient, SGD watches the validation loss plateau."), 4: ("這一頁所有學習率都是一個全域數字。損失在不同方向彎曲程度差很多時，一個學習率顧得了所有維度嗎？這是 KP10 的主題。", "Every rate here is one global number. When curvature differs a lot between directions, can one rate serve all dimensions? That is KP10's topic.")})
    closing(d, [(RED, REDL, "「SGD 永遠不收斂」", "“SGD never converges”", "固定步長收斂到一個分布;遞減步長滿足條件時收斂到最小點", "fixed step: to a distribution; decaying step with the conditions: to the minimum"), (ORG, ORGL, "「$O(1/t)$ 就是實際速度」", "“$O(1/t)$ is the actual speed”", "它只是上界;強凸時全梯度是線性收斂", "it is only a bound; strongly convex full gradient is linear"), (PUR, PURL, "「步長縮得越快越好」", "“the faster the step shrinks the better”", "$\\sum\\eta_k$ 有限時走不到最小點", "if $\\sum\\eta_k$ is finite it never reaches the minimum")],
            ("損失在不同方向彎曲程度差很多時,一個全域的學習率顧得了所有維度嗎?", "When curvature differs a lot between directions, can one global learning rate serve every dimension?"), ("下一個單元:AdaGrad", "next unit: AdaGrad"))
    return d.pages
