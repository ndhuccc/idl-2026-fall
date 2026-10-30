import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lec03_kit import *

KP = "KP11"; N = 7
f = lambda x, y: 0.5 * (x * x + 25 * y * y)
def run(kind, n, eta=0.5, eps=1e-8, gam=0.9):
    x, y = 10.0, 2.0; G = [0.0, 0.0]; v = [0.0, 0.0]; out = []
    for k in range(1, n + 1):
        g = (x, 25 * y)
        if kind == "ada": G = [G[0] + g[0] ** 2, G[1] + g[1] ** 2]; den = [math.sqrt(eps + G[0]), math.sqrt(eps + G[1])]
        else: v = [gam * v[0] + (1 - gam) * g[0] ** 2, gam * v[1] + (1 - gam) * g[1] ** 2]; den = [math.sqrt(eps + v[0]), math.sqrt(eps + v[1])]
        st = eta / den[0]; x, y = x - eta * g[0] / den[0], y - eta * g[1] / den[1]; out.append((k, st, f(x, y)))
    return out
A = run("ada", 200); R = run("rms", 200)
assert [round(A[k - 1][1], 4) for k in (1, 5, 10, 20, 40)] == [0.05, 0.0242, 0.0182, 0.0141, 0.0114] and [round(R[k - 1][1], 4) for k in (1, 5, 10, 20, 40)] == [0.1581, 0.1034, 0.1088, 0.164, 0.4655]
assert [round(A[k - 1][2], 2) for k in (1, 5, 10, 20, 40)] == [73.25, 41.39, 30.10, 21.25, 12.86] and [round(R[k - 1][2], 3) for k in (1, 5, 10, 20)] == [37.632, 13.753, 4.542, 0.249] and R[39][2] < 1e-6
assert round(min(R[:40], key=lambda t: t[1])[1], 4) == 0.1025 or round(min(R[:40], key=lambda t: t[1])[1], 3) == 0.102
assert [round(R[k - 1][2], 2) for k in (80, 100, 150, 200)] == [1.35, 0.61, 0.84, 0.81] and round(A[199][2], 3) == 0.614
W = [0.9 ** k for k in (1, 5, 10, 22)]; assert [round(w, 4) for w in W] == [0.9, 0.5905, 0.3487, 0.0985]
HL = [math.log(0.5) / math.log(g) for g in (0.9, 0.99, 0.999)]; assert [round(h, 2) for h in HL] == [6.58, 68.97, 692.8]
C1 = [1 / (1 - 0.9 ** t) for t in (1, 2, 10, 50)]; C2 = [1 / (1 - 0.999 ** t) for t in (1, 2, 3000)]
assert [round(c, 3) for c in C1] == [10.0, 5.263, 1.535, 1.005] and [round(c, 2) for c in C2] == [1000.0, 500.25, 1.05]
M1 = 0.1 * 10; V1 = 0.001 * 100; X_FIX = 10 - 0.5 * (M1 / 0.1) / math.sqrt(1e-8 + V1 / 0.001); X_RAW = 10 - 0.5 * M1 / math.sqrt(1e-8 + V1)
assert round(X_FIX, 3) == 9.5 and round(X_RAW, 3) == 8.419 and round(0.1 / math.sqrt(0.001), 2) == 3.16

def make(lang):
    d = D(lang, KP, N); T = d.T
    g = d.page("KP11 讓累積學會遺忘", "KP11 Teaching the accumulator to forget")
    g.beat = 1
    box(g, 50, 126, 570, 170, RED, REDL, T("AdaGrad", "AdaGrad"), r"$G_t=G_{t-1}+g_t^2$" + T("\n只增不減:早期大梯度永遠留在分母", "\nonly grows: early large gradients stay in the denominator"), ts=22, bs=24)
    g.beat = 2
    box(g, 660, 126, 570, 170, GRN, GRNL, T("RMSProp:改成指數移動平均", "RMSProp: an exponential moving average"), r"$v_t=\gamma\,v_{t-1}+(1-\gamma)\,g_t^2$" + "\n" + r"$w_t=w_{t-1}-\eta\,g_t/\sqrt{\varepsilon+v_t}$", ts=21, bs=24)
    g.beat = 3
    chain(g, 50, 330, 1180, 130, [(BLUE, BLUEL, T("權重", "weights"), r"$v_t=(1-\gamma)\sum_s\gamma^{t-s}g_s^2$"), (ORG, ORGL, T("$k$ 步前的梯度平方", "a gradient $k$ steps back"), r"$\gamma^{k}$" + T(" 倍,越遠越小", " times, smaller if farther"))], ts=20, bs=24, gap=100, beat0=3)
    g.beat = 5
    gold(g, 150, 490, 980, 120, T("出處要如實:RMSProp 出自 Hinton 2012 年的網路課程講義(Coursera),不是正式發表的論文", "state the source truthfully: RMSProp comes from Hinton's 2012 Coursera lecture notes, not a published paper"), size=22)
    banner(g, T("只改一個地方:累加 → 指數移動平均", "one change only: summation → exponential moving average"))
    d.done(g, "讓累積遺忘", "Forgetting", {0: ("AdaGrad 的累積量只增不減，前期遇到的大梯度會永遠留在分母裡。", "AdaGrad's accumulator only grows, so large early gradients stay in the denominator forever."), 1: ("RMSProp 只改一個地方：把累加換成指數移動平均。", "RMSProp changes one thing: summation becomes an exponential moving average."), 2: ("展開之後，離現在 k 步以前的梯度平方，權重是 gamma 的 k 次方，", "Unrolled, a squared gradient k steps back carries weight gamma to the k,"), 3: ("離得越遠權重越小。", "smaller the farther back."), 4: ("", ""), 5: ("出處要如實：RMSProp 出自 Hinton 在 2012 年網路課程的講義，不是正式發表的論文。", "To be truthful about the source: RMSProp comes from Hinton's 2012 online course notes, not a published paper.")})

    g = d.page("記憶有多長", "How long is the memory")
    g.beat = 1
    for i, (k, w) in enumerate(zip((1, 5, 10, 22), W)):
        x = 70 + i * 140
        g.rect(x, 520 - w * 300, 100, w * 300, fill=BLUEL, stroke=BLUE, sw=3, radius=6, z=3); g.text(x - 20, 520 - w * 300 - 42, 140, 36, f"${w:.4f}$", size=22, color=BLUE, bold=True, align="center", z=8); g.text(x - 20, 524, 140, 60, T(f"$k={k}$ 步前", f"$k={k}$ steps back"), size=18, color=INK, bold=True, align="center", z=8)
    g.line([(60, 520), (600, 520)], INK, sw=3, z=3)
    g.text(60, 112, 540, 36, T("$\\gamma=0.9$ 時,過去的梯度平方的權重 $\\gamma^k$", "weight $\\gamma^k$ of a past squared gradient when $\\gamma=0.9$"), size=20, color=MUTED, bold=True, z=8)
    g.beat = 2
    rows = [[f"{gm:g}", f"{1 / (1 - gm):.0f}" + T(" 步", " steps"), f"{h:.2f}" + T(" 步", " steps")] for gm, h in zip((0.9, 0.99, 0.999), HL)]
    table(g, 640, 130, [140, 190, 220], [r"$\gamma$", T("記憶窗 $\\approx\\frac{1}{1-\\gamma}$", "window $\\approx\\frac{1}{1-\\gamma}$"), T("半衰期", "half-life")], rows, rh=60, fs=22, hfs=18, styles=[None, None, None], gap=6) if False else table(g, 600, 150, [130, 220, 250], [r"$\gamma$", T("記憶窗 $\\approx1/(1-\\gamma)$", "window $\\approx1/(1-\\gamma)$"), T("半衰期 $\\ln0.5/\\ln\\gamma$", "half-life $\\ln0.5/\\ln\\gamma$")], rows, rh=60, fs=22, hfs=17, styles=[(GRNL, GRN), (CARD, GRAY), (CARD, GRAY)], gap=6)
    g.beat = 3
    gold(g, 600, 370, 600, 130, T("記憶窗是經驗的估計;半衰期是權重降到一半所需的步數", "the window is an empirical estimate; the half-life is the steps for a weight to fall to a half"), size=20)
    banner(g, T("$\\gamma=0.9$:約記 10 步;$\\gamma=0.999$:約記 1000 步", "$\\gamma=0.9$: about 10 steps of memory; $\\gamma=0.999$: about 1000"))
    d.done(g, "記憶長度", "Memory length", {0: ("以 gamma 等於 0.9 為例，一步前的權重 0.9，五步前 0.5905，十步前 0.3487，二十二步前只剩 0.0985。", "With gamma 0.9 the weight is 0.9 one step back, 0.5905 five back, 0.3487 ten back and only 0.0985 twenty-two back."), 1: ("記憶有多長？兩個常用的量：記憶窗約一除以一減 gamma 步；半衰期是權重降到一半所需的步數。gamma 0.9：窗約 10 步，半衰期 6.58 步；0.99：約 100 與 68.97；0.999：約 1000 與 692.8。", "How long is the memory? Two common measures: the window is about one over one minus gamma steps; the half-life is the steps for the weight to halve. Gamma 0.9: about 10 and 6.58; 0.99: about 100 and 68.97; 0.999: about 1000 and 692.8."), 2: ("記憶窗是經驗的估計。", "The window is an empirical estimate."), 3: ("", "")})

    g = d.page("同一個碗上，步長會回升", "On the same bowl the step recovers")
    g.text(50, 108, 1180, 34, T("$f=\\frac12(x^2+25y^2)$,起點 $(10,2)$,$\\eta=0.5$,$\\varepsilon=10^{-8}$,RMSProp $\\gamma=0.9$;$x$ 坐標的有效步長", "$f=\\frac12(x^2+25y^2)$, start $(10,2)$, $\\eta=0.5$, $\\varepsilon=10^{-8}$, RMSProp $\\gamma=0.9$; the effective step of $x$"), size=19, color=MUTED, bold=True, align="center", z=8)
    ks = (1, 5, 10, 20, 40)
    rows = [[str(k), f"{A[k - 1][1]:.4f}", f"{R[k - 1][1]:.4f}", f"{A[k - 1][2]:.4g}", (f"{R[k - 1][2]:.4g}" if k < 40 else r"$2.3\times10^{-7}$")] for k in ks]
    g.beat = 1
    table(g, 60, 150, [100, 260, 260, 250, 250], [T("步 $k$", "step $k$"), T("AdaGrad 有效步長", "AdaGrad step"), T("RMSProp 有效步長", "RMSProp step"), T("AdaGrad 損失", "AdaGrad loss"), T("RMSProp 損失", "RMSProp loss")], rows, rh=54, fs=22, hfs=18, styles=[None, None, None, (GOLDL, GOLD), (GRNL, GRN)], beats=[1, 1, 1, 2, 3], gap=6)
    g.beat = 4
    gold(g, 150, 540, 980, 80, T("AdaGrad 從 $0.05$ 降到 $0.0114$ 從不回頭;RMSProp 降到第 6 步最低 $0.1025$ 後升到 $0.4655$,是 AdaGrad 的 $40.9$ 倍", "AdaGrad falls from $0.05$ to $0.0114$ and never returns; RMSProp bottoms at $0.1025$ (step 6) then rises to $0.4655$, $40.9\\times$ AdaGrad's"), size=20)
    banner(g, T("梯度從 $10$ 降到 $1$:AdaGrad 的帳本被大梯度壓著,RMSProp 已經忘掉", "gradients fall from $10$ to $1$: AdaGrad's ledger stays weighed down, RMSProp has forgotten"))
    d.done(g, "步長回升", "The step recovers", {0: ("用同一個碗，兩個方法設定相同：學習率 0.5，RMSProp 的 gamma 是 0.9。第 1 到第 10 步，RMSProp 的有效步長 0.1581、0.1034、0.1088，AdaGrad 一路降到 0.0182。", "On the same bowl with equal settings: learning rate 0.5 and RMSProp gamma 0.9. For steps 1 to 10 RMSProp's effective step is 0.1581, 0.1034, 0.1088, while AdaGrad drops to 0.0182."), 1: ("第 20 步：RMSProp 步長 0.164，損失 0.249；AdaGrad 步長 0.0141，損失 21.25。", "Step 20: RMSProp's step is 0.164 with loss 0.249; AdaGrad's 0.0141 with loss 21.25."), 2: ("第 40 步：RMSProp 步長 0.4655，損失降到十的負七次方；AdaGrad 步長 0.0114，損失 12.86。", "Step 40: RMSProp's step is 0.4655 and its loss falls to about 10 to the minus 7; AdaGrad's step is 0.0114 with loss 12.86."), 3: ("AdaGrad 的步長從不回頭；RMSProp 隨梯度變小而回升，到第 40 步是 AdaGrad 的 40.9 倍。", "AdaGrad's step never returns; RMSProp's rises as the gradient shrinks, 40.9 times AdaGrad's at step 40."), 4: ("", "")})

    g = d.page("回升也有代價", "Recovering has a price")
    xr, yr = (60, 200), (0, 9)
    ux, uy = axes(g, 60, 120, 760, 460, xr, yr)
    g.beat = 1
    g.line([(ux(k), uy(min(R[k - 1][2], 8.9))) for k in range(60, 201)], GRN, sw=4, z=5)
    g.beat = 2
    g.line([(ux(k), uy(min(A[k - 1][2], 8.9))) for k in range(60, 201)], BLUE, sw=4, z=5)
    g.text(60, 586, 760, 34, T("第 60–200 步的損失(綠:RMSProp,藍:AdaGrad)", "loss over steps 60–200 (green: RMSProp, blue: AdaGrad)"), size=18, color=MUTED, bold=True, align="center", z=8)
    g.beat = 1
    box(g, 860, 126, 370, 200, GRN, GRNL, T("RMSProp", "RMSProp"), T("固定 $\\eta$ 在最小點附近來回彈跳\n第 80 步 $1.35$,100 / 150 / 200 步 $0.608,\\ 0.836,\\ 0.812$", "a fixed $\\eta$ bounces around the minimum\nstep 80: $1.35$; steps 100/150/200: $0.608,\\ 0.836,\\ 0.812$"), ts=21, bs=18)
    g.beat = 2
    box(g, 860, 350, 370, 150, BLUE, BLUEL, T("AdaGrad", "AdaGrad"), T("較慢但穩\n第 200 步 $0.614$", "slower but steady\nstep 200: $0.614$"), ts=21, bs=22)
    g.beat = 3
    gold(g, 860, 520, 370, 100, T("步長要不要再搭配遞減排程:常見實務", "pair it with a decay schedule: common practice"), size=19)
    banner(g, T("沒有收斂到零:回升讓它能繼續走,也讓它會彈跳", "no convergence to zero: recovery keeps it moving, and bouncing"))
    d.done(g, "回升的代價", "The price of recovery", {0: ("回升也有代價。步長變大以後，固定的學習率會讓 RMSProp 在最小點附近來回彈跳：第 80 步損失又升到 1.35，第 100、150、200 步是 0.608、0.836、0.812，沒有收斂到零。", "Recovery has a price. Once the step is large a fixed rate lets RMSProp bounce around the minimum: at step 80 the loss rises to 1.35 and at steps 100, 150, 200 it is 0.608, 0.836, 0.812, not converging to zero."), 1: ("AdaGrad 較慢但穩，第 200 步是 0.614。", "AdaGrad is slower but steady: 0.614 at step 200."), 2: ("步長要不要再搭配遞減排程，是常見的實務。", "Whether to add a decay schedule is a common practical question."), 3: ("", "")})

    g = d.page("Adam:方向與尺度", "Adam: direction and scale")
    g.beat = 1
    box(g, 50, 126, 570, 190, BLUE, BLUEL, T("方向 $m_t$(KP05 的動量)", "direction $m_t$ (KP05's momentum)"), r"$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$" + T("\n來回震盪的分量互相抵消", "\noscillating parts cancel"), ts=21, bs=24)
    g.beat = 2
    box(g, 660, 126, 570, 190, ORG, ORGL, T("尺度 $v_t$(RMSProp 的 $v$)", "scale $v_t$ (RMSProp's $v$)"), r"$v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$" + T("\n誰的梯度大,誰的步子小", "\nthe larger the gradient, the smaller the step"), ts=21, bs=24)
    g.beat = 3
    g.rect(190, 350, 900, 100, fill=CARD, stroke=GRAY, sw=3, radius=16, z=3); g.text(200, 350, 880, 100, r"$w_t=w_{t-1}-\eta\,\dfrac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}$".replace(r"\dfrac", r"\frac"), size=38, color=INK, align="center", vcenter=True, z=8)
    g.beat = 4
    gold(g, 150, 480, 980, 130, T("常見預設 $\\beta_1=0.9$(約記 10 步)、$\\beta_2=0.999$(約記 1000 步)、$\\varepsilon=10^{-8}$、$\\eta=10^{-3}$:只是常見的起點,不是定理", "usual defaults $\\beta_1=0.9$ (about 10 steps), $\\beta_2=0.999$ (about 1000), $\\varepsilon=10^{-8}$, $\\eta=10^{-3}$: common starting points, not theorems"), size=21)
    banner(g, T("方向除以尺度,每個坐標各自縮放", "direction divided by scale, per coordinate"))
    d.done(g, "Adam", "Adam", {0: ("RMSProp 只管各個坐標的尺度，沒有方向的記憶。KP05 的動量補的是方向：把梯度做指數平均，來回震盪的分量互相抵消。", "RMSProp handles each coordinate's scale only, with no memory of direction. KP05's momentum supplies direction: an exponential average of the gradient cancels oscillating parts."), 1: ("Adam 同時維護梯度的指數平均，也就是方向，", "Adam keeps an exponential average of the gradient, the direction,"), 2: ("和梯度平方的指數平均，也就是尺度，就是 RMSProp 的那個量。", "and one of the squared gradient, the scale, which is RMSProp's quantity."), 3: ("更新式把方向除以尺度，每個坐標各自縮放。", "The update divides the direction by the scale, per coordinate."), 4: ("常見預設值是 0.9、0.999、10 的負 8 次方與學習率 0.001，只是常見的起點，不是定理。", "The usual defaults 0.9, 0.999, 10 to the minus 8 and learning rate 0.001 are common starting points, not theorems.")})

    g = d.page("偏差修正不是可有可無", "Bias correction is not optional")
    g.beat = 1
    box(g, 50, 126, 570, 150, RED, REDL, T("從零開始,前幾步一定偏低", "starting from zero, early steps are biased low"), T("$m_1=0.1\\,g_1$(只剩十分之一)\n$v_1=0.001\\,g_1^2$(只剩千分之一)", "$m_1=0.1\\,g_1$ (a tenth)\n$v_1=0.001\\,g_1^2$ (a thousandth)"), ts=20, bs=22)
    g.beat = 2
    box(g, 660, 126, 570, 150, GRN, GRNL, T("除掉權重和 $1-\\beta^t$", "divide by the weight sum $1-\\beta^t$"), r"$\hat m_t=\dfrac{m_t}{1-\beta_1^{t}},\quad \hat v_t=\dfrac{v_t}{1-\beta_2^{t}}$".replace(r"\dfrac", r"\frac"), ts=20, bs=26)
    rows = [[T("$1/(1-0.9^t)$", "$1/(1-0.9^t)$")] + [f"{c:.4g}" for c in C1], [T("$1/(1-0.999^t)$", "$1/(1-0.999^t)$")] + [f"{c:.5g}" for c in C2] + [""]]
    g.beat = 3
    table(g, 60, 300, [250, 190, 190, 190, 190], ["$t$", "1", "2", "10 / 3000", "50"], [[rows[0][0], rows[0][1], rows[0][2], rows[0][3], rows[0][4]], [rows[1][0], rows[1][1], rows[1][2], rows[1][3], ""]], rh=50, fs=22, hfs=18, styles=[(CARD, GRAY), (CARD, GRAY)], gap=6) if False else None
    g.text(60, 300, 1160, 34, T("修正倍率:$t$ 小時很大,$t$ 變大後趨近 $1$", "correction factor: large for small $t$, tending to $1$ as $t$ grows"), size=20, color=MUTED, bold=True, align="center", z=8)
    box(g, 50, 340, 570, 100, ORG, ORGL, r"$1/(1-0.9^t)$", T(f"$t=1,2,10,50$:${C1[0]:.0f},\\ {C1[1]:.2f},\\ {C1[2]:.2f},\\ {C1[3]:.3f}$", f"$t=1,2,10,50$: ${C1[0]:.0f},\\ {C1[1]:.2f},\\ {C1[2]:.2f},\\ {C1[3]:.3f}$"), ts=19, bs=22)
    box(g, 660, 340, 570, 100, PUR, PURL, r"$1/(1-0.999^t)$", T(f"$t=1,2,3000$:${C2[0]:.0f},\\ {C2[1]:.2f},\\ {C2[2]:.2f}$", f"$t=1,2,3000$: ${C2[0]:.0f},\\ {C2[1]:.2f},\\ {C2[2]:.2f}$"), ts=19, bs=22)
    g.beat = 4
    box(g, 50, 470, 570, 140, GRN, GRNL, T("第 1 步(修正後)", "step 1 (corrected)"), T("$x:\\ 10\\to9.5$\n$\\hat m_1=g_1$,$\\hat v_1=g_1^2$", "$x:\\ 10\\to9.5$\n$\\hat m_1=g_1$, $\\hat v_1=g_1^2$"), ts=20, bs=22)
    g.beat = 5
    box(g, 660, 470, 570, 140, RED, REDL, T("不修正", "uncorrected"), T("$x:\\ 10\\to8.419$;恆定梯度時步長 $3.16\\,\\eta$\n不是「停擺」,而是被放大", "$x:\\ 10\\to8.419$; with a constant gradient the step is $3.16\\,\\eta$\nnot “stalling” but amplified"), ts=20, bs=19)
    banner(g, T("預設 $\\beta$ 下不修正,早期步長被放大 3 至 6.6 倍,約 3000 步才退場", "with default $\\beta$ and no correction, early steps are 3 to 6.6× too big until about step 3000"))
    d.done(g, "偏差修正", "Bias correction", {0: ("從零開始，前幾步的 m 與 v 一定偏低：第 1 步 m 只剩真梯度的十分之一，v 只剩千分之一。", "Starting from zero, early m and v are biased low: at step 1 m is a tenth of the true gradient and v a thousandth."), 1: ("原因是指數平均的權重加起來只有 1 減 beta 的 t 次方。除掉它就無偏了。", "The reason is that the exponential weights sum to only 1 minus beta to the t. Dividing by it removes the bias."), 2: ("修正倍率在 t 小時很大，t 變大後趨近 1。", "The correction factor is large for small t and tends to 1 as t grows."), 3: ("第 1 步修正後，x 從 10 到 9.5，正好是學習率乘符號。", "At step 1 corrected, x goes from 10 to 9.5, exactly the learning rate times the sign."), 4: ("", ""), 5: ("不修正，x 從 10 到 8.419。來源說開頭會幾乎停擺，用預設值計算結果相反：步長被放大。", "Uncorrected, x goes from 10 to 8.419. The source says it nearly stalls; with the default values the result is the opposite: the step is amplified.")})

    g = d.page("比較最佳化器是一場受控實驗", "Comparing optimizers is a controlled experiment")
    rows = [[T("固定不變", "hold fixed"), T("同一份資料、模型、初始化、學習率排程與算力預算", "the same data, model, initialization, schedule and compute budget"), T("差異才能歸因於最佳化器", "so the difference is due to the optimizer")], [T("重複多次", "repeat"), T("多個隨機種子;每個最佳化器各自調過學習率", "several seeds; tune every optimizer's learning rate"), T("分辨真差異與隨機波動;只調偏愛的那個不公平", "tell real gaps from noise; tuning only your favorite is unfair")], [T("要記錄", "record"), T("訓練損失、驗證指標、牆鐘時間、跨種子變異", "training loss, validation metric, wall-clock time, seed variance"), T("快、好、穩要一起比", "compare fast, good, stable together")]]
    g.beat = 1
    table(g, 60, 120, [180, 540, 460], ["", T("怎麼做", "how"), T("為什麼", "why")], rows, rh=110, fs=19, hfs=20, styles=[(BLUEL, BLUE), (ORGL, ORG), (GRNL, GRN)], beats=[1, 2, 3], gap=8)
    g.beat = 4
    gold(g, 150, 500, 980, 110, T("結論要附條件:「在這組資料、模型與預算下,A 比 B 好」,而不是「A 比 B 好」", "state conclusions with conditions: “on this data, model and budget A beats B”, not “A beats B”"), size=24)
    banner(g, T("Adam 是常用預設之一;沒有定理保證它處處最好", "Adam is a common default; no theorem makes it best everywhere"))
    d.done(g, "受控實驗", "A controlled experiment", {0: ("Adam 是目前最常用的預設之一，但沒有定理保證它處處最好。要比較兩個最佳化器，要像做實驗。", "Adam is one of the most common defaults, but no theorem makes it best everywhere. Comparing two optimizers must be done like an experiment."), 1: ("固定不變：同一份資料、模型、初始化、學習率排程與算力預算。", "Hold fixed: the same data, model, initialization, schedule and compute budget."), 2: ("重複多次：多個隨機種子，每個最佳化器各自調過學習率。", "Repeat: several random seeds, with every optimizer's learning rate tuned."), 3: ("要記錄：訓練損失、驗證指標、牆鐘時間與跨種子的變異。", "Record: training loss, validation metric, wall-clock time and variance across seeds."), 4: ("結論也要附條件。", "Conclusions must carry conditions too.")})
    closing(d, [(RED, REDL, "「Adam 一定最好」", "“Adam is always best”", "沒有定理保證;比較要用受控實驗", "no theorem guarantees it; compare with controlled experiments"), (ORG, ORGL, "「偏差修正可有可無」", "“bias correction is optional”", "不修正,早期步長被放大 3 至 6.6 倍", "uncorrected, early steps are inflated 3 to 6.6×"), (PUR, PURL, "「RMSProp 步長回升就是更好」", "“a recovering RMSProp step is simply better”", "固定 $\\eta$ 也會在最小點附近彈跳", "a fixed $\\eta$ also bounces near the minimum")],
            ("我們比較的訓練損失與驗證指標可能差很多。訓練損失低就夠了嗎?", "The training loss and validation metric we compare can differ a lot. Is a low training loss enough?"), ("下一個單元:平坦極小值與 SAM 的動機", "next unit: flat minima and the motivation for SAM"))
    return d.pages
