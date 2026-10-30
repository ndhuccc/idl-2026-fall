import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lec03_kit import *

KP = "KP05"; N = 7
f = lambda x, y: 0.5 * (x * x + 25 * y * y)
def run(eta, al, n=40):
    p = [10.0, 2.0]; v = [0.0, 0.0]; out = [f(*p)]; ps = [tuple(p)]
    for _ in range(n):
        g = (p[0], 25 * p[1]); v = [al * v[0] - eta * g[0], al * v[1] - eta * g[1]]; p = [p[0] + v[0], p[1] + v[1]]; out.append(f(*p)); ps.append(tuple(p))
    return out, ps
GD, GDP = run(0.06, 0); MO, MOP = run(0.06, 0.8)
assert [round(GD[k], 4) for k in (1, 2, 10, 20, 40)] == [56.68, 42.1624, 14.5054, 4.2081, 0.3542] and [round(MO[k], 4) for k in (1, 2, 10, 20, 40)] == [56.68, 80.0364, 3.1677, 0.827, 0.0163]
assert [round(v, 3) for v in MOP[2]] == [8.356, -1.9] and [round(v, 3) for v in GDP[2]] == [8.836, 0.5]
LR = {e: run(e, 0)[0][40] for e in (0.01, 0.04, 0.08, 0.09)}; assert LR[0.09] > 1e9 and round(LR[0.04], 2) == 1.91
M09 = run(0.09, 0.8)[0][40]; M20 = run(0.2, 0.8)[0][40]; assert round(M09, 4) == 0.003 and M20 > 1e30
assert abs(2 / 26 - 0.0769) < 1e-4 and round(24 / 26, 3) == 0.923 and abs(2 * 1.8 / 25 - 0.144) < 1e-12 and abs(0.06 / 0.2 - 0.3) < 1e-12
def sci(v): 
    e = int(math.floor(math.log10(v))); return f"${v / 10 ** e:.2f}\\times10^{{{e}}}$"

def make(lang):
    d = D(lang, KP, N); T = d.T
    g = d.page("KP05 狹長的碗:一個學習率顧不了兩個方向", "KP05 A long bowl needs more than one rate")
    g.beat = 1
    chain(g, 50, 120, 1180, 150, [(BLUE, BLUEL, T("$x$ 方向 $\\lambda=1$", "$x$: $\\lambda=1$"), r"$x\times(1-\eta)$"), (ORG, ORGL, T("$y$ 方向 $\\lambda=25$", "$y$: $\\lambda=25$"), r"$y\times(1-25\eta)$"), (PUR, PURL, T("條件數", "condition number"), r"$\kappa=25$")], ts=22, bs=28, beat0=1)
    rows = [[f"{e:g}", f"{1 - e:.2f}".rstrip("0").rstrip("."), f"{1 - 25 * e:.2f}".rstrip("0").rstrip("."), (f"{v:.3g}" if v < 1e6 else sci(v))] for e, v in LR.items()]
    g.beat = 4
    table(g, 60, 290, [140, 190, 190, 300], [r"$\eta$", T("$x$ 倍率", "$x$ factor"), T("$y$ 倍率", "$y$ factor"), T("40 步後損失", "loss after 40 steps")], rows, rh=54, fs=22, hfs=20, styles=[None, None, (GOLDL, GOLD), (REDL, RED)], gap=6)
    g.beat = 5
    gold(g, 910, 290, 320, 270, T("穩定要 $\\eta<2/\\lambda_{\\max}=0.08$;最好的固定 $\\eta^*\\approx0.0769$ 誤差每步最多縮到 $0.923$", "stability needs $\\eta<0.08$; the best fixed $\\eta^*\\approx0.0769$ only shrinks the error to $0.923$ per step"), size=20)
    banner(g, T("調大 $\\eta$ 則 $y$ 發散,調小則 $x$ 走不動", "raise $\\eta$ and $y$ diverges; lower it and $x$ barely moves"))
    d.done(g, "狹長的碗", "The long bowl", {0: ("回到玩具碗。x 方向曲率 1，y 方向曲率 25，條件數是 25。", "Back to the toy bowl. Curvature 1 along x and 25 along y: condition number 25."), 1: ("固定學習率的梯度下降，x 每步乘 1 減 eta，", "With a fixed learning rate, x is multiplied by 1 minus eta each step,"), 2: ("y 每步乘 1 減 25 eta。", "and y by 1 minus 25 eta."), 3: ("", ""), 4: ("調大學習率想讓 x 走快，y 就發散；調小讓 y 穩定，x 就走不動。", "Raise the rate to speed up x and y diverges; lower it to steady y and x hardly moves."), 5: ("穩定要求學習率小於 0.08。即使取最好的固定學習率 0.0769，誤差每步最多只縮小到 0.923。", "Stability needs a rate below 0.08. Even the best fixed rate 0.0769 only shrinks the error to 0.923 per step.")})

    g = d.page("動量:讓最佳化器記得剛才往哪走", "Momentum: remember where you just went")
    g.beat = 1
    ids = chain(g, 50, 140, 1180, 210, [(BLUE, BLUEL, T("上一步更新量的一部分", "a fraction of the last update"), r"$\alpha\,\Delta w^{(k-1)}$"), (ORG, ORGL, T("這一步的負梯度", "this step's negative gradient"), r"$-\eta\nabla f(w^{(k)})$"), (GRN, GRNL, T("這一步的更新量", "this step's update"), r"$\Delta w^{(k)}$")], ts=22, bs=30, beat0=1, gap=90)
    g.beat = 4
    g.rect(190, 390, 900, 100, fill=CARD, stroke=GRAY, sw=3, radius=16, z=3); g.text(200, 390, 880, 100, r"$w^{(k+1)}=w^{(k)}+\Delta w^{(k)},\quad \Delta w^{(-1)}=\mathbf{0}$", size=36, color=INK, align="center", vcenter=True, z=8)
    g.beat = 5
    gold(g, 190, 520, 900, 90, T("$\\alpha\\in[0,1)$ 是動量係數,常取 $0.9$;$\\alpha=0$ 時精確退化成純梯度下降", "$\\alpha\\in[0,1)$ is the momentum coefficient, often $0.9$; $\\alpha=0$ is exactly plain gradient descent"), size=22)
    banner(g, T("像重鐵球滾下山谷:慣性讓前進的力一直疊加", "like a heavy ball rolling down a valley: inertia accumulates the forward push"))
    d.done(g, "動量更新", "The momentum update", {0: ("純梯度下降只看此刻的斜率。動量讓最佳化器記得剛才往哪走：更新量等於上一步更新量的一部分，", "Plain gradient descent only sees the slope now. Momentum remembers where it just went: the update is a fraction of the last update,"), 1: ("加上這一步的負梯度。", "plus this step's negative gradient."), 2: ("合起來就是這一步的更新量。", "Together that is this step's update."), 3: ("", ""), 4: ("再用更新量移動參數，起點的更新量取零。", "Then move the parameters by that update, starting with a zero update."), 5: ("動量係數常取 0.9；係數為零時，精確退化成純梯度下降。", "The momentum coefficient is often 0.9; at zero it is exactly plain gradient descent.")})

    g = d.page("手算兩步:損失不一定每步都降", "Two steps by hand: the loss need not fall every step")
    xr, yr = (-12, 12), (-4, 4)
    ux, uy = axes(g, 50, 120, 700, 480, xr, yr)
    g.beat = 1
    bowl_contours(g, ux, uy, levels=(4, 16, 36, 64))
    g.line([(ux(x), uy(y)) for x, y in GDP[:12]], BLUE, sw=4, z=6); g.line([(ux(x), uy(y)) for x, y in MOP[:12]], ORG, sw=4, z=6)
    for x, y in GDP[:3]: dot(g, ux(x), uy(y), 7, BLUE)
    for x, y in MOP[:3]: dot(g, ux(x), uy(y), 7, ORG)
    g.text(50, 604, 700, 34, T("藍:純梯度下降　橘:動量($\\alpha=0.8$)", "blue: plain GD · orange: momentum ($\\alpha=0.8$)"), size=18, color=MUTED, bold=True, align="center", z=8)
    g.beat = 2
    box(g, 790, 126, 440, 170, ORG, ORGL, T("動量 $\\theta_2$", "momentum $\\theta_2$"), r"$\Delta w^{(1)}=(-1.044,\,-0.9)$" + "\n" + r"$\theta_2=(8.356,\,-1.9)$", ts=22, bs=24)
    g.beat = 3
    box(g, 790, 316, 440, 150, BLUE, BLUEL, T("純梯度下降 $\\theta_2$", "plain GD $\\theta_2$"), r"$\theta_2=(8.836,\,0.5)$", ts=22, bs=26)
    g.beat = 4
    gold(g, 790, 490, 440, 120, T("$f(\\theta_1)=56.68$,$f(\\theta_2)=80.04$:比上一步高!(GD 為 $42.16$)", "$f(\\theta_1)=56.68$, $f(\\theta_2)=80.04$: higher than before! (GD: $42.16$)"), size=21)
    banner(g, T("第 1 步和純梯度下降一樣:因為還沒有「上一步」", "step 1 equals plain GD: there is no “last step” yet"))
    d.done(g, "手算兩步", "Two steps", {0: ("玩具碗，起點 10、2，學習率 0.06，動量係數 0.8。第一步和純梯度下降一樣，到 9.4、負 1.0。", "Toy bowl, start 10, 2, learning rate 0.06, momentum 0.8. The first step equals plain gradient descent: 9.4, minus 1.0."), 1: ("第二步：更新量是 0.8 乘上一步，加上這一步的負梯度，得到負 1.044、負 0.9，位置 8.356、負 1.9。", "Step two: the update is 0.8 times the last one plus this step's negative gradient, minus 1.044, minus 0.9, giving the position 8.356, minus 1.9."), 2: ("純梯度下降的第二步是 8.836、0.5。", "Plain gradient descent's second step is 8.836, 0.5."), 3: ("動量的損失不是單調的：第二步損失 80.04，比上一步 56.68 高；純梯度下降是 42.16。動量帶著慣性衝，會衝過頭。", "Momentum's loss is not monotone: at step two it is 80.04, above the previous 56.68, while plain gradient descent has 42.16. It rushes with inertia and can overshoot."), 4: ("", "")})

    g = d.page("為什麼有效:抵消與累加", "Why it works: cancellation and accumulation")
    g.beat = 1
    g.rect(190, 118, 900, 100, fill=CARD, stroke=GRAY, sw=3, radius=16, z=3); g.text(200, 118, 880, 100, r"$\Delta w^{(k)}=-\eta\sum_{i=0}^{k}\alpha^i\,\nabla f\big(w^{(k-i)}\big)$", size=36, color=INK, align="center", vcenter=True, z=8)
    g.text(190, 220, 900, 36, T("歷史梯度的指數加權和:越舊權重 $\\alpha^i$ 越小", "an exponentially weighted sum of past gradients: older ones weigh $\\alpha^i$"), size=20, color=MUTED, bold=True, align="center", z=8)
    g.beat = 2
    box(g, 50, 280, 570, 150, RED, REDL, T("抵消", "cancellation"), T("$y$ 方向梯度每步正負交替(碗很陡)\n相加時互相抵消,擺動變小", "the $y$ gradient alternates in sign (steep bowl)\nsummed, they cancel and the swing shrinks"), ts=22, bs=20)
    g.beat = 3
    box(g, 660, 280, 570, 150, GRN, GRNL, T("累加", "accumulation"), T("$x$ 方向梯度每步同號(碗很平)\n相加時越疊越大,有效步長 $\\eta/(1-\\alpha)$", "the $x$ gradient keeps one sign (flat bowl)\nsummed, they build up: effective step $\\eta/(1-\\alpha)$"), ts=22, bs=20)
    g.beat = 4
    rows = [[T("純梯度下降 $f$", "plain GD $f$")] + [f"{GD[k]:.4g}" for k in (1, 2, 10, 20, 40)], [T("動量 $f$($\\alpha=0.8$)", "momentum $f$ ($\\alpha=0.8$)")] + [f"{MO[k]:.4g}" for k in (1, 2, 10, 20, 40)]]
    table(g, 60, 440, [330, 150, 150, 150, 150, 150], [T("步 $k$", "step $k$"), "1", "2", "10", "20", "40"], rows, rh=46, fs=20, hfs=18, styles=[(CARD, GRAY), (GOLDL, GOLD)], gap=6)
    banner(g, T("第 2 步較差,第 10 步已好 4 倍以上,第 40 步好 20 倍以上", "worse at step 2, over 4× better by step 10, over 20× better by step 40"))
    d.done(g, "抵消與累加", "Cancel and accumulate", {0: ("把遞迴往回展開，更新量是歷史梯度的指數加權和，越舊的梯度權重越小。", "Unrolled, the update is an exponentially weighted sum of past gradients, older ones weighing less."), 1: ("結果之一是抵消：y 方向的梯度每步正負交替，相加時互相抵消，橫向擺動變小。", "One result is cancellation: the y gradient alternates in sign and cancels when summed, so the swing shrinks."), 2: ("結果之二是累加：x 方向的梯度每步同號，越疊越大，有效步長是學習率除以 1 減動量係數，動量 0.8 時是 5 倍。", "The other is accumulation: the x gradient keeps one sign and builds up; the effective step is the learning rate over 1 minus the coefficient, 5 times at 0.8."), 3: ("", ""), 4: ("同樣 40 步，動量在第 2 步比較差，第 10 步已好 4 倍以上，第 40 步好 20 倍以上。", "Over the same 40 steps momentum is worse at step 2, over 4 times better by step 10 and over 20 times by step 40.")})

    g = d.page("動量的限制", "The limits of momentum")
    g.beat = 1
    box(g, 50, 126, 570, 150, RED, REDL, T("損失不是每步都降", "the loss does not fall every step"), T("帶著慣性衝,會衝過頭;整體下降但不保證單調", "it rushes with inertia and can overshoot; downhill overall, not monotone"), ts=22, bs=22)
    g.beat = 2
    box(g, 660, 126, 570, 150, ORG, ORGL, T("學習率仍有上限", "the rate still has a ceiling"), r"$0<\eta<\dfrac{2(1+\alpha)}{\lambda_{\max}}=0.144$".replace(r"\dfrac", r"\frac") + T("\n($\\alpha=0.8$;純 GD 是 $0.08$)", "\n($\\alpha=0.8$; plain GD: $0.08$)"), ts=22, bs=24)
    g.beat = 3
    rows = [["0.09", "0", sci(LR[0.09]), T("發散", "diverges")], ["0.09", "0.8", f"{M09:.3g}", T("收斂", "converges")], ["0.2", "0.8", sci(M20), T("仍發散", "still diverges")]]
    table(g, 60, 300, [180, 180, 440, 340], [r"$\eta$", r"$\alpha$", T("40 步後 $f$", "$f$ after 40 steps"), T("結果", "result")], rows, rh=64, fs=26, styles=[(REDL, RED), (GRNL, GRN), (REDL, RED)], gap=8)
    g.beat = 4
    gold(g, 190, 570, 900, 58, T("$\\alpha$ 太大衝過頭、來回很久;它也沒改變「只有梯度、沒有曲率」", "too large an $\\alpha$ overshoots; it still has gradients only, no curvature"), size=21)
    banner(g, T("動量放寬了上限,但不能讓任何學習率都安全", "momentum relaxes the ceiling but does not make every rate safe"))
    d.done(g, "動量的限制", "Limits", {0: ("動量的損失不是每步都降，它帶著慣性衝，會衝過頭。", "Momentum's loss does not fall every step; it carries inertia and can overshoot."), 1: ("學習率仍有上限：動量的穩定範圍是 2 乘 1 加 alpha 除以最大曲率，alpha 0.8 時上限從 0.08 放大到 0.144。", "The rate still has a ceiling: the stable range is 2 times 1 plus alpha over the largest curvature; at alpha 0.8 the ceiling rises from 0.08 to 0.144."), 2: ("所以學習率 0.09 在純梯度下降會發散，在動量下反而收斂；但學習率 0.2 仍然發散。", "So a rate of 0.09 diverges for plain gradient descent but converges with momentum; a rate of 0.2 still diverges."), 3: ("動量多一個要調的超參數，係數太大會衝過頭、來回很久。它也沒改變只有梯度、沒有曲率這件事。", "Momentum adds a hyperparameter; too large a coefficient overshoots for long. It also does not change that we have gradients only, no curvature."), 4: ("", "")})
    closing(d, [(RED, REDL, "「動量一定讓每一步都更好」", "“momentum improves every step”", "第 2 步損失還上升", "the loss rose at step 2"), (ORG, ORGL, "「$\\alpha$ 越大越好」", "“a larger $\\alpha$ is always better”", "太大會衝過頭、來回很久", "too large overshoots and swings"), (PUR, PURL, "「動量用了二階資訊」", "“momentum uses second-order information”", "它只累積過去的一階梯度", "it only accumulates past first-order gradients")],
            ("整章到這裡,我們都把梯度當成「已經算好」。這個梯度到底是怎麼算出來的?", "So far we treated the gradient as already computed. How is it actually computed?"), ("下一個單元:反向傳播", "next unit: backpropagation"))
    return d.pages
