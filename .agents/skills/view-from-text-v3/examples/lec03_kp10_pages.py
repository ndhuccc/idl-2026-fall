import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lec03_kit import *

KP = "KP10"; N = 7
def gd(eta, n=30):
    x, y = 10.0, 2.0
    for _ in range(n): x, y = x - eta * x, y - eta * 25 * y
    return x, y
GDR = {e: gd(e) for e in (0.01, 0.07, 0.09)}
assert round(GDR[0.07][0], 3) == 1.134 and GDR[0.09][1] > 1000 and abs(GDR[0.01][0] - 7.397) < 1e-3
def ada(n, eta=1.0, eps=1e-8):
    x, y = 10.0, 2.0; G = [0.0, 0.0]; tr = []
    for _ in range(n):
        g = (x, 25 * y); G = [G[0] + g[0] ** 2, G[1] + g[1] ** 2]; st = [eta / math.sqrt(eps + G[0]), eta / math.sqrt(eps + G[1])]
        x, y = x - st[0] * g[0], y - st[1] * g[1]; tr.append((g, tuple(G), tuple(st), (st[0] * g[0], st[1] * g[1]), (x, y)))
    return tr
TR = ada(30)
assert TR[0][1] == (100.0, 2500.0) and [round(v, 4) for v in TR[0][2]] == [0.1, 0.02] and [round(v) for v in TR[0][4]] == [9, 1]
assert [round(v) for v in TR[1][1]] == [181, 3125] and [round(v, 4) for v in TR[1][2]] == [0.0743, 0.0179] and [round(v, 3) for v in TR[1][3]] == [0.669, 0.447] and [round(v, 3) for v in TR[1][4]] == [8.331, 0.553] and round(TR[29][4][0], 3) == 2.657
EFF_B = [1 / math.sqrt(c) for c in (1, 2, 3)]; assert [round(v, 4) for v in EFF_B] == [1.0, 0.7071, 0.5774] and 1 / math.sqrt(100) == 0.1
EPS = {e: 0.001 / math.sqrt(e + 1e-6) for e in (1e-8, 1e-6, 1e-4)}; assert [round(v, 4) for v in EPS.values()] == [0.995, 0.7071, 0.0995]

def make(lang):
    d = D(lang, KP, N); T = d.T
    g = d.page("KP10 一個學習率顧不了所有維度", "KP10 One learning rate cannot serve every dimension")
    g.beat = 1
    chain(g, 50, 120, 1180, 140, [(BLUE, BLUEL, T("$x$ 方向", "the $x$ direction"), r"$x\leftarrow x(1-\eta)$"), (ORG, ORGL, T("$y$ 方向(陡 25 倍)", "the $y$ direction (25× steeper)"), r"$y\leftarrow y(1-25\eta)$")], ts=22, bs=26, gap=100, beat0=1)
    rows = [[f"{e:g}", f"{1 - e:.2f}", f"{1 - 25 * e:.2f}".replace("-0.00", "0"), (f"$x={GDR[e][0]:.3f}$ , $y\\approx0$" if abs(GDR[e][1]) < 1 else T("發散($y\\to10^{3}$)", "diverges ($y\\to10^{3}$)"))] for e in (0.01, 0.07, 0.09)]
    g.beat = 3
    table(g, 60, 290, [180, 200, 220, 480], [r"$\eta$", T("$x$ 倍率", "$x$ factor"), T("$y$ 倍率", "$y$ factor"), T("30 步後(起點 $(10,2)$)", "after 30 steps (start $(10,2)$)")], rows, rh=60, fs=24, hfs=20, styles=[(REDL, RED), (GOLDL, GOLD), (REDL, RED)], gap=8)
    g.beat = 4
    gold(g, 150, 540, 980, 80, T("要兩個座標都收斂:$\\eta<2/25=0.08$。陡的座標限制上限,平的座標因此走得慢", "both coordinates converge only if $\\eta<2/25=0.08$: the steep one caps the rate, the flat one crawls"), size=22)
    banner(g, T("AdaGrad(Duchi 等人,2011):讓每個座標依看過的梯度各配一個步長", "AdaGrad (Duchi et al., 2011): give each coordinate its own step from the gradients it has seen"))
    d.done(g, "兩難", "The dilemma", {0: ("KP09 的遞減步長是人工訂的排程，對所有座標一視同仁。但各座標的梯度尺度常差很多。玩具碗上 y 的曲率是 x 的 25 倍。", "KP09's decaying step is a hand-made schedule treating every coordinate alike. But gradient scales differ a lot. In the toy bowl y's curvature is 25 times x's."), 1: ("固定學習率下，兩個座標各自乘上固定的係數：x 乘 1 減 eta，y 乘 1 減 25 eta。", "With a fixed rate each coordinate is multiplied by a constant: x by 1 minus eta, y by 1 minus 25 eta."), 2: ("", ""), 3: ("學習率 0.01：x 走得很慢；0.07：x 還是慢；0.09：y 發散。陡的座標限制了學習率的上限，平的座標因此走得慢。", "Rate 0.01: x crawls; 0.07: x is still slow; 0.09: y diverges. The steep coordinate caps the rate, so the flat one crawls."), 4: ("AdaGrad 的想法：讓演算法根據每個座標看過的梯度，各配一個步長。", "AdaGrad's idea: let the algorithm give each coordinate its own step from the gradients it has seen.")})

    g = d.page("更新規則:以歷史梯度平方和縮放", "Scaling by the squared-gradient history")
    g.beat = 1
    box(g, 50, 126, 570, 170, BLUE, BLUEL, T("每個座標一本帳", "a ledger per coordinate"), r"$G_{t,j}=\sum_{s\le t}g_{s,j}^{2}$", ts=22, bs=32)
    g.beat = 2
    box(g, 660, 126, 570, 170, ORG, ORGL, T("更新", "the update"), r"$w_j\leftarrow w_j-\dfrac{\eta}{\sqrt{\varepsilon+G_{t,j}}}\,g_{t,j}$".replace(r"\dfrac", r"\frac"), ts=22, bs=28)
    g.beat = 3
    chain(g, 50, 330, 1180, 140, [(PUR, PURL, T("向量寫法", "vector form"), r"$G\leftarrow G+g\odot g$"), (GRN, GRNL, T("再更新", "then update"), r"$w\leftarrow w-\eta\,g/\sqrt{\varepsilon+G}$"), (RED, REDL, T("代價", "cost"), T("多一個和 $w$ 一樣長的向量\n運算量 $O(d)$", "one extra vector as long as $w$\ncost $O(d)$"))], ts=20, bs=22, gap=50, beat0=3)
    g.beat = 6
    gold(g, 150, 500, 980, 100, T("有效步長 $\\eta/\\sqrt{\\varepsilon+G_{t,j}}$:被推得越用力的座標,之後步子越小心;很少被推的座標,步子大", "effective step $\\eta/\\sqrt{\\varepsilon+G_{t,j}}$: a coordinate pushed hard steps carefully; a rarely pushed one steps boldly"), size=21)
    banner(g, T("沒有人需要事先告訴它哪個方向陡", "nobody has to tell it beforehand which direction is steep"))
    d.done(g, "更新規則", "The update rule", {0: ("為每個座標各記一個累積量，把看過的梯度平方加起來。", "Keep one accumulator per coordinate, summing the squared gradients seen."), 1: ("更新時用步長除以根號下累積量，乘上這一步的梯度。", "The update divides the step by the square root of the accumulator and multiplies by this step's gradient."), 2: ("整個向量一起算，就是逐元素運算。", "Over a whole vector it is an elementwise operation."), 3: ("先把梯度平方加進累積量，", "First add the squared gradient to the accumulator,"), 4: ("再用開根號的累積量縮放步長。", "then scale the step by the square root of the accumulator."), 5: ("代價是多一個和參數一樣長的向量，運算量是參數個數的一次方。", "The cost is one extra vector as long as the parameters, linear work."), 6: ("被推得越用力的座標，之後步子越小心；很少被推的座標，步子就大。", "A coordinate pushed hard steps carefully afterwards; one pushed rarely steps boldly.")})

    g = d.page("在碗上跑兩步", "Two steps on the bowl")
    g.text(50, 108, 1180, 36, T("起點 $(10,2)$,$\\eta=1$,$\\varepsilon=10^{-8}$", "start $(10,2)$, $\\eta=1$, $\\varepsilon=10^{-8}$"), size=22, color=MUTED, bold=True, align="center", z=8)
    g.beat = 1
    rows = []
    for k in (0, 1):
        gg, G, st, mv, pos = TR[k]
        rows.append([str(k + 1), f"$({gg[0]:g},\\,{gg[1]:g})$", f"$({G[0]:.0f},\\,{G[1]:.0f})$", f"$({st[0]:.4f},\\,{st[1]:.4f})$", f"$({mv[0]:.3g},\\,{mv[1]:.3g})$", f"$({pos[0]:.4g},\\,{pos[1]:.3g})$"])
    table(g, 50, 150, [90, 210, 210, 270, 210, 190], [T("步", "step"), T("梯度", "gradient"), T("累積 $G$", "accumulator $G$"), T("有效步長", "effective step"), T("移動量", "move"), T("新位置", "new position")], rows, rh=80, fs=22, hfs=19, styles=[(GRNL, GRN), (CARD, GRAY)], beats=[1, 2], gap=10)
    g.beat = 3
    box(g, 50, 400, 570, 200, PUR, PURL, T("第 1 步:每個座標都走 $\\eta$", "step 1: every coordinate moves $\\eta$"), T("$\\eta g/\\sqrt{\\varepsilon+g^2}\\approx\\eta\\,\\mathrm{sign}(g)$\n梯度是 10 或 50 都走 1", "$\\eta g/\\sqrt{\\varepsilon+g^2}\\approx\\eta\\,\\mathrm{sign}(g)$\ngradient 10 or 50, both move 1"), ts=21, bs=22)
    g.beat = 4
    box(g, 660, 400, 570, 200, GRN, GRNL, T("有效步長相差 5 倍", "effective steps differ 5×"), T("$0.1$ 對 $0.02$:$y$ 的梯度是 $x$ 的 5 倍\n陡的 $y$ 步長小,不會越衝越遠", "$0.1$ versus $0.02$: $y$'s gradient is 5× $x$'s\nthe steep $y$ steps small and does not run away"), ts=21, bs=21)
    banner(g, T("第 2 步有效步長都比第 1 步小:累積只增不減", "step 2's effective steps are both below step 1's: the accumulator only grows"))
    d.done(g, "碗上兩步", "Two steps", {0: ("第 1 步：梯度 10、50，累積 100、2500，有效步長 0.1、0.02，移動量 1、1，到 9、1。", "Step 1: gradient 10, 50, accumulator 100, 2500, effective steps 0.1, 0.02, moves 1, 1, landing at 9, 1."), 1: ("第 2 步：梯度 9、25，累積 181、3125，有效步長 0.0743、0.0179，移動量 0.669、0.447，到 8.331、0.553。", "Step 2: gradient 9, 25, accumulator 181, 3125, effective steps 0.0743, 0.0179, moves 0.669, 0.447, landing at 8.331, 0.553."), 2: ("第一步每個座標都走學習率那麼遠，不管梯度是 10 還是 50，方向由符號決定。", "In the first step every coordinate moves the learning rate, whether the gradient is 10 or 50; the sign sets the direction."), 3: ("", ""), 4: ("但有效步長相差 5 倍，因為 y 的梯度是 x 的 5 倍。陡的 y 座標步長小，不會越衝越遠。", "But the effective steps differ 5 times because y's gradient is 5 times x's. The steep y steps small and does not run away.")})

    g = d.page("稀疏座標得到較大的步長", "Sparse coordinates get larger steps")
    g.beat = 1
    g.text(50, 108, 1180, 36, T("兩個座標,梯度大小在出現時都是 $1$", "two coordinates, each gradient has size $1$ when it appears"), size=22, color=MUTED, bold=True, align="center", z=8)
    box(g, 50, 160, 570, 170, RED, REDL, T("座標 A:每一步都出現", "coordinate A: appears every step"), T("第 100 步累積量 $100$\n有效步長 $0.1$", "after 100 steps $G=100$\neffective step $0.1$"), ts=22, bs=26)
    g.beat = 2
    box(g, 660, 160, 570, 170, GRN, GRNL, T("座標 B:只在第 1、50、100 步出現", "coordinate B: only at steps 1, 50, 100"), T("累積量只有 $1,2,3$\n有效步長 $1,\\ 0.7071,\\ 0.5774$", "$G$ is only $1,2,3$\neffective steps $1,\\ 0.7071,\\ 0.5774$"), ts=20, bs=26)
    g.beat = 3
    xs = [1, 50, 100]
    g.text(50, 360, 1180, 36, T("B 的有效步長", "B's effective step"), size=20, color=GRN, bold=True, z=8)
    for i, (k, v) in enumerate(zip(xs, EFF_B)):
        x = 120 + i * 350
        g.rect(x, 560 - v * 150, 150, v * 150, fill=GRNL, stroke=GRN, sw=3, radius=6, z=3); g.text(x - 20, 560 - v * 150 - 40, 190, 36, f"${v:.4f}$", size=22, color=GRN, bold=True, align="center", z=8); g.text(x - 20, 564, 190, 34, T(f"第 {k} 步", f"step {k}"), size=20, color=INK, bold=True, align="center", z=8)
    g.line([(80, 560), (1180, 560)], INK, sw=3, z=3)
    g.beat = 4
    gold(g, 700, 400, 500, 130, T("罕見的座標每次出現都能走大步,常出現的座標被壓小", "a rare coordinate takes big steps each time it appears; a frequent one is damped"), size=22)
    banner(g, T("這就是 AdaGrad 常用在稀疏特徵的原因", "this is why AdaGrad is popular with sparse features"))
    d.done(g, "稀疏座標", "Sparse coordinates", {0: ("兩個座標，梯度大小出現時都是 1。座標 A 每一步都出現，到第 100 步累積量是 100，有效步長 0.1。", "Two coordinates whose gradient has size 1 when present. Coordinate A appears every step; at step 100 its accumulator is 100 and its effective step 0.1."), 1: ("座標 B 只在第 1、50、100 步出現，累積量只有 1、2、3。", "Coordinate B appears only at steps 1, 50 and 100, so its accumulator is only 1, 2, 3."), 2: ("B 的有效步長是 1、0.7071、0.5774，比 A 的 0.1 大得多。", "B's effective steps are 1, 0.7071, 0.5774, much larger than A's 0.1."), 3: ("罕見的座標每次出現都能走大步，常出現的座標被壓小。這就是 AdaGrad 常用在稀疏特徵的原因。", "A rare coordinate takes big steps each time it appears and a frequent one is damped. That is why AdaGrad is popular with sparse features."), 4: ("", "")})

    g = d.page("$\\varepsilon$ 不是裝飾", "$\\varepsilon$ is not decoration")
    g.beat = 1
    box(g, 50, 126, 570, 150, BLUE, BLUEL, T("作用 1", "role 1"), T("避免 $G=0$ 時 $0/0$:\n某座標還沒出現過梯度,$\\varepsilon=0$ 會得到 NaN", "avoid $0/0$ when $G=0$:\na coordinate with no gradient yet gives NaN if $\\varepsilon=0$"), ts=22, bs=20)
    g.beat = 2
    box(g, 660, 126, 570, 150, ORG, ORGL, T("作用 2", "role 2"), T("決定微小梯度的第一步有多大", "sets how big the first step of a tiny gradient is"), ts=22, bs=22)
    rows = [[f"$10^{{{round(math.log10(e))}}}$", f"${v:.4f}$"] for e, v in EPS.items()]
    g.beat = 3
    g.text(50, 296, 1180, 36, T("第一次出現的梯度 $g=0.001$,$\\eta=1$:第一步移動", "a first gradient $g=0.001$ with $\\eta=1$: the first move"), size=22, color=MUTED, bold=True, align="center", z=8)
    table(g, 190, 340, [320, 580], [r"$\varepsilon$", T("第一步移動量", "first-step move")], rows, rh=60, fs=28, hfs=20, styles=[(GRNL, GRN), (GOLDL, GOLD), (REDL, RED)], gap=8)
    banner(g, T("$\\varepsilon$ 太大會把微小梯度的步子壓得很小", "a large $\\varepsilon$ squashes the step of a tiny gradient"))
    d.done(g, "epsilon 的作用", "What epsilon does", {0: ("epsilon 不是裝飾。第一個作用是避免累積量為零時零除以零：某座標還沒出現過梯度，沒有 epsilon 會得到 NaN。", "Epsilon is not decoration. First it avoids 0 over 0 when the accumulator is zero: a coordinate with no gradient yet would give NaN without it."), 1: ("第二個作用是決定微小梯度的第一步有多大。", "Second it sets how big the first step of a tiny gradient is."), 2: ("第一次出現的梯度是 0.001，學習率 1：epsilon 是 10 的負 8 次方時移動 0.9950；10 的負 6 次方時 0.7071；10 的負 4 次方時只有 0.0995。", "A first gradient of 0.001 with learning rate 1: epsilon 10 to the minus 8 moves 0.9950; 10 to the minus 6 moves 0.7071; 10 to the minus 4 moves only 0.0995."), 3: ("", "")})

    g = d.page("累積只增不減:有效步長只降不升", "Accumulation only grows: the effective step only falls")
    g.beat = 1
    chain(g, 50, 120, 1180, 150, [(BLUE, BLUEL, T("梯度平方非負", "squared gradients are nonneg."), T("$G_{t,j}$ 對 $t$ 單調不減", "$G_{t,j}$ is nondecreasing in $t$")), (ORG, ORGL, T("所以", "so"), T("有效步長只降不升", "the effective step only falls")), (RED, REDL, T("梯度大小恆為 $g$", "constant gradient size $g$"), r"$\dfrac{\eta}{|g|\sqrt{k}}$".replace(r"\dfrac", r"\frac") + T("\n以 $1/\\sqrt k$ 下降", "\nfalls like $1/\\sqrt k$"))], ts=20, bs=24, gap=50, beat0=1)
    g.beat = 4
    box(g, 50, 300, 570, 150, RED, REDL, T("缺點", "drawbacks"), T("早期的大梯度永遠留在累積量裡\n後期步長可能太小,走不動", "early large gradients stay in the accumulator forever\nlate steps can become too small to move"), ts=22, bs=21)
    g.beat = 5
    box(g, 660, 300, 570, 150, PUR, PURL, T("修正的方向", "the direction of the fix"), T("讓累積量能遺忘舊梯度\n讓有效步長有機會回升(KP11)", "let the accumulator forget old gradients\nso the effective step can recover (KP11)"), ts=22, bs=21)
    g.beat = 6
    gold(g, 150, 490, 980, 120, T("玩具碗:$\\eta=0.07$ 的梯度下降 30 步後 $x=1.134$;$\\eta=1$ 的 AdaGrad 是 $2.657$。AdaGrad 的好處是不必知道各座標的尺度,不是任何情況都更快", "toy bowl: GD with $\\eta=0.07$ reaches $x=1.134$ after 30 steps; AdaGrad with $\\eta=1$ reaches $2.657$. Its benefit is not needing the coordinate scales, not being faster everywhere"), size=20)
    banner(g, T("記帳本只會越寫越厚,步子只會越來越小", "the ledger only grows thicker, so the steps only get smaller"))
    d.done(g, "只增不減", "Only grows", {0: ("平方是非負的，所以累積量對步數單調不減。", "Squares are nonnegative, so the accumulator is nondecreasing in the step count."), 1: ("有效步長因此只降不升。", "So the effective step only falls."), 2: ("若某座標梯度大小一直是 g，有效步長是學習率除以 g 乘根號 k，以根號 k 分之一下降。", "If a coordinate's gradient size stays g, the effective step is the rate over g times root k, falling like one over root k."), 3: ("", ""), 4: ("而且早期的大梯度永遠留在累積量裡，訓練後期步長可能太小，走不動。", "Early large gradients stay in the accumulator forever, so late steps can become too small to move."), 5: ("修正的方向是讓累積量能遺忘舊的梯度，讓有效步長有機會回升，這是 KP11 的問題。", "The fix is to let the accumulator forget old gradients so the effective step can recover: KP11's question."), 6: ("AdaGrad 不一定比 SGD 好：玩具碗上學習率 0.07 的梯度下降 30 步後 x 是 1.134，AdaGrad 是 2.657。它的好處是不必知道各座標的尺度。", "AdaGrad is not always better than SGD: on the toy bowl gradient descent with 0.07 reaches x of 1.134 after 30 steps, AdaGrad 2.657. Its benefit is not needing the coordinate scales.")})
    closing(d, [(RED, REDL, "「AdaGrad 一定比 SGD 好」", "“AdaGrad is always better than SGD”", "不一定:好處是不必知道各座標的尺度", "not necessarily: its benefit is not needing the scales"), (ORG, ORGL, "「有效步長會自己調回來」", "“the effective step recovers by itself”", "不會:分母單調不減,只降不升", "no: the denominator never decreases"), (PUR, PURL, "「第一步的移動量和梯度大小成正比」", "“the first move is proportional to the gradient”", "不是:每個座標都約等於 $\\eta$(符號 $\\times\\eta$)", "no: every coordinate moves about $\\eta$ (sign $\\times\\eta$)")],
            ("怎麼讓累積量能遺忘舊的梯度,讓有效步長有機會回升?", "How can the accumulator forget old gradients so the effective step can recover?"), ("下一個單元:RMSProp 與 Adam", "next unit: RMSProp and Adam"))
    return d.pages
