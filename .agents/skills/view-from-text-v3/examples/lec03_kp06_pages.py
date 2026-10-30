import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lec03_kit import *

KP = "KP06"; N = 7
X, W1, W2, Y = 1, 2, 3, 0
A = W1 * X; Z = A; YH = W2 * Z; E = 0.5 * (YH - Y) ** 2; DO = YH - Y; G2 = DO * Z; DA = DO * W2 * 1; G1 = DA * X
assert (A, Z, YH, E, DO, G2, DA, G1) == (2, 2, 6, 18, 6, 12, 18, 18)
ef = lambda w1: 0.5 * (W2 * w1 * X - Y) ** 2; EPS = 0.001
EP, EM = ef(W1 + EPS), ef(W1 - EPS); CD = (EP - EM) / (2 * EPS); assert round(EP, 7) == 18.0180045 and round(EM, 7) == 17.9820045 and round(CD, 6) == 18
sg = lambda a: 1 / (1 + math.exp(-a)); sgp = lambda a: sg(a) * (1 - sg(a))
SA = 0.5; SZ = sg(SA); SYH = 2 * SZ; SE = 0.5 * (SYH - 1) ** 2; SDO = SYH - 1; SDA = SDO * 2 * sgp(SA)
assert [round(v, 4) for v in (SZ, SYH, SDO, SDA)] == [0.6225, 1.2449, 0.2449, 0.1151] and round(SE, 4) == 0.03 and round(sgp(5), 4) == 0.0066 and sgp(0) == 0.25

def make(lang):
    d = D(lang, KP, N); T = d.T
    g = d.page("KP06 隱藏層看得到損失，卻看不到標籤", "KP06 A hidden layer sees the loss, not the labels")
    g.beat = 1
    box(g, 50, 126, 560, 210, RED, REDL, T("逐一擾動每個參數", "perturb every parameter"), T("每擾動一個就重跑一次前向\n成本 $O(W)\\times W=O(W^2)$\n$W=10^6$:$10^{12}$ 次運算", "each perturbation reruns a forward pass\ncost $O(W)\\times W=O(W^2)$\n$W=10^6$: $10^{12}$ operations"), ts=22, bs=22)
    g.beat = 2
    box(g, 670, 126, 560, 210, GRN, GRNL, T("反向傳播(鏈式法則)", "backpropagation (chain rule)"), T("一次前向(保存每個 $a,z$)\n加一次反向(由後往前)\n全部梯度精確,成本 $O(W)$", "one forward pass (keep every $a,z$)\nplus one backward pass\nall gradients exact, cost $O(W)$"), ts=22, bs=22)
    g.beat = 3
    gold(g, 190, 390, 900, 120, T("隱藏層的參數只透過下游一連串運算影響損失:每個參數該負多少責任?", "a hidden parameter reaches the loss only through downstream operations: how much is it responsible for?"), size=24)
    banner(g, T("1986 年 Rumelhart、Hinton 與 Williams 系統化並普及(更早還有別人的研究)", "popularized by Rumelhart, Hinton and Williams in 1986 (earlier work too)"))
    d.done(g, "為什麼需要反向傳播", "Why backpropagation", {0: ("單層模型的梯度一步就推得出來。多層網路不同：隱藏層從不直接接觸標籤與損失。", "A single-layer model's gradient follows in one step. In a multilayer network a hidden layer never touches the labels or the loss directly."), 1: ("最直接的辦法是逐一擾動，每個參數都要重跑一次前向，總成本是參數個數的平方。一百萬個參數就是十的十二次方次運算，不可行。", "The direct way perturbs each parameter, rerunning the forward pass each time: cost grows with the square of the parameter count. A million parameters means 10 to the 12 operations, infeasible."), 2: ("反向傳播用鏈式法則：一次前向加一次反向，得到全部參數的精確梯度，成本是參數個數的一次方。", "Backpropagation uses the chain rule: one forward and one backward pass give every parameter's exact gradient, at cost linear in the parameter count."), 3: ("隱藏層的參數只透過下游的運算影響損失，問題是每個參數該負多少責任。", "A hidden parameter affects the loss only through downstream operations; the question is how much responsibility each carries.")})

    g = d.page("單一神經元:梯度 = $\\delta\\times z$", "One neuron: gradient = $\\delta\\times z$")
    g.beat = 1
    chain(g, 50, 130, 1180, 170, [(BLUE, BLUEL, T("前向兩步", "two forward steps"), r"$a_i=\sum_\ell w_{i\ell}z_\ell,\quad z_i=\sigma(a_i)$"), (ORG, ORGL, T("鏈式法則", "chain rule"), r"$\dfrac{\partial e}{\partial w_{i\ell}}=\dfrac{\partial e}{\partial a_i}\cdot\dfrac{\partial a_i}{\partial w_{i\ell}}$".replace(r"\dfrac", r"\frac"))], ts=22, bs=26, gap=60, beat0=1)
    g.beat = 3
    box(g, 50, 330, 570, 150, PUR, PURL, T("誤差敏感度", "error sensitivity"), r"$\delta_i:=\dfrac{\partial e}{\partial a_i}$".replace(r"\dfrac", r"\frac") + T("\n$a_i$ 動一點,損失動多少", "\nhow much the loss moves when $a_i$ moves"), ts=22, bs=26)
    box(g, 660, 330, 570, 150, GRN, GRNL, T("乾淨的因子", "the clean factor"), r"$\dfrac{\partial a_i}{\partial w_{i\ell}}=z_\ell$".replace(r"\dfrac", r"\frac") + T("\n流進這條連線的前層訊號", "\nthe signal flowing into this connection"), ts=22, bs=26)
    g.beat = 4
    gold(g, 190, 510, 900, 100, T("權重的梯度 = 這個神經元的 $\\delta_i$ × 流進來的訊號 $z_\\ell$;所以前向要把每層的 $z$ 存起來", "weight gradient = $\\delta_i\\times z_\\ell$; that is why the forward pass stores every $z$"), size=22)
    banner(g, T("這句話無論在輸出層或第十個隱藏層都成立", "true in the output layer and in the tenth hidden layer alike"))
    d.done(g, "梯度等於 delta 乘 z", "Gradient = delta times z", {0: ("先看一個神經元的前向：先線性加總，再過活化函數。", "First one neuron's forward pass: a linear sum, then the activation."), 1: ("損失對權重的偏導，用鏈式法則先經過活化前的加權和。", "The loss's partial derivative with respect to a weight goes through the pre-activation sum by the chain rule."), 2: ("", ""), 3: ("第一個因子叫誤差敏感度 delta：活化前加權和動一點，損失動多少。第二個因子很乾淨，就是流進這條連線的前層訊號 z。", "The first factor is the error sensitivity delta: how much the loss moves when the pre-activation sum moves. The second is clean: the signal z flowing into this connection."), 4: ("所以權重的梯度等於 delta 乘上流進來的訊號。這也是為什麼前向要把每一層的 z 存起來。", "So a weight's gradient is delta times the incoming signal. That is why the forward pass stores every layer's z.")})

    g = d.page("手算:前向,再反向", "By hand: forward, then backward")
    g.beat = 0
    nodes = []
    for i, (nm, v, col, light) in enumerate((("x", "1", BLUE, BLUEL), ("a", "2", ORG, ORGL), ("z", "2", ORG, ORGL), (r"\hat y", "6", GRN, GRNL))):
        x = 70 + i * 310
        nodes.append(g.mark()); g.ellipse(x + 60, 210, 58, fill=light, stroke=col, sw=3, z=3); g.text(x + 2, 168, 116, 84, f"${nm}$", size=28, color=col, bold=True, align="center", vcenter=True, z=8)
    g.beat = 1
    for i in range(4):
        g.text(70 + i * 310 + 2, 282, 116, 44, f"$={[1, 2, 2, 6][i]}$", size=26, color=INK, bold=True, align="center", z=8)
    for i, (w, t) in enumerate((("w_1=2", ""), ("", ""), ("w_2=3", ""))):
        if w: g.link(nodes[i], nodes[i + 1], 4, 12, GRAY, 4); g.text(70 + i * 310 + 120, 150, 190, 40, f"${w}$", size=22, color=MUTED, bold=True, align="center", z=8)
    g.link(nodes[1], nodes[2], 4, 12, GRAY, 4)
    g.text(50, 340, 1180, 40, T("前向(由前往後並保存):$a=2$,$z=2$,$\\hat y=6$,$e=\\frac12(6-0)^2=18$", "forward (keep the values): $a=2$, $z=2$, $\\hat y=6$, $e=\\frac12(6-0)^2=18$"), size=22, color=BLUE, bold=True, align="center", z=8)
    g.beat = 2
    box(g, 50, 410, 370, 130, ORG, ORGL, T("輸出端敏感度", "output sensitivity"), r"$\delta_{\mathrm{out}}=\hat y-y=6$", ts=20, bs=26)
    g.beat = 3
    box(g, 455, 410, 370, 130, GRN, GRNL, T("$w_2$ 的梯度", "gradient of $w_2$"), r"$\delta_{\mathrm{out}}\cdot z=6\cdot2=12$", ts=20, bs=26)
    g.beat = 4
    box(g, 860, 410, 370, 130, PUR, PURL, T("$w_1$ 的梯度", "gradient of $w_1$"), r"$\delta_a=6\cdot3\cdot1=18$" + "\n" + r"$18\cdot x=18$", ts=20, bs=24)
    banner(g, T("每個因子都有角色:$6$ 輸出差多少、$3$ $z$ 透過 $w_2$ 的強度、$1$ 活化斜率、$x$ 流進 $w_1$ 的訊號", "each factor has a role: output error, $w_2$, activation slope, input signal"))
    d.done(g, "完整手算", "A complete hand computation", {0: ("最小例子：輸入 1，兩個權重 2 與 3，答案 0，活化是恆等，損失是平方差的一半。", "The smallest example: input 1, weights 2 and 3, answer 0, identity activation, half squared error."), 1: ("前向由前往後並保存數值：a 等於 2，z 等於 2，輸出 6，損失 18。", "Forward, keeping the values: a is 2, z is 2, the output 6, the loss 18."), 2: ("反向由後往前。輸出端的敏感度是輸出減答案，等於 6。", "Backward from the end. The output sensitivity is the output minus the answer, 6."), 3: ("w2 的梯度是 6 乘 z，等於 12。", "The gradient of w2 is 6 times z, equal to 12."), 4: ("隱藏單元的敏感度是 6 乘 w2 乘活化斜率 1，等於 18；w1 的梯度是 18 乘 x，等於 18。", "The hidden sensitivity is 6 times w2 times the activation slope 1, equal to 18; the gradient of w1 is 18 times x, equal to 18.")})

    g = d.page("怎麼確定算對了:梯度檢查", "How do we know it is right: gradient checking")
    g.beat = 1
    g.rect(190, 118, 900, 100, fill=CARD, stroke=GRAY, sw=3, radius=16, z=3); g.text(200, 118, 880, 100, r"$\dfrac{\partial e}{\partial w}\approx\dfrac{e(w+\epsilon)-e(w-\epsilon)}{2\epsilon}$".replace(r"\dfrac", r"\frac"), size=38, color=INK, align="center", vcenter=True, z=8)
    g.beat = 2
    chain(g, 50, 260, 1180, 170, [(BLUE, BLUEL, r"$e(w_1+\epsilon)$", f"${EP:.7f}$"), (ORG, ORGL, r"$e(w_1-\epsilon)$", f"${EM:.7f}$"), (GRN, GRNL, T("差除以 $2\\epsilon$", "difference over $2\\epsilon$"), r"$\frac{18.0180045-17.9820045}{0.002}=18$")], ts=22, bs=22, gap=60, beat0=2)
    g.beat = 5
    gold(g, 190, 460, 900, 70, T("與手算的 $18$ 相符($\\epsilon=0.001$)", "matches the hand result $18$ ($\\epsilon=0.001$)"), size=26)
    g.beat = 6
    gold(g, 190, 548, 900, 70, T("中心差分誤差 $O(\\epsilon^2)$;前向差分 $O(\\epsilon)$", "central difference error $O(\\epsilon^2)$; forward difference $O(\\epsilon)$"), size=22)
    banner(g, T("再比較手算、autograd、有限差分三者;autograd 是獨立證人,不取代理解", "compare hand, autograd, finite differences; autograd does not replace understanding"))
    d.done(g, "梯度檢查", "Gradient checking", {0: ("反向傳播的公式很容易寫錯，所以要獨立檢查：用中心差分估計導數，再和反向傳播比較。", "Backpropagation formulas are easy to get wrong, so check independently: estimate the derivative by central differences and compare."), 1: ("w1 加一點點，損失 18.0180045。", "With w1 nudged up, the loss is 18.0180045."), 2: ("w1 減一點點，損失 17.9820045。", "With w1 nudged down, the loss is 17.9820045."), 3: ("", ""), 4: ("差除以兩倍的擾動，得到 18。", "The difference over twice the nudge gives 18."), 5: ("與手算的 18 相符。", "It matches the hand result of 18."), 6: ("用中心差分是因為偶數階項互相抵消，誤差是 epsilon 平方量級，前向差分是 epsilon 量級。自動微分是另一個獨立的證人，實務上比較手算梯度、autograd 與有限差分三者。", "Central differences are used because even-order terms cancel, giving error of order epsilon squared against order epsilon for forward differences. Automatic differentiation is another independent witness; in practice compare hand gradients, autograd and finite differences.")})

    g = d.page("隱藏層的 $\\delta$:下游責任倒灌回來", "Hidden-layer $\\delta$: downstream responsibility flows back")
    g.beat = 1
    g.rect(120, 118, 1040, 110, fill=CARD, stroke=GRAY, sw=3, radius=16, z=3); g.text(130, 118, 1020, 110, r"$\delta_i=\sigma'(a_i)\sum_j\delta_j\,w_{ji}$", size=42, color=INK, align="center", vcenter=True, z=8)
    g.beat = 2
    ids = chain(g, 50, 260, 1180, 160, [(BLUE, BLUEL, T("下游責任 $\\delta_j$", "downstream $\\delta_j$"), T("沿連線以 $w_{ji}$ 加權倒灌", "weighted by $w_{ji}$, flows back")), (ORG, ORGL, T("加總", "sum"), r"$\sum_j\delta_jw_{ji}$"), (GRN, GRNL, T("乘局部斜率", "times the local slope"), r"$\sigma'(a_i)$")], ts=22, bs=26, gap=60, beat0=2)
    g.beat = 5
    rows = [[T("$a_i=5$:$\\sigma'=0.0066$", "$a_i=5$: $\\sigma'=0.0066$"), r"$\delta_i=2\cdot0.5\cdot0.0066\approx0.0066$"], [T("$a_i=0$:$\\sigma'=0.25$", "$a_i=0$: $\\sigma'=0.25$"), r"$\delta_i=2\cdot0.5\cdot0.25=0.25$"]]
    table(g, 60, 440, [420, 720], [T("飽和與否", "saturated or not"), r"$\delta_i$($w_{ji}=2,\ \delta_j=0.5$)"], rows, rh=54, fs=22, hfs=18, styles=[(REDL, RED), (GRNL, GRN)], gap=6)
    banner(g, T("飽和使 $\\sigma'\\approx0$:前面的層幾乎收不到訊號(梯度消失)", "saturation makes $\\sigma'\\approx0$: earlier layers hardly receive a signal (vanishing gradient)"))
    d.done(g, "隱藏層的 delta", "Hidden-layer delta", {0: ("隱藏層神經元的敏感度，是下游每個神經元的敏感度乘上連線權重加總，再乘自己的局部斜率。", "A hidden neuron's sensitivity is the sum of each downstream sensitivity times its connection weight, then times its own local slope."), 1: ("下游的責任沿著連線，以權重加權倒灌回來。", "Downstream responsibility flows back along the connections, weighted by the weights."), 2: ("加總之後，", "Summed,"), 3: ("再經過這個神經元自己的局部斜率過濾。", "then filtered by this neuron's own local slope."), 4: ("", ""), 5: ("用 sigmoid 比較：活化前加權和是 5 時斜率只有 0.0066，delta 幾乎消失；在 0 時斜率 0.25，delta 是 0.25。飽和使前面的層幾乎收不到訊號，這就是梯度消失。", "With a sigmoid: at a pre-activation of 5 the slope is only 0.0066 and the delta nearly vanishes; at 0 the slope is 0.25 and delta is 0.25. Saturation leaves earlier layers almost no signal: the vanishing gradient.")})
    closing(d, [(RED, REDL, "「反向傳播就是梯度下降」", "“backpropagation is gradient descent”", "不是,它只算梯度;更新是最佳化器的事", "no: it only computes gradients; updating is the optimizer's job"), (ORG, ORGL, "「隱藏層能直接看到標籤」", "“a hidden layer sees the labels directly”", "不能,它透過下游的 $\\delta$ 間接得知責任", "no: it learns its responsibility through downstream $\\delta$"), (PUR, PURL, "「autograd 通過就不用懂原理」", "“if autograd passes I need not understand”", "它是獨立證人,不取代理解", "it is an independent witness, not a substitute for understanding")],
            ("資料有上百萬筆時,連算一次完整的梯度都很貴。怎麼辦?", "With millions of samples even one full gradient is costly. What then?"), ("下一個單元:隨機梯度下降", "next unit: stochastic gradient descent"))
    return d.pages
