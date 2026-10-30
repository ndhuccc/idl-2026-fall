import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lec03_kit import *

KP = "KP01"; N = 7
f = lambda x, y: 0.5 * (x * x + 25 * y * y)
grad = lambda x, y: (x, 25 * y)
TH0 = (10, 2); G0 = grad(*TH0); assert G0 == (10, 50) and f(*TH0) == 100
TH1 = (TH0[0] - 0.06 * G0[0], TH0[1] - 0.06 * G0[1]); assert [round(v, 6) for v in TH1] == [9.4, -1.0] and round(f(*TH1), 2) == 56.68
H = (1, 25); NEWT = (TH0[0] - G0[0] / H[0], TH0[1] - G0[1] / H[1]); assert NEWT == (0, 0)
PATH = [TH0]
for _ in range(30): x, y = PATH[-1]; PATH.append((x * 0.94, y * -0.5))
assert [round(f(*PATH[k]), 1) for k in (10, 30)] == [14.5, 1.2]
ROWS = [(d, 4 * d, 4 * d * d, d ** 3, d ** 3 / 1e13) for d in (10 ** 2, 10 ** 4, 10 ** 6)]
assert ROWS[0][2] == 40000 and ROWS[1][2] == 4e8 and ROWS[2][2] == 4e12 and round(ROWS[2][4] / 3600, 1) == 27.8
def hum(b):
    for u, s in ((1e12, "TB"), (1e9, "GB"), (1e6, "MB"), (1e3, "KB"), (1, "B")):
        if b >= u: return f"{b / u:g} {s}"
def hum_t(t):
    if t < 1e-3: return f"{t * 1e6:g} µs"
    if t < 100: return f"{t:g} s"
    return f"$10^{{{round(math.log10(t))}}}$ s ≈ {t / 3600:.1f} h"

def make(lang):
    d = D(lang, KP, N); T = d.T
    # 1 hiker
    g = d.page("KP01 訓練就是在山上找最低的谷", "KP01 Training is looking for the lowest valley")
    g.beat = 1
    ids = chain(g, 50, 150, 1180, 300, [(BLUE, BLUEL, T("山", "the mountain"), T("每一組參數\n對應一個損失", "every set of parameters\nhas a loss")), (ORG, ORGL, T("坡度", "the slope"), T("損失在這一點\n的變化方向\n(梯度)", "the direction the loss\nchanges at this point\n(the gradient)")), (GRN, GRNL, T("走路的人", "the hiker"), T("最佳化器:\n決定下一步\n往哪裡走", "the optimizer:\ndecides where to\nstep next"))], ts=28, bs=24, beat0=1)
    g.beat = 4
    gold(g, 190, 490, 900, 100, T("蒙住眼睛，只能用腳感覺此刻哪個方向最陡", "blindfolded, you only feel which direction is steepest right now"), size=26)
    banner(g, T("訓練 = 在參數空間裡，沿著坡度走向低處", "training = walking downhill through parameter space"))
    d.done(g, "山、坡度、走路的人", "Mountain, slope, hiker", {0: ("想像被蒙住眼睛放在一座山上，要走到最低的山谷。訓練網路就是這件事。", "Imagine being blindfolded on a mountain and having to reach the lowest valley. Training a network is exactly that."), 1: ("山是每一組參數對應的損失。", "The mountain is the loss of every set of parameters."), 2: ("坡度是損失在這一點的變化方向，也就是梯度。", "The slope is the direction the loss changes at this point, the gradient."), 3: ("走路的人就是最佳化器。", "The hiker is the optimizer."), 4: ("你看不到整座山，只能用腳感覺此刻哪個方向最陡。", "You cannot see the whole mountain; you only feel which direction is steepest right now.")})

    # 2 three roles
    g = d.page("三個角色，一行目標", "Three roles and one line of goal")
    g.beat = 1
    chain(g, 50, 130, 1180, 200, [(BLUE, BLUEL, T("假設類別", "hypothesis class"), T("模型能畫出哪些函數\n= 山的形狀", "which functions the model can draw\n= the mountain's shape")), (ORG, ORGL, T("損失", "loss"), T("把差距換成分數\n= 每個點的高度", "turns the gap into a score\n= the height of each point")), (GRN, GRNL, T("最佳化器", "optimizer"), T("看著分數，決定\n參數下一步往哪調", "reads the score and decides\nwhere to adjust next"))], ts=26, bs=22, beat0=1)
    g.beat = 4
    g.rect(190, 370, 900, 100, fill=CARD, stroke=GRAY, sw=3, radius=16, z=3); g.text(200, 370, 880, 100, r"$\min_{\theta}\ \mathcal{L}(\theta)$", size=48, color=INK, align="center", vcenter=True, z=8)
    g.beat = 5
    gold(g, 190, 500, 900, 110, T("這一行只說「要找什麼」，沒說「怎麼找」。損失只排名次，不指方向。", "This line says what to find, not how. A loss only ranks; it gives no direction."), size=24)
    banner(g, T("怎麼走，是最佳化器的工作，也是整個訓練真正的引擎", "how to move is the optimizer's job: the real engine of training"))
    d.done(g, "三個角色", "Three roles", {0: ("機器學習把訓練拆成三個角色。", "Machine learning splits training into three roles."), 1: ("假設類別決定模型能畫出哪些函數，也就是山可能的形狀。", "The hypothesis class decides which functions the model can draw, the shape the mountain may have."), 2: ("損失把預測與答案的差距換成分數，每一組參數都有一個。", "The loss turns the gap between prediction and answer into a score, one for every set of parameters."), 3: ("最佳化器看著分數，決定參數下一步往哪裡調。", "The optimizer reads the score and decides where to adjust the parameters next."), 4: ("目標寫成一行：在所有參數裡找讓損失最小的。", "The goal in one line: among all parameters, find the one with the smallest loss."), 5: ("這一行只說要找什麼，沒說怎麼找。損失能說這組比那組好，卻不會告訴你下一步往哪走。", "This line says what to find, not how. A loss can say one set is better than another, but not which way to step next.")})

    # 3 one step
    g = d.page("最佳化器的一步，重複到停止", "One optimizer step, repeated until stopping")
    st = [(BLUE, BLUEL, T("1 預測", "1 predict"), r"$\theta_t$ " + T("與資料", "and data")), (ORG, ORGL, T("2 算損失", "2 loss"), r"$\mathcal{L}(\theta_t)$"), (PUR, PURL, T("3 求梯度", "3 gradient"), r"$\nabla\mathcal{L}(\theta_t)$"), (GRN, GRNL, T("4 更新", "4 update"), r"$\theta_{t+1}$")]
    ids = chain(g, 50, 170, 1180, 200, st, ts=26, bs=32, beat0=1)
    g.beat = 5
    g.link(ids[3], ids[0], 8, 8, RED, 5)
    g.text(300, 440, 680, 50, T("回到第 1 步，重複到停止(何時停:KP03)", "back to step 1, repeat until stopping (when to stop: KP03)"), size=22, color=RED, bold=True, align="center", z=8)
    g.beat = 6
    gold(g, 190, 520, 900, 90, T("梯度 = 損失上升最快的方向;負梯度就是下坡", "the gradient is the direction of fastest increase; the negative gradient is downhill"), size=24)
    banner(g, T("四步一圈:預測、算損失、求梯度、更新", "four steps a turn: predict, loss, gradient, update"))
    d.done(g, "一步四個動作", "Four actions per step", {0: ("接下來看最佳化器每一步做什麼。", "Now let us see what the optimizer does at each step."), 1: ("第一步，用目前的參數和資料做預測。", "Step one: predict with the current parameters and data."), 2: ("第二步，和標準答案比，算出損失。", "Step two: compare with the answers and compute the loss."), 3: ("第三步，求損失在這一點的梯度，也就是上升最快的方向。", "Step three: take the gradient of the loss at this point, the direction of fastest increase."), 4: ("第四步，選出下一組參數。", "Step four: choose the next parameters."), 5: ("然後回到第一步，重複到停止為止。何時停止是 KP03 的主題。", "Then return to step one and repeat until stopping. When to stop is the topic of KP03."), 6: ("負梯度就是下坡的方向。", "The negative gradient is the downhill direction.")})

    # 4 toy bowl one step
    g = d.page("玩具碗:走一步", "The toy bowl: one step")
    xr, yr = (-12, 12), (-4, 4)
    ux, uy = axes(g, 50, 120, 760, 480, xr, yr)
    g.beat = 1
    bowl_contours(g, ux, uy, levels=(4, 16, 36, 64))
    dot(g, ux(10), uy(2), 12, RED); lab(g, ux(10) - 150, uy(2) - 46, 160, "$\\theta_0=(10,2)$", 20, RED)
    g.beat = 2
    g.line([(ux(10), uy(2)), (ux(10 - 0.2 * 10), uy(2 - 0.2 * 50 / 25 * 0.5))], GRAY, sw=1, z=1) if False else None
    g.line([(ux(TH0[0]), uy(TH0[1])), (ux(TH1[0]), uy(TH1[1]))], GRN, sw=6, arrow="end", head=20, z=6)
    dot(g, ux(TH1[0]), uy(TH1[1]), 12, GRN); lab(g, ux(TH1[0]) - 170, uy(TH1[1]) + 14, 190, "$\\theta_1=(9.4,-1)$", 20, GRN)
    g.text(50, 604, 760, 34, T("等高線:$f=\\frac12(x^2+25y^2)$,y 方向陡得多", "contours of $f=\\frac12(x^2+25y^2)$: much steeper in $y$"), size=18, color=MUTED, bold=True, align="center", z=8)
    g.beat = 1
    box(g, 850, 126, 380, 150, BLUE, BLUEL, T("起點", "start"), r"$f(10,2)=100$" + "\n" + r"$\nabla f=(10,\,50)$", ts=22, bs=24)
    g.beat = 2
    box(g, 850, 290, 380, 150, GRN, GRNL, T("走一步 $\\eta=0.06$", "one step, $\\eta=0.06$"), r"$\theta_1=(9.4,\,-1.0)$" + "\n" + r"$f=56.68<100$", ts=22, bs=24)
    g.beat = 3
    gold(g, 850, 470, 380, 130, T("負梯度幾乎指向 $y$ 方向:碗在 $y$ 陡得多", "the negative gradient points almost along $y$: the bowl is steeper there"), size=21 if zh else 18) if False else gold(g, 850, 470, 380, 130, T("負梯度幾乎指向 $y$ 方向:碗在 $y$ 陡得多", "the negative gradient points almost along $y$: steeper there"), size=21)
    banner(g, T("損失由 $100$ 降到 $56.68$:確實往下走了", "the loss fell from $100$ to $56.68$: a real step downhill"))
    d.done(g, "走一步", "One step", {0: ("用一個只有兩個參數的碗來看這四步。起點 10、2，損失是 100，梯度是 10、50，幾乎指向 y 方向，因為碗在 y 方向陡得多。", "Use a two-parameter bowl to see the four steps. The start is 10, 2 with loss 100 and gradient 10, 50, almost along y because the bowl is much steeper there."), 1: ("取學習率 0.06 走一步，到 9.4、負 1.0。", "A step with learning rate 0.06 lands at 9.4, minus 1.0."), 2: ("新位置的損失是 56.68，比 100 小，確實往下走了。", "The new loss is 56.68, below 100: a real step downhill."), 3: ("負梯度幾乎指向 y 方向，因為這個碗在 y 方向陡得多。", "The negative gradient points almost along y because the bowl is much steeper there.")})

    # 5 newton vs gradient
    g = d.page("多用曲率:牛頓法一步到底", "Using curvature: Newton lands in one step")
    ux, uy = axes(g, 50, 120, 760, 480, xr, yr)
    g.beat = 1
    bowl_contours(g, ux, uy, levels=(4, 16, 36, 64))
    g.line([(ux(x), uy(y)) for x, y in PATH[:9]], ORG, sw=4, z=6)
    for k in range(9): dot(g, ux(PATH[k][0]), uy(PATH[k][1]), 7, ORG)
    g.beat = 2
    g.line([(ux(10), uy(2)), (ux(0), uy(0))], PUR, sw=6, arrow="end", head=22, z=7); dot(g, ux(0), uy(0), 12, PUR)
    g.beat = 1
    box(g, 850, 130, 380, 170, ORG, ORGL, T("負梯度:之字形", "negative gradient: zigzag"), T("$y$ 每步乘 $-0.5$\n$x$ 每步乘 $0.94$\n第 10 步 $f=14.5$,第 30 步 $1.2$", "$y$ times $-0.5$ each step\n$x$ times $0.94$ each step\nstep 10: $f=14.5$, step 30: $1.2$"), ts=21, bs=20)
    g.beat = 2
    box(g, 850, 330, 380, 150, PUR, PURL, T("牛頓法", "Newton"), r"$H=\mathrm{diag}(1,25)$" + "\n" + r"$\theta_0-H^{-1}\nabla f=(0,0)$", ts=22, bs=22)
    g.beat = 3
    gold(g, 850, 510, 380, 100, T("二次函數:牛頓法精確,第 1 步損失就是 $0$", "quadratic: Newton is exact, loss $0$ at step 1"), size=20)
    banner(g, T("梯度只知道此刻最陡;曲率告訴你碗有多彎", "the gradient knows only the steepest direction; curvature says how curved the bowl is"))
    d.done(g, "牛頓法", "Newton's method", {0: ("只用梯度，路徑是之字形：y 每步乘負 0.5，正負交替地縮小，x 每步只乘 0.94，橫向前進很慢。第 10 步損失剩 14.5，第 30 步還有 1.2。", "With gradients only the path zigzags: y is multiplied by minus 0.5 each step, shrinking with alternating signs, while x is multiplied by only 0.94, so sideways progress is slow. The loss is 14.5 at step 10 and still 1.2 at step 30."), 1: ("如果多用曲率，海森矩陣是 1、25 的對角矩陣。牛頓法用它的逆縮放梯度，一步走到原點。", "If we also use curvature, the Hessian is diag 1, 25. Newton rescales the gradient by its inverse and lands at the origin in one step."), 2: ("因為這個碗剛好是二次函數，牛頓法是精確的，第一步損失就是 0。", "Because the bowl is exactly quadratic, Newton is exact and the loss is 0 after the first step."), 3: ("", "")})

    # 6 cost
    g = d.page("曲率的帳單:隨參數數平方成長", "Curvature's bill grows with d squared")
    rows = []
    for dd, gb, hb, inv, t in ROWS: rows.append([f"$10^{round(math.log10(dd))}$", hum(gb), hum(hb), f"$10^{{{round(math.log10(inv))}}}$", hum_t(t)])
    table(g, 60, 130, [150, 210, 230, 230, 330], ["$d$", T("梯度 $4d$ bytes", "gradient $4d$ bytes"), T("海森 $4d^2$ bytes", "Hessian $4d^2$ bytes"), T("求逆 $d^3$ 次", "inversion $d^3$ ops"), T("時間($10^{13}$ 次/秒)", "time ($10^{13}$ ops/s)")], rows, rh=84, fs=26, styles=[(CARD, GRAY), (CARD, GRAY), (REDL, RED)], beats=[1, 2, 3])
    g.beat = 4
    gold(g, 190, 520, 900, 90, T("$d=10^6$:光存 $H$ 要 4 TB;求逆約 27.8 小時,而且只是一步", "$d=10^6$: storing $H$ alone is 4 TB; inversion about 27.8 hours, for a single step"), size=24)
    banner(g, T("時間用了假設的硬體速度,只是量級估計", "times assume a hypothetical hardware speed: order-of-magnitude only"))
    d.done(g, "曲率的帳單", "The curvature bill", {0: ("曲率這麼有用，為什麼深度學習不用？因為代價隨參數個數 d 快速成長。", "Curvature is so useful; why does deep learning not use it? Because the cost grows quickly with the number of parameters d."), 1: ("d 等於 100：梯度 400 位元組，海森 40 KB，求逆一百萬次運算，0.1 微秒。", "With d equal 100: gradient 400 bytes, Hessian 40 KB, inversion a million operations, 0.1 microseconds."), 2: ("d 等於一萬：海森 400 MB，求逆 10 的 12 次方次，0.1 秒。", "With d equal ten thousand: Hessian 400 MB, inversion 10 to the 12 operations, 0.1 second."), 3: ("d 等於一百萬，對今天的網路還算小：海森 4 TB，求逆要約 27.8 小時，而且這只是一步。", "With d equal a million, small for today's networks: Hessian 4 TB, inversion about 27.8 hours, and that is only one step."), 4: ("時間用了假設的硬體速度，只是量級估計。", "The times assume a hypothetical hardware speed, so they are only order-of-magnitude estimates.")})
    closing(d, [(RED, REDL, "「損失會告訴最佳化器怎麼走」", "“the loss tells the optimizer which way to go”", "不會，損失只替參數打分", "no: the loss only scores parameters"), (ORG, ORGL, "「梯度指向最低點」", "“the gradient points at the lowest point”", "不一定:它只是此刻最陡的上坡方向", "not necessarily: it is only the steepest uphill direction right now"), (PUR, PURL, "「二階方法一定比較好」", "“second-order methods are always better”", "小問題很強,$d=10^6$ 時成本無法負擔", "strong on small problems, unaffordable at $d=10^6$")],
            ("所以深度學習選一階方法:每一步只要一個梯度,反向傳播(KP06)便宜算出整個梯度。但只有方向,這一步該走多遠?", "So deep learning picks first-order methods: each step needs one gradient, which backpropagation (KP06) computes cheaply. But with only a direction, how far should one step go?"), ("下一個單元:梯度下降與學習率", "next unit: gradient descent and the learning rate"))
    return d.pages
