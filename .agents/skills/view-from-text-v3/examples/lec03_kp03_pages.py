import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lec03_kit import *

KP = "KP03"; N = 7
f = lambda x, y: 0.5 * (x * x + 25 * y * y)
P = [(10.0, 2.0)]
for _ in range(80): x, y = P[-1]; P.append((x * 0.94, y * -0.5))
gn = lambda k: math.hypot(P[k][0], 25 * P[k][1]); fk = lambda k: f(*P[k]); df = lambda k: abs(fk(k + 1) - fk(k))
ROWS = [(k, gn(k), df(k), fk(k)) for k in (0, 10, 28, 29, 64)]
assert [(round(a, 3), round(b, 3), round(c, 3)) for _, a, b, c in ROWS] == [(50.99, 43.32, 100.0), (5.386, 1.688, 14.505), (1.768, 0.182, 1.564), (1.662, 0.161, 1.382), (0.191, 0.002, 0.018)]
EPS = 0.2; K_LOSS = next(k for k in range(80) if df(k) <= EPS) + 1; K_GRAD = next(k for k in range(80) if gn(k) <= EPS)
assert K_LOSS == 29 and K_GRAD == 64 and round(0.06 * gn(29) ** 2, 3) == 0.166

def make(lang):
    d = D(lang, KP, N); T = d.T
    # 1 zero gradient
    g = d.page("KP03 梯度為零代表什麼", "KP03 What a zero gradient means")
    g.beat = 1
    box(g, 50, 126, 540, 190, BLUE, BLUEL, T("局部最小點 $w^*$", "a local minimum $w^*$"), T("沿任何方向走都不再下降\n每個方向的一階變化都是零", "no direction goes lower\nthe first-order change is zero in every direction"), ts=24, bs=22)
    g.beat = 2
    box(g, 690, 126, 540, 190, ORG, ORGL, T("一階最優性條件", "first-order optimality"), r"$\nabla f(w^*)=\mathbf{0}$", ts=24, bs=34)
    g.link(g.mark() - 1, g.mark() - 1, 4, 12) if False else None
    g.beat = 3
    gold(g, 50, 350, 1180, 100, T("方向:局部最小 $\\Rightarrow$ 梯度為零;反過來不成立。梯度為零只是必要條件,不是充分條件。", "direction: local minimum $\\Rightarrow$ zero gradient, not the other way round. A necessary condition, not a sufficient one."), size=24)
    g.beat = 4
    g.rect(190, 490, 900, 110, fill=CARD, stroke=GRAY, sw=3, radius=16, z=3); g.text(200, 490, 880, 110, r"$\|\nabla f(w^{(k+1)})\|_2\le\varepsilon$", size=44, color=INK, align="center", vcenter=True, z=8)
    banner(g, T("程式裡不可能要求梯度剛好是零,只能設一個小門檻 $\\varepsilon$", "in code the gradient is never exactly zero: use a small threshold $\\varepsilon$"))
    d.done(g, "梯度為零", "Zero gradient", {0: ("假設損失可微，w 星是局部最小點。在最小點沿任何方向走都不再下降，所以每個方向的一階變化都是零。", "Suppose the loss is differentiable and w star is a local minimum. No direction goes lower there, so the first-order change is zero in every direction."), 1: ("這叫一階最優性條件：梯度等於零。梯度為零的點叫駐點。", "This is the first-order optimality condition: the gradient is zero. A point with zero gradient is a stationary point."), 2: ("注意方向：局部最小蘊含梯度為零，反過來不成立。梯度為零只是必要條件，不是充分條件。", "Note the direction: a local minimum implies zero gradient, not the other way round. Zero gradient is necessary, not sufficient."), 3: ("實際寫成程式，只能設一個小門檻，梯度長度小於門檻就停下。", "In code we can only set a small threshold and stop when the gradient length falls below it."), 4: ("", "")})

    # 2 five indicators
    g = d.page("常用的五個停止指標", "Five common stopping indicators")
    rows = [[T("梯度範數", "gradient norm"), r"$\|\nabla f_{k+1}\|\le\varepsilon$", T("坡度夠平了嗎?", "is the slope flat enough?")], [T("損失變化", "loss change"), r"$|f_{k+1}-f_k|\le\varepsilon$", T("再走一步還降得動嗎?", "can one more step still lower it?")], [T("梯度變化", "gradient change"), r"$\|\nabla f_{k+1}-\nabla f_k\|\le\varepsilon$", T("前進的方向穩定了嗎?", "has the direction become stable?")], [T("迭代預算", "iteration budget"), r"$k\ge K_{\max}$", T("算力用完了嗎?", "has the compute run out?")], [T("驗證損失", "validation loss"), T("驗證集損失不再改善", "validation loss stops improving"), T("再練下去會過度擬合嗎?", "would more training overfit?")]]
    cols = [(BLUE, BLUEL), (BLUE, BLUEL), (BLUE, BLUEL), (ORG, ORGL), (GRN, GRNL)]
    table(g, 60, 112, [220, 480, 460], [T("指標", "indicator"), T("條件", "condition"), T("回答的問題", "the question it answers")], rows, rh=70, fs=22, styles=[(l, c) for c, l in cols], beats=[1, 1, 1, 2, 3], gap=8)
    g.beat = 1
    g.text(60, 596, 340, 36, T("前三個看訓練過程本身", "the first three watch training itself"), size=17, color=BLUE, bold=True, z=8)
    g.beat = 2; g.text(420, 596, 300, 36, T("第四個看資源", "the fourth watches resources"), size=17, color=ORG, bold=True, z=8)
    g.beat = 3; g.text(740, 596, 300, 36, T("第五個看泛化", "the fifth watches generalization"), size=17, color=GRN, bold=True, z=8)
    banner(g, T("深度學習常用後兩個:完整梯度很貴,SGD 的梯度有雜訊、長度永遠不是零", "deep learning often uses the last two: full gradients are costly and SGD gradients are noisy"))
    d.done(g, "五個指標", "Five indicators", {0: ("梯度範數只是其中一個。前三個看訓練過程本身：坡度夠平嗎、再走一步還降得動嗎、前進的方向穩定了嗎。", "The gradient norm is only one. The first three watch training itself: is the slope flat, can another step lower the loss, has the direction stabilized."), 1: ("第四個看資源：迭代預算，算力用完了嗎。", "The fourth watches resources: the iteration budget, has the compute run out."), 2: ("第五個看泛化：驗證損失不再改善，再練下去會不會過度擬合。深度學習常用後兩個，因為完整訓練集的梯度很貴，而隨機梯度下降的梯度帶雜訊，長度永遠不會真正是零。", "The fifth watches generalization: validation loss stops improving, would more training overfit. Deep learning often uses the last two because full-training-set gradients are costly and SGD gradients are noisy, never exactly zero."), 3: ("", "")})

    # 3 same path, different stops
    g = d.page("同一條路徑，準則停在不同的地方", "On one path, the criteria stop in different places")
    rows = [[str(k), f"{a:.4g}", f"{b:.4g}", f"{c:.4g}"] for k, a, b, c in ROWS]
    styles = [None, None, None, (GOLDL, GOLD), (GRNL, GRN)]
    g.text(50, 108, 1180, 36, T("玩具碗,$\\eta=0.06$,門檻 $\\varepsilon=0.2$", "toy bowl, $\\eta=0.06$, threshold $\\varepsilon=0.2$"), size=20, color=MUTED, bold=True, align="center", z=8)
    table(g, 60, 150, [130, 230, 290, 230], [T("步 $k$", "step $k$"), r"$\|\nabla f\|_2$", r"$|f_{k+1}-f_k|$", r"$f(w^{(k)})$"], rows, rh=62, fs=24, styles=styles, beats=[1, 1, 1, 2, 3], gap=8)
    g.beat = 2
    box(g, 960, 150, 270, 220, ORG, ORGL, T("損失變化準則", "loss-change rule"), T("第 29 步觸發\n$\\|\\nabla f\\|=1.662$\n$f=1.38$", "triggers at step 29\n$\\|\\nabla f\\|=1.662$\n$f=1.38$"), ts=20, bs=20)
    g.beat = 3
    box(g, 960, 390, 270, 220, GRN, GRNL, T("梯度範數準則", "gradient-norm rule"), T("第 64 步才觸發\n$\\|\\nabla f\\|=0.1906$\n$f=0.0182$", "triggers only at step 64\n$\\|\\nabla f\\|=0.1906$\n$f=0.0182$"), ts=20, bs=20)
    g.beat = 4
    gold(g, 60, 570, 860, 56, T("同一條路徑,兩個準則相差 35 步", "on the same path the two rules differ by 35 steps"), size=24)
    banner(g, T("損失變化準則停得早:這時梯度還是門檻的 8 倍多", "the loss-change rule stops early: the gradient is still over 8 times the threshold"))
    d.done(g, "同一條路徑", "One path", {0: ("用 KP01 的玩具碗，學習率 0.06，看各步的梯度範數與損失變化。", "On KP01's toy bowl with learning rate 0.06, look at the gradient norm and loss change at each step."), 1: ("取門檻 0.2。損失變化準則在第 29 步就觸發，這時梯度範數還有 1.662，是門檻的八倍多，損失 1.38，離最小值還有一段距離。", "Take threshold 0.2. The loss-change rule triggers at step 29, when the gradient norm is still 1.662, over eight times the threshold, and the loss 1.38 is still far from the minimum."), 2: ("梯度範數準則要到第 64 步才觸發，這時損失是 0.0182。", "The gradient-norm rule only triggers at step 64, when the loss is 0.0182."), 3: ("同一條路徑，兩個準則相差 35 步。", "On the same path the two criteria differ by 35 steps."), 4: ("", "")})

    # 4 why inconsistent
    g = d.page("為什麼不一致:平方關係", "Why they disagree: a square")
    g.beat = 1
    ids = chain(g, 50, 130, 1180, 200, [(BLUE, BLUEL, T("損失變化", "loss change"), r"$\approx\eta\,\|\nabla f\|^2$"), (ORG, ORGL, T("梯度範數", "gradient norm"), r"$\|\nabla f\|$")], ts=26, bs=40, gap=120, beat0=1)
    g.beat = 3
    g.text(560, 190, 160, 80, T("是它的\n平方×$\\eta$", "its square\n× $\\eta$"), size=20, color=RED, bold=True, align="center", z=8)
    box(g, 190, 370, 900, 130, PUR, PURL, T("第 29 步的數字", "the numbers at step 29"), r"$0.06\times1.662^2\approx0.166\ \ (\text{vs}\ 0.161)$", ts=22, bs=30)
    g.beat = 4
    gold(g, 190, 520, 900, 90, T("梯度範數還有 $1.66$ 時,損失變化已經只剩 $0.16$:「損失幾乎不降」不等於「坡度已經平了」", "with a gradient norm of $1.66$ the loss change is already only $0.16$: “barely falling” is not “flat”"), size=21)
    banner(g, T("損失變化是梯度範數的平方量級(KP02 的一階近似)", "the loss change is of the order of the squared gradient norm (KP02's first-order estimate)"))
    d.done(g, "平方關係", "The square", {0: ("為什麼不一致？損失變化量約等於學習率乘梯度範數的平方。", "Why the disagreement? The loss change is about the learning rate times the squared gradient norm."), 1: ("梯度範數是一次方。", "The gradient norm is first power."), 2: ("前者是後者的平方再乘上學習率。第 29 步：0.06 乘 1.662 的平方，約 0.166，和 0.161 相符。", "The former is the square of the latter times the learning rate. At step 29: 0.06 times 1.662 squared, about 0.166, close to 0.161."), 3: ("所以梯度範數還有 1.66 的時候，損失變化已經只剩 0.16。損失幾乎不降，不等於坡度已經平了。", "So with a gradient norm of 1.66 the loss change is already only 0.16. Barely falling does not mean flat.")})

    # 5 saddle & plateau
    g = d.page("看起來該停，其實不該停", "Looks like time to stop, but it is not")
    g.beat = 1
    xr, yr = (-2, 2), (-2, 2)
    ux, uy = axes(g, 50, 130, 440, 440, xr, yr)
    for c in (0.25, 1.0, 2.25):
        r_ = math.sqrt(c)
        g.line([(ux(x), uy(math.sqrt(x * x - c))) for x in [r_ + 0.02 * i for i in range(0, 45) if r_ + 0.02 * i <= 2]], BLUE, sw=3, z=3); g.line([(ux(x), uy(-math.sqrt(x * x - c))) for x in [r_ + 0.02 * i for i in range(0, 45) if r_ + 0.02 * i <= 2]], BLUE, sw=3, z=3)
        g.line([(ux(-x), uy(math.sqrt(x * x - c))) for x in [r_ + 0.02 * i for i in range(0, 45) if r_ + 0.02 * i <= 2]], BLUE, sw=3, z=3); g.line([(ux(-x), uy(-math.sqrt(x * x - c))) for x in [r_ + 0.02 * i for i in range(0, 45) if r_ + 0.02 * i <= 2]], BLUE, sw=3, z=3)
    dot(g, ux(0), uy(0), 14, RED); lab(g, ux(0) + 16, uy(0) - 16, 160, T("原點:$\\nabla f=0$", "origin: $\\nabla f=0$"), 18, RED)
    g.text(50, 574, 440, 50, T("$f=x^2-y^2$:沿 $x$ 上升、沿 $y$ 下降", "$f=x^2-y^2$: up along $x$, down along $y$"), size=18, color=MUTED, bold=True, align="center", z=8)
    box(g, 530, 130, 700, 200, RED, REDL, T("鞍點", "saddle point"), T("原點梯度範數是 $0$,任何梯度準則都立刻宣布到站\n但它不是最小點:高維非凸曲面上鞍點很多", "at the origin the gradient norm is $0$ and any gradient rule declares arrival\nyet it is not a minimum: high-dimensional non-convex surfaces are full of saddles"), ts=24, bs=21)
    g.beat = 2
    box(g, 530, 360, 700, 210, ORG, ORGL, T("平原", "plateau"), T("飽和區或廣闊平原:連續很多步損失只降 $0.00001$\n太早用損失變化準則停下,後面可能還有陡下坡", "saturated regions or wide plains: the loss falls by only $0.00001$ for many steps\nstopping early on the loss change may miss a steep descent ahead"), ts=24, bs=21)
    banner(g, T("只看梯度範數會被鞍點騙,只看損失變化會被平原騙", "the gradient norm alone is fooled by saddles; the loss change alone by plateaus"))
    d.done(g, "鞍點與平原", "Saddles and plateaus", {0: ("鞍點：取 x 平方減 y 平方，原點的梯度是零，梯度準則立刻宣布到站。但沿 x 軸往兩邊上升，沿 y 軸往兩邊下降，原點是鞍點，不是最小點。", "A saddle: take x squared minus y squared; the gradient at the origin is zero and a gradient rule declares arrival at once. But it rises along x and falls along y, so the origin is a saddle, not a minimum."), 1: ("平原：神經網路的飽和區或廣闊平原上，曲面非常平緩，連續很多步損失只降十萬分之一。太早用損失變化準則停下，模型會停在半路。", "Plateaus: in saturated regions or wide plains the surface is very flat and the loss falls by only a hundred-thousandth for many steps. Stopping early on the loss change leaves the model halfway."), 2: ("", "")})

    # 6 remedies
    g = d.page("怎麼辦:不單押一個指標", "What to do: never bet on one indicator")
    chain(g, 50, 140, 1180, 330, [(BLUE, BLUEL, T("1 組合準則", "1 combine rules"), T("梯度範數夠小,\n且損失變化連續 $N$ 步都小,\n或每步移動 $\\|w_{k+1}-w_k\\|\\le\\varepsilon_w$", "small gradient norm,\nand loss change small for $N$ steps,\nor step length $\\|w_{k+1}-w_k\\|\\le\\varepsilon_w$")), (ORG, ORGL, T("2 預算上限", "2 a budget cap"), T("不論如何,\n到 $K_{\\max}$ 就停,\n避免無止境", "whatever happens,\nstop at $K_{\\max}$,\nnever endless")), (GRN, GRNL, T("3 驗證損失早停", "3 early stopping"), T("每隔一段時間在驗證集量損失,\n不再改善就停;\n不要求訓練損失降到零", "measure the validation loss now and then,\nstop when it stops improving;\ndo not push training loss to zero"))], ts=24, bs=22, beat0=1, th=70)
    g.beat = 4
    gold(g, 190, 510, 900, 100, T("「訓練集收斂反而是壞事」說得太直接:訓練損失低不保證泛化好,所以需要驗證損失", "“training convergence is bad” is too blunt: low training loss does not guarantee generalization, hence the validation loss"), size=21)
    banner(g, T("「停在哪裡」不是純粹的數學問題", "“where to stop” is not purely a mathematical question"))
    d.done(g, "組合使用", "Use them together", {0: ("實務上不單押一個指標。", "In practice never bet on a single indicator."), 1: ("第一，組合準則：梯度夠小，並且損失變化連續很多步都小，或每步移動距離夠小。", "First, combine rules: a small gradient and a loss change that stays small for many steps, or a small step length."), 2: ("第二，預算上限：不論如何，到最大步數就停，避免無止境。", "Second, a budget: whatever happens, stop at the maximum number of steps."), 3: ("第三，驗證損失早停：深度學習不要求訓練損失降到零，每隔一段時間在驗證集上量損失，不再改善就停。", "Third, early stopping on validation loss: deep learning does not push training loss to zero; measure the validation loss now and then and stop when it stops improving."), 4: ("訓練集收斂反而是壞事說得太直接。更精確的講法是訓練損失降低不保證泛化好，所以需要驗證損失。", "Saying training convergence is bad is too blunt. More precisely, lower training loss does not guarantee good generalization, so we need the validation loss.")})
    closing(d, [(RED, REDL, "「梯度很小就是到了最小點」", "“a small gradient means a minimum”", "鞍點與平原都可能梯度很小", "saddles and plateaus can have small gradients too"), (ORG, ORGL, "「損失不再降就是收斂」", "“no more loss decrease means convergence”", "損失變化是梯度範數的平方量級,可以小很多", "the loss change is of squared order and can be far smaller"), (PUR, PURL, "「訓練損失越低越好」", "“the lower the training loss the better”", "要看驗證損失,否則可能過度擬合", "watch the validation loss or you may overfit")],
            ("停之前,每一步的步長能不能自動、安全地決定?", "Before stopping: can the step size of each step be chosen automatically and safely?"), ("下一個單元:線搜索與最陡下降", "next unit: line search and steepest descent"))
    return d.pages
