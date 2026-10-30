import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lec03_kit import *

KP = "KP14"; N = 7
f2 = lambda a, b: 0.5 * (a * b - 1) ** 2
def sharp(a, rho=0.5):
    w = (a, 1 / a); return max(f2(w[0] + rho * math.cos(t / 3600 * 2 * math.pi), w[1] + rho * math.sin(t / 3600 * 2 * math.pi)) for t in range(3600)) - f2(*w)
AS = (0.25, 0.5, 1, 2, 4, 8); SHP = [sharp(a) for a in AS]; assert [round(v, 3) for v in SHP] == [2.069, 0.614, 0.346, 0.614, 2.069, 8.049]
G0 = (10, 50); NG = math.hypot(*G0); EPS = (0.5 * G0[0] / NG, 0.5 * G0[1] / NG); assert [round(v, 4) for v in EPS] == [0.0981, 0.4903]
BETA = 0.5; MASKS = [(0, 0), (1, 0), (0, 1), (1, 1)]; EH = [(m[0] / BETA * EPS[0], m[1] / BETA * EPS[1]) for m in MASKS]
assert [(round(a, 4), round(b, 4)) for a, b in EH] == [(0, 0), (0.1961, 0), (0, 0.9806), (0.1961, 0.9806)]
assert abs(sum(a for a, _ in EH) / 4 - EPS[0]) < 1e-12 and abs(sum(b for _, b in EH) / 4 - EPS[1]) < 1e-12 and abs(sum(a * a + b * b for a, b in EH) / 4 - 0.25 / BETA) < 1e-12 and abs(math.hypot(*EH[3]) - 1.0) < 1e-3
def asam(w, g, rho, gam):
    T = [abs(x) + gam for x in w]; tg = [T[i] * g[i] for i in range(len(w))]; nt = math.sqrt(sum(v * v for v in tg)); return [rho * T[i] * T[i] * g[i] / nt for i in range(len(w))]
E3 = asam((1, -2, 0.5), (0.3, 0.1, -0.8), 0.1, 0.01); assert [round(v, 4) for v in E3] == [0.056, 0.0739, -0.0381]
E2 = asam((10, 2), (10, 50), 0.1, 0.01); assert [round(v, 3) for v in E2] == [0.706, 0.142]
REL_S = [EPS[0] / 10 * 100, EPS[1] / 2 * 100]; assert [round(v, 2) for v in REL_S] == [0.98, 24.51] and round(E2[0] / 10 * 100, 1) == 7.1 and round(E2[1] / 2 * 100, 1) == 7.1
COST = lambda r: (4 + 2 * r) / 3; assert COST(1) == 2 and abs(COST(0) - 4 / 3) < 1e-12

def make(lang):
    d = D(lang, KP, N); T = d.T
    g = d.page("KP14 SAM 的兩個痛點", "KP14 Two pain points of SAM")
    g.beat = 1
    box(g, 50, 126, 570, 200, RED, REDL, T("成本", "cost"), T("每步 2 次前向 + 2 次反向\n成本模型(前向 1、反向 2):SGD 3 單位,SAM 6 單位,約 2 倍", "two forward and two backward passes per step\ncost model (forward 1, backward 2): SGD 3 units, SAM 6, about 2×"), ts=24, bs=21)
    g.beat = 2
    box(g, 660, 126, 570, 200, ORG, ORGL, T("尺度", "scale"), T("固定 $\\rho$ 的球對每個權重一視同仁\n重新縮放權重,銳度就會變", "a fixed-$\\rho$ ball treats every weight alike\nrescale the weights and the sharpness changes"), ts=24, bs=21)
    g.beat = 3
    chain(g, 50, 360, 1180, 130, [(BLUE, BLUEL, T("對症下藥:成本", "for cost"), T("少算一點:SWP 少擾動權重,SDS 少處理樣本", "compute less: SWP perturbs fewer weights, SDS handles fewer samples")), (GRN, GRNL, T("對症下藥:尺度", "for scale"), T("改鄰域的形狀:ASAM", "change the neighborhood shape: ASAM"))], ts=22, bs=20, gap=100, beat0=3)
    banner(g, T("三者改的是不同的東西:計算預算、樣本選擇、鄰域幾何", "the three change different things: compute budget, sample selection, neighborhood geometry"))
    d.done(g, "兩個痛點", "Two pain points", {0: ("SAM 有兩個痛點。第一是成本：SGD 每步算一次梯度，SAM 先算一次得到擾動，再到擾動點算第二次，所以每步兩次前向兩次反向。成本模型下 SGD 是 3 個單位，SAM 是 6 個，約 2 倍。", "SAM has two pain points. First, cost: SGD takes one gradient per step; SAM takes one to get the perturbation and a second at the perturbed point, so two forward and two backward passes. In the cost model SGD is 3 units and SAM 6, about twice."), 1: ("第二是尺度：固定半徑的球對每個權重一視同仁，重新縮放權重，銳度就會變。", "Second, scale: a ball of fixed radius treats every weight alike, so rescaling the weights changes the sharpness."), 2: ("成本靠少算一點：SWP 少擾動權重，SDS 少處理樣本；", "Cost is reduced by computing less: SWP perturbs fewer weights, SDS handles fewer samples;"), 3: ("尺度要靠改鄰域的形狀，這是 ASAM。", "scale needs a change of the neighborhood's shape: ASAM.")})

    g = d.page("SWP:隨機遮罩的擾動", "SWP: a randomly masked perturbation")
    g.beat = 1
    g.rect(120, 118, 1040, 100, fill=CARD, stroke=GRAY, sw=3, radius=16, z=3); g.text(130, 118, 1020, 100, r"$\hat\epsilon=\dfrac{1}{\beta}\,m\odot\epsilon^{*},\qquad m_i\sim\mathrm{Bernoulli}(\beta)$".replace(r"\dfrac", r"\frac"), size=34, color=INK, align="center", vcenter=True, z=8)
    g.beat = 2
    g.text(50, 236, 1180, 36, T("玩具碗:$\\epsilon^{*}=(0.0981,\\,0.4903)$,$\\beta=0.5$,四種遮罩各有 $1/4$ 機率", "toy bowl: $\\epsilon^{*}=(0.0981,\\,0.4903)$, $\\beta=0.5$, each of four masks has probability $1/4$"), size=19, color=MUTED, bold=True, align="center", z=8)
    rows = [[f"$({m[0]},{m[1]})$", f"$({a:.4f},\\ {b:.4f})$", f"${math.hypot(a, b):.4f}$".replace("1.0000", "1.0")] for m, (a, b) in zip(MASKS, EH)]
    table(g, 140, 280, [220, 460, 260], [T("遮罩 $m$", "mask $m$"), r"$\hat\epsilon$", r"$\|\hat\epsilon\|$"], rows, rh=48, fs=22, hfs=18, styles=[None, None, None, (REDL, RED)], gap=6)
    g.beat = 3
    gold(g, 150, 560, 980, 66, T("四個值的平均 $=(0.0981,\\,0.4903)=\\epsilon^{*}$;$\\mathbb{E}\\|\\hat\\epsilon\\|^2=0.5=\\rho^2/\\beta$", "the mean of the four $=(0.0981,\\,0.4903)=\\epsilon^{*}$; $\\mathbb{E}\\|\\hat\\epsilon\\|^2=0.5=\\rho^2/\\beta$"), size=21)
    banner(g, T("除以 $\\beta$ 的理由:補回被丟掉的部分,使 $\\mathbb{E}[\\hat\\epsilon]=\\epsilon^{*}$", "dividing by $\\beta$ restores what was dropped, so $\\mathbb{E}[\\hat\\epsilon]=\\epsilon^{*}$"))
    d.done(g, "SWP", "SWP", {0: ("SWP 每步抽一個遮罩，各分量獨立，以機率 beta 保留，擾動取成遮罩逐元素乘最壞擾動再除以 beta。沒被選中的權重這一步不受擾動。", "SWP draws a mask each step, each component independently kept with probability beta, and takes the perturbation as the mask elementwise times the worst perturbation over beta. Weights not selected are unperturbed this step."), 1: ("在玩具碗上，最壞擾動是 0.0981、0.4903，beta 取 0.5，四種遮罩得到零、0.1961 與 0、0 與 0.9806、0.1961 與 0.9806。", "On the toy bowl the worst perturbation is 0.0981, 0.4903 and beta is 0.5; the four masks give zero, 0.1961 and 0, 0 and 0.9806, and 0.1961 and 0.9806."), 2: ("", ""), 3: ("四個值的平均正好是最壞擾動，所以期望相等；這就是除以 beta 的理由，補回被丟掉的部分。", "The mean of the four is exactly the worst perturbation, so the expectation matches; that is why we divide by beta, restoring what was dropped.")})

    g = d.page("SWP:不要把結論說過頭", "SWP: do not over-read the result")
    g.beat = 1
    box(g, 50, 126, 570, 190, RED, REDL, T("單次的 $\\hat\\epsilon$ 可以比 $\\rho$ 長", "a single $\\hat\\epsilon$ can be longer than $\\rho$"), T("$(1,1)$ 那一格長 $1.0=\\rho/\\beta$\n已跑出半徑 $0.5$ 的球", "the $(1,1)$ mask has length $1.0=\\rho/\\beta$\nit leaves the radius-$0.5$ ball"), ts=21, bs=22)
    g.beat = 2
    box(g, 660, 126, 570, 190, ORG, ORGL, T("梯度的期望不保證無偏", "the expected gradient is not guaranteed unbiased"), T("SAM 真正用的是 $\\nabla L(w+\\hat\\epsilon)$\n損失對擾動不是線性的", "SAM actually uses $\\nabla L(w+\\hat\\epsilon)$\nthe loss is not linear in the perturbation"), ts=19, bs=21)
    g.beat = 3
    box(g, 50, 350, 570, 150, PUR, PURL, T("省多少時間?", "how much time is saved"), T("取決於實作,不是由 $\\beta$ 決定的固定倍數;本講義不給倍數", "depends on the implementation, not a fixed multiple of $\\beta$; none is claimed"), ts=21, bs=21)
    g.beat = 4
    box(g, 660, 350, 570, 150, BLUE, BLUEL, T("像 Dropout 的正則化?", "a Dropout-like regularizer?"), T("來源沒有證據,這裡不採用", "the source gives no evidence, so it is not adopted"), ts=21, bs=21)
    gold(g, 150, 530, 980, 90, T("碗上 60 步($\\eta=0.05$):SAM 終點損失 $8.77$;SWP 五個種子 $26.3,\\ 34.0,\\ 1.28,\\ 0.198,\\ 1.29$——這不是速度或效果的比較", "60 steps on the bowl ($\\eta=0.05$): SAM ends at $8.77$; SWP with five seeds $26.3,\\ 34.0,\\ 1.28,\\ 0.198,\\ 1.29$ — not a speed or quality comparison"), size=19) if False else None
    banner(g, T("只能說 $\\hat\\epsilon$ 本身的期望等於 $\\epsilon^{*}$", "all we can say is that $\\mathbb{E}[\\hat\\epsilon]=\\epsilon^{*}$"))
    d.done(g, "SWP 的限縮", "Limits of SWP", {0: ("不要把結論說過頭，有兩點要限縮。第一，單次的擾動可以比 rho 長：兩個分量都保留的那一格長度是 1.0，已經跑出半徑 0.5 的球。", "Do not over-read it; two limits. First, a single perturbation can exceed rho: the mask keeping both components has length 1.0, leaving the radius-0.5 ball."), 1: ("第二，SAM 真正用到的是在擾動點的梯度，損失對擾動不是線性的，所以梯度的期望不保證無偏。", "Second, what SAM uses is the gradient at the perturbed point, and the loss is not linear in the perturbation, so the expected gradient is not guaranteed unbiased."), 2: ("省多少時間取決於實作，不是由 beta 決定的固定倍數，本講義不給倍數。", "How much time is saved depends on the implementation, not a fixed multiple of beta; no multiple is claimed."), 3: ("隨機遮罩的雜訊是否有類似 Dropout 的正則化效果，來源沒有證據，這裡不採用。", "Whether the mask's noise acts like Dropout regularization has no evidence in the source and is not adopted here.")})

    g = d.page("SDS:只讓銳度敏感的樣本做第二次反傳", "SDS: only sharp samples take a second backward")
    g.beat = 1
    chain(g, 50, 120, 1180, 170, [(BLUE, BLUEL, T("1 整批算擾動", "1 batch perturbation"), r"$\epsilon^{*}$"), (ORG, ORGL, T("2 逐筆量漲幅", "2 per-sample rise"), r"$\ell_i(w+\epsilon^{*})-\ell_i(w)$"), (GRN, GRNL, T("3 漲幅 $>\\alpha$", "3 rise $>\\alpha$"), r"$\mathcal{B}^{+}$" + T("\n做第二次反傳", "\ndoes the second backward pass"))], ts=20, bs=24, gap=50, beat0=1)
    g.beat = 4
    rows = [[f"{r:g}", f"{4 + 2 * r:g}", f"{COST(r):.3f}".rstrip("0").rstrip(".")] for r in (0.25, 0.5, 1.0)]
    g.text(50, 316, 1180, 34, T("成本:每步 $(1+2)+1+2r=4+2r$ 單位,SGD 是 3 單位(前向 1、反向 2)", "cost: $(1+2)+1+2r=4+2r$ units per step against SGD's 3 (forward 1, backward 2)"), size=19, color=MUTED, bold=True, align="center", z=8)
    table(g, 190, 358, [260, 300, 340], [T("保留比例 $r$", "kept fraction $r$"), T("每步單位", "units per step"), T("相對 SGD", "relative to SGD")], rows, rh=50, fs=24, hfs=18, styles=[(GRNL, GRN), (CARD, GRAY), (REDL, RED)], gap=6)
    g.beat = 5
    gold(g, 150, 556, 980, 70, T("下限 $4/3$;第二次梯度偏向「銳度高的樣本」,不再是完整批次 SAM 梯度的無偏估計", "floor $4/3$; the second gradient leans toward sharp samples and is no longer an unbiased estimate of the full-batch SAM gradient"), size=20)
    banner(g, T("SDS 不省掉所有第二輪的工作:第一次反傳與擾動點前向仍是整批", "SDS keeps the first backward and perturbed forward full-batch"))
    d.done(g, "SDS", "SDS", {0: ("SDS 從資料這一維省。先用整批算出擾動，", "SDS saves along the data dimension. First compute the perturbation on the whole batch,"), 1: ("再對每筆樣本做一次前向，看擾動讓它的損失漲了多少。", "then run a forward pass per sample to see how much the perturbation raises its loss."), 2: ("漲幅大於門檻 alpha 的樣本，才做第二次反傳。", "Only samples whose rise exceeds the threshold alpha take the second backward pass."), 3: ("", ""), 4: ("成本要誠實算：第一次反傳整批都要做，擾動點的前向整批都要做，只有第二次反傳按保留比例縮小。每步 4 加 2 倍保留比例單位，相對 SGD 的 3 單位，下限是三分之四。", "Count the cost honestly: the first backward on the whole batch and the forward at the perturbed point on the whole batch remain; only the second backward shrinks with the kept fraction. 4 plus 2 times the fraction units per step against SGD's 3, with floor four thirds."), 5: ("還有一個提醒：第二次梯度只在被選中的樣本上算，所以不再是完整批次梯度的無偏估計。", "One more caution: the second gradient is computed only on the selected samples, so it is no longer an unbiased estimate of the full-batch gradient.")})

    g = d.page("ASAM:讓鄰域跟著權重的尺度走", "ASAM: letting the neighborhood follow the weights' scale")
    g.beat = 1
    g.rect(120, 118, 1040, 100, fill=CARD, stroke=GRAY, sw=3, radius=16, z=3); g.text(130, 118, 1020, 100, r"$\max_{\|T_w^{-1}\epsilon\|\le\rho}L(w+\epsilon)-L(w),\quad T_w=\mathrm{diag}(|w_i|+\gamma)$", size=28, color=INK, align="center", vcenter=True, z=8)
    g.beat = 2
    g.rect(190, 240, 900, 100, fill="#fef3c7", stroke=GOLD, sw=3, radius=16, z=3); g.text(200, 240, 880, 100, r"$\epsilon^{*}=\rho\,\dfrac{T_w^{2}\,g}{\|T_w\,g\|}$".replace(r"\dfrac", r"\frac"), size=40, color=INK, align="center", vcenter=True, z=8)
    g.beat = 3
    box(g, 50, 370, 570, 150, BLUE, BLUEL, T("驗算", "a check"), T("$w=(1,-2,0.5)$,$g=(0.3,0.1,-0.8)$,$\\rho=0.1$,$\\gamma=0.01$\n$\\epsilon^{*}=(0.0560,\\,0.0739,\\,-0.0381)$,$\\|T^{-1}\\epsilon^{*}\\|=0.1$", "$w=(1,-2,0.5)$, $g=(0.3,0.1,-0.8)$\n$\\rho=0.1$, $\\gamma=0.01$\n$\\epsilon^{*}=(0.0560,\\,0.0739,\\,-0.0381)$"), ts=20, bs=17)
    g.beat = 4
    box(g, 660, 370, 570, 150, GRN, GRNL, T("玩具碗 $(10,2)$,相對權重的擾動", "toy bowl $(10,2)$: perturbation relative to the weights"), T("SAM:$(0.98\\%,\\ 24.5\\%)$\nASAM:$(7.1\\%,\\ 7.1\\%)$", "SAM: $(0.98\\%,\\ 24.5\\%)$\nASAM: $(7.1\\%,\\ 7.1\\%)$"), ts=18, bs=24)
    g.beat = 5
    gold(g, 150, 545, 980, 80, T("ASAM 仍是兩次前向加兩次反向,約 $2$ 倍成本:它改的是鄰域的形狀,不是計算量", "ASAM still costs two forward and two backward passes, about $2\\times$: it changes the shape, not the compute"), size=21)
    banner(g, T("大的權重有大的容許擾動,小的權重有小的:球變成沿各軸的橢球", "large weights allow large perturbations, small ones small: an axis-aligned ellipsoid"))
    d.done(g, "ASAM", "ASAM", {0: ("SAM 的鄰域是各向同性的球，對每個權重一視同仁。ASAM 改用權重尺度正規化的鄰域：大的權重有大的容許擾動，小的有小的，球變成橢球。", "SAM's neighborhood is an isotropic ball treating every weight alike. ASAM uses a neighborhood normalized by the weight scale: large weights get large allowed perturbations and small ones small, so the ball becomes an ellipsoid."), 1: ("p 等於 2 時的最壞擾動是 rho 乘 T 平方乘梯度，除以 T 乘梯度的長度。", "For p equal 2 the worst perturbation is rho times T squared times the gradient over the length of T times the gradient."), 2: ("驗算：權重 1、負 2、0.5，梯度 0.3、0.1、負 0.8，rho 0.1，gamma 0.01，得到 0.0560、0.0739、負 0.0381，正規化長度剛好 0.1。", "A check: weights 1, minus 2, 0.5, gradient 0.3, 0.1, minus 0.8, rho 0.1, gamma 0.01 give 0.0560, 0.0739, minus 0.0381, with normalized length exactly 0.1."), 3: ("在玩具碗上，SAM 的擾動相對權重是 0.98% 與 24.5%，y 方向被擾動得很大；ASAM 兩個方向都是 7.1%。", "On the toy bowl SAM's perturbation is 0.98 percent and 24.5 percent of the weights, very large in y; ASAM gives 7.1 percent in both."), 4: ("", ""), 5: ("ASAM 不省計算，仍是兩次前向加兩次反向，約 2 倍成本。", "ASAM does not save compute: still two forward and two backward passes, about twice the cost.")})

    g = d.page("尺度不變性:同一個函數,不同的銳度", "Scale invariance: one function, different sharpness")
    g.beat = 1
    g.text(50, 108, 1180, 36, T("兩層線性網路的最小點 $(a,\\,1/a)$ 都是同一個函數;固定 $\\rho=0.5$ 的 SAM 銳度", "the minima $(a,\\,1/a)$ of the two-layer linear net are the same function; SAM sharpness at fixed $\\rho=0.5$"), size=20, color=MUTED, bold=True, align="center", z=8)
    base = 530; sc = 52
    for i, (a, sv) in enumerate(zip(AS, SHP)):
        g.beat = 1 if i < 3 else 2
        x = 80 + i * 190
        g.rect(x, base - sv * sc, 130, sv * sc, fill=(GRNL if sv < 1 else (ORGL if sv < 3 else REDL)), stroke=(GRN if sv < 1 else (ORG if sv < 3 else RED)), sw=3, radius=6, z=3)
        g.text(x - 20, base - sv * sc - 42, 170, 36, f"${sv:.3f}$", size=22, color=INK, bold=True, align="center", z=8); g.text(x - 20, base + 4, 170, 34, f"$a={'1/4' if a == 0.25 else ('1/2' if a == 0.5 else f'{a:g}')}$", size=20, color=INK, bold=True, align="center", z=8)
    g.line([(60, base), (1220, base)], INK, sw=3, z=3)
    g.beat = 3
    gold(g, 150, 574, 980, 56, T("同一個函數的銳度差 $23$ 倍以上(最小 $0.346$,最大 $8.049$)", "the same function's sharpness varies over $23\\times$ (min $0.346$, max $8.049$)"), size=20)
    banner(g, T("ASAM 要求 $T_{Aw}=AT_w$:對 $\\gamma>0$ 只是近似,$\\gamma=0$ 時才相等", "ASAM wants $T_{Aw}=AT_w$: only approximate for $\\gamma>0$, exact at $\\gamma=0$"))
    d.done(g, "尺度不變性", "Scale invariance", {0: ("把兩層線性網路的最小點 a 與 1 除以 a 的縮放從四分之一掃到 8，它們都是同一個函數。固定半徑的 SAM 銳度是 2.069、0.614、0.346。", "Sweeping the scaling a of the two-layer minima (a, 1 over a) from a quarter to 8, they are all the same function. SAM's fixed-radius sharpness is 2.069, 0.614, 0.346."), 1: ("後面是 0.614、2.069、8.049。", "Then 0.614, 2.069, 8.049."), 2: ("", ""), 3: ("同一個函數的銳度差 23 倍以上。ASAM 要求 T 滿足縮放不變，對 gamma 大於零只是近似，來源說恆成立並不準確。", "The same function's sharpness varies by over 23 times. ASAM asks T to be scale-invariant; for gamma above zero this is only approximate, so the source's claim that it always holds is not accurate.")})

    g = d.page("三種方法改的是不同的東西", "The three methods change different things")
    rows = [["SWP", T("計算預算:每步只擾動隨機一部分權重", "compute budget: perturb a random part of the weights"), T("$\\mathbb{E}[\\hat\\epsilon]=\\epsilon^{*}$,變異變大", "$\\mathbb{E}[\\hat\\epsilon]=\\epsilon^{*}$, larger variance"), T("梯度的期望不保證無偏;省多少看實作", "expected gradient not guaranteed unbiased; savings depend on implementation")], ["SDS", T("樣本選擇:漲幅大於 $\\alpha$ 才做第二次反傳", "sample selection: second backward only above $\\alpha$"), T("成本 $(4+2r)/3$ 倍 SGD(下限 $4/3$)", "cost $(4+2r)/3\\times$ SGD (floor $4/3$)"), T("第二次梯度偏向高銳度樣本", "the second gradient leans toward sharp samples")], ["ASAM", T("鄰域幾何:球換成 $\\|T_w^{-1}\\epsilon\\|\\le\\rho$", "neighborhood geometry: $\\|T_w^{-1}\\epsilon\\|\\le\\rho$"), T("成本不變(約 2 倍),銳度對正縮放近乎不變", "cost unchanged (about $2\\times$), sharpness nearly scale-invariant"), T("$\\gamma>0$ 時只是近似;$\\rho$ 不能與 SAM 的比", "approximate for $\\gamma>0$; $\\rho$ is not comparable with SAM's")]]
    g.beat = 1
    table(g, 50, 120, [130, 360, 340, 350], ["", T("改的是什麼", "what it changes"), T("相對 SAM", "relative to SAM"), T("不保證什麼", "what is not guaranteed")], rows, rh=130, fs=17, hfs=19, styles=[(REDL, RED), (ORGL, ORG), (GRNL, GRN)], beats=[1, 2, 3], gap=8)
    g.beat = 4
    gold(g, 150, 560, 980, 66, T("三者可以疊在一起用;不要把「省了計算」與「改了鄰域」混為一談", "the three can be combined; do not confuse “saving compute” with “changing the neighborhood”"), size=22)
    banner(g, T("從最佳化理論看,它們到底保證了什麼?(KP15)", "from optimization theory, what do they actually guarantee? (KP15)"))
    d.done(g, "三種方法", "Three methods", {0: ("三種方法改的是不同的東西。", "The three methods change different things."), 1: ("SWP 改計算預算：每步只擾動隨機一部分權重，擾動的期望等於最壞擾動，變異變大。", "SWP changes the compute budget: each step perturbs a random part of the weights; the perturbation's expectation equals the worst perturbation and its variance grows."), 2: ("SDS 改樣本選擇：成本是四加二倍保留比例除以三倍的 SGD。ASAM 改鄰域幾何：成本不變，約兩倍，銳度對正縮放近乎不變。", "SDS changes sample selection: cost is 4 plus 2 times the fraction over 3 times SGD. ASAM changes the neighborhood geometry: cost unchanged, about twice, sharpness nearly invariant to positive scaling."), 3: ("", ""), 4: ("三者可以疊在一起用，但各自解決的問題不同，不要把省了計算與改了鄰域混為一談。", "The three can be combined, but they solve different problems; do not confuse saving compute with changing the neighborhood.")})
    return d.pages
