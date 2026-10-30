import math, sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lec03_kit import *

KP = "KP08"; N = 7
GR = (3, 1, -1, -3); SIG2 = sum(x * x for x in GR) / 4; assert SIG2 == 5
BAV = sorted(sum(c) / 2 for c in itertools.combinations(GR, 2)); assert BAV == [-2, -1, 0, 0, 1, 2]
STD2 = math.sqrt(sum(a * a for a in BAV) / 6); assert round(STD2, 3) == 1.291 and abs(STD2 - math.sqrt(SIG2 / 2 * (4 - 2) / (4 - 1))) < 1e-12 and round(math.sqrt(SIG2), 3) == 2.236
NB, BS = 1000, 32; FULL = NB // BS; assert FULL == 31 and FULL * BS == 992 and NB - FULL * BS == 8
assert NB // 10 == 100 and 1000 // 10 == 100 and 1000 // 100 == 10 and round(1 - 1 / math.e, 3) == 0.632

def make(lang):
    d = D(lang, KP, N); T = d.T
    g = d.page("KP08 兩個極端，與黃金折衷", "KP08 Two extremes and the compromise")
    rows = [[r"$b=1$" + T("(單筆)", " (single)"), T("最低", "lowest"), T("最大", "largest"), T("差:一次一筆", "poor: one at a time")], [r"$1<b<n$" + T("(小批次)", " (minibatch)"), T("中", "medium"), T("中", "medium"), T("好:矩陣運算並行", "good: parallel matrix ops")], [r"$b=n$" + T("(全批)", " (full batch)"), T("最高", "highest"), T("零", "zero"), T("常裝不下記憶體", "often exceeds memory")]]
    g.beat = 1
    table(g, 60, 120, [300, 220, 220, 400], [T("批次大小", "batch size"), T("每步成本", "cost per step"), T("梯度雜訊", "gradient noise"), T("硬體利用", "hardware use")], rows, rh=84, fs=22, hfs=20, styles=[(REDL, RED), (GRNL, GRN), (REDL, RED)], beats=[1, 2, 3], gap=10)
    g.beat = 4
    g.rect(190, 450, 900, 90, fill=CARD, stroke=GRAY, sw=3, radius=16, z=3); g.text(200, 450, 880, 90, r"$g_B(w)=\frac{1}{|B|}\sum_{i\in B}\nabla f_i(w)$", size=36, color=INK, align="center", vcenter=True, z=8)
    g.beat = 5
    gold(g, 150, 556, 980, 66, T("仍然無偏:均勻抽樣時 $\\mathbb{E}[g_B(w)]=\\nabla F(w)$;批次平均壓低方差", "still unbiased: $\\mathbb{E}[g_B(w)]=\\nabla F(w)$ under uniform sampling; averaging lowers the variance"), size=21)
    banner(g, T("常見 $b$ 是 32 到 256", "common batch sizes: 32 to 256"))
    d.done(g, "兩個極端", "Two extremes", {0: ("單筆 SGD 便宜但每步方差大，路徑抖得像醉漢，GPU 大部分閒著；全批沒有雜訊，但每步掃過全部資料，記憶體常常裝不下。", "Single-sample SGD is cheap but each step has high variance and the path jitters like a drunkard, leaving most of the GPU idle; the full batch has no noise but scans all data per step and often exceeds memory."), 1: ("單筆：成本最低，雜訊最大，硬體利用差。", "Single sample: lowest cost, largest noise, poor hardware use."), 2: ("小批次取中間：成本中，雜訊中，批次的前向反向可以打包成矩陣乘法，在 GPU 上並行。", "A minibatch sits in the middle: medium cost and noise, with a batch's forward and backward packed into matrix multiplications that run in parallel on a GPU."), 3: ("全批：成本最高，雜訊為零，常常裝不下記憶體。", "Full batch: highest cost, zero noise, often too big for memory."), 4: ("小批次用這一批的平均梯度更新。", "A minibatch updates with the average gradient of the batch."), 5: ("它仍然無偏，批次平均把方差壓低。", "It remains unbiased and averaging lowers the variance.")})

    g = d.page("資料怎麼分，什麼叫一個 epoch", "How to cut the data, and what an epoch is")
    g.beat = 1
    g.text(50, 108, 1180, 36, T("$n=1000$,$b=32$:先隨機洗牌,再依序切成互斥的批次", "$n=1000$, $b=32$: shuffle, then cut into disjoint batches"), size=22, color=MUTED, bold=True, align="center", z=8)
    cols = 31; bw = 33; x0 = 50
    for i in range(cols):
        g.rect(x0 + i * 36, 170, bw, 90, fill=BLUEL, stroke=BLUE, sw=2, radius=4, z=3)
    g.rect(x0 + cols * 36 + 4, 170, 60, 90, fill=REDL, stroke=RED, sw=3, radius=4, z=3)
    g.text(x0, 270, 1100, 40, T("31 個完整批次,共 $31\\cdot32=992$ 筆", "31 full batches, $31\\cdot32=992$ samples"), size=22, color=BLUE, bold=True, align="left", z=8)
    g.beat = 2
    g.text(x0 + cols * 36 - 120, 270, 260, 40, T("剩 $8$ 筆", "$8$ left over"), size=22, color=RED, bold=True, align="center", z=8)
    g.beat = 3
    box(g, 50, 340, 570, 170, GRN, GRNL, T("一個 epoch", "one epoch"), T("所有批次各用一次\n= $\\lfloor n/b\\rfloor=31$ 次更新\n每個樣本恰好被看過一次", "every batch used once\n$=\\lfloor n/b\\rfloor=31$ updates\neach sample seen exactly once"), ts=22, bs=22)
    g.beat = 4
    box(g, 660, 340, 570, 170, ORG, ORGL, T("尾巴 8 筆", "the 8 leftovers"), T("可當作一個較小的批次\n或丟掉(drop_last):下一輪重新洗牌", "use as a smaller batch\nor drop them (drop_last): the next epoch reshuffles"), ts=22, bs=22)
    g.beat = 5
    gold(g, 150, 540, 980, 84, T("有放回抽樣:批次可重疊;抽 $n$ 次約 $1-1/e\\approx63.2\\%$ 的獨特樣本被抽中。深度學習偏好無放回。", "with replacement batches can overlap: drawing $n$ times covers about $1-1/e\\approx63.2\\%$ unique samples. Deep learning prefers without replacement."), size=20)
    banner(g, T("互斥:任兩批沒有共同樣本;聯集是整個資料集(扣掉尾巴)", "disjoint: no two batches share a sample; their union is the dataset (minus the tail)"))
    d.done(g, "切批次與 epoch", "Batches and epochs", {0: ("標準做法：先把整個訓練集隨機洗牌，再依序切成互斥的批次。n 是 1000、批次 32，得到 31 個完整批次，用掉 992 筆。", "The standard way: shuffle the whole training set, then cut it into disjoint batches. With n 1000 and batch 32 we get 31 full batches using 992 samples."), 1: ("剩下 8 筆。", "8 samples are left."), 2: ("把所有批次各用一次，叫一個 epoch，共 31 次更新，每個樣本恰好被看過一次。", "Using every batch once is one epoch: 31 updates, each sample seen exactly once."), 3: ("剩下的 8 筆可以當作一個較小的批次，或丟掉，下一輪重新洗牌，換一批被丟。", "The 8 leftovers can serve as a smaller batch or be dropped; the next epoch reshuffles so a different set is dropped."), 4: ("", ""), 5: ("另一種是有放回抽樣，批次可以重疊；抽 n 次約有 63.2% 的獨特樣本被抽中。深度學習偏好無放回。", "The other is sampling with replacement, where batches may overlap; drawing n times covers about 63.2 percent unique samples. Deep learning prefers without replacement.")})

    g = d.page("批次估計有多準", "How accurate is a batch estimate")
    g.beat = 1
    g.text(50, 108, 1180, 36, T("在 $w=5$,單筆梯度 $(3,1,-1,-3)$:平均 $0$,$\\sigma^2=5$", "at $w=5$ the single-sample gradients are $(3,1,-1,-3)$: mean $0$, $\\sigma^2=5$"), size=22, color=MUTED, bold=True, align="center", z=8)
    chain(g, 50, 160, 1180, 150, [(BLUE, BLUEL, r"$\{1,2\}$", r"$(3+1)/2=2$"), (ORG, ORGL, r"$\{3,4\}$", r"$(-1-3)/2=-2$"), (GRN, GRNL, T("完整平均", "full average"), r"$0$")], ts=22, bs=30, gap=60, beat0=1)
    g.beat = 4
    gold(g, 150, 335, 980, 70, T("小批次的估計可以和完整梯度的方向完全相反,但兩批平均起來剛好是 $0$", "a minibatch estimate can point opposite to the full gradient, yet the batches average to exactly $0$"), size=22)
    g.beat = 5
    base = 590; sc = 52
    g.line([(80, base), (620, base)], INK, sw=3, z=3)
    for i, (b_, sd, col) in enumerate(((1, 2.236, RED), (2, 1.291, ORG), (4, 0.0, GRN))):
        x = 120 + i * 180
        g.rect(x, base - max(sd * sc, 4), 120, max(sd * sc, 4), fill=REDL if col == RED else (ORGL if col == ORG else GRNL), stroke=col, sw=3, radius=6, z=3)
        g.text(x - 20, base - max(sd * sc, 4) - 40, 160, 36, f"${sd:g}$", size=22, color=col, bold=True, align="center", z=8); g.text(x - 20, base + 4, 160, 34, f"$b={b_}$", size=20, color=INK, bold=True, align="center", z=8)
    box(g, 680, 430, 550, 170, PUR, PURL, T("標準差隨批次變大而下降", "the standard deviation falls as the batch grows"), T("六種兩筆批次平均:\n$2,1,0,0,-1,-2$ → $\\sqrt{10/6}=1.291$", "six two-sample batch averages:\n$2,1,0,0,-1,-2$ → $\\sqrt{10/6}=1.291$"), ts=20, bs=20)
    banner(g, T("方差隨批次大小下降:$b=1,2,4$ 的標準差是 $2.236,\\,1.291,\\,0$", "variance falls with batch size: stds $2.236,\\,1.291,\\,0$ for $b=1,2,4$"))
    d.done(g, "批次估計", "A batch estimate", {0: ("用四筆資料算：在 w 等於 5，單筆梯度是 3、1、負 1、負 3，平均 0。", "With the four samples at w equal 5 the single gradients are 3, 1, minus 1, minus 3 with mean 0."), 1: ("批次一二的平均是 2。", "The batch of samples one and two averages 2."), 2: ("批次三四的平均是負 2。", "The batch of samples three and four averages minus 2."), 3: ("完整平均是 0。", "The full average is 0."), 4: ("小批次的估計可以和完整梯度的方向完全相反，但兩個批次平均起來剛好是 0。", "A minibatch estimate can point opposite to the full gradient, yet the two batches average exactly 0."), 5: ("六種可能的兩筆批次平均是 2、1、0、0、負 1、負 2，標準差 1.291；單筆是 2.236；全批是 0。方差隨批次大小下降。", "The six possible two-sample batch averages are 2, 1, 0, 0, minus 1, minus 2 with standard deviation 1.291; a single sample has 2.236; the full batch 0. Variance falls with batch size.")})

    g = d.page("方差公式:獨立與無放回", "Variance: independent vs without replacement")
    g.beat = 1
    box(g, 50, 126, 570, 220, BLUE, BLUEL, T("獨立(有放回)抽樣", "independent (with replacement)"), r"$\mathrm{Var}(g_B)=\dfrac{\sigma^2}{b}$".replace(r"\dfrac", r"\frac") + T("\n標準差 $\\sigma/\\sqrt b$", "\nstandard deviation $\\sigma/\\sqrt b$"), ts=22, bs=30)
    g.beat = 2
    box(g, 660, 126, 570, 220, ORG, ORGL, T("無放回抽樣", "without replacement"), r"$\mathrm{Var}(g_B)=\dfrac{\sigma^2}{b}\cdot\dfrac{n-b}{n-1}$".replace(r"\dfrac", r"\frac") + T("\n多一個有限母體修正", "\nplus a finite-population correction"), ts=22, bs=28)
    g.beat = 3
    box(g, 190, 380, 900, 150, GRN, GRNL, T("驗算 $n=4,\\ b=2$", "check $n=4,\\ b=2$"), r"$\frac{5}{2}\cdot\frac{4-2}{4-1}=\frac53,\quad \sqrt{5/3}=1.291$", ts=22, bs=32)
    g.beat = 4
    gold(g, 190, 550, 900, 70, T("和前一頁六種批次直接算出的 $1.291$ 相同", "equal to the $1.291$ computed directly from the six batches"), size=24)
    banner(g, T("「方差一律是 $\\sigma^2/b$」只在獨立抽樣時成立", "“variance is always $\\sigma^2/b$” holds only for independent draws"))
    d.done(g, "方差公式", "Variance formulas", {0: ("獨立有放回抽樣，批次平均的方差是 sigma 平方除以 b。", "For independent sampling with replacement, the batch average's variance is sigma squared over b."), 1: ("無放回抽樣，還要乘上有限母體修正，n 減 b 除以 n 減 1。", "Without replacement multiply by the finite-population correction, n minus b over n minus 1."), 2: ("驗算：n 等於 4、b 等於 2，5 除以 2 乘 2 除以 3，等於 5 除以 3，標準差 1.291。", "Check: n equal 4, b equal 2: 5 over 2 times 2 over 3 equals 5 over 3, standard deviation 1.291."), 3: ("和前一頁六種批次直接算出的 1.291 相同。", "It equals the 1.291 computed directly from the six batches."), 4: ("", "")})

    g = d.page("批次大小的取捨，與 epoch 的限制", "The batch-size trade-off and the limits of an epoch")
    g.beat = 1
    chain(g, 50, 120, 1180, 170, [(RED, REDL, T("雜訊", "noise"), T("越大越小", "smaller as $b$ grows")), (ORG, ORGL, T("記憶體", "memory"), T("越大越吃", "heavier as $b$ grows")), (BLUE, BLUEL, T("每個 epoch 的更新數", "updates per epoch"), r"$\lfloor n/b\rfloor$" + T(",越大越少", ", fewer as $b$ grows"))], ts=22, bs=24, gap=40, beat0=1)
    g.beat = 4
    rows = [[T("甲", "A"), "$n=1000,\\ b=10$", r"$1000/10=100$" + T(" 次/epoch", " /epoch"), r"$\times10=1000$"], [T("乙", "B"), "$n=1000,\\ b=100$", r"$1000/100=10$" + T(" 次/epoch", " /epoch"), r"$\times10=100$"]]
    table(g, 60, 330, [120, 340, 400, 280], ["", T("設定(10 個 epoch)", "setting (10 epochs)"), T("每個 epoch 更新", "updates per epoch"), T("10 個 epoch 共", "over 10 epochs")], rows, rh=68, fs=24, hfs=18, styles=[(GRNL, GRN), (REDL, RED)], gap=8)
    g.beat = 5
    gold(g, 150, 540, 980, 84, T("同樣是 10 個 epoch,更新次數差 10 倍:比較設定要同時看更新步數與看過的樣本總數", "the same 10 epochs differ 10× in updates: compare setups by update count and samples seen, not epochs alone"), size=21)
    banner(g, T("沒有放諸四海皆準的最佳批次大小", "there is no universally best batch size"))
    d.done(g, "取捨", "Trade-offs", {0: ("批次大小控制三件事。第一是雜訊，越大越小。", "Batch size controls three things. First, noise: smaller as the batch grows."), 1: ("第二是記憶體，越大越吃。", "Second, memory: heavier as the batch grows."), 2: ("第三是每個 epoch 的更新次數，越大越少。", "Third, updates per epoch: fewer as the batch grows."), 3: ("", ""), 4: ("epoch 不是通用的算力單位：同樣訓練十個 epoch，批次 10 是一千次更新，批次 100 只有一百次更新。", "An epoch is not a universal unit of compute: ten epochs with batch 10 is a thousand updates, with batch 100 only a hundred."), 5: ("所以比較兩個訓練設定，要同時看更新步數與看過的樣本總數，不能只看 epoch 數。", "So compare two training setups by update count and samples seen, not by epochs alone.")})
    closing(d, [(RED, REDL, "「批次越大越好」", "“a larger batch is always better”", "每步更精確,但更新少、記憶體緊", "more precise steps, but fewer updates and tighter memory"), (ORG, ORGL, "「小批次的估計一定接近完整梯度」", "“a minibatch estimate is always near the full gradient”", "單一批次可以方向完全相反", "a single batch can point the opposite way"), (PUR, PURL, "「epoch 數相同,訓練量就相同」", "“same epochs means same amount of training”", "更新步數與看過的樣本數才是量", "update count and samples seen are the amounts")],
            ("雜訊存在時,「收斂」到底是什麼意思?", "With noise present, what does “convergence” even mean?"), ("下一個單元:SGD 收斂分析", "next unit: SGD convergence analysis"))
    return d.pages
