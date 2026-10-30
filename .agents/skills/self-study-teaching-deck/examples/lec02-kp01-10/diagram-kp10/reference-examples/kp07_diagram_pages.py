"""第 2 週 KP07 損失函數 —— 圖解版(完整覆蓋講義,29 頁)。
用法(在本目錄):  PYTHONPATH=kit python3 kit/build_kp.py kp07_diagram_pages OUT "KP07 …" "KP07 …" lec02-kp07-loss-functions
內容規格(標題、結論、錨點、承接)在 ../kp07_shared.py,與導學版共用。"""
import math, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "kit")); sys.path.insert(0, os.path.join(HERE, ".."))
from lec03_kit import *
import kp07_shared as SH

KP, N = "KP07", 29
RES = (1, 1, 1, 10); DELTA = 1
sq = lambda e: e * e; ab = abs
hub = lambda e, d=DELTA: 0.5 * e * e if abs(e) <= d else d * (abs(e) - 0.5 * d)
VAL = {"sq": [sq(e) for e in RES], "ab": [ab(e) for e in RES], "hub": [hub(e) for e in RES]}
assert VAL["sq"] == [1, 1, 1, 100] and sum(VAL["sq"]) == 103 and sum(VAL["ab"]) == 13 and VAL["hub"] == [0.5, 0.5, 0.5, 9.5] and sum(VAL["hub"]) == 11
_x = [math.exp(v) for v in (2, 1, 0)]; PX = [v / sum(_x) for v in _x]; assert [round(v, 3) for v in PX] == [0.665, 0.245, 0.09]
CE = [-math.log(p) for p in PX]; assert [round(c, 3) for c in CE] == [0.408, 1.408, 2.408]
Y = (0.7, 0.2, 0.1); KLT = [y * math.log(y / p) for y, p in zip(Y, PX)]; KL = sum(KLT); assert round(KL, 4) == 0.0058
HY = -sum(y * math.log(y) for y in Y); CES = -sum(y * math.log(p) for y, p in zip(Y, PX)); assert abs(CES - (HY + KL)) < 1e-12
f = lambda v, n=3: ("%." + str(n) + "f") % v


# 只有圖形、沒有文字的揭示步驟,補導讀稿(否則 check_slides 會 FAIL)
EXTRA = {10: {1: ("曲線 $-\\ln p$:機率 $p$ 越接近 0,損失升得越快;$p$ 接近 1,損失趨近 0。", "The curve $-\\ln p$: the closer $p$ is to 0, the faster the loss climbs; near $p=1$ the loss goes to 0.")},
         19: {1: ("綠色是 Huber 曲線:中間二次、兩側線性。", "The green line is Huber: quadratic in the middle, linear at the sides.")}}


def pg(d, i):
    S = SH.S[i - 1]; g = d.page(S[1], S[2]); bz, be = SH.BR[i - 1]
    g.text(50, 92, 1180, 32, d.T(bz, be), size=18, color=MUTED, z=8)
    return g

def fin(d, g, i, ex):
    S = SH.S[i - 1]; g.beat = 3; banner(g, d.T(S[3], S[4]))
    nt = {0: ex, 3: (S[3], S[4])}; nt.update(EXTRA.get(i, {})); d.done(g, S[1], S[2], nt)

def fml(g, x, y, w, h, tex, size=30, fill=CARD, stroke=GRAY, col=INK):
    g.rect(x, y, w, h, fill=fill, stroke=stroke, sw=3, radius=14, z=3)
    g.text(x + 8, y + 2, w - 16, h - 4, tex, size=size, color=col, align="center", vcenter=True, z=8)

def ticks(g, ux, uy, xs, y, size=18):
    for v in xs: g.text(ux(v) - 30, y, 60, 30, f"${v:g}$", size=size, color=MUTED, align="center", z=8)

def make(lang):
    d = D(lang, KP, N); T = d.T
    pages = []

    # 1 ───────────────────────────────────────────── 損失是一個數
    g = pg(d, 1); g.beat = 1
    a = g.mark(); g.rect(60, 190, 280, 130, fill=BLUEL, stroke=BLUE, sw=3, radius=16, z=3); g.text(60, 215, 280, 90, T("網路輸出(預測)", "network output\n(prediction)"), size=22, color=BLUE, bold=True, align="center", z=8)
    b = g.mark(); g.rect(60, 380, 280, 130, fill=GRNL, stroke=GRN, sw=3, radius=16, z=3); g.text(60, 405, 280, 90, T("正確答案(目標)", "right answer\n(target)"), size=22, color=GRN, bold=True, align="center", z=8)
    c = g.mark(); g.rect(480, 270, 300, 150, fill=ORGL, stroke=ORG, sw=4, radius=20, z=3); g.text(480, 280, 300, 56, T("損失函數", "loss function"), size=28, color=ORG, bold=True, align="center", z=8); g.text(480, 340, 300, 66, r"$\ell$", size=40, color=INK, align="center", z=8)
    e_ = g.mark(); g.rect(940, 270, 280, 150, fill=REDL, stroke=RED, sw=3, radius=16, z=3); g.text(940, 290, 280, 110, T("一個數\n越小越好", "one number\nsmaller is better"), size=28, color=RED, bold=True, align="center", z=8)
    g.link(a, c, 4, 12, GRAY, 5); g.link(b, c, 4, 12, GRAY, 5); g.link(c, e_, 4, 12, GRAY, 5)
    g.beat = 2
    for k, (col, light, t) in enumerate(((BLUE, BLUEL, T("回歸:MSE、MAE、Huber", "regression: MSE, MAE, Huber")), (ORG, ORGL, T("分類:交叉熵、KL", "classification: cross-entropy, KL")), (GRN, GRNL, T("間隔:Hinge", "margin: hinge")))):
        g.rect(60 + k * 400, 555, 380, 60, fill=light, stroke=col, sw=3, radius=14, z=3); g.text(60 + k * 400, 555, 380, 60, t, size=22, color=col, bold=True, align="center", vcenter=True, z=8)
    fin(d, g, 1, T("損失函數拿網路的預測和正確答案比,把「差多少」變成一個數。有了這個數,訓練才知道往哪個方向調整權重,而且這個數越小越好。本單元看六種常用的損失:回歸用 MSE、MAE、Huber;分類用交叉熵與 KL 散度;二分類的間隔用 Hinge。", "A loss function compares the network's prediction with the right answer and turns how far off into one number. With it, training knows which way to adjust the weights, and smaller is better. This unit covers six common losses: MSE, MAE and Huber for regression; cross-entropy and KL divergence for classification; hinge for binary margins."))

    # 2 ───────────────────────────────────────────── 殘差
    g = pg(d, 2); g.beat = 0
    fml(g, 330, 130, 620, 76, r"$e_i=f_o(\mathbf x_i)-y_i$", size=32)
    x0, un = 200, 80
    g.beat = 1
    for k, e in enumerate(RES):
        y = 270 + k * 80
        g.line([(x0 - 30, y), (x0 + 10 * un + 40, y)], GRAY, sw=2, z=2); lab(g, 60, y - 22, 110, f"$y_{k+1}$", size=24, col=PUR, align="center"); dot(g, x0, y, 11, PUR)
    g.beat = 2
    for k, e in enumerate(RES):
        y = 270 + k * 80
        g.line([(x0, y - 22), (x0 + e * un, y - 22)], RED if e == 10 else BLUE, sw=4, z=4); dot(g, x0 + e * un, y, 11, BLUE)
        lab(g, x0 + e * un + 18, y - 38, 150, f"$e_{k+1}={e}$", size=22, col=RED if e == 10 else BLUE)
    g.beat = 0
    dot(g, 330, 610, 9, PUR); lab(g, 346, 594, 220, T("目標 $y$", "target $y$"), size=20, col=PUR); dot(g, 600, 610, 9, BLUE); lab(g, 616, 594, 300, T("預測 $f_o$", "prediction $f_o$"), size=20, col=BLUE)
    lab(g, 880, 586, 350, T("$e>0$:預測偏高", "$e>0$: prediction too high"), size=20, col=MUTED)
    fin(d, g, 2, T("殘差 $e_i$ 是預測減掉目標。這裡有四筆資料,目標都畫在 0,預測落在 1、1、1、10,所以殘差是 1、1、1、10。正負號表示預測偏高或偏低,至於怎麼懲罰這個差,要看你選哪一把損失函數。", "The residual $e_i$ is the prediction minus the target. Here are four data points with target 0 and predictions 1, 1, 1, 10, so the residuals are 1, 1, 1, 10. The sign tells whether the prediction is too high or too low; how that gap is punished depends on the loss you choose."))

    # 3 ───────────────────────────────────────────── 平方
    g = pg(d, 3); g.beat = 0
    fml(g, 330, 130, 620, 76, r"$L_{\rm sq,sum}=\sum_{i=1}^{b}e_i^2$", size=32)
    g.beat = 1; base = 470
    for k, (e, x) in enumerate(zip(RES, (150, 340, 530, 760))):
        s = abs(e) * 18; g.rect(x, base - s, s, s, fill=RED, fillOpacity=0.8 if e == 10 else 1, radius=3, z=3)
        lab(g, x - 30, base + 8, 240, f"$e_{k+1}^2={e*e}$", size=22, col=RED)
    g.text(780, base - 120, 160, 50, r"$10^2=100$", size=26, color="#ffffff", bold=True, align="center", z=8)
    g.beat = 2
    gold(g, 200, 548, 880, 82, r"$1^2+1^2+1^2+10^2=103$", size=32)
    fin(d, g, 3, T("平方把每筆殘差變成正方形的面積:邊長是殘差大小。三個小殘差各只有 1,但最後一筆 10 變成 100,四筆加總是 103。這是還沒取平均的總和。平方的特點就是把大誤差放得特別大。", "Squaring turns each residual into the area of a square whose side is its size. Three small residuals give 1 each, but the last, 10, becomes 100; the sum is 103, before averaging. Squaring blows up large errors."))

    # 4 ───────────────────────────────────────────── 絕對值
    g = pg(d, 4); g.beat = 0
    fml(g, 330, 130, 620, 76, r"$L_{\rm abs,sum}=\sum_{i=1}^{b}|e_i|$", size=32)
    g.beat = 1; base = 475
    for k, (e, x) in enumerate(zip(RES, (190, 400, 610, 820))):
        h = e * 25; g.rect(x, base - h, 80, h, fill=BLUE, radius=6, z=3); lab(g, x - 40, base + 8, 160, f"$|e_{k+1}|={e}$", size=22, col=BLUE, align="center")
    g.line([(130, base), (1000, base)], INK, sw=3, z=3)
    g.beat = 2
    gold(g, 200, 556, 880, 76, r"$|1|+|1|+|1|+|10|=13$", size=32)
    fin(d, g, 4, T("絕對值不平方,直接用殘差的大小:長條的高度就等於殘差。四筆加總是 13。殘差變成兩倍,懲罰也剛好變成兩倍,所以離群值不會被特別放大。", "The absolute value skips the square: each bar's height is just the residual size. The four sum to 13. Double the residual and the penalty doubles, so an outlier is not blown up."))

    # 5 ───────────────────────────────────────────── Huber
    g = pg(d, 5); g.beat = 0
    xr, yr = (-3, 3), (0, 5); ux, uy = axes(g, 60, 140, 560, 410, xr, yr); ticks(g, ux, uy, (-3, -1, 0, 1, 3), 556)
    lab(g, 70, 144, 120, r"$h_1(e)$", size=22, col=GRN)
    g.beat = 1
    g.rect(ux(-1), 142, ux(1) - ux(-1), 406, fill=GOLDL, fillOpacity=0.8, z=2); g.line([(ux(-1), 142), (ux(-1), 548)], GOLD, sw=3, dash=True, z=3); g.line([(ux(1), 142), (ux(1), 548)], GOLD, sw=3, dash=True, z=3)
    curve(g, ux, uy, hub, GRN, xr, yr, sw=6)
    lab(g, ux(-0.9), uy(4.3), ux(0.9) - ux(-0.9), T("二次段", "quadratic"), size=20, col=ORG, align="center"); lab(g, ux(1.2), uy(3.6), 150, T("線性段", "linear"), size=20, col=GRN)
    g.beat = 2
    fml(g, 660, 150, 570, 100, r"$|e|\le\delta:\ h_\delta(e)=\tfrac12e^2$", size=28, fill=ORGL, stroke=ORG)
    fml(g, 660, 270, 570, 100, r"$|e|>\delta:\ h_\delta(e)=\delta\big(|e|-\tfrac12\delta\big)$", size=26, fill=GRNL, stroke=GRN)
    fml(g, 660, 390, 570, 70, r"$\delta>0$" + T(",本例 $\\delta=1$", ", here $\\delta=1$"), size=26, fill=GOLDL, stroke=GOLD)
    fin(d, g, 5, T("Huber 把平方和絕對值接起來。殘差落在 $-\\delta$ 到 $\\delta$ 之間(黃色帶)時用二分之一 $e^2$,像平方一樣平滑;超出門檻就改用直線 $\\delta(|e|-\\tfrac12\\delta)$,像絕對值一樣不暴衝。門檻 $\\delta$ 必須是正數,本例取 1。", "Huber joins the square and the absolute value. Inside the yellow band $-\\delta$ to $\\delta$ it uses $\\tfrac12e^2$, smooth like the square; outside it switches to the line $\\delta(|e|-\\tfrac12\\delta)$, tame like the absolute value. $\\delta$ must be positive; here it is 1."))

    # 6 ───────────────────────────────────────────── Huber 算例
    g = pg(d, 6); g.beat = 0
    zz = (RED, REDL)
    rows = [[r"$e=1$", T(r"$|e|\le\delta$:二次段", r"$|e|\le\delta$: quadratic"), r"$\tfrac12\cdot1^2$", "$0.5$"]] * 3 + [[r"$e=10$", T(r"$|e|>\delta$:線性段", r"$|e|>\delta$: linear"), r"$1\cdot(10-\tfrac12)$", "$9.5$"]]
    table(g, 60, 140, (200, 340, 400, 220), [T("殘差", "residual"), T("落在哪一段", "which piece"), T("代入公式 ($\\delta=1$)", "plug in ($\\delta=1$)"), T("Huber 值", "Huber value")], rows, rh=60, fs=24, styles=[(ORGL, ORG)] * 3 + [(GRNL, GRN)], beats=[1, 1, 1, 2])
    g.beat = 2; gold(g, 200, 510, 880, 86, r"$\tfrac12+\tfrac12+\tfrac12+9.5=11$", size=32)
    fin(d, g, 6, T("同一組殘差代入 Huber。三個 $e=1$ 都不超過門檻,用二次段:各得 0.5。$e=10$ 超過門檻,用線性段:$1\\times(10-0.5)=9.5$。四筆加總是 11。", "Plug the same residuals into Huber. The three $e=1$ are within the threshold, so the quadratic piece gives 0.5 each. $e=10$ exceeds it, so the linear piece gives $1\\times(10-0.5)=9.5$. The four sum to 11."))

    # 7 ───────────────────────────────────────────── 加總 vs 平均
    g = pg(d, 7); g.beat = 0
    rows = [[T("MSE(平方)", "MSE (square)"), "$103$", r"$\div4$", r"$\tfrac{103}{4}=25.75$"], [T("MAE(絕對值)", "MAE (absolute)"), "$13$", r"$\div4$", r"$\tfrac{13}{4}=3.25$"], ["Huber", "$11$", r"$\div4$", r"$\tfrac{11}{4}=2.75$"]]
    table(g, 60, 140, (300, 220, 260, 380), [T("尺", "ruler"), T("加總", "sum"), T("批次大小 $b$", "batch size $b$"), T("平均", "mean")], rows, rh=70, fs=26, styles=[(REDL, RED), (BLUEL, BLUE), (GRNL, GRN)], beats=[1, 1, 1])
    g.beat = 2; gold(g, 150, 470, 980, 130, T("比較前先對齊約定:\n都用加總,或都用平均;不能一個加總、一個平均", "Align the convention before comparing:\nall sums or all means, never mixed"), size=26)
    fin(d, g, 7, T("第 1 節的公式是「對批次求和」。若要取平均,就再除以批次大小 $b$。本例 $b=4$,所以 103、13、11 的平均分別是 25.75、3.25、2.75。比較不同的損失,或和別人的結果對照時,要先確認大家用的是加總還是平均。", "The formulas in section 1 sum over the batch. To get a mean, divide by the batch size $b$. Here $b=4$, so 103, 13 and 11 become 25.75, 3.25 and 2.75. When comparing losses, or with someone else's numbers, first check whether they use sums or means."))

    # 8 ───────────────────────────────────────────── 離群值占比
    g = pg(d, 8); g.beat = 0
    TOT = 740; items = [("MSE", VAL["sq"], RED, REDL), ("MAE", VAL["ab"], BLUE, BLUEL), ("Huber", VAL["hub"], GRN, GRNL)]
    for k, (nm, v, col, light) in enumerate(items):
        y = 170 + k * 120; s = sum(v); w_small = TOT * sum(v[:3]) / s; w_big = TOT - w_small
        g.beat = 1; lab(g, 60, y + 14, 150, nm, size=26, col=col); g.rect(230, y, w_small, 70, fill=light, stroke=col, sw=3, radius=6, z=3)
        g.beat = 2; g.rect(230 + w_small, y, w_big, 70, fill=col, radius=6, z=3)
        g.text(230 + w_small + 6, y + 8, w_big - 12, 54, f"${v[3]:g}/{s:g}\\approx{v[3]/s*100:.1f}\\%$", size=24, color="#ffffff", bold=True, align="center", vcenter=True, z=8)
    g.beat = 1; g.rect(230, 548, 40, 26, fill=GRAY, radius=4, z=3); lab(g, 280, 543, 300, T("前三筆小殘差", "three small residuals"), size=20, col=MUTED); g.rect(620, 548, 40, 26, fill=INK, radius=4, z=3); lab(g, 670, 543, 300, T("離群的最後一筆", "the outlier"), size=20, col=MUTED)
    fin(d, g, 8, T("看總和還不夠,要看最後那筆離群值占了幾成。MSE 的 103 裡,離群值占 100,約 97.1%;MAE 的 13 裡占 10,約 76.9%;Huber 的 11 裡占 9.5,約 86.4%。占比越高,這把尺就越被那一筆左右。這個比較是講義補充的算例。", "Totals are not enough; look at what share the outlier takes. Of MSE's 103 the outlier is 100, about 97.1%; of MAE's 13 it is 10, about 76.9%; of Huber's 11 it is 9.5, about 86.4%. The higher the share, the more that ruler is dominated by one point. This comparison is an added example in the lecture."))

    # 9 ───────────────────────────────────────────── 三條曲線
    g = pg(d, 9); g.beat = 0
    xr, yr = (-3, 3), (0, 9); ux, uy = axes(g, 60, 140, 620, 440, xr, yr); ticks(g, ux, uy, (-3, -2, -1, 0, 1, 2, 3), 584)
    g.beat = 1; curve(g, ux, uy, sq, RED, xr, yr, sw=6); curve(g, ux, uy, ab, BLUE, xr, yr, sw=6)
    g.beat = 2; curve(g, ux, uy, hub, GRN, xr, yr, sw=6, dash=True)
    g.beat = 1; fml(g, 730, 150, 500, 80, r"$e^2\ \to\ 9$", size=30, fill=REDL, stroke=RED, col=RED); fml(g, 730, 250, 500, 80, r"$|e|\ \to\ 3$", size=30, fill=BLUEL, stroke=BLUE, col=BLUE)
    g.beat = 2; fml(g, 730, 350, 500, 80, r"$h_1(e)\ \to\ 2.5$", size=30, fill=GRNL, stroke=GRN, col=GRN)
    lab(g, 730, 440, 500, T("以上都是 $e=3$ 時的值", "all values at $e=3$"), size=20, col=MUTED)
    gold(g, 730, 490, 500, 100, T("曲線比的是「成長方式」,\n不同尺的值不能直接比大小", "Curves compare growth,\nnot values across rulers"), size=22)
    fin(d, g, 9, T("把三把尺畫在同一個座標,殘差 $e$ 從 -3 到 3。紅色平方是拋物線,殘差越大升得越快;藍色絕對值是兩條直線;綠色虛線 Huber 中間像拋物線、兩側像直線,而且比絕對值低。看的是成長方式,不是誰的值比較大。", "Three rulers on one axis, residual $e$ from -3 to 3. The red square is a parabola rising faster as $e$ grows; the blue absolute value is two straight lines; the green dashed Huber is parabola-like in the middle and line-like at the sides, and lies below the absolute value. Compare the growth pattern, not who is larger."))

    # 10 ──────────────────────────────────────────── 交叉熵
    g = pg(d, 10); g.beat = 0
    xr, yr = (0, 1), (0, 3); ux, uy = axes(g, 60, 140, 560, 430, xr, yr); ticks(g, ux, uy, (0, 0.5, 1), 576)
    lab(g, 70, 144, 200, r"$-\ln p$", size=22, col=ORG)
    g.beat = 1; curve(g, ux, uy, lambda p: -math.log(p), ORG, (0.05, 1), yr, sw=6, n=120)
    g.beat = 2
    for p, c_, col in zip(PX, CE, (GRN, BLUE, RED)):
        dot(g, ux(p), uy(c_), 12, col); lab(g, ux(p) + (14 if p < 0.5 else -196), uy(c_) - 34, 190, f"$p={p:.3f}\\to{c_:.3f}$", size=18, col=col)
    fml(g, 660, 150, 570, 90, r"$L_{\rm CE}=-\sum_{i}\sum_{l}y_{il}\ln p_{il}$", size=26)
    fml(g, 660, 260, 570, 80, r"$\ell=-\ln p_{\rm true}$", size=30, fill=ORGL, stroke=ORG)
    fml(g, 660, 360, 570, 80, T(r"輸出要像機率:$\sum_l p_l=1$", r"outputs look like probabilities: $\sum_l p_l=1$"), size=22, fill=GRNL, stroke=GRN)
    lab(g, 660, 460, 570, T("多類接 softmax;二元接 sigmoid", "softmax for multi-class; sigmoid for binary"), size=20, col=MUTED)
    fin(d, g, 10, T("交叉熵的核心是「給正確類別的機率,取負對數」。曲線 $-\\ln p$ 在 $p$ 接近 1 時幾乎是 0,在 $p$ 接近 0 時急速升高,所以預測給正解的機率越小,損失越大。它要求輸出像機率,所以輸出層要接 softmax(多類)或 sigmoid(二元)。", "The core of cross-entropy is the negative log of the probability given to the right class. The curve $-\\ln p$ is almost 0 near $p=1$ and shoots up near 0, so the less probability the prediction gives the right answer, the larger the loss. It needs outputs that look like probabilities, hence softmax (multi-class) or sigmoid (binary)."))

    # 11 ──────────────────────────────────────────── 交叉熵算例
    g = pg(d, 11); g.beat = 0
    lab(g, 300, 135, 330, T("預測機率 $p$(正解那一類)", "probability $p$ of the right class"), size=20, col=MUTED); lab(g, 720, 135, 400, T("損失 $-\\ln p$", "loss $-\\ln p$"), size=20, col=MUTED)
    for k, (p, c_, col, light) in enumerate(zip(PX, CE, (GRN, BLUE, RED), (GRNL, BLUEL, REDL))):
        y = 180 + k * 115; g.beat = 1 if k == 0 else 2
        g.rect(60, y, 200, 80, fill=light, stroke=col, sw=3, radius=14, z=3); g.text(60, y, 200, 80, T(f"正解:第 {k+1} 類", f"right: class {k+1}"), size=22, color=col, bold=True, align="center", vcenter=True, z=8)
        g.rect(300, y + 10, 300 * p, 60, fill=col, radius=6, z=3); lab(g, 300 + 300 * p + 10, y + 22, 200, f"${p:.3f}$", size=24, col=col)
        g.rect(720, y + 10, 120 * c_, 60, fill=col, radius=6, z=3); lab(g, 720 + 120 * c_ + 10, y + 22, 150, f"${c_:.3f}$", size=24, col=col)
    g.beat = 0; fml(g, 130, 540, 1020, 80, r"$\mathbf a=(2,1,0)\ \Rightarrow\ \mathbf p=(0.665,\,0.245,\,0.090)$", size=28, fill=GOLDL, stroke=GOLD)
    fin(d, g, 11, T("沿用 KP06 的 softmax 輸出:輸入 $(2,1,0)$ 得到機率 $(0.665,0.245,0.090)$。同一個預測,正解是第 1、2、3 類時,損失分別是 0.408、1.408、2.408。正解的機率越小,長條就越長。計算時保留完整的中間值,最後才取位數。", "Reuse KP06's softmax output: input $(2,1,0)$ gives probabilities $(0.665,0.245,0.090)$. For the same prediction, the loss is 0.408, 1.408, 2.408 when the right class is 1, 2, 3. The smaller the right class's probability, the longer the bar. Keep full precision in the middle and round only at the end."))

    # 12 ──────────────────────────────────────────── 二元交叉熵
    g = pg(d, 12); g.beat = 0
    xr, yr = (0, 1), (0, 3); ux, uy = axes(g, 60, 140, 560, 430, xr, yr); ticks(g, ux, uy, (0, 0.5, 1), 576)
    g.beat = 1; curve(g, ux, uy, lambda p: -math.log(p), BLUE, (0.05, 1), yr, sw=6, n=120); lab(g, ux(0.1), uy(2.7), 160, T("$y=1$", "$y=1$"), size=22, col=BLUE)
    g.beat = 2; curve(g, ux, uy, lambda p: -math.log(1 - p), ORG, (0, 0.95), yr, sw=6, n=120); lab(g, ux(0.72), uy(2.7), 160, T("$y=0$", "$y=0$"), size=22, col=ORG)
    fml(g, 660, 150, 570, 90, r"$\ell=-y\ln p-(1-y)\ln(1-p)$", size=26)
    fml(g, 660, 260, 570, 70, r"$y=1:\ \ell=-\ln p$", size=26, fill=BLUEL, stroke=BLUE); fml(g, 660, 350, 570, 70, r"$y=0:\ \ell=-\ln(1-p)$", size=26, fill=ORGL, stroke=ORG)
    fml(g, 660, 440, 570, 70, r"$y\in\{0,1\},\ \ 0<p<1$", size=26, fill=GOLDL, stroke=GOLD)
    fin(d, g, 12, T("只有兩類時,一個輸出 $p$ 代表第一類的機率,另一類就是 $1-p$。式子把兩種結果都寫進來:$y=1$ 時只剩 $-\\ln p$,$y=0$ 時只剩 $-\\ln(1-p)$。所以只要一個 sigmoid 輸出,不必讓單一輸出的和為 1。此頁的完整式是補充說明。", "With two classes one output $p$ is the probability of the first class and the other is $1-p$. The formula writes both outcomes: for $y=1$ only $-\\ln p$ remains, for $y=0$ only $-\\ln(1-p)$. So a single sigmoid output is enough; a single output need not sum to 1. This full form is an added explanation."))

    # 13 ──────────────────────────────────────────── KL
    g = pg(d, 13); g.beat = 0
    base, sc_ = 520, 330
    g.line([(80, base), (680, base)], INK, sw=3, z=3)
    for l in range(3):
        x = 130 + l * 190
        g.beat = 1; g.rect(x, base - Y[l] * sc_, 70, Y[l] * sc_, fill=PUR, radius=6, z=3); lab(g, x - 10, base - Y[l] * sc_ - 34, 90, f"${Y[l]:g}$", size=20, col=PUR, align="center")
        g.beat = 2; g.rect(x + 80, base - PX[l] * sc_, 70, PX[l] * sc_, fill=BLUE, radius=6, z=3); lab(g, x + 70, base - PX[l] * sc_ - 34, 100, f"${PX[l]:.3f}$", size=20, col=BLUE, align="center")
        lab(g, x, base + 8, 150, f"$l={l+1}$", size=22, col=MUTED, align="center")
    g.beat = 1; g.rect(100, 140, 36, 24, fill=PUR, radius=4, z=3); lab(g, 144, 134, 250, T("目標分布 $\\mathbf y$", "target $\\mathbf y$"), size=20, col=PUR)
    g.beat = 2; g.rect(400, 140, 36, 24, fill=BLUE, radius=4, z=3); lab(g, 444, 134, 250, T("預測分布 $\\mathbf p$", "prediction $\\mathbf p$"), size=20, col=BLUE)
    g.beat = 2
    fml(g, 730, 150, 500, 100, r"$D_{\rm KL}(\mathbf y\Vert\mathbf p)=\sum_l y_l\ln\frac{y_l}{p_l}$", size=26)
    fml(g, 730, 270, 500, 70, r"$\sum_l y_l=\sum_l p_l=1$", size=26, fill=GRNL, stroke=GRN)
    fml(g, 730, 360, 500, 70, r"$y_l>0\Rightarrow p_l>0$", size=26, fill=ORGL, stroke=ORG)
    gold(g, 730, 450, 500, 100, T("方向不能交換:\n以目標 $\\mathbf y$ 為基準", "Direction matters:\nit is measured from the target $\\mathbf y$"), size=22)
    fin(d, g, 13, T("當目標本身就是一份機率分布(軟標籤,例如知識蒸餾)時,用 KL 散度比較目標 $\\mathbf y$ 和預測 $\\mathbf p$。圖中紫色是目標、藍色是預測,兩份越像,KL 越小。目標有機率的位置,預測也不能是 0;而且 KL 以目標為基準,方向不能交換。", "When the target itself is a probability distribution (a soft label, as in knowledge distillation), KL divergence compares target $\\mathbf y$ with prediction $\\mathbf p$. Purple is the target and blue the prediction; the more alike, the smaller the KL. Where the target has probability, the prediction cannot be 0, and KL is measured from the target, so direction matters."))

    # 14 ──────────────────────────────────────────── KL 算例
    g = pg(d, 14); g.beat = 0
    rows = [[f"${l+1}$", f"${Y[l]:g}$", f"${PX[l]:.3f}$", f"${KLT[l]:+.4f}$"] for l in range(3)]
    styles = [(GRNL, GRN) if KLT[l] >= 0 else (REDL, RED) for l in range(3)]
    table(g, 120, 150, (140, 220, 260, 360), [T("類別 $l$", "class $l$"), "$y_l$", "$p_l$", r"$y_l\ln(y_l/p_l)$"], rows, rh=70, fs=26, styles=styles, beats=[1, 1, 1])
    g.beat = 2; gold(g, 220, 470, 840, 80, r"$0.0357-0.0404+0.0105\approx0.0058$", size=30)
    lab(g, 120, 566, 1040, T("個別項可以是負的(紅色);加起來的總 KL 才一定不小於 0", "single terms may be negative (red); only the total KL is never below 0"), size=22, col=MUTED, align="center")
    fin(d, g, 14, T("目標 $(0.7,0.2,0.1)$ 對預測 $(0.665,0.245,0.090)$,逐項算 $y_l\\ln(y_l/p_l)$:+0.0357、-0.0404、+0.0105,加起來約 0.0058,很接近 0,因為兩份分布很像。第二項是負的,但總 KL 不會小於 0。", "For target $(0.7,0.2,0.1)$ against prediction $(0.665,0.245,0.090)$, compute $y_l\\ln(y_l/p_l)$ term by term: +0.0357, -0.0404, +0.0105; the sum is about 0.0058, close to 0 because the distributions are alike. The second term is negative, yet the total KL is never below 0."))
    assert [round(v, 4) for v in KLT] == [0.0357, -0.0404, 0.0105]

    # 15 ──────────────────────────────────────────── 交叉熵 = 熵 + KL
    g = pg(d, 15); g.beat = 0
    fml(g, 230, 132, 820, 80, r"$H(\mathbf y,\mathbf p)=H(\mathbf y)+D_{\rm KL}(\mathbf y\Vert\mathbf p)$", size=32)
    fml(g, 330, 226, 620, 56, r"$H(\mathbf y)=-\sum_l y_l\ln y_l$", size=24, fill=GRNL, stroke=GRN)
    SC = 480
    for k, (ttl, hh, kk) in enumerate(((T("軟標籤 $\\mathbf y=(0.7,0.2,0.1)$", "soft target $\\mathbf y=(0.7,0.2,0.1)$"), HY, KL), (T("one-hot $\\mathbf y=(1,0,0)$", "one-hot $\\mathbf y=(1,0,0)$"), 0.0, CE[0]))):
        y = 336 + k * 156; g.beat = 1; lab(g, 60, y - 40, 600, ttl, size=22, col=INK)
        wh = hh * SC; wk = max(kk * SC, 6)
        if wh > 0: g.rect(120, y, wh, 70, fill=GRN, radius=6, z=3); g.text(120, y, wh, 70, f"$H(\\mathbf y)={hh:.4f}$", size=22, color="#ffffff", bold=True, align="center", vcenter=True, z=8)
        else: lab(g, 120, y + 18, 220, r"$H(\mathbf y)=0$", size=22, col=GRN)
        g.beat = 2; g.rect(120 + wh + (0 if wh else 200), y, wk, 70, fill=ORG, radius=6, z=3); lab(g, 120 + wh + (0 if wh else 200) + wk + 12, y + 18, 420, f"$D_{{\\rm KL}}={kk:.4f}$" + ("" if k else T("(很小)", " (tiny)")), size=22, col=ORG)
    g.beat = 2; lab(g, 120, 410, 700, f"$H(\\mathbf y,\\mathbf p)={CES:.4f}$", size=22, col=INK); lab(g, 120, 566, 700, f"$H(\\mathbf y,\\mathbf p)={CE[0]:.3f}=D_{{\\rm KL}}$", size=22, col=INK)
    fin(d, g, 15, T("交叉熵等於目標自己的熵加上 KL。固定目標時,熵是常數,所以最小化交叉熵和最小化 KL 是同一件事。軟標籤例:0.8018 + 0.0058 = 0.8076。目標是 one-hot 時熵為 0,交叉熵就等於 KL,都是 0.408。這個 one-hot 算例是講義補充的。", "Cross-entropy is the target's own entropy plus the KL. With a fixed target the entropy is constant, so minimizing cross-entropy and minimizing KL are the same thing. Soft-label example: 0.8018 + 0.0058 = 0.8076. With a one-hot target the entropy is 0, so cross-entropy equals KL, both 0.408. The one-hot case is an added example in the lecture."))
    pages_part1 = d.pages
    import kp07_diagram_pages2 as P2
    P2.build(d, pg, fin, fml, ticks, globals())
    return d.pages

